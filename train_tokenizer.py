from __future__ import annotations

import os
from typing import IO, Any, BinaryIO, List, Tuple, Union
import regex as re
import collections
from tqdm import trange
import multiprocessing
from utils.chunking import find_chunk_boundaries



def process_chunk(input_path: str, start: int, end: int, special_tokens: List[str], PAT) -> collections.Counter:
    """
    这个函数由单个子进程执行。
    它打开文件，读取指定的区块，并返回该区块的词频统计。
    """
    # 1. 打开文件并读取指定范围的字节
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    
    # 2. 解码成字符串
    chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
    
    # 3. 在当前区块内部，根据特殊 token 进行切分
    split_pattern = "|".join(re.escape(token) for token in special_tokens)
    text_split = re.split(split_pattern, chunk_str)
    
    # 4. 对切分后的每个部分进行预分词和频率统计
    chunk_total_counts = collections.Counter()

    # 5. 返回当前区块的统计结果
    for text_item in text_split:
        for m in re.finditer(PAT, text_item):
            bword = m.group().encode("utf-8")
            chunk_total_counts[bword] += 1
    return chunk_total_counts

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 16,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.
        num_processes: number of processes to work

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, special_tokens[0].encode("utf-8")) #这里需要适配special_tokens列表
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    chunk_definitions = list(zip(boundaries[:-1], boundaries[1:]))
    job_args = [(input_path, start, end, special_tokens, PAT) for start, end in chunk_definitions]
    total_counts = collections.Counter()
    with multiprocessing.Pool(processes=num_processes) as pool:
        # pool.starmap 会将 job_args 中的每个元组解包作为 process_chunk 的参数
        results = pool.starmap(process_chunk, job_args)
        # 聚合所有子进程返回的结果
        print("All processes have completed. Aggregating results...")
        for chunk_counts in results:
            total_counts.update(chunk_counts)

    initial__bytes = {i: bytes([i]) for i in range(256)}
    pair_counts = collections.defaultdict(int)
    vocab = collections.defaultdict()
    for key, value in total_counts.items():
        alpha_list = [bytes([b]) for b in key]
        if len(alpha_list) < 2:
            continue
        vocab[key] = (alpha_list, value)
        for index1, index2 in zip(alpha_list, alpha_list[1:]):
            pair_counts[(index1, index2)] += value

    vocab_cur = len(special_tokens) + len(initial__bytes) #未合并前是特殊token数+初始字符数（256）
    merges = []
    #持续合并
    for _ in trange(vocab_size - vocab_cur):
        pair = max(pair_counts, key=lambda p: (pair_counts[p], p))#找到频率最高的字符对,相同同频率下优先选字典序大的
        del pair_counts[pair] #删除频率最高的
        merge_alp = pair[0] + pair[1]
        merges.append((pair[0], pair[1])) #记录合并

        for key in list(vocab.keys()):
            alp_list, alp_count = vocab[key]
            i = 0
            while i < len(alp_list) - 1:
                cur_pair = (alp_list[i], alp_list[i+1])
                if cur_pair == pair: #因为合并需要减少左右字符配对频率
                    if i > 0:
                        l_alp = alp_list[i-1]
                        l_pair = (l_alp, alp_list[i])
                        pair_counts[l_pair] -= alp_count
                        pair_counts[(l_alp,merge_alp)] += alp_count
                    if i + 2 < len(alp_list):
                        r_alp = alp_list[i+2]
                        r_pair = (alp_list[i+1], r_alp)
                        pair_counts[r_pair] -= alp_count
                        pair_counts[(merge_alp, r_alp)] += alp_count
                    
                    alp_list[i:i+2] = [merge_alp]
                else:
                    i += 1
            #更新vocab中key的alp_list
            vocab[key]= (alp_list, alp_count)

    final_tokens = initial__bytes.copy()
    nxt_id = len(final_tokens)
    #添加特殊字符
    for s_token in special_tokens:
        final_tokens[nxt_id] = s_token.encode("utf-8")
        nxt_id += 1
    #添加训练得到的字符组合
    for mer_pair in merges:
        merged = mer_pair[0] + mer_pair[1] 
        final_tokens[nxt_id] = merged
        nxt_id += 1
    
    return final_tokens, merges