from typing import Iterable, Iterator
import json
import regex as re
import multiprocessing
from utils.chunking import find_chunk_boundaries

class bpe_tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        '''
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        '''
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.merges = merges
        #self.special_tokens = special_tokens
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        else:
            self.special_tokens = special_tokens
        self.num_processes = 16
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        '''
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        因为不知道输入对应的文件类型和格式，需要进一步适配各种格式
        '''
        with open(vocab_filepath, "rb") as f:
            vocab = json.load(f)
        with open(merges_filepath, "rb") as f:
            merges = json.load(f)
        return cls(vocab, merges, special_tokens)  # 用 cls 创建类的实例

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        token_list = []
        if self.special_tokens is None: #assume to be short text
            #1. Pre-tokenize
            pretok_lst = [[bytes([i]) for i in word.encode("utf-8")] for word in re.findall(self.PAT, text)]
            #2. Apply the merges
            for tokens in pretok_lst:
                token_list.extend(self.merge_token(tokens))

        elif len(text) < 1000:
            #split_pattern = "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
            split_pattern = "|".join(re.escape(token) for token in self.special_tokens)
            segments = re.split(f"({split_pattern})", text)  # 保留特殊token作为独立段
            for seg in segments:
                if seg in self.special_tokens:
                    token_list.append(seg.encode("utf-8"))  # 或直接 append 特殊token对应的 bytes
                else:
                    words = re.findall(self.PAT, seg)
                    for word in words:
                        tokens = [bytes([b]) for b in word.encode("utf-8")]
                        token_list.extend(self.merge_token(tokens))
            
        else:
            boundaries = find_chunk_boundaries(text, self.num_processes, self.special_tokens[0]) #按照第一个special_token分块
            chunk_definitions = list(zip(boundaries[:-1], boundaries[1:]))
            job_args = [(text, start, end) for start, end in chunk_definitions]
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                results = pool.starmap(self.process_chunk, job_args)
                for result in results:
                    token_list.extend(result)
        
        #将token_list转化为int list
        token_id = [self.vocab_inv[i] for i in token_list]
        return token_id
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            if not line:
                continue
            for token_id in self.encode(line):
                yield token_id
    
    def decode(self, ids: list[int]) -> str:
        mer_bytes = b"".join(self.vocab[i] for i in ids)
        decoded_text = mer_bytes.decode("utf-8", errors="replace")

        return decoded_text
    
    def process_chunk(self, text:str, start:int, end:int) -> list[bytes]:
        chunk_text = text[start:end]
        #split_pattern = "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
        split_pattern = "|".join(re.escape(token) for token in self.special_tokens)
        text_split = re.split(f"({split_pattern})", chunk_text)
        #text_split = self.split_with_special_tokens(chunk_text, split_pattern)
        token_list = []
        for segment in text_split:
            if not segment:
                continue
            if segment in self.special_tokens:
                token_list.append(segment.encode("utf-8"))
            else:
                # 用正则做 pre-tokenization（假设 PAT 是合法正则）
                words = re.findall(self.PAT, segment)
                for word in words:
                    # 字符串编码为 UTF-8 字节序列
                    token_bytes = [bytes([b]) for b in word.encode("utf-8")]
                    # 应用 BPE 合并
                    merged = self.merge_token(token_bytes)  # 返回 List[bytes]
                    token_list.extend(merged)
        return token_list
    
    #错误的合并方法，先出现的不一定是合并优先级最高的
    #def merge_token(self, tokens: list[bytes]) -> list[bytes]:
    #    #合并
    #    while True:
    #        for i in range(len(tokens) - 1):
    #            pair = (tokens[i], tokens[i + 1])
    #            if pair in self.merges:
    #                tokens[i:i+2] = [pair[0] + pair[1]]
    #                break
    #        else:
    #            break
    #    return tokens
    
    def merge_token(self, tokens: list[bytes]) -> list[bytes]:
        
        while True:
            # 找出所有可合并的 pair 和位置
            pairs = [(i, (tokens[i], tokens[i+1]))
                    for i in range(len(tokens)-1)
                    if (tokens[i], tokens[i+1]) in self.merge_ranks]
            if not pairs:
                break
            # 贪心选出 rank 最小的 pair
            i, pair = min(pairs, key=lambda x: self.merge_ranks[x[1]])
            tokens[i:i+2] = [pair[0] + pair[1]]
        return tokens
    
    def split_with_special_tokens(self, text: str, split_pattern) -> list[str]:
        
        result = []
        last_end = 0
        for match in re.finditer(split_pattern, text):
            start, end = match.span()
            if start > last_end:
                result.append(text[last_end:start])  # 普通文本段
            result.append(match.group(0))  # 特殊 token 本身
            last_end = end
        if last_end < len(text):
            result.append(text[last_end:])
        return result
