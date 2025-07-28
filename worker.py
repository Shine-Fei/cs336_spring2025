import regex as re
import collections
from typing import List

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
    for text_item in text_split:
        for m in re.finditer(PAT, text_item):
            bword = m.group().encode("utf-8")
            chunk_total_counts[bword] += 1
    return chunk_total_counts