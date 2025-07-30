from typing import Iterable, Iterator
import json
import regex as re
import multiprocessing
import base64
from tqdm import tqdm
import heapq
from utils.chunking import find_chunk_boundaries
from tqdm.contrib.concurrent import process_map

class Node:
    """双向链表中的节点"""
    _id_counter = 0 
    def __init__(self, value):
        self.value = value  # 当前 token 的字节内容
        self.prev = None    # 指向前一个节点
        self.next = None    # 指向后一个节点
        self.active = True  # 节点是否有效（用于懒惰删除）
        # 为每个新创建的节点分配一个唯一的、递增的ID
        self.id = Node._id_counter
        Node._id_counter += 1

    def __lt__(self, other):
        """
        定义 Node 对象之间的小于 (<) 比较规则。
        我们根据节点的唯一 ID 来比较。
        """
        return self.id < other.id

class bpe_tokenizer():
    def __init__(self, vocab, merges, special_tokens=None, num_processes = 10):
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
        self.num_processes = num_processes
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None, num_processes = 10):
        '''
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        '''
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_b64 = json.load(f)
        vocab = {int(k): base64.b64decode(v) for k, v in vocab_b64.items()}
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = [
                (
                    base64.b64decode(a_str),
                    base64.b64decode(b_str)
                )
                for a_str, b_str in (line.strip().split() for line in f)
            ]
        return cls(vocab, merges, special_tokens, num_processes)  # 用 cls 创建类的实例
    
    #原版
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


    def encode_iterable(self, iterable: Iterable[str], total=None) -> Iterator[int]:
        for line in tqdm(iterable, total=total):
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
    
    #def merge_token(self, tokens: list[bytes]) -> list[bytes]:    
    ##simple implementation
    #    while True:
    #        # 找出所有可合并的 pair 和位置
    #        pairs = [(i, (tokens[i], tokens[i+1]))
    #                for i in range(len(tokens)-1)
    #                if (tokens[i], tokens[i+1]) in self.merge_ranks]
    #        if not pairs:
    #            break
    #        # 贪心选出 rank 最小的 pair
    #        i, pair = min(pairs, key=lambda x: self.merge_ranks[x[1]])
    #        tokens[i:i+2] = [pair[0] + pair[1]]
    #    return tokens
    
    def merge_token(self, tokens: list[bytes]) -> list[bytes]:
        if len(tokens) < 2:
            return tokens

        # === 步骤 1: 初始化 - 构建双向链表 ===
        head = Node(tokens[0])
        prev_node = head
        for i in range(1, len(tokens)):
            current_node = Node(tokens[i])
            prev_node.next = current_node
            current_node.prev = prev_node
            prev_node = current_node

        # === 步骤 2: 初始化 - 填充优先队列 (Heap) ===
        heap = []
        current_node = head
        while current_node and current_node.next:
            pair = (current_node.value, current_node.next.value)
            if pair in self.merge_ranks:
                rank = self.merge_ranks[pair]
                # 堆中存储: (优先级, 左节点, 右节点)
                heapq.heappush(heap, (rank, current_node, current_node.next))
            current_node = current_node.next

        # === 步骤 3: 主循环 - 合并最高优先级的词对 ===
        while heap:
            # 取出当前优先级最高的词对
            rank, node1, node2 = heapq.heappop(heap)

            # 关键的有效性检查（懒惰删除）
            # 如果 node1 或 node2 因为之前的合并已经失效，则跳过
            if not node1.active or not node2.active or node1.next != node2:
                continue

            # --- 执行合并 ---
            # 1. 标记旧节点为失效
            node1.active = False
            node2.active = False

            # 2. 创建合并后的新节点
            new_value = node1.value + node2.value
            new_node = Node(new_value)

            # 3. 重新连接链表
            #    A <-> node1 <-> node2 <-> D  ==>  A <-> new_node <-> D
            left_neighbor = node1.prev
            right_neighbor = node2.next

            new_node.prev = left_neighbor
            if left_neighbor:
                left_neighbor.next = new_node
            else:
                head = new_node # 如果合并的是头节点，则更新头节点

            new_node.next = right_neighbor
            if right_neighbor:
                right_neighbor.prev = new_node

            # 4. 将新产生的邻居对加入堆中
            # a) 检查左侧新邻居对 (A, new_node)
            if left_neighbor:
                pair = (left_neighbor.value, new_node.value)
                if pair in self.merge_ranks:
                    new_rank = self.merge_ranks[pair]
                    heapq.heappush(heap, (new_rank, left_neighbor, new_node))
            
            # b) 检查右侧新邻居对 (new_node, D)
            if right_neighbor:
                pair = (new_node.value, right_neighbor.value)
                if pair in self.merge_ranks:
                    new_rank = self.merge_ranks[pair]
                    heapq.heappush(heap, (new_rank, new_node, right_neighbor))

        # === 步骤 4: 收尾 - 从链表中提取结果 ===
        final_tokens = []
        current_node = head
        while current_node:
            final_tokens.append(current_node.value)
            current_node = current_node.next
            
        return final_tokens

    
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
