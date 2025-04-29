import collections
import math
import torch
from typing import List


def ngrams_iterator(token_list, ngrams):
    """
    返回一个迭代器，该迭代器生成给定的token和它们的ngrams。
    Args:
        token_list: 一个包含tokens的列表。
        ngrams: ngrams的数量。
    """

    def _get_ngrams(n):
        return zip(*[token_list[i:] for i in range(n)])

    for x in token_list:
        yield x
    for n in range(2, ngrams + 1):
        for x in _get_ngrams(n):
            yield " ".join(x)


def _compute_ngram_counter(tokens, max_n):
    """
    计算给定的tokens的ngrams的计数器。
    Args:
        tokens: 一个可迭代对象，包含tokens。
        max_n: 最大的ngram的数量。
    Returns:
        一个collections.Counter，包含ngrams的计数器。
    """
    ngrams_counter = collections.Counter(
        tuple(x.split(" ")) for x in ngrams_iterator(tokens, max_n)
    )
    return ngrams_counter


def bleu_score(
    candidate_corpus: List[List[str]],  # 候选翻译的token列表的列表。
    references_corpus: List[List[str]], # 参考翻译的token列表的列表，数量应与候选一一对应。
    max_n=4, # 最大的ngram的数量，默认4。
    weights: List[float] = [0.25] * 4, # 用于计算加权几何平均时的权重列表，长度应为max_n。
) -> float: # BLEU分数（0到1之间）。
    """
    计算候选翻译语料库和参考翻译语料库之间的BLEU分数。
    """
    # =====================
    # 1. 计算各n-gram的累计剪辑计数和总计数
    # =====================
    # 初始化剪辑后计数和总计数的列表
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n

    # 用于brevity penalty的长度统计
    c_len = 0  # 候选总长度
    r_len = 0  # 参考总长度（最接近匹配）

    # 遍历每一对候选和参考
    for cand_tokens, ref_tokens in zip(candidate_corpus, references_corpus):
        # 更新长度
        c_len += len(cand_tokens)
        # 选择参考长度与候选最接近的
        # （这里只取单参考，故直接使用ref长度）
        r_len += len(ref_tokens)

        # 获取候选和参考各自的n-gram计数器
        cand_counter = _compute_ngram_counter(cand_tokens, max_n)
        ref_counter = _compute_ngram_counter(ref_tokens, max_n)

        # 遍历n-gram的各个order，由长度tuple决定
        for ngram_tuple, count in cand_counter.items():
            n = len(ngram_tuple)  # 当前n-gram的order
            if n <= max_n:
                # 剪辑计数：candidate出现次数与reference出现次数的最小值
                clipped = min(count, ref_counter.get(ngram_tuple, 0))
                clipped_counts[n - 1] += clipped
                # 累计candidate的n-gram总数
                total_counts[n - 1] += count

    # =====================
    # 2. 计算每个n-gram order的Precision
    # =====================
    precisions = []
    for i in range(max_n):
        if total_counts[i] == 0:
            # 如果某个order的candidate完全没有n-gram，则Precision定义为0
            precisions.append(0.0)
        else:
            precisions.append(clipped_counts[i] / total_counts[i])

    # =====================
    # 3. 计算brevity penalty (BP)
    # =====================
    # 当候选长度大于参考长度时，BP=1；否则BP=exp(1 - r/c)
    if c_len > r_len:
        bp = 1.0
    else:
        # 防止除0
        bp = math.exp(1 - (r_len / c_len)) if c_len > 0 else 0.0

    # =====================
    # 4. 计算加权几何平均（log域）并得到BLEU分数
    # =====================
    # 如果任何precision为0，则整体几何平均为0
    if min(precisions) == 0:
        geo_mean = 0.0
    else:
        # sum(weights_i * ln precision_i)
        geo_mean = math.exp(
            sum(weights[i] * math.log(precisions[i]) for i in range(max_n))
        )

    bleu = bp * geo_mean
    return bleu

# 示例：
# cand = [['the', 'cat', 'is', 'on', 'the', 'mat']]
# ref = [['there', 'is', 'a', 'cat', 'on', 'the', 'mat']]
# print(bleu_score(cand, ref))  # 输出BLEU分数  

