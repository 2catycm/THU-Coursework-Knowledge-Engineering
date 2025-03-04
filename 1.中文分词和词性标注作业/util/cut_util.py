from typing import List, Tuple
import jieba
from tqdm import tqdm


def maximum_match_cut(
    text: str,  # input text to be parsed
    vocab: set,  # word set
    max_size: int = 4,  # considered maximum length of words
) -> List[
    Tuple[int, int]
]:  # result, list of index pair indicating parsed words, e.g. [(0, 3), (3, 5), ...]
    """
    maximum matching algo
    """
    result = []
    left, right = 0, max_size
    while left < len(text):  # 左边指针的左边是已经分词好的，右边是未分词的
        while right > left:  # 从最大长度开始找，直到找到一个词
            current_str = text[left:right]
            if current_str in vocab:
                result.append((left, right))
                break  # break 内循环
            right -= 1
        if right == left:  # 如果找不到词，就把一个字当做一个词
            right = left + 1
            result.append((left, right))
        # 现在已经有一个词了，更新左右指针，开始下一个词
        left = right
        right = left + max_size

    return result


from util.trie import Trie


def maximum_match_cut_fast(
    text: str,  # input text to be parsed
    vocab: Trie,  # word set trie
    max_size: int = 4,  # considered maximum length of words
) -> List[
    Tuple[int, int]
]:  # result, list of index pair indicating parsed words, e.g. [(0, 3), (3, 5), ...]
    """
    maximum matching algo
    """
    return vocab.search(text)


# 双向最大匹配
def count_single_words(result: List[Tuple[int, int]]) -> int:
    """
    count single words in the result
    """
    return sum(1 for start, end in result if end - start == 1)


from collections import Counter

running_stats = Counter()


def get_final_result(
    backward_result: List[Tuple], forward_result: List[Tuple]
) -> List[Tuple]:  # result
    """
    return final result given backward matching result and forward matching result
    """
    # 如果两个结果一样，就返回
    if backward_result == forward_result:  # python的list == 是正确的。
        return backward_result
    else:
        results = [backward_result, forward_result]
        idx = min(
            (0, 1),
            key=lambda idx: (
                len(results[idx]),  # 先看分词数量，越少越好
                count_single_words(results[idx]),  # 再看单字词数量，越少越好
                idx,  # 最后看元组中的顺序，
            ),
        )
        running_stats.update([idx])
        return results[idx]


import jieba

from collections import Counter


def jieba_cut(
    valid_text: List[str],
    train_set=None,  # 我们新增加的参数，让jieba可以优先使用训练集的词汇
) -> List[List[Tuple]]:  # jieba_result
    """use jieba to cut"""
    # 增加词库
    if train_set is not None:
        counter = Counter([word for sent in train_set for word in sent])
        for word, freq in counter.items():
            jieba.add_word(word, freq)
    # 分词
    return [
        [res[1:] for res in jieba.tokenize(text, "default", True)]
        for text in valid_text
    ]


def evaluate(
    prediction: List[List[Tuple[int, int]]],  # [sentence, word] -> [start, end]
    target: List[List[Tuple[int, int]]],  # [sentence, word] -> [start, end]
    macro_or_micro: bool = True,  # True for macro, False for micro
    beta: float = 1.0,  # beta for f beta
) -> Tuple[float, float, float]:  # precision, recall, f beta
    # Span-level metric calculation, return precision, recall, and f beta
    true_positives: List[int] = []  # 每一个句子计算 有多少个 正确
    positives: List[int] = []  # 预测了多少个东西
    real_positives: List[int] = []  # 实际有多少个东西
    for i in range(len(prediction)):
        pred = set(prediction[i])
        tar = set(target[i])
        true_positives.append(len(pred & tar))
        positives.append(len(pred))
        real_positives.append(len(tar))
    if macro_or_micro:
        # macro average
        # 每个句子单独求 P, R 然后求平均
        precision = sum(
            map(
                lambda i: true_positives[i] / positives[i] if positives[i] > 0 else 0,
                range(len(prediction)),
            )
        ) / len(prediction)
        recall = sum(
            map(
                lambda i: true_positives[i] / real_positives[i]
                if real_positives[i] > 0
                else 0,
                range(len(prediction)),
            )
        ) / len(prediction)
    else:
        # micro average
        # 求出单词级别的总体的 P, R
        precision = sum(true_positives) / sum(positives) if sum(positives) > 0 else 0
        recall = (
            sum(true_positives) / sum(real_positives) if sum(real_positives) > 0 else 0
        )
    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    return precision, recall, f_beta
