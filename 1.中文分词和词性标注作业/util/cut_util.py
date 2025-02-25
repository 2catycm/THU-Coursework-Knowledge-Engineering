from typing import List, Tuple
import jieba
from tqdm import tqdm


def maximum_match_cut(
    text: str, vocab: set, max_size: int = 4
) -> List[Tuple[int, int]]:
    """
    maximum matching algo
    Args:
      text: str, input text to be parsed
      vocab: set, word set
      max_size: considered maximum length of words
    Returns:
      result: List[tuple], list of index pair indicating parsed words, e.g. [(0, 3), (3, 5), ...]
    """
    result = []
    # TODO
    return result


def get_final_result(backward_result: List[Tuple], forward_result: List[Tuple]):
    """
    return final result given backward matching result and forward matching result
    Args:
      backward_result: List[Tuple]
      forward_result: List[Tuple]
    Returns:
      result: List[Tuple]
    """
    # TODO
    raise NotImplementedError


import jieba

from collections import Counter
def jieba_cut(
    valid_text: List[str], 
    train_set = None # 我们新增加的参数，让jieba可以优先使用训练集的词汇
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
    beta:float = 1.0, # beta for f beta
) -> Tuple[float, float, float]: # precision, recall, f beta
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
        precision = sum(map(lambda i:true_positives[i]/positives[i] if positives[i] >0 else 0,
                             range(len(prediction)))) / len(prediction)
        recall = sum(map(lambda i:true_positives[i]/real_positives[i] if real_positives[i] >0 else 0
                         , range(len(prediction))) ) / len(prediction)
    else:
        # micro average
        # 求出单词级别的总体的 P, R
        precision = sum(true_positives) / sum(positives) if sum(positives) > 0 else 0
        recall = sum(true_positives) / sum(real_positives) if sum(real_positives) > 0 else 0
    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    return precision, recall, f_beta
