from typing import *
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


def jieba_cut(valid_text: List[str]):
    """
    use jieba to cut
    Args:
      valid_text: List[str]
    Returns:
      jieba_result: List[List[Tuple]]
    """
    jieba_result = []
    # TODO
    return jieba_result


def evaluate(prediction: List[List[tuple]], target: List[List[tuple]]):
    """
    Span-level metric calculation, return precision, recall, and f1
    Args:
      prediction: List[List[tuple]], each tuple is an index pair indicating one parsed word
      target: List[List[tuple]], same as above
    Returns:
      (precision, recall, f1): Tuple[float]
    """
    # TODO
    raise NotImplementedError
