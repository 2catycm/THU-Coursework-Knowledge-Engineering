import numpy as np
import re
from typing import *


def preprocess(lines):
    lines = list(filter(lambda x: len(x.strip()) > 0, lines))
    all_text, all_labels = [], []

    for line in lines:
        all_text.append([])
        all_labels.append([])
        for word in line.strip().split(" "):
            clean_word = re.sub(r"^\[|\][a-zA-Z]+", "", word)
            text, tag = clean_word.strip().split("/")
            all_text[-1].append(text)
            all_labels[-1].append(tag)
    print(all_text[0], "\n", all_labels[0])
    return all_text, all_labels


def compute_count_matrix(
    train_text: List[List[int]],
    train_labels: List[List[int]],
    text_vocab: dict,
    tag_vocab: dict,
):
    """
    compute frequency matrix for training data
    Args:
      train_text: training data
      train_labels: training tag labels
      text_vocab: Dict[str:int], word to index
      tag_vocab: Dict[str:int], tag to index
    Returns:
      initial: np.array, with size (tag_size, ), initial[i] means number of times that tag_i appears at the beginning of a sentence
      transmission: np.array, with size (tag_size, tag_size), transmission[i,j] means the number of times tag_j follows tag_i
      emission: np.array, with size (tag_size, vocab_size), where emission[i,j] means the number of times word_j is labeled as tag_i
    """
    initial = np.zeros(len(tag_vocab))
    transmission = np.zeros((len(tag_vocab), len(tag_vocab)))
    emission = np.zeros((len(tag_vocab), len(text_vocab)))

    # TODO

    return initial, transmission, emission
