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


# | export
import numpy as np


def compute_count_matrix(
    train_text: List[List[int]],  # training data
    train_labels: List[List[int]],  # training tag labels
    text_vocab: dict,  # word to index
    tag_vocab: dict,  # tag to index
) -> Tuple[
    np.ndarray,  # initial with size (tag_size, ), initial[i] means number of times that tag_i appears at the beginning of a sentence
    np.ndarray,  # transmission with size (tag_size, tag_size), transmission[i,j] means the number of times tag_j follows tag_i
    np.ndarray,  # emission with size (tag_size, vocab_size), where emission[i,j] means the number of times word_j is labeled as tag_i
]:
    """
    compute frequency matrix for training data
    """
    initial = np.zeros(len(tag_vocab))
    transmission = np.zeros((len(tag_vocab), len(tag_vocab)))
    emission = np.zeros((len(tag_vocab), len(text_vocab)))

    for sentence, labels in zip(train_text, train_labels):
        # 句子首词的词性
        initial[labels[0]] += 1
        # 词性转移
        for i in range(1, len(labels)):
            transmission[labels[i - 1], labels[i]] += 1
        # 词性转移到某个词语的次数
        for word, label in zip(sentence, labels):
            emission[label, word] += 1

    return initial, transmission, emission
