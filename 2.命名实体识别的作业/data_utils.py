"""Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task."""

from __future__ import absolute_import, division, print_function

import logging
import os
from io import open
from typing import *

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid: str, words: List[str], labels: List[str]):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        assert len(words) == len(labels)

    def __len__(self):
        return len(self.words)


def read_examples_from_file(file_path: str, mode: str):
    """
    Read file and load into a list of `InputExample`s

    Args:
        file_path: str, file path to load
        mode: str, "train" or "test"
    Returns:
        examples = List[InputExample]
    """
    examples = []
    # TODO
    return examples


def word2features(sent: InputExample, i: int):
    """
    get discrete features for a single word
    Args:
        sent: InputExample
        i: int, index for target word
    Returns:
        features: List[str]

    Please design features for one word in the sentence as input into pycrfsuite model (https://python-crfsuite.readthedocs.io/en/latest/)
    """
    word = sent.words[i]
    features = []
    # TODO
    return features


def sent2features(sent: InputExample):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent: InputExample):
    return sent.labels


def sent2tokens(sent: InputExample):
    return sent.words
