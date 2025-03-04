"""Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task."""

from __future__ import absolute_import, division, print_function

import logging
import os
from io import open
from typing import *

logger = logging.getLogger(__name__)

from pyhanlp import HanLP

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
    
    def get_full_sentence(self):
        return "".join(self.words)
    
    def get_postags(self):
        if hasattr(self, "postags"):
            return self.postags
        HanLP.parse('我的希望是希望张晚霞的背影被晚霞映红。', tasks='pos/pku')





from typing import List
from uuid import uuid4

def read_examples_from_file(
    file_path: str,  # file path to load
    mode: str,  # "train" or "test"
) -> List[InputExample]:
    """
    Read file and load into a list of `InputExample`s
    """
    examples = []

    with open(file_path, "r", encoding="utf-8") as f:
        file = f.read()

    for sentence in file.strip().split("\n\n"):
        words = []
        labels = [] if mode!="real_test" else None
        for line in sentence.split("\n"):
            word, label = line.split("\t")
            words.append(word)
            if mode!="real_test":
                labels.append(label)
        examples.append(InputExample(guid=str(uuid4()), words=words, labels=labels))

    return examples

def is_chinese_char(char):
    """判断字符是否为中文"""
    return '\u4e00' <= char <= '\u9fff'


def single_word2features(word: str, prefix=""):
    return [f'{prefix}word.lower={ word.lower()}',
            f'{prefix}word.istitle={word.istitle()}',
            f'{prefix}word.isupper={word.isupper()}',
            f'{prefix}word.isdigit={word.isdigit()}',
            f'{prefix}word.ischinese={all([is_chinese_char(c) for c in word])}']



def word2features(sent: InputExample, # InputExample
                i: int, # index for target word
                use_simple_feature_only: bool = True # 
                )-> List[str]:
    """
    get discrete features for a single word
    Please design features for one word in the sentence as input into pycrfsuite model (https://python-crfsuite.readthedocs.io/en/latest/)
    """
    word = sent.words[i]
    # if not hasattr(sent, "postags"):
        
    features = single_word2features(word)
    if i > 0:
        word1 = sent.words[i-1]
        features += single_word2features(word1, "-1:")
    else:
        features.append('BOS') # 句子开头
        
    if i < len(sent)-1:
        word1 = sent.words[i+1]
        features += single_word2features(word1, "+1:")
    else:
        features.append('EOS')
                
    return features


def sent2features(sent: InputExample):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent: InputExample):
    return sent.labels


def sent2tokens(sent: InputExample):
    return sent.words
