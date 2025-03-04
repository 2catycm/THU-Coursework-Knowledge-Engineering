"""Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task."""

from __future__ import absolute_import, division, print_function

import logging
import os
from io import open
from typing import *

logger = logging.getLogger(__name__)

# from pyhanlp import HanLP

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
    
    # def get_postags(self):
        





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
            word, label = line.split()
            words.append(word)
            if mode!="real_test":
                labels.append(label)
        examples.append(InputExample(guid=str(uuid4()), words=words, labels=labels))

    return examples

from pypinyin import pinyin

from typing import List
def is_chinese_char(char):
    """判断字符是否为中文"""
    return '\u4e00' <= char <= '\u9fff'

from cnradical import Radical, RunOption
radical = Radical(RunOption.Radical)


strokes = []

def get_stroke(c, strokes_path="data/strokes.txt"):
    # 如果返回 0, 则也是在unicode中不存在kTotalStrokes字段
    global strokes
    if not strokes:
        with open(strokes_path, 'r') as fr:
            for line in fr:
                strokes.append(int(line.strip()))
 
    unicode_ = ord(c)
 
    if 13312 <= unicode_ <= 64045:
        return strokes[unicode_-13312]
    elif 131072 <= unicode_ <= 194998:
        return strokes[unicode_-80338]
    else:
        # print("c should be a CJK char, or not have stroke in unihan data.")
        # can also 
        return 0

def single_word2features(
    word: str, prefix="", use_simple_feature_only: bool = True
) -> List[str]:
    features = [
        f"{prefix}word.lower={word.lower()}",
        f"{prefix}word.istitle={word.istitle()}",
        f"{prefix}word.isupper={word.isupper()}",
        f"{prefix}word.isdigit={word.isdigit()}",
        f"{prefix}word.ischinese={is_chinese_char(word)}",
    ]
    if not use_simple_feature_only:
        features += [
            f"{prefix}word.pinyin={pinyin(word)[0][0]}",
            f"{prefix}word.radical={radical.trans_ch(word)}",
            f"{prefix}word.stroke={get_stroke(word)}",
        ]
    return features


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
        
    features = single_word2features(word, "", use_simple_feature_only)
    if i > 0:
        word1 = sent.words[i-1]
        features += single_word2features(word1, "-1:", use_simple_feature_only)
    else:
        features.append('BOS') # 句子开头
        
    if i < len(sent)-1:
        word1 = sent.words[i+1]
        features += single_word2features(word1, "+1:", use_simple_feature_only)
    else:
        features.append('EOS')
                
    return features


def sent2features(sent: InputExample, use_simple_feature_only: bool = True):
    return [word2features(sent, i, use_simple_feature_only) for i in range(len(sent))]


def sent2labels(sent: InputExample):
    return sent.labels


def sent2tokens(sent: InputExample):
    return sent.words
