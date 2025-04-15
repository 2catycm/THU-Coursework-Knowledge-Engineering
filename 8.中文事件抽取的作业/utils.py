# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task."""

from __future__ import absolute_import, division, print_function

import logging
import os
from io import open
from typing import List
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list[str]. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_ids):
        """
        Args:
            input_ids: List[int]
            input_mask: List[int], expected to containing only 1 and 0, where 1 is for unmasked tokens and 0 for masked tokens
            label_ids: List[int]

        all 3 arguments should have same length
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_ids = label_ids


def read_examples_from_file(data_dir, mode):
    """
    read file and convert to a list of `InputExample`s
    """
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        file = f.read()
    samples = file.split("\n\n")
    for guid_index, sample in enumerate(samples):
        lines = sample.split("\n")
        words = []
        labels = []
        for line in lines:
            if line.strip() == "":
                continue
            line = line.split(" ")
            if len(line) == 2:
                words.append(line[0])
                labels.append(line[1])
            else: 
                raise ValueError(
                    "Error in line format: {} in file {}".format(line, file_path)
                )

        examples.append(
            InputExample(guid=f"{mode:s}-{guid_index:d}", words=words, labels=labels)
        )
    return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list,  # a list of all unique labels
    max_seq_length: int,  # all sequence should be padded or truncated to `max_seq_length`
    tokenizer,  # PretrainedTokenizer
    pad_token_label_id: int,  # label id for pad token
) -> List[InputFeatures]:  # features
    """Loads a list of `InputExample`s into a list of `InputFeatures`s"""
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id  # padded token id
    # tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for ex_index, example in enumerate(examples):
        # Hint: remember to add `[CLS]` and `[SEP]` tokens for BERT model
        # e.g. [CLS] the dog is hairy . [SEP]
        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)

            # Bert模型中，一个单词可能会被切分成多个子词，我们需要将标签分配给这些子词
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # 只有第一个子词保留原始标签，其余子词使用特殊标签 "X"
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # 添加[CLS]和[SEP]标记
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # 注意mask的处理，只有真实的token对应的mask值为1，padding的token对应的mask值为0
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        label_ids = label_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids, input_mask=input_mask, label_ids=label_ids
            )
        )
    return features


def load_and_cache_examples(
    args, tokenizer, labels, pad_token_label_id, mode, max_length
):
    logger.info("Creating features from dataset file at %s", args.data_dir)
    examples = read_examples_from_file(args.data_dir, mode)
    features = convert_examples_to_features(
        examples, labels, max_length, tokenizer, pad_token_label_id=pad_token_label_id
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    return dataset


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
        print(len(labels), labels)
    if "O" not in labels:
        labels = ["O"] + labels
    return labels
