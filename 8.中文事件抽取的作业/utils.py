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
        words = []
        labels = []
        guid_index = ""
        #############
        # TODO

        examples.append(
            InputExample(
                guid="%s-%d".format(mode, guid_index), words=words, labels=labels
            )
        )
    return examples


def convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, pad_token_label_id
):
    """Loads a list of `InputExample`s into a list of `InputFeatures`s

    Args:
        examples: List[InputExample]
        label_list: a list of all unique labels
        max_seq_length: int, all sequence should be padded or truncated to `max_seq_length`
        tokenizer: PretrainedTokenizer
        pad_token_label_id: label id for pad token
    Returns:
        features: List[InputFeatures]
    """
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id  # padded token id
    # tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for ex_index, example in enumerate(examples):
        input_ids = []
        input_mask = []
        label_ids = []
        ##############
        # TODO
        # Hint: remember to add `[CLS]` and `[SEP]` tokens for BERT model
        # e.g. [CLS] the dog is hairy . [SEP]

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
