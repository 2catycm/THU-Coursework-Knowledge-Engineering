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
"""Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score

# from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from utils import get_labels, load_and_cache_examples

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
)

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id):
    """Train the model"""
    # if args.local_rank in [-1, 0]:
    #     tb_writer = SummaryWriter()

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    t_total = len(train_dataloader) * args.epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    best_metric = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
            }
            outputs = model(**inputs)
            loss = outputs[
                0
            ]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss.backward()

            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            # model.zero_grad()
            optimizer.zero_grad()
            global_step += 1

        results, _ = evaluate(
            args, model, tokenizer, labels, pad_token_label_id, mode="dev"
        )
        if results["f1"] > best_metric:
            best_metric = results["f1"]
            output_dir = os.path.join(args.output_dir, "checkpoint-best")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            tokenizer.save_pretrained(output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples(
        args,
        tokenizer,
        labels,
        pad_token_label_id,
        mode=mode,
        max_length=args.max_length,
    )

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
            }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[0], outputs[1]
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()

        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )
        # memory management
        del outputs, tmp_eval_loss, logits
        torch.cuda.empty_cache()
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

    logger.info("***** Eval results %s *****", prefix)
    print(results)
    return results, preds_list


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default="./data/processed", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--output_dir", default="./checkpoint", type=str)
    parser.add_argument("--mode", default="trigger", type=str)
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument("--model_type", default="bert", type=str)
    parser.add_argument("--eval_only", action="store_true")

    args = parser.parse_args()

    args.output_dir = args.output_dir + "/" + args.mode
    args.data_dir = args.data_dir + "/" + args.mode
    args.labels = args.data_dir + "/" + "labels.txt"

    model_name = "hfl/chinese-bert-wwm-ext"

    # Set seed
    set_seed(args)

    # Prepare CONLL-2003 task
    labels = get_labels(args.labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # load tokenier and plm
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    if args.mode == "argument":
        # add special markers
        tokenizer.add_tokens(["<event>", "<event/>"])
    model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)

    if not args.eval_only:
        train_dataset = load_and_cache_examples(
            args,
            tokenizer,
            labels,
            pad_token_label_id,
            mode="train",
            max_length=args.max_length,
        )
        global_step, tr_loss = train(
            args, train_dataset, model, tokenizer, labels, pad_token_label_id
        )
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    best_checkpoint = os.path.join(args.output_dir, "checkpoint-best")
    results = {}
    tokenizer = BertTokenizer.from_pretrained(best_checkpoint)
    checkpoint = best_checkpoint

    global_step = ""
    model = BertForTokenClassification.from_pretrained(checkpoint)
    model.to(args.device)
    if args.mode == "argument":
        result, predictions = evaluate(
            args,
            model,
            tokenizer,
            labels,
            pad_token_label_id,
            mode="test",
            prefix=global_step,
        )
    else:
        result, predictions = evaluate(
            args,
            model,
            tokenizer,
            labels,
            pad_token_label_id,
            mode="dev",
            prefix=global_step,
        )
    if global_step:
        result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
    results.update(result)
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        for key in sorted(results.keys()):
            writer.write("{} = {}\n".format(key, str(results[key])))

    output_test_predictions_file = os.path.join(best_checkpoint, "eval_predictions.txt")
    with open(output_test_predictions_file, "w") as writer:
        with open(os.path.join(args.data_dir, "dev.txt"), "r") as f:
            example_id = 0
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    writer.write(line)
                    if not predictions[example_id]:
                        example_id += 1
                elif predictions[example_id]:
                    output_line = (
                        line.split()[0] + " " + predictions[example_id].pop(0) + "\n"
                    )
                    writer.write(output_line)
                else:
                    logger.warning(
                        "Maximum sequence length exceeded: No prediction for '%s'.",
                        line.split()[0],
                    )

    return results


if __name__ == "__main__":
    main()
