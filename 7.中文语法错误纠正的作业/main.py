# 加载elmoformanylangs简体中文预训练模型
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
from elmoformanylangs import Embedder, logger
from typing import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import re
from util import GECDataset, load_vocab_dict
import einops
import numpy as np
from torch import nn
from model.model import GECModel
from model.generator import BeamSearchGenerator

import time
from torch.cuda.amp import autocast
from metrics.maxmatch import maxmatch_metric

train_cache = "./data/processed/seg.train"
elmo_model = Embedder("zhs.model", batch_size=16)
logger.setLevel(logging.WARNING)

# sents = [['今', '天', '天气', '真', '好', '啊', '', '', ''],
# ['潮水', '退', '了', '就', '知道', '谁', '没', '穿', '裤子']]
# embeddings = elmo_model.sents2elmo(sents)
# embeddings

# 读入训练和测试集，初始化dataloader
device = "cuda:0" if torch.cuda.is_available() else "cpu"

vocab_dict = load_vocab_dict("./zhs.model/word.dic")

train_dataset = GECDataset("./data/processed/seg.train", vocab_dict=vocab_dict, max_length=200)
test_dataset = GECDataset("./data/processed/seg.txt", vocab_dict=vocab_dict, max_length=200)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=10)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=test_dataset.collate_fn, num_workers=10)

# 定义模型、损失函数、优化器等

model = GECModel(elmo_model, len(vocab_dict)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

reverse_vocab_dict = {v:k for k, v in vocab_dict.items()}
generator = BeamSearchGenerator(model, reverse_vocab_dict, device)

# 模型训练，输出测试结果，对测试结果调用脚本进行评测
total_epoch = 10
gradient_accumulation = 1

for epoch in range(total_epoch):
    train_loss = 0
    start = time.time()
    correct, total = 0, 0
    model.train()
    torch.set_grad_enabled(True)
    MA = lambda old, new, step: old * step / (step + 1) + new / (step + 1)
    for batch_idx, batch in enumerate(train_dataloader):
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        labels = batch.pop("labels")
        with autocast():
            output = model(**batch)
            flatten = lambda tensor: einops.rearrange(tensor, "b s h -> (b s) h" if len(tensor.shape) == 3 else "b s -> (b s)")
            loss = criterion(flatten(output), flatten(labels)) / gradient_accumulation
        loss.backward()
        if (batch_idx + 1) % gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
        train_loss = MA(train_loss, loss.item() if not np.isnan(loss.item()) else train_loss, batch_idx)
        valid_labels = labels != -100
        correct += (output.max(dim=-1).indices[valid_labels] == labels[valid_labels]).sum().item()
        total += valid_labels.sum().item()
        elasped_time = time.time() - start
        print(f"\rEpoch {epoch} Step {batch_idx + 1}/{len(train_dataloader)}: {int(elasped_time / 60)}min {round(elasped_time) % 60}s | Loss = {train_loss:.4f} | Accuracy = {correct / total * 100:.4f}%    ", end="")
    print()

    torch.save(model.state_dict(), f"{epoch}.ckpt")
    model.eval()
    output = []
    start = time.time()
    torch.set_grad_enabled(False)
    for batch_idx, batch in enumerate(test_dataloader):
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        with autocast():
            generated_string = generator.generate(**batch)
        output.extend(generated_string)
        elasped_time = time.time() - start
        print(f"\rEpoch {epoch} Step {batch_idx + 1}/{len(test_dataloader)} Test | {int(elasped_time / 60)}min {round(elasped_time) % 60}s", end="")
    print()
    prediction_file = f"test_{epoch}.txt"
    open(prediction_file, "w").write("\n".join(output))
    
    # metric calculation
    metrics = maxmatch_metric(prediction_file, "./data/raw/gold.01")
    print(metrics)
