import json
from typing import List

import torch
from torch.optim.lr_scheduler import LambdaLR


def pad_list(sequences: List[List[int]], pad_token: int):
    '''将一个列表的列表填充到最大长度'''
    max_len = max([len(seq) for seq in sequences])
    ids = []
    mask = []
    for seq in sequences:
        ids.append(seq + [pad_token] * (max_len - len(seq)))
        mask.append([1] * len(seq) + [0] * (max_len - len(seq)))
    return ids, mask

def collate_fn(data: List[dict]):
    source = [item["source"] for item in data]
    target = [item["target"] for item in data]
    source_ids, source_mask = pad_list(source, 0)
    target_ids, target_mask = pad_list(target, 0)
    source_ids = torch.tensor(source_ids, dtype=torch.long)
    source_mask = torch.tensor(source_mask, dtype=torch.long)
    target_ids = torch.tensor(target_ids, dtype=torch.long)
    target_mask = torch.tensor(target_mask, dtype=torch.long)
    return source_ids, source_mask, target_ids, target_mask

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    '''创建学习率衰减策略'''
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def greedy_decoder(model, target_vocab_path: str, source_ids: List[List[int]], source_mask: List[List[int]], max_len=64):
    '''
    贪心解码，每次只输入一个句子，从<BOS>开始解码，每次只输出一个token，直到输出<EOS>或者达到最大长度，需要学生自行实现
    Args:
        model: 训练好的模型
        target_vocab_path: 目标语言词表路径
        source_ids: 输入序列，每次解码只能输入一个句子
        source_mask: 输入序列mask
        max_len: 最大输出序列长度
    '''
    with open(target_vocab_path, "r", encoding="utf-8") as f:
        target_vocab = json.load(f)
    encoder_output = model.encoder(source_ids, source_mask)
    decoder_input = torch.tensor([[]], dtype=torch.long)
    next_token = target_vocab["<BOS>"]
    while decoder_input.shape[1] < max_len:
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]], dtype=torch.long)], dim=1)
        target_mask = torch.ones_like(decoder_input)
        decoder_output = model.decoder(decoder_input, encoder_output, source_mask, target_mask)
        logits = model.linear(decoder_output)
        next_token = torch.argmax(logits[:, -1, :]).item()
        if next_token == target_vocab["<EOS>"]:
            break
    return decoder_input[:, 1:]

def decode(target_vocab_path, ids):
    '''使用目标语言词表将id序列转换为词序列，需要学生自行实现'''
    # 读取词表
    with open(target_vocab_path, "r", encoding="utf-8") as f:
        target_vocab = json.load(f)
    # 建立反向词表
    target_vocab = {v: k for k, v in target_vocab.items()}
    # 将id转换为词，如果token以##开头，则将其与前一个词拼接
    tokens = []
    for id in ids:
        id = id.item()
        if id == 0:
            continue
        token = target_vocab[id]
        if token.startswith("##"):
            tokens[-1] += token[2:]
        else:
            tokens.append(token)
    return tokens