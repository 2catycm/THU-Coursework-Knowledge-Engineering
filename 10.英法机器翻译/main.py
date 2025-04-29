import json
import torch
import os
import argparse
from metrics import bleu_score
from model.transformer import Transformer, TransformerConfig
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from utils import collate_fn, get_linear_schedule_with_warmup, greedy_decoder, decode
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_path", type=str, default="output")
    parser.add_argument("--train_data_path", type=str, default="data/processed/train.json")
    parser.add_argument("--valid_data_path", type=str, default="data/processed/valid.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--config_path", type=str, default="config.json")
    parser.add_argument("--num_warmup_steps", type=int, default=32)
    parser.add_argument("--num_training_steps", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--ckpt", type=int, default=0)
    return parser.parse_args()

args = parse_args()

# 自定义数据集
class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        for line in open(data_path, "r", encoding="utf-8"):
            self.data.append(json.loads(line))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
# 从config.json加载模型
with open(args.config_path, "r", encoding="utf-8") as f:
    config_dict = json.load(f)
config = TransformerConfig(config_dict)
model = Transformer(config)
# 加载模型参数
if os.path.exists(os.path.join(args.model_path, "model_"+str(args.ckpt)+".bin")):
    model.load_state_dict(torch.load(os.path.join(args.model_path, "model_"+str(args.ckpt)+".bin")))
else:
    os.mkdir(args.model_path)
# 训练模式
if args.mode == "train":
    model.train()
    # 使用自定义数据集和collate_fn初始化DataLoader
    dataloader = DataLoader(MyDataset(args.train_data_path), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = Adam(model.parameters(), lr=args.lr)
    # 加载优化器参数
    if os.path.exists(os.path.join(args.model_path, "optimizer_"+str(args.ckpt)+".bin")):
        optimizer.load_state_dict(torch.load(os.path.join(args.model_path, "optimizer_"+str(args.ckpt)+".bin")))
    # 定义学习率调度器
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.num_training_steps)
    # 加载调度器参数
    if os.path.exists(os.path.join(args.model_path, "lr_scheduler_"+str(args.ckpt)+".bin")):
        lr_scheduler.load_state_dict(torch.load(os.path.join(args.model_path, "lr_scheduler_"+str(args.ckpt)+".bin")))
    # 训练循环
    for epoch in range(args.num_epochs):
        for batch in tqdm(dataloader):
            source_ids, source_mask, target_ids, target_mask = batch
            output = model(source_ids, source_mask, target_ids, target_mask)
            loss = criterion(output.view(-1, output.size(-1)), target_ids.view(-1))
            print(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
        # 保存模型参数
        torch.save(model.state_dict(), os.path.join(args.model_path, "model_"+str(epoch)+".bin"))
        # 保存优化器参数
        torch.save(optimizer.state_dict(), os.path.join(args.model_path, "optimizer_"+str(epoch)+".bin"))
        # 保存调度器参数
        torch.save(lr_scheduler.state_dict(), os.path.join(args.model_path, "lr_scheduler_"+str(epoch)+".bin"))
# 评估模式
else:
    model.eval()
    dataloader = DataLoader(MyDataset(args.valid_data_path), batch_size=1, shuffle=False, collate_fn=collate_fn)
    candidate_corpus = []
    references_corpus = []
    # 评估循环
    for batch in dataloader:
        source_ids, source_mask, target_ids, target_mask = batch
        prediction = greedy_decoder(model, config_dict["target_vocab_path"], source_ids, source_mask)
        candidate_corpus.append(decode(config_dict["target_vocab_path"], prediction.squeeze(0)))
        references_corpus.append(decode(config_dict["target_vocab_path"], target_ids[1:-1]))
    # 计算BLEU
    bleu = bleu_score(references_corpus, candidate_corpus, max_n=4)
    print(f"Valid BLEU score={bleu}")
