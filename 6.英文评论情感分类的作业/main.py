import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader
# from model.cnn import TextCNN  # 已移除，采用动态选择模型
from data_util import MyDataset, collate_fn
import functools
from tqdm import tqdm
import time
import argparse
parser = argparse.ArgumentParser(description="英文评论情感分类作业")
parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "lstm", "rwkv"], help="选择模型类型：cnn, lstm 或 rwkv")
parser.add_argument("--batch_size", type=int, default=64, help="训练时的批处理大小")
parser.add_argument("--dropout", type=float, default=0.5, help="模型的dropout概率")
args = parser.parse_args()

max_length = 64
vector_size = 1024
device = "cuda:0" if torch.cuda.is_available() else "cpu"
options_file = "./data/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "./data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
max_train_example = 10000
print("max_train_example:", max_train_example)


def evaluate(prediction, label, flag):
    accuracy = accuracy_score(label, prediction)
    if flag:
        print("Accuracy:", float("{:.4f}".format(accuracy)))
    return accuracy

scale = 1

print("loading data")
train_loader = DataLoader(
    MyDataset(
        "./data/train.tsv",
        max_length=max_length,
        train=True,
        max_example_num=max_train_example,
    ),
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=functools.partial(collate_fn, device=device),
)
val_loader = DataLoader(
    MyDataset("./data/dev.tsv", max_length=max_length, train=True),
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=functools.partial(collate_fn, device=device),
)


if args.model == "cnn":
    from model.cnn import TextCNN
    model = TextCNN(options_file, weight_file, vector_size=vector_size, max_length=max_length, dropout=args.dropout)
elif args.model == "lstm":
    from model.lstm import TextLSTM
    model = TextLSTM(options_file, weight_file, vector_size=vector_size, max_length=max_length, dropout=args.dropout)
elif args.model == "rwkv":
    from model.rwkv import TextRWKV
    model = TextRWKV(options_file, weight_file, vector_size=vector_size, max_length=max_length, dropout=args.dropout)
else:
    raise ValueError("Invalid model type")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5*scale, weight_decay=1e-4)
loss_function = nn.CrossEntropyLoss()
total_epoch = 10
max_acc = 0.0
print("start training")
tic = time.time()
out_bar = tqdm(range(total_epoch))
for epoch in out_bar: 
    model.train()
    bar = tqdm(train_loader)
    for text, label in bar:
        prediction = model(text)
        loss = loss_function(prediction, label)
        loss.backward()
        optimizer.step()
        bar.set_description(f"Loss: {loss.item():.4f}")
    model.eval()
    val_prediction = []
    with torch.no_grad():
        val_label = []
        for text, label in val_loader:
            prediction = model(text).max(dim=1).indices
            val_prediction.extend(prediction.detach().cpu().tolist())
            val_label.extend(label.cpu().tolist())
    acc = evaluate(val_prediction, val_label, False)
    out_bar.set_description(f"Epoch: {epoch} Max Accuracy: {max_acc:.4f}")
    if acc > max_acc:
        max_acc = acc
        torch.save(model.state_dict(), "data/best_model.pkl")
model.load_state_dict(torch.load("data/best_model.pkl"))
test_prediction = []
model.eval()
with torch.no_grad():
    val_label = []
    for text, label in val_loader:
        prediction = model(text).max(dim=1).indices
        test_prediction.extend(prediction.detach().cpu().tolist())
        val_label.extend(label.cpu().tolist())
evaluate(test_prediction, val_label, True)
toc = time.time()
print("Total Time:", toc - tic, "s")
