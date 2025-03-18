import numpy as np
import gensim
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from dataset import MyDataset
from util import load_word2vec_model, get_word_embeddings
from model.cnn import TextCNN
from tqdm import tqdm

max_length = 1024
vector_size = 100
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pad_token = "<PAD>"
unk_token = "<UNK>"
batch_size = 50


def evaluate(prediction, label, print_):
    precision, recall, f1, _ = precision_recall_fscore_support(label, prediction, average=None, labels=sorted(list(set(label))))
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(label, prediction, average="micro")
    if print_:
        print("各类别Precision:", [float('{:.4f}'.format(i)) for i in precision])
        print("各类别Recall:", [float('{:.4f}'.format(i)) for i in recall])
        print("各类别F1:", [float('{:.4f}'.format(i)) for i in f1])
        print("整体微平均Precision:", float('{:.4f}'.format(micro_precision)))
        print("整体微平均Recall:", float('{:.4f}'.format(micro_recall)))
        print("整体微平均F1:", float('{:.4f}'.format(micro_f1)))
    return micro_f1


print("loading word2vec model")
# use training data only
word2vec_model = load_word2vec_model(file="./data/raw/cnews.train.txt", vector_size=vector_size)
text_vocab = word2vec_model.wv.key_to_index
# add unk_token and pad_token
unk_index = text_vocab[unk_token] = len(text_vocab)
pad_index = text_vocab[pad_token] = len(text_vocab)

print("loading dataset")
train_dataset = MyDataset("./data/raw/cnews.train.txt", text_vocab=text_vocab, pad_token=pad_token, unk_token=unk_token, max_length=max_length)
label2index = train_dataset.label2index
val_dataset = MyDataset("./data/raw/cnews.val.txt", text_vocab=text_vocab, label2index=label2index, pad_token=pad_token, unk_token=unk_token, max_length=max_length)
test_dataset = MyDataset("./data/raw/cnews.test.txt", text_vocab=text_vocab, label2index=label2index, pad_token=pad_token, unk_token=unk_token, max_length=max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# get embedding data
word_embeddings = get_word_embeddings(word2vec_model, vector_size=vector_size)

print("preparing model")
model = TextCNN(word_embeddings, vector_size, label2index, pad_index, max_length = 1024).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_function = nn.CrossEntropyLoss()
total_epoch = 50
max_micro_f1 = 0.0
for epoch in tqdm(range(total_epoch)):
    model.train()
    for text, label in train_loader:
        text = text.to(device)
        label = label.to(device)
        prediction = model(text)
        loss = loss_function(prediction, label)
        loss.backward()
        optimizer.step()
    model.eval()
    val_prediction = []
    val_label = []
    with torch.no_grad():
        for text, label in val_loader:
            text = text.to(device)
            label = label.to(device)
            prediction = model(text).max(dim=1).indices
            val_prediction.extend(prediction.detach().cpu().tolist())
            val_label.extend(label.detach().cpu().tolist())
    micro_f1 = evaluate(val_prediction, val_label, False)
    if micro_f1 > max_micro_f1:
        max_micro_f1 = micro_f1
        torch.save(model.state_dict(), "best_model.pkl")
model.load_state_dict(torch.load("best_model.pkl"))
test_prediction = []
test_label = []
model.eval()
with torch.no_grad():
    for text, label in test_loader:
        text = text.to(device)
        label = label.to(device)
        prediction = model(text).max(dim=1).indices
        test_prediction.extend(prediction.detach().cpu().tolist())
        test_label.extend(label.detach().cpu().tolist())
evaluate(test_prediction, test_label, True)