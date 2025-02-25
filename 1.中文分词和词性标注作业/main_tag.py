import re
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from util.tag_util import preprocess, compute_count_matrix
from util.viterbi import HMM

# 数据集预处理
lines = open("./data/corpus_POS.txt", "r+", encoding="gbk").readlines()
all_text, all_labels = preprocess(lines)

# 划分训练集和验证集
random.seed(19260817)
index = list(range(len(all_text)))
random.shuffle(index)
all_text = [all_text[i] for i in index]
all_labels = [all_labels[i] for i in index]

train_size = round(len(all_text) * 4 / 5)
train_text, train_labels = all_text[:train_size], all_labels[:train_size]
valid_text, valid_labels = all_text[train_size:], all_labels[train_size:]
print(len(all_text), len(train_text), len(valid_text))

# 用训练集构建词表和标签表，训练文本和标签、测试文本转为索引
text_vocab, tag_vocab = {"<UNK>": 0}, {"<UNK>": 0}
for i in range(train_size):
    for text, tag in zip(train_text[i], train_labels[i]):
        tag_vocab.setdefault(tag, len(tag_vocab))
        text_vocab.setdefault(text, len(text_vocab))
tag_list = list(tag_vocab.keys())
print(tag_vocab)

transform = lambda _list, func: [list(map(func, item)) for item in _list]
train_text = transform(train_text, lambda x: text_vocab.get(x, 0))
valid_text = transform(valid_text, lambda x: text_vocab.get(x, 0))
train_labels = transform(train_labels, lambda x: tag_vocab.get(x, 0))
print(train_text[0], "\n", train_labels[0])

# 计算初始概率、转移概率和发射概率矩阵
initial, transmission, emission = compute_count_matrix(
    train_text, train_labels, text_vocab, tag_vocab
)
# smoothing
smooth = 0.5
initial += 0.5
transmission += 0.5
emission += 0.5

"""
convert to log probability to ensure computational stability
"""
normalize = lambda matrix: np.log(
    np.einsum(
        "ij,i->ij" if len(matrix.shape) == 2 else "i,->i",
        matrix,
        1 / (matrix.sum(axis=-1) + 1e-8),
    )
)
initial, transmission, emission = (
    normalize(initial),
    normalize(transmission),
    normalize(emission),
)
print(f"{initial=},\n{transmission=},\n{emission=}")

# 使用viterbi动态规划在验证集上解码
model = HMM(len(tag_vocab), initial, transmission, emission)
valid_prediction = []

for text_list in valid_text:
    valid_prediction.append(model.viterbi(text_list))

# 计算实验结果指标
# 计算micro指标 对标签列表展平
valid_prediction_flatten = [tag_list[tag] for item in valid_prediction for tag in item]
valid_labels_flatten = [tag for item in valid_labels for tag in item]
print(valid_prediction_flatten[:10], "\n", valid_labels_flatten[:10])

precision, recall, f1, _ = precision_recall_fscore_support(
    valid_labels_flatten, valid_prediction_flatten, average="micro"
)
print(
    f"total={len(valid_prediction_flatten)},correct={sum(i1 == i2 for i1, i2 in zip(valid_prediction_flatten, valid_labels_flatten))}"
)
print("micro")
print(f"{precision=},{recall=},{f1=}")
