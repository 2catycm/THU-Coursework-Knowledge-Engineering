import re
import random
from tqdm import tqdm
import jieba
from typing import *
from util.cut_util import maximum_match_cut, get_final_result, jieba_cut, evaluate

# 预处理
lines = open("./data/corpu.txt", "r+", encoding="gbk").readlines()
corpus = list(
    map(
        lambda line: list(
            map(
                lambda item: re.sub(r"^\[|/[a-zA-Z]+", "", item),
                line.strip().split(" "),
            )
        ),
        lines,
    )
)
print("Corpus Size:", len(corpus))
print("Corpus Samples:", corpus[:3])

# 训练集验证集划分
random.seed(19260817)
random.shuffle(corpus)
train_size = round(len(corpus) * 4 / 5)
train_set, valid_set = corpus[:train_size], corpus[train_size:]
print("Train:", len(train_set), ", Valid:", len(valid_set))

# 词表构建
vocab = set([word for sent in train_set for word in sent])
inverted_vocab = set(map(lambda x: x[::-1], vocab))
print("Vocab Size:", len(vocab))

# 验证集重构
valid_text, valid_label = [], []
for words in valid_set:
    valid_text.append("".join(words))
    valid_label.append([])
    index = 0
    for word in words:
        valid_label[-1].append((index, index + len(word)))
        index += len(word)
print("Valid Sample:\n", valid_text[0], "\n", valid_label[0])

# # 计算双向匹配法分词结果
# max_size = 4
# valid_result = []
# for item in tqdm(valid_text):
#     forward_result = maximum_match_cut(item, vocab, max_size=max_size)
#     backward_result = maximum_match_cut(item[::-1], inverted_vocab, max_size=max_size)
#     # re-compute backward matching index
#     backward_result = [
#         (len(item) - i[1], len(item) - i[0]) for i in backward_result[::-1]
#     ]
#     result = get_final_result(backward_result, forward_result)
#     valid_result.append(result)
# print("Result Sample:", valid_result[0])

# # 计算效果指标

# p, r, f = evaluate(valid_result, valid_label)
# print(f"双向最大匹配算法, precision={p}, recall={r}, f1={f}")

# 使用jieba进行分词
jieba_result = jieba_cut(valid_text)
jieba_result_with_vocab = jieba_cut(valid_text, train_set=train_set)

for macro_or_micro in [True, False]:
    for result, name in zip([jieba_result, jieba_result_with_vocab], ["默认模式", "增加训练集词库"]):
        p, r, f = evaluate(result, valid_label, macro_or_micro=macro_or_micro)
        print(f"jieba分词({name}), precision={p}, recall={r}, f1={f}, macro_or_micro={macro_or_micro}")

