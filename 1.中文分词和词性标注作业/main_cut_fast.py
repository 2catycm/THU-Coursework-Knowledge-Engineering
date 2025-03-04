import re
import random
from tqdm import tqdm
import jieba
from typing import *
from util.cut_util import maximum_match_cut, get_final_result, jieba_cut, evaluate
from util.cut_util import running_stats
from util.trie import Trie

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--my_method", action="store_true", default=False)
parser.add_argument("--jieba_method", action="store_true", default=False)
args = parser.parse_args()

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
# vocab = set([word for sent in train_set for word in sent])
# inverted_vocab = set(map(lambda x: x[::-1], vocab))
# print("Vocab Size:", len(vocab))

vocab_trie = Trie()
inverted_vocab_trie = Trie(reverse=True)
for sent in train_set:
    for word in sent:
        if len(word) <= 4:
            vocab_trie.insert(word)
            inverted_vocab_trie.insert(word[::-1])

print("Vocab Size:", len(vocab_trie))


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


from util.cut_util import maximum_match_cut_fast

if args.my_method:
    # 计算双向匹配法分词结果
    max_size = 4
    valid_result = []
    for item in tqdm(valid_text):
        forward_result = maximum_match_cut_fast(item, vocab_trie, max_size=max_size)
        backward_result = maximum_match_cut_fast(
            item[::-1], inverted_vocab_trie, max_size=max_size
        )
        # re-compute backward matching index
        backward_result = [
            (len(item) - i[1], len(item) - i[0]) for i in backward_result[::-1]
        ]
        result = get_final_result(backward_result, forward_result)
        valid_result.append(result)
    print("Result Sample:", valid_result[0])
    print(f"双向匹配法使用统计 {running_stats}")

    # 计算效果指标
    for macro_or_micro in [True, False]:
        p, r, f = evaluate(valid_result, valid_label, macro_or_micro=macro_or_micro)
        print(
            f"双向最大匹配算法, precision={p}, recall={r}, f1={f}, macro_or_micro={macro_or_micro}"
        )

if args.jieba_method:
    # 使用jieba进行分词

    jieba_result = jieba_cut(valid_text)
    jieba_result_with_vocab = jieba_cut(valid_text, train_set=train_set)

    for macro_or_micro in [True, False]:
        for result, name in zip(
            [jieba_result, jieba_result_with_vocab], ["默认模式", "增加训练集词库"]
        ):
            p, r, f = evaluate(result, valid_label, macro_or_micro=macro_or_micro)
            print(
                f"jieba分词({name}), precision={p}, recall={r}, f1={f}, macro_or_micro={macro_or_micro}"
            )
