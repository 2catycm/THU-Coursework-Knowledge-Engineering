from tqdm import tqdm
import jieba
import os

# 数据预处理，训练集使用官方训练集，转化为平行语料对（测试集直接使用官方测试集不需要处理）

train_cache = "./processed/seg.train"
if not os.path.exists(train_cache):
    lines = open(
        "./raw/NLPCC2018_GEC_TrainingData/data.train", "r", encoding="utf-8"
    ).readlines()
    output = []
    for line in tqdm(lines[:1000]):
        _, count, source, *targets = line.strip().split("\t")
        cut = lambda text: " ".join(jieba.lcut(text))
        cut_source = cut(source)
        if int(count) != len(targets):
            continue
        output.extend([f"{cut_source}\t{cut(target)}\n" for target in targets])
    output = list([x for x in output if len(x.strip().split("\t")) == 2])
    open(train_cache, "w", encoding="utf-8").writelines(output)
