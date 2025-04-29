import json
from transformers import BertTokenizer
import numpy as np


def build_vocab(tokenizer, file_path: str, vocab_path: str, vocab_size=20000):
    '''从文件中构建词汇到id的映射，以dict的形式存储到json文件中，词汇表的大小最大限制为vocab_size'''
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    vocab = {}
    for line in lines:
        tokens = tokenizer.tokenize(line.strip())
        for token in tokens:
            if token not in vocab:
                vocab[token] = 1
            else:
                vocab[token] += 1
    word2id = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    for word, _ in sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:vocab_size]:
        word2id[word] = len(word2id)
    json.dump(word2id, open(vocab_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)


def preprocess(tokenizer, file_path: str, source_vocab_path: str, target_vocab_path: str, max_length=64, ratio=0.9):
    '''将平行语料文件转换为id序列文件，随机打乱并按照比例ratio切分为训练集和验证集'''
    source_vocab = json.load(open(source_vocab_path, "r", encoding="utf-8"))
    target_vocab = json.load(open(target_vocab_path, "r", encoding="utf-8"))
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    source = []
    target = []
    for line in lines:
        source_line, target_line = line.strip().split("\t")[:2]
        source_tokens = tokenizer.tokenize(source_line)
        target_tokens = tokenizer.tokenize(target_line)
        if len(source_tokens) > max_length or len(target_tokens) > max_length:
            continue
        source_tokens = ["<BOS>"] + source_tokens + ["<EOS>"]
        target_tokens = ["<BOS>"] + target_tokens + ["<EOS>"]
        source_ids = [source_vocab.get(token, source_vocab["<UNK>"]) for token in source_tokens]
        target_ids = [target_vocab.get(token, target_vocab["<UNK>"]) for token in target_tokens]
        source.append(source_ids)
        target.append(target_ids)
    # 随机打乱
    all_data = list(zip(source, target))
    np.random.shuffle(all_data)
    # 切分训练集和验证集
    train_data = all_data[:int(len(all_data) * ratio)]
    valid_data = all_data[int(len(all_data) * ratio):]
    return train_data, valid_data


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    source_file_path = "data/raw/english.txt"
    target_file_path = "data/raw/french.txt"
    source_vocab_path = "data/processed/source_vocab.json"
    target_vocab_path = "data/processed/target_vocab.json"
    build_vocab(tokenizer, source_file_path, source_vocab_path)
    build_vocab(tokenizer, target_file_path, target_vocab_path)
    train_data, valid_data = preprocess(tokenizer, "data/raw/fra.txt", source_vocab_path, target_vocab_path)
    with open("data/processed/train.json", "w", encoding="utf-8") as f:
        for source, target in train_data:
            f.write(json.dumps({"source": source, "target": target}, ensure_ascii=False) + "\n")
    with open("data/processed/valid.json", "w", encoding="utf-8") as f:
        for source, target in valid_data:
            f.write(json.dumps({"source": source, "target": target}, ensure_ascii=False) + "\n")
