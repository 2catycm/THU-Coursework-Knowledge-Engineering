import os
import re
import numpy as np
import gensim
from tqdm import tqdm

def load_text(file):
    print(f"loading from {file}")
    text = []
    f = open(file, "r+", encoding="utf-8")
    for i in tqdm(f.readlines()):
        content = i.strip().split("\t")
        _text = re.sub("\s+", "", content[1])
        temp = []
        for char in _text:
            temp.append(char)
        text.append(temp)
    return text


def load_word2vec_model(file=None, vector_size=100):
    # train word2vec with gensim
    if os.path.exists("word2vec"):
        word2vec_model = gensim.models.word2vec.Word2Vec.load("word2vec")
    else:
        text = load_text(file)
        # TODO
        # hint: use gensim.models.word2vec.Word2Vec
        raise NotImplementedError
        word2vec_model.save("word2vec")
    return word2vec_model

def get_word_embeddings(word2vec_model, vector_size=100, pad_token="<PAD>", unk_token="<UNK>"):
    text_vocab = word2vec_model.wv.key_to_index
    word_embeddings = np.zeros((len(text_vocab), vector_size))
    for char, index in text_vocab.items():
        if char not in [pad_token, unk_token]:
            word_embeddings[index] = word2vec_model.wv[char]
    return word_embeddings