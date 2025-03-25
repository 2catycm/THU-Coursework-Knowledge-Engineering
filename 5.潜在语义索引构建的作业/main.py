import os
import jieba
import numpy as np
from scipy.linalg import svd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from util import (
    load_words,
    build_term_doc_matrix,
    cal_tfidf_matrix,
    search_key_similarity,
    classification,
    search_topn_for_each_key,
)

filePath = "./data"
fileList = sorted(os.listdir(filePath))


# 评测函数
def evaluate(prediction, label, flag):
    precision, recall, f1, _ = precision_recall_fscore_support(
        label, prediction, average=None, labels=sorted(list(set(label)))
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        label, prediction, average="micro"
    )
    if flag:
        print("各类别Precision:", [float("{:.4f}".format(i)) for i in precision])
        print("各类别Recall:", [float("{:.4f}".format(i)) for i in recall])
        print("各类别F1:", [float("{:.4f}".format(i)) for i in f1])
        print("整体微平均Precision:", float("{:.4f}".format(micro_precision)))
        print("整体微平均Recall:", float("{:.4f}".format(micro_recall)))
        print("整体微平均F1:", float("{:.4f}".format(micro_f1)))
    return micro_f1


if __name__ == "__main__":
    # 预处理文件得到中文分词
    documents = []
    terms = []
    label = []
    keys = []
    for file in fileList:
        terms_tmp, documents_tmp = load_words(os.path.join(filePath, file))
        documents.extend(documents_tmp)
        terms.extend(terms_tmp)

        # load labels and keys for search
        _label = file.split(".")[0]
        # a list of labels for each document
        label.extend([len(keys)] * len(documents_tmp))
        keys.append(_label)

    terms = sorted(list(set(terms)))

    # 构造Term-Document矩阵
    term_doc = build_term_doc_matrix(documents, terms)
    np.save("term_doc.npy", term_doc)
    # 加载Term-Document矩阵
    term_doc = np.load("term_doc.npy")
    # 计算TF-IDF
    TF_IDF = cal_tfidf_matrix(term_doc, documents, terms)
    print(
        "TOP10单词的TF-TDF如下所示\n",
        sorted(TF_IDF.items(), key=lambda x: x[1], reverse=True)[:10],
    )
    # SVD奇异值分解
    U, s, VT = svd(term_doc)
    np.save("U.npy", U)
    np.save("s.npy", s)
    np.save("VT.npy", VT)
    U = np.load("U.npy")
    s = np.load("s.npy")
    VT = np.load("VT.npy")
    print("U:", U.shape)
    print("s:", s.shape)
    print("VT:", VT.shape)
    # 改变LSI矩阵的K值
    n = 5  # 查询top-n
    for k in [10, 20, 30, 40, 50, 100]:
        sim_matrix = search_key_similarity(U, s, VT, terms, term_doc, keys, k=k)
        searched_topn = search_topn_for_each_key(sim_matrix, n=n).astype(int)
        searched_topn_label = np.array(
            [label[i] for i in searched_topn.reshape(-1)]
        ).reshape(searched_topn.shape[0], -1)
        print(searched_topn_label[:5])
        print("查询关键词为:", keys)
        # print("分类结果为:", [np.sum(predict == i) for i in range(len(keys))])
        # 查询
        print(f"查询结果top-{n}:", searched_topn)
        print(
            f"查询结果top-{n}准确率:",
            [
                np.sum(searched_topn_label[i] == i)
                / np.sum(searched_topn_label[i] >= 0)
                for i in range(len(keys))
            ],
        )
        print("矩阵K值为:", k)

        # 分类，将key当作类别标签
        print("分类结果:")
        predict = classification(sim_matrix)
        evaluate(predict, label, True)
