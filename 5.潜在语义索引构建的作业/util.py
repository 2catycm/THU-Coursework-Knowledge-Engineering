import jieba
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List


import chardet

import joblib

memory = joblib.Memory(location="cache", verbose=0)


@memory.cache
def load_words(
    filepath: str,  # 要加载的文件路径
) -> Tuple[List[str], List[List[str]]]:  # (所有词汇列表, 文档分词列表)
    """读取文件并使用jieba分词"""
    with open(filepath, "rb") as f:
        raw_data = f.read()
        # 使用 chardet 检测编码
        result = chardet.detect(raw_data)
    encoding = result["encoding"]
    confidence = result["confidence"]
    if confidence < 0.7:
        print(
            f"Warning: low confidence ({confidence}) for encoding{encoding}. Using utf-8 instead."
        )
        encoding = "utf-8"
    print(
        f"Loading file {filepath} with encoding {encoding}, confidence {confidence}. "
    )

    with open(filepath, "r", encoding=encoding) as f:
        documents = []
        all_terms = set()

        for line in f:
            line = line.strip()
            if not line:
                continue

            # 使用jieba分词
            words = jieba.lcut(line)
            documents.append(words)
            all_terms.update(words)

    return all_terms, documents


def build_term_doc_matrix(
    documents: List[List[str]],  # 文档分词列表 (N个文档)
    terms: List[str],  # 所有词汇列表 (D个词汇)
) -> np.ndarray:  # N×D的矩阵，term_doc[i,j]表示词j在文档i中的出现次数
    """
    build term-document matrix
    """
    # 创建词汇到索引的映射
    term_index = {term: idx for idx, term in enumerate(terms)}

    # 初始化矩阵
    matrix = np.zeros((len(documents), len(terms)), dtype=int)

    # 填充矩阵
    for doc_idx, doc in enumerate(documents):
        for word in doc:
            if word in term_index:
                matrix[doc_idx, term_index[word]] += 1

    return matrix


def cal_tfidf_matrix(
    term_doc: np.ndarray,  # 词-文档矩阵
    documents: List[List[str]],
    terms: List[str],
) -> dict:  # TF-IDF值字典 {word: tfidf_value}
    """
    calculate TF-IDF value for each word
    """
    # TF计算
    tf = term_doc.astype(float)
    doc_lens = tf.sum(axis=1)
    tf = tf / doc_lens[:, np.newaxis]  # 归一化

    # IDF计算
    df = (term_doc > 0).sum(axis=0)
    idf = np.log(len(documents) / (df + 1e-6))  # 避免除以0

    # TF-IDF计算
    tfidf = tf * idf

    # 转换为字典
    return {term: tfidf[:, idx].mean() for idx, term in enumerate(terms)}


def search_key_similarity(
    U: np.ndarray,  # SVD分解的U矩阵
    s: np.ndarray,  # 奇异值数组
    VT: np.ndarray,  # SVD分解的VT矩阵
    terms: List[str],  # 词汇列表
    term_doc: np.ndarray,
    keys: List[str],  # 查询关键词列表
    k: int = 10,  # 保留的奇异值数量
) -> np.ndarray:  # 相似度矩阵 (N文档, K关键词)
    """
    计算LSI相似度矩阵
    """
    # 构建查询向量
    term_index = {term: idx for idx, term in enumerate(terms)}
    query_vectors = []

    for key in keys:
        words = jieba.lcut(key)
        vec = np.zeros(len(terms))
        for word in words:
            if word in term_index:
                vec[term_index[word]] += 1
        query_vectors.append(vec)

    query_matrix = np.array(query_vectors).T  # D x K

    # 降维处理
    sigma_k = np.diag(s[:k])
    U_k = U[:, :k]
    VT_k = VT[:k, :]

    # 文档在隐空间中的表示
    doc_rep = U_k @ sigma_k  # N x k

    # 查询在隐空间中的表示
    query_rep = np.linalg.pinv(sigma_k) @ VT_k @ query_matrix  # k x K

    # 计算余弦相似度
    return cosine_similarity(doc_rep, query_rep.T)


def classification(
    sim_matrix: np.ndarray,  # 相似度矩阵 (N文档, K关键词)
) -> np.ndarray:  # 预测的类别索引 (N文档,)
    """
    文档分类：为每个文档选择最相似的关键词
    """
    return np.argmax(sim_matrix, axis=1)


def search_topn_for_each_key(
    sim_matrix: np.ndarray,  # 相似度矩阵 (N文档, K关键词)
    n: int = 10,  # 保留的Top-N结果数量
) -> np.ndarray:  # 搜索结果矩阵 (K关键词, n文档索引)
    """
    为每个关键词搜索Top-N文档
    """
    # 按列排序（每个关键词对应一列）
    return np.argsort(-sim_matrix, axis=0)[:n, :].T
