import jieba
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List


def load_words(filepath: str) -> Tuple[List[str], List[List[str]]]:
    """读取文件并使用jieba分词
    filepath: str # 要加载的文件路径
    returns: 
        tuple[List[str], List[List[str]]] # (所有词汇列表, 文档分词列表)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
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
    
    # 转换为有序列表
    terms = sorted(list(all_terms))
    return terms, documents


def build_term_doc_matrix(documents: List[List[str]], terms: List[str]):
    """
    build term-document matrix
    documents: List[List[str]] # 文档分词列表 (N个文档)
    terms: List[str] # 所有词汇列表 (D个词汇)
    returns: np.ndarray # N×D的矩阵，term_doc[i,j]表示词j在文档i中的出现次数
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


def cal_tfidf_matrix(term_doc: np.ndarray, documents: List[List[str]], terms: List[str]):
    """
    calculate TF-IDF value for each word
    term_doc: np.ndarray # 词-文档矩阵
    returns: dict # TF-IDF值字典 {word: tfidf_value}
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
    U: np.ndarray,
    s: np.ndarray,
    VT: np.ndarray,
    terms: List[str],
    term_doc: np.ndarray,
    keys: List[str],
    k: int = 10,
):
    """
    计算LSI相似度矩阵
    U: np.ndarray # SVD分解的U矩阵
    s: np.ndarray # 奇异值数组
    VT: np.ndarray # SVD分解的VT矩阵
    terms: List[str] # 词汇列表
    keys: List[str] # 查询关键词列表
    k: int # 保留的奇异值数量
    returns: np.ndarray # 相似度矩阵 (N文档, K关键词)
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


def classification(sim_matrix: np.ndarray):
    """
    文档分类：为每个文档选择最相似的关键词
    sim_matrix: np.ndarray # 相似度矩阵 (N文档, K关键词)
    returns: np.ndarray # 预测的类别索引 (N文档,)
    """
    return np.argmax(sim_matrix, axis=1)


def search_topn_for_each_key(sim_matrix: np.ndarray, n: int = 10):
    """
    为每个关键词搜索Top-N文档
    sim_matrix: np.ndarray # 相似度矩阵 (N文档, K关键词)
    n: int # 保留的Top-N结果数量
    returns: np.ndarray # 搜索结果矩阵 (K关键词, n文档索引)
    """
    # 按列排序（每个关键词对应一列）
    return np.argsort(-sim_matrix, axis=0)[:n, :].T
