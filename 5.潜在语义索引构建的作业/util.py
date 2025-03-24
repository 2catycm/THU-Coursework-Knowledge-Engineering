import jieba
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import *


def load_words(filepath: str) -> Tuple[List[str], List[List[str]]]:
    """
    read file and load parsed words with jieba, each line is one document
    Args:
        filepath: file path to load
    Returns:
        terms: List[str], a list of all words
        documents: List[List[str]], a list of documents, each sublist contains words in the document
    """
    f = open(filepath)
    documents = []
    terms = []

    # TODO

    f.close()
    return terms, documents


def build_term_doc_matrix(documents: List[List[str]], terms: List[str]):
    """
    build term-document matrix
    Args:
        documents: List[List[str]], a list of documents, each sublist contains words in the document, len(documents) = N
        terms: List[str], a list of all words, len(terms) = D
    Returns:
        term_doc: np.array, of size N * D, where term_doc[i, j] means the number of times word_j appears in document_i
    """
    term_doc = []

    # TODO

    return term_doc


def cal_tfidf_matrix(term_doc: np.array, documents: List[List[str]], terms: List[str]):
    """
    calculate TF-IDF value for each word
    Args:
        term_doc: N * D term-document matrix
        documents: List[List[str]], a list of documents, each sublist contains words in the document, len(documents) = N
        terms: List[str], a list of all words, len(terms) = D
    Returns:
        TF-IDF: Dict[int, float], where TF-IDF[word_i] is the tf-odf value for word_i
    """
    TF_IDF = {}

    # TODO

    return TF_IDF


def search_key_similarity(
    U: np.array,
    s: np.array,
    VT: np.array,
    terms: List[str],
    term_doc: np.array,
    keys: List[str],
    k: int = 10,
):
    """
    calculate cosine similarity for each pair of key and document
    U, s, VT are SVD results of term-document matrix
    ```
    U, s, VT = scipy.linalg.svd(term_doc)
    ```
    Args:
        U: numpy matrix of size (D,D)
        s: numpy array of size (N,)
        VT: numpy array of size
        terms: List[str], a list of all words, len(terms) = D
        term_doc: N * D term-document matrix
        keys: a list of search keys
        k: number of features used in matrix approximation
    Returns:
        sim_matrix: np.array of size (N, len(keys))
    """
    # TODO
    raise NotImplementedError


def classification(sim_matrix: np.array):
    """
    view as a document classification problem, return the most similar key index for each document
    Args:
        sim_matrix: np.array of size (N, len(keys))
    Returns:
        predict: np.array of size (N,)
    """
    # TODO
    raise NotImplementedError


def search_topn_for_each_key(sim_matrix: np.array, n: int = 10):
    """
    view as a search problem, return the top-n most similar document index for each keyword
    Args:
        sim_matrix: np.array of size (N, len(keys))
    Returns:
        searched: np.array of size (len(keys), n)
    """
    # TODO
    raise NotImplementedError
