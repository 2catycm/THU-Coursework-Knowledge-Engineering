import collections
import math
import torch
from typing import List


def ngrams_iterator(token_list, ngrams):
    """
    返回一个迭代器，该迭代器生成给定的token和它们的ngrams。
    Args:
        token_list: 一个包含tokens的列表。
        ngrams: ngrams的数量。
    """

    def _get_ngrams(n):
        return zip(*[token_list[i:] for i in range(n)])

    for x in token_list:
        yield x
    for n in range(2, ngrams + 1):
        for x in _get_ngrams(n):
            yield " ".join(x)


def _compute_ngram_counter(tokens, max_n):
    """
    计算给定的tokens的ngrams的计数器。
    Args:
        tokens: 一个可迭代对象，包含tokens。
        max_n: 最大的ngram的数量。
    Returns:
        一个collections.Counter，包含ngrams的计数器。
    """
    ngrams_counter = collections.Counter(
        tuple(x.split(" ")) for x in ngrams_iterator(tokens, max_n)
    )
    return ngrams_counter


def bleu_score(
    candidate_corpus: List[List[str]],  # 候选翻译的token列表的列表。
    references_corpus: List[
        List[str]
    ],  # 参考翻译的token列表的列表，数量应与候选一一对应。
    max_n=4,  # 最大的ngram的数量，默认4。
    weights: List[float] = [0.25]
    * 4,  # 用于计算加权几何平均时的权重列表，长度应为max_n。
    verbose: bool = True,
) -> float:  # BLEU分数（0到1之间）。
    """
    计算候选翻译语料库和参考翻译语料库之间的BLEU分数。
    """
    assert len(candidate_corpus) == len(references_corpus), (
        "候选翻译和参考翻译的数量必须一致。"
    )
    assert len(weights) == max_n, "权重列表的长度必须等于最大的ngram数量。"

    total_clip_count = [0] * max_n
    total_candidate_ngrams = [0] * max_n
    total_candidate_length = 0
    total_reference_length = 0

    for candidate, references in zip(candidate_corpus, references_corpus):
        candidate_ngrams = _compute_ngram_counter(candidate, max_n)
        reference_ngrams = [_compute_ngram_counter(ref, max_n) for ref in references]
        max_reference_ngrams = collections.Counter()
        for ref_ngrams in reference_ngrams:
            for ngram, count in ref_ngrams.items():
                max_reference_ngrams[ngram] = max(max_reference_ngrams[ngram], count)

        for n in range(1, max_n + 1):
            for ngram, count in candidate_ngrams.items():
                if len(ngram) == n:
                    total_candidate_ngrams[n - 1] += count
                    total_clip_count[n - 1] += min(count, max_reference_ngrams[ngram])

        candidate_length = len(candidate)
        total_candidate_length += candidate_length
        reference_lengths = [len(ref) for ref in references]
        closest_ref_length = min(
            reference_lengths, key=lambda x: abs(x - candidate_length)
        )
        total_reference_length += closest_ref_length

    precisions = []
    for clip_count, candidate_ngrams in zip(total_clip_count, total_candidate_ngrams):
        if candidate_ngrams == 0:
            precisions.append(0)
        else:
            precisions.append(clip_count / candidate_ngrams)

    if verbose:
        print(f"Precisions: {precisions}")
        print(f"Total candidate length: {total_candidate_length}")
        print(f"Total reference length: {total_reference_length}")

    if total_candidate_length == 0:
        return 0

    brevity_penalty = (
        1
        if total_candidate_length >= total_reference_length
        else math.exp(1 - total_reference_length / total_candidate_length)
    )

    if verbose:
        print(f"Brevity penalty: {brevity_penalty}")

    log_precisions = [math.log(p) if p > 0 else float("-inf") for p in precisions]
    bleu = brevity_penalty * math.exp(
        sum(w * p for w, p in zip(weights, log_precisions))
    )

    return bleu
