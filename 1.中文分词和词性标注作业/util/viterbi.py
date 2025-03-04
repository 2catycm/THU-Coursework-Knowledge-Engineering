#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
from fastcore.all import store_attr, patch
import numpy as np


class HMM:
    """Hidden Markov Model"""

    def __init__(
        self,
        total_states: int,  # number of states, N
        pi: np.ndarray,  # shape (N,) initial state probability, N
        A: np.ndarray,
        # shape (N, N) log transition probability.
        #   A[Xi, j] means log transition prob from state i to state j.
        #   A.T[i, j] means log transition prob from state j to state i.
        B: np.ndarray,
        # shape (N, T) log emitting probability.
        #   B[i, k] means log emitting prob from state i to observation k.
    ):
        store_attr()


@patch
def viterbi(
    self: HMM,  # 把方法绑定到类上
    ob: np.ndarray,  # shape (T, ) (o0, o1, ..., oT-1), observations
) -> np.ndarray:  # best_path: shape T, the best state sequence
    """Viterbi Decoding Algorithm.
    Variables:
    delta (array, with shape(T, N)): delta[t, s] means max probability torwards state s at
        timestep t given the observation ob[0:t+1]
        给定观察ob[0:t+1]情况下t时刻到达状态s的概率最大的路径的概率
    phi (array, with shape(T, N)): phi[t, s] means prior state s' for delta[t, s]
        给定观察ob[0:t+1]情况下t时刻到达状态s的概率最大的路径的t-1时刻的状态s'
    """
    T = len(ob)
    delta = np.zeros((T, self.total_states))
    phi = np.zeros((T, self.total_states), int)
    best_path = np.zeros((T,), dtype=int)

    # TODO, note that we are using log probability matrix here
    # 初始化
    # 当我看到 o0 的时候，实际状态 s0 是各个状态的概率
    delta[0] = self.pi + self.B[:, ob[0]]
    # 递推
    for t in range(1, T):
        for s in range(self.total_states):
            any_state_to_s_prob = delta[t - 1] + self.A[:, s]
            best_prior_state = np.argmax(any_state_to_s_prob)
            max_prob_to_s = any_state_to_s_prob[best_prior_state]
            local_observe_prob = self.B[s, ob[t]]
            # 当我看到 ot 的时候，实际状态 st 是 s 的时候，最大可能从哪个状态转移过来的
            delta[t, s] = max_prob_to_s + local_observe_prob
            phi[t, s] = best_prior_state
    # 回溯
    # 首先确定最后一个状态是什么
    best_path[-1] = np.argmax(delta[-1])
    # 然后根据phi回溯
    for t in range(-2, -(T + 1), -1):  # -2 一直到 -T
        best_path[t] = phi[t + 1, best_path[t + 1]]
    return best_path


# %%
