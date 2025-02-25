#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import numpy as np


class HMM:
    """Hidden Markov Model

    Attributes:
        total_states (int): number of states, N
        pi (array, with shape (N,)): initial state probability
        A (array, with shape (N, N)): log transition probability. 
                                      A[i, j] means log transition prob from state i to state j.
                                      A.T[i, j] means log transition prob from state j to state i.
        B (array, with shape (N, T)): log emitting probability.
                                      B[i, k] means log emitting prob from state i to observation k.

    """

    def __init__(self, total_state, pi, A, B):
        self.total_states = total_state
        self.pi = pi
        self.A = A
        self.B = B


    def viterbi(self, ob):
        """Viterbi Decoding Algorithm.

        Args:
            ob (array, with shape(T,)): (o1, o2, ..., oT), observations

        Variables:
            delta (array, with shape(T, N)): delta[t, s] means max probability torwards state s at
                                             timestep t given the observation ob[0:t+1]
                                             给定观察ob[0:t+1]情况下t时刻到达状态s的概率最大的路径的概率
            phi (array, with shape(T, N)): phi[t, s] means prior state s' for delta[t, s]
                                           给定观察ob[0:t+1]情况下t时刻到达状态s的概率最大的路径的t-1时刻的状态s'

        Returns:
            best_path: np.array, shape T, the best state sequence

        """
        T = len(ob)
        delta = np.zeros((T, self.total_states))
        phi = np.zeros((T, self.total_states), int)
        best_path = np.zeros((T,), dtype=int)

        # TODO, note that we are using log probability matrix here

        return best_path

# %%
