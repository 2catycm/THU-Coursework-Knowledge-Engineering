import numpy as np
import torch
from torch import nn


class TextCNN(nn.Module):
    def __init__(self, word_embeddings, vector_size, label2index, pad_index, filter_size=[2,3,4,5], channels=64, max_length=1024):
        # Args:
        #   word_embeddings: np.array of size N*D, containing pretrianed word2vec embedding
        #   vector_size: int, word embedding dim
        #   label2index: Dict
        #   pad_index: int
        #   filter_size: List[int], kernel size for each layer of CNN
        #   channels: int, output channel for CNN
        #   max_length: int
        # Returns:
        #   None
        super(TextCNN, self).__init__()
        ####################
        # TODO
        # 1.initialize embedding layer with word_embeddings
        # 2.build a stack of 1-d CNNs with designated kernel size
        # e.g. with filter_size=[2,3,4,5], 4 layers of CNN should be built and kernel size is set to 2,3,4,5, respectively.
        # 3. the last linear layer for label prediction
        #####################
        raise NotImplementedError

    def forward(self, inputs):
        # Args:
        #   inputs: torch.tensor of size N*L
        # Returns:
        #   predicted_logits: torch.tensor of size N*C (number of classes)
        #############
        # TODO
        ##############
        raise NotImplementedError