import numpy as np
import torch
from torch import nn
from allennlp.modules.elmo import Elmo


class TextCNN(nn.Module):
    def __init__(
        self,
        options_file,
        weight_file,
        vector_size,
        filter_size=[2, 3, 4, 5],
        channels=64,
        max_length=1024,
    ):
        # Args:
        #   options_file: elmo file
        #   weight_file: elmo weight file
        #   vector_size: int, word embedding dim
        #   filter_size: List[int], kernel size for each layer of CNN
        #   channels: int, output channel for CNN
        #   max_length: int
        # Returns:
        #   None
        super(TextCNN, self).__init__()
        self.embedding = Elmo(options_file, weight_file, 1, dropout=0)
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
        output = []
        inputs = self.embedding(inputs)["elmo_representations"][0]
        #############
        # TODO
        ##############
        return output
