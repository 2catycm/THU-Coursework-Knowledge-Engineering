import numpy as np  # type: ignore
import torch  # type: ignore
from torch import nn  # type: ignore


class TextCNN(nn.Module):
    def __init__(
        self,
        word_embeddings,
        vector_size,
        label2index,
        pad_index,
        filter_size=[2, 3, 4, 5],
        channels=64,
        max_length=1024,
    ):
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
        # Initialize embedding layer with pre-trained word_embeddings
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(word_embeddings), freeze=False, padding_idx=pad_index
        )
        # Build a stack of 1D CNN layers for each filter size
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=vector_size, out_channels=channels, kernel_size=k)
                for k in filter_size
            ]
        )
        # Final linear layer for label prediction; number of classes equals len(label2index)
        num_class = len(label2index)
        self.linear = nn.Linear(channels * len(filter_size), num_class)

    def forward(self, inputs):
        # Args:
        #   inputs: torch.tensor of size N*L
        # Returns:
        #   predicted_logits: torch.tensor of size N*C (number of classes)
        #############
        # TODO
        ##############
        raise NotImplementedError
