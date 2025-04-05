import einops
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch
from typing import *

class GECModel(nn.Module):
    """
    使用GRU构建Encoder-Decoder
    Elmo作为Encoder Embedding
    Decoder Embedding 也可以用elmo

    Args:
        elmo_model: ELmo 模型
        vocab_size: 词表大小

    """
    def __init__(self, elmo_model, vocab_size:int):
        super().__init__()
        # TODO

    def encode(self, source_inputs: List[List[str]], source_mask: torch.Tensor, **kwargs):
        """
        Encode input source text
        Args:
            source_inputs: a list of input text
            source_mask: torch.tensor of size (batch_size, sequence_length)
        Returns:
            encoder_outputs: torch.Tensor of size (batch_size, sequence_length, hidden_states)
        """
        
        # TODO
        raise NotImplementedError

    def decode(self, encoder_outputs: torch.Tensor, source_mask: torch.Tensor, target_input_ids: torch.Tensor, target_inputs: List[List[str]], target_mask: torch.Tensor=None, hidden_states: torch.Tensor=None, **kwargs):
        """
        decode for output sequence
        Args
            encoder_outputs: a torch.Tensor of (batch_size, sequence_length, hidden_states) output by encoder
            source_mask: torch.tensor of size (batch_size, sequence_length)
            target_input_ids: torch.tensor of size (batch_size, sequence_length)
            target_inputs: a list of target text
            target_mask: torch.tensor of size (batch_size, sequence_length)
        Returns:
            decoder_output: a torch.Tensor of (batch_size, sequence_length, vocab_size)
        """
        
        # TODO
        raise NotImplementedError

    def forward(self, **kwargs):
        encoder_outputs = self.encode(**kwargs)
        decoder_outputs = self.decode(encoder_outputs, **kwargs)
        return decoder_outputs