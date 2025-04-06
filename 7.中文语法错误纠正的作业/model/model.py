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

    def __init__(self, elmo_model, vocab_size: int, hidden_size: int = 512, num_layers: int = 1, dropout: float = 0.5):
        super().__init__()
        self.elmo = elmo_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoder_gru = nn.GRU(
            input_size=elmo_model.get_output_dim(),
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )
        self.decoder_gru = nn.GRU(
            input_size=elmo_model.get_output_dim() + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.linear = nn.Linear(hidden_size, vocab_size)
# Removed
        self.target_embedding = nn.Embedding(vocab_size, elmo_model.get_output_dim())
        self.dropout_layer = nn.Dropout(dropout)

    def encode(
        self, source_inputs: List[List[str]], source_mask: torch.Tensor, **kwargs
    ):
        """
        Encode input source text
        Args:
            source_inputs: a list of input text
            source_mask: torch.tensor of size (batch_size, sequence_length)
        Returns:
            encoder_outputs: torch.Tensor of size (batch_size, sequence_length, hidden_states)
        """

        # Encode the source inputs using ELMo
        with torch.no_grad():
            elmo_outputs = self.elmo(source_inputs)
        # Apply the encoder GRU
        encoder_outputs, _ = self.encoder_gru(elmo_outputs)
        return encoder_outputs

    def decode(
        self,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_inputs: List[List[str]],
        target_mask: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ):
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

        # Use target embedding and decoder GRU with attention
        embedded = self.target_embedding(target_input_ids)
        embedded = self.dropout_layer(embedded)
        # Apply attention
        context, _ = self.attention(embedded, encoder_outputs, encoder_outputs)
        combined = torch.cat((embedded, context), dim=2)
        decoder_outputs, _ = self.decoder_gru(combined)
        outputs = self.linear(decoder_outputs)
        return outputs

    def forward(self, **kwargs):
        encoder_outputs = self.encode(**kwargs)
        decoder_outputs = self.decode(encoder_outputs, **kwargs)
        return decoder_outputs
