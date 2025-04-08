import einops
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch
from typing import List, Optional, Union, Tuple
import torch.nn.functional as F

from elmoformanylangs import Embedder

from fla.layers.rwkv7 import RWKV7Attention

class GECModel(nn.Module):
    """使用GRU构建的Encoder-Decoder模型，用于中文语法纠错任务"""

    def __init__(
        self,
        elmo_model: Embedder,  # ELMo模型，用于获取词向量表示
        vocab_size: int,  # 词表大小，用于构建目标端词嵌入层
        hidden_size: int = 512,  # 隐藏层维度，默认512
        num_layers: int = 1,  # GRU层数，默认1
        dropout: float = 0.5,  # Dropout概率，默认0.5
        # bidirectional=True, # 是否使用双向GRU，默认True
        elmo_dim=1024,  # ELMo projection_dim  这个比较难读取，所以作为参数
    ):
        super().__init__()
        self.elmo = elmo_model
        hidden_size = elmo_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_rwkv = False
        if self.use_rwkv:
            self.encoder_gru = RWKV7Attention(mode="chunk", 
            hidden_size=hidden_size, 
            head_dim=64)
        else:
            self.encoder_gru = nn.GRU(
                input_size=elmo_dim,  # ELMo projection_dim from config
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                # bidirectional=bidirectional
                dropout=dropout if num_layers > 1 else 0,
            )


        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, batch_first=True
        )
        self.decoder_gru = nn.GRU(
            input_size=elmo_dim
            + hidden_size,  # ELMo dim + hidden_size for attention concat
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.target_embedding = nn.Embedding(vocab_size, elmo_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding2query = nn.Linear(elmo_dim, hidden_size)

    def encode(
        self,
        source_inputs: List[List[str]],  # a list of input text
        source_mask: torch.Tensor,  # size (batch_size, sequence_length)
        **kwargs,
    ) -> (
        torch.Tensor
    ):  # encoder_outputs, size (batch_size, sequence_length, hidden_states)
        """
        Encode input source text, using source_mask to pack padded sequences.
        """
        # Encode the source inputs using ELMo
        device = next(self.parameters()).device
        with torch.no_grad():
            elmo_outputs = self.elmo.sents2elmo(source_inputs)
            elmo_outputs = torch.from_numpy(np.array(elmo_outputs)).float().to(device)
        
        if self.use_rwkv:
            return self.encoder_gru(elmo_outputs)
        else:
            # Compute lengths from source_mask (assumes mask with 1 for valid tokens)
            lengths = source_mask.sum(dim=1)
            # Pack the padded sequence using the computed lengths
            packed_input = pack_padded_sequence(
                elmo_outputs, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            # Apply the encoder GRU
            packed_outputs, _ = self.encoder_gru(packed_input)
            # Unpack the sequence
            encoder_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
            return encoder_outputs

    def decode(
        self,
        encoder_outputs: torch.Tensor,  # a torch.Tensor of (batch_size, sequence_length, hidden_states) output by encoder
        source_mask: torch.Tensor,  # a torch.Tensor of (batch_size, sequence_length) mask for source inputs
        target_input_ids: torch.Tensor,  # torch.tensor of size (batch_size, sequence_length)
        target_inputs: Optional[List[List[str]]] = None,  # a list of target text
        target_mask: Optional[
            torch.Tensor
        ] = None,  # torch.tensor of size (batch_size, sequence_length)
        hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.Tensor, torch.Tensor
    ]:  # decoder_output: a torch.Tensor of (batch_size, sequence_length, vocab_size)
        """
        decode for output sequence
        """

        # Use target embedding and decoder GRU with attention
        embedded = self.target_embedding(target_input_ids)
        embedded = self.dropout_layer(embedded)
        # Apply attention
        # We need to ensure all tensors have the same device
        device = next(self.parameters()).device
        embedded = embedded.to(device)
        encoder_outputs = encoder_outputs.to(device)
        embedded_q = self.embedding2query(embedded)
        context, _ = self.attention(
            embedded_q,
            encoder_outputs,
            encoder_outputs,
            key_padding_mask=~source_mask.bool(),
        )
        combined = torch.cat((embedded, context), dim=2)
        decoder_outputs, hidden_states = self.decoder_gru(combined, hidden_states)
        outputs = self.linear(decoder_outputs)
        return outputs, hidden_states

    def forward(self, **kwargs):
        encoder_outputs = self.encode(**kwargs)
        decoder_outputs = self.decode(encoder_outputs, **kwargs)
        return decoder_outputs
