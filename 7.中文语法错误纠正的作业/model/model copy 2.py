import einops
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch
from typing import List, Dict, Tuple, Optional
from elmoformanylangs import Embedder


class GECModel(nn.Module):
    """
    使用GRU构建的Encoder-Decoder模型，用于中文语法纠错任务
    """

    def __init__(
        self,
        elmo_model: "Embedder",  # ELMo模型，用于获取词向量表示
        vocab_size: int,  # 词表大小，用于构建目标端词嵌入层
        hidden_size: int = 512,  # 隐藏层维度，默认512
        num_layers: int = 1,  # GRU层数，默认1
        dropout: float = 0.5,  # Dropout概率，默认0.5
        bidirectional: bool = True,  # 是否使用双向GRU，默认True
        elmo_dim: int = 1024,  # ELMo projection_dim
    ):
        super().__init__()
        self.elmo = elmo_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.elmo_dim = elmo_dim

        # Encoder
        self.encoder_gru = nn.GRU(
            input_size=elmo_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder
        self.decoder_embedding = nn.Embedding(vocab_size, hidden_size)
        self.decoder_gru = nn.GRU(
            input_size=hidden_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            kdim=hidden_size * 2 if bidirectional else hidden_size,
            vdim=hidden_size * 2 if bidirectional else hidden_size,
            num_heads=8,
            dropout=dropout,
        )
        self.decoder_fc = nn.Linear(
            hidden_size * 2 if bidirectional else hidden_size, vocab_size
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

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
        with torch.no_grad():
            elmo_outputs = self.elmo.sents2elmo(source_inputs)
            elmo_outputs = (
                torch.from_numpy(np.array(elmo_outputs)).float().to(source_mask.device)
            )

        # Pack padded sequences
        lengths = source_mask.sum(dim=1).cpu()
        packed_inputs = pack_padded_sequence(
            elmo_outputs, lengths, batch_first=True, enforce_sorted=False
        )

        # Encoder GRU
        encoder_outputs, _ = self.encoder_gru(packed_inputs)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs, batch_first=True)

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
        torch.Tensor,  # decoder_output: a torch.Tensor of (batch_size, sequence_length, vocab_size)
        torch.Tensor,
    ]:  # hidden states
        """
        decode for output sequence
        """
        batch_size, seq_length = target_input_ids.size()

        # Embedding
        decoder_inputs = self.decoder_embedding(target_input_ids)
        decoder_inputs = self.dropout(decoder_inputs)

        # Decoder GRU
        if hidden_states is None:
            hidden_states = self._init_hidden(batch_size, encoder_outputs.device)

        # Ensure hidden_states has the correct shape for the decoder GRU
        if self.bidirectional:
            hidden_states = hidden_states.view(
                self.num_layers, batch_size, self.hidden_size * 2
            )
        else:
            hidden_states = hidden_states.view(
                self.num_layers, batch_size, self.hidden_size
            )

        decoder_outputs, new_hidden_states = self.decoder_gru(
            decoder_inputs, hidden_states
        )

        # Attention
        encoder_outputs = encoder_outputs.permute(
            1, 0, 2
        )  # (seq_length, batch_size, hidden_size)
        decoder_outputs = decoder_outputs.permute(
            1, 0, 2
        )  # (seq_length, batch_size, hidden_size)
        attention_output, _ = self.attention(
            query=decoder_outputs,
            key=encoder_outputs,
            value=encoder_outputs,
            key_padding_mask=~source_mask.bool(),
        )
        attention_output = attention_output.permute(
            1, 0, 2
        )  # (batch_size, seq_length, hidden_size)

        # Concatenate decoder outputs and attention outputs
        combined_outputs = torch.cat(
            [decoder_outputs.permute(1, 0, 2), attention_output], dim=-1
        )

        # Final projection
        decoder_output = self.decoder_fc(combined_outputs)
        decoder_output = self.dropout(decoder_output)

        return decoder_output, new_hidden_states

    def forward(self, **kwargs):
        encoder_outputs = self.encode(**kwargs)
        decoder_outputs, _ = self.decode(encoder_outputs, **kwargs)
        return decoder_outputs

    def _init_hidden(self, batch_size, device):
        num_directions = 2 if self.bidirectional else 1
        return torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )
