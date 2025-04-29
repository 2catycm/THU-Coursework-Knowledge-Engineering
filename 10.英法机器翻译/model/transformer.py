import json
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerConfig:
    def __init__(self, config: dict):
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_layers = config["num_hidden_layers"]
        self.filter_size = config["intermediate_size"]
        self.dropout = config["dropout_prob"]
        self.max_len = config["max_position_embeddings"]
        with open(config["source_vocab_path"], "r", encoding="utf-8") as f:
            self.source_vocab_size = len(json.load(f))
        with open(config["target_vocab_path"], "r", encoding="utf-8") as f:
            self.target_vocab_size = len(json.load(f))


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.linear = nn.Linear(config.hidden_size, config.target_vocab_size)

    def forward(self, source_ids, source_mask, target_ids, target_mask):
        encoder_output = self.encoder(source_ids, source_mask)
        decoder_output = self.decoder(
            target_ids, encoder_output, source_mask, target_mask
        )
        output = self.linear(decoder_output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout, max_len=512):
        """实现位置编码"""
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """输入x的形状为[seq_len, batch_size, hidden_size]"""
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        """实现Scaled Dot-Product Attention"""
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask):
        """
        q, k, v的形状为[batch_size, num_heads, seq_len, hidden_size / num_heads]
        mask的形状为[batch_size, 1, seq_len, seq_len]
        output:
            - context: 输出值
            - attention: 计算得到的注意力矩阵
        """
        context, attention = None, None
        # TODO
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        """实现Multi-Head Attention"""
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, q, k, v, mask):
        """
        q, k, v的形状为[batch_size, seq_len, hidden_size]
        mask的形状为[batch_size, 1, seq_len, seq_len]
        """
        residual = q
        batch_size = q.size(0)
        output, attention = None, None
        # TODO
        return output, attention


class FeedForward(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout):
        """实现FFN"""
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, filter_size)
        self.linear_2 = nn.Linear(filter_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """输入x的形状为[batch_size, seq_len, hidden_size]"""
        residual = x
        output = self.linear_2(F.relu(self.linear_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(
            config.hidden_size, config.num_heads, config.dropout
        )
        self.feed_forward = FeedForward(
            config.hidden_size, config.filter_size, config.dropout
        )

    def forward(self, x, mask):
        """输入x的形状为[batch_size, seq_len, hidden_size]"""
        x, attention = self.attention(x, x, x, mask)
        x = self.feed_forward(x)
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.source_vocab_size, config.hidden_size)
        self.position_embedding = PositionalEncoding(config.hidden_size, config.dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_layers)]
        )

    def forward(self, source_ids, source_mask):
        x = self.embedding(source_ids)
        x = self.position_embedding(x)
        source_mask = source_mask.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, source_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.attention_1 = MultiHeadAttention(
            config.hidden_size, config.num_heads, config.dropout
        )
        self.attention_2 = MultiHeadAttention(
            config.hidden_size, config.num_heads, config.dropout
        )
        self.feed_forward = FeedForward(
            config.hidden_size, config.filter_size, config.dropout
        )

    def forward(self, x, encoder_output, source_mask, target_mask):
        x, attention_1 = self.attention_1(x, x, x, target_mask)
        x, attention_2 = self.attention_2(
            x, encoder_output, encoder_output, source_mask
        )
        x = self.feed_forward(x)
        return x, attention_1, attention_2


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(config.target_vocab_size, config.hidden_size)
        self.position_embedding = PositionalEncoding(config.hidden_size, config.dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_layers)]
        )

    def forward(self, target_ids, encoder_output, source_mask, target_mask):
        x = self.embedding(target_ids)
        x = self.position_embedding(x)
        source_mask = source_mask.unsqueeze(1)
        target_mask = target_mask.unsqueeze(1)
        for layer in self.layers:
            x, attention_1, attention_2 = layer(
                x, encoder_output, source_mask, target_mask
            )
        return x
