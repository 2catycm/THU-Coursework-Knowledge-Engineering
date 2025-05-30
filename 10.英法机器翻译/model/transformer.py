import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, repeat


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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,  # [batch_size, num_heads, seq_len, hidden_size / num_heads]
        mask: torch.Tensor  # [batch_size, 1, seq_len, seq_len]
    ):
        """
        output:
            - context: 输出值
            - attention: 计算得到的注意力矩阵
        """
        d_k = q.size(-1)
        sqrt_d_k = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        # 计算点积，使用einops的einsum
        
        # b,h 独立，做乘法的是 i,j，
        # d 维度会进行求和操作，因为它只在输入中出现，不在输出中出现（被reduce掉了）。
        scores = einsum(q, k, 'b h i d, b h j d -> b h i j') / sqrt_d_k 

        if mask is not None:
            # print("scores.shape", scores.shape)
            # print("mask.shape", mask.shape)
            scores = scores.masked_fill(mask == 0, -1e9) # mask到的位置不能被 attention 注意到，本来是赋值为0，但是待会有softmax，应该给-inf。

        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # 计算context，使用einops的einsum
        # b,h 独立，i， d乘法；对 j 求和
        context = einsum(attention, v, 'b h i j, b h j d -> b h i d')
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
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,  # [batch_size, num_heads, seq_len, hidden_size / num_heads]
        mask: torch.Tensor  # [batch_size, 1, seq_len, seq_len]
    ):
        mask = mask.unsqueeze(1)
        residual = q
        batch_size = q.size(0)

        # 线性变换
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # 分割成多个头
        head_dim = self.hidden_size // self.num_heads
        q = q.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        # 应用缩放点积注意力
        context, attention = self.scaled_dot_product_attention(q, k, v, mask)

        # 拼接多个头的输出
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)

        # 通过线性层
        output = self.linear(context)
        output = self.dropout_layer(output)

        # 残差连接和层归一化
        output = self.layer_norm(output + residual)

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


import torch
from functools import partial

def make_attention_mask(pad_mask: torch.Tensor, is_decoder: bool) -> torch.Tensor:
    """
    生成 Encoder 或 Decoder 的 [batch, 1, seq, seq] 自注意力 mask。

    参数：
      pad_mask: [batch, seq_len]，1 表示有效 token，0 表示 padding。
      is_decoder: True 则生成下三角 causal mask，否则全 1。

    返回：
      mask: [batch, 1, seq_len, seq_len]，bool dtype。
    """
    batch, seq_len = pad_mask.shape
    device = pad_mask.device

    # —— 第一步：底板 mask（bool）
    if is_decoder:
        # 下三角 causal
        base = torch.tril(torch.ones((seq_len, seq_len), 
                                     dtype=torch.bool, 
                                     device=device))
    else:
        # 全 1
        base = torch.ones((seq_len, seq_len), 
                          dtype=torch.bool, 
                          device=device)

    # 扩展到 batch 维度
    base = base.unsqueeze(0).expand(batch, seq_len, seq_len)  # [b, seq, seq]

    # —— 第二步：根据 pad_mask 屏蔽行 & 列
    # row_ok[b,i,j] = pad_mask[b,i]
    row_ok = pad_mask.bool().unsqueeze(2).expand(batch, seq_len, seq_len)
    # col_ok[b,i,j] = pad_mask[b,j]
    col_ok = pad_mask.bool().unsqueeze(1).expand(batch, seq_len, seq_len)

    mask2d = base & row_ok & col_ok  # [b, seq, seq]
    return mask2d.unsqueeze(1)       # -> [b, 1, seq, seq]

make_decoder_mask = partial(make_attention_mask, is_decoder=True)
make_encoder_mask = partial(make_attention_mask, is_decoder=False)

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.source_vocab_size, config.hidden_size)
        self.position_embedding = PositionalEncoding(config.hidden_size, config.dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.num_layers)]
        )

    def forward(self, source_ids, 
            source_mask:torch.Tensor # [batch_size, seq_len]
            ):
        # print(f"Encoder source_mask.shape is {source_mask.shape}")
        # print(f"Encoder source_mask is {source_mask}")
        
        x = self.embedding(source_ids)
        x = self.position_embedding(x)
        # 将 source_mask 从 [batch_size, seq_len] 转换为 [batch_size, 1, seq_len, seq_len]
        # source_mask = make_encoder_mask(source_mask)
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
        # print(f"Decoder source_mask.shape is {source_mask.shape}")
        # print(f"Decoder target_mask.shape is {target_mask.shape}")
        # print(f"Decoder source_mask is {source_mask}")
        # print(f"Decoder target_mask is {target_mask}")
        # print(f"target_ids.device = {target_ids.device}")
        x = self.embedding(target_ids)
        x = self.position_embedding(x)

        # source_mask = make_decoder_mask(source_mask)
        # target_mask = make_decoder_mask(target_mask)
        source_mask = source_mask.unsqueeze(1)
        target_mask = target_mask.unsqueeze(1)
        for layer in self.layers:
            x, attention_1, attention_2 = layer(
                x, encoder_output, source_mask, target_mask
            )
        return x
