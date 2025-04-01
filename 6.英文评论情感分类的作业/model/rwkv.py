import torch
from torch import nn
from allennlp.modules.elmo import Elmo
from typing import List

# 引入 RWKV7Attention 模块
from fla.layers.rwkv7 import RWKV7Attention

class TextRWKV(nn.Module):
    def __init__(
        self,
        options_file: str,   # elmo选项文件
        weight_file: str,    # elmo权重文件
        vector_size: int,    # 词向量维度
        filter_size: List[int] = [2, 3, 4, 5],  # 保持接口一致，但不使用
        channels: int = 64,  # 保持接口一致，可用于其他用途
        max_length: int = 1024,  # 最大句子长度
        dropout: float = 0.5,    # dropout 概率
    ):
        super(TextRWKV, self).__init__()
        # 使用ELMo构造嵌入层
        self.embedding = Elmo(options_file, weight_file, 1, dropout=0)
        # 构造 RWKV 模块，采用 chunk 模式，hidden_size 使用 vector_size
        self.rwkv = RWKV7Attention(mode='chunk', hidden_size=vector_size, head_dim=64)
        self.dropout = nn.Dropout(dropout)
        # 最后全连接分类层，将特征映射到二分类问题
        self.linear = nn.Linear(vector_size, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        利用ELMo嵌入和RWKV7Attention模块进行前向传播

        Parameters:
            inputs: torch.Tensor
                输入句子（形状为 N x L）
        
        Returns:
            torch.Tensor: 分类预测 logits（形状为 N x 2）
        """
        # 获取ELMo嵌入表示，形状为 (N, L, vector_size)
        x = self.embedding(inputs)["elmo_representations"][0]
        # 通过RWKV模块，输出形状假设为 (N, L, vector_size)
        o, _, _, _ = self.rwkv(x)
        # 对时间步进行最大池化，得到 (N, vector_size) 表示
        h, _ = torch.max(o, dim=1)
        # 应用ReLU激活增强非线性，然后dropout
        h = torch.relu(h)
        h = self.dropout(h)
        # 全连接分类层，输出二分类 logits
        out = self.linear(h)
        return out
