import torch
from torch import nn
from allennlp.modules.elmo import Elmo
from typing import List


class TextLSTM(nn.Module):
    def __init__(
        self,
        options_file: str,  # elmo options file
        weight_file: str,   # elmo weight file
        vector_size: int,   # word embedding dim
        filter_size: List[int] = [2, 3, 4, 5],  # 保留接口，与CNN一致，但不使用
        channels: int = 64,   # 作为LSTM隐藏层维度
        max_length: int = 1024,  # 最大句子长度
        dropout: float = 0.5,  # dropout rate
    ):
        super(TextLSTM, self).__init__()
        self.embedding = Elmo(options_file, weight_file, 1, dropout=0)
        # 使用LSTM进行特征抽取，使用channels作为隐藏层维度
        self.lstm = nn.LSTM(input_size=vector_size, hidden_size=channels, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # 最后全连接分类层
        self.linear = nn.Linear(channels, 2)  # 二分类问题

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        利用ELMo嵌入和LSTM进行前向传播
        """
        # 获取ELMo嵌入表示, 输出形状为 (N, L, vector_size)
        x = self.embedding(inputs)["elmo_representations"][0]
        # 通过LSTM，输出h_n形状为 (num_layers, N, hidden_size)
        _, (h_n, _) = self.lstm(x)
        # 提取最后一层隐藏状态，形状为 (N, hidden_size)
        h = h_n[-1]
        # 应用Dropout
        h = self.dropout(h)
        # 分类
        out = self.linear(h)
        return out