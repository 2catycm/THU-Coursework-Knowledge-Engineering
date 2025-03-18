import numpy as np
import torch
from torch import nn

import torch
import torch.nn as nn

from kan_convolutional.KANConv import KAN_Convolutional_Layer


class Conv1dViaConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        conv_2d=nn.Conv2d,
    ):
        super(Conv1dViaConv2d, self).__init__()
        self.conv2d = conv_2d(
            in_channels,
            out_channels,
            (1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            dilation=(1, dilation),
            # groups=groups,
            # bias=bias,
        )

    def forward(self, x):
        # 调整输入维度
        x = x.unsqueeze(2)  # 添加一个高度维度
        # 执行 Conv2d
        x = self.conv2d(x)
        # 移除多余维度
        x = x.squeeze(2)
        return x


class TextCNN(nn.Module):
    def __init__(
        self,
        word_embeddings: np.ndarray,  # 预训练词向量矩阵(N*D)
        vector_size: int,  # 词向量维度 D
        label2index: dict,  # 标签到索引的映射
        pad_index: int,  # 填充token的索引
        filter_size: list[int] = [2, 3, 4, 5],  # CNN卷积核大小
        channels: int = 64,  # CNN输出通道数
        max_length: int = 1024,  # 最大序列长度
    ) -> None:
        super(TextCNN, self).__init__()
        # Initialize embedding layer with pre-trained word_embeddings
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(word_embeddings), freeze=False, padding_idx=pad_index
        )
        # Build a stack of 1D CNN layers for each filter size
        self.convs = nn.ModuleList(
            [
                # nn.Conv1d(in_channels=vector_size, out_channels=channels, kernel_size=k)
                Conv1dViaConv2d(
                    in_channels=vector_size,
                    out_channels=channels,
                    kernel_size=k,
                    conv_2d=KAN_Convolutional_Layer,
                )
                for k in filter_size
            ]
        )
        # Final linear layer for label prediction; number of classes equals len(label2index)
        num_class = len(label2index)
        self.linear = nn.Linear(channels * len(filter_size), num_class)
        self.max_length = max_length

    def forward(
        self,
        inputs: torch.Tensor,  # 输入张量(N*L)
    ) -> torch.Tensor:  # 返回预测logits(N*K)， 不需要softmax
        # check max_length
        if inputs.size(1) > self.max_length:
            inputs = inputs[:, : self.max_length]
        # Embedding layer
        x = self.embedding(inputs)  # 得到 (N*L*D)
        # Convolutional layer
        x = x.transpose(1, 2)  # 卷积需要将词向量维度放在最后 (N*D*L)
        x = [conv(x) for conv in self.convs]
        x = [nn.functional.gelu(i) for i in x]  # 每一个 i是 (N*C*Li) ， Li = L - ki + 1
        # Pooling layer
        x = [
            nn.functional.max_pool1d(
                i,
                kernel_size=i.size(2),  # 对 Li 去做 max_pooling
            ).squeeze(2)
            for i in x  # 每一个 i是 (N*C*Li)
        ]  # 每一个 item 变为 (N*C)
        # Concatenate all pooling results
        x = torch.cat(x, dim=1)  # 把每一个 item 拼接起来，变为 (N, C*len(filter_size))
        # Linear layer
        x = self.linear(x)  # 分类，得到 (N*K)
        return x
