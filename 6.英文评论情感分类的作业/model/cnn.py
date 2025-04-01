import numpy as np
import torch
from torch import nn
from allennlp.modules.elmo import Elmo
from typing import List


class TextCNN(nn.Module):
    def __init__(
        self,
        options_file:str, # elmo file
        weight_file:str, # elmo weight file
        vector_size:int, # word embedding dim
        filter_size:List[int]=[2, 3, 4, 5], # kernel size for each layer of CNN
        channels:int=64, # output channel for CNN
        max_length:int =1024, # max length of input sentence
        dropout = 0.5, # dropout rate
    ):
        super(TextCNN, self).__init__()
        self.embedding = Elmo(options_file, weight_file, 1, dropout=0)
        ####################
        # 初始化嵌入层已经通过Elmo完成
        # 直接用上次作业的代码
        # Build a stack of 1D CNN layers for each filter size
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channels=vector_size, 
                          out_channels=channels, 
                          kernel_size=k)
                # Conv1dViaConv2d(
                #     in_channels=vector_size,
                #     out_channels=channels,
                #     kernel_size=k,
                #     conv_2d=KAN_Convolutional_Layer,
                # )
                for k in filter_size
            ]
        )
        # Final linear layer for label prediction; number of classes equals len(label2index)
        # CNN的输出通道数 × 不同卷积核的数量
        self.linear = nn.Linear(channels * len(filter_size), 2) # 二分类问题（正面/负面）
        #  Dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length


    def forward(self, 
                inputs:torch.Tensor, # input sentence, size N*L
                ) -> torch.Tensor: # predicted_logits: torch.tensor of size N*C (number of classes)
        # 获取ELMo嵌入表示
        inputs = self.embedding(inputs)["elmo_representations"][0]  # [N, L, vector_size]
        
        # Convolutional layer
        x = inputs.transpose(1, 2)  # 卷积需要将词向量维度放在最后 (N*D*L)
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
        # 应用dropout
        x = self.dropout(x)
        # Linear layer
        x = self.linear(x)  # 分类，得到 (N*K)
        return x
