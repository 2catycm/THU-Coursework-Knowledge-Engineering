import re
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional


class MyDataset(Dataset):
    def __init__(
        self,
        file: str,  # 文件路径
        text_vocab: dict,  # 文本词汇
        max_length: int = 1024,  # 最大长度
        pad_token: str = "<PAD>",  # 填充标记
        unk_token: str = "<UNK>",  # 未知标记
        label2index: Optional[dict] = None,  # 标签映射
    ) -> None:
        # 先写self是更加规范的。
        self.text_vocab = text_vocab
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.max_length = max_length
        # 直接保存的参数写完了，接下来才写计算逻辑

        # 加载原始文本和标签
        # 这里还没变成张量，不要搞混淆了
        raw_text, raw_labels = self.load(file)
        assert len(raw_text) == len(raw_labels), "text: {}, label: {}".format(
            len(raw_text), len(raw_labels)
        )
        # assert condition, error_message 才是规范写法，助教写print有误。

        # 初始化或使用标签映射
        if label2index is None:
            self.label2index = dict(
                zip(sorted(set(raw_labels)), range(len(set(raw_labels))))
            )
        else:
            self.label2index = label2index

        # 转换标签为整数
        # convert_label2index 函数不应该暴露到外面，而且只有一行，直接在这里实现
        self._labels = [self.label2index[label] for label in raw_labels]
        assert len(self._labels) == len(raw_labels), "_labels: {}, raw_labels: {}".format(
            len(self._labels), len(raw_labels)
        )

        # 转换文本为词索引
        indexed_text = self.word2index(raw_text)
        assert len(indexed_text) == len(raw_text), "indexed_text: {}, raw_text: {}".format(
            len(indexed_text), len(raw_text)
        )

        # 填充并转换为张量
        # 合理的接口设计不应该使用 self传递参数，而是应该明确传递。
        padded_text = self.pad(indexed_text)
        self._text_tensor = torch.tensor(padded_text)
    
    def convert_label2index(self) -> None:  # 将字符串标签转换为整数索引
        """已在__init__中实现,此方法保留以兼容旧代码"""
        pass

    def word2index(self,
                   text: list[str]  # 输入文本列表
                   ) -> list[list[int]]:  # 返回词索引列表的列表
        """
        convert loaded text to word_index with text_vocab
        self.text_vocab is a dict
        """
        _text = []
        #############################
        # TODO
        ###########################
        return _text

    def load(self,
             file: str  # 输入文件路径
             ) -> tuple[list[str], list[str]]:  # 返回(文本列表,标签列表)
        """
        read file and load into text (a list of strings) and label (a list of class labels)
        """
        text, label = [], []
        #####################
        # TODO
        #####################
        return text, label

    def pad(self, text: list[list[int]]  # 待填充的词索引列表
            ) -> list[list[int]]:  # 返回填充后的词索引列表
        """
        pad word indices to max_length
        """
        pad_text = []
        for _text in text:
            ################
            # TODO
            # hint: use pad_token index to pad
            pass
            ################
        return pad_text

    def __len__(self) -> int:  # 返回数据集大小
        return len(self._text_tensor)

    def __getitem__(self, item: int  # 数据索引
                     ) -> tuple[torch.Tensor, int]:  # 返回(文本张量,标签)
        return self._text_tensor[item], self._labels[item]
