from typing import List, Tuple
from torch.utils.data import DataLoader, Dataset
from allennlp.modules.elmo import Elmo, batch_to_ids
import torch.nn.functional as F
import torch
import random


class MyDataset(Dataset):
    def __init__(
        self, filename, max_length=64, 
        train=True, max_example_num=None,
        random_state=0
    ):
        self.max_length = max_length
        self.text, self.label = self.load(filename, train=train)
        text_ids = batch_to_ids(self.text)
        self.text = self.pad(text_ids).tolist()
        if max_example_num:
            random.seed(0)
            sampled_index = random.sample(
                range(len(self.text)), min(max_example_num, len(self.text))
            )
            self.text = [self.text[i] for i in sampled_index]
            self.label = [self.label[i] for i in sampled_index]

    def load(
        self,
        file: str,  # file path
        train: bool = True,  # whether is training file
    ) -> Tuple[List[List[str]], List[int]]:  # Returns (text, label), text input and label
        """
        load file into texts and labels
        """
        import pandas as pd
        
        # 使用pandas读取文件，自动推断分隔符
        try:
            # 首先尝试tab分隔符，因为这是期望的格式
            df = pd.read_csv(file, sep='\t')
        except Exception as e:
            print(f"Error reading file with tab separator: {e}")
            print("Trying to read with auto-detected separator...")
            # 如果失败，让pandas尝试自动推断分隔符
            df = pd.read_csv(file, sep=None, engine='python')

        text = df['sentence'].astype(str).tolist()
        # 分词
        text = [sentence.split() for sentence in text]
        
        if train:
            # 训练集格式: sentence  label
            label = df['label'].astype(int).tolist()
        else:
            # 测试集可能没有标签，默认为 -1 表示不知道
            label = [-1] * len(text)
            
        return text, label

    def pad(self, 
            text_ids: torch.Tensor,  # size N*L*D
            )-> torch.Tensor: # Returns padded_text_id, size N*max_length*D
        """
        pad text_ids to max_length
        """
        N, L, D = text_ids.shape
        print("text_ids shape: ", text_ids.shape)
        if L >= self.max_length:
            # 如果文本长度大于等于最大长度，截断
            return text_ids[:, :self.max_length, :]
        else:
            # 如果文本长度小于最大长度，填充
            padding = torch.zeros(N, self.max_length - L, D, dtype=text_ids.dtype, device=text_ids.device)
            padded_text_ids = torch.cat([text_ids, padding], dim=1)
            return padded_text_ids

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        return self.text[item], self.label[item]


def collate_fn(data, device):
    text, label = [], []
    for _text, _label in data:
        label.append(_label)
        text.append(_text)
    return torch.tensor(text).to(device), torch.tensor(label).to(device)


if __name__ == "__main__":
    dataset = MyDataset("./data/dev.tsv")
    print(dataset[0])
