import re
import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(
        self,
        file,
        text_vocab,
        max_length=1024,
        pad_token="<PAD>",
        unk_token="<UNK>",
        label2index=None,
    ):
        self.text, self.label = self.load(file)

        assert len(self.text) == len(self.label), print(
            "text: {}, label: {}".format(len(self.text), len(self.label))
        )

        if label2index is None:
            self.label2index = dict(
                zip(sorted(set(self.label)), range(len(set(self.label))))
            )
        else:
            self.label2index = label2index
        self.convert_label2index()
        assert len(self.text) == len(self.label), print(
            "text: {}, label: {}".format(len(self.text), len(self.label))
        )

        self.text_vocab = text_vocab
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.max_length = max_length
        # convert strings to indices with pre-trained word2vec model
        self.text = self.word2index(self.text)
        assert len(self.text) == len(self.label), print(
            "text: {}, label: {}".format(len(self.text), len(self.label))
        )

        self.pad()
        assert len(self.text) == len(self.label), print(
            "text: {}, label: {}".format(len(self.text), len(self.label))
        )

    def convert_label2index(self):
        self.label = [self.label2index[_label] for _label in self.label]

    def word2index(self, text):
        """
        convert loaded text to word_index with text_vocab
        self.text_vocab is a dict
        Args:
           text: List[str]
        Return:
          _text: List[List[int]]
        """
        _text = []
        #############################
        # TODO
        ###########################
        return _text

    def load(self, file):
        """
        read file and load into text (a list of strings) and label (a list of class labels)
        Args:
          file: str
        Returns
          text: List[str]
          label: list[str]
        """
        text, label = [], []
        #####################
        # TODO
        #####################
        return text, label

    def pad(self):
        """
        pad word indices to max_length and convert it to torch.Tensor
        """
        pad_text = []
        for _text in self.text:
            ################
            # TODO
            # hint: use pad_token index to pad
            pass
            ################
        self.text = torch.tensor(pad_text)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        return self.text[item], self.label[item]
