import numpy as np
import torch
import functools

class CORE_Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, tokenizer, use_ratio=1, which_label='void_fraction'):
        label_dict = {
            'void_fraction': 2,
            'pld': 3,
            'lcd': 4
        }
        self.data = data[:int(len(data) * use_ratio)]
        self.mofid = self.data[:, 1].astype(str)
        self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True, padding='max_length') for i in self.mofid])
        self.label = self.data[:, label_dict[which_label]].astype(float)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.label)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        X = torch.from_numpy(np.asarray(self.tokens[index]))
        y = torch.from_numpy(np.asarray(self.label[index])).view(-1, 1)
        return X, y.float()


class MOF_ID_Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, tokenizer, use_ratio=1):
        self.data = data[:int(len(data) * use_ratio)]
        self.mofid = self.data[:, 1].astype(str)
        self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True, padding='max_length') for i in self.mofid])
        self.label = self.data[:, 2].astype(float)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.label)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        X = torch.from_numpy(np.asarray(self.tokens[index]))
        y = torch.from_numpy(np.asarray(self.label[index])).view(-1, 1)
        return X, y.float()


class MOF_pretrain_Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, tokenizer, use_ratio=1):
        self.data = data[:int(len(data) * use_ratio)]
        self.mofid = self.data.astype(str)
        self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True, padding='max_length') for i in self.mofid])
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.mofid)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        X = torch.from_numpy(np.asarray(self.tokens[index]))
        return X.type(torch.LongTensor)


class MOF_tsne_Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, tokenizer, use_ratio=1):
        self.data = data[:int(len(data) * use_ratio)]
        self.mofname = self.data[:, 0].astype(str)
        self.mofid = self.data[:, 1].astype(str)
        self.tokens = np.array([tokenizer.encode(i, max_length=512, truncation=True, padding='max_length') for i in self.mofid])
        self.label = self.data[:, 2].astype(float)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.label)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        X = torch.from_numpy(np.asarray(self.tokens[index]))
        return X, self.label[index], self.mofname[index], self.mofid[index]


# Define a CustomDataset class to satisfy finetune_transformer.py imports
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, tokenizer=None, transform=None):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.tokenizer:
            sample = self.tokenizer.encode(sample, max_length=512, truncation=True, padding='max_length')
            sample = torch.tensor(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample, torch.tensor(label).float()
