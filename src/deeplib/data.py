import math
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler


def train_valid_loaders(dataset, batch_size, train_split=0.8, shuffle=True):
    num_data = len(dataset)
    indices = np.arange(num_data)

    if shuffle:
        np.random.shuffle(indices)

    split = math.floor(train_split * num_data)
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader
