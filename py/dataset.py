from PIL import Image
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np


def split_data(path, class_list, p=0.9):
    train_data = []
    test_data = []
    for i, class_str in enumerate(class_list):
        tmp_list = [((path + class_str + '/' + image), i) for image in os.listdir(path + class_str)]
        idx = int(len(tmp_list) * p)
        train_data += tmp_list[:idx]
        test_data += tmp_list[idx:]

    return train_data, test_data


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, transforms):
        super().__init__()

        self.data = data

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx]
        # label = torch.Tensor(label)
        data = Image.open(data).convert('RGB')

        if self.transforms is not None:
            data = self.transforms(data)

        return data, label