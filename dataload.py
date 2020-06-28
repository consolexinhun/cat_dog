import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader

import os
import csv
import numpy as np
import pandas as pd
from PIL import Image
import time

train_transform = transforms.Compose([
    lambda x:Image.open(x).convert('RGB'),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                          [0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    lambda x:Image.open(x).convert('RGB'),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                          [0.229,0.224,0.225])
])


class MyDataSet(Dataset):
    def __init__(self, paths, mode, labels=None):
        super(MyDataSet, self).__init__()
        self.paths  = paths
        self.mode = mode
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = self.paths[item]
        if self.mode == 'train' or self.mode == 'val':
            image = train_transform(img_path)
            label = self.labels[item]

            return image, torch.tensor(label)

        if self.mode == 'test':
            image = test_transform(img_path)
            return image

def split(dir, batch_size, mode):
    img_paths, labels = [], []
    if mode == 'train' or mode == 'val':
        train_path = os.listdir(os.path.join(dir))
        train_path.sort(key=lambda x : int(x[4:-4]))

        for i, filename in enumerate(train_path):
            img_paths.append(os.path.join(dir, filename))
            labels.append((0 if filename[:3] == "cat" else 1))

        print(len(labels)) # train ：20000  val:2000
        train_loader = DataLoader(MyDataSet(img_paths, mode, labels), batch_size, shuffle=True)
        return train_loader
    elif mode == 'test':
        test_path = os.listdir(os.path.join(dir))
        test_path.sort(key=lambda x: int(x[: -4]))

        for filename in test_path:
            img_paths.append(os.path.join(dir, filename))

        # print(len(img_paths)) # 2000

        test_loader = DataLoader(MyDataSet(img_paths, mode), batch_size)
        return test_loader


if __name__ == '__main__':
    # TestLoader = split('test', 8, 'test')
    TrainLoader = split('val', 32, 'val')

    print(len(TrainLoader.dataset))

    sample = next(iter(TrainLoader))
    print(sample[1])


    # sample = next(iter(TestLoader))
    # print(len(sample))