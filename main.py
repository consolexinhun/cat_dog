import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader

# import os
# os.environ['']
import csv
import numpy as np
import pandas as pd
from PIL import Image

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from dataload import split

TrainLoad = split('train', batch_size=64, mode='train')
ValLoad = split('val', batch_size=64, mode='val')
TestLoad = split('test', batch_size=1, mode='test')

resnet_model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(resnet_model.children())[:-1],
                      Flatten(),
                      nn.Linear(512, 2)).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criteon = nn.CrossEntropyLoss().to(device)


def evaluate(loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total


############ train

epochs = 40
for epoch in range(epochs):
    for step, (x, y) in enumerate(TrainLoad):
        model.train()
        x, y = x.to(device), y.to(device)
        logits = model(x)

        loss = criteon(logits, y)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print('step:{}, loss:{}'.format(step, loss.item()))

    print('epoch:{}, loss:{}'.format(epoch, loss.item()))


    acc = evaluate(ValLoad)
    print("evaluate acc : {}".format(acc))

###############

keys, values = [], []
for i in range(len(TestLoad.dataset)):
    keys.append(i)

model.eval()
for x in TestLoad:
    x = x.to(device)
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(dim=1)

    values.append(pred[0].cpu().numpy())

with open('key.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(zip(keys, values))



