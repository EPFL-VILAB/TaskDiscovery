import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import argparse
import tqdm

from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize

import sys 

sys.path.append('../task-discovery')
from models.resnet import ResNet18
from itertools import combinations

def split_dataset(dataset, seed=0, val_split=0.1):
    len_dataset = len(dataset)
    splits = get_splits(len_dataset, val_split)
    dataset_train, dataset_val = random_split(dataset, splits, generator=torch.Generator().manual_seed(seed))

    return dataset_train, dataset_val

def adv_split_dataset(dataset, split_ids):
    dataset_train = Subset(dataset, split_ids[0])
    dataset_val = Subset(dataset, split_ids[1])
    return dataset_train, dataset_val

def get_splits(len_dataset, val_split=0.1):
    val_len = int(val_split * len_dataset)
    train_len = len_dataset - val_len
    splits = [train_len, val_len]
    return splits

def cifar10_normalization():
    normalize = Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )
    return normalize

def image_transform():
    return Compose([ToTensor(), cifar10_normalization()])

REAL_TASKS = list(combinations(range(10), 5))[:126]

parser = argparse.ArgumentParser()
parser.add_argument('--adversarial_splits_path', type=str, default='assets/adversarial_splits/cifar10/cifar_splits.pt')
parser.add_argument('--split_idx', type=int, default=0)
parser.add_argument('--task_idx', type=int, default=0)
parser.add_argument('--task_type', type=str, default='real2')
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

dataset = CIFAR10(root='.', train=True, transform=image_transform(), download=True)

if args.task_type == 'real2':
    real_task = REAL_TASKS[args.task_idx]
    new_target = list(map(lambda x: int(x in real_task), dataset.targets))
    dataset.targets = new_target

print("LOADED TASK: ", args.task_type)

if args.adversarial_splits_path != '':
    print(f"USING ADVERSARIAL SPLIT, task_idx = {args.task_idx}, split_idx = {args.split_idx}")
    split_ids = torch.load(args.adversarial_splits_path)[args.task_idx][args.split_idx]
    dataset_train, dataset_val = adv_split_dataset(dataset, split_ids)
else:
    print(f"USING RANDOM SPLIT")
    dataset_train, dataset_val = split_dataset(dataset, val_split=0.5)

dataloader_train = DataLoader(dataset_train, batch_size=512, num_workers=32, shuffle=True, pin_memory=True, persistent_workers=True)
dataloader_val = DataLoader(dataset_val, batch_size=512, num_workers=32, pin_memory=True, persistent_workers=True)

if args.task_type == 'real2':
    K = 2
elif args.task_type == 'real10':
    K = 10
else:
    raise NotImplementedError

model1 = ResNet18(out_dim=K).to(args.device)

optimizer = torch.optim.Adam(model1.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in tqdm.tqdm(range(100)):
    for x, y in dataloader_train:
        optimizer.zero_grad()
        x, y = x.to(args.device), y.to(args.device)
        pred1 = model1(x)
        loss = criterion(pred1, y)
        loss.backward()
        optimizer.step()

preds1 = []

with torch.no_grad():
    model1.eval()
    for x, y in dataloader_val:
        x = x.to(args.device)
        y = y.to(args.device)
        pred1 = model1(x).argmax(1)
        preds1.append((pred1 == y).float())

preds4acc = torch.cat(preds1)
print("Accuracy=", preds4acc.mean().item())