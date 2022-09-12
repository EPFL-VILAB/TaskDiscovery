import torch
import pandas as pd
import torch.nn as nn
import argparse
import tqdm

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from models.resnet import ResNet18
from itertools import combinations


def split_dataset(dataset, seed=0):
    len_dataset = len(dataset)
    splits = get_splits(len_dataset)
    dataset_train, dataset_val = random_split(dataset, splits, generator=torch.Generator().manual_seed(seed))

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
parser.add_argument('--tasks_path', type=str, default='assets/tasks/discovered.csv')
parser.add_argument('--task_type', type=str, default='resnet18')
parser.add_argument('--task_idx', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

dataset = CIFAR10(root='.', train=True, transform=image_transform(), download=True)

if args.task_type == 'random':
    g = torch.Generator().manual_seed(0)
    new_target = torch.randint(0, 2, (len(dataset),), generator=g).tolist()

elif args.task_type == 'real':
    real_task = REAL_TASKS[args.task_idx]
    new_target = list(map(lambda x: int(x in real_task), dataset.targets))

elif args.task_type in ('resnet18', 'vit', 'mlp', 'resnet18_d8'):
    tasks = pd.read_csv(args.tasks_path, index_col=False)
    new_target = tasks[args.task_type+'-'+str(args.task_idx)].tolist()
else:
    raise NotImplementedError()

dataset.targets = new_target

print(f"LOADED TASK: {args.task_type}-{args.task_idx}")
dataset_train, dataset_val = split_dataset(dataset)

dataloader_train = DataLoader(dataset_train, batch_size=512, num_workers=32, shuffle=True, pin_memory=True, persistent_workers=True)
dataloader_val = DataLoader(dataset_val, batch_size=512, num_workers=32, pin_memory=True, persistent_workers=True)

model1 = ResNet18(out_dim=2).to(args.device)
model2 = ResNet18(out_dim=2).to(args.device)

optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in tqdm.tqdm(range(100)):
    for x, y in dataloader_train:
        optimizer.zero_grad()
        x, y = x.to(args.device), y.to(args.device)
        pred1 = model1(x)
        pred2 = model2(x)
        loss = criterion(pred1, y) + criterion(pred2, y)
        loss.backward()
        optimizer.step()

preds1 = []
preds2 = []

with torch.no_grad():
    model1.eval()
    model2.eval()
    for x, _ in dataloader_val:
        x = x.to(args.device)
        pred1 = model1(x).argmax(1)
        pred2 = model2(x).argmax(1)
        preds1.append(pred1)
        preds2.append(pred2)

t_preds1 = torch.cat(preds1)
t_preds2 = torch.cat(preds2)

as_score = (t_preds1 == t_preds2).float().mean().item()

print("Agreement", as_score)


