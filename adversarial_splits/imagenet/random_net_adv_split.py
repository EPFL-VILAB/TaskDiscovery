import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.models as models
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

torch.set_grad_enabled(False)
device = torch.device('cuda')


parser = argparse.ArgumentParser()
parser.add_argument('data', metavar='DIR', default='imagenet', help='path to imagenet *train* images')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 16)')
parser.add_argument('-b', '--batch-size', default=512, type=int)
parser.add_argument('--seed', default=0, type=int, help='seed for network initialization')
parser.add_argument('--class_split_seed', default=0, type=int, help='seed for splitting ImageNet classes')
parser.add_argument('--out_path', default='adv-split.torch')
args, _ = parser.parse_known_args()


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

model = models.resnet50()

# Change the last layer to output a single number as logit for sorting
model.fc = torch.nn.Linear(2048, 1)
model.to(device)
model.eval()

# Create the dataset
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_dataset = datasets.ImageFolder(
    args.data,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers,
)

# Obtaing logits corresponding to a randomly initialized network
logits, targets = [], []
for x, y in tqdm(train_loader):
    with torch.no_grad():
        l = model(x.to(device)).cpu().numpy()
    logits.append(l)
    targets.append(y.numpy())
logits = np.concatenate(logits)[:, 0]
targets = np.concatenate(targets)


def get_class_balanced_task(logits, targets, train_size=0.5):
    '''
    Creates a random net-based task that splits original ImageNet classes equally
    '''
    task = np.zeros_like(logits).astype(int)
    for y in np.unique(targets):
        y_idxs = np.where(targets == y)[0]
        idxs_sorted = sorted(y_idxs, key=lambda i: logits[i])
        t = int(len(y_idxs) * train_size)
        task[idxs_sorted[:t]] = 1
    return task

random_net_bin_class_balanced_task = get_class_balanced_task(logits, targets)

# randomly split original 1000 ImageNet calsses into 500/500
cls1, _ = train_test_split(np.arange(1000), train_size=500, random_state=args.class_split_seed)
real_bin_task = (targets[None] == cls1[:, None]).any(0)


def get_split(base_task, adv_task):
    '''
    Creates an adversarial split for the base_task based on the adv_task
    '''
    print(f'Train: {(base_task == adv_task).mean():.3f}')
    print(f'  Val: {(base_task != adv_task).mean():.3f}')
    return np.where(base_task == adv_task)[0], np.where(base_task != adv_task)[0]


train_idxs, val_idxs = get_split(real_bin_task, random_net_bin_class_balanced_task)
torch.save([train_idxs, val_idxs], args.out_path)
print(f'==> Save the split to: {args.out_path}')
