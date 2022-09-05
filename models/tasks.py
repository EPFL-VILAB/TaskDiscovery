from abc import abstractmethod
from typing import List, OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from itertools import combinations
from abc import ABC
from torch import Tensor

from .resnet import ResNet18
from .td_encoder import TaskDiscoveryEncoder


CIFAR_REAL_BIN_TASKS = [cls1 for cls1 in list(combinations(range(10), 5))[:126]]
CORE_REAL_TASKS_IDX = [0, 11, 18, 27, 31, 38, 40]

class Task(ABC, nn.Module):
    DIM = None

    def __init__(self):
        super().__init__()

    @abstractmethod    
    def loss(self, a, b):
        pass

    @abstractmethod
    def metrics(self, a, b):
        pass

class BaseClassificationTask(Task):
    def loss(self, prediction, target):
        assert (prediction.dim() == 2) & (target.shape[0] == prediction.shape[0]), f'{prediction.shape=}, {target.shape=}'
        if target.dim() == 1:
            # assume targets are classes
            return F.cross_entropy(prediction, target)
        elif target.dim() == 2:
            # assume targets are probabilities 
            return self.cross_entropy_loss(target, prediction)

    def metrics(self, prediction, target):
        labels = target if target.dim() == 1 else target.argmax(1)
        return {
            'cross_entropy': self.loss(prediction, target).item(),
            'acc': (prediction.argmax(1) == labels).float().mean().item(),
            'rate': prediction.argmax(1).float().mean().item()
        }

    @staticmethod
    def cross_entropy_loss(target, q):
        assert target.dim() == 2 and q.dim() == 2
        loss = -(F.softmax(target, 1) * F.log_softmax(q, 1)).sum(1)
        # loss = (F.softmax(p, 1) * F.log_softmax(q, 1)).sum(1) + (F.softmax(p, 1) * F.log_softmax(q, 1)).sum(1)
        return loss.mean()
    

class CIFARClassificationTask(BaseClassificationTask):
    N = 50000
    DIM = 2

    def __init__(
        self,
        task_type: str ='random',
        task_idx: int = 0,
        net_arch: str = 'resnet18',
    ):
        super().__init__()

        self.task_type = task_type
            
        if self.task_type == 'random':
            g = torch.Generator().manual_seed(task_idx)
            table = torch.randint(0, self.DIM, (self.N,), generator=g)
            print(f'[TASK] ===> Random task: {table[:20]}')
        elif self.task_type == 'real':
            assert self.DIM == 2, 'Only two-way classification tasks are supported'
            label2val = [int(i in CIFAR_REAL_BIN_TASKS[task_idx]) for i in range(10)]
            print(f'[TASK] ===> Real task: {label2val}')
            table = torch.LongTensor(label2val)
        elif self.task_type == 'real10':
            label2val = [i for i in range(10)]
            self.DIM = 10
            print(f'[TASK] ===> Real 10-way task: {label2val}')
            table = torch.LongTensor(label2val)
        elif self.task_type.startswith('net'):
            if net_arch == 'resnet18':
                self.task_net = ResNet18(out_dim=self.DIM)
            else:
                raise NotImplementedError
            table = torch.zeros(self.N,).long()
        elif self.task_type == 'table':
            table = torch.zeros(self.N).long()
            print(f'[TASK] ===> Table task: {table[:20]}')

        self.lookup_table = nn.parameter.Parameter(table, requires_grad=False)

    def forward(self, x=None, y=None, idx=None):
        if self.task_type in ["random", 'table']:
            assert idx is not None
            t = self.lookup_table[idx]
        elif self.task_type == "real" or self.task_type == "real10":
            assert y is not None
            t = self.lookup_table[y]
        elif self.task_type.startswith("net"):
            assert x is not None
            t = self.task_net(x)
            if not self.task_type.endswith('logits'):
                t = t.argmax(1)
        else:
            raise NotImplementedError

        return t

    def load_state_dict(self, state_dict: OrderedDict[str, Tensor], strict: bool = True):
        if self.task_type == 'net':
            res = self.task_net.load_state_dict(state_dict, strict=strict)
        else:
            res = super().load_state_dict(state_dict, strict=strict)

        print(f'[TASK] ===> Loaded "{self.task_type}" task: example={self.lookup_table[:20].tolist()}, mean={self.lookup_table.float().mean():.2f}')
        return res


class CIFAREmbeddingClassificationTask(BaseClassificationTask):
    DIM = 2

    def __init__(self, h_dim: int, in_dim: List[int] = (3,), out_type: str = 'logits', arch: str = 'resnet18', proj: str = 'linear'):
        super().__init__()
        # self.n_linear_tasks = n_linear_tasks or h_dim
        # self.task_idx = task_idx
        self.out_type = out_type

        self.encoder = TaskDiscoveryEncoder(
            in_dim=in_dim,
            h_dim=h_dim,
            arch=arch,
            proj=proj,
        )

    def forward(self, x=None, y=None, idx=None):
        self.encoder.eval()

        p = self.encoder(x)[0]
        if self.out_type == 'class':
            p = p.argmax(1)
        elif self.out_type == 'logits':
            pass
        else:
            raise ValueError(f'{self.out_type=}')
        return p