import numpy as np
from torch import nn
import torch

from models.resnet import ResNet18, ResNet50
from models.vit import ViT4
from models.fcnet import FCNet


class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False
        self.bias.zero_()

def id_backbone(x, **kwargs):
    return x

def flatten(x, **kwargs):
    return torch.flatten(x, start_dim=1)


class TaskDiscoveryEncoder(nn.Module):
    def __init__(self, in_dim=(3, -1, -1), h_dim=512, h_loss=None, nonlinearity='relu', arch='resnet18', proj='linear', freeze_backbone=False) -> None:
        super().__init__()
        if arch == 'resnet18':
            self.backbone = ResNet18(in_dim=in_dim[0], out_dim=2, nonlinearity=nonlinearity)
            backbone_dim = 512
            for p in self.backbone.linear.parameters():
                p.requires_grad = False
        elif arch == 'resnet50':
            self.backbone = ResNet50(out_dim=2)
            backbone_dim = 2048
        elif arch == 'id':
            self.backbone = id_backbone
            backbone_dim = in_dim
        elif arch == 'flatten':
            self.backbone = flatten
            backbone_dim = np.prod(in_dim)
        elif arch == 'vitbn':
            self.backbone = ViT4(out_dim=2, batch_norm=True)
            backbone_dim = 256
        elif arch  == 'mlpbn':
            self.backbone = FCNet(out_dim=2, batch_norm=True)
            backbone_dim=64
        else:
            raise ValueError(f'{arch=}')

        self.h_dim = h_dim if proj not in ['id', 'flatten'] else backbone_dim
        # h_loss is an additional loss on h (basically, uniformity in the current settting)
        self.h_loss = h_loss
        if proj == 'linear':
            self.projector = nn.Sequential(
                nn.Linear(backbone_dim, self.h_dim, bias=False),
                BatchNorm1dNoBias(self.h_dim),
            )
        elif proj == 'nlp':
            self.projector = nn.Sequential(
                nn.Linear(backbone_dim, backbone_dim, bias=False),
                nn.BatchNorm1d(backbone_dim),
                nn.ReLU(),
                nn.Linear(backbone_dim, self.h_dim, bias=False),
                BatchNorm1dNoBias(self.h_dim),
            )
        elif proj == 'id':
            self.projector = nn.Identity()
        else:
            raise ValueError(f'{proj=}')

        self.classifier = nn.Linear(self.h_dim, 2, bias=False)
        for p in self.classifier.parameters():
            p.requires_grad = False
        
        self.freeze_backbone = freeze_backbone
    
    def load_pretrained_encoder(self, encoder_path):
        print("Loading pretraining encoder doesn't work")
        model = ResNet18(2)

        ch_dict = torch.load(encoder_path)['state_dict']
        new_dict = {}
        for key in ch_dict:
            if "task_net" in key or 'models.0' in key:
                new_dict[key[9:]] = ch_dict[key]
        model.load_state_dict(new_dict)
        self.backbone.load_state_dict(new_dict, strict=False)

    def forward(self, x, out='y', get_loss=True):
        h = self.projector(self.backbone(x, penultimate=True))

        loss = torch.zeros(1).to(h.device)
        if self.h_loss is not None and get_loss:
            loss = self.h_loss(h)

        if out == 'h':
            return h, loss

        y = self.classifier(h)

        return y, loss
    
    def train(self, mode: bool = True):
        self.training = mode
        if self.freeze_backbone:
            self.projector.train()
        else:
            self.backbone.train()
            self.projector.train()