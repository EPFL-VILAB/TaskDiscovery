'''Fully Connected Network in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class FCNet(nn.Module):
    def __init__(self, out_dim=2, width_factor=1, batch_norm=False):
        super(FCNet, self).__init__()
        self.wf = width_factor
        
        self.fc1   = nn.Linear(3*32*32, round(1024 * self.wf))
        self.fc2   = nn.Linear(round(1024 * self.wf), round(512 * self.wf))
        self.fc3   = nn.Linear(round(512 * self.wf), round(256 * self.wf))
        self.fc4   = nn.Linear(round(256 * self.wf), round(64 * self.wf))
        self.fc5   = nn.Linear(round(64 * self.wf), out_dim)

        self.bn1 = nn.BatchNorm1d(round(1024 * self.wf)) if batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(round(512 * self.wf)) if batch_norm else nn.Identity()
        self.bn3 = nn.BatchNorm1d(round(256 * self.wf)) if batch_norm else nn.Identity()

    def forward(self, x, penultimate=False):
        out = x.view(x.size(0), -1)
        out = self.bn1(F.relu(self.fc1(out)))
        out = self.bn2(F.relu(self.fc2(out)))
        out = self.bn3(F.relu(self.fc3(out)))
        out = F.relu(self.fc4(out))
        if penultimate:
            return out
        out = self.fc5(out)
        return out
