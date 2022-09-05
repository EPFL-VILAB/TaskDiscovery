import functools
import os
import wandb
import torch
import glob
import random
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
from itertools import combinations, chain

from torch import nn
from typing import Union, Callable, Type, Any, Optional, List
import abc
import argparse
from copy import deepcopy
from torch import autograd
import higher
from tqdm import tqdm
import numpy as np
import typing as t
from torch.distributions import Categorical
from torch.utils.checkpoint import checkpoint

from collections import defaultdict
from datautils import MyCIFAR10DataModule
from models.resnet import ResNet18, ResNet50
from models.fcnet import FCNet
from models.vit import ViT4
from models.tasks import CIFARClassificationTask
import utils

def smooth_max(inputs, dim=0, alpha=1.):
    return torch.logsumexp(inputs * alpha, dim) / alpha

def cross_entropy_loss(target, q):
    assert target.dim() == 2 and q.dim() == 2
    loss = -(F.softmax(target, 1) * F.log_softmax(q, 1)).sum(1)
    # loss = (F.softmax(p, 1) * F.log_softmax(q, 1)).sum(1) + (F.softmax(p, 1) * F.log_softmax(q, 1)).sum(1)
    return loss.mean()

def kl_loss(inputs, target):
    kl = F.softmax(target, 1) * (F.log_softmax(target, 1) - F.log_softmax(inputs, 1))
    return kl.mean()

def entropy_with_logits(logits):
    return Categorical(logits=logits).entropy()

def mean_categorical_with_logits(logits):
    logp = torch.log_softmax(logits, dim=1)
    logp_mean = torch.logsumexp(logp - np.log(logits.shape[0]*1.), dim=0)
    return logp_mean


def hsic(x, y):
    x = x - x.mean(1)[:, None]
    y = y - y.mean(1)[:, None]
    return torch.sum(x.t() * y) / (x.shape[0] - 1)**2


def features2cka(x, y):    
    x = x - x.mean(0)[None]
    y = y - y.mean(0)[None]

    x = x @ x.t()
    y = y @ y.t()

    return hsic(x, y)/np.sqrt(hsic(x, x) * hsic(y, y))

def pairwise_cka(features):
    cka = []
    for x, y in combinations(features, 2):
        cka.append(features2cka(x, y))
    return np.array(cka)


class TwoSupervisedClassifiers(pl.LightningModule):
    def __init__(
        self,
        arch: str = 'resnet18',
        learning_rate: float = 1e-3,
        save_hparams: bool = True,
        automatic_optimization: bool = True,
        K: int = 2,
        random_init: bool = True,
        labels: str = 'classes',
        opt: str = 'sgd',
        task_net: str = '',
        models_weights: str = '',
        width_factor: float = 1.,
        ag_loss_norm_logits: bool = False,
        ag_loss: str = 'kl',
        entropy_reg: float = 0.,
        log: bool = True,
        in_dim: int = 3,
        **kwargs
    ) -> None:
        super().__init__()

        self._log = log

        self.save_hyperparameters()

        if arch == 'resnet18':
            model_f = lambda: ResNet18(in_dim=in_dim, out_dim=K, width_factor=width_factor)
        elif arch == 'resnet50':
            model_f = lambda: ResNet50(out_dim=K)
        elif arch == 'mlpbn':
            model_f = lambda: FCNet(out_dim=K, batch_norm=True)
        elif arch == 'vitbn':
            model_f = lambda: ViT4(out_dim=K, batch_norm=True)
            
        # elif arch == 'efficient_net':
        #     model_f = lambda: timm.create_model('efficientnet_b3', num_classes=2)

        self.models = nn.ModuleList([model_f(), model_f()])
        if models_weights != '':
            new_dict = torch.load(models_weights)
            state_dict = {}
            for k,v in new_dict.items():
                if k.startswith('models'):
                    k = '.'.join(k.split('.')[1:])
                state_dict[k] = v
            self.models.load_state_dict(state_dict)
            
        self._models_init = deepcopy(self.models.state_dict())
        self.task_net = None
        if isinstance(task_net, str) and task_net != '':
            self.task_net = model_f()
            self.task_net.load_state_dict(torch.load(task_net))
            for p in self.task_net.parameters():
                p.requires_grad = False
        elif isinstance(task_net, nn.Module):
            self.task_net = task_net
            for p in self.task_net.parameters():
                p.requires_grad = False

        self._data = None

        self.automatic_optimization = automatic_optimization

        self._logger = None

    def reset_init(self):
        print('===> Reset models init')
        for m in self.models.modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.GroupNorm)):
                m.reset_parameters()

        del self._models_init
        self._models_init = deepcopy(self.models.state_dict())

    def reset(self):
        if self.hparams.random_init:
            for m in self.models.modules():
                if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.GroupNorm)):
                    m.reset_parameters()
        else:
            self.models.load_state_dict(self._models_init)

    def log(self, name, value, *args, **kwargs):
        if not self._log:
            return

        if self._logger is not None:
            self._logger.log_metrics({
                name: value
            })
        super().log(name, value, *args, **kwargs)

    def log_metrics(self, metrics, step=None):
        if self._logger is not None:
            self._logger.log_metrics(metrics, step=step)
        else:
            for k, v in metrics.items():
                self.log(k, v)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--arch', type=str, default='resnet18')
        parser.add_argument('--width_factor', type=float, default=1.)
        parser.add_argument('--opt', type=str, default='sgd')
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--entropy_reg', type=float, default=0.)
        parser.add_argument('--k', '--K', dest='K', type=int, default=2)
        parser.add_argument('--random_init', action='store_true', default=False)
        parser.add_argument('--ag_loss_norm_logits', action='store_true', default=False)
        parser.add_argument('--labels', type=str, default='classes')
        parser.add_argument('--data_mode', type=str, default='meta')
        parser.add_argument('--ag_loss', type=str, default='ce')
        parser.add_argument('--train_loss', type=str, default='ce')
        return parser

    def _train_loss(self, y_hat, ys):
        if self.hparams.labels == 'classes':
            loss1 = F.cross_entropy(y_hat[0], ys[0])
            loss2 = F.cross_entropy(y_hat[1], ys[1])
            return loss1, loss2
        elif self.hparams.labels == 'logits':
            if self.hparams.train_loss == 'kl':
                loss1 = kl_loss(y_hat[0], ys[0])
                loss2 = kl_loss(y_hat[1], ys[1])
            elif self.hparams.train_loss == 'ce':
                loss1 = cross_entropy_loss(ys[0], y_hat[0])
                loss2 = cross_entropy_loss(ys[1], y_hat[1])
            return loss1, loss2
        else:
            raise NotImplementedError

    def training_step(self, batch, batch_idx, **kwargs):
        # batch = [[x1, y1], [x2, y2]]
        if self.hparams.data_mode == 'meta':
            xs, ys = torch.chunk(batch[0], 2, 0), torch.chunk(batch[1], 2, 0)
        elif self.hparams.data_mode == 'full':
            x, y = batch
            xs = x.unsqueeze(0).repeat(2, *([1]*x.dim()))
            ys = y.unsqueeze(0).repeat(2, *([1]*y.dim()))
        elif self.hparams.data_mode == 'independent':
            batch_1, batch_2 = batch
            xs = (batch_1[0], batch_2[0])
            ys = (batch_1[1], batch_2[1])
        else:
            raise NotImplementedError


        y_hat = self(*xs, **kwargs)
        
        model0_rate = y_hat[0].argmax(1).float().mean()
        model1_rate = y_hat[1].argmax(1).float().mean()
        
        if self.task_net is not None:
            self.task_net.eval()
            if self.hparams.data_mode == 'independent':
                ys = self.task_net(torch.cat((batch_1[0], batch_2[0]), dim=0))
            else:
                ys = self.task_net(batch[0])

            if self.hparams.labels == 'classes':
                ys = ys.argmax(1).long()
            
            if self.hparams.data_mode == 'full':
                ys = ys.unsqueeze(0).repeat(2, *([1]*ys.dim()))
            else:
                ys = torch.chunk(ys, 2, 0)

        loss_0, loss_1 = self._train_loss(y_hat, ys)
        total_loss = 0.5 * (loss_0 + loss_1)

        if self.hparams.labels == 'logits':
            ys = list(map(lambda y: y.argmax(1), ys))

        task_rate_0 = ys[0].float().mean()
        task_rate_1 = ys[1].float().mean()

        logs = {
            'entropy': (entropy_with_logits(y_hat[0]).mean() + entropy_with_logits(y_hat[1]).mean()).item() * 0.5,
            'loss_0': loss_0.item(),
            'loss_1': loss_1.item(),
            'task_rate_0': task_rate_0.item(),
            'task_rate_1': task_rate_1.item(),
            'train_acc_0': (y_hat[0].argmax(1) == ys[0]).float().mean().item(),
            'train_acc_1': (y_hat[1].argmax(1) == ys[1]).float().mean().item(),
            'model0_rate': model0_rate.item(),
            'model1_rate': model1_rate.item(),
            'entropy_0': entropy_with_logits(y_hat[0]).mean().item(),
            'entropy_1': entropy_with_logits(y_hat[1]).mean().item(),
            'agreement_loss': self._agreement_loss(*y_hat),
            'agreement_acc': (y_hat[0].argmax(1) == y_hat[1].argmax(1)).float().mean().item()
        }
        self.log_metrics(logs)
        return {
            'loss': total_loss,
            'acc': 0.5 * (logs['train_acc_0'] + logs['train_acc_1']),
            'loss_0': loss_0.item(),
            'loss_1': loss_1.item(),
            'task_rate_0': task_rate_0.item(),
            'task_rate_1': task_rate_1.item(),
            'train_acc_0': (y_hat[0].argmax(1) == ys[0]).float().mean().item(),
            'train_acc_1': (y_hat[1].argmax(1) == ys[1]).float().mean().item(),
            'model0_rate': model0_rate.item(),
            'model1_rate': model1_rate.item(),
            'entropy_0': entropy_with_logits(y_hat[0]).mean().item(),
            'entropy_1': entropy_with_logits(y_hat[1]).mean().item(),
            'agreement_loss': self._agreement_loss(*y_hat),
            'agreement_acc': (y_hat[0].argmax(1) == y_hat[1].argmax(1)).float().mean().item()
        }

    def forward(self, *xs):
        out = [self.models[0](xs[0]), self.models[1](xs[1])]
        # for x, midx in zip(xs, range(2)):
            # out.append(self.models[midx](x))
        return tuple(out)

    def _agreement_loss(self, p1, p2):
        if self.hparams.ag_loss == 'kl':
            agreement_loss = 0.5 * (kl_loss(p1, p2) + kl_loss(p2, p1))
        elif self.hparams.ag_loss == 'ce':
            agreement_loss = 0.5 * (cross_entropy_loss(p1, p2) + cross_entropy_loss(p2, p1))
            if self.hparams.entropy_reg != 0.:
                assert self.hparams.entropy_reg < 1.
                agreement_loss += self.hparams.entropy_reg * (entropy_with_logits(p1) + entropy_with_logits(p1)).mean()
        else:
            raise NotImplementedError

        return agreement_loss

    def _shared_step(self, batch, batch_idx, **kwargs):
        if len(batch) == 1:
            x = batch[0]
        elif len(batch) == 2:
            x, y = batch
        else:
            raise NotImplementedError

        if self.task_net is not None:
            with torch.no_grad():
                self.task_net.eval()
                y = self.task_net(x).argmax(1)

        p1, p2 = self(x, x, **kwargs)

        logs = {
            'entropy_0': entropy_with_logits(p1).mean().item(),
            'entropy_1': entropy_with_logits(p2).mean().item(),
        }

        if self.hparams.ag_loss_norm_logits:
            p1 = (p1 - p1.mean())/p1.std() * 5
            p2 = (p2 - p2.mean())/p2.std() * 5

        model0_rate = p1.argmax(1).float().mean()
        model1_rate = p2.argmax(1).float().mean()
        task_rate = y.float().mean()

        agreement_loss = self._agreement_loss(p1, p2)

        agreement_acc = (p1.argmax(1) == p2.argmax(1)).float().mean().item()
        all_equal = ((p1.argmax(1) == y) & (p2.argmax(1) == y)).float().mean().item()
        acc0 = (p1.argmax(1) == y).float().mean().item()
        acc1 = (p2.argmax(1) == y).float().mean().item()

        logs.update({
            'loss': agreement_loss,
            'agreement_loss': agreement_loss.item(),
            'acc_0': acc0,
            'acc_1': acc1,
            'model0_rate': model0_rate,
            'model1_rate': model1_rate,
            'task_rate': task_rate,
            'all_equal': all_equal,
            'agreement_acc': agreement_acc,
        })

        return logs
    
    def test_step(self, *args, **kwargs):
        logs = self._shared_step(*args, **kwargs)
        logs = {f'test_{k}':v for k, v in logs.items()}
        self.log_metrics(logs)
        return logs

    def validation_step(self, batch, batch_idx, **kwargs):
        if len(batch) == 1:
            x = batch[0]
        elif len(batch) == 2:
            x, y = batch
        else:
            raise NotImplementedError

        if self.task_net is not None:
            with torch.no_grad():
                self.task_net.eval()
                y = self.task_net(x).argmax(1)

        p1, p2 = self(x, x, **kwargs)

        logs = {
            'val_entropy_0': entropy_with_logits(p1).mean().item(),
            'val_entropy_1': entropy_with_logits(p2).mean().item(),
        }

        if self.hparams.ag_loss_norm_logits:
            p1 = (p1 - p1.mean())/p1.std() * 5
            p2 = (p2 - p2.mean())/p2.std() * 5

        model0_rate = p1.argmax(1).float().mean()
        model1_rate = p2.argmax(1).float().mean()
        task_rate = y.float().mean()

        if self.hparams.ag_loss == 'kl':
            agreement_loss = 0.5 * (kl_loss(p1, p2) + kl_loss(p2, p1))
        elif self.hparams.ag_loss == 'ce':
            agreement_loss = 0.5 * (cross_entropy_loss(p1, p2) + cross_entropy_loss(p2, p1))
            if self.hparams.entropy_reg != 0.:
                assert self.hparams.entropy_reg < 1.
                agreement_loss += self.hparams.entropy_reg * (entropy_with_logits(p1) + entropy_with_logits(p1)).mean()
        else:
            raise NotImplementedError

        agreement_acc = (p1.argmax(1) == p2.argmax(1)).float().mean().item()
        all_equal = ((p1.argmax(1) == y) & (p2.argmax(1) == y)).float().mean().item()
        acc0 = (p1.argmax(1) == y).float().mean().item()
        acc1 = (p2.argmax(1) == y).float().mean().item()

        logs.update({
            'loss': agreement_loss,
            'val_acc_0': acc0,
            'val_acc_1': acc1,
            'model0_rate_val': model0_rate,
            'model1_rate_val': model1_rate,
            'task_rate_val': task_rate,
            'all_equal': all_equal,
            'acc': agreement_acc,
        })

        self.log_metrics(logs)

        return logs

    def configure_optimizers(self, learning_rate=None):
        learning_rate = learning_rate or self.hparams.learning_rate
        if self.hparams.opt == 'sgd':
            opt = torch.optim.SGD(self.models.parameters(), lr=learning_rate)
        elif self.hparams.opt == 'adam':
            opt = torch.optim.Adam(self.models.parameters(), lr=learning_rate)
        self._opt_init = deepcopy(opt.state_dict())
        return opt

    def reset_parameters(self) -> None:
        for m in self.models.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

class AgreementLoss(pl.LightningModule):
    def __init__(
        self,
        in_dim: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # propagate named params to the models
        kwargs['automatic_optimization'] = False
        kwargs.pop('learning_rate')
        kwargs.pop('arch')
        self.models = TwoSupervisedClassifiers(
            **kwargs,
            in_dim=in_dim,
            learning_rate=kwargs['as_learning_rate'],
            arch=kwargs['as_arch'],
            log=False,
        )
        print(f'====> Models: \n{self.models}')
        self.fmodels = higher.monkeypatch(self.models)

        assert kwargs['opt'] == 'sgd', 'Implement optimizer stated init restoring first!'
        self.opt = self.models.configure_optimizers(learning_rate=self.hparams.as_learning_rate)
        print(f'====> Models Optimizer: \n{self.opt}')
        self.diffopt = higher.optim.get_diff_optim(
            opt=self.opt,
            reference_params=self.models.parameters(),
            device=self.device,
        )
        self._models_logs = {}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = TwoSupervisedClassifiers.add_model_specific_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--nsteps', type=int, default=50)
        parser.add_argument('--nvalsteps', type=int, default=1)
        parser.add_argument('--as_learning_rate', type=float, default=0.01)
        parser.set_defaults(labels='logits')
        return parser

    def _save_init_models(self):
        if self.logger is not None:
            path = os.path.join(self.logger.experiment.dir, 'init_models.ckpt')
        else:
            path = 'init_models.ckpt'

        torch.save(self.models._models_init, path)

    def _opt_step(self, task, x, h_loss, *params):
        is_first_pass = not torch.is_grad_enabled()
        detach = is_first_pass
        self.diffopt._track_higher_grads = not detach
        with torch.enable_grad():
            y, _h_loss = task(x)
            logs = self.fmodels.training_step([x, y], 0, params=params)
            new_params = self.diffopt.step(logs['loss'], params)

            self._models_logs.update({f'models/{k}': v for k, v in logs.items()})

        # !!! IMPORTANT !!! note, that it MUST NOT be nested, otherwise no grads are propagated
        return (h_loss + _h_loss, ) + tuple(tensor if tensor.requires_grad else tensor.clone().requires_grad_(True) for tensor in new_params)

    def _val_step(self, task, x, *params):
        with torch.no_grad():
            y = task(x, get_loss=False)[0].argmax(1)

        logs = self.fmodels.validation_step([x, y], 0, params=params)
        
        return logs['loss'], torch.FloatTensor([logs['acc']]), logs['task_rate_val']

    def _train_models(self, task, dataloader):
        new_params = tuple(self.models.parameters())
        iterator = iter(dataloader)
        h_loss = torch.zeros(1).to(self.device)

        for _ in range(self.hparams.nsteps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            x = batch[0].to(self.device)
            opt_step = functools.partial(self._opt_step, task)
            h_loss, *new_params = checkpoint(opt_step, x, h_loss, *new_params)
        
        if self.hparams.nsteps == 0:
            batch = next(iterator)
            x = batch[0].to(self.device)
            _, h_loss = task(x)

        return h_loss / max(self.hparams.nsteps, 1), new_params

    def _eval_models(self, task, params, dataloader):
        val_loss = 0
        agreement_acc = 0
        rate = 0
        task.eval()

        iterator = iter(dataloader)
        for _ in range(self.hparams.nvalsteps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            x = batch[0].to(self.device)
            if len(batch) == 4:
                # => tasks is on factors
                task_f = functools.partial(task, factors=batch[3].to(self.device))
            else:
                task_f = task

            val_step = functools.partial(self._val_step, task_f)
            _loss, _acc, _rate = checkpoint(val_step, x, *params)
            val_loss += _loss / self.hparams.nvalsteps
            agreement_acc += _acc / self.hparams.nvalsteps
            rate += _rate / self.hparams.nvalsteps

        task.train()

        return val_loss, agreement_acc, rate

    def _test_models(self, params, dataloader):
        test_loss = 0
        agreement_acc = 0
        for batch in dataloader:
            x = batch[0].to(self.device)
            _loss, _acc, _rate = checkpoint(self._val_step, x, *params)
            test_loss += _loss / len(dataloader)
            agreement_acc += _acc / len(dataloader)
                
        return test_loss, agreement_acc

    def forward(self, task, train_loader, val_loader):
        logs = {}
        self.reset()

        h_loss, trained_params = self._train_models(task, train_loader)
        agreement_loss, agreement_acc, rate = self._eval_models(task, trained_params, val_loader)

        logs['loss'] = agreement_loss
        logs['h_loss'] = h_loss
        logs['agreement_loss'] = agreement_loss.item()
        logs['agreement_acc'] = agreement_acc.item()
        logs['rate'] = rate.item()

        # add models' logs from the last step
        logs.update({k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in self._models_logs.items()})

        return logs

    def reset(self):
        self.models.reset()

        # TODO: seems to address the memory leakage over multiple meta-steps, BUT WHY?
        # del self.diffopt
        # del self.fmodels
        # torch.cuda.empty_cache()

        self.models.zero_grad()

        self.fmodels = higher.monkeypatch(self.models)
        self.diffopt = higher.optim.get_diff_optim(
            opt=self.opt,
            reference_params=self.models.parameters(),
            device=self.device,
        )

        torch.cuda.empty_cache()

    def log(self, *args, **kwargs):
        pass


class FOXASLoss(AgreementLoss):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._models_init = deepcopy(self.models.state_dict())
        self._opt_init = deepcopy(self.opt.state_dict())

    def forward(self, task, train_batch, val_batch, backward_fn, reset=True):
        if reset: self.reset()

        logs = {}
        x_train = train_batch[0].to(self.device)
        with higher.innerloop_ctx(self.models, self.opt, copy_initial_weights=True) as (fmodels, diffopt):
            y, h_loss = task(x_train)
            logs['h_loss'] = h_loss
            logs_train = fmodels.training_step([x_train, y], 0)
            diffopt.step(logs_train['loss'])
            logs.update({f'models_train/{k}': v for k, v in logs_train.items()})

            x_val = val_batch[0].to(self.device)
            with torch.no_grad():
                task.eval()
                y_val = task(x_val, get_loss=False)[0].argmax(1)
                task.train()

            logs_val = fmodels.validation_step([x_val, y_val], 0)

            # TODO: this is quite ugly
            backward_fn(logs_val['loss'], h_loss)

            logs.update({f'models_val/{k}': v for k, v in logs_val.items()})

            self.models.load_state_dict(fmodels.state_dict())

        return {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in logs.items()}

    def reset(self):
        self.models.load_state_dict(self._models_init)
        self.opt.load_state_dict(self._opt_init)

class TrainingXASLoss(AgreementLoss):
    def __init__(self, track_loss_from=0, decay_rate=1., early_stop_xas_n_steps=-1, early_stop_xas_threshold=-1, min_nsteps=5,  **kwargs) -> None:
        super().__init__(**kwargs)
        assert self.hparams.nsteps >= self.hparams.min_nsteps

    def _opt_step(self, task, x, h_loss, loss, *params):
        is_first_pass = not torch.is_grad_enabled()
        detach = is_first_pass
        self.diffopt._track_higher_grads = not detach
        with torch.enable_grad():
            y, _h_loss = task(x)
            logs = self.fmodels.training_step([x, y], 0, params=params)
            new_params = self.diffopt.step(logs['loss'], params)

            self._models_logs.update({f'models/{k}': v for k, v in logs.items()})

        # !!! IMPORTANT !!! note, that it MUST NOT be nested, otherwise no grads are propagated
        return (h_loss + _h_loss, loss + logs['agreement_loss']) + tuple(tensor if tensor.requires_grad else tensor.clone().requires_grad_(True) for tensor in new_params)

    def _train_models(self, task, dataloader):
        logs = {}
        new_params = tuple(self.models.parameters())
        h_loss = torch.zeros(1).to(self.device)
        loss = torch.zeros(1).to(self.device)
        xas_acc = []
        i = 0

        for i, batch in zip(range(1, self.hparams.nsteps + 1), dataloader):
            if self.hparams.track_loss_from == i:
                # discard previous track
                loss = torch.zeros(1).to(self.device)

            x = batch[0].to(self.device)
            if len(batch) == 4:
                # => tasks is on factors
                task_f = functools.partial(task, factors=batch[3].to(self.device))
            else:
                task_f = task
            opt_step = functools.partial(self._opt_step, task_f)
            h_loss, loss, *new_params = checkpoint(opt_step, x, h_loss, loss, *new_params)
            loss = loss * self.hparams.decay_rate

            if self.hparams.early_stop_xas_n_steps != -1:
                assert self.hparams.early_stop_xas_threshold != -1
                xas_acc.append(self._models_logs['models/agreement_acc'])
                xas_acc = xas_acc[-self.hparams.early_stop_xas_n_steps:]
                if len(xas_acc) == self.hparams.early_stop_xas_n_steps and (np.array(xas_acc) > self.hparams.early_stop_xas_threshold).all() and i >= self.hparams.min_nsteps:
                    break

        if self.hparams.nsteps == 0:
            batch = next(iter(dataloader))
            x = batch[0].to(self.device)
            _, h_loss = task(x)

        logs['loss'] = loss / max(i - self.hparams.track_loss_from, 1)
        logs['h_loss'] = h_loss / max(i, 1)
        logs['steps'] = i

        return logs, new_params

    def forward(self, task, train_loader, val_loader=None):
        assert len(train_loader) >= self.hparams.nsteps, 'TrainingXAS loss should not be used for more than 1 epoch!'
        self.reset()

        logs, trained_params = self._train_models(task, train_loader)
        with torch.no_grad():
            if val_loader is not None:
                agreement_loss, agreement_acc, rate = self._eval_models(task, trained_params, val_loader)
            else:
                agreement_loss, agreement_acc, rate = torch.zeros(3).to(self.device)

        logs['agreement_loss'] = agreement_loss.item()
        logs['agreement_acc'] = agreement_acc.item()
        logs['rate'] = rate.item()

        # add models' logs from the last step
        logs.update({k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in self._models_logs.items()})

        return logs


class TrainingProxyASLoss(TrainingXASLoss):
    def _opt_step(self, task, x, h_loss, loss, *params):
        is_first_pass = not torch.is_grad_enabled()
        detach = is_first_pass
        self.diffopt._track_higher_grads = not detach
        with torch.enable_grad():
            y, _h_loss = task(x)
            logs = self.fmodels.training_step([x, y], 0, params=params)
            new_params = self.diffopt.step(logs['loss'], params)

            self._models_logs.update({f'models/{k}': v for k, v in logs.items()})

        # !!! IMPORTANT !!! note, that it MUST NOT be nested, otherwise no grads are propagated
        return (h_loss + _h_loss, loss + logs['loss']) + tuple(tensor if tensor.requires_grad else tensor.clone().requires_grad_(True) for tensor in new_params)


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    d = torch.device('cuda')

    task_net = ResNet18().to(d).train()
    task_net_init = deepcopy(task_net.state_dict())

    tscore = TasknessScoreWithTaskNet(
        netf=ResNet18,
        nsteps=4,
        nvalsteps=1,
        batch_size=128,
        task_entropy_coef=0,
        learning_rate=1e-2,
    ).to(d)
    tscore_init = deepcopy(tscore.state_dict())

    score = tscore._get_score_and_backward_higher(task_net)
    dl_dtask_gt = []
    for p in task_net.parameters():
        dl_dtask_gt.append(torch.clone(p.grad))
        p.grad = None
    print(f'===> GT score={score.item():.3f}')

    task_net.load_state_dict(task_net_init)
    tscore.load_state_dict(tscore_init)
    score = tscore.get_score(task_net)
    grads = tscore.get_task_grads(task_net)

    print(f'===> MY score={score.item():.3f}')

    print(*[f'{torch.norm(g_gt-g)/torch.norm(g_gt):.4f}' for g_gt, g in zip(dl_dtask_gt, grads)])
