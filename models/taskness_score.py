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
        n_classes: int = 2,
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
            model_f = lambda: ResNet18(in_dim=in_dim, out_dim=n_classes, width_factor=width_factor)
        elif arch == 'resnet50':
            model_f = lambda: ResNet50(out_dim=n_classes)
        elif arch == 'mlp':
            model_f = lambda: FCNet(out_dim=n_classes, batch_norm=True)
        elif arch == 'vit':
            model_f = lambda: ViT4(out_dim=n_classes, batch_norm=True)
            
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


class TasknessScore(pl.LightningModule):
    def __init__(
        self,
        netf: Callable[..., Type[nn.Module]] = ResNet18,
        nsteps: int = 10,
        nvalsteps: int = 1,
        batch_size: int = 512,
        shuffle_data: bool = False,
        task_entropy_coef: float = 1.,
        acc_threshold: float = 0.98,
        labels: str = 'logits',
        train_on_data_labelles: bool = False,
        sim_loss_norm_logits: bool = False,
        reset_models_init: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        print(f'====> {self.hparams.reset_models_init=}')

        kwargs['automatic_optimization'] = False
        # propagate named params to the models
        kwargs.update(dict(self.hparams))
        kwargs['task_net'] = ''
        self.models = TwoSupervisedClassifiers(
            **kwargs,
        )
        print(f'====> Models: \n{self.models}')
        self.fmodels = higher.monkeypatch(self.models)

        self.opt = self.models.configure_optimizers()
        print(f'====> Models Optimizer: \n{self.opt}')
        self.diffopt = higher.optim.get_diff_optim(
            opt=self.opt,
            reference_params=self.models.parameters(),
            device=self.device,
        )

        print('=======>>>>>>', kwargs.get('random_labelling', False))
        self._data_module = MyCIFAR10DataModule(
            val_split=self.hparams.n_val_images,
            n_train_images=self.hparams.n_train_images,
            shuffle=shuffle_data,
            batch_size=batch_size,
            n_classes=kwargs.get('K', 10),
            random_labelling=kwargs.get('random_labelling', False),
            gt2class=kwargs.get('gt2class', None) if train_on_data_labelles else '|airplane,automobile,ship,truck,horse,bird,cat,deer,dog,frog',
            return_indicies=self.hparams.use_task_lookup_table,
        )

        self._data_module.setup(None)
        self._train_dataloader = self._data_module.train_dataloader()
        self._val_dataloader = self._data_module.val_dataloader()
        self._test_dataloader = self._data_module.test_dataloader()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = TwoSupervisedClassifiers.add_model_specific_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--nsteps', type=int, default=50)
        parser.add_argument('--nvalsteps', type=int, default=1)
        parser.add_argument('--n_simsteps', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=512)
        parser.add_argument('--n_train_images', type=int, default=-1)
        parser.add_argument('--n_val_images', type=int, default=5000)
        parser.add_argument('--task_entropy_coef', type=float, default=1.)
        parser.add_argument('--mean_task_entropy_coef', type=float, default=1.)
        parser.add_argument('--similarity_loss_coef', type=float, default=1e-3)
        parser.add_argument('--sim_loss_ent_coef', type=float, default=0.3)
        parser.add_argument('--task_entropy_reg_type', type=str, default='sin')
        parser.add_argument('--rate_lb', type=float, default=0.)
        parser.add_argument('--rate_ub', type=float, default=1.)
        parser.add_argument('--max_type', type=str, default='smoothmax')
        parser.add_argument('--similarity_alpha', type=float, default=1.)
        parser.add_argument('--no_ats_optimization', action='store_true', default=False)
        parser.add_argument('--task_learning_rate', type=float, default=1e-3)
        parser.add_argument('--acc_threshold', type=float, default=.8)
        parser.add_argument('--sim_threshold', type=float, default=.7)
        parser.add_argument('--task_net', type=str, default='resnet18')
        parser.add_argument('--pretraining', action='store_true', default=False)
        parser.add_argument('--entropy_decay', type=float, default=0.1)
        parser.add_argument('--adaptive_coefs', action='store_true', default=False)
        parser.add_argument('--ats_coef', type=float, default=1.)
        parser.add_argument('--sim_loss_norm_logits', action='store_true', default=False)
        parser.add_argument('--n_presteps', type=int, default=2)
        parser.add_argument('--path2pool', type=str, default='')
        parser.add_argument('--save_every_epoch', action='store_true', default=False)
        parser.add_argument('--track_labels', action='store_true', default=False)
        parser.add_argument('--no_use_task_lookup_table', dest='use_task_lookup_table', action='store_false', default=True)
        parser.add_argument('--shuffle_data', dest='shuffle_data', action='store_true')
        parser.set_defaults(shuffle_data=False)
        parser.add_argument('--discovery_mode', dest='discovery_mode', action='store_true')
        parser.set_defaults(discovery_mode=False)
        parser.add_argument('--train_on_data_labelles', dest='train_on_data_labelles', action='store_true')
        parser.set_defaults(train_on_data_labelles=False)
        parser.set_defaults(labels='logits')
        parser.add_argument('--path2task_net', type=str, default='')
        parser.add_argument('--no_reset_models_init', dest='reset_models_init', action='store_false', default=True)
        return parser

    def log_metrics(self, metrics, step=None):
        for k, v in metrics.items():
            self.log(k, v)


class TasknessScoreWithTaskNet(TasknessScore):
    def __init__(self, task_net='resnet18', path2task_net='', *args, **kwargs):
        super(TasknessScoreWithTaskNet, self).__init__(*args, **kwargs)
        if task_net == 'resnet18':
            self.task_net = ResNet18(out_dim=kwargs['K'])   
        elif task_net == 'resnet18_nobn':
            self.task_net = ResNet18(out_dim=kwargs['K'], no_bn=True)
        else:
            raise NotImplementedError

        if path2task_net != '':
            self._load_task_net(path2task_net)

        if self.hparams.discovery_mode:
            self._task = deepcopy(self.task_net)
            for p in self._task.parameters():
                p.requires_grad = False

            self._task.eval()
            self._task_pool = []
            self._task_pool_pred = []

        self._rates = []
        val_batch_size = self._data_module.test_batch_size
        self._labels = torch.zeros(size=(len(self._val_dataloader) * val_batch_size,))
        self._conditions = []
        self._global_step = 0
        self._dummy_data = torch.utils.data.TensorDataset(torch.FloatTensor(1))

    def training_step(self, batch, batch_idx):
        if self.hparams.track_labels:
            self._track_labels()
        
        rate_condition = True
        similarity_condition = True
        ats_condition = True
        
        similarity_loss, logs = self._reg_step()
        logs['loss'] = similarity_loss

        if self.hparams.no_ats_optimization is False:
            task_reg, trained_params = self()
            self.phase = 'val'
            agreement_loss, agreement_acc, rate = self._eval_models(trained_params, self._val_dataloader)
            logs['loss'] = logs['loss'] + self.hparams.ats_coef * agreement_loss + task_reg
            logs['agreement_loss'] = agreement_loss
            logs['agreement_acc'] = agreement_acc
            logs['rate'] = rate.item()

            with torch.no_grad():
                self.phase = 'test'
                test_agreement_loss, test_agreement_acc = self._test_models(
                    trained_params,
                    self._test_dataloader,
                )
                logs['test_agreement_loss'] = test_agreement_loss.item()
                logs['test_agreement_acc'] = test_agreement_acc.item()
            
            self.log('agreement_loss', agreement_loss.item())
            self.log('agreement_acc', agreement_acc.item())
            self.log('test_agreement_loss', test_agreement_loss.item())
            self.log('test_agreement_acc', test_agreement_acc.item())
            self.log('mean_task_reg', task_reg.item() / self.hparams.nsteps)
            self.log('rate', rate.item())

            rate_condition = (self.hparams.rate_lb <= rate <= self.hparams.rate_ub).item()
            ats_condition = (agreement_acc > self.hparams.acc_threshold).item()

        if self.hparams.no_ats_optimization is True:
            self.log('rate', logs['rate'])
            self.log('task_entropy_avg', logs['task_entropy_avg']) 
            self.log('mean_task_entropy', logs['mean_task_entropy'])

        self.log('sim_entropy_loss', logs['entropy_loss'])
        self.log('total_loss', logs['loss'].item())
        self.log('similarity', logs['similarity'])
        self.log('similarity_loss', logs['similarity_loss'])

        similarity_condition = (logs['similarity'] < self.hparams.sim_threshold).item()
        discovery_condition = similarity_condition & ats_condition & rate_condition
        self._rates.append(logs['rate'])

        if self.hparams.adaptive_coefs:
            self._change_coefs(similarity_condition, ats_condition)
        
        if self.hparams.discovery_mode & discovery_condition:
            self._add_current_task_to_pool()
            self.reset_task()
            if self.hparams.reset_models_init: self.models.reset_init()
            self.reset()
            self.log('n_tasks', len(self._task_pool))
            logs['loss'] = torch.tensor(0.0, requires_grad=True)
            self._rates = []
            self._conditions = []

        if len(self._rates) > 50:
            del self._rates[0]

        # if len(self._rates) == 50 and 0.01 < np.mean(self._rates) < 0.99:
        #     self.hparams.sim_loss_ent_coef *= self.hparams.entropy_decay
        #     self.log('ent_loss_coef', self.hparams.sim_loss_ent_coef)
        #     self._rates = []
            
        return logs

    def _add_current_task_to_pool(self):
        # self._save_task_net(name=f'task_net_{len(self._task_pool)}.ckpt')
        # cpu_dict = {k: v.detach().cpu() for k, v in self.task_net.state_dict().items()}
        # self._task_pool.append(deepcopy(cpu_dict))
        # if self.hparams.use_task_lookup_table: self._add_task_pred(cpu_dict)

        task = CIFARClassificationTask(task_type='table')
        with torch.no_grad():
            for x, _, idxs in chain(self._train_dataloader, self._val_dataloader):
                self.task_net.eval()
                p = self.task_net(x.to(self.device)).argmax(1)
                task.lookup_table[idxs] = p.cpu()
        
        if self.logger is not None:
            torch.save(
                task.state_dict(),
                os.path.join(self.logger.experiment.dir, f'task_{len(self._task_pool)}.ckpt'),
            )
        self._task_pool.append(task)

    def _add_task_pred(self, state):
        self._task.load_state_dict(deepcopy({k: v.to(self.device) for k, v in state.items()}))

        n = len(self._train_dataloader.dataset)
        # train_dataset contains a subset of CIFAR10, but indicies are preserved so indicies can be out of bound
        task_pred = torch.zeros(len(self._train_dataloader.dataset.dataset)).long() - 1
        with torch.no_grad():
            for x, _, idxs in self._train_dataloader:
                self._task.eval()
                p = self._task(x.to(self.device)).argmax(1)
                task_pred[idxs] = p.cpu()
                n -= p.shape[0]
        assert n == 0, n
        self._task_pool_pred.append(task_pred)

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def _get_score_and_backward(self):
        self.models.reset()
        self.opt.load_state_dict(self.models._opt_init)

        with higher.innerloop_ctx(self.models, self.opt, copy_initial_weights=False) as (fmodels, diffopt):
            for i, batch in zip(range(self.hparams.nsteps), self._train_dataloader()):
                x = batch[0].to(self.device)
                loss = fmodels.training_step([x, self.task_net(x)], i)['loss']
                diffopt.step(loss)

            task_loss = 0.
            for i, batch in zip(range(self.hparams.nvalsteps), self._val_dataloader()):
                task_loss += fmodels.validation_step([batch[0].to(self.device),], i)['loss']

        return -task_loss

    def load_task_pool(self, dirpath):
        print('"=====> Loading the task pool...')
        task_pool = glob.glob(dirpath+'/task_[0-9]*.ckpt')

        for task_name in tqdm(task_pool):
            task = CIFARClassificationTask(task_type='table')
            res = task.load_state_dict(torch.load(task_name, map_location='cpu'), strict=False)
            assert len(res.missing_keys) == 0
            self._task_pool.append(task)

        # task_pool = glob.glob(dirpath+'/task_net_[0-9]*.ckpt')
        # for task_name in tqdm(task_pool):
        #     state = torch.load(task_name, map_location='cpu')
        #     self._task_pool.append(state)
        #     if self.hparams.use_task_lookup_table: self._add_task_pred(state)

        print(f"=====> Loaded {len(task_pool)} tasks")

    def _save_task_pool(self):
        if self.logger is not None:
            dirpath = self.logger.experiment.dir
        else:
            raise NotImplementedError

        for i in range(len(self._task_pool)):
            path = os.path.join(dirpath, f'task_{i}.ckpt')
            torch.save(self._task_pool[i], path)

    def _save_task_net(self, name='task_net.ckpt'):
        if self.logger is not None:
            path = os.path.join(self.logger.experiment.dir, name)
        else:
            path = name
        
        torch.save(self.task_net.state_dict(), path)

    def _save_init_models(self):
        if self.logger is not None:
            path = os.path.join(self.logger.experiment.dir, 'init_models.ckpt')
        else:
            path = 'init_models.ckpt'

        torch.save(self.models._models_init, path)
    
    def _load_task_net(self, path):
        state_dict = self.task_net.state_dict()
        new_dict = torch.load(path, map_location=self.device)
        for k, v in zip(state_dict.keys(), new_dict.values()):
            state_dict[k] = v
        self.task_net.load_state_dict(state_dict)

    def on_epoch_end(self):
        if self.hparams.save_every_epoch:
            name = f'epoch_{self.current_epoch}.ckpt' 
            self._save_task_net(name=name)

    def on_fit_start(self):
        if self.logger is not None:
            self._save_task_pool()
            self._save_task_net('init_task_net.ckpt')
            self._save_init_models()
            self.logger.log_hyperparams({'path2new_pool' : self.logger.experiment.dir})

    def _opt_step(self, x, task_reg, *params):
        is_first_pass = not torch.is_grad_enabled()
        detach = is_first_pass
        self.diffopt._track_higher_grads = not detach
        with torch.enable_grad():
            y = self.task_net(x)
            logs = self.fmodels.training_step([x, y], 0, params=params)

            task_entropy_avg = entropy_with_logits(y).mean()
            mean_task_entropy = entropy_with_logits(mean_categorical_with_logits(y))
            loc_task_reg = (self.hparams.task_entropy_coef * task_entropy_avg + self.hparams.mean_task_entropy_coef * (np.log(self.models.hparams.K) - mean_task_entropy))
            logs['task_entropy_avg'] = task_entropy_avg.item()
            logs['mean_task_entropy'] = mean_task_entropy.item()
            # TODO: generalize to K classes
            logs['class1_rate'] = (y.argmax(1) == 1).float().mean()
            logs['task_reg'] = loc_task_reg.item()

            new_params = self.diffopt.step(logs['loss'], params)

        if is_first_pass and self.logger is not None:
            self.logger.experiment.log({f'{"models_" if "task" not in k else ""}{k}': v for k, v in logs.items()})

        return (task_reg + loc_task_reg,) + tuple(tensor if tensor.requires_grad else tensor.clone().requires_grad_(True) for tensor in new_params)

    def _val_step(self, x, *params):
        is_first_pass = not torch.is_grad_enabled()
        with torch.no_grad():
            self.task_net.eval()
            y = self.task_net(x).argmax(1)
            self.task_net.train()

        logs = self.fmodels.validation_step([x, y], 0, params=params)

        if is_first_pass and self.logger is not None:
            phase = self.phase
            self.logger.experiment.log({f'models_{self.phase}_{k}': v for k, v in logs.items()},
                                        commit=False)
        
        return logs['loss'], torch.FloatTensor([logs['acc']]), logs['task_rate_val']

    def _reg_step(self):
        if not self.hparams.discovery_mode or len(self._task_pool) == 0:
            return torch.tensor(0.0, requires_grad=True), {
                'loss': torch.tensor(0.0, requires_grad=True),
                'similarity_loss': 0,
                'similarity': torch.tensor(0.0),
                'entropy_loss': 0,
                'task_entropy_avg': 0,
                'mean_task_entropy': 0,
                'rate': 0,
            }

        alpha = self.hparams.similarity_alpha
        coef = self.hparams.similarity_loss_coef
        
        similarity = 0.0
        loss = 0.0
        loss_1 = 0.0
        j = 0
        n_obj = 0
        entropy_loss = 0.
        rate = 0.0
        task_entropy_avg = 0.0
        mean_task_entropy = 0.0

        iterator = iter(self._train_dataloader)
        for j in range(self.hparams.n_simsteps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self._train_dataloader)
                batch = next(iterator)
            
            x = batch[0].to(self.device)
            batch_size = x.shape[0]
            n_obj += batch_size

            p = self.task_net(x)
            rate += (p.argmax(1) == 1).float().mean() / self.hparams.n_simsteps
            _task_entropy_avg = entropy_with_logits(p).mean().item()
            _mean_task_entropy = entropy_with_logits(mean_categorical_with_logits(p)).item()
            task_entropy_avg += _task_entropy_avg / self.hparams.n_simsteps
            mean_task_entropy += _mean_task_entropy / self.hparams.n_simsteps

            if self.hparams.sim_loss_norm_logits:
                p = (p - p.mean())/p.std() * 5

            with torch.no_grad():
                tasks_pred = []
                if self.hparams.use_task_lookup_table:
                    for i in range(len(self._task_pool)):
                        idxs = batch[2]
                        _p = self._task_pool[i](idx=idxs).to(self.device)
                        assert (_p != -1).all().item(), 'There is discrepancy between training objects'
                        tasks_pred.append(_p)
                        tasks_pred.append(_p^1)
                else:
                    raise NotImplementedError
                    for i in range(len(self._task_pool)):
                        gpu_dict = {k: v.to(self.device) for k, v in self._task_pool[i].items()}
                        self._task.load_state_dict(deepcopy(gpu_dict))
                        self._task.eval()
                        _p = self._task(x).argmax(1)
                        tasks_pred.append(_p)
                        tasks_pred.append(_p^1)

                tasks_pred = torch.cat(tasks_pred)

            new_pred = p.repeat(len(self._task_pool) * 2, 1)
            p1 = new_pred[:, 1] - new_pred[:, 0]

            tmp1 = tasks_pred * torch.sigmoid(p1) + (1 - tasks_pred) * torch.sigmoid(-p1)
            tmp1 = tmp1.reshape(-1, batch_size).sum(1)
            loss_1 += tmp1

            p1 = new_pred[0, 1] - new_pred[0, 0]
            if self.hparams.task_entropy_reg_type == 'gaussian':
                a, b = 6., 15.
                entropy_loss += (torch.exp(-(torch.sigmoid(p1)**2 + torch.sigmoid(-p1)**2)*b + a) - np.exp(-b + a)).sum()
            elif self.hparams.task_entropy_reg_type == 'sin':
                entropy_loss += (-torch.sin(torch.sigmoid(p1) * 2 * np.pi + np.pi/2) + 1.).sum()*np.log(2)/2
            elif self.hparams.task_entropy_reg_type == 'entropy':
                ent = entropy_with_logits(p)
                assert ent.dim() == 1
                entropy_loss += ent.sum()
            elif self.hparams.task_entropy_reg_type == 'none':
                pass
            else:
                raise NotImplementedError

            new_pred = new_pred.argmax(1).reshape(-1, batch_size)
            tasks_pred = tasks_pred.reshape(-1, batch_size)
            
            tmp_sim = (new_pred == tasks_pred).float().mean(1)
            similarity += tmp_sim

        # Multiply the number of used elements by the number of tasks
        # take max for each task to get similarity loss per task
        if self.hparams.max_type == 'smoothmax':
            similarity_loss = smooth_max(loss_1/n_obj, alpha=self.hparams.similarity_alpha) 
        elif self.hparams.max_type == 'softmax':
            #TODO fix softmax
            similarity_loss = (F.softmax(loss_1/n_obj, 0) * loss).sum(0)
        else:
            raise NotImplementedError

        # decay sim_loss_ent_coef linearly with the number of tasks to avoid overoptimizing it 
        # [empirical observation]
        ent_coef = self.hparams.sim_loss_ent_coef / (2 * len(self._task_pool))

        loss += coef * similarity_loss + ent_coef * entropy_loss
        similarity /= self.hparams.n_simsteps
        similarity = torch.max(similarity, 1-similarity).max()

        logs = {
            'loss': loss,
            'similarity_loss': similarity_loss.item(),
            'similarity': similarity,
            'entropy_loss': entropy_loss.item(),
            'task_entropy_avg': task_entropy_avg,
            'mean_task_entropy': mean_task_entropy,
            'rate': rate.item(),
        }
        return loss, logs

    def _pretrain(self):
        tmp_opt = torch.optim.Adam(self.models.parameters(), lr=1e-3)
        iterator = iter(self._train_dataloader)
        for _ in range(self.hparams.n_presteps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self._train_dataloader)
                batch = next(iterator)

            x = batch[0].to(self.device)
            with torch.no_grad():
                self.task_net.eval()
                y = self.task_net(x)
            tmp_opt.zero_grad()
            logs = self.models.training_step([x, y], 0)
            if self.logger is not None:
                self.logger.experiment.log({f'{"models_" if "task" not in k else ""}{k}': v for k, v in logs.items()})
            
            logs['loss'].backward()
            tmp_opt.step()
        torch.cuda.empty_cache()
    
    def _track_labels(self):
        if self._labels.device != self.device:
            self._labels = self._labels.to(self.device)

        with torch.no_grad():
            self.task_net.eval()
            cur_labels = []
            for batch in self._val_dataloader:
                x = batch[0].to(self.device)
                pred = self.task_net(x).argmax(1)
                cur_labels.append(pred)
            
            self.task_net.train()
        
        cur_labels = torch.cat(cur_labels)
        changed = (self._labels != cur_labels).float().mean()
        self.log('changed_rate', changed)
        self._labels = cur_labels

    def _change_coefs(self, similarity_condition, ats_condition):
        if not similarity_condition and not ats_condition:
            self._conditions.append(0)
        elif not similarity_condition:
            self._conditions.append(1)
        elif not ats_condition:
            self._conditions.append(0)

        if len(self._conditions) > 50:
            del self._conditions[0]
    
        if len(self._conditions) == 50:
            if np.mean(self._conditions) < 0.01:
                self.hparams.ats_coef *= 2
                self.hparams.similarity_loss_coef /= 2
                self._conditions = []
            elif np.mean(self._conditions) > 0.99:
                self.hparams.ats_coef /= 2
                self.hparams.similarity_loss_coef *= 2
                self._conditions = []

        self.log('ats_coef', self.hparams.ats_coef)
        self.log('similarity_loss_coef', self.hparams.similarity_loss_coef)

    def _train_models(self):
        task_reg = torch.zeros(1).to(self.device)
        
        if self.hparams.pretraining:
            self._pretrain()
        
        new_params = tuple(self.models.parameters())
        iterator = iter(self._train_dataloader)
        for _ in range(self.hparams.nsteps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self._train_dataloader)
                batch = next(iterator)

            x = batch[0].to(self.device)
            task_reg, *new_params = checkpoint(self._opt_step, x, task_reg, *new_params)

        task_reg /= self.hparams.nsteps
        return task_reg, new_params
    
    def _eval_models(self, params, dataloader):
        val_loss = 0
        agreement_acc = 0
        rate = 0

        iterator = iter(dataloader)
        for _ in range(self.hparams.nvalsteps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            x = batch[0].to(self.device)
            _loss, _acc, _rate = checkpoint(self._val_step, x, *params)
            val_loss += _loss / self.hparams.nvalsteps
            agreement_acc += _acc / self.hparams.nvalsteps
            rate += _rate / self.hparams.nvalsteps

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

    def forward(self):
        self.reset()

        return self._train_models()

    def reset_task(self):
        for m in self.task_net.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 1.

        self.optimizers().__setstate__({'state': defaultdict(dict)})

    def reset(self):
        self.models.reset()

        # TODO: seems to address the memory leakage over multiple meta-steps, BUT WHY?
        del self.diffopt
        del self.fmodels
        torch.cuda.empty_cache()
        
        self.models.zero_grad()
        
        self.fmodels = higher.monkeypatch(self.models)
        self.diffopt = higher.optim.get_diff_optim(
            opt=self.opt,
            reference_params=self.models.parameters(),
            device=self.device,
        )
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.task_net.parameters(), lr=self.hparams.task_learning_rate)
        print(f'========> Task Optimizer: \n {opt}')
        return opt

    def on_before_zero_grad(self, optimizer) -> None:
        print({n: p.grad for n, p in self.task_net.named_parameters()})
        return super().on_before_zero_grad(optimizer)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._dummy_data,
        )


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
