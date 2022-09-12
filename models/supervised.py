import pytorch_lightning as pl
import torch
import argparse
from torch import nn
from itertools import count

from models.resnet import ResNet18, ResNet50
from models.vit import ViT4
from models.fcnet import FCNet
from models.agreement_score import AgreementScore
from models.tasks import Task


class TwoSupervisedModels(pl.LightningModule):
    def __init__(
        self,
        arch: str = 'resnet18',
        learning_rate: float = 1e-3,
        save_hparams: bool = True,
        opt: str = 'adam',
        task: Task = None,
        agreement_score: AgreementScore = None,
        models_weights: str = '',
        width_factor: float = 1.,
        same_init: bool = False,
        breaking_point: int = -1,
        data_mode: str = 'meta',
        in_dim: int = 3, 
        **kwargs
    ) -> None:
        super().__init__()

        self.task = task
        self.agreement_score = agreement_score

        assert breaking_point == -1 or data_mode == 'meta'

        if arch == 'resnet18':
            model_f = lambda: ResNet18(out_dim=self.task.DIM, width_factor=width_factor)
        elif arch == 'resnet50':
            model_f = lambda: ResNet50(out_dim=self.task.DIM)
        elif arch == 'vit':
            model_f = lambda: ViT4(out_dim=self.task.DIM, batch_norm=True, width_factor=width_factor)
        elif arch == 'mlp':
            model_f = lambda: FCNet(out_dim=self.task.DIM, batch_norm=True, width_factor=width_factor)
        else:
            raise NotImplementedError

        self.models = nn.ModuleList([model_f(), model_f()])

        if same_init:
            print('===> SAME INIT')
            self.models[0].load_state_dict(self.models[1].state_dict())

        if models_weights != '':
            new_dict = torch.load(models_weights)
            state_dict = {}
            for k,v in new_dict.items():
                if k.startswith('models'):
                    k = '.'.join(k.split('.')[1:])
                state_dict[k] = v
            self.models.load_state_dict(state_dict)

        if save_hparams:
            self.save_hyperparameters(ignore='task')

    def log(self, name, value, *args, **kwargs):
        if hasattr(self, '_logger') and self._logger is not None:
            self._logger.log_metrics({
                name: value
            })
        super().log(name, value, *args, **kwargs)

    def log_metrics(self, metrics, step=None):
        if hasattr(self, '_logger') and self._logger is not None:
            self._logger.log_metrics(metrics, step=step)
        else:
            for k, v in metrics.items():
                self.log(k, v, prog_bar=('acc' in k))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--arch', type=str, default='resnet18')
        parser.add_argument('--opt', type=str, default='adam')
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--data_mode', type=str, default='meta')
        parser.add_argument('--same_init', default=False, action='store_true')
        parser.add_argument('--breaking_point', type=int, default=-1)
        parser.add_argument('--width_factor', type=float, default=1.0)
        return parser

    def training_step(self, batch, batch_idx, **kwargs):
        # TODO: change batch/inputs to dict?
        if self.hparams.data_mode == 'meta':
            if self.hparams.breaking_point == -1 or self.hparams.breaking_point < self.global_step:
                inputs = tuple(torch.chunk(b, 2, 0) for b in batch)
                inputs = tuple(zip(*inputs))
            else:
                # means we didn't reach the breaking point => use do full mode
                inputs = (batch, batch)
        elif self.hparams.data_mode == 'full':
            inputs = (batch, batch)
        else:
            raise NotImplementedError

        # models prediction based on x (assumed to be the 1st)
        y_hat = self(inputs[0][0], inputs[1][0], **kwargs)

        # get predictions based on all the available inputs
        ys = tuple([self.task(*inp) for inp in inputs])

        loss_0, loss_1 = (self.task.loss(p, y) for p, y in zip(y_hat, ys))

        logs = {
            'loss': 0.5 * (loss_0 + loss_1),
        }
        logs.update(self.agreement_score(*y_hat))

        for i, p, y in zip(count(), y_hat, ys):
            _m = self.task.metrics(p, y)
            logs.update({f'{k}_{i}':v for k, v in _m.items()})

        self.log_metrics(logs)

        return logs

    def forward(self, *xs):
        out = [self.models[0](xs[0]), self.models[1](xs[1])]
        return tuple(out)

    def _shared_val_step(self, batch, **kwargs):
        x = batch[0]

        y_hat = self(x, x, **kwargs)
        y = self.task(*batch)

        logs = {
            'loss': (self.task.loss(y_hat[0], y) + self.task.loss(y_hat[1], y)) / 2
        }

        for i, p in zip(count(), y_hat):
            _m = self.task.metrics(p, y)
            logs.update({f'{k}_{i}':v for k, v in _m.items()})

        logs.update(self.agreement_score(*y_hat))

        return logs
    
    def test_step(self, batch, batch_idx, **kwargs):
        logs = self._shared_val_step(batch)
        logs = {f'test_{k}':v for k, v in logs.items()}
        self.log_metrics(logs)
        return logs

    def validation_step(self, batch, batch_idx, **kwargs):
        logs = self._shared_val_step(batch, **kwargs)
        logs = {f'val_{k}':v for k, v in logs.items()}
        self.log_metrics(logs)
        return logs

    def configure_optimizers(self):
        if self.hparams.opt == 'sgd':
            opt = torch.optim.SGD(self.models.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.opt == 'adam':
            opt = torch.optim.Adam(self.models.parameters(), lr=self.hparams.learning_rate)
        return opt

    def reset_parameters(self) -> None:
        raise RuntimeError
        for m in self.models.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()