import os
import pytorch_lightning as pl
import torch
import yaml
import argparse
from pytorch_lightning.loggers import WandbLogger

from models.agreement_score import ClassificationAgreementScore
from models.tasks import CIFARClassificationTask, CIFAREmbeddingClassificationTask
from models.taskness_score import TwoSupervisedClassifiers
from models.supervised import TwoSupervisedModels
from datautils import MyCIFAR10DataModule

from tiny_imagenet import TinyImageNetDataModule
import utils

config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                           help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='tmp')
parser.add_argument('--notes', type=str, default='')
parser.add_argument('--group', type=str, default='ats')
parser.add_argument('--tags', type=str, nargs='*', default=[])
parser.add_argument('--tmp', dest='tmp', action='store_true')
parser.set_defaults(tmp=False)
parser.add_argument('--task', type=str, default='classification')
parser.add_argument('--task_net', type=str, default='')
parser.add_argument('--task_arch', type=str, default='resnet18')
parser.add_argument('--task_type', type=str, default='real')
parser.add_argument('--task_idx', type=int, default=0)
parser.add_argument('--task_mix', type=str, default='')
parser.add_argument('--task_ckpts', type=str, nargs='*', default=[])
parser.add_argument('--task_h_dim', type=int, default=512)
parser.add_argument('--task_out_type', type=str, default='class')
parser.add_argument('--emb_lin_task', type=str, default='learned')
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--noise', type=float, default=None)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--entity', type=str, default='task-discovery')
parser.add_argument('--project', type=str, default='task-discovery-repo')
# Trainer
parser.add_argument('--nologger', action='store_true', default=False)
parser.add_argument('--save_ckpt', action='store_true', default=False)
parser.add_argument('--save_dir', type=str, default=None)

parser = TwoSupervisedModels.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
parser = MyCIFAR10DataModule.add_argparse_args(parser)
parser.set_defaults(random_labelling=False)
parser.set_defaults(val_split=0.1)
parser.set_defaults(automatic_optimization=True)
parser.set_defaults(shuffle=True)

# first, load config if any
args_config, remaining = config_parser.parse_known_args()
if args_config.config:
    with open(args_config.config, 'r') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
# The main arg parser parses the rest of the args, the usual
# defaults will have been overridden if config file specified.
args = parser.parse_args(remaining)

utils.set_seeds(args.seed)

# Define datamodule
tmp_data_module = MyCIFAR10DataModule(
    data_dir=os.environ.get('DATA_ROOT', os.getcwd()),
    **{k: v for k, v in vars(args).items() if k not in ['data_dir', 'val_split', 'drop_last']},
    drop_last=False,
)

tmp_data_module.setup()

if args.dataset == 'cifar10':
    data_module = MyCIFAR10DataModule(
    data_dir=os.environ.get('DATA_ROOT', os.getcwd()),
        **{k: v for k, v in vars(args).items() if k not in ['data_dir']}
    )
elif args.dataset == 'tiny_imagenet':
    data_module = TinyImageNetDataModule(
        data_dir=os.environ.get('DATA_ROOT', os.getcwd()),
        **{k: v for k, v in vars(args).items() if k not in ['data_dir', 'val_split']}
    )
elif args.dataset == 'tiny_imagenet_64':
    data_module = TinyImageNetDataModule(
        data_dir=os.environ.get('DATA_ROOT', os.getcwd()),
        image_size=64,
    **{k: v for k, v in vars(args).items() if k not in ['data_dir', 'val_split']}
)


# Defining a task
if args.task == 'classification':
    if args.task_type == 'emb':
        task_fn = lambda: CIFAREmbeddingClassificationTask(
            h_dim=args.task_h_dim,
            in_dim=data_module.dims,
            out_type=args.task_out_type,
            arch=args.task_arch,
            n_classes=args.n_classes,
        )
    else:
        assert args.task_out_type == 'class'
        task_fn = lambda: CIFARClassificationTask(
            task_type=args.task_type,
            task_idx=args.task_idx,
            dataset=args.dataset,
            n_classes=args.n_classes,
        )
    agreement_score = ClassificationAgreementScore()


task = task_fn()
if len(args.task_ckpts) == 1 and args.task_ckpts[0] != '':
    if args.task_type == 'emb':
        from models.as_uniformity import ASUniformityTraining
        model = ASUniformityTraining.load_from_checkpoint(args.task_ckpts[0], dataset=args.dataset, arch=args.task_arch)
        if args.emb_lin_task == 'learned':
            model.set_task(idx=args.task_idx)
        elif args.emb_lin_task == 'random':
            model.set_random_task(seed=args.task_idx)
        else:
            raise ValueError(f'{args.emb_lin_task=}')

        task.encoder.backbone.load_state_dict(model.encoder.backbone.state_dict())
        task.encoder.projector.load_state_dict(model.encoder.projector.state_dict())

        if args.n_classes != model.hparams.get('n_classes', 2):
            print(f'===> !!!! Warning !!!  {args.n_classes} != {model.hparams.get("n_classes", 2)}')
            ws = utils.random_k_way_linear_task(args.n_classes, model.hparams.h_dim, args.task_idx)
            task.encoder.classifier.weight.copy_(model.hparams.task_temp * torch.FloatTensor(ws.T))
        else:
            task.encoder.classifier.load_state_dict(model.encoder.classifier.state_dict())

    else:
        task.load_state_dict(torch.load(args.task_ckpts[0]))
elif len(args.task_ckpts) > 1:
    raise RuntimeError

for p in task.parameters():
    p.requires_grad = False

task.eval()

# Two models module
task_discovery_model = TwoSupervisedModels(
    **{k: v for k, v in vars(args).items() if k not in ['task']},
    agreement_score=agreement_score,
    task=task,
    in_dim=data_module.dims[0],
)


name = ('tmp-' if args.tmp else '') + args.name.format(**vars(args))

if not args.nologger:
    logger = WandbLogger(
        name=name,
        project=args.project,
        entity=args.entity,
        save_dir=args.save_dir if not args.tmp else '/tmp/exps/',
        tags=['ats'] + args.tags,
        group=args.group.format(**vars(args)),
        notes=args.notes
    )
else:
    logger = None
    
trainer = pl.Trainer(
    gpus=torch.cuda.device_count(),
    logger=logger,
    log_every_n_steps=args.log_every_n_steps,
    max_epochs=args.max_epochs,
    max_steps=args.max_steps,
    val_check_interval=args.val_check_interval,
    check_val_every_n_epoch=args.check_val_every_n_epoch,
    limit_val_batches=args.limit_val_batches,
    deterministic=args.deterministic,
    checkpoint_callback=args.save_ckpt,
    default_root_dir=args.save_dir,
)

trainer.fit(task_discovery_model, datamodule=data_module)
trainer.test(task_discovery_model, datamodule=data_module)
