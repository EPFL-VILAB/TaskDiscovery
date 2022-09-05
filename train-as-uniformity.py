import argparse
import os
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.loggers import WandbLogger

from datautils import MyCIFAR10DataModule
from models.as_uniformity import ASUniformityTraining
import utils

if __name__ == "__main__":
    config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                            help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--group', type=str, default='as-uniformity')
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--tags', type=str, nargs='*', default=[])
    parser.add_argument('--nologger', action='store_true', default=False)
    parser.add_argument('--resume_id', default=os.environ.get('JOB_UUID', ''))
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--no_resume', dest='resume', action='store_false', default=True)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--meta_steps', type=int, default=int(1e5))
    parser.add_argument('--encoder_learning_rate', type=float, default=1e-3)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--save_step_frequency', type=int, default=100)
    parser.add_argument('--noise', type=float, default=None)
    parser = ASUniformityTraining.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MyCIFAR10DataModule.add_argparse_args(parser)
    parser.set_defaults(num_sanity_val_steps=1)

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    utils.set_seeds(args.seed)

    SAVE_DIR = SAVE_DIR if not args.tmp else '/tmp/exps/'

    if not args.nologger:
        name = ('tmp-' if args.tmp else '') + args.name.format(**vars(args))
        logger = WandbLogger(
            name=name,
            project=PROJECT_NAME,
            entity='task-discovery',
            save_dir=SAVE_DIR,
            tags=['as', 'uniformity'] + args.tags,
            group=args.group,
            notes=args.notes,
            id=args.resume_id,
        )
        run = logger.experiment
        print(f'{run.resumed=}')
        checkpoint_callbacks = [
            utils.CheckpointEveryNSteps(
                save_step_frequency=args.save_step_frequency,
            )
        ]
    else:
        logger = None
        checkpoint_callbacks = None

    # check if there is checkpoint from the previous run
    ckpt_path = os.path.join(SAVE_DIR, PROJECT_NAME, args.resume_id, 'checkpoints', 'checkpoint.ckpt')
    if not os.path.exists(ckpt_path):
        if logger is not None and run.resumed:
            print(f'====> FAILED to find a checkpoint from the previous run: {ckpt_path}')
        ckpt_path = None

    if not args.resume:
        ckpt_path = None

    if args.ckpt and ckpt_path is None:
        model = ASUniformityTraining.load_from_checkpoint(args.ckpt, **vars(args))
        print(f'====> Loaded from checkpoint: {args.ckpt}')
    else:
        model = ASUniformityTraining(**vars(args))

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        logger=logger,
        log_every_n_steps=1,
        callbacks=checkpoint_callbacks,
        max_steps=args.meta_steps,
        num_sanity_val_steps=args.num_sanity_val_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        val_check_interval=args.val_check_interval,
        resume_from_checkpoint=ckpt_path,
    )

    if not args.test:
        trainer.fit(model, ckpt_path=ckpt_path)
        trainer.test(model)
    else:
        trainer.test(model, ckpt_path=ckpt_path)

