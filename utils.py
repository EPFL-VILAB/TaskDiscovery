import os
import numpy as np
import torch
import pytorch_lightning as pl
import random
import functools

from torch.distributions import Categorical
from itertools import combinations
from sklearn.metrics import pairwise_distances
from models.tasks import CIFAR_REAL_BIN_TASKS


def hamming_sym(a, b=None, binary=True):
    s = 1 - pairwise_distances(a, b, metric='hamming', n_jobs=-1)
    if binary:
        s[s < 0.5] = 1 - s[s < 0.5]
    return s


def tonp(x):
    if isinstance(x, (np.ndarray, float, int)):
        return np.array(x)
    return x.detach().cpu().numpy()


def viz_array_grid(array, rows, cols, padding=0, channels_last=False, normalize=False, **kwargs):
    # normalization
    '''
    Args:
        array: (N_images, N_channels, H, W) or (N_images, H, W, N_channels)
        rows, cols: rows and columns of the plot. rows * cols == array.shape[0]
        padding: padding between cells of plot
        channels_last: for Tensorflow = True, for PyTorch = False
        normalize: `False`, `mean_std`, or `min_max`
    Kwargs:
        if normalize == 'mean_std':
            mean: mean of the distribution. Default 0.5
            std: std of the distribution. Default 0.5
        if normalize == 'min_max':
            min: min of the distribution. Default array.min()
            max: max if the distribution. Default array.max()
    '''
    array = tonp(array)
    if not channels_last:
        array = np.transpose(array, (0, 2, 3, 1))

    array = array.astype('float32')

    if normalize:
        if normalize == 'mean_std':
            mean = kwargs.get('mean', 0.5)
            mean = np.array(mean).reshape((1, 1, 1, -1))
            std = kwargs.get('std', 0.5)
            std = np.array(std).reshape((1, 1, 1, -1))
            array = array * std + mean
        elif normalize == 'min_max':
            min_ = kwargs.get('min', array.min())
            min_ = np.array(min_).reshape((1, 1, 1, -1))
            max_ = kwargs.get('max', array.max())
            max_ = np.array(max_).reshape((1, 1, 1, -1))
            array -= min_
            array /= max_ + 1e-9

    batch_size, H, W, channels = array.shape
    assert rows * cols == batch_size

    if channels == 1:
        canvas = np.ones((H * rows + padding * (rows - 1),
                          W * cols + padding * (cols - 1)))
        array = array[:, :, :, 0]
    elif channels == 3:
        canvas = np.ones((H * rows + padding * (rows - 1),
                          W * cols + padding * (cols - 1),
                          3))
    else:
        raise TypeError('number of channels is either 1 of 3')

    for i in range(rows):
        for j in range(cols):
            img = array[i * cols + j]
            start_h = i * padding + i * H
            start_w = j * padding + j * W
            canvas[start_h: start_h + H, start_w: start_w + W] = img

    canvas = np.clip(canvas, 0, 1)
    canvas *= 255.0
    canvas = canvas.astype('uint8')
    return canvas


def _entropy_with_logits(logits):
    return Categorical(logits=logits).entropy()

def _mean_categorical_with_logits(logits):
    logp = torch.log_softmax(logits, dim=1)
    logp_mean = torch.logsumexp(logp - np.log(logits.shape[0]*1.), dim=0)
    return logp_mean

def _l2_reg(hparams, params):
    # l2 biased regularization
    return sum([((b - p) ** 2).sum() for b, p in zip(hparams, params)])


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    from https://github.com/PyTorchLightning/pytorch-lightning/issues/2534#issuecomment-674582085
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_train_batch_end(self, trainer: pl.Trainer, *_, force_save=False):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if self.save_step_frequency == -1: return

        if force_save or global_step % self.save_step_frequency == 0 or global_step == 1:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
            print(f'[Checkpoint] ===> Saved to {ckpt_path}')

    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        print('[Checkpoint] ===> Saving on interruption...')
        self.on_train_batch_end(trainer, force_save=True)


def rvs(dim=3, seed=None):
    """
    Return dim random perpendicular vectors in R^dim
    """
    if seed is None:
        random_state = np.random
    else:
        random_state = np.random.default_rng(seed)

    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim-n+1,))
        D[n-1] = np.sign(x[0])
        x[0] -= D[n-1]*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
    # Fix the last sign such that the determinant is 1
    D[-1] = (-1)**(1-(dim % 2))*D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D*H.T).T
    return H


def rot_by_alpha_deg(v, deg):
    theta = np.deg2rad(deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return v @ rot


def random_k_way_linear_task(K, d, seed):
    basis = rvs(d, seed)[:2]
    w = np.array([0, 1.])
    ws = np.stack([rot_by_alpha_deg(w, 360/K*i) for i in range(K)]).T
    return np.sum(basis[:, None] * ws[..., None], 0).T


def get_all_binary_tasks(classes):
    tasks = list(combinations(classes, len(classes)//2))
    return tasks

def get_main_tasks_idxs_from_included_classes(classes):
    classes = set(classes)
    reduced_tasks = list(map(set, get_all_binary_tasks(classes)))
    main_task_idxs = []
    _taken_tasks = []
    for i, cls1 in enumerate(CIFAR_REAL_BIN_TASKS):
        upd_cls1 = set(cls1).intersection(classes)
        if upd_cls1 in reduced_tasks and upd_cls1 not in _taken_tasks:
            main_task_idxs.append(i)
            _taken_tasks.append(upd_cls1)
    return main_task_idxs


def partialclass(cls, *args, **kwds):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls