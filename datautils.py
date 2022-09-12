from functools import partial
from io import DEFAULT_BUFFER_SIZE
import pandas as pd
from pytorch_lightning.utilities.cloud_io import load
from sklearn.preprocessing import StandardScaler
import torch
import argparse
import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule
from torchvision.datasets import STL10, CIFAR100
from typing import Any, Callable, Optional, Union, List
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import random
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import glob
from itertools import combinations
from torchvision.datasets import CIFAR10
from kornia.color.lab import RgbToLab
from torchvision import transforms
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

import utils


CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
REAL_TASKS = list(combinations(CIFAR_CLASSES, 5))[:126]


class Rgb2L(RgbToLab):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = super().forward(image)
        return x[:1]


class MyCIFAR10(CIFAR10):
    FACTORS_DF_PATH='./data/cifar-factors.csv'

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        include_classes: List[int] = None,
        return_indicies: bool = False,
        factors: List[str] = None,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)

        self.include_classes = include_classes
        self.return_indicies = return_indicies

        if self.include_classes is not None:
            include = np.array([t in self.include_classes for t in self.targets])
            self.data = self.data[include]
            self.targets = np.array(self.targets)[include]

        self.factors = None
        if factors is not None:
            self.factors = []
            factors_df = pd.read_csv(self.FACTORS_DF_PATH, index_col=0)
            for f in factors:
                if f == 'mean_color':
                    self.factors.append(self.data.mean((1, 2)))
                elif f == 'color_minmax_diff':
                    self.factors.append((self.data.max((1, 2, 3)) - self.data.min((1, 2, 3)))[..., None])
                else:
                    self.factors.append(torch.load(factors_df.loc[f][f'path_{"train" if train else "test"}']))

            self.factors = np.concatenate(self.factors, axis=1).astype(np.float32)

    def __getitem__(self, index):
        out = list(super().__getitem__(index))
        if self.return_indicies:
            out.append(index)

        if self.factors is not None:
            out.append(self.factors[index])

        return tuple(out)

    @property
    def factors_dim(self) -> int:
        return self.factors.shape[1] if self.factors is not None else 0

    def __repr__(self) -> str:
        return super().__repr__() + f'\nClasses: {self.include_classes}'


class MyCIFAR10DataModule(CIFAR10DataModule):
    dataset_cls = MyCIFAR10
    def __init__(
        self,
        data_dir: Optional[str] =  os.environ.get('DATA_ROOT', os.getcwd()),
        val_split: Union[int, float] = 0.1,
        num_workers: int = 16,
        normalize: bool = True,
        batch_size: int = 32,
        test_batch_size: Optional[int] = None,
        data_seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = True,
        drop_last: bool = True,
        random_labelling: bool = False,
        random_labelling_seed: Optional[int] = None,
        n_classes: int = 10,
        gt2class: Optional[str] = None,
        n_train_images: int = -1,
        multi_task: bool = "",
        path2pool: str = '',
        n_tasks: int = 1,
        persistent_workers: bool = False,
        return_indicies: bool = False,
        to_lightness: bool = False,
        include_classes: List[int] = None,
        augs: bool = False,
        factors: Optional[List[str]] = None,
        train_val_split: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=data_seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        self.EXTRA_ARGS['download'] = True
        self.dataset_cls = partial(MyCIFAR10, return_indicies=return_indicies, include_classes=include_classes, factors=factors)

        assert n_train_images == -1 or n_train_images >= batch_size or not drop_last
        self.test_batch_size = test_batch_size or self.batch_size
        self.n_train_images = n_train_images
        self.random_labelling_seed = random_labelling_seed if random_labelling_seed is not None else self.seed
        print(f'[Datamodule] ===> : Random_labelling={random_labelling}, Shuffle={shuffle}, Data_seed={data_seed}, Persistent_workers={persistent_workers}')
        self.random_labelling = random_labelling
        self._num_classes = n_classes
        # if not random_labelling:
        #     print(type(gt2class))
        #     assert self.num_classes == 10 or isinstance(gt2class, str)
        self._gt2class = None
        if isinstance(gt2class, str) and gt2class != '' and not self.random_labelling:
            self._gt2class = {gt: i for i, clss in enumerate(gt2class.split('|')) for gt in clss.split(',') }
            print(self._gt2class)

        self.persistent_workers = persistent_workers
        self.to_lightness = to_lightness
        if self.to_lightness:
            self.dims = (1, 32, 32)

        self.augs = augs
        self.train_transforms=self.get_transforms(train=True)
        self.val_transforms=self.get_transforms(train=False)
        self.test_transforms=self.get_transforms(train=False)

        self.train_val_split = train_val_split

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--data_seed', type=int, default=42)
        parser.add_argument('--random_labelling_seed', type=int, default=42)
        parser.add_argument('--n_train_images', type=int, default=-1)
        parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
        parser.add_argument('--shuffle', dest='shuffle', action='store_true')
        parser.set_defaults(shuffle=True)
        parser.add_argument('--multi_task', type=str, default='')
        # parser.add_argument('--n_tasks', type=int, default=1)
        parser.add_argument('--val_split', type=float, default=0.1)
        parser.add_argument('--gt2class', type=str, default="")
        parser.add_argument('--path2pool', type=str, default="")
        parser.add_argument('--random_labelling', action='store_true', default=False)
        parser.add_argument('--no_drop_last', dest='drop_last', action='store_false', default=True)
        parser.add_argument('--persistent_workers', action='store_true', default=False)
        parser.add_argument('--return_indicies', action='store_true', default=False)
        parser.add_argument('--to_lightness', action='store_true', default=False)
        parser.add_argument('--normalize', action='store_true', default=True)
        parser.add_argument('--no_normalize', dest='normalize', action='store_false', default=True)
        parser.add_argument('--include_classes', type=int, default=None, nargs='+')
        parser.add_argument('--augs', action='store_true', default=False)
        parser.add_argument('--no_augs', dest='augs', action='store_false', default=False)
        parser.add_argument('--factors', type=str, default=None, nargs='*')
        parser.add_argument('--train_val_split', type=str, default=None)
        parser.add_argument('--dataset_path', type=str, default='')
        return parser

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def get_transforms(self, train=False) -> Callable:
        t = [transforms.ToTensor()]

        if self.augs and train:
            t += [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]

        if self.to_lightness:
            t.append(Rgb2L())

        if self.normalize:
            if self.to_lightness: raise ValueError(f'{self.normalize=} and {self.to_lightness=}')
            t.append(cifar10_normalization())

        return transforms.Compose(t)
    
    @property
    def factors_dim(self) -> int:
        return self.dataset_train.dataset.factors_dim

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates train, val, and test dataset
        """

        # prepare all datasets
        super().setup()
        if self.dataset_train.dataset.factors is not None:
            scaler = StandardScaler()
            self.dataset_train.dataset.factors = scaler.fit_transform(self.dataset_train.dataset.factors)

            self.dataset_val.dataset.factors = scaler.transform(self.dataset_val.dataset.factors)
            self.dataset_test.factors = scaler.transform(self.dataset_test.factors)

        print(f'[Datamodule] ===> {self.dataset_train.indices[:20]=}')

        if self.random_labelling:
            g = torch.Generator().manual_seed(self.random_labelling_seed)
            self.dataset_train.dataset.targets = torch.randint(0, self.num_classes, (len(self.dataset_train.dataset),), generator=g).tolist()
            self.dataset_val.dataset.targets = torch.randint(0, self.num_classes, (len(self.dataset_val.dataset),), generator=g).tolist()
            self.dataset_test.targets = torch.randint(0, self.num_classes, (len(self.dataset_test),), generator=g).tolist()
        elif self._gt2class is not None:
            classes = self.dataset_train.dataset.classes
            self.dataset_train.dataset.targets = [self._gt2class[classes[t]] for t in self.dataset_train.dataset.targets]
            self.dataset_val.dataset.targets = [self._gt2class[classes[t]] for t in self.dataset_val.dataset.targets]
            self.dataset_test.targets = [self._gt2class[classes[t]] for t in self.dataset_test.targets]

    def _split_dataset(self, dataset: Dataset, train: bool = True) -> Dataset:
        """
        Splits the dataset into train and validation set
        """
        if self.train_val_split is None:
            len_dataset = len(dataset)  # type: ignore[arg-type]
            splits = self._get_splits(len_dataset)
            dataset_train, _, dataset_val = random_split(dataset, splits, generator=torch.Generator().manual_seed(self.seed))
        else:
            splits = torch.load(self.train_val_split)
            dataset_train, dataset_val = [Subset(dataset, indices) for indices in splits]

        if train:
            return dataset_train
        return dataset_val

    def _get_splits(self, len_dataset: int) -> List[int]:
        """
        Computes split lengths for train and validation set
        """
        if isinstance(self.val_split, int):
            val_len = self.val_split
        elif isinstance(self.val_split, float):
            val_len = int(self.val_split * len_dataset)
        else:
            raise ValueError(f'Unsupported type {type(self.val_split)}')
        
        if self.n_train_images == -1:
            train_len = len_dataset - val_len
        else:
            train_len = self.n_train_images

        splits = [train_len, len_dataset - train_len - val_len, val_len]
        print('train/_/val splits :', splits)

        return splits

    def _data_loader(
        self,
        dataset: torch.utils.data.Dataset,
        generator: Any = None,
        shuffle: bool = False,
        persistent_workers: bool = False,
        batch_size: int = None,
        drop_last: bool = None,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=self.num_workers,
            drop_last=self.drop_last if drop_last is None else drop_last,
            pin_memory=self.pin_memory,
            worker_init_fn=MyCIFAR10DataModule._worker_init_fn,
            persistent_workers=persistent_workers,
        )

    def train_dataloader(
        self,
        generator: Optional[torch.Generator] = None,
        persistent_workers: bool = False,
        batch_size: int = None,
        drop_last: bool = None,

    ) -> torch.utils.data.DataLoader:
        """ The train dataloader """
        persistent_workers = persistent_workers or self.persistent_workers
        return self._data_loader(self.dataset_train, shuffle=self.shuffle, generator=generator, persistent_workers=persistent_workers, batch_size=batch_size, drop_last=drop_last)

    def val_dataloader(self, persistent_workers: bool = False, batch_size: int = None) -> torch.utils.data.DataLoader:
        """ The val dataloader """
        persistent_workers = persistent_workers or self.persistent_workers
        batch_size = batch_size or self.test_batch_size
        return self._data_loader(self.dataset_val, persistent_workers=persistent_workers, batch_size=batch_size, drop_last=False)

    def test_dataloader(self, persistent_workers: bool = False, batch_size: int = None) -> torch.utils.data.DataLoader:
        """ The val dataloader """
        batch_size = batch_size or self.test_batch_size
        return self._data_loader(self.dataset_test, persistent_workers=persistent_workers, batch_size=batch_size, drop_last=False)

    @staticmethod
    def _worker_init_fn(_id):
        seed = torch.utils.data.get_worker_info().seed % 2**32
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)