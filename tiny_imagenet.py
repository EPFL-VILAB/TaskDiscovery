import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pytorch_lightning as pl
from typing import Any, Optional, Union, List
import argparse
import random
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import glob
from tqdm import tqdm
from torchvision import datasets, transforms

class MyTinyImageNetTrainDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        return *super().__getitem__(index), index
    
class MyTinyImageNetValDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        return *super().__getitem__(index), index + 100000

class MyTinyImageNetTestDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        return *super().__getitem__(index), index + 100000 + 10000

class TinyImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: Optional[str] =  '',
        num_workers: int = 16,
        batch_size: int = 32,
        test_batch_size: Optional[int] = None,
        data_seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = True,
        drop_last: bool = True,
        task_type: str = 'real',
        random_labelling_seed: Optional[int] = None,
        n_classes: int = 2,
        persistent_workers: bool = False,
        return_indicies: bool = False,
        image_size: int = 32,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = data_seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.return_indicies = return_indicies
        self.pin_memory=pin_memory
        self.dims = (3, image_size, image_size)

        self.random_labelling_seed = random_labelling_seed if random_labelling_seed is not None else self.seed
        self.task_type = task_type
        print(f'[TinyImageNetDatamodule] ===> : Shuffle={shuffle}, Data_seed={data_seed}, Persistent_workers={persistent_workers}, Drop_last={drop_last}')
        self._num_classes = n_classes
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        
        dataset_train_cls = MyTinyImageNetTrainDataset if return_indicies else datasets.ImageFolder
        dataset_val_cls = MyTinyImageNetValDataset if return_indicies else datasets.ImageFolder
        dataset_test_cls = MyTinyImageNetTestDataset if return_indicies else datasets.ImageFolder
        
        self.dataset_train = dataset_train_cls(
            dataset_path + '/train',
            transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                normalize,
        ]))

        self.dataset_val = dataset_val_cls(
            dataset_path + '/val/images',
            transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                normalize,
        ]))

        self.dataset_test = dataset_test_cls(
            dataset_path + '/test',
            transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                normalize,
        ]))
        self.test_batch_size = test_batch_size or self.batch_size
        self.persistent_workers = persistent_workers

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--data_seed', type=int, default=42)
        parser.add_argument('--random_labelling_seed', type=int, default=42)
        parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
        parser.set_defaults(shuffle=True)
        pasers.add_argument('--n_classes', type=int, default=2)
        parser.add_argument('--no_drop_last', dest='drop_last', action='store_false', default=True)
        parser.add_argument('--return_indicies', action='store_true', default=False)
        parser.add_argument('--persistent_workers', action='store_true', default=False)
        parser.add_argument('--dataset_path', type=str, default='')
        return parser

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates train, val, and test dataset
        """

        # prepare all datasets
        super().setup()
        
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
            worker_init_fn=TinyImageNetDataModule._worker_init_fn,
            persistent_workers=persistent_workers,
        )

    def train_dataloader(
        self,
        generator: Optional[torch.Generator] = None,
        persistent_workers: bool = False,
        batch_size: int = None,

    ) -> torch.utils.data.DataLoader:
        """ The train dataloader """
        persistent_workers = persistent_workers or self.persistent_workers
        return self._data_loader(self.dataset_train, shuffle=self.shuffle, generator=generator, persistent_workers=persistent_workers, batch_size=batch_size)

    def val_dataloader(self, persistent_workers: bool = False, batch_size: int = None) -> torch.utils.data.DataLoader:
        """ The val dataloader """
        persistent_workers = persistent_workers or self.persistent_workers
        batch_size = batch_size or self.test_batch_size
        return self._data_loader(self.dataset_val, persistent_workers=persistent_workers, batch_size=batch_size, drop_last=False)
    
    def test_dataloader(self, persistent_workers: bool = False, batch_size: int = None) -> torch.utils.data.DataLoader:
        """ The train dataloader """
        batch_size = batch_size or self.test_batch_size
        return self._data_loader(self.dataset_val, persistent_workers=persistent_workers, batch_size=batch_size, drop_last=False)

    @staticmethod
    def _worker_init_fn(_id):
        seed = torch.utils.data.get_worker_info().seed % 2**32
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)