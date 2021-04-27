from typing import Union

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

from neuroAPI.database.models import BorderPointType as _BPType


# credit to https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
class FastDataLoader(object):
    def __init__(self,
                 file_path: str = None,
                 batch_size: int = 64,
                 shuffle: bool = False,
                 borders: dict[str, dict[_BPType, float]] = None,
                 ):

        # Assertions
        if not file_path:
            raise NotImplementedError
        if borders:  # TODO: optimize assertions? move to self.normalize()?
            assert isinstance(borders, dict), TypeError(f'{type(borders)} passed. dict expected')
            assert len(borders) == 3, ValueError(f'{len(borders)} items in `borders` passed. 3 items expected')
            assert 'x' in borders, ValueError('No `x` in borders.')
            assert 'y' in borders, ValueError('No `y` in borders.')
            assert 'z' in borders, ValueError('No `z` in borders.')
            assert isinstance(borders['x'], dict), TypeError(f'{type(borders["x"])} passed. dict expected')
            assert isinstance(borders['y'], dict), TypeError(f'{type(borders["y"])} passed. dict expected')
            assert isinstance(borders['z'], dict), TypeError(f'{type(borders["z"])} passed. dict expected')
            assert all(k in borders['x'] for k in [_BPType.min, _BPType.max]), ValueError('no min-max border in `x`')
            assert all(k in borders['y'] for k in [_BPType.min, _BPType.max]), ValueError('no min-max border in `y`')
            assert all(k in borders['z'] for k in [_BPType.min, _BPType.max]), ValueError('no min-max border in `z`')
            assert borders['x'][_BPType.min] < borders['x'][_BPType.max], ValueError('`x.min` >= `x.max')
            assert borders['y'][_BPType.min] < borders['y'][_BPType.max], ValueError('`x.min` >= `x.max')
            assert borders['z'][_BPType.min] < borders['z'][_BPType.max], ValueError('`x.min` >= `x.max')
            # TODO: change other type-asserts in project from `type(obj) == cls` to `isinstance(obj, cls)`
        if batch_size:
            try:
                batch_size = int(batch_size)
            except ValueError:
                ValueError('`batch_size` is not int-able')

        # Load data from file
        self.raw_data = pd.read_csv(file_path,
                                    usecols=['center.x', 'center.y', 'center.z', 'code.index'])

        # Normalization
        # TODO: implement normalization as staticmethod
        X_df = self.raw_data[['center.x', 'center.y', 'center.z']]
        pd.options.mode.chained_assignment = None
        if not borders:
            for col in X_df.columns:
                X_df[col] = (X_df[col] - X_df[col].min()) / (X_df[col].max() - X_df[col].min())
        else:
            raise NotImplementedError
        pd.options.mode.chained_assignment = 'warn'

        self.tensors = (torch.from_numpy(X_df.to_numpy()).type(torch.FloatTensor),
                        torch.from_numpy(self.raw_data['code.index'].to_numpy()).type(torch.LongTensor))
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate batches
        n_batches, remainder = divmod(self.dataset_len, batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch
