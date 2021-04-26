from typing import Union

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from neuroAPI.database.models import BorderPointType as _BPType


class GeologyDataset(Dataset):
    def __init__(self, file_path: str = None, borders: dict[str, dict[_BPType, float]] = None):
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

        self.raw_data = pd.read_csv(file_path,
                                    usecols=['center.x', 'center.y', 'center.z', 'code.index'])

        self.data = pd.DataFrame()
        self.data['X_1'] = self.raw_data['center.x']
        self.data['X_2'] = self.raw_data['center.y']
        self.data['X_3'] = self.raw_data['center.x']
        self.data['Y'] = self.raw_data['code.index']

        self.normalize(borders=borders)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx, :]
        return {'X': np.array([row['X_1'], row['X_2'], row['X_3']]).astype(np.float32),
                'Y': int(row['Y'])}

    def normalize(self, borders: Union[None, dict[str, dict[_BPType, float]]]):
        if not borders:
            gen = (col for col in self.data.columns if col.startswith('X'))
            for col in gen:
                self.data[col] = (self.data[col] - self.data[col].min()) / (self.data[col].max() - self.data[col].min())
