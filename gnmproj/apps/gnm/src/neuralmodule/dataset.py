import pandas as pd
import numpy as np
import torch

from ...models import DepositBorders


# credit to https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
class FastDataLoader(object):  # TODO: refactor after init from database
    def __init__(self,
                 dataframe: pd.DataFrame = None,
                 file_path: str = None,
                 batch_size: int = 64,
                 shuffle: bool = False,
                 borders: dict[str, dict[DepositBorders.PointTypeEnum, float]] = None
                 ):

        assert (dataframe is not None) != bool(file_path)

        if batch_size:
            try:
                batch_size = int(batch_size)
            except ValueError:
                ValueError('`batch_size` is not int-able')

        if not file_path:
            self.raw_data = dataframe
            self.raw_data.columns = ['id', 'x', 'y', 'z', 'index']

        else:
            self.raw_data = pd.read_csv(file_path,
                                        usecols=['id', 'x', 'y', 'z', 'index'])

        prep_data = self.raw_data.rename(columns={self.raw_data.columns[0]: 'id',
                                                  self.raw_data.columns[1]: 'X_x',
                                                  self.raw_data.columns[2]: 'X_y',
                                                  self.raw_data.columns[3]: 'X_z',
                                                  self.raw_data.columns[4]: 'Y'})
        self.input_columns = ['X_x', 'X_y', 'X_z']

        if not borders:
            self.borders = {_cn: {DepositBorders.PointTypeEnum.MIN: prep_data[_cn].min(),
                                  DepositBorders.PointTypeEnum.MAX: prep_data[_cn].max()} for _cn in self.input_columns}
        else:
            self.borders = borders

        prep_data = self.preprocess(prep_data, input_columns=self.input_columns, borders=self.borders)

        self.data = (torch.from_numpy(prep_data[self.input_columns].to_numpy().astype(np.float64)).type(torch.FloatTensor),
                     torch.from_numpy(prep_data['Y'].to_numpy().astype(np.int8)).type(torch.LongTensor))
        self.dataset_len = self.data[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate batches
        n_batches, remainder = divmod(self.dataset_len, batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def denormalize(self, df: pd.DataFrame):
        pd.options.mode.chained_assignment = None

        for _cn in df.columns:
            cn = self.input_columns[_cn]
            _min = np.float64(self.borders[cn][DepositBorders.PointTypeEnum.MIN])
            _max = np.float64(self.borders[cn][DepositBorders.PointTypeEnum.MAX])
            df[_cn] = df[_cn] * (_max - _min) + _min

        pd.options.mode.chained_assignment = 'warn'

        return df

    @classmethod
    def preprocess(cls,
                   df: pd.DataFrame,
                   input_columns: list[str],
                   borders: dict[str, dict[DepositBorders.PointTypeEnum, float]] = None):
        data = cls._normalize(df, input_columns, borders)
        data = cls._undublicate(data, input_columns)
        return data

    @classmethod
    def _normalize(cls,
                   df: pd.DataFrame,
                   input_columns: list[str],
                   borders: dict[str, dict[DepositBorders.PointTypeEnum, float]] = None) -> pd.DataFrame:

        if not input_columns:
            input_columns = list(df.columns)

        # TODO: implement partial borders or partial border types pass
        assert isinstance(borders, dict), TypeError(f'{type(borders)} passed. dict expected')
        assert len(borders) == len(input_columns), \
            ValueError(f'{len(borders)} items in `borders` passed. {len(input_columns)} items expected')
        for _cn in input_columns:
            assert _cn in borders, ValueError(f'No {_cn} in borders.')
            assert isinstance(borders[_cn], dict), TypeError(f'{type(borders[_cn])} passed. dict expected')
            assert all(k in borders[_cn] for k in [
                DepositBorders.PointTypeEnum.MIN,
                DepositBorders.PointTypeEnum.MAX
            ]), ValueError(f'no min-max borders in {_cn}')
            assert borders[_cn][DepositBorders.PointTypeEnum.MIN] < borders[_cn][DepositBorders.PointTypeEnum.MAX],\
                ValueError(f'`{_cn}.min` >= `{_cn}.max')
        # TODO: change other type-asserts in project from `type(obj) == cls` to `isinstance(obj, cls)`

        pd.options.mode.chained_assignment = None

        for _cn in input_columns:
            _min = borders[_cn][DepositBorders.PointTypeEnum.MIN]
            _max = borders[_cn][DepositBorders.PointTypeEnum.MAX]
            df[_cn] = (df[_cn] - _min) / (_max - _min)

        pd.options.mode.chained_assignment = 'warn'

        return df

    @classmethod
    def _undublicate(cls, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if not columns:
            columns = list(df.columns)

        return df.drop_duplicates(subset=columns, ignore_index=True)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.data = [t[r] for t in self.data]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.data)
        self.i += self.batch_size
        return batch
