import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class GeologyDataset(Dataset):
    def __init__(self, file_path: str = None):
        if not file_path:
            raise NotImplementedError
        self.raw_data = pd.read_csv(file_path,
                                    usecols=['center.x', 'center.y', 'center.z', 'code.index'])

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        row = self.raw_data.iloc[idx, :]
        return {'X': np.array([row['center.x'], row['center.y'], row['center.z']]).astype(np.float32),
                'Y': int(row['code.index'])}

    def get_all(self):  # TODO: move call to __init__
        df = pd.DataFrame()
        df['X'] = self.raw_data[['center.x', 'center.y', 'center.z']].values.tolist()
        df['Y'] = self.raw_data['code.index']
        return df
