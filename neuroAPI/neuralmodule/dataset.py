import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class GeologyDataset(Dataset):
    def __init__(self, file_path: str = None):
        if not file_path:
            raise NotImplementedError
        self.data = pd.read_csv(file_path,
                                usecols=['center.x', 'center.y', 'center.z', 'code.index'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx, :]
        return {'X': np.array([row['center.x'], row['center.y'], row['center.z']]).astype(np.float32),
                'Y': int(row['code.index'])}
