"""Read prepared arrays. DataLoader workers keep independent lazy HDF5 handles."""
import h5py
import numpy as np
from torch.utils.data import Dataset


def inspect(path):
    with h5py.File(path, 'r') as f:
        if not f.attrs.get('complete', False):
            raise ValueError('Incomplete prepared data; fix preparation errors and rebuild')
        return str(f.attrs['dataset_id']), {key: f['geometry/' + key][:] for key in f['geometry']}


class Pairs(Dataset):
    def __init__(self, path, split):
        self.path = str(path)
        self.file = None
        with h5py.File(path, 'r') as f:
            self.rows = f['splits/' + split][:]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, 'r')
        row = int(self.rows[index])
        return (self.file['lr'][row].astype(np.float32),
                self.file['hr'][row].astype(np.float32))

    def __getstate__(self):
        state = self.__dict__.copy()
        state['file'] = None
        return state
