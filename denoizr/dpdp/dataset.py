from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .dpdp import dpdp_wfst


class DPDPUnitsDataset(Dataset):
    def __init__(
        self,
        features_dataset_path: Path,
        codebook_path: Path,
        lmbda: float,
        num_neighbors: int,
    ):
        assert Path(features_dataset_path).suffix == ".h5"
        self.lmbda = lmbda
        self.num_neighbors = num_neighbors
        self.codebook = torch.from_numpy(np.load(codebook_path))

        self.file = h5py.File(features_dataset_path, "r")
        self.refs = {}
        for key in self.file.keys():
            self.refs[key] = self.file[key]
        self.keys = sorted(list(self.refs.keys()))

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, idx):
        key = self.keys[idx]
        features = self.refs[key][:]
        features = torch.from_numpy(features).float()
        units = dpdp_wfst(
            features=features,
            codebook=self.codebook,
            lmbda=self.lmbda,
            num_neighbors=self.num_neighbors,
        )
        units = units.numpy()
        return key, units

    def close(self):
        if hasattr(self, "file"):
            self.file.close()

    def __del__(self):
        self.close()
