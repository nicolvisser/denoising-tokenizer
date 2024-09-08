from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from denoizr.dpdp import dpdp_wfst

from .data import TokenizerTrainingDatasetItem


class TokenizerDataset(Dataset):
    def __init__(
        self,
        features_data_paths: List[str],
        codebook_path: str,
        dpdp_lmbda: float,
        dpdp_num_neighbors: int,
        dedupe_tokens: bool,
    ):
        assert Path(codebook_path).exists(), f"Codebook {codebook_path} does not exist"
        assert Path(codebook_path).suffix == ".npy", "Codebook must be a .npy file"
        self.codebook = torch.from_numpy(np.load(codebook_path))
        assert (
            self.codebook.dim() == 2
        ), "Codebook must be a 2D array of shape (n_tokens, dim_features)"
        self.lmbda = dpdp_lmbda
        self.num_neighbors = dpdp_num_neighbors
        self.dedupe = dedupe_tokens

        # assert that data_paths is list of strings and not a single string
        assert isinstance(
            features_data_paths, list
        ), "Features paths must be a list of strings"

        # ensure that all data paths exist
        for p in features_data_paths:
            assert Path(p).exists(), f"Data path {p} does not exist"

        self.features_files = [
            h5py.File(data_path, "r") for data_path in features_data_paths
        ]

        # make a lookup table with a dataset reference for each key
        self.features_refs = {}
        for file in self.features_files:
            for key in sorted(list(file.keys())):
                self.features_refs[key] = file[key]

        # store keys
        self.keys = sorted(list(self.features_refs.keys()))
        print(f"Found {len(self.keys)} samples in dataset")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        features = self.features_refs[key][:]
        features = torch.from_numpy(features).float()
        tokens = dpdp_wfst(
            features=features,
            codebook=self.codebook,
            lmbda=self.lmbda,
            num_neighbors=self.num_neighbors,
        ).long()
        if self.dedupe:
            tokens = torch.cat([tokens[:1], tokens[1:][tokens[1:] != tokens[:-1]]])
        return TokenizerTrainingDatasetItem(
            features=features,
            tokens=tokens,
        )

    def close(self):
        # close all .h5 files
        if hasattr(self, "features_files"):
            for file in self.features_files:
                file.close()

    def __del__(self):
        self.close()
