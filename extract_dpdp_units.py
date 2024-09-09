from pathlib import Path
from typing import List

import click
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from denoizr.dpdp.dataset import DPDPUnitsDataset


@click.command()
@click.option(
    "--features-dir",
    "-i",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to LibriSpeech features",
    required=True,
)
@click.option(
    "--units-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    help="Path to save extracted units",
    required=True,
)
@click.option(
    "--subsets",
    "-s",
    multiple=True,
    type=click.Choice(
        [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]
    ),
    help="Subsets to extract features from",
    required=True,
)
@click.option(
    "--codebook-path",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to numpy codebook of shape (num_units, num_features)",
    required=True,
)
@click.option(
    "--lmbda",
    "-l",
    type=float,
    default=0.0,
    help="Lambda parameter for DPDP",
)
@click.option(
    "--num-neighbors",
    "-n",
    type=int,
    default=1,
    help="Number of neighbors to consider in DPDP",
)
def extract_features(
    features_dir: Path,
    units_dir: Path,
    subsets: List[str],
    codebook_path: Path,
    lmbda: float,
    num_neighbors: int,
):
    units_dir.mkdir(parents=True, exist_ok=True)
    for subset in subsets:
        features_dataset_path = features_dir / f"{subset}.h5"
        units_dataset_path = units_dir / f"{subset}.h5"
        if units_dataset_path.exists():
            click.secho(
                f"Units for {subset} already extracted. Skipping...", fg="yellow"
            )
            continue

        click.secho(f"Extracting units for {subset}...", fg="green")

        dataset = DPDPUnitsDataset(
            features_dataset_path=features_dataset_path,
            codebook_path=codebook_path,
            lmbda=lmbda,
            num_neighbors=num_neighbors,
        )

        with h5py.File(units_dataset_path, "w") as f:
            for key, units in tqdm(dataset):
                f.create_dataset(key, data=units)


if __name__ == "__main__":
    extract_features()
