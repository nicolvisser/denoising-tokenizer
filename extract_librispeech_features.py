from pathlib import Path
from typing import List

import click
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm


@click.command()
@click.option(
    "--librispeech-dir",
    "-i",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to LibriSpeech dataset root",
    required=True,
)
@click.option(
    "--features-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    help="Path to save extracted features",
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
    "--layer-idx",
    "-l",
    type=int,
    help="Layer index to extract features from",
    required=True,
)
def extract_features(
    librispeech_dir: Path,
    features_dir: Path,
    subsets: List[str],
    layer_idx: int,
):

    wavlm = torch.hub.load(
        "bshall/knn-vc", "wavlm_large", trust_repo=True, pretrained=True
    )
    wavlm.eval()

    for subset in subsets:
        features_dataset_path = features_dir / f"layer-{layer_idx}" / f"{subset}.h5"
        if features_dataset_path.exists():
            click.secho(
                f"Features for {subset} already extracted. Skipping...", fg="yellow"
            )
            continue

        click.secho(f"Extracting features for {subset}...", fg="green")

        dataset = LIBRISPEECH(root=librispeech_dir.parent, url=subset, download=False)

        loader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=collate_fn,
            num_workers=16,
            pin_memory=True,
        )

        with h5py.File(features_dataset_path, "w") as f:
            for key, waveform in tqdm(loader):
                waveform = F.pad(waveform, ((400 - 320) // 2, (400 - 320) // 2))
                with torch.no_grad():
                    features, _ = wavlm.extract_features(
                        source=waveform.cuda(),
                        padding_mask=None,
                        mask=False,
                        ret_conv=False,
                        output_layer=layer_idx,
                        ret_layer_results=False,
                    )
                features = features.squeeze(0).cpu().numpy()
                f.create_dataset(key, data=features)


def collate_fn(batch):
    waveform, _, _, speaker_id, chapter_id, utterance_id = batch[0]
    key = f"{speaker_id}-{chapter_id}-{utterance_id:04d}"
    return key, waveform


if __name__ == "__main__":
    extract_features()
