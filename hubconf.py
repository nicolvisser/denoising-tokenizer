dependencies = ["torch", "torchaudio", "xformers", "simple_parsing"]

import requests
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from denoizr.tokenizer import TokenizerModel, TokenizerModelArgs


def _tokenizer(
    tag: str,
    progress: bool = True,
) -> TokenizerModel:
    response = requests.get(
        f"https://github.com/nicolvisser/denoising-tokenizer/releases/download/{tag}/model_config.json",
    )
    if response.status_code == 200:
        config = response.json()
    else:
        raise Exception(f"Failed to fetch JSON file: {response.status_code}")

    model_args = TokenizerModelArgs.from_dict(config)
    model = TokenizerModel(model_args)
    checkpoint = torch.hub.load_state_dict_from_url(
        f"https://github.com/nicolvisser/denoising-tokenizer/releases/download/{tag}/best.pth",
        progress=progress,
    )
    model.load_state_dict(checkpoint)
    return model


def tokenizer(tag=str, progress: bool = True) -> TokenizerModel:
    return torch.hub.load(
        f"nicolvisser/denoising-tokenizer:{tag}",
        "_tokenizer",
        tag=tag,
        trust_repo=True,
    )
