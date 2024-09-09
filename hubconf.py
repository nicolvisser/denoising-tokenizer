dependencies = ["torch", "torchaudio", "xformers", "simple_parsing"]

import requests
import torch

from denoizr.tokenizer import TokenizerModel, TokenizerModelArgs


def tokenizer(
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
