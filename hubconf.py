dependencies = ["torch", "torchaudio", "xformers", "simple_parsing"]

import requests
import torch

from denoizr.tokenizer import TokenizerModel, TokenizerModelArgs

CONFIG_FILES = {
    "vivid-river-boost": "model_config-9a8dfcf9.json",
    "flash-river-speak": "model_config-9a8dfcf9.json",
}

CHECKPOINT_FILES = {
    "vivid-river-boost": "best-0d896416.pth",
    "flash-river-speak": "best-df8f58e4.pth",
}


def tokenizer(
    tag: str,
    progress: bool = True,
) -> TokenizerModel:
    response = requests.get(
        f"https://github.com/nicolvisser/denoising-tokenizer/releases/download/{tag}/{CONFIG_FILES[tag]}.json",
    )
    config = response.json()
    model_args = TokenizerModelArgs.from_dict(config)
    model = TokenizerModel(model_args)
    checkpoint = torch.hub.load_state_dict_from_url(
        f"https://github.com/nicolvisser/denoising-tokenizer/releases/download/{tag}/{CHECKPOINT_FILES[tag]}.pth",
        progress=progress,
    )
    model.load_state_dict(checkpoint)
    return model
