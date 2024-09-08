from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class TokenizerTrainingDatasetItem:
    features: torch.Tensor  # (num_features, dim_features)
    tokens: torch.Tensor  # (num_tokens,)

    @property
    def num_features(self) -> int:
        return self.features.shape[0]

    @property
    def num_tokens(self) -> int:
        return self.tokens.shape[0]

    @property
    def dim_features(self) -> int:
        return self.features.shape[1]


def _make_decoder_types(
    features_lengths: List[int],
    tokens_lengths: List[int],
    device: Optional[torch.device] = None,
):
    """
    - A utility function to create a tensor of types for the decoder input and output.
    - This type can then be used to create masks for the input and output of the decoder.
    e.g.:
    -----------------------------------------------------------------------------------------------------
                                    |                item1                |             item2           |
    -----------------------------------------------------------------------------------------------------
    types                   logic   |  0   1   1   1   1   2   3   3   4  |  0   1   1   1   2   3   4  |
    ----------------------------------------------------------------------------------------------------|
                                    | CLS <f> <f> <f> <f> SEP <t> <t> <t> | CLS <f> <f> <f> SEP <t> <t> |
    input_CLS_token_mask    0       |  X                                  |  X                          |
    input_feature_mask      1       |      X   X   X   X                  |      X   X   X              |
    input_SEP_token_mask    2       |                      X              |                  X          |
    input_token_mask        3|4     |                          X   X   X  |                      X   X  |
    ----------------------------------------------------------------------------------------------------|
                                    |                     <t> <t> <t> END |                 <t> <t> END |
    output_token_mask       2|3     |                      X   X   X      |                  X   X      |
    output_END_token_mask   4       |                                  X  |                          X  |
    output_loss_mask        2|3|4   |                      X   X   X   X  |                  X   X   X  |
    ----------------------------------------------------------------------------------------------------|
    """
    types = []
    for f, t in zip(features_lengths, tokens_lengths):
        types.append(0)
        types.extend([1] * f)
        types.append(2)
        types.extend([3] * (t - 1))
        types.append(4)
    return torch.tensor(types, device=device, dtype=torch.int8)


@dataclass
class TokenizerTrainingInputBatch:
    features: torch.Tensor  # (total_num_features, dim_features)
    tokens: torch.Tensor  # (total_num_tokens,)
    features_lengths: List[int]  # (batch_size,)
    tokens_lengths: List[int]  # (batch_size,)
    decoder_types: torch.Tensor  # (total_num_features + total_num_tokens + batch_size,)

    @property
    def device(self) -> torch.device:
        return self.features.device

    @property
    def encoder_q_seqlen(self) -> List[int]:
        return self.features_lengths

    @property
    def decoder_q_seqlen(self) -> List[int]:
        return [f + t + 2 for f, t in zip(self.features_lengths, self.tokens_lengths)]

    @property
    def decoder_input_length(self) -> int:
        return self.features.shape[0] + self.tokens.shape[0] + self.batch_size * 2

    @property
    def decoder_input_CLS_token_mask(self) -> torch.Tensor:
        return self.decoder_types == 0

    @property
    def decoder_input_features_mask(self) -> torch.Tensor:
        return self.decoder_types == 1

    @property
    def decoder_input_SEP_token_mask(self) -> torch.Tensor:
        return self.decoder_types == 2

    @property
    def decoder_input_token_mask(self) -> torch.Tensor:
        return (self.decoder_types == 3) | (self.decoder_types == 4)

    @property
    def decoder_output_token_mask(self) -> torch.Tensor:
        return (self.decoder_types == 2) | (self.decoder_types == 3)

    @property
    def decoder_output_END_token_mask(self) -> torch.Tensor:
        return self.decoder_types == 4

    @property
    def decoder_output_loss_mask(self) -> torch.Tensor:
        return (
            (self.decoder_types == 2)
            | (self.decoder_types == 3)
            | (self.decoder_types == 4)
        )

    @property
    def total_num_features(self) -> int:
        return self.features.shape[0]

    @property
    def total_num_tokens(self) -> int:
        return self.tokens.shape[0]

    @property
    def dim_features(self) -> int:
        return self.features.shape[1]

    @property
    def batch_size(self) -> int:
        return len(self.features_lengths)

    def to(self, device) -> "TokenizerTrainingInputBatch":
        self.features = self.features.to(device)
        self.tokens = self.tokens.to(device)
        self.decoder_types = self.decoder_types.to(device)
        return self

    def __post_init__(self) -> None:
        assert self.features.ndim == 2
        assert self.tokens.ndim == 1
        assert sum(self.features_lengths) == self.features.shape[0]
        assert sum(self.tokens_lengths) == self.tokens.shape[0]
        assert len(self.features_lengths) == len(self.tokens_lengths)


def collate_fn(
    batch: List[TokenizerTrainingDatasetItem],
    device: Optional[torch.device] = None,
) -> TokenizerTrainingInputBatch:
    features = []
    tokens = []
    feature_lengths = []
    tokens_lengths = []
    for data in batch:
        features.append(data.features)
        tokens.append(data.tokens)
        feature_lengths.append(data.num_features)
        tokens_lengths.append(data.num_tokens)
    features = torch.cat(features, axis=0)
    tokens = torch.cat(tokens, axis=0)
    decoder_types = _make_decoder_types(feature_lengths, tokens_lengths, device=device)
    return TokenizerTrainingInputBatch(
        features=features,
        tokens=tokens,
        features_lengths=feature_lengths,
        tokens_lengths=tokens_lengths,
        decoder_types=decoder_types,
    )


@dataclass
class TokenizerTrainingOutputBatch:
    loss: torch.Tensor
    accuracy: torch.Tensor
