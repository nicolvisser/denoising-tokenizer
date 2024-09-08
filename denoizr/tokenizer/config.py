from dataclasses import dataclass
from typing import List, Optional, Tuple

from simple_parsing import Serializable

from denoizr.transformer import TransformerModelArgs


@dataclass
class TokenizerModelArgs(Serializable):
    codebook_size: int
    encoder: TransformerModelArgs
    decoder: TransformerModelArgs

    @property
    def dim(self):
        return self.encoder.dim

    def __post_init__(self):
        assert self.encoder.dim == self.decoder.dim
        assert self.codebook_size > 0


@dataclass
class TokenizerTrainingArgs(Serializable):
    train_features_data_paths: List[str]
    valid_features_data_paths: List[str]
    codebook_path: str
    dpdp_lmbda: float
    dpdp_num_neighbors: int
    dedupe_tokens: bool
    betas: Tuple[float, float]
    weight_decay: float
    lr_max: float
    lr_final: float
    warmup_steps: int
    decay_steps: int
    batch_size: int
    num_workers: int
    precision: str
    accumulate_grad_batches: int
    gradient_clip_val: float
    log_every_n_steps: int
    val_check_interval: float
    save_last: bool
    save_last_weights_only: bool
    save_best: bool
    save_best_top_k: int
    save_best_weights_only: bool
    early_stopping: bool
    early_stopping_patience: Optional[int]
    max_epochs: Optional[int]
    max_steps: Optional[int]
