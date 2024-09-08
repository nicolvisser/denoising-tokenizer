from dataclasses import dataclass

from simple_parsing import Serializable


@dataclass
class TransformerModelArgs(Serializable):
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    rope_theta: float = 1_000_000
