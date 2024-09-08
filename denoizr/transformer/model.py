import operator
from functools import reduce
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
from xformers.ops.fmha import memory_efficient_attention
from xformers.ops.fmha.attn_bias import AttentionBias, BlockDiagonalMask

from .config import TransformerModelArgs
from .rope import apply_rotary_emb, precompute_freqs_cis


def repeat_kv(
    keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys.contiguous(), values.contiguous()


class Attention(nn.Module):
    def __init__(self, args: TransformerModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.head_dim: int = args.head_dim
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: AttentionBias,
    ) -> torch.Tensor:
        seqlen_sum, _ = x.shape

        xq = self.wq(x).view(seqlen_sum, self.n_heads, self.args.head_dim)
        xk = self.wk(x).view(seqlen_sum, self.n_kv_heads, self.args.head_dim)
        xv = self.wv(x).view(seqlen_sum, self.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        key, val = xk, xv

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(xq, key, val, mask)

        return self.wo(output.view(seqlen_sum, -1))


class FeedForward(nn.Module):
    def __init__(self, args: TransformerModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: TransformerModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.dim = args.dim

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.attention = Attention(args)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.feed_forward = FeedForward(args=args)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, att_mask: AttentionBias
    ) -> torch.Tensor:
        r = self.attention(self.attention_norm(x), freqs_cis, att_mask)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out


def positions_from_sizes(sizes: Iterable[int], device):
    return torch.tensor(
        reduce(operator.iadd, [list(range(s)) for s in sizes], []),
        dtype=torch.long,
        device=device,
    )


class TransformerModel(nn.Module):
    def __init__(self, args: TransformerModelArgs):
        super().__init__()
        self.args = args
        self.n_layers = args.n_layers

        self.layers = torch.nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args=args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        # set lazily
        self._freqs_cis = None

        self.attn_mask_fn = BlockDiagonalMask

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freqs_cis(self):
        # lazy init
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis(
                self.args.head_dim, 128_000, self.args.rope_theta, self.device
            )
        return self._freqs_cis

    def forward(
        self,
        embeddings: torch.Tensor,
        q_seqlen: List[int],
    ) -> torch.Tensor:
        assert embeddings.dim() == 2, embeddings.shape
        assert sum(q_seqlen) == embeddings.shape[0], (
            sum(q_seqlen),
            embeddings.shape[0],
        )

        positions = positions_from_sizes(q_seqlen, self.freqs_cis.device)
        att_mask = self.attn_mask_fn.from_seqlens(q_seqlen=q_seqlen)
        freqs_cis = self.freqs_cis[positions].to(device=embeddings.device)
        for layer in self.layers:
            embeddings = layer(embeddings, freqs_cis, att_mask)
        return self.norm(embeddings)
