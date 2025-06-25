from functools import partial
import math
from .registry import register_block
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba, Block

def create_mamba_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mix_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm, eps=norm_epsilon, **factory_kwargs)
    block = Block(
        d_model,
        mix_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def mean_unpadded(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
    masked = x * mask.unsqueeze(-1)
    summed = masked.sum(dim=1)
    return summed / lengths.unsqueeze(-1).float()


@register_block("mamba")
class MambaBlock(nn.Module):
    """
    RiboNN-compatible encoder block using Mamba layers from Orthrus.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_layer: int = 6,
        ssm_cfg=None,
        aggregation: str = "mean",
        norm_epsilon: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.aggregation = aggregation
        self.length_dummy = None  # used for representation()

        self.embedding = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            create_mamba_block(
                d_model=d_model,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                layer_idx=i,
            )
            for i in range(n_layer)
        ])
        self.norm = nn.LayerNorm(d_model, eps=norm_epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) → transpose to (B, L, C)
        x = x.transpose(1, 2)
        lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        self.length_dummy = lengths

        h = self.embedding(x)
        residual = None
        for layer in self.layers:
            h, residual = layer(h, residual)

        h = (h + residual) if residual is not None else h
        h = self.norm(h)

        return self.aggregate(h, lengths)

    def aggregate(self, h, lengths):
        if self.aggregation == "mean":
            return mean_unpadded(h, lengths)
        elif self.aggregation == "last":
            idxs = lengths - 1
            return h[torch.arange(h.size(0), device=h.device), idxs]
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation}")

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, L) input, for compatibility with Saluki
        return self.forward(x)