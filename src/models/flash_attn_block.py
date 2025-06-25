from .registry import register_block
import torch
import torch.nn as nn
from flash_attn.modules.mha import MHA

@register_block("flash_attn")
class FlashAttnBlock(nn.Module):
    def __init__(
        self,
        dim=1536,
        heads=8,
        dropout=0.15,
        rotary_emb_base=20000.0,
        rotary_emb_scale_base=None,
        aggregation="mean",
    ):
        super().__init__()
        self.aggregation = aggregation

        self.attn = MHA(
            use_flash_attn=True,
            embed_dim=dim,
            num_heads=heads,
            num_heads_kv=(heads // 2),
            qkv_proj_bias=True,
            out_proj_bias=True,
            dropout=dropout,
            softmax_scale=(dim / heads) ** -0.5,
            causal=False,
            rotary_emb_dim=128,
            rotary_emb_base=rotary_emb_base,
            rotary_emb_scale_base=rotary_emb_scale_base,
            fused_bias_fc=False,
        )

        # Init
        nn.init.kaiming_normal_(self.attn.Wqkv.weight, nonlinearity='relu')
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.ones_(self.attn.Wqkv.bias)

    def forward(self, x, lengths=None):
        out = self.attn(x)

        if lengths is not None:
            return self.representation(out, lengths)
        return out

    def representation(self, x: torch.Tensor, lengths: torch.Tensor):
        """Aggregate token embeddings into per-sample representations."""
        if self.aggregation == "mean":
            return mean_unpadded(x, lengths)
        elif self.aggregation == "last":
            return last_unpadded(x, lengths)
        else:
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")


def mean_unpadded(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
    masked = x * mask.unsqueeze(-1)
    summed = masked.sum(dim=1)
    return summed / lengths.unsqueeze(-1).float()


def last_unpadded(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    idx = lengths.long() - 1
    return x[torch.arange(x.size(0), device=x.device), idx]