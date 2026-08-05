"""Vendored LongCLIP-B Vision Transformer used by VHM-B/16.

Architecture matches OpenAI CLIP ViT-B/16 / Beichuan LongCLIP visual tower.
Weights are loaded exclusively from ``VHM-B-16.pth`` — no external LongCLIP
package or ``longclip-B.pt`` bootstrap is required.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torch.nn as nn


class LayerNorm(nn.LayerNorm):
    """LayerNorm that preserves input dtype (fp16-safe)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor | None = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


def infer_vit_config_from_state(
    visual_state: Dict[str, torch.Tensor],
) -> Tuple[int, int, int, int, int, int]:
    """Return (input_resolution, patch_size, width, layers, heads, output_dim)."""
    if "conv1.weight" not in visual_state:
        raise KeyError("visual state missing conv1.weight")
    width = int(visual_state["conv1.weight"].shape[0])
    patch_size = int(visual_state["conv1.weight"].shape[-1])
    layers = len(
        [
            k
            for k in visual_state
            if k.endswith(".attn.in_proj_weight") and k.startswith("transformer.")
        ]
    )
    if layers <= 0:
        raise ValueError("Could not infer transformer depth from visual state")
    grid_size = int(round((visual_state["positional_embedding"].shape[0] - 1) ** 0.5))
    if grid_size * grid_size + 1 != int(visual_state["positional_embedding"].shape[0]):
        raise ValueError(
            "positional_embedding is not a square patch grid + CLS: "
            f"{tuple(visual_state['positional_embedding'].shape)}"
        )
    input_resolution = patch_size * grid_size
    output_dim = int(visual_state["proj"].shape[1])
    heads = width // 64
    return input_resolution, patch_size, width, layers, heads, output_dim


def build_vhm_visual_from_state(
    visual_state: Dict[str, torch.Tensor],
    *,
    strict: bool = True,
) -> VisionTransformer:
    """Construct a VisionTransformer and load ``visual.*`` weights (prefix stripped)."""
    state = {
        (k[len("visual.") :] if k.startswith("visual.") else k): v
        for k, v in visual_state.items()
    }
    (
        input_resolution,
        patch_size,
        width,
        layers,
        heads,
        output_dim,
    ) = infer_vit_config_from_state(state)
    visual = VisionTransformer(
        input_resolution=input_resolution,
        patch_size=patch_size,
        width=width,
        layers=layers,
        heads=heads,
        output_dim=output_dim,
    )
    missing, unexpected = visual.load_state_dict(state, strict=bool(strict))
    if missing or unexpected:
        # Caller may log; keep return usable for non-strict loads.
        pass
    return visual.eval()
