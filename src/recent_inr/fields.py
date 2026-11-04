"""Conditional adaptations; see REPRODUCTION.md for fidelity and deviations."""
import importlib
import math
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import ConditionalField


class TorchHashGrid(nn.Module):
    """2D Wisp-compatible bilinear hash lookup in PyTorch, no CUDA extension.

    Preserves upstream coordinate convention, hash, table sizes, and exclusive
    LOD masking, including its zero last level for lod_idx=num_lods-1.
    """
    @classmethod
    def from_geometric(cls, **kw):
        return cls(**kw)

    def __init__(self, feature_dim=2, num_lods=10, multiscale_type='cat',
                 feature_std=.01, feature_bias=0., codebook_bitwidth=10,
                 min_grid_res=8, max_grid_res=128, blas=None, coord_dim=2):
        super().__init__()
        assert coord_dim == 2 and multiscale_type == 'cat' and feature_bias == 0
        b = np.exp((np.log(max_grid_res)-np.log(min_grid_res))/(num_lods-1))
        self.resolutions = [int(np.floor(min_grid_res*b**i)) for i in range(num_lods)]
        self.active_lods = list(range(num_lods))
        self.feature_dim = feature_dim
        self.size = 2**codebook_bitwidth
        self.tables = nn.ParameterList([nn.Parameter(torch.randn(min(r*r, self.size), feature_dim)*feature_std)
                                       for r in self.resolutions])

    def interpolate(self, coords, lod_idx):
        outputs = []
        for level, (r, table) in enumerate(zip(self.resolutions, self.tables)):
            if level >= lod_idx:
                outputs.append(coords.new_zeros(*coords.shape[:-1], self.feature_dim))
                continue
            p = (r*(coords*.5+.5)).clamp(0., r-1-1e-5)
            base = p.floor().long()
            frac = p-base
            value = 0.
            for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1)):
                ix, iy = base[..., 0]+dx, base[..., 1]+dy
                index = ix+iy*r if r*r < self.size else (ix ^ (iy*2654435761)) % self.size
                weight = (frac[..., 0] if dx else 1-frac[..., 0])*(frac[..., 1] if dy else 1-frac[..., 1])
                value = value+table[index]*weight[..., None]
            outputs.append(value)
        return torch.cat(outputs, -1)


class RecentField(ConditionalField):
    def __init__(self, kind, width=128):
        # Reuse exactly the existing event encoder and PE count renderer.
        nn.Module.__init__(self)
        self.config = dict(model=kind, width=width, depth=3, latent=32, coord_scale=650.)
        self.encoder = nn.Sequential(nn.Linear(170, width), nn.ReLU(), nn.Linear(width, 32))
        if kind == 'sasnet':
            sys.path.insert(0, str(Path(__file__).parent/'vendor'/'SASNet_inr'))
            official = importlib.import_module('sasnet.models.sasnet')
            # Dependency injection only: official frequency/group-mask layers unchanged.
            official._import_hash_grid = lambda: TorchHashGrid
            self.net = official.SASNet(2, [width]*3, 2, 1, period=2/.95,
                omega_0=30., bandlimit=42, low_range=9, perc_low_freqs=.5,
                num_high_freq_bands=4, masked_hidden_layer_groups=[8, 8],
                use_masks=True, hash_grid_kwargs=dict(min_grid_res=8, max_grid_res=128))
            # Condition both the backbone and masks on the same event latent.
            layer = self.net.middle_layers[0]
            layer.linear = nn.Linear(width+32, width)
            nn.init.uniform_(layer.linear.weight, -math.sqrt(6/(width+32))/30,
                             math.sqrt(6/(width+32))/30)
            self.net.decoder.hidden = nn.Linear(20+32, 48)
            nn.init.constant_(self.net.output_layer.bias, math.log(math.expm1(1.)))
        else:
            raise ValueError(kind)

    def density(self, latent, query_mm):
        xy = query_mm/self.config['coord_scale']
        cond = latent[:, None, :].expand(-1, len(xy), -1)
        feat = self.net.grid.interpolate(xy, len(self.net.grid.active_lods)-1)
        feat = feat[None].expand(len(latent), -1, -1)
        masks = self.net.decoder(torch.cat([feat, cond], -1))
        chunks = torch.split(masks, self.net._mask_sizes, -1)
        coords = xy[None].expand(len(latent), -1, -1)
        h = self.net.input_layer(coords, input_masks=chunks[0])
        h = torch.cat([h, cond], -1)
        for layer, mask in zip(self.net.middle_layers, chunks[1:]):
            h = layer(h, hidden_masks=mask)
        return F.softplus(self.net.output_layer(h)).squeeze(-1)
