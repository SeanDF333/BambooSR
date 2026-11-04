"""FFN backbone with a zero-initialized optical feature adapter.

Optical responses are phenomenological features, not calibrated PAFs.
The backbone is exactly ConditionalField('ffn'); no truth xy is an input.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models import ConditionalField


class BambooSRV3(ConditionalField):
    def __init__(self, lr_centers, mask_indices=(2,), width=128, latent=32,
                 features=64, sigma=3., coord_scale=650., variant='full',
                 wall_radius=592.5, adapter_width=32):
        super().__init__('ffn', width=width, depth=3, latent=latent,
                         features=features, sigma=sigma, coord_scale=coord_scale)
        if variant not in ('full', 'direct', 'neural_control', 'ffn'):
            raise ValueError('Unknown v3 variant')
        centers = torch.as_tensor(lr_centers, dtype=torch.float32)
        if centers.shape != (169, 2):
            raise ValueError('Require audited 169-channel geometry')
        self.register_buffer('centers', centers)
        valid = torch.ones(169, dtype=torch.bool)
        valid[list(mask_indices)] = False
        self.register_buffer('valid', valid)
        self.config.update(variant=variant, wall_radius=wall_radius,
                           adapter_width=adapter_width)
        # Preserve backbone initialization AND the downstream shuffle RNG.
        # A separate stream isolates extra parameters from the baseline stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed()+3000)
            if variant != 'ffn':
                self.feature_head = nn.Linear(latent, 6)
                nn.init.zeros_(self.feature_head.weight)
                nn.init.zeros_(self.feature_head.bias)
                if variant != 'neural_control':
                    with torch.no_grad():
                        self.feature_head.bias[2] = math.log(math.expm1(.4))
                        self.feature_head.bias[3] = math.log(math.expm1(1.9))
                        self.feature_head.bias[4] = math.log(math.expm1(2.2))
                        self.feature_head.bias[5] = -2.2
                self.adapter = nn.Sequential(nn.Linear(latent+6, adapter_width),
                    nn.SiLU(), nn.Linear(adapter_width, width))
                nn.init.zeros_(self.adapter[-1].weight)
                nn.init.zeros_(self.adapter[-1].bias)
        angle = torch.arange(24)*2*math.pi/24
        self.register_buffer('boundary', wall_radius*torch.stack((angle.cos(), angle.sin()), -1))

    def encode(self, pe):
        return super().encode(pe*self.valid)

    def optical_context(self, pe, latent):
        observed = pe*self.valid
        fractions = observed/observed.sum(-1, keepdim=True).clamp_min(1e-8)
        p = self.feature_head(latent)
        # Effective source only. Do not impose an unverified PTFE radius cap.
        source = fractions@self.centers+120*p[:, :2].tanh()
        return dict(source=source, height=10+100*F.softplus(p[:, 2:3]),
                    mix=10+100*F.softplus(p[:, 3:4]),
                    spread=10+100*F.softplus(p[:, 4:5]),
                    diffuse=p[:, 5:6].sigmoid())

    def physics_features(self, ctx, query):
        scale = self.config['coord_scale']
        delta = query[None]-ctx['source'][:, None]
        d2 = delta.square().sum(-1)
        height = ctx['height']
        direct = height/(2*math.pi*(d2+height.square()).pow(1.5))
        reflected = torch.zeros_like(direct)
        if self.config['variant'] == 'full':
            distance = (self.boundary[None]-ctx['source'][:, None]).square().sum(-1)
            mixing = torch.softmax(-distance/(2*ctx['mix'].square()), -1)
            bd2 = (query[:, None]-self.boundary[None]).square().sum(-1)
            variance = ctx['spread'].square()
            kernels = torch.exp(-bd2[None]/(2*variance[:, :, None]))/(2*math.pi*variance[:, :, None])
            local = (kernels*mixing[:, None]).sum(-1)
            reflected = (1-ctx['diffuse'])*local+ctx['diffuse']/(math.pi*self.config['wall_radius']**2)
        # Dimensionless, gently compressed optical responses. Six features.
        direct_log = torch.log1p(direct*scale**2)
        reflect_log = torch.log1p(reflected*scale**2)
        query_gap = (self.config['wall_radius']-query.norm(dim=-1))/scale
        source_gap = (self.config['wall_radius']-ctx['source'].norm(dim=-1))/scale
        return torch.cat((delta/scale, direct_log[..., None], reflect_log[..., None],
            query_gap[None, :, None].expand(len(delta), -1, -1),
            source_gap[:, None, None].expand(-1, len(query), -1)), -1)

    def density(self, latent, query_mm, context=None):
        phase = 2*math.pi*(query_mm/self.config['coord_scale'])@self.fourier
        coords = torch.cat((phase.sin(), phase.cos()), -1)
        coords = coords[None].expand(len(latent), -1, -1)
        z = latent[:, None].expand(-1, len(query_mm), -1)
        hidden = self.decoder[0](torch.cat((coords, z), -1))
        variant = self.config['variant']
        if variant != 'ffn':
            if variant == 'neural_control':
                # Six learned features, identical active head/adapter sizes.
                extra = torch.tanh(coords[..., :6]+self.feature_head(latent)[:, None])
            else:
                if context is None:
                    raise ValueError('Optical density requires LR-derived context')
                extra = self.physics_features(context, query_mm)
            hidden = hidden+self.adapter(torch.cat((z, extra), -1))
        return F.softplus(self.decoder[1:](hidden)).squeeze(-1)

    def counts(self, pe, query_mm, weights):
        latent, total = self.encode(pe)
        context = self.optical_context(pe, latent) if self.config['variant'] in ('full', 'direct') else None
        channels, samples, _ = query_mm.shape
        density = self.density(latent, query_mm.reshape(-1, 2), context)
        return (density.reshape(len(pe), channels, samples)*weights).sum(-1)*total

    def field(self, pe, query_mm):
        """Physical density PE/mm^2, for continuous-field visualization."""
        latent, total = self.encode(pe)
        ctx = self.optical_context(pe, latent) if self.config['variant'] in ('full', 'direct') else None
        return self.density(latent, query_mm, ctx)*total/(169*math.pi*32**2)
