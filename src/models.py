"""Conditional SIREN, Fourier-feature MLP, and WIRE with one count renderer."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SineLayer(nn.Module):
    def __init__(self, inputs, outputs, omega, first=False):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs)
        self.omega = omega
        bound = 1 / inputs if first else math.sqrt(6 / inputs) / omega
        nn.init.uniform_(self.linear.weight, -bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class ComplexGaborLayer(nn.Module):
    """Complex Gabor activation from the official WIRE implementation."""
    def __init__(self, inputs, outputs, omega=30., scale=10., first=False):
        super().__init__()
        self.omega = omega
        self.scale = scale
        self.linear = nn.Linear(inputs, outputs,
                                dtype=torch.float if first else torch.cfloat)

    def forward(self, x):
        z = self.linear(x)
        return torch.exp(1j * self.omega * z - (self.scale * z).abs().square())


class WireDecoder(nn.Module):
    def __init__(self, inputs, width, depth, omega, scale):
        super().__init__()
        # A complex parameter stores two real values, so follow WIRE and shrink
        # hidden width by sqrt(2) for a fair parameter budget.
        hidden = max(1, int(width / math.sqrt(2)))
        layers = [ComplexGaborLayer(inputs, hidden, omega, scale, first=True)]
        layers += [ComplexGaborLayer(hidden, hidden, omega, scale)
                   for _ in range(depth - 1)]
        layers.append(nn.Linear(hidden, 1, dtype=torch.cfloat))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).real


class ConditionalField(nn.Module):
    def __init__(self, model='siren', width=128, depth=3, latent=32,
                 features=64, sigma=3., omega=30., wire_scale=10., coord_scale=650.):
        super().__init__()
        self.config = dict(model=model, width=width, depth=depth, latent=latent,
                           features=features, sigma=sigma, omega=omega,
                           wire_scale=wire_scale, coord_scale=coord_scale)
        if model not in ('siren', 'ffn', 'wire') or min(width, depth, latent, features) < 1:
            raise ValueError('Invalid architecture')
        if min(sigma, omega, wire_scale, coord_scale) <= 0:
            raise ValueError('Scales must be positive')
        self.encoder = nn.Sequential(nn.Linear(170, width), nn.ReLU(), nn.Linear(width, latent))
        if model == 'ffn':
            self.register_buffer('fourier', torch.randn(2, features) * sigma)
        inputs = (2 if model == 'siren' else 2 * features) + latent
        if model == 'wire':
            inputs = 2 + latent
            self.decoder = WireDecoder(inputs, width, depth, omega, wire_scale)
            return
        layers = []
        for i in range(depth):
            if model == 'siren':
                layers.append(SineLayer(inputs if i == 0 else width, width, omega, first=i == 0))
            else:
                layers.extend([nn.Linear(inputs if i == 0 else width, width), nn.ReLU()])
        final = nn.Linear(width, 1)
        if model == 'siren':
            nn.init.uniform_(final.weight, -math.sqrt(6 / width) / omega, math.sqrt(6 / width) / omega)
        nn.init.constant_(final.bias, math.log(math.expm1(1.)))
        layers.append(final)
        self.decoder = nn.Sequential(*layers)

    def encode(self, pe):
        total = pe.sum(-1, keepdim=True)
        condition = torch.cat([pe / total.clamp_min(1e-8), torch.log1p(total) / 10.], dim=-1)
        return self.encoder(condition), total

    def density(self, latent, query_mm):
        coords = query_mm / self.config['coord_scale']
        if self.config['model'] == 'ffn':
            phase = 2 * math.pi * coords @ self.fourier
            coords = torch.cat([phase.sin(), phase.cos()], dim=-1)
        coords = coords.unsqueeze(0).expand(len(latent), -1, -1)
        cond = latent[:, None, :].expand(-1, coords.shape[1], -1)
        return F.softplus(self.decoder(torch.cat([coords, cond], dim=-1))).squeeze(-1)

    def counts(self, pe, query_mm, weights):
        latent, total = self.encode(pe)
        channels, samples, _ = query_mm.shape
        density = self.density(latent, query_mm.reshape(-1, 2))
        return (density.reshape(len(pe), channels, samples) * weights).sum(-1) * total


def count_loss(pred, target):
    # Constant log(target!) omitted. Zeros are included. No BCE or log-count MSE.
    return (pred - target * torch.log(pred.clamp_min(1e-8))).mean()
