"""Finite-area disk quadrature shared by the training protocol."""
import math
import torch
from torch import nn


def disk_quadrature(centers, radius, samples=128):
    if radius <= 0 or samples < 16:
        raise ValueError('Positive radius and at least 16 samples required')
    k = torch.arange(samples, device=centers.device, dtype=centers.dtype) + .5
    r = radius * torch.sqrt(k / samples)
    angle = k * (math.pi * (3 - math.sqrt(5)))
    offsets = torch.stack((r * angle.cos(), r * angle.sin()), -1)
    points = centers[:, None] + offsets
    weights = centers.new_full((len(centers), samples), math.pi * radius**2 / samples)
    return points, weights


