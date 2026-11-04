"""Geometric LR-to-HR baselines on the corrected v3 channel convention."""
import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np


def circle_intersection(d, a, b):
    d = np.asarray(d, dtype=np.float64)
    out = np.zeros_like(d)
    contained = d <= abs(a - b)
    out[contained] = math.pi * min(a, b) ** 2
    partial = (d < a + b) & ~contained
    x = d[partial]
    if len(x):
        aa = np.arccos(np.clip((x*x + a*a - b*b) / (2*x*a), -1, 1))
        bb = np.arccos(np.clip((x*x + b*b - a*a) / (2*x*b), -1, 1))
        rad = np.maximum(0, (-x+a+b)*(x+a-b)*(x-a+b)*(x+a+b))
        out[partial] = a*a*aa + b*b*bb - .5*np.sqrt(rad)
    return out


def method_matrix(method, lr_xy, hr_xy, excluded, lr_radius=32., hr_radius=12.5):
    excluded = np.asarray(excluded, dtype=int)
    if method == 'area':
        d = np.linalg.norm(hr_xy[:, None] - lr_xy[None], axis=2)
        matrix = circle_intersection(d, hr_radius, lr_radius) / (math.pi*lr_radius**2)
        matrix[:, excluded] = 0
        return matrix.astype('f4')
    k = 1 if method == 'nearest' else 4
    keep = np.ones(len(lr_xy), dtype=bool); keep[excluded] = False
    source = np.flatnonzero(keep)
    d = np.linalg.norm(hr_xy[:, None] - lr_xy[None, source], axis=2)
    near = np.argpartition(d, kth=k-1, axis=1)[:, :k]
    local = np.take_along_axis(d, near, axis=1)
    weight = np.ones_like(local) if k == 1 else 1/np.maximum(local, 1e-6)**2
    weight /= weight.sum(1, keepdims=True)
    matrix = np.zeros((len(hr_xy), len(lr_xy)), dtype='f4')
    matrix[np.arange(len(hr_xy))[:, None], source[near]] = weight
    matrix *= (hr_radius/lr_radius)**2
    return matrix


