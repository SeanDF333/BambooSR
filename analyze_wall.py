"""Measure the inward width of cross-calibrated 24-sector wall residuals."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
import yaml


def geometry(xy, sides, apothem):
    radius = np.linalg.norm(xy, axis=1)
    phi = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2 * np.pi)
    sector = np.floor(phi / (2 * np.pi / sides)).astype(int).clip(0, sides - 1)
    delta = np.mod(phi, 2 * np.pi / sides) - np.pi / sides
    return radius, sector, apothem / np.cos(delta)


def calibrate(xy, fold, sides, apothem):
    radius, sector, wall = geometry(xy, sides, apothem)
    ratio = radius / wall
    factors = np.full((2, sides), np.nan)
    residual = np.empty(len(radius))
    for held in range(2):
        for k in range(sides):
            cal = (fold != held) & (sector == k)
            test = (fold == held) & (sector == k)
            if cal.sum() < 10 or np.median(ratio[cal]) <= 0:
                raise ValueError('Insufficient sector calibration')
            factors[held, k] = 1 / np.median(ratio[cal])
            residual[test] = radius[test] * factors[held, k] - wall[test]
    return residual, factors


def cdf(x, mu, left, right):
    return np.where(x < mu, 2 * left / (left + right) * ndtr((x - mu) / left),
                    left / (left + right) + 2 * right / (left + right) * (ndtr((x - mu) / right) - .5))


def fit_width(residual, config):
    step = config['fit_bin_mm']
    edges = np.arange(config['fit_min_mm'], config['fit_max_mm'] + step, step)
    counts, _ = np.histogram(residual, edges)

    def objective(parameters):
        probability = np.diff(cdf(edges, parameters[0], np.exp(parameters[1]), np.exp(parameters[2])))
        probability /= probability.sum()
        return -np.dot(counts, np.log(np.maximum(probability, 1e-300)))

    fits = [minimize(objective, [0, np.log(s), np.log(s)], method='L-BFGS-B',
                     bounds=[(-25, 25), (np.log(.3), np.log(200)), (np.log(.3), np.log(200))])
            for s in (3., 10., 30.)]
    best = min(fits, key=lambda item: item.fun)
    mu = best.x[0]
    left, right = np.exp(best.x[1:])
    valid = bool(best.success and abs(mu) < 24.99 and min(left, right) > .301 and max(left, right) < 199.9)
    return {'fit_valid': valid, 'mu_mm': float(mu), 'sigma_left_mm': float(left),
            'sigma_right_mm': float(right), 'fit_events': int(counts.sum()),
            'inward_tail_beyond_20mm': float(np.mean(residual < -20))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--wall-data', default='data/bamboosr_wall_run2_5583_v1.npz')
    parser.add_argument('--output')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    with np.load(args.wall_data, allow_pickle=False) as wall, np.load(args.input, allow_pickle=False) as prediction:
        if not np.array_equal(wall['event_index'], prediction['event_index']):
            raise ValueError('Wall event order mismatch')
        if not np.array_equal(wall['lr_pe'], prediction['lr_pe']):
            raise ValueError('Wall charges mismatch')
        lr = wall['lr_pe'].astype('f8')
        xy = {'lr': lr @ prediction['lr_centers'] / lr.sum(1)[:, None],
              str(prediction['method']): prediction['cog_xy']}
    rng = np.random.default_rng(config['seed'])
    order = rng.permutation(len(lr))
    fold = np.empty(len(lr), dtype=int)
    fold[order] = np.arange(len(lr)) % 2
    result = {'events': len(lr), 'measure': 'fitted split-normal inward width, sigma_L',
              'methods': {}}
    for name, positions in xy.items():
        radius, _, wall_radius = geometry(positions, config['polygon_sides'], config['apothem_mm'])
        residual, factors = calibrate(positions, fold, config['polygon_sides'], config['apothem_mm'])
        fit = fit_width(residual, config)
        fit['valid_wall_reconstruction'] = bool(np.median(radius / wall_radius) > .5 and fit['fit_valid'])
        fit['raw_bias_mm'] = float(np.mean(radius - wall_radius))
        fit['scale_min'] = float(factors.min())
        fit['scale_max'] = float(factors.max())
        result['methods'][name] = fit
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
