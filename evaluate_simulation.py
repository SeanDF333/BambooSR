"""Evaluate a validation-selected checkpoint on the fixed simulation test split."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from release_core import load_checkpoint, predict


def take(dataset, rows):
    order = np.argsort(rows)
    return dataset[rows[order]][np.argsort(order)]


def metrics(pred, truth, source_xy, geometry):
    y = truth.astype('f8')
    mu = pred.astype('f8')
    error = mu - y
    eps = 1e-8
    total_true = y.sum(1)
    total_pred = mu.sum(1)
    positive = total_true > 0
    edge = np.linalg.norm(source_xy, axis=1) >= 520
    pmt3 = geometry['lr_centers'][int(geometry['lr_mask_indices'][0])]
    region = np.linalg.norm(geometry['hr_centers'] - pmt3, axis=1) <= 32
    deviance = 2 * (mu - y + y * (np.log(np.maximum(y, eps)) - np.log(np.maximum(mu, eps))))
    shape_tv = .5 * np.abs(mu / np.maximum(total_pred[:, None], eps) -
                              y / np.maximum(total_true[:, None], eps)).sum(1)
    return {
        'events': len(y), 'rmse': float(np.sqrt(np.mean(error**2))),
        'mae': float(np.mean(np.abs(error))), 'poisson_deviance': float(np.mean(deviance)),
        'total_mae': float(np.mean(np.abs(total_pred - total_true))),
        'total_relative_mae_pct': float(100 * np.mean(np.abs(total_pred[positive] - total_true[positive]) / total_true[positive])),
        'total_bias_pct': float(100 * (total_pred.sum() - total_true.sum()) / total_true.sum()),
        'shape_tv': float(np.mean(shape_tv[positive])),
        'edge_events': int(edge.sum()),
        'edge_rmse': float(np.sqrt(np.mean(error[edge]**2))) if edge.any() else None,
        'pmt3_region_rmse': float(np.sqrt(np.mean(error[:, region]**2))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    model, checkpoint, geometry, method = load_checkpoint(args.checkpoint, args.device)
    if checkpoint['args'].get('smoke_events', 0) and not args.limit:
        raise ValueError('Smoke checkpoints require --limit and are not paper results')
    with h5py.File(args.data, 'r') as source:
        if str(source.attrs['dataset_id']) != checkpoint['dataset_id']:
            raise ValueError('Checkpoint and benchmark dataset differ')
        rows = source['splits/test'][:]
        if args.limit:
            rows = rows[:args.limit]
        pe = take(source['lr'], rows).astype('f4')
        truth = take(source['hr'], rows).astype('f4')
        source_xy = take(source['metadata/truth_xy'], rows)
    pred = predict(model, pe, geometry, args.device,
                   checkpoint['args']['batch_size'], checkpoint['args']['eval_channels'])
    result = {'method': method, 'dataset_id': checkpoint['dataset_id'],
              'selection': 'best validation Poisson NLL', **metrics(pred, truth, source_xy, geometry)}
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
