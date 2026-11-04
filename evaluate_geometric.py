"""Evaluate Area, Nearest or IDW on the released simulation test split."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from baselines.geometric import method_matrix
from evaluate_simulation import metrics, take


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--method', choices=['area', 'nearest', 'idw'], required=True)
    parser.add_argument('--output')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    with h5py.File(args.data, 'r') as source:
        geometry = {key: source['geometry/' + key][:] for key in
                    ('lr_centers', 'hr_centers', 'lr_mask_indices')}
        rows = source['splits/test'][:]
        if args.limit:
            rows = rows[:args.limit]
        lr = take(source['lr'], rows).astype('f4')
        truth = take(source['hr'], rows).astype('f4')
        source_xy = take(source['metadata/truth_xy'], rows)
        dataset_id = str(source.attrs['dataset_id'])
    matrix = method_matrix(args.method, geometry['lr_centers'], geometry['hr_centers'],
                           geometry['lr_mask_indices'])
    prediction = lr @ matrix.T
    result = {'method': args.method, 'dataset_id': dataset_id,
              **metrics(prediction, truth, source_xy, geometry)}
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
