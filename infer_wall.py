"""Predict virtual HR counts for the released fixed wall sample."""
import argparse
from pathlib import Path

import numpy as np

from release_core import load_checkpoint, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--wall-data', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    with np.load(args.wall_data, allow_pickle=False) as wall:
        pe = wall['lr_pe'].astype('f4')
        event_index = wall['event_index']
    if pe.shape != (5583, 169) or np.any(pe[:, 2] != 0):
        raise ValueError('Wrong wall sample or PMT3 mask')
    model, checkpoint, geometry, method = load_checkpoint(args.checkpoint, args.device)
    if checkpoint['args'].get('smoke_events', 0):
        raise ValueError('Smoke checkpoints cannot be used for wall results')
    hr = predict(model, pe, geometry, args.device,
                 checkpoint['args']['batch_size'], checkpoint['args']['eval_channels'])
    centers = geometry['hr_centers']
    xy = hr.astype('f8') @ centers / hr.sum(1)[:, None]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, event_index=event_index, lr_pe=pe,
                        hr_pe=hr, cog_xy=xy, lr_centers=geometry['lr_centers'],
                        hr_centers=centers, method=np.array(method))


if __name__ == '__main__':
    main()
