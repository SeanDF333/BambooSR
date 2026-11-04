"""Validate benchmark identity, geometry, split integrity and release files."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
PRIVATE = re.compile(r'/home/|/lustre/|[A-Za-z]:\\|(?:password|api[_-]?key|secret)\s*[:=]', re.I)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_sim(path):
    with h5py.File(path, 'r') as source:
        assert source.attrs['complete'] and source.attrs['version'] == 'aligned_mask3_v3'
        assert source.attrs['lr_channel_order'] == 'lr_ids_geometry_row_order'
        assert source['lr'].shape == (112890, 169)
        assert source['hr'].shape == (112890, 1261)
        assert source['metadata/truth_xy'].shape == (112890, 2)
        assert source['metadata/grid_id'].shape == (112890,)
        assert source['event_index'].shape == (112890,)
        g = source['geometry']
        for name, shape in {'lr_centers': (169, 2), 'hr_centers': (1261, 2),
                            'lr_ids': (169,), 'hr_ids': (1261,),
                            'query_mm': (1261, 128, 2), 'weights': (1261, 128),
                            'real_to_mc_index': (169,), 'native_mc_centers': (169, 2)}.items():
            assert g[name].shape == shape, name
        mask = g['lr_mask_indices'][:].astype(int)
        assert np.array_equal(mask, [2]) and np.array_equal(g['lr_ids'][mask], [3])
        assert np.array_equal(g['real_to_mc_index'][mask], [163])
        assert len(np.unique(g['lr_ids'][:])) == 169 and len(np.unique(g['hr_ids'][:])) == 1261
        assert np.all(g['coverage'][:] == 1)
        assert np.allclose(g['weights'][:].sum(1) * 169 * math.pi * 32**2,
                           math.pi * 12.5**2, rtol=1e-4)
        for key in ('lr_centers', 'hr_centers', 'query_mm', 'weights'):
            assert np.isfinite(g[key][:]).all(), key
        splits = [source['splits/' + key][:] for key in ('train', 'val', 'test')]
        assert [len(part) for part in splits] == [90330, 11280, 11280]
        combined = np.concatenate(splits)
        assert np.array_equal(np.sort(combined), np.arange(112890))
        grid = source['metadata/grid_id'][:]
        groups = [set(grid[part].tolist()) for part in splits]
        assert all(not groups[i].intersection(groups[j]) for i in range(3) for j in range(i))
        assert len(np.unique(grid)) == 11289
        assert np.unique(np.bincount(np.unique(grid, return_inverse=True)[1])).tolist() == [10]
        assert np.array_equal(source['event_index'][:], np.arange(112890))
        assert np.isfinite(source['metadata/truth_xy'][:]).all()
        for key in ('lr', 'hr'):
            array = source[key]
            assert array.dtype.kind in 'fiu', key
            for start in range(0, len(array), 1024):
                chunk = array[start:start + 1024]
                assert np.isfinite(chunk).all() and (chunk >= 0).all(), key
                if key == 'lr':
                    assert np.all(chunk[:, 2] == 0)
        pmt3 = g['lr_centers'][2]
        near = np.linalg.norm(g['hr_centers'][:] - pmt3, axis=1) <= 32
        assert near.any() and (source['hr'][:1024, near] > 0).any()
        def inspect(name, item):
            for value in item.attrs.values():
                assert not PRIVATE.search(str(value)), f'Private metadata: {name}'
        for value in source.attrs.values():
            assert not PRIVATE.search(str(value)), 'Private root metadata'
        source.visititems(inspect)
        return {'dataset_id': str(source.attrs['dataset_id']), 'events': 112890,
                'source_positions': len(np.unique(grid))}


def validate_wall(path):
    with np.load(path, allow_pickle=False) as wall:
        assert set(wall.files) == {'lr_pe', 'event_index'}
        assert wall['lr_pe'].shape == (5583, 169)
        assert np.array_equal(wall['event_index'], np.arange(5583))
        assert wall['lr_pe'].dtype.kind == 'f'
        assert np.isfinite(wall['lr_pe']).all() and np.all(wall['lr_pe'][:, 2] == 0)
    return {'events': 5583, 'truth_xy': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim-data', default='data/bamboosr_sim_benchmark_v1.h5')
    parser.add_argument('--wall-data', default='data/bamboosr_wall_run2_5583_v1.npz')
    args = parser.parse_args()
    sim = Path(args.sim_data)
    wall = Path(args.wall_data)
    report = {'simulation': validate_sim(sim), 'wall': validate_wall(wall),
              'sha256': {sim.name: sha256(sim), wall.name: sha256(wall)}}
    for path in ROOT.rglob('*'):
        if path.suffix in {'.py', '.yaml', '.md', '.txt', '.json'} and path.is_file():
            if path.name == 'validate_release.py':
                continue
            assert not PRIVATE.search(path.read_text(encoding='utf-8')), f'Private path or secret-like text: {path}'
    manifest = ROOT / 'MANIFEST.json'
    if manifest.exists():
        expected = json.loads(manifest.read_text(encoding='utf-8'))
        assert expected['simulation_dataset_sha256'] == report['sha256'][sim.name]
        assert expected['wall_dataset_sha256'] == report['sha256'][wall.name]
        for name, digest in expected['config_sha256'].items():
            assert sha256(ROOT / 'configs' / name) == digest
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
