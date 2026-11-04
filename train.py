"""Run a released training configuration."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parent
ENTRY = {
    'bamboosr': 'bamboo_sr_v3/train.py',
    'ffn': 'bamboo_sr/train.py',
    'siren': 'bamboo_sr/train.py',
    'wire': 'wire_sr_stable/train.py',
    'sasnet': 'recent_inr/train.py',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke-events', type=int, default=0)
    parser.add_argument('--smoke-epochs', type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if config['protocol'] != {'optimizer': 'Adam', 'grad_clip': 1,
                              'scheduler_factor': 0.5,
                              'checkpoint_selection': 'validation_poisson_nll'}:
        raise ValueError('Config protocol differs from the frozen training implementation')
    method = config['method']
    command = [sys.executable, '-u', str(ROOT / 'src' / ENTRY[method]),
               '--data', str(Path(args.data).resolve()), '--output', str(Path(args.output).resolve()),
               '--device', args.device]
    if method == 'bamboosr':
        command += ['--variant', config['variant']]
    else:
        command += ['--model', method]
    for key, value in config['training'].items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command += [flag, str(value)]
    if args.smoke_events:
        command += ['--smoke-events', str(args.smoke_events)]
    if args.smoke_epochs:
        if not args.smoke_events:
            parser.error('--smoke-epochs requires --smoke-events')
        command += ['--epochs', str(args.smoke_epochs)]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    main()
