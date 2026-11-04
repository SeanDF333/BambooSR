"""v3: reuse audited data, observation operator and baseline training protocol."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bamboo_sr import train as shared
from bamboo_sr_v3.model import BambooSRV3


def arguments(p):
    p.set_defaults(epochs=160, patience=15, min_delta=1e-4, selection='nll',
                   scheduler_patience=5, max_minutes=0, baseline_protocol=True,
                   consistency=0., residual_penalty=0.)
    p.add_argument('--variant', choices=['full', 'direct', 'neural_control', 'ffn'], default='full')


def build_model(a, g):
    if a.model != 'bamboo' or a.consistency or a.residual_penalty or a.no_reflection or a.no_residual:
        raise ValueError('v3 uses variant switch and HR-only Poisson loss')
    if a.lr_radius != 32.:
        raise ValueError('v3 field normalization assumes audited 32 mm LR cathodes')
    return BambooSRV3(g['lr_centers'], g['lr_mask_indices'], width=a.width,
                      variant=a.variant, wall_radius=a.wall_radius)


if __name__ == '__main__':
    shared.main(arguments, build_model, checkpoint_format='bamboo_sr_arch_v3')
