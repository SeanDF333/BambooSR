"""Load released models and render virtual HR counts."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    args = checkpoint['args']
    geometry = checkpoint['geometry']
    method = args['model']
    if checkpoint.get('dataset_id') is None:
        raise ValueError('Missing training dataset identity')
    if checkpoint['format'] == 'bamboo_sr_arch_v3' and method == 'bamboo':
        from bamboo_sr_v3.model import BambooSRV3
        model = BambooSRV3(geometry['lr_centers'], geometry['lr_mask_indices'],
                           width=args['width'], variant=args['variant'], wall_radius=args['wall_radius'])
        name = 'bamboosr'
    elif checkpoint['format'] == 'conditional_field_release_v1' and method in ('ffn', 'siren'):
        from models import ConditionalField
        model = ConditionalField(method, width=args['width'])
        name = method
    elif checkpoint['format'] == 'wire_bounded_condition_v1' and method == 'wire':
        from wire_sr_stable.train import StableWire
        model = StableWire(args['width'])
        name = method
    elif checkpoint['format'] == 'recent_inr_conditional_v1' and method == 'sasnet':
        from recent_inr.fields import RecentField
        model = RecentField('sasnet', args['width'])
        name = method
    else:
        raise ValueError('Unsupported checkpoint format or method')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device).eval()
    return model, checkpoint, geometry, name


@torch.inference_mode()
def predict(model, pe, geometry, device='cpu', batch_size=8, channel_chunk=64):
    points = torch.as_tensor(geometry['query_mm'], dtype=torch.float32, device=device)
    weights = torch.as_tensor(geometry['weights'], dtype=torch.float32, device=device)
    output = []
    for start in range(0, len(pe), batch_size):
        q = torch.as_tensor(pe[start:start + batch_size], dtype=torch.float32, device=device)
        pieces = [model.counts(q, points[j:j + channel_chunk], weights[j:j + channel_chunk])
                  for j in range(0, len(points), channel_chunk)]
        output.append(torch.cat(pieces, 1).cpu().numpy())
    result = np.concatenate(output, axis=0)
    if result.shape != (len(pe), 1261) or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Invalid HR prediction')
    return result
