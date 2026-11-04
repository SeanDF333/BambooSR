"""Audited upstream-only trainer. Test data are never used for model selection."""
import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import Pairs
from models import ConditionalField, count_loss
if __package__:
    from .model import disk_quadrature
else:
    from model import disk_quadrature


def audited_geometry(path, lr_radius, hr_radius):
    with h5py.File(path, 'r') as f:
        if not f.attrs.get('complete') or f.attrs.get('version') != 'aligned_mask3_v3':
            raise ValueError('Require completed channel-corrected v3 dataset')
        if f.attrs.get('lr_channel_order') != 'lr_ids_geometry_row_order':
            raise ValueError('Unverified LR channel order')
        g = {k: f['geometry'][k][:] for k in f['geometry']}
        masked = g['lr_mask_indices'].astype(int)
        if not np.array_equal(g['lr_ids'][masked], [3]):
            raise ValueError('Expected physical PMT3 mask')
        if not np.array_equal(g['real_to_mc_index'][masked], [163]):
            raise ValueError('Native/real channel mapping mismatch')
        if not np.all(g['coverage'] == 1):
            raise ValueError('Require full HR labels, including the missing PMT region')
        reference_area = len(g['lr_ids'])*math.pi*lr_radius**2
        areas = g['weights'].sum(-1)*reference_area
        if not np.allclose(areas, math.pi*hr_radius**2, rtol=1e-4):
            raise ValueError('Quadrature normalization/radius mismatch')
        if np.max(np.linalg.norm(g['query_mm']-g['hr_centers'][:, None], axis=-1)) > hr_radius+.01:
            raise ValueError('HR query points outside cathodes')
        split_rows = [f['splits/'+s][:] for s in ('train', 'val', 'test')]
        joined = np.concatenate(split_rows)
        if len(np.unique(joined)) != len(joined) or len(joined) != len(f['lr']):
            raise ValueError('Splits overlap or omit events')
        groups = f['metadata/grid_id'][:]
        group_sets = [set(groups[r].tolist()) for r in split_rows]
        if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i)):
            raise ValueError('Source grid positions leak across splits')
        return str(f.attrs['dataset_id']), g, reference_area


def build_model(args, geometry):
    if args.model == 'bamboo':
        raise ValueError('Use the BambooSR v3 entrypoint for the published model')
    return ConditionalField(model=args.model, width=args.width)


def render(model, pe, points, weights, reference_area, context=None):
    return model.counts(pe, points, weights/reference_area), pe.new_zeros(())


@torch.no_grad()
def evaluate(model, loader, points, weights, reference_area, chunk=64):
    model.eval()
    squared = absolute = nll = total_error = 0.
    elements = events = 0
    for pe, truth in loader:
        pe, truth = pe.to(points.device), truth.to(points.device)
        context = None
        predicted_total = torch.zeros(len(pe), device=points.device)
        for start in range(0, len(points), chunk):
            sl = slice(start, start+chunk)
            pred, _ = render(model, pe, points[sl], weights[sl], reference_area, context)
            target = truth[:, sl]
            squared += (pred-target).square().sum().item()
            absolute += (pred-target).abs().sum().item()
            nll += count_loss(pred, target).item()*pred.numel()
            elements += pred.numel()
            predicted_total += pred.sum(-1)
        total_error += (predicted_total-truth.sum(-1)).abs().sum().item()
        events += len(pe)
    if not elements:
        raise ValueError('Empty evaluation split')
    return dict(rmse=math.sqrt(squared/elements), mae=absolute/elements,
                nll=nll/elements, total_mae=total_error/events, events=events)


def save(path, content):
    temp = path.with_suffix('.tmp')
    torch.save(content, temp)
    temp.replace(path)


def main(add_arguments=None, model_factory=None, checkpoint_format='bamboo_sr_v1', protocol_lr=1e-4):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', choices=['bamboo', 'siren', 'ffn', 'wire'], default='bamboo')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--min-delta', type=float, default=.001, help='Absolute validation RMSE improvement, PE/channel')
    p.add_argument('--selection', choices=['rmse', 'nll'], default='rmse',
                   help='validation metric for scheduler, best.pt, and early stopping')
    p.add_argument('--scheduler-patience', type=int, default=3)
    p.add_argument('--max-minutes', type=float, default=120, help='Checked at epoch boundaries; 0 disables time limit')
    p.add_argument('--baseline-protocol', action='store_true',
                   help='Require old v3 protocol and its min-delta checkpoint rule')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--train-channels', type=int, default=32)
    p.add_argument('--eval-channels', type=int, default=64)
    p.add_argument('--lr-channels', type=int, default=16)
    p.add_argument('--consistency', type=float, default=.1)
    p.add_argument('--residual-penalty', type=float, default=.001)
    p.add_argument('--lr-radius', type=float, default=32.)
    p.add_argument('--hr-radius', type=float, default=12.5)
    p.add_argument('--wall-radius', type=float, default=592.5)
    p.add_argument('--no-reflection', action='store_true')
    p.add_argument('--no-residual', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke-events', type=int, default=0, help='Train/val subset for integration tests only')
    if add_arguments is not None:
        add_arguments(p)
    a = p.parse_args()
    if min(a.epochs, a.patience, a.scheduler_patience, a.batch_size, a.train_channels, a.lr_channels, a.eval_channels, a.width) < 1:
        p.error('Counts must be positive')
    if min(a.lr, a.lr_radius, a.hr_radius, a.wall_radius) <= 0 or min(a.max_minutes, a.workers, a.smoke_events, a.min_delta, a.consistency, a.residual_penalty) < 0:
        p.error('Invalid scale, weight or limit')
    if a.baseline_protocol:
        expected = dict(patience=15, min_delta=1e-4, selection='nll', scheduler_patience=5,
                        max_minutes=0., batch_size=8, lr=protocol_lr, train_channels=32,
                        eval_channels=64, seed=42, width=128)
        if not a.smoke_events:
            expected['epochs'] = 160
        for key, value in expected.items():
            if getattr(a, key) != value:
                p.error(f'Baseline protocol requires --{key.replace("_", "-")} {value}')
        if a.model != 'bamboo' and (a.consistency or a.residual_penalty):
            p.error('Plain INR baselines require --consistency 0 --residual-penalty 0')
    torch.set_num_threads(2)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(a.device)
    # Do not change HR channel sampling or event shuffling when ablating LR loss.
    lr_generator = torch.Generator(device=device).manual_seed(a.seed+10000)
    dataset_id, g, area = audited_geometry(a.data, a.lr_radius, a.hr_radius)
    model = (model_factory or build_model)(a, g).to(device)
    points = torch.tensor(g['query_mm'], dtype=torch.float32, device=device)
    weights = torch.tensor(g['weights']*area, dtype=torch.float32, device=device)
    lr_points, lr_weights = disk_quadrature(torch.tensor(g['lr_centers'], dtype=torch.float32, device=device), a.lr_radius)
    healthy = torch.tensor(np.setdiff1d(np.arange(len(g['lr_ids'])), g['lr_mask_indices']), device=device)
    loaders = {}
    for split in ('train', 'val'):
        ds = Pairs(a.data, split)
        if a.smoke_events:
            ds = Subset(ds, range(min(a.smoke_events, len(ds))))
        loaders[split] = DataLoader(ds, batch_size=a.batch_size, shuffle=split=='train',
                                   num_workers=a.workers, pin_memory=device.type=='cuda')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if not a.resume and any(out.iterdir()):
        raise ValueError('Use an empty run directory or --resume')
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=a.scheduler_patience)
    start, best, anchor, stale, elapsed = 0, float('inf'), float('inf'), 0, 0.
    if a.resume:
        ck = torch.load(out/'last.pt', map_location=device, weights_only=False)
        if ck.get('format') != checkpoint_format:
            raise ValueError('Resume checkpoint architecture format mismatch')
        if ck['dataset_id'] != dataset_id:
            raise ValueError('Resume data mismatch')
        for key, value in vars(a).items():
            if key not in ('resume', 'device', 'workers', 'output', 'data', 'epochs', 'max_minutes') and ck['args'][key] != value:
                raise ValueError('Resume configuration mismatch: '+key)
        model.load_state_dict(ck['state_dict'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        start, best, anchor, stale, elapsed = [ck[k] for k in ('epoch', 'best_val', 'anchor', 'stale', 'elapsed')]
        torch.set_rng_state(ck['rng'].cpu())
        if 'lr_rng' in ck:
            lr_generator.set_state(ck['lr_rng'].cpu())
        if device.type == 'cuda' and ck['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([r.cpu() for r in ck['cuda_rng']])
    else:
        (out/'config.json').write_text(json.dumps(dict(args=vars(a), dataset_id=dataset_id,
            selection=f'validation {a.selection}; test held out', units='PE and mm',
            parameters=sum(t.numel() for t in model.parameters()), torch=torch.__version__), indent=2))
    print(json.dumps(dict(model=a.model, device=str(device), parameters=sum(t.numel() for t in model.parameters()),
        dataset_id=dataset_id, train_events=len(loaders['train'].dataset), val_events=len(loaders['val'].dataset),
        selection=a.selection, baseline_protocol=a.baseline_protocol, test='HELD OUT')), flush=True)
    begin = time.monotonic()
    for epoch in range(start, a.epochs):
        if stale >= a.patience or (a.max_minutes > 0 and elapsed+time.monotonic()-begin >= a.max_minutes*60):
            break
        tick = time.monotonic()
        model.train()
        train_sum = n = 0
        hr_sum = lr_sum = residual_sum = 0.
        for pe, truth in loaders['train']:
            pe, truth = pe.to(device), truth.to(device)
            ids = torch.randperm(len(points), device=device)[:a.train_channels]
            context = None
            pred, reg = render(model, pe, points[ids], weights[ids], area, context)
            hr_loss = count_loss(pred, truth[:, ids])
            lr_loss = pe.new_zeros(())
            loss = hr_loss+a.residual_penalty*reg
            if a.consistency:
                selected = healthy[torch.randperm(len(healthy), device=device, generator=lr_generator)[:a.lr_channels]]
                lr_pred, _ = render(model, pe, lr_points[selected], lr_weights[selected], area, context)
                # Poisson deviance/target total makes consistency insensitive to
                # the differing PE scale of large and small photocathodes.
                observed = pe[:, selected]
                deviance = lr_pred-observed+observed*(torch.log(observed.clamp_min(1e-8))-torch.log(lr_pred.clamp_min(1e-8)))
                lr_loss = (deviance.mean(1)/observed.mean(1).clamp_min(1.)).mean()
                loss = loss+a.consistency*lr_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            train_sum += loss.item()*len(pe)
            hr_sum += hr_loss.item()*len(pe)
            lr_sum += lr_loss.item()*len(pe)
            residual_sum += reg.item()*len(pe)
            n += len(pe)
        metrics = evaluate(model, loaders['val'], points, weights, area, a.eval_channels)
        if not np.isfinite(list(metrics.values())).all():
            raise FloatingPointError('Nonfinite validation')
        selected = metrics[a.selection]
        scheduler.step(selected)
        improved = selected < best-(a.min_delta if a.baseline_protocol else 0.)
        if improved:
            best = selected
        if selected < anchor-a.min_delta:
            anchor, stale = selected, 0
        else:
            stale += 1
        duration = elapsed+time.monotonic()-begin
        record = dict(epoch=epoch+1, train_objective=train_sum/n, train_nll=hr_sum/n,
                      train_lr_deviance=lr_sum/n, train_residual_penalty=residual_sum/n,
                      **{'val_'+k:v for k,v in metrics.items()},
                      lr=optimizer.param_groups[0]['lr'], seconds=time.monotonic()-tick,
                      elapsed_seconds=duration, stale=stale)
        with (out/'history.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        ck = dict(format=checkpoint_format, args=vars(a), geometry=g, reference_area=area,
                  dataset_id=dataset_id, epoch=epoch+1, best_val=best, anchor=anchor, stale=stale,
                  metrics=metrics, elapsed=duration, state_dict=model.state_dict(),
                  optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                  rng=torch.get_rng_state(), lr_rng=lr_generator.get_state(),
                  cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
        save(out/'last.pt', ck)
        if improved:
            save(out/'best.pt', ck)
        print(json.dumps(record), flush=True)
    print(f'Finished: test held out; best.pt selected by validation {a.selection}.', flush=True)


if __name__ == '__main__':
    main(checkpoint_format='conditional_field_release_v1')
