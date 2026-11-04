"""One-time frozen v3 test evaluation with the unified upstream metrics."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import torch
from bamboo_sr.train import audited_geometry
from bamboo_sr_v3.train import build_model


def take(ds, rows):
    order=np.argsort(rows)
    return ds[np.asarray(rows)[order]][np.argsort(order)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args(); torch.set_num_threads(2)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ck.get('format')!='bamboo_sr_arch_v3':
        raise ValueError('Checkpoint is not BambooSR architecture v3')
    config=SimpleNamespace(**ck['args'])
    if config.smoke_events:
        raise ValueError('Smoke checkpoint is not a scientific result')
    identity,g,area=audited_geometry(a.data,config.lr_radius,config.hr_radius)
    if identity!=ck['dataset_id']:
        raise ValueError('Dataset identity mismatch')
    with h5py.File(a.data,'r') as f:
        rows=f['splits/test'][:]
        lr=take(f['lr'],rows).astype('f4')
        truth=take(f['hr'],rows).astype('f4')
        xy=take(f['metadata/truth_xy'],rows).astype('f4')
    model=build_model(config,g).to(a.device).eval()
    model.load_state_dict(ck['state_dict'])
    points=torch.tensor(g['query_mm'],dtype=torch.float32,device=a.device)
    weights=torch.tensor(g['weights'],dtype=torch.float32,device=a.device)
    pred=np.empty_like(truth)
    with torch.no_grad():
        for start in range(0,len(lr),config.batch_size):
            pe=torch.tensor(lr[start:start+config.batch_size],device=a.device)
            chunks=[]
            for j in range(0,len(points),config.eval_channels):
                chunks.append(model.counts(pe,points[j:j+config.eval_channels],
                                           weights[j:j+config.eval_channels]).cpu())
            pred[start:start+len(pe)]=torch.cat(chunks,1).numpy()
            if start%1600==0:
                print(start,len(lr),flush=True)
    if not np.isfinite(pred).all() or (pred<0).any():
        raise FloatingPointError('Invalid test prediction')
    y=truth.astype('f8'); mu=pred.astype('f8'); err=mu-y; eps=1e-8
    dev=2*(mu-y+y*(np.log(np.maximum(y,eps))-np.log(np.maximum(mu,eps))))
    total=y.sum(1); predicted_total=mu.sum(1); nz=total>0
    shape=.5*np.abs(mu/np.maximum(predicted_total[:,None],eps)-
                    y/np.maximum(total[:,None],eps)).sum(1)
    pmt3=g['lr_centers'][int(g['lr_mask_indices'][0])]
    missing=np.linalg.norm(g['hr_centers']-pmt3,axis=1)<=32
    edge=np.linalg.norm(xy,axis=1)>=520
    result=dict(model='bamboo_v3_'+config.variant,events=len(y),checkpoint_epoch=ck['epoch'],
        selection='best checkpoint by validation NLL; test evaluated once',dataset_id=identity,
        rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(abs(err))),
        poisson_deviance=float(np.mean(dev)),total_mae=float(np.mean(abs(predicted_total-total))),
        total_relative_mae_pct=float(100*np.mean(abs(predicted_total[nz]-total[nz])/total[nz])),
        total_bias_pct=float(100*(predicted_total.sum()-total.sum())/total.sum()),
        shape_tv=float(np.mean(shape[nz])),
        pmt3_region_rmse=float(np.sqrt(np.mean(err[:,missing]**2))),
        edge_definition='truth source radius >= 520 mm',edge_events=int(edge.sum()),
        edge_rmse=float(np.sqrt(np.mean(err[edge]**2))),parameters=sum(x.numel() for x in model.parameters()),
        units='PE/channel except percentages and dimensionless shape/deviance')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
