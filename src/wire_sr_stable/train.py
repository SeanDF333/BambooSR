"""WIRE with bounded event conditioning and conservative optimizer groups."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models import ConditionalField
from wire_sr_stable import protocol


class StableWire(ConditionalField):
    def __init__(self, width=128):
        super().__init__('wire',width=width,omega=10.,wire_scale=5.)
        self.config.update(condition_bound=.25, adaptation='bounded_condition_v1')
        self._diagnose=False

    def encode(self,pe):
        latent,total=super().encode(pe)
        # Near-identity for small latent values, bounded input to localized Gabor.
        return .25*torch.tanh(latent/.25),total

    def train(self,mode=True):
        self._diagnose=not mode
        return super().train(mode)

    def counts(self,pe,query_mm,weights):
        if self._diagnose:
            self._diagnose=False
            with torch.no_grad():
                latent,_=self.encode(pe)
                density=self.density(latent,query_mm.reshape(-1,2))
                cv=(density.std(-1)/density.mean(-1).clamp_min(1e-12)).median().item()
                print('WIRE_HEALTH latent_abs_max=%.6g spatial_cv=%.6g' %
                      (latent.abs().max().item(),cv),flush=True)
                if cv<1e-7:
                    raise RuntimeError('WIRE collapsed to a spatially constant field; refusing misleading baseline')
        return super().counts(pe,query_mm,weights)


def arguments(p):
    p.set_defaults(epochs=160,patience=15,min_delta=1e-4,selection='nll',
                   scheduler_patience=5,max_minutes=0,baseline_protocol=True,
                   consistency=0.,residual_penalty=0.,lr=5e-4)


def optimizer(m,a):
    return torch.optim.Adam([{'params':m.encoder.parameters(),'lr':1e-4},
                             {'params':m.decoder.parameters(),'lr':a.lr}])


if __name__=='__main__':
    protocol.main(arguments,lambda a,g:StableWire(a.width),
                  checkpoint_format='wire_bounded_condition_v1',protocol_lr=5e-4,
                  optimizer_factory=optimizer)
