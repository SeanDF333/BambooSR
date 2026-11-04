"""Train the SASNet conditional baseline under the matched v3 protocol."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol
from fields import RecentField


def arguments(p):
    p.set_defaults(epochs=160, patience=15, min_delta=1e-4, selection='nll',
        scheduler_patience=5, max_minutes=0, baseline_protocol=True,
        consistency=0., residual_penalty=0., lr=.001)


def optimizer(model, a):
    mask = list(model.net.grid.parameters())+list(model.net.decoder.parameters())
    ids = {id(p) for p in mask}
    return torch.optim.Adam([
        {'params': [p for p in model.parameters() if id(p) not in ids], 'lr': a.lr},
        {'params': mask, 'lr': a.lr*5}], betas=(.9, .99))


if __name__ == '__main__':
    protocol.main(arguments, lambda a,g: RecentField(a.model, a.width),
                  checkpoint_format='recent_inr_conditional_v1', protocol_lr=.001,
                  optimizer_factory=optimizer)
