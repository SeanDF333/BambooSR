# BambooSR reproducibility package

**Repository:** https://github.com/SeanDF333/BambooSR

This repository contains code for BambooSR and the paper baselines, plus documentation for the final channel-aligned simulation benchmark and the fixed 5,583-event Run 2 wall sample. The versioned data files are prepared for separate distribution as GitHub Release assets. See [RELEASE_DECISIONS.md](RELEASE_DECISIONS.md) for the owner's release confirmation, [LICENSE](LICENSE) for the MIT code license, and [DATA_LICENSE.md](DATA_LICENSE.md) for the CC BY 4.0 dataset license.

The model consumes 169 top-PMT counts with physical PMT3 masked. It predicts a continuous nonnegative field, then integrates it over 1,261 virtual photocathodes. The released HR labels are geometry-derived counts from the same simulated photon realization, not a full electronics or R7057 response simulation.

## Setup

After the versioned GitHub Release is published, download `bamboosr_sim_benchmark_v1.h5` and `bamboosr_wall_run2_5583_v1.npz` into `data/`. Check their SHA256 hashes against `MANIFEST.json`. Create an environment using `environment.yml`, or install `requirements.txt` into a Python 3.10 environment. A CUDA GPU is recommended for full neural training. Run commands from this directory. Checkpoints are produced locally; none are bundled.

```bash
python validate_release.py --sim-data data/bamboosr_sim_benchmark_v1.h5 --wall-data data/bamboosr_wall_run2_5583_v1.npz
python train.py --config configs/bamboosr.yaml --data data/bamboosr_sim_benchmark_v1.h5 --output runs/bamboosr_seed42
python evaluate_simulation.py --config configs/bamboosr.yaml --data data/bamboosr_sim_benchmark_v1.h5 --checkpoint runs/bamboosr_seed42/best.pt --output results/bamboosr_test.json
python infer_wall.py --config configs/bamboosr.yaml --wall-data data/bamboosr_wall_run2_5583_v1.npz --checkpoint runs/bamboosr_seed42/best.pt --output wall_predictions/bamboosr.npz
python analyze_wall.py --config configs/wall.yaml --wall-data data/bamboosr_wall_run2_5583_v1.npz --input wall_predictions/bamboosr.npz --output results/bamboosr_wall.json
```

The wall analysis reports the fitted inward split-normal width `sigma_left_mm` after a shared two-fold, 24-sector cross-calibration. This is a boundary-width proxy, not event-level XY error.

Use `configs/direct.yaml` and `configs/neural_control.yaml` for the controlled variants. Use `configs/ffn.yaml`, `siren.yaml`, `wire.yaml`, or `sasnet.yaml` with `train.py` for neural baselines. Evaluate each validation-selected `best.pt` with `evaluate_simulation.py`. Geometric baselines need no training:

```bash
python evaluate_geometric.py --data data/bamboosr_sim_benchmark_v1.h5 --method area
python evaluate_geometric.py --data data/bamboosr_sim_benchmark_v1.h5 --method nearest
python evaluate_geometric.py --data data/bamboosr_sim_benchmark_v1.h5 --method idw
```

Read [DATA_CARD.md](DATA_CARD.md) for data semantics and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for protocol details. [RELEASE_AUDIT.md](RELEASE_AUDIT.md) records checks and unresolved items. The model source is `src/bamboo_sr_v3/model.py`; `src/bamboo_sr/train.py` is the shared released training loop. The source retains the final event encoder, physics head, adapter, count renderer, and Poisson loss. Only obsolete v1-only branches were removed from the copied shared trainer.

The benchmark's original test split was inspected during method development; report it as a held-out split for checkpoint selection, not as a wholly untouched method-development holdout. Do not load an untrusted checkpoint: PyTorch checkpoint deserialization can execute code.
