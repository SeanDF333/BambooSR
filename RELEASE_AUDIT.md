# Release audit — staging v1

This is a local release candidate, not a published repository or an approved data deposit. `MANIFEST.json` lists SHA256 values for both datasets, every Python source file, and every YAML config. The simulation dataset ID is `b5c64859-694c-441d-b7a1-d87ba65aa064`.

## Source mapping and transformations

| Research source | Release destination | Change |
|---|---|---|
| `code/new/bamboo_sr_v3/model.py` | `src/bamboo_sr_v3/model.py` | Byte-identical final architecture. |
| `code/new/bamboo_sr_v3/train.py` | `src/bamboo_sr_v3/train.py` | Byte-identical v3 entrypoint. |
| `code/new/models.py`, `data.py` | `src/models.py`, `src/data.py` | Byte-identical shared encoder/decoder and H5 loader. |
| `code/new/bamboo_sr/train.py`, `model.py` | `src/bamboo_sr/train.py`, `model.py` | Kept the training path and disk quadrature; removed unreachable historical v1 model branches. FFN/SIREN checkpoints use a release-specific format tag. |
| `code/new/wire_sr_stable/train.py`, `protocol.py` | `src/wire_sr_stable/` | Kept the repaired WIRE architecture and optimization; removed unreachable v1 model branch in the copied protocol. |
| `code/new/recent_inr/fields.py`, `train.py`, `protocol.py` | `src/recent_inr/` | Kept SASNet adaptation; removed the excluded PIN branch and unreachable v1 branch. |
| `code/new/recent_inr/vendor/SASNet_inr/sasnet/models/` | `src/recent_inr/vendor/SASNet_inr/sasnet/models/` | Included only the SASNet model modules and their upstream license. |
| `code/new/aligned_v3/baseline_v3.py` | `baselines/geometric.py` | Kept the exact Area/Nearest/IDW transfer-matrix functions; omitted Oracle and server-specific output code. |
| `code/new/wall_cog/analyze.py` | `analyze_wall.py` | Retained the 24-sector, opposite-fold median scaling and split-normal fit equations in a standalone path-free CLI. |

Top-level training, evaluation, inference, validation, and geometric-evaluation CLIs are release entrypoints. They do not replace the final BambooSR model. All staged Python comments are in English. No checkpoint, Condor file, old 5,637-event wall product, parent 6,991-event wall sample, old wrong-order dataset, PAF, BQ, BQG, PIN, or scratch plot is included.

## Data audit

- Simulation file: 112,890 × 169 LR, 112,890 × 1,261 HR; split sizes 90,330/11,280/11,280. Exactly 11,289 `grid_id` positions have ten events each; positions do not cross splits.
- Channel audit: physical PMT ID 3 is LR column 2, mapped to native MC column 163; that LR column is zero in all events. HR remains unmasked. LR/HR column counts match geometry IDs and centers.
- Geometry audit: 1,261 × 128 query points and weights; per-channel weight sums match a 12.5 mm photocathode disk relative to the 169-disk LR reference area. All HR channels have full target coverage.
- Wall file: exactly 5,583 × 169 PE counts and public sequential event indices 0–5,582. It contains no truth XY, parent selection rows, original run/event IDs, or unrelated Run 2 fields.
- Removed private server-location attributes from the copied H5. Scientific arrays, split indices, and dataset ID were not rewritten. H5 and wall SHA256 values are in `MANIFEST.json`.

## Checks performed

`validate_release.py` passed on the staged data: required shapes, dtypes, finite values, channel mask, split isolation, full HR targets, quadrature geometry, wall size, data hashes, and text-path scan. BambooSR v3 model and entrypoint SHA256 values match their research sources. One four-event, one-epoch CPU training smoke run produced a validation-selected checkpoint. Four-event simulation evaluator and Area baseline smoke evaluations passed. FFN, SIREN, repaired WIRE, and SASNet each produced a finite two-channel forward pass. The standalone wall analysis was compared with the verified 5,583-event COG cache: LR and BambooSR fitted inward widths agreed within 1.3e-5 mm.

The clean-room copy test passed using only the release source and its two data files: validation, one-epoch four-event CPU training, two-event model evaluation, Area evaluation, Area wall inference, and wall analysis. Smoke outputs remain outside the release and are not paper results. Full baseline retraining has not been completed. `git_commit` is null because this staging workspace is not a Git repository.

## Approval and limitations

**Staging status at the time of this audit:** the owner's release confirmation and final licensing terms had not yet been recorded. They were subsequently recorded in `RELEASE_DECISIONS.md`, `LICENSE`, and `DATA_LICENSE.md` on 2026-10-26. The underlying institutional authorization document is not bundled. **Still not verified:** fresh-install environment resolution on another machine; full retraining of every released baseline; complete neural wall inference from a freshly trained formal checkpoint; numerical parity with historical FFN/SIREN checkpoints. The simulation benchmark's test split was inspected during method development, so it is not a fresh method-development holdout. Virtual HR counts use a geometric photocathode footprint and do not model R7057-specific QE, gain, or electronics. Hardware/PyTorch changes may alter retraining results.
