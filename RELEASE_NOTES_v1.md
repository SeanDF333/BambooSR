# BambooSR v1 release notes

This release accompanies the BambooSR manuscript and its reproducibility
package. The repository contains code, configurations, and validation scripts.
Two versioned data files are planned as GitHub Release attachments rather
than Git commits:

| File | Size | SHA256 |
| --- | ---: | --- |
| `bamboosr_sim_benchmark_v1.h5` | 651,061,260 bytes | `6a3480adbc49f3e06f42c099f22bd42eb93b9cc30b99ebdcac323680a6f58f02` |
| `bamboosr_wall_run2_5583_v1.npz` | 3,166,548 bytes | `342c9d3a3b69c0c4dc1950e02fb83973a44993ed8106b31353aefe7f1a187e18` |

The simulation benchmark has 112,890 paired 169-channel/1,261-channel
readouts. The real-data file is the fixed 5,583-event Run 2 wall sample used
in the paper, without original event IDs or event-level truth positions.

Code is MIT-licensed; data are CC BY 4.0. See `DATA_CARD.md`,
`REPRODUCIBILITY.md`, and `RELEASE_AUDIT.md` before use. The simulation test
split was inspected during method development and is not a fresh
method-development holdout. Virtual HR counts use geometric photocathode
footprints and do not include tube-specific electronics response.
