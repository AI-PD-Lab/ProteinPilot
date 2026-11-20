# Licensing

HARMONIA is a combination of original code and code derived from BSD-licensed upstream projects. Two licences apply.

## PolyForm Noncommercial 1.0.0 ([`LICENSE`](LICENSE))

The original HARMONIA contributions:

- `rfd4/harmonia_ext/` (all files)
- `rfd3/scripts/`, `rfd3/docs/`, `checkpoints/`
- the SS/ADJ and ss_ratio additions to RFD3:
  `rfd3/foundry/src/foundry/data/ss_adj.py`, and under `rfd3/foundry/models/rfd3/`:
  `src/rfd3/model/layers/ss_adj_adapters.py`, `src/rfd3/model/layers/ss_conditioning.py`, `src/rfd3/model/ss_adj_cfg.py`,
  `src/rfd3/transforms/ss_adj_transforms.py`
- the changes HARMONIA made inside upstream files (the SS/ADJ hooks in `TokenInitializer`, the pipelines)
- the fine-tuned weights `Harmonia-RFD3.ckpt` and `Harmonia-RFD4/` (the changes the fine-tuning added to the base weights)

## BSD 3-Clause ([`LICENSE-BSD-3-Clause`](LICENSE-BSD-3-Clause))

Everything derived from RFdiffusion3 / foundry (Copyright (c) 2025 Institute for Protein Design, University of Washington) that is not
listed above, i.e. the rest of `rfd3/foundry/`, see also [`rfd3/foundry/LICENSE.md`](rfd3/foundry/LICENSE.md). RFD4-Proteina is not
redistributed here (install it separately; BSD 3-Clause, as are its released weights). The base weights the two checkpoints were
fine-tuned from stay under their own BSD 3-Clause licence, and its copyright and disclaimer notices must be retained wherever the
base-derived parts are redistributed.
