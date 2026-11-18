"""rfd4-generate for Harmonia-RFD4 (the script ProteinPilot's RFD4 export runs).

  python -m harmonia_ext.rfd4_generate inputs=<dir of .cif + .adj.npy> hf_ckpt_path=<Harmonia-RFD4 dir> out_dir=<dir> \
      harmonia_w=1.5 [any rfd4-generate Hydra override]

Needs harmonia_ext on PYTHONPATH and an RFD4-Proteina install (see rfd4/README.md). On top of the stock pipeline:
  1. harmonia_ext.adj_model.register(): the trunk with adj_embed, without which the
     checkpoint does not load;
  2. every input <stem>.cif gets feats['adj_state'] from <stem>.adj.npy beside it,
     after RFD4's own pipeline, checked against the example's token count
     (collate pads it with PAD = 5, which adds nothing);
  3. harmonia_ext.generate.install_guidance(): pred = masked + w (cond - masked),
     the masked pass with adj_state all MASKED and C_SEC all NONE.
Extra argument, removed before Hydra parses the rest: harmonia_w=<float> (1.0 = off).
"""
import os
import sys

import numpy as np
import torch

HARMONIA_W = 1.5
_args = []
for _a in sys.argv[1:]:
    if _a.startswith("harmonia_w="):
        HARMONIA_W = float(_a.split("=", 1)[1])
    else:
        _args.append(_a)
# The patches below live in this process. Dataloader workers would get a fresh
# copy of the dataset class without them, so load in-process unless asked otherwise.
if not any(_a.split("=", 1)[0].lstrip("+~") == "num_workers" for _a in _args):
    _args.append("num_workers=0")
sys.argv = [sys.argv[0]] + _args

from harmonia_ext.adj_model import register  # noqa: E402

register()

from harmonia_ext.generate import install_guidance  # noqa: E402
import rfproteina.generate as G  # noqa: E402
from rfproteina.conditions import C_SEC  # noqa: E402
from rfproteina.inference_utils import datasets as DS  # noqa: E402

_orig_getitem = DS.DesignProblemDataset.__getitem__
_reported = set()


def adj_path_for(cif):
    p = str(cif)
    for ext in (".cif.zst", ".cif.gz", ".bcif", ".cif"):
        if p.endswith(ext):
            return p[: -len(ext)] + ".adj.npy"
    return os.path.splitext(p)[0] + ".adj.npy"


def getitem_with_adj(self, idx):
    data = _orig_getitem(self, idx)
    cif = self.file_paths[idx]
    path = adj_path_for(cif)
    if not os.path.isfile(path):
        raise FileNotFoundError("HARMONIA: no adjacency map for %s (expected %s)" % (cif, path))
    feats = data["feats"]
    n = int(feats["residue_type"].shape[0])
    adj = np.load(path)
    if adj.shape != (n, n):
        raise ValueError("HARMONIA: %s is %s but %s has %d tokens after RFD4's pipeline"
                         % (path, adj.shape, cif, n))
    feats["adj_state"] = torch.from_numpy(adj.astype(np.int16))
    # <stem>.ss.npy, when present, must say what the CIF's C_SEC says (ProteinPilot
    # codes 0 H / 1 E / 2 L / 3 masked -> C_SEC 1 / 2 / 3 / 0).
    ss_path = path[: -len(".adj.npy")] + ".ss.npy"
    if os.path.isfile(ss_path) and C_SEC.full_name in feats:
        want = np.array([(1, 2, 3, 0)[int(v)] for v in np.load(ss_path)], dtype=np.int64)
        got = torch.as_tensor(feats[C_SEC.full_name]).reshape(-1)[:n].long().numpy()
        if want.shape != got.shape or (want != got).any():
            raise ValueError("HARMONIA: %s disagrees with the C_SEC of %s - re-download both from the same drawing"
                             % (ss_path, cif))
    if path not in _reported:
        _reported.add(path)
        vals, cnt = np.unique(adj, return_counts=True)
        print(">>> HARMONIA adj_state %s: %d tokens, state counts %s"
              % (os.path.basename(path), n, dict(zip(vals.tolist(), cnt.tolist()))), flush=True)
    return data


DS.DesignProblemDataset.__getitem__ = getitem_with_adj

_orig_load = G.load_ckpt_n_configure_inference


def load_with_guidance(cfg, model=None):
    m = _orig_load(cfg, model=model)
    if not hasattr(m.nn, "adj_embed"):
        raise RuntimeError("HARMONIA: the loaded trunk has no adj_embed - not a HARMONIA-RFD4 bundle?")
    print(">>> HARMONIA-RFD4 trunk %s, adj_embed max|w| %.4f, guidance w %.2f"
          % (type(m.nn).__name__, float(m.nn.adj_embed.weight.detach().abs().max()), HARMONIA_W), flush=True)
    install_guidance(m.fm, C_SEC.full_name, {"w": HARMONIA_W}, m)
    return m


G.load_ckpt_n_configure_inference = load_with_guidance

from script_utils.entrypoints import generate  # noqa: E402

generate()
