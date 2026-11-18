#!/usr/bin/env python
"""Adapter diagnostics + tree/checkpoint pre-flight (WP-EVAL). Needs torch (CPU is fine), nothing else.

  diag      row norms / contrast / cosines of the ADJ + SS adapter embedding tables of a checkpoint (EMA `shadow.*`
            = what inference uses, and raw `model.*`); the table size (3 = phase-8 tree, 5 = v2 tree) is READ from
            the checkpoint, never assumed. Mirrors A4_adj_rows.py / A5 2.2:
              norm_row<i>        |W[i]|        contrast = |W[1]-W[0]|        cos01 = cos(W[0], W[1])
              (5 rows) cos_1_3 = cos(W[1], W[3])  cos_0_4 = cos(W[0], W[4])  hot_gap = |W[3]-W[1]|  cold_gap = |W[4]-W[0]|
            Row 2 (masked) is exactly 0 by construction. The 'adapter-contrast growth saturating' gate of the
            retrain design (REVIEW 3.2.A) is read off this number across checkpoints (compare mode of score_designs.py).
  preflight refuse to evaluate a checkpoint in a tree whose adapter tables have a different size: the loader
            silently falls back to zero-initialised adapters on a shape mismatch (A5 F1, job 785335), which makes
            every conditioned arm meaningless. The expected size is obtained by constructing the tree's own
            ADJAdapter/SSAdapter (loaded from <repo>/foundry/models/rfd3/src/rfd3/model/layers/ss_adj_adapters.py).
            Exit 0 ok (also for the stock checkpoint, which has no adapter keys), 2 mismatch, 3 unreadable.

  python adapter_diag.py diag --ckpt X.ckpt [--out adapter_diag.json]
  python adapter_diag.py preflight --ckpt X.ckpt --repo /path/to/worktree [--allow-missing]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import eval_common as ec  # noqa: E402

ADJ_KEY = "token_initializer.adj_adapter.embed.weight"
SS_KEY = "token_initializer.ss_adapter.embed.weight"
ADAPTER_SRC = os.path.join("foundry", "models", "rfd3", "src", "rfd3", "model", "layers", "ss_adj_adapters.py")


def _stub_pickle_module():
    """Permissive unpickler: checkpoints pickle training-only classes (configs, lightning objects) that the
    inspection env does not have; unknown classes become empty stubs (same trick as the N002 A4 scripts)."""
    class Dummy:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, s):
            self._state = s

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            try:
                return super().find_class(mod, name)
            except Exception:
                return type(name, (Dummy,), {})

    pm = types.ModuleType("harmonia_stub_pickle")
    pm.__dict__.update({k: getattr(pickle, k) for k in dir(pickle) if not k.startswith("__") and k != "Unpickler"})
    pm.Unpickler = U
    return pm


def load_state(path: str) -> dict:
    """The checkpoint's `model` state dict (keys 'model.*' raw and 'shadow.*' EMA), mmap'd, CPU."""
    import torch

    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False, pickle_module=_stub_pickle_module())
    return ck["model"] if isinstance(ck, dict) and "model" in ck else ck


def table_stats(w) -> dict:
    """Geometry of one adapter embedding table W [S, c]."""
    import torch

    w = w.float()
    S = w.shape[0]
    out: dict = {"n_states": int(S), "norms": [float(w[i].norm()) for i in range(S)]}
    for i in range(S):
        out[f"norm_row{i}"] = out["norms"][i]

    def cos(a, b):
        na, nb = w[a].norm(), w[b].norm()
        return float(torch.nn.functional.cosine_similarity(w[a], w[b], dim=0)) if na > 0 and nb > 0 else float("nan")

    if S >= 2:
        out["contrast"] = float((w[1] - w[0]).norm())
        out["cos01"] = cos(0, 1)
    if S >= 5:
        out["cos_1_3"], out["cos_0_4"] = cos(1, 3), cos(0, 4)
        out["hot_gap"], out["cold_gap"] = float((w[3] - w[1]).norm()), float((w[4] - w[0]).norm())
    return out


def adapter_diagnostics(ckpt: str) -> dict:
    sd = load_state(ckpt)
    res: dict = {"ckpt": ckpt, "has_adapters": False}
    for src in ("shadow", "model"):
        for tag, key in (("adj", ADJ_KEY), ("ss", SS_KEY)):
            k = f"{src}.{key}"
            if k in sd:
                res.setdefault(src, {})[tag] = table_stats(sd[k])
                res["has_adapters"] = True
    return res


def expected_rows(repo: str) -> dict:
    """Adapter table sizes of the tree at `repo`, by constructing its own modules."""
    mod = ec.load_module_from_path(os.path.join(repo, ADAPTER_SRC), "harmonia_tree_adapters")
    return {"adj": int(mod.ADJAdapter(8).embed.weight.shape[0]), "ss": int(mod.SSAdapter(8).embed.weight.shape[0])}


def preflight(ckpt: str, repo: str) -> tuple[int, dict]:
    try:
        sd = load_state(ckpt)
    except Exception as e:  # noqa: BLE001
        return 3, {"ok": False, "message": f"cannot read checkpoint: {type(e).__name__}: {e}"}
    tree = expected_rows(repo)
    ck, bad = {}, []
    for tag, key in (("adj", ADJ_KEY), ("ss", SS_KEY)):
        rows = None
        for src in ("shadow", "model"):
            k = f"{src}.{key}"
            if k in sd:
                r = int(sd[k].shape[0])
                if rows is not None and r != rows:
                    bad.append(f"{tag}: shadow/model tables differ ({rows} vs {r} rows)")
                rows = r
        ck[tag] = rows
        if rows is not None and rows != tree[tag]:
            bad.append(f"{tag}_adapter: checkpoint has {rows} rows, tree at {repo} expects {tree[tag]}")
    info = {"ok": not bad, "ckpt_rows": ck, "tree_rows": tree, "has_adapters": any(v is not None for v in ck.values())}
    if bad:
        info["message"] = ("; ".join(bad) + ". The loader would silently re-initialise the adapter to ZERO (every "
                           "conditioned arm = unconditional). Evaluate with the matching tree (REPO=...) or pad the table.")
        return 2, info
    info["message"] = ("stock-style checkpoint without adapters: the tree's zero-init adapters are used (expected for "
                       "the stock reference)" if not info["has_adapters"] else "adapter table sizes match the tree")
    return 0, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diag")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--out", default=None)
    p = sub.add_parser("preflight")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--repo", default=ec.REPO_DEFAULT)
    a = ap.parse_args(argv)
    if a.cmd == "diag":
        res = adapter_diagnostics(a.ckpt)
        txt = json.dumps(res, indent=1)
        if a.out:
            with open(a.out, "w") as fh:
                fh.write(txt)
        print(txt)
        return 0
    rc, info = preflight(a.ckpt, os.path.abspath(a.repo))
    print(json.dumps(info))
    return rc


if __name__ == "__main__":
    sys.exit(main())
