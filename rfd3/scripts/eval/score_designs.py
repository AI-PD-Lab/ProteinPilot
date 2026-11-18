#!/usr/bin/env python
"""Scorer + compare of the HARMONIA-v2 guardrail harness (WP-EVAL step 3). CPU, minutes per checkpoint.

  score    read ONE eval tar (<tag>.tar.gz written by eval_checkpoint.sbatch), score every design against the spec
           manifest, write  <out>/<tag>.designs.csv  (one row per design),  <out>/<tag>.cells.csv  (per
           target x arm x CFG: mean/CI/median of every metric),  <out>/<tag>.scores.json  (cells + per-design values
           + meta + adapter diagnostics; the input of `compare`)  and  <out>/<tag>.summary.md  (the one-pager).
  compare  checkpoint vs stock vs baseline table from scores.json files, bootstrap CIs of mean differences, with the
           measured noise floor (run-to-run sd of a 32-design ferredoxin half_sep ~1 A: differences < 2 A are noise)
           and the retrain kill criterion (REVIEW 8: CFG-off half_sep > stock-ref + 1.5 A).

  /path/to/env/bin/python -I scripts/eval/score_designs.py score \
      --tar /path/to/work_rfd3/v2/eval/<tag>.tar.gz --out <dir> [--procs 8]
  .../score_designs.py compare --ckpt <dir>/<tag>.scores.json --stock <dir>/stock.scores.json \
      [--baseline <dir>/p0716.scores.json] --out <dir>/compare_<tag>.md

Metric definitions: design_metrics.py. The spec manifest is read from --specs-tar (default: the pack path recorded in
the eval tar's meta.json, else build_specs.SPECS_TAR_DEFAULT).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import re
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import build_specs as bs  # noqa: E402
import design_metrics as dm  # noqa: E402
import eval_common as ec  # noqa: E402

NOISE_FLOOR_A = 2.0       # half_sep differences below this are run-to-run noise (coordinator measurement)
KILL_HALF_SEP_A = 1.5     # REVIEW 8 kill criterion: CFG-off half_sep > stock-ref + 1.5 A

_MANIFEST: dict = {}


def _init_worker(manifest):
    global _MANIFEST
    _MANIFEST = manifest


def score_member(name: str, data: bytes, manifest: dict | None = None) -> dict | None:
    """Score one design file (`designs/cfg<c>/<key>_<batch>_model_<m>.cif.gz`); None for non-design members."""
    manifest = manifest or _MANIFEST
    parts = name.split("/")
    base = parts[-1]
    mt = ec.DESIGN_RE.match(base)
    if not mt:
        return None
    cm = ec.CFG_DIR_RE.search(parts[-2]) if len(parts) >= 2 else None   # designs/cfg1.5/<file>  or legacy <tag>_cfg1.0/<file>
    cfg = float(cm.group("cfg")) if cm else float("nan")
    key = mt.group("key")
    row = {"key": key, "cfg": cfg, "batch": int(mt.group("batch")), "model": int(mt.group("model")),
           "design": base.split(".")[0], "error": ""}
    arm = manifest["arms"].get(key)
    unc = manifest["unconditional"].get(key)
    rec = arm or unc
    if rec is None:
        row.update(target="?", arm="?", error="key not in manifest")
        return row
    row.update(target=rec["target"], arm=rec["arm"])
    try:
        atoms = ec.read_structure(data, base)
        chains = ec.protein_chains(atoms)
        if not chains:
            raise ValueError("no protein chain")
        row["n_chains"] = len(chains)
        target = manifest["targets"].get(rec["target"]) if arm else None
        row.update(dm.design_row(chains[0], target, arm))
    except Exception as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return row


def _score_star(args):
    return score_member(*args)


def iter_tar_designs(tar_path: str, layout: str = "designs"):
    """Yield (name, bytes) of every design file of the eval tar, streaming. layout 'designs' = this harness
    (`<tag>/designs/cfg<c>/<key>_<b>_model_<m>.cif.gz`); 'flat' = any design-named member (legacy N002 tars,
    `<tag>_cfg1.0/<key>_<b>_model_<m>.cif.gz`; the CFG is read from the directory name)."""
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf:
            if not m.isfile() or not ec.DESIGN_RE.match(os.path.basename(m.name)):
                continue
            if layout == "designs" and "/designs/" not in m.name:
                continue
            yield m.name, tf.extractfile(m).read()


def read_tar_extras(tar_path: str) -> dict:
    """meta.json / adapter_diag.json / plan.json of the eval tar (any may be missing)."""
    out = {}
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf:
            base = os.path.basename(m.name)
            if m.isfile() and base in ("meta.json", "adapter_diag.json", "plan.json") and "/designs/" not in m.name:
                out[base[:-5]] = json.load(tf.extractfile(m))
    return out


def score_tar(tar_path: str, manifest: dict, procs: int = 1, layout: str = "designs") -> list[dict]:
    rows = []
    gen = ((n, d) for n, d in iter_tar_designs(tar_path, layout))
    if procs > 1:
        with mp.get_context("fork").Pool(procs, initializer=_init_worker, initargs=(manifest,)) as pool:
            for r in pool.imap(_score_star, ((n, d, None) for n, d in gen), chunksize=8):
                if r:
                    rows.append(r)
    else:
        for n, d in gen:
            r = score_member(n, d, manifest)
            if r:
                rows.append(r)
    return rows


# --------------------------------------------------------------------------- aggregation
def _cell_name(target, arm, cfg):
    return f"{target}|{arm}|{cfg:.1f}"


def aggregate(rows: list[dict], boot: int = 2000) -> dict:
    """cells: name -> {target, arm, cfg, n, n_err, n_len_bad, metrics{m:{mean,lo,hi,median,n}}, values{m:[...]}}"""
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["target"], r["arm"], r["cfg"]), []).append(r)
    cells = {}
    for (target, arm, cfg), rs in sorted(groups.items()):
        good = [r for r in rs if not r["error"]]
        cell = {"target": target, "arm": arm, "cfg": cfg, "n": len(rs), "n_err": len(rs) - len(good),
                "n_len_bad": sum(1 for r in good if not r.get("len_ok", True)), "metrics": {}, "values": {}}
        use = [r for r in good if r.get("len_ok", True)]
        for m in dm.CELL_METRICS:
            vals = [float(r[m]) for r in use if m in r and r[m] is not None]
            vals = [v for v in vals if math.isfinite(v)]
            if not vals:
                continue
            mean, lo, hi = ec.bootstrap_ci(vals, n_boot=boot, seed=1)
            cell["metrics"][m] = {"mean": mean, "lo": lo, "hi": hi, "median": float(np.median(vals)), "n": len(vals)}
            cell["values"][m] = [round(v, 4) for v in vals]
        cells[_cell_name(target, arm, cfg)] = cell
    return cells


def write_designs_csv(rows: list[dict], path: str):
    cols = ["target", "arm", "cfg", "design", "key", "batch", "model", "error", "n_chains", "len_ok"] + dm.CELL_METRICS + ["N"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["target"], r["arm"], r["cfg"], r["batch"], r["model"])):
            w.writerow({c: ("" if r.get(c) is None else (f"{r[c]:.5g}" if isinstance(r.get(c), float) else r.get(c, ""))) for c in cols})


def write_cells_csv(cells: dict, path: str):
    cols = ["target", "arm", "cfg", "n", "n_err", "n_len_bad"]
    for m in dm.CELL_METRICS:
        cols += [f"{m}_mean", f"{m}_lo", f"{m}_hi", f"{m}_median"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for c in cells.values():
            row = {k: c[k] for k in ("target", "arm", "cfg", "n", "n_err", "n_len_bad")}
            for m, s in c["metrics"].items():
                row.update({f"{m}_mean": f"{s['mean']:.5g}", f"{m}_lo": f"{s['lo']:.5g}", f"{m}_hi": f"{s['hi']:.5g}",
                            f"{m}_median": f"{s['median']:.5g}"})
            w.writerow(row)


# --------------------------------------------------------------------------- markdown
def _ci(cell, m, nd=2):
    s = cell["metrics"].get(m) if cell else None
    if not s:
        return "-"
    return f"{ec.fmt(s['mean'], nd)} [{ec.fmt(s['lo'], nd)},{ec.fmt(s['hi'], nd)}]"


def _row_sort(c):
    return (c["target"], ["full", "sparse8", "dense0_true", "ss_only", "unconditional"].index(c["arm"]) if c["arm"] in
            ("full", "sparse8", "dense0_true", "ss_only", "unconditional") else 9, c["cfg"])


def summary_md(tag: str, cells: dict, manifest: dict, meta: dict | None, adapter: dict | None, stock_cells: dict | None = None) -> str:
    L = [f"# HARMONIA-v2 guardrail summary: {tag}", ""]
    if meta:
        L.append(f"ckpt `{meta.get('ckpt_path', '?')}` | seed base {meta.get('seed', '?')} | n/key {meta.get('n_per_key', '?')} | "
                 f"steps {meta.get('num_timesteps', '?')} | git {meta.get('git_commit', '?')[:10]} | job {meta.get('slurm_job_id', '?')} | "
                 f"valid {meta.get('valid', '?')}")
    L.append(f"specs v{manifest['spec_version']} ({manifest['dialect']}); metric defs: scripts/eval/design_metrics.py; "
             f"mean [95% bootstrap CI]; native = reference structure measured by the same code. Differences < "
             f"{NOISE_FLOOR_A:.0f} A in half_sep are run-to-run noise (n=32).")
    nerr = sum(c["n_err"] + c["n_len_bad"] for c in cells.values())
    if nerr:
        L.append(f"**WARNING: {nerr} designs unreadable / wrong length and excluded** (see cells.csv n_err, n_len_bad).")
    L += ["", "## Conditioned arms", "",
          "| target | arm | CFG | n | half_sep (native) | recovery | violation | extent_ratio | Rg/N^0.4 | lr_cb | ss_acc | breaks | heavy clash |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted((c for c in cells.values() if c["arm"] != "unconditional"), key=_row_sort):
        nat = manifest["targets"].get(c["target"], {}).get("native", {})
        L.append(f"| {c['target']} | {c['arm']} | {c['cfg']:.1f} | {c['n']} | {_ci(c, 'half_sep', 1)} ({ec.fmt(nat.get('half_sep'), 1)}) | "
                 f"{_ci(c, 'recovery')} | {_ci(c, 'violation')} | {_ci(c, 'extent_ratio')} | {_ci(c, 'rg_n04', 3)} ({ec.fmt(nat.get('rg_n04'), 3)}) | "
                 f"{_ci(c, 'lr_cb')} ({ec.fmt(nat.get('lr_cb'))}) | {_ci(c, 'ss_acc')} | {_ci(c, 'ca_breaks')} | {_ci(c, 'heavy_clash')} |")
    # CFG effect on half_sep
    L += ["", "## CFG effect (half_sep, A; mean difference vs CFG 1.0 with 95% CI; negative = tighter)", "",
          "| target | arm | CFG 1.5 - 1.0 | CFG 2.0 - 1.0 |", "|---|---|---|---|"]
    by = {}
    for c in cells.values():
        by.setdefault((c["target"], c["arm"]), {})[c["cfg"]] = c
    for (t, a), d in sorted(by.items()):
        if a == "unconditional" or 1.0 not in d:
            continue
        outs = []
        for cfg in (1.5, 2.0):
            if cfg in d and "half_sep" in d[cfg]["values"] and "half_sep" in d[1.0]["values"]:
                m, lo, hi = ec.diff_ci(d[cfg]["values"]["half_sep"], d[1.0]["values"]["half_sep"], seed=2)
                outs.append(f"{ec.fmt(m, 1)} [{ec.fmt(lo, 1)},{ec.fmt(hi, 1)}]")
            else:
                outs.append("-")
        L.append(f"| {t} | {a} | {outs[0]} | {outs[1]} |")
    # unconditional panel
    unc = sorted((c for c in cells.values() if c["arm"] == "unconditional"), key=lambda c: (c["target"], c["cfg"]))
    if unc:
        L += ["", "## Unconditional panel (no ss/adj input)", "",
              "| L | n | Rg/N^0.4 | lr_cb | split | helix | strand | breaks | CA clash | heavy clash |" + (" d Rg/N^0.4 vs stock | d lr_cb vs stock |" if stock_cells else ""),
              "|---|---|---|---|---|---|---|---|---|---|" + ("---|---|" if stock_cells else "")]
        for c in unc:
            ln = manifest["unconditional"][f"{c['target']}__unconditional"]["length"]
            extra = ""
            if stock_cells:
                s = stock_cells.get(_cell_name(c["target"], c["arm"], c["cfg"]))
                cols = []
                for m in ("rg_n04", "lr_cb"):
                    if s and m in s["values"] and m in c["values"]:
                        d, lo, hi = ec.diff_ci(c["values"][m], s["values"][m], seed=3)
                        cols.append(f"{ec.fmt(d, 3)} [{ec.fmt(lo, 3)},{ec.fmt(hi, 3)}]")
                    else:
                        cols.append("-")
                extra = f" {cols[0]} | {cols[1]} |"
            L.append(f"| {ln} | {c['n']} | {_ci(c, 'rg_n04', 3)} | {_ci(c, 'lr_cb')} | {_ci(c, 'split')} | {_ci(c, 'helix')} | "
                     f"{_ci(c, 'strand')} | {_ci(c, 'ca_breaks')} | {_ci(c, 'ca_clash')} | {_ci(c, 'heavy_clash')} |{extra}")
    if adapter and adapter.get("has_adapters"):
        L += ["", "## Adapter diagnostics (EMA `shadow.*`, what inference uses)", "",
              "| table | states | row norms | contrast |W1-W0| | cos(W0,W1) | extra |", "|---|---|---|---|---|---|"]
        for tag_, st in (adapter.get("shadow") or {}).items():
            ex = ""
            if "cos_1_3" in st:
                ex = f"cos(W1,W3) {ec.fmt(st['cos_1_3'])}, cos(W0,W4) {ec.fmt(st['cos_0_4'])}, |W3-W1| {ec.fmt(st['hot_gap'])}, |W4-W0| {ec.fmt(st['cold_gap'])}"
            L.append(f"| {tag_} | {st['n_states']} | {', '.join(ec.fmt(x) for x in st['norms'])} | {ec.fmt(st.get('contrast'))} | {ec.fmt(st.get('cos01'))} | {ex} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- compare
def _status(delta, lo, hi, noise=NOISE_FLOOR_A):
    if not math.isfinite(delta):
        return "-"
    if lo > 0 and delta > noise:
        return "WORSE"
    if hi < 0 and delta < -noise:
        return "better"
    return "~noise"


def compare_md(sets: dict, noise: float = NOISE_FLOOR_A, kill: float = KILL_HALF_SEP_A) -> str:
    """sets: {'ckpt': scores, 'stock': scores|None, 'baseline': scores|None}; checkpoint vs stock vs baseline."""
    ck = sets["ckpt"]
    stock, base = sets.get("stock"), sets.get("baseline")
    L = [f"# Compare: {ck['meta'].get('tag', 'ckpt')}"
         f" vs stock {stock['meta'].get('tag') if stock else '(none)'} vs baseline {base['meta'].get('tag') if base else '(none)'}", "",
         f"half_sep: status WORSE/better only when |mean diff| > {noise:.1f} A (noise floor) AND the 95% bootstrap CI of the difference excludes 0. "
         f"KILL? = CFG 1.0 half_sep of the checkpoint exceeds the stock reference by > {kill:.1f} A with CI > 0 (REVIEW 8).", "",
         "| target | arm | CFG | half_sep ckpt | stock | baseline | d(ckpt-stock) [CI] | status | KILL? | recovery ckpt/stock/base | violation ckpt/stock/base | extent_ratio ckpt/stock/base |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def trio(m, nd=2):
        out = []
        for s in (ck, stock, base):
            cell = s["cells"].get(key) if s else None
            out.append(ec.fmt(cell["metrics"][m]["mean"], nd) if cell and m in cell["metrics"] else "-")
        return "/".join(out)

    kills = []
    for key, c in sorted(ck["cells"].items(), key=lambda kv: _row_sort(kv[1])):
        if c["arm"] == "unconditional":
            continue
        hs = [s["cells"].get(key) if s else None for s in (ck, stock, base)]
        h = [ec.fmt(x["metrics"]["half_sep"]["mean"], 1) if x and "half_sep" in x["metrics"] else "-" for x in hs]
        d = lo = hi = float("nan")
        if hs[1] and "half_sep" in hs[1]["values"] and "half_sep" in c["values"]:
            d, lo, hi = ec.diff_ci(c["values"]["half_sep"], hs[1]["values"]["half_sep"], seed=5)
        st = _status(d, lo, hi, noise)
        kl = "KILL" if (c["cfg"] == 1.0 and math.isfinite(d) and d > kill and lo > 0) else ""
        if kl:
            kills.append(f"{c['target']}/{c['arm']}")
        L.append(f"| {c['target']} | {c['arm']} | {c['cfg']:.1f} | {h[0]} | {h[1]} | {h[2]} | {ec.fmt(d, 1)} [{ec.fmt(lo, 1)},{ec.fmt(hi, 1)}] | {st} | {kl} | "
                 f"{trio('recovery')} | {trio('violation')} | {trio('extent_ratio')} |")
    L += ["", "## Unconditional panel (ckpt vs stock vs baseline; mean Rg/N^0.4, lr_cb)", "",
          "| L | Rg/N^0.4 ckpt/stock/base | d(ckpt-stock) [CI] | lr_cb ckpt/stock/base | d(ckpt-stock) [CI] |", "|---|---|---|---|---|"]
    for key, c in sorted(ck["cells"].items()):
        if c["arm"] != "unconditional":
            continue
        row = [str(c["target"])]
        for m, nd in (("rg_n04", 3), ("lr_cb", 2)):
            vals = [(s["cells"].get(key) if s else None) for s in (ck, stock, base)]
            row.append("/".join(ec.fmt(v["metrics"][m]["mean"], nd) if v and m in v["metrics"] else "-" for v in vals))
            sv = vals[1]
            if sv and m in sv["values"] and m in c["values"]:
                dd, l2, h2 = ec.diff_ci(c["values"][m], sv["values"][m], seed=6)
                row.append(f"{ec.fmt(dd, 3)} [{ec.fmt(l2, 3)},{ec.fmt(h2, 3)}]")
            else:
                row.append("-")
        L.append("| " + " | ".join(row) + " |")
    L += ["", "## Adapter contrast (|W1-W0|, EMA)", ""]
    for name, s in (("ckpt", ck), ("stock", stock), ("baseline", base)):
        ad = (s or {}).get("adapter") or {}
        st = (ad.get("shadow") or {}).get("adj")
        L.append(f"- {name}: " + (f"{st['n_states']} states, contrast {ec.fmt(st.get('contrast'))}, cos01 {ec.fmt(st.get('cos01'))}, norms {', '.join(ec.fmt(x) for x in st['norms'])}" if st else "no adapter table"))
    L.append("")
    L.append("KILL flags: " + (", ".join(kills) if kills else "none"))
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- CLI
def cmd_score(a):
    extras = read_tar_extras(a.tar)
    meta = extras.get("meta", {})
    specs_tar = a.specs_tar or meta.get("specs_tar") or ec.SPECS_TAR_DEFAULT
    manifest = ec.read_json(a.manifest) if a.manifest else bs.read_manifest_from_tar(specs_tar)
    tag = a.tag or meta.get("tag") or re.sub(r"\.tar(\.gz)?$", "", os.path.basename(a.tar))
    rows = score_tar(a.tar, manifest, a.procs, a.layout)
    if not rows:
        raise SystemExit(f"no designs found in {a.tar}")
    cells = aggregate(rows)
    os.makedirs(a.out, exist_ok=True)
    meta = dict(meta, tag=tag)
    adapter = extras.get("adapter_diag")
    if adapter is None and a.ckpt:
        import adapter_diag as ad

        adapter = ad.adapter_diagnostics(a.ckpt)
    stock_cells = ec.read_json(a.stock)["cells"] if a.stock else None
    base = os.path.join(a.out, tag)
    write_designs_csv(rows, base + ".designs.csv")
    write_cells_csv(cells, base + ".cells.csv")
    ec.write_json(base + ".scores.json", {"meta": meta, "cells": cells, "adapter": adapter, "n_designs": len(rows)}, indent=None)
    md = summary_md(tag, cells, manifest, meta, adapter, stock_cells)
    with open(base + ".summary.md", "w") as fh:
        fh.write(md)
    print(md)
    print(f"wrote {base}.designs.csv .cells.csv .scores.json .summary.md  ({len(rows)} designs)")


def cmd_compare(a):
    sets = {"ckpt": ec.read_json(a.ckpt), "stock": ec.read_json(a.stock) if a.stock else None,
            "baseline": ec.read_json(a.baseline) if a.baseline else None}
    md = compare_md(sets, a.noise_floor, a.kill)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(md)
    print(md)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--tar", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--specs-tar", default=None)
    s.add_argument("--manifest", default=None, help="manifest.json instead of reading it from the spec pack")
    s.add_argument("--tag", default=None)
    s.add_argument("--procs", type=int, default=1)
    s.add_argument("--layout", choices=("designs", "flat"), default="designs",
                   help="'flat' scores legacy N002-style tars (use with --manifest <legacy manifest>)")
    s.add_argument("--ckpt", default=None, help="compute adapter diagnostics from this checkpoint if the tar has none")
    s.add_argument("--stock", default=None, help="stock scores.json: adds delta-vs-stock columns to the unconditional panel")
    c = sub.add_parser("compare")
    c.add_argument("--ckpt", required=True, help="scores.json of the checkpoint under test")
    c.add_argument("--stock", default=None)
    c.add_argument("--baseline", default=None)
    c.add_argument("--noise-floor", type=float, default=NOISE_FLOOR_A)
    c.add_argument("--kill", type=float, default=KILL_HALF_SEP_A)
    c.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    (cmd_score if a.cmd == "score" else cmd_compare)(a)


if __name__ == "__main__":
    main()
