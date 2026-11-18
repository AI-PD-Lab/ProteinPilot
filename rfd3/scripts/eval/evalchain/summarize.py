#!/usr/bin/env python
"""summarize.py - evalchain CPU step: one report per checkpoint evaluation, every pre-registered kill criterion PASS/FAIL.

Reads the chain record written by evalchain.sh (<results item>/eval_<TAG>/chain.json), then
  1. scores every tar of the chain with the N007 kit score_all.sh into the SHARED cache (v2/eval/cache: designs/tm csv), the
     N019 baseline tags included - baselines are always re-scored by the same code, never copied numbers;
  2. runs the N029 score_anatomy.py (imported) on specs_v2 at every CFG with --compare v2_E2_e0596_h02r1cen2m0 (seed 1; the
     --full seed tags against the matching baseline seed tags), and on the held-out beta pack at CFG 1.5 with --compare
     v2_E2_e0596_heldout_s1 (manifest + refs from the N029 probe pack);
  3. guardrails per goal arm x CFG from the designs csv (kit split3 nat-relaxed SPLIT rule, breaks, clash > 4);
  4. designability pooled over the goal arms per CFG (Wilson CI) from designability.sbatch's json;
  5. the Rossmann stress panel (ross_panel.sbatch tar) with the kit ross_metrics.py rule, vs the N019 baseline tars;
  6. sacct GPU-h of every stage job;
  7. criteria.json evaluated -> summary.json + summary.md (+ anatomy/guardrail csv) in the results dir.
Re-runnable at any time (cache hits are free): python summarize.py --chain <dir>/chain.json
Plain python (esmfold2 env), NOT -I: score_anatomy sets sys.path itself. Exit 0 once the report is written (the verdict is in it).
Importers (tests) must register the module: sys.modules["summarize"] = m before spec.loader.exec_module(m), because the Rossmann
scorer pickles its fork-Pool worker by module + name. Unit tests: test_summarize.py."""
import argparse
import collections
import csv
import glob
import importlib.util
import json
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np

W = "/path/to/work_rfd3"
REPO = W + "/harmonia_study"
RESULTS = REPO + "/.agent-research/results"
EVAL_DIR = W + "/v2/eval"
HERE = os.path.dirname(os.path.abspath(__file__))
CRIT = json.load(open(os.path.join(HERE, "criteria.json")))
KIT = sorted(glob.glob(RESULTS + "/N007-*/kit"))[0]
PROBE = sorted(glob.glob(RESULTS + "/N029-*/probe"))[0]
HELDOUT_MANIFEST = PROBE + "/specs_beta_heldout_manifest.json"
HELDOUT_REFS = PROBE + "/specs_beta_heldout.tar.gz"
GOAL_ARMS = tuple(CRIT["goal_arms"])
ERR_OK = ("", "0", "False", "None")
T_SPLIT, T_LR, T_RG = 0.67, 0.92, 2.15     # kit ross_metrics.py: strict fold-independent SPLIT rule (natural p5/p5/p95)
SD_RECIPE = float(CRIT["sd_recipe"])

sys.path.insert(0, REPO + "/scripts/eval")
sys.path.insert(0, KIT)
import eval_common as ec  # noqa: E402
import design_metrics as dm  # noqa: E402
import split3  # noqa: E402  (kit: nat-relaxed universal split rule; loads the kit manifest = specs_v2 manifest)


def _load_sa():
    spec = importlib.util.spec_from_file_location("score_anatomy", PROBE + "/score_anatomy.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["score_anatomy"] = m        # registered: its fork-Pool worker (_geom) is pickled by module + name
    spec.loader.exec_module(m)
    return m


sa = _load_sa()


# ----------------------------------------------------------------------------- small helpers
def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def fin(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def fmt(v, nd=3):
    return f"{v:.{nd}f}" if fin(v) else "-"


def wilson(k, n, z=1.96):
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def tar_meta(tar_path, member=None):
    """meta.json of a harness tar (<TAG>/meta.json) or of a ross panel tar (meta.json); None if the tar is missing/unreadable"""
    if not os.path.exists(tar_path):
        return None
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            names = [m.name for m in tf.getmembers() if m.isfile() and os.path.basename(m.name) == "meta.json"]
            if member:
                names = [n for n in names if n == member] or names
            if not names:
                return {"_error": "no meta.json in tar"}
            return json.load(tf.extractfile(names[0]))
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {e}"}


# ----------------------------------------------------------------------------- 1+2. anatomy (score_anatomy.py imported)
def run_anatomy(tags, base, cfg, cache, out_csv, manifest=None, refs=None, procs=16):
    """score_anatomy.main on tags (+ base via --compare); scores missing designs.csv with the kit; returns the tables dict or None"""
    argv = ["--cache", cache, "--cfg", str(cfg), "--out", out_csv, "--procs", str(procs)]
    if manifest:
        argv += ["--manifest", manifest]
    if refs:
        argv += ["--refs", refs]
    if base:
        argv += ["--compare", base]
    argv += list(tags)
    try:
        return sa.main(argv)
    except SystemExit as e:
        print(f"[summarize] score_anatomy exited: {e}", flush=True)
        return None


def overall(tables, tag):
    return tables[tag][("ALL", "ALL")] if tables and tag in tables else None


def sfar_stats(tables, tag, base=None):
    s = overall(tables, tag)
    if s is None:
        return None
    out = {"n": int(s["n"]), "pass": float(s["pass"]), "lo": float(s["pass_lo"]), "hi": float(s["pass_hi"]),
           "rec_lt09": s["rec_lt09"], "sep_gt115": s["sep_gt115"], "breaks": s["breaks"], "clash_gt4": s["clash_gt4"],
           "pass_native": s["pass_native"]}
    b = overall(tables, base) if base else None
    if b is not None:
        d, lo, hi = ec.diff_ci(s["_pass_vec"], b["_pass_vec"], n_boot=2000, seed=0)
        out.update(base_tag=base, base_pass=float(b["pass"]), base_n=int(b["n"]), delta=d, delta_lo=lo, delta_hi=hi,
                   sig=bool(np.isfinite(lo) and (lo > 0 or hi < 0)), flag=bool(abs(d) > 2 * SD_RECIPE))
    return out


def sandwich_stats(tables, tag, base=None):
    """sandwich_1tit ALL row (goal arms pooled): pass, hairpin p12/L12, p14/L14; hairpin12_cut vs base"""
    if not tables or tag not in tables or ("sandwich_1tit", "ALL") not in tables[tag]:
        return None
    s = tables[tag][("sandwich_1tit", "ALL")]
    out = {k: (float(s[k]) if fin(s.get(k)) else float("nan")) for k in ("pass", "pass_lo", "pass_hi", "p12", "L12", "p14", "L14", "p05")}
    out["n"] = int(s["n"])
    if base and base in tables and ("sandwich_1tit", "ALL") in tables[base]:
        b = tables[base][("sandwich_1tit", "ALL")]
        out.update(base_tag=base, base_pass=float(b["pass"]), base_p12=float(b["p12"]), base_p14=float(b["p14"]), base_L12=float(b["L12"]))
        out["hairpin12_cut"] = 1.0 - out["p12"] / b["p12"] if fin(b["p12"]) and b["p12"] > 0 else float("nan")
        d, lo, hi = ec.diff_ci(s["_pass_vec"], b["_pass_vec"], n_boot=2000, seed=0)
        out.update(delta=d, delta_lo=lo, delta_hi=hi)
    return out


# ----------------------------------------------------------------------------- 3. guardrails from the designs csv
def guardrails(cache, tag, cfgs):
    """{(cfg, arm|'ALL'): n, split, split_cut/lr/rg, breaks, clash_mean, clash_gt4} for the goal arms (kit split3 rule)"""
    path = f"{cache}/{tag}.designs.csv"
    if not os.path.exists(path):
        return None
    rows = [r for r in csv.DictReader(open(path)) if r["error"] in ERR_OK and r["arm"] in GOAL_ARMS]
    out = {}
    for cfg in cfgs:
        for arm in GOAL_ARMS + ("ALL",):
            R = [r for r in rows if (arm == "ALL" or r["arm"] == arm) and abs(f(r["cfg"]) - cfg) < 1e-6]
            if not R:
                continue
            F = [split3.fl(r) for r in R]
            hc = [f(r["heavy_clash"]) for r in R]
            out[(cfg, arm)] = dict(n=len(R), split=float(np.mean([any(x) for x in F])),
                                   split_cut=float(np.mean([x[0] for x in F])), split_lr=float(np.mean([x[1] for x in F])),
                                   split_rg=float(np.mean([x[2] for x in F])), breaks=float(np.nanmean([f(r["ca_breaks"]) for r in R])),
                                   clash_mean=float(np.nanmean(hc)), clash_gt4=float(np.mean([h > 4 for h in hc])),
                                   rec_ge09=float(np.mean([f(r["recovery"]) >= 0.9 for r in R])))
    return out


# ----------------------------------------------------------------------------- 4. designability pooling
def designability_pool(path):
    """{cfg: {k, n, p, lo, hi, on_fold_k, joint_k, joint_p}} over the goal-arm cells of a designability json"""
    if not path or not os.path.exists(path):
        return None
    cells = json.load(open(path))["cells"].values()
    out = {}
    for cfg in sorted({float(c["cfg"]) for c in cells}):
        C = [c for c in cells if abs(float(c["cfg"]) - cfg) < 1e-6 and c["arm"] in GOAL_ARMS]
        k = sum(c["designable"]["k"] for c in C)
        n = sum(c["designable"]["n"] for c in C)
        kj = sum(c["designable_on_fold"]["k"] for c in C)
        k70 = sum(c["designable70"]["k"] for c in C)
        p, lo, hi = wilson(k, n)
        out[cfg] = dict(k=k, n=n, p=p, lo=lo, hi=hi, n_cells=len(C), joint_k=kj, joint_p=(kj / n if n else float("nan")),
                        p70=(k70 / n if n else float("nan")))
    return out


# ----------------------------------------------------------------------------- 5. Rossmann stress panel
def _ross_one(job):
    name, data = job
    try:
        ch = ec.protein_chains(ec.read_structure(data, os.path.basename(name)), min_len=40)[0]
        r = dm.design_row(ch)
    except Exception:  # noqa: BLE001
        return name, None
    return name, {k: float(r[k]) for k in ("split", "lr_cb", "rg_n04", "helix", "strand", "heavy_clash", "ca_breaks")}


def ross_scores(tar_path, procs=16):
    """{cell: {case: {n, split_pct, cut_pct, lr_pct, rg_pct, rg_n04, lr_cb, cutdens, helix, strand, clash, breaks}}} + '_meta'"""
    if not tar_path or not os.path.exists(tar_path):
        return None
    jobs, meta = [], None
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            n = m.name[2:] if m.name.startswith("./") else m.name
            if os.path.basename(n) == "meta.json":
                meta = json.load(tf.extractfile(m))
            elif n.endswith(".cif.gz") and "/" in n:
                jobs.append((n, tf.extractfile(m).read()))
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(min(procs, max(1, len(jobs)))) as pool:
        res = dict(pool.imap_unordered(_ross_one, jobs, chunksize=8))
    by = collections.defaultdict(list)
    for n, r in res.items():
        if r is None:
            continue
        cell = n.split("/")[0]
        case = re.sub(r"_\d+_model_\d+.*$", "", os.path.basename(n))
        by[(cell, case)].append(r)
    out = {}
    for (cell, case), L in sorted(by.items()):
        sp = [int(x["split"] < T_SPLIT or x["lr_cb"] < T_LR or x["rg_n04"] > T_RG) for x in L]
        out.setdefault(cell, {})[case] = dict(
            n=len(L), split_pct=100 * float(np.mean(sp)), cut_pct=100 * float(np.mean([x["split"] < T_SPLIT for x in L])),
            lr_pct=100 * float(np.mean([x["lr_cb"] < T_LR for x in L])), rg_pct=100 * float(np.mean([x["rg_n04"] > T_RG for x in L])),
            rg_n04=float(np.mean([x["rg_n04"] for x in L])), lr_cb=float(np.mean([x["lr_cb"] for x in L])),
            cutdens=float(np.mean([x["split"] for x in L])), helix=float(np.mean([x["helix"] for x in L])),
            strand=float(np.mean([x["strand"] for x in L])), clash=float(np.mean([x["heavy_clash"] for x in L])),
            breaks=float(np.mean([x["ca_breaks"] for x in L])))
    out["_meta"] = meta
    out["_n_failed"] = sum(1 for r in res.values() if r is None)
    return out


def ross_case_mean(scores, case="full_rossmann_like", key="split_pct"):
    if not scores:
        return float("nan")
    v = [c[case][key] for cell, c in scores.items() if not cell.startswith("_") and case in c]
    return float(np.mean(v)) if v else float("nan")


# ----------------------------------------------------------------------------- 6. sacct
def parse_sacct(text):
    out = {}
    for line in text.strip().splitlines():
        p = line.split("|")
        if len(p) < 6:
            continue
        jid, name, el, tres, state, part = p[:6]
        m = re.search(r"gres/gpu=(\d+)", tres)
        gpus = int(m.group(1)) if m else 0
        el = int(el) if el.isdigit() else 0
        out[jid] = dict(name=name, elapsed_s=el, gpus=gpus, gpuh=gpus * el / 3600.0, state=state, partition=part)
    return out


def sacct_info(job_ids):
    ids = [str(j) for j in job_ids if j and str(j).isdigit()]
    if not ids:
        return {}
    try:
        r = subprocess.run(["sacct", "-j", ",".join(ids), "-X", "-P", "-n", "--format=JobID,JobName,ElapsedRaw,AllocTRES,State,Partition"],
                           capture_output=True, text=True, timeout=120)
        return parse_sacct(r.stdout) if r.returncode == 0 else {}
    except Exception as e:  # noqa: BLE001
        print(f"[summarize] sacct unavailable: {e}")
        return {}


# ----------------------------------------------------------------------------- 7. criteria
def _cmp(op, v, thr):
    return {">=": v >= thr, "<=": v <= thr, ">": v > thr, "<": v < thr, "==": v == thr}[op]


def evaluate(metrics, full=False):
    """criteria.json -> rows [{id, kind, metric, value, threshold, status, note}]; status PASS | FAIL | UNVERIFIED | SKIPPED"""
    rows = []
    M = metrics
    for c in CRIT["criteria"]:
        cid, op, thr = c["id"], c["op"], c["threshold"]
        val, note = None, ""
        if cid == "valid":
            V = M.get("validity") or {}
            bad = [k for k, v in V.items() if v.get("exists") and v.get("valid") is False]      # produced but invalid = evidence
            missing = [k for k, v in V.items() if not v.get("exists")]                           # no output (job failed/pending) = unverified
            if bad:
                val, note = False, "invalid output: " + ", ".join(f"{k} ({V[k].get('note')})" for k in bad)
            elif missing:
                val, note = None, "missing outputs (rerun the stage): " + ", ".join(f"{k} (job {V[k].get('job_state') or '?'})" for k in missing)
            elif V:
                val, note = True, f"{len(V)} stage outputs valid"
        elif cid == "sfar15":
            s = (M.get("sfar") or {}).get(1.5)
            if s:
                val = s["pass"]
                note = f"n {s['n']} CI [{fmt(s['lo'])},{fmt(s['hi'])}]; baseline {fmt(s.get('base_pass'))} delta {fmt(s.get('delta'))} [{fmt(s.get('delta_lo'))},{fmt(s.get('delta_hi'))}]{' sig' if s.get('sig') else ''}"
        elif cid in ("sandwich_pass", "sandwich_target"):
            s = (M.get("sandwich") or {}).get(1.5)
            if s:
                val = s["pass"]
                note = f"n {s['n']} CI [{fmt(s['pass_lo'], 2)},{fmt(s['pass_hi'], 2)}]; baseline {fmt(s.get('base_pass'))}"
        elif cid == "hairpin12_cut":
            s = (M.get("sandwich") or {}).get(1.5)
            if s and fin(s.get("hairpin12_cut")):
                val = s["hairpin12_cut"]
                note = f"p12 {fmt(s['p12'])} vs baseline {fmt(s.get('base_p12'))} (L12 {fmt(s['L12'], 2)} vs {fmt(s.get('base_L12'), 2)})"
            elif s:
                note = "baseline p12 unavailable"
        elif cid == "paired14":
            s = (M.get("sandwich") or {}).get(1.5)
            if s:
                val = s["p14"]
                note = f"baseline {fmt(s.get('base_p14'))}; L14 {fmt(s['L14'], 2)}"
        elif cid == "split":
            G = M.get("guardrails")
            if G:
                cells = [(v["split"], f"{a}@{cfg:.1f}") for (cfg, a), v in G.items() if a != "ALL"]
                val, worst = max(cells)
                note = f"worst {worst}; " + " ".join(f"{a}@{cfg:.1f}={100 * v['split']:.1f}%" for (cfg, a), v in sorted(G.items()) if a != "ALL")
        elif cid == "breaks":
            G = M.get("guardrails")
            if G:
                cells = [(v["breaks"], f"ALL@{cfg:.1f}") for (cfg, a), v in G.items() if a == "ALL"]
                val, worst = max(cells)
                note = f"worst {worst}; " + " ".join(f"CFG {cfg:.1f}={v['breaks']:.4f}" for (cfg, a), v in sorted(G.items()) if a == "ALL")
        elif cid == "designability":
            D = M.get("designability")
            if D and 2.0 in D:
                d = D[2.0]
                val = d["p"]
                note = f"{d['k']}/{d['n']} CI [{fmt(d['lo'], 2)},{fmt(d['hi'], 2)}] ({d['n_cells']} cells); designable&on-fold {fmt(d['joint_p'], 2)}" + \
                       (f"; CFG 1.5: {fmt(D[1.5]['p'], 2)} ({D[1.5]['k']}/{D[1.5]['n']})" if 1.5 in D else "")
            elif D:
                note = "no CFG 2.0 cells in the designability json"
        elif cid == "heldout":
            H = M.get("heldout")
            if H and fin(H.get("delta")):
                d, lo, hi, sig = H["delta"], H["delta_lo"], H["delta_hi"], H["sig"]
                worse = d < thr and sig
                val = d
                note = f"tag {fmt(H['pass'])} (n {H['n']}) vs baseline {fmt(H['base_pass'])} (n {H['base_n']}); CI [{fmt(lo)},{fmt(hi)}]{' sig' if sig else ''}" + \
                       ("; IMPROVED" if d > -thr and sig else "")
                rows.append(dict(id=cid, kind=c["kind"], metric=c["metric"], value=val, threshold=thr,
                                 status="FAIL" if worse else "PASS", note=note, source=c["source"]))
                continue
            elif H and fin(H.get("pass")):
                note = f"tag SFAR@1.5 {fmt(H['pass'])} (n {H['n']}); baseline tag {H.get('base_tag')} not scored yet - re-run summarize when its tar lands"
            elif H and H.get("error"):
                note = H["error"]
        elif cid == "ross":
            R = M.get("ross")
            if R and fin(R.get("full_split_pct")):
                val = R["full_split_pct"]
                note = f"cells {R.get('cells')}; adjonly {fmt(R.get('adjonly_split_pct'), 1)}% ssonly {fmt(R.get('ssonly_split_pct'), 1)}%; baseline full {fmt(R.get('base_full_split_pct'), 1)}% (N019 recorded 21.1)"
        elif cid == "sfar15_3seed":
            if not full:
                rows.append(dict(id=cid, kind=c["kind"], metric=c["metric"], value=None, threshold=thr, status="SKIPPED",
                                 note="screen only (no --full)", source=c["source"]))
                continue
            S = M.get("sfar_seeds") or {}
            vals = [v["pass"] for v in S.values() if v and fin(v.get("pass"))]
            if S and len(vals) == len(S):
                val = float(np.mean(vals))
                note = "seeds " + " ".join(f"{k}={fmt(v['pass'])}" for k, v in sorted(S.items())) + \
                       "; baseline seeds " + " ".join(f"{fmt(v.get('base_pass'))}" for k, v in sorted(S.items()))
            else:
                note = f"{len(vals)}/{len(S)} seed tags scored"
        if val is None:
            status = "UNVERIFIED"
        elif isinstance(val, bool) or op == "==":
            status = "PASS" if val == thr else "FAIL"
        else:
            status = "PASS" if _cmp(op, val, thr) else "FAIL"
        rows.append(dict(id=cid, kind=c["kind"], metric=c["metric"], value=val, threshold=thr, status=status, note=note, source=c["source"]))
    return rows


def verdict(rows):
    kill = [r for r in rows if r["kind"] == "kill"]
    if any(r["status"] == "FAIL" for r in kill):
        return "KILL"
    if any(r["status"] == "UNVERIFIED" for r in kill):
        return "INCOMPLETE"
    return "PASS"


# ----------------------------------------------------------------------------- report
def _md_table(header, rows):
    L = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        L.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(L)


def write_summary(chain, M, rows, out_dir, acct, gpuh_total):
    tag = chain["tag"]
    V = verdict(rows)
    L = [f"# evalchain summary: {tag}", "",
         f"verdict: **{V}**  (KILL = a pre-registered kill criterion failed; INCOMPLETE = a kill criterion is unverified; PASS = all kill criteria pass)", "",
         f"checkpoint `{chain['ckpt']}` ({chain.get('weights', 'ema')} weights), seed {chain.get('seed')}, CFGS {chain.get('cfgs')}, N {chain.get('n')}, "
         f"recipe {chain.get('recipe')}{(' + ' + chain['extra']) if chain.get('extra') else ''}; item {chain.get('item')}; submitted {chain.get('submitted')}; "
         f"summarized {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
         "## Criteria", "",
         _md_table(["id", "kind", "value", "op", "threshold", "status", "note"],
                   [(r["id"], r["kind"], (fmt(r["value"]) if not isinstance(r["value"], bool) else r["value"]),
                     next(c["op"] for c in CRIT["criteria"] if c["id"] == r["id"]), r["threshold"], r["status"], r["note"]) for r in rows]),
         "", "## Stages and cost", ""]
    srows = []
    for name, st in chain["stages"].items():
        a = acct.get(str(st.get("job")), {})
        v = (M.get("validity") or {}).get(name, {})
        srows.append((name, st.get("job"), a.get("state", "?"), a.get("elapsed_s", "-"), a.get("gpus", "-"), fmt(a.get("gpuh"), 2),
                      st.get("expected_gpuh", "-"), v.get("exists", "-"), v.get("valid", "-"),
                      st.get("tag") or os.path.basename(st.get("tar") or st.get("json") or "")))
    L += [_md_table(["stage", "job", "state", "elapsed s", "GPUs", "GPU-h (sacct)", "expected GPU-h", "output exists", "valid", "output"], srows), "",
          f"total GPU-h used (sacct, GPU stages): **{fmt(gpuh_total, 2)}**; submit-time estimate {chain.get('estimate', {}).get('expected_gpuh', '-')} expected / "
          f"{chain.get('estimate', {}).get('cap_gpuh', '-')} walltime cap", ""]
    S = M.get("sfar") or {}
    if S:
        L += ["## SFAR (specs_v2, goal arms full/sparse8/dense0_true pooled)", "",
              _md_table(["CFG", "n", "SFAR", "95 % CI", "baseline", "delta", "delta CI", "sig", "rec<.9", "sep>1.15", "breaks", "clash>4", "pass&native-hand"],
                        [(f"{cfg:.1f}", s["n"], fmt(s["pass"]), f"[{fmt(s['lo'])},{fmt(s['hi'])}]", fmt(s.get("base_pass")), fmt(s.get("delta")),
                          f"[{fmt(s.get('delta_lo'))},{fmt(s.get('delta_hi'))}]", "sig" if s.get("sig") else "", fmt(s["rec_lt09"]), fmt(s["sep_gt115"]),
                          fmt(s["breaks"], 4), fmt(s["clash_gt4"]), fmt(s["pass_native"])) for cfg, s in sorted(S.items())]), ""]
    SS = M.get("sfar_seeds") or {}
    if SS:
        L += ["### --full seeds (each vs the matching N019 baseline seed tag)", "",
              _md_table(["seed", "tag", "n", "SFAR@1.5", "CI", "baseline", "delta", "delta CI", "sig"],
                        [(k, v.get("tag"), v.get("n"), fmt(v.get("pass")), f"[{fmt(v.get('lo'))},{fmt(v.get('hi'))}]", fmt(v.get("base_pass")),
                          fmt(v.get("delta")), f"[{fmt(v.get('delta_lo'))},{fmt(v.get('delta_hi'))}]", "sig" if v.get("sig") else "") for k, v in sorted(SS.items())]), ""]
    SW = M.get("sandwich") or {}
    if SW:
        L += ["## Sandwich anatomy (sandwich_1tit, goal arms pooled; p = pairing fraction >= 2 CA-CA < 5.5 A, L = soft ladder)", "",
              _md_table(["CFG", "n", "pass", "baseline", "p12 (hairpin)", "base p12", "hairpin12 cut", "L12", "p14", "base p14", "L14", "p05"],
                        [(f"{cfg:.1f}", s["n"], fmt(s["pass"]), fmt(s.get("base_pass")), fmt(s["p12"]), fmt(s.get("base_p12")), fmt(s.get("hairpin12_cut")),
                          fmt(s["L12"], 2), fmt(s["p14"]), fmt(s.get("base_p14")), fmt(s["L14"], 2), fmt(s["p05"])) for cfg, s in sorted(SW.items())]), ""]
    G = M.get("guardrails")
    if G:
        L += ["## Guardrails (specs_v2 goal arms; kit split3 nat-relaxed SPLIT rule)", "",
              _md_table(["CFG", "arm", "n", "SPLIT %", "cut/lr/rg %", "breaks", "clash mean", "clash>4 %", "rec>=.9"],
                        [(f"{cfg:.1f}", a, v["n"], f"{100 * v['split']:.1f}", f"{100 * v['split_cut']:.0f}/{100 * v['split_lr']:.0f}/{100 * v['split_rg']:.0f}",
                          fmt(v["breaks"], 4), fmt(v["clash_mean"], 2), f"{100 * v['clash_gt4']:.1f}", fmt(v["rec_ge09"])) for (cfg, a), v in sorted(G.items())]), ""]
        GB = M.get("guardrails_base")
        if GB:
            L += ["baseline (" + CRIT["baseline"]["specs_v2_tag"] + "): " + "; ".join(
                f"{a}@{cfg:.1f} split {100 * v['split']:.1f}% breaks {v['breaks']:.4f}" for (cfg, a), v in sorted(GB.items())), ""]
    D = M.get("designability")
    if D:
        L += ["## Designability (goal arms x 5 targets x <= 8 designs; LigandMPNN K=8 -> ESMFold2; designable = scRMSD < 2 A and pLDDT > 80)", "",
              _md_table(["CFG", "designable", "k/n", "Wilson CI", "pLDDT>70", "designable & on-fold", "cells"],
                        [(f"{cfg:.1f}", fmt(d["p"]), f"{d['k']}/{d['n']}", f"[{fmt(d['lo'], 2)},{fmt(d['hi'], 2)}]", fmt(d["p70"], 2), fmt(d["joint_p"], 2), d["n_cells"])
                         for cfg, d in sorted(D.items())]), ""]
        DB = M.get("designability_base")
        if DB:
            L += ["baseline N019 (CFG 2.0): " + "; ".join(f"CFG {cfg:.1f} {fmt(d['p'])} ({d['k']}/{d['n']})" for cfg, d in sorted(DB.items())), ""]
    H = M.get("heldout")
    if H:
        L += ["## Held-out beta pack (fn3_1fna, ferredoxin_1ulr, prealbumin_1f41, barrel_1stp; CFG 1.5, goal arms pooled)", ""]
        if H.get("error"):
            L += [H["error"], ""]
        else:
            L += [_md_table(["tag", "n", "SFAR@1.5", "CI", "baseline", "base n", "delta", "delta CI", "sig"],
                            [(H.get("tag"), H.get("n"), fmt(H.get("pass")), f"[{fmt(H.get('lo'))},{fmt(H.get('hi'))}]", fmt(H.get("base_pass")), H.get("base_n", "-"),
                              fmt(H.get("delta")), f"[{fmt(H.get('delta_lo'))},{fmt(H.get('delta_hi'))}]", "sig" if H.get("sig") else "")]), ""]
            if H.get("per_target"):
                L += [_md_table(["target", "n", "pass", "baseline", "delta"], [(t, v["n"], fmt(v["pass"]), fmt(v.get("base_pass")), fmt(v.get("delta")))
                                                                            for t, v in sorted(H["per_target"].items())]), ""]
    R = M.get("ross")
    if R:
        L += ["## Rossmann stress panel (old_rossmann_inputs, CFG 2.0, strict kit SPLIT rule: cut < 0.67 or lr_cb < 0.92 or Rg/N^0.4 > 2.15)", "",
              _md_table(["cell", "case", "n", "SPLIT %", "cut/lr/rg %", "Rg/N^.4", "lr_cb", "helix", "strand", "clash", "breaks"],
                        [(cell, case, v["n"], f"{v['split_pct']:.1f}", f"{v['cut_pct']:.0f}/{v['lr_pct']:.0f}/{v['rg_pct']:.0f}", fmt(v["rg_n04"]), fmt(v["lr_cb"], 2),
                          fmt(v["helix"], 2), fmt(v["strand"], 2), fmt(v["clash"], 1), fmt(v["breaks"], 2))
                         for cell, C in sorted(R.get("cells_detail", {}).items()) for case, v in sorted(C.items())]), "",
              f"full_rossmann_like SPLIT % mean over seeds: tag {fmt(R.get('full_split_pct'), 1)}, baseline {fmt(R.get('base_full_split_pct'), 1)} (N019 recorded 21.1; E2 29.7; p0694 28.9)", ""]
    if M.get("errors"):
        L += ["## Errors during summarize", ""] + [f"- {e}" for e in M["errors"]] + [""]
    L += ["## Files", "", f"- results: `{out_dir}` (summary.json, anatomy_cfg*.csv, heldout_cfg1.5.csv, guardrails.csv, ross.csv, chain.json, jobs.json)",
          f"- cache (designs/tm csv, shared by both session instances): `{chain.get('cache')}`",
          f"- tars: `{chain.get('eval_dir')}/<tag>.tar.gz`, `<TAG>.ross.tar.gz`, designability json", ""]
    md = "\n".join(L)
    with open(os.path.join(out_dir, "summary.md"), "w") as fh:
        fh.write(md)
    return md, V


def _jsonable(o):
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else "|".join(str(x) for x in k) if isinstance(k, tuple) else str(k)): _jsonable(v) for k, v in o.items()
                if not (isinstance(k, str) and k.startswith("_pass"))}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# ----------------------------------------------------------------------------- main
def summarize(chain, procs=16, out_dir=None, cache=None, no_sacct=False):
    tag = chain["tag"]
    out_dir = out_dir or chain["out_dir"]
    cache = cache or chain.get("cache") or EVAL_DIR + "/cache"
    eval_dir = chain.get("eval_dir") or EVAL_DIR
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    cfgs = [float(c) for c in str(chain.get("cfgs", "1.5 2.0")).split()]
    stages = chain["stages"]
    B = CRIT["baseline"]
    M = {"errors": []}

    def guard(name, fn, *a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            M["errors"].append(f"{name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None

    # --- sacct + validity
    acct = {} if no_sacct else sacct_info([st.get("job") for st in stages.values()] + [os.environ.get("SLURM_JOB_ID")])
    V = {}
    for name, st in stages.items():
        if name == "summarize":
            continue
        a = acct.get(str(st.get("job")), {})
        if st.get("tag"):
            tp = f"{eval_dir}/{st['tag']}.tar.gz"
            meta = tar_meta(tp, f"{st['tag']}/meta.json")
            V[name] = dict(path=tp, exists=meta is not None and "_error" not in meta, valid=(meta.get("valid") if meta and "_error" not in meta else None),
                           note=(meta.get("note") or meta.get("_error")) if meta else "tar missing", n_found=meta.get("n_designs_found") if meta else None,
                           n_expected=meta.get("n_designs_expected") if meta else None, wall_seconds=meta.get("wall_seconds") if meta else None,
                           ckpt_path=meta.get("ckpt_path") if meta else None, job_state=a.get("state", ""))
        elif st.get("tar"):       # ross panel: the tar is the output; meta.json (ross_panel.sbatch) carries validity, older panels have none
            meta = tar_meta(st["tar"])
            has_meta = meta is not None and "_error" not in meta
            V[name] = dict(path=st["tar"], exists=os.path.exists(st["tar"]), valid=(meta.get("valid") if has_meta else None),
                           note=(meta.get("note") if has_meta else (meta.get("_error") if meta else "tar missing")), job_state=a.get("state", ""))
        elif st.get("json"):
            ok = os.path.exists(st["json"])
            V[name] = dict(path=st["json"], exists=ok, valid=(True if ok else None), note="" if ok else "designability json missing", job_state=a.get("state", ""))
    M["validity"] = V
    if chain.get("weights") == "raw":
        for name, v in V.items():
            if v.get("ckpt_path") and not os.path.basename(v["ckpt_path"]).endswith("_raw.ckpt"):
                M["errors"].append(f"{name}: weights=raw requested but the harness ran {v['ckpt_path']}")
    gpuh_total = sum(a["gpuh"] for jid, a in acct.items() if jid != str(os.environ.get("SLURM_JOB_ID")))

    # --- specs_v2 anatomy per CFG (+ baseline)
    if V.get("eval", {}).get("exists"):
        M["sfar"], M["sandwich"] = {}, {}
        for cfg in cfgs:
            T = guard(f"anatomy cfg {cfg}", run_anatomy, [tag], B["specs_v2_tag"], cfg, cache, os.path.join(out_dir, f"anatomy_cfg{cfg:.1f}.csv"), procs=procs)
            if T:
                M["sfar"][cfg] = sfar_stats(T, tag, B["specs_v2_tag"])
                M["sandwich"][cfg] = sandwich_stats(T, tag, B["specs_v2_tag"])
        M["guardrails"] = guard("guardrails", guardrails, cache, tag, cfgs)
        M["guardrails_base"] = guard("guardrails base", guardrails, cache, B["specs_v2_tag"], cfgs)
        for t in (tag,):
            src = f"{cache}/{t}.cells.csv"
            if os.path.exists(src):
                shutil.copy(src, out_dir)
    # --- --full seeds
    seed_stages = {n: st for n, st in stages.items() if n.startswith("eval_s")}
    if seed_stages:
        M["sfar_seeds"] = {}
        S1 = M.get("sfar", {}).get(1.5)
        if S1:
            M["sfar_seeds"][str(chain.get("seed"))] = dict(S1, tag=tag)
        for name, st in sorted(seed_stages.items()):
            if V.get(name, {}).get("exists"):
                T = guard(f"anatomy {name}", run_anatomy, [st["tag"]], st.get("base_tag"), 1.5, cache, os.path.join(out_dir, f"anatomy_cfg1.5_{st['tag']}.csv"), procs=procs)
                s = sfar_stats(T, st["tag"], st.get("base_tag")) if T else None
                M["sfar_seeds"][str(st.get("seed"))] = dict(s, tag=st["tag"]) if s else None
            else:
                M["sfar_seeds"][str(st.get("seed"))] = None
    # --- designability
    if stages.get("design"):
        M["designability"] = guard("designability", designability_pool, stages["design"].get("json"))
        M["designability_base"] = guard("designability base", designability_pool, B["designability_json"])
        if stages["design"].get("json") and os.path.exists(stages["design"]["json"].replace(".json", ".md")):
            shutil.copy(stages["design"]["json"].replace(".json", ".md"), out_dir)
    # --- held-out pack
    if stages.get("heldout"):
        ht = stages["heldout"]["tag"]
        H = {"tag": ht, "base_tag": B["heldout_tag"]}
        if V.get("heldout", {}).get("exists"):
            base = B["heldout_tag"] if os.path.exists(f"{eval_dir}/{B['heldout_tag']}.tar.gz") else None
            T = guard("heldout anatomy", run_anatomy, [ht], base, 1.5, cache, os.path.join(out_dir, "heldout_cfg1.5.csv"),
                      manifest=HELDOUT_MANIFEST, refs=HELDOUT_REFS, procs=procs)
            s = sfar_stats(T, ht, base) if T else None
            if s:
                H.update(s)
                H["per_target"] = {}
                for t in sorted({k[0] for k in T[ht] if k[0] != "ALL"}):
                    r = T[ht][(t, "ALL")]
                    rec = dict(n=int(r["n"]), pass_=float(r["pass"]))
                    rec["pass"] = rec.pop("pass_")
                    if base and (t, "ALL") in T[base]:
                        b = T[base][(t, "ALL")]
                        d, lo, hi = ec.diff_ci(r["_pass_vec"], b["_pass_vec"], n_boot=2000, seed=0)
                        rec.update(base_pass=float(b["pass"]), delta=d, delta_lo=lo, delta_hi=hi)
                    H["per_target"][t] = rec
            else:
                H["error"] = "held-out tag could not be scored (see log)"
        else:
            H["error"] = f"held-out tar missing: {V.get('heldout', {}).get('path')}"
        M["heldout"] = H
    # --- Rossmann panel
    if stages.get("ross"):
        R = {"tar": stages["ross"].get("tar")}
        if V.get("ross", {}).get("exists"):
            sc = guard("ross", ross_scores, stages["ross"]["tar"], procs)
            if sc:
                R["cells_detail"] = {c: v for c, v in sc.items() if not c.startswith("_")}
                R["cells"] = len(R["cells_detail"])
                R["full_split_pct"] = ross_case_mean(sc, "full_rossmann_like")
                R["adjonly_split_pct"] = ross_case_mean(sc, "adjonly_rossmann_like")
                R["ssonly_split_pct"] = ross_case_mean(sc, "ssonly_rossmann_like")
                R["n_failed"] = sc.get("_n_failed", 0)
            bases = [p for p in B["ross_tars"] if os.path.exists(p)]
            if bases:
                bsc = {}
                for p in bases:
                    s = guard(f"ross base {p}", ross_scores, p, procs)
                    if s:
                        bsc.update({c: v for c, v in s.items() if not c.startswith("_")})
                R["base_cells_detail"] = bsc
                R["base_full_split_pct"] = ross_case_mean(bsc, "full_rossmann_like")
        else:
            R["error"] = "ross tar missing"
        M["ross"] = R
    # --- criteria + report
    rows = evaluate(M, full=bool(seed_stages))
    V_ = verdict(rows)
    # csv side files
    if M.get("guardrails"):
        with open(os.path.join(out_dir, "guardrails.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["tag", "cfg", "arm", "n", "split", "split_cut", "split_lr", "split_rg", "breaks", "clash_mean", "clash_gt4", "rec_ge09"])
            for key, G in (("tag", M["guardrails"]), ("base", M.get("guardrails_base") or {})):
                for (cfg, a), v in sorted(G.items()):
                    w.writerow([tag if key == "tag" else B["specs_v2_tag"], cfg, a, v["n"], v["split"], v["split_cut"], v["split_lr"], v["split_rg"], v["breaks"], v["clash_mean"], v["clash_gt4"], v["rec_ge09"]])
    if M.get("ross", {}).get("cells_detail"):
        with open(os.path.join(out_dir, "ross.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cell", "case", "n", "split_pct", "cut_pct", "lr_pct", "rg_pct", "rg_n04", "lr_cb", "cutdens", "helix", "strand", "clash", "breaks"])
            for det in (M["ross"]["cells_detail"], M["ross"].get("base_cells_detail") or {}):
                for cell, C in sorted(det.items()):
                    for case, v in sorted(C.items()):
                        w.writerow([cell, case] + [v[k] for k in ("n", "split_pct", "cut_pct", "lr_pct", "rg_pct", "rg_n04", "lr_cb", "cutdens", "helix", "strand", "clash", "breaks")])
    md, _ = write_summary(chain, M, rows, out_dir, acct, gpuh_total)
    out = dict(tag=tag, verdict=V_, chain=chain, criteria=rows, metrics=_jsonable(M), sacct=acct, gpuh_total=gpuh_total,
               summarized=time.strftime("%Y-%m-%d %H:%M:%S"), summarize_job=os.environ.get("SLURM_JOB_ID"))
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(_jsonable(out), fh, indent=1)
    print(md)
    print(f"\nwrote {out_dir}/summary.md + summary.json; verdict {V_}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain", required=True, help="chain.json written by evalchain.sh")
    ap.add_argument("--procs", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--out", default=None, help="override the results dir")
    ap.add_argument("--cache", default=None, help="override the shared scoring cache")
    ap.add_argument("--no-sacct", action="store_true")
    a = ap.parse_args(argv)
    chain = json.load(open(a.chain))
    summarize(chain, procs=a.procs, out_dir=a.out, cache=a.cache, no_sacct=a.no_sacct)
    return 0


if __name__ == "__main__":
    sys.exit(main())
