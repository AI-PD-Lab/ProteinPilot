#!/usr/bin/env python3
"""TRIAD-R S2 gate table (one page) from scored design runs + the oracle matrix. Pure python + stdlib (runs with the login-node /usr/bin/python3, 3.9).

Inputs
  --run GROUP[@SEED]=SCORE,TAR   one scored run. GROUP = LABEL/MODE (LABEL e.g. B1000, Z1000, base; MODE masked | generated | stock). SCORE = the scorer's
                                 SLURM log (tj_score_<job>.out: lines ``{'run': 'disco', 'design': 'design_0000', 'scrmsd': .., 'plddt': ..}``) or its
                                 tj_score_<TAG>_codesign.json; TAR = the design archive written by eval_ckpt.sbatch (sequences = CA residue names of the PDBs).
                                 SEED = ``@SEED``, else read from the archive name (``..._s<seed>_<len>_<job>.tar.gz``), else unknown (never pairs).
  --manifest FILE                lines ``GROUP[@SEED] SCORE TAR`` (# comments allowed) instead of / in addition to --run
  --base LABEL                   label of the released-model runs (default ``base``); without such runs the built-in N003 reference is used
  --control LABEL                label prefix of the control arm (arm Z), matched to an arm label by its checkpoint suffix (B1000 -> Z1000 with ``--control Z``)
  --oracle FILE[=LABEL]          oracle_s2.py JSON of a checkpoint (repeatable)
  --step0 FILE                   oracle_s2.py JSON of the RELEASED model (ring_embed == 0, ``BUNDLE=.../rfd4_weights``): the step-0 control of gate (b). Without it an
                                 --oracle file whose model has ring_embed == 0 is used; without any, gate (b) is INVALID (``--allow-no-step0`` for exploration only)
  --eval-log FILE                eval_ckpt log(s): the ``EVAL ring`` lines (generated label density) are echoed under gate (c)
  --gate-c FILE                  json of ``s2/gate_c.py`` (RING run on the decoded designs): its density / agreement / verdict section is printed under gate (c)

Definitions (same judge as every number of this project)
  designable          own-sequence co-design scRMSD < 2 A and pLDDT > 70
  low complexity (LC) top residue of the design > 25 %          natural = designable and not LC          (join3.py of N003)
Gate (a) / (a')  need COMPLETE runs: scored designs == archived designs (same design ids) == --expect-n (default 48) == design_map rows, else INCOMPLETE (no PASS/FAIL).
  band      2 x SD of the per-seed fractions of the supplied BASE runs when >= 2 base seeds are present, else 2 x 1.5/48 tagged ``default band`` (N003 addendum)
  (a)       masked-stream designability >= base - band for designable AND natural (``--require designable``: first only). The arm's seeds must be a subset of the base
            seeds (base restricted to the arm's seeds, paired by seed), else UNPAIRED (numbers shown, no verdict). Without base runs: built-in N003 reference, not seed-paired.
  (a')      generated >= masked - band, ONLY for identical seed sets (else UNPAIRED); IMPROVED if >= masked + band.
Gate (b)  oracle cells. A file is INVALID (never PASS) when: evaluated chains < --min-chains (200), sequence clusters < --min-units (40), any common-random-number violation, a
          required cell is missing, the interactions|structure cell is not the fixed-t_lat contrast, no step-0 control was given, or the step-0 control (released model, ring_embed ==
          0) is itself invalid (same checks) or has a nonzero delta (|delta| > --step0-tol, default exact 0) in any stream cell.
          A cell is POS when the 95 % CI of delta (loss(all-MASK) - loss(true map)) > 0 AND the CI of margin (shuffled - true) > 0. A TERM is POS when its cell is POS at >= 2 of the
          4 levels t_ca in {0.2, 0.5, 0.8, 0.95} INCLUDING 0.5 and no level is NEG (multiplicity: one lucky level is not enough); MIXED = POS and NEG levels; else WEAK / NS / NEG.
          structure axis = bb_ca POS AND ca_x0_rmsd not NEG/MIXED; sequence axis = seq_ce POS; interactions axis = ring_ce (t_ca 0.5 -> 0.9 at fixed t_lat) POS and auc not NEG.
          PASS = all three axes POS on a valid file.
Gate (c)  generated vs RING-measured labels of the decoded designs: NOT computed here (needs RING on the designs: run ``s2/gate_c.py``); the label density of the generated maps is
          echoed, and with ``--gate-c FILE`` the gate_c.py result (RING-measured density vs natural chains, generated-vs-measured P/R/F1, shuffled control, distance baselines, verdict) is printed.
Exit status 0 always (this is a report); ``--json OUT`` writes every number.
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import math
import os
import re
import statistics
import sys
import tarfile

THREE = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P",
         "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"}
SCRMSD_MAX, PLDDT_MIN, LC_MAX = 2.0, 70.0, 0.25
# N003 addendum 2026-10-22: reference sampler = latent TSR k=1.5, L=100, seeds 0..2 x 48 = 144 designs (jobs 1093297/98, 1099370/71, 1099372/73)
REF_BASE = {"source": "N003 addendum (latent TSR k=1.5, L=100, 3 seeds x 48)", "n": 144, "designable": 121, "natural": 103, "sd_per_48": 1.5, "seeds": 3}
REQUIRED_LEVELS = (0.2, 0.5, 0.8, 0.95)       # t_ca of the stream cells counted by gate (b)
MIN_POS_LEVELS = 2
SHOWN_TERMS = {"structure|interactions": ("bb_ca", "ca_x0_rmsd", "local_latents"), "sequence|interactions": ("seq_ce",), "interactions|structure": ("ring_ce", "auc", "ap", "auc_hard")}
STREAM_AXES = ("structure|interactions", "sequence|interactions")
KEYS = ("designable", "natural")


def sequence_cluster(cluster, fallback: str = "") -> str:
    """Sequence-cluster id of a manifest cluster string = the part before the first '+' (the rule of ``ring_data.cluster_group``): ``abc+HEM`` and ``abc`` are one
    cluster; empty / missing -> ``fallback`` (a row without a cluster is its own unit). Shared by oracle_s2 (bootstrap units) and the gate report."""
    if cluster is None or cluster != cluster:
        return fallback
    c = str(cluster).split("+", 1)[0]
    return c if c else fallback


# ------------------------------------------------------------------------------------------------------------------ run inputs
def read_scores(path: str) -> dict:
    """{design name: {'scrmsd', 'plddt'}} from a scorer SLURM log or a codesign json (last record of a design wins)."""
    out = {}
    if path.endswith(".json"):
        for r in json.load(open(path)):
            out[str(r["design"])] = {"scrmsd": float(r["scrmsd"]), "plddt": float(r["plddt"])}
        return out
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{'run'"):
                continue
            try:
                r = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            if "design" in r and "scrmsd" in r and "plddt" in r:
                out[str(r["design"])] = {"scrmsd": float(r["scrmsd"]), "plddt": float(r["plddt"])}
    return out


def read_archive(tar_path: str) -> tuple:
    """-> ({design name: one-letter sequence from the CA records}, number of design_map.tsv rows or None) of an archive (members ``*/design_*.pdb``)."""
    seqs, map_rows = {}, None
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            base = os.path.basename(m.name)
            if m.isfile() and base == "design_map.tsv":
                map_rows = len([l for l in tf.extractfile(m).read().decode("utf-8", "replace").splitlines() if l.strip()])
            if not (m.isfile() and base.startswith("design_") and base.endswith(".pdb")):
                continue
            s = []
            for line in tf.extractfile(m).read().decode("utf-8", "replace").splitlines():
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    s.append(THREE.get(line[17:20].strip(), "X"))
            seqs[base[:-4]] = "".join(s)
    return seqs, map_rows


def read_sequences(tar_path: str) -> dict:
    return read_archive(tar_path)[0]


def run_stats(scores: dict, seqs: dict) -> dict:
    """join3.py logic over the designs that have BOTH a score and a sequence: designable, low complexity, designable AND natural, pooled composition entropy, Glu."""
    n = des = lc = nat = 0
    allc = collections.Counter()
    sc, pl = [], []
    for name, r in sorted(scores.items()):
        s = seqs.get(name)
        if not s:
            continue
        c = collections.Counter(s)
        allc.update(s)
        d = r["scrmsd"] < SCRMSD_MAX and r["plddt"] > PLDDT_MIN
        low = max(c.values()) / len(s) > LC_MAX
        n += 1
        des += d
        lc += low
        nat += d and not low
        sc.append(r["scrmsd"])
        pl.append(r["plddt"])
    tot = sum(allc.values())
    ent = -sum(v / tot * math.log2(v / tot) for v in allc.values()) if tot else float("nan")
    med = lambda x: sorted(x)[len(x) // 2] if x else float("nan")
    return {"n": n, "designable": des, "lc": lc, "natural": nat, "entropy": ent, "glu": allc["E"] / tot if tot else float("nan"), "med_scrmsd": med(sc),
            "med_plddt": med(pl), "n_scores": len(scores), "n_seqs": len(seqs)}


def check_run(scores: dict, seqs: dict, map_rows, expect_n: int) -> list:
    """Completeness problems of one scored run (empty = complete): the join score <-> archive must be one-to-one and have the requested size."""
    bad = []
    want = expect_n if expect_n > 0 else len(seqs)
    if len(seqs) != want:
        bad.append(f"archive holds {len(seqs)} designs, expected {want}")
    if map_rows is not None and map_rows != len(seqs):
        bad.append(f"design_map.tsv has {map_rows} rows, archive {len(seqs)} pdbs")
    if len(scores) != want:
        bad.append(f"scored {len(scores)} of {want} designs")
    no_score, no_arch = sorted(set(seqs) - set(scores)), sorted(set(scores) - set(seqs))
    if no_score:
        bad.append(f"{len(no_score)} archived designs without a score (e.g. {no_score[0]})")
    if no_arch:
        bad.append(f"{len(no_arch)} scored designs not in the archive (e.g. {no_arch[0]}): score log of another archive?")
    return bad


def pool(runs: list) -> dict:
    """Sum of the counts of several runs (pooled fractions); entropy / medians are n-weighted means (informational)."""
    n = sum(r["n"] for r in runs)
    w = lambda k: (sum(r[k] * r["n"] for r in runs) / n) if n else float("nan")
    return {"n": n, "designable": sum(r["designable"] for r in runs), "lc": sum(r["lc"] for r in runs), "natural": sum(r["natural"] for r in runs), "entropy": w("entropy"),
            "glu": w("glu"), "med_scrmsd": w("med_scrmsd"), "med_plddt": w("med_plddt"), "seeds": len(runs)}


def parse_group(spec: str) -> tuple:
    """'B1000/masked@0' -> ('B1000', 'masked', '0'); the seed part is optional."""
    seed = None
    if "@" in spec:
        spec, seed = spec.rsplit("@", 1)
    if "/" not in spec:
        raise SystemExit(f"gates_report: group '{spec}' must be LABEL/MODE (masked|generated|stock)")
    label, mode = spec.rsplit("/", 1)
    return label, mode, seed


def seed_of(spec_seed, tar: str) -> tuple:
    if spec_seed is not None:
        return str(spec_seed), "explicit"
    m = re.search(r"_s(\d+)_(?:short|medium|long)_", os.path.basename(tar))
    return (m.group(1), "archive name") if m else (None, "unknown")


def collect(run_specs: list, expect_n: int = 48) -> dict:
    """{(label, mode): {seed: run record}}; a record = stats + 'problems' (completeness) + file names."""
    groups = collections.defaultdict(dict)
    for spec, score, tar in run_specs:
        label, mode, sd = parse_group(spec)
        seed, src = seed_of(sd, tar)
        scores = read_scores(score)
        seqs, map_rows = read_archive(tar)
        st = run_stats(scores, seqs)
        st["problems"] = check_run(scores, seqs, map_rows, expect_n)
        st["score_file"], st["tar"], st["seed_src"] = score, tar, src
        key = seed if seed is not None else f"?{len(groups[(label, mode)])}"
        if key in groups[(label, mode)]:
            st["problems"].append(f"seed {key} given twice in {label}/{mode}")
        groups[(label, mode)][key] = st
    return dict(groups)


# ------------------------------------------------------------------------------------------------------------------ gates (a), (a')
def frac(p: dict, key: str) -> float:
    return p[key] / p["n"] if p["n"] else float("nan")


def band_from(base_runs: dict, default_sd: float) -> dict:
    """{'designable': band, 'natural': band, 'tag': ..} bands in FRACTION units. >= 2 base seeds: 2 x sample SD of their per-seed fractions (per metric); else the default."""
    if len(base_runs) >= 2:
        out = {k: 2.0 * statistics.stdev([frac(r, k) for r in base_runs.values()]) for k in KEYS}
        out["tag"] = f"2 x SD of {len(base_runs)} base seeds (SD {out['designable'] * 24:.1f} / {out['natural'] * 24:.1f} designs per 48)"
        out["default"] = False
        return out
    d = 2.0 * default_sd / 48.0
    return {"designable": d, "natural": d, "tag": f"DEFAULT band 2 x {default_sd:g}/48 (fewer than 2 base seeds supplied)", "default": True}


def gate_a(arm: dict, base: dict, band: dict, require: str = "both") -> dict:
    """arm/base: pooled stats over the same seeds. Pass = arm fraction >= base fraction - band for designable (and natural)."""
    res = {}
    for key in KEYS:
        thr = frac(base, key) - band[key]
        a = frac(arm, key)
        res[key] = {"arm": a, "base": frac(base, key), "threshold": thr, "margin_per48": (a - thr) * 48.0, "pass": a >= thr - 1e-12}
    need = ("designable",) if require == "designable" else KEYS
    res["pass"] = all(res[k]["pass"] for k in need)
    return res


def gate_a_prime(gen: dict, msk: dict, band: dict) -> dict:
    out = {}
    for key in KEYS:
        d = frac(gen, key) - frac(msk, key)
        out[key] = {"gen": frac(gen, key), "masked": frac(msk, key), "diff": d,
                    "status": "IMPROVED" if d >= band[key] else ("NOT_WORSE" if d >= -band[key] - 1e-12 else "WORSE")}
    return out


def problems_of(runs: dict) -> list:
    return [f"seed {s}: {p}" for s, r in sorted(runs.items()) for p in r["problems"]]


def compare_masked_to_base(arm_runs: dict, base_runs: dict, default_sd: float, require: str) -> dict:
    """Gate (a) for one arm: INCOMPLETE | UNPAIRED | PASS | FAIL, with the numbers whenever they exist."""
    out = {"arm_seeds": sorted(arm_runs), "base_seeds": sorted(base_runs)}
    bad = problems_of(arm_runs) + [f"base {p}" for p in problems_of(base_runs)]
    if bad:
        return dict(out, status="INCOMPLETE", reasons=bad)
    if not set(arm_runs) <= set(base_runs):
        arm = pool(list(arm_runs.values()))
        return dict(out, status="UNPAIRED", reasons=[f"arm seeds {sorted(arm_runs)} are not a subset of the base seeds {sorted(base_runs)}"], arm=arm)
    band = band_from(base_runs, default_sd)
    paired_base = pool([base_runs[s] for s in sorted(arm_runs)])
    arm = pool([arm_runs[s] for s in sorted(arm_runs)])
    g = gate_a(arm, paired_base, band, require)
    return dict(out, status="PASS" if g["pass"] else "FAIL", gate=g, band=band, arm=arm, base=paired_base, reasons=[])


def compare_builtin(arm_runs: dict, default_sd: float, require: str) -> dict:
    out = {"arm_seeds": sorted(arm_runs), "base_seeds": []}
    if problems_of(arm_runs):
        return dict(out, status="INCOMPLETE", reasons=problems_of(arm_runs))
    base = {"n": REF_BASE["n"], "designable": REF_BASE["designable"], "natural": REF_BASE["natural"]}
    band = {"designable": 2 * default_sd / 48.0, "natural": 2 * default_sd / 48.0, "tag": f"DEFAULT band 2 x {default_sd:g}/48; built-in {REF_BASE['source']}, NOT seed-paired", "default": True}
    arm = pool(list(arm_runs.values()))
    g = gate_a(arm, base, band, require)
    return dict(out, status="PASS" if g["pass"] else "FAIL", gate=g, band=band, arm=arm, base=base, reasons=[])


def compare_gen_masked(gen_runs: dict, msk_runs: dict, band: dict) -> dict:
    out = {"gen_seeds": sorted(gen_runs), "masked_seeds": sorted(msk_runs)}
    bad = problems_of(gen_runs) + problems_of(msk_runs)
    if bad:
        return dict(out, status="INCOMPLETE", reasons=bad)
    if set(gen_runs) != set(msk_runs):
        return dict(out, status="UNPAIRED", reasons=[f"generated seeds {sorted(gen_runs)} != masked seeds {sorted(msk_runs)}"])
    seeds = sorted(gen_runs)
    g = gate_a_prime(pool([gen_runs[s] for s in seeds]), pool([msk_runs[s] for s in seeds]), band)
    per_seed = {s: gen_runs[s]["designable"] - msk_runs[s]["designable"] for s in seeds}
    worst = "WORSE" if any(g[k]["status"] == "WORSE" for k in KEYS) else ("IMPROVED" if all(g[k]["status"] == "IMPROVED" for k in KEYS) else "NOT_WORSE")
    return dict(out, status=worst, gate=g, per_seed_designable_diff=per_seed, band=band, reasons=[])


# ---------------------------------------------------------------------------------------------------------------------- gate (b)
def t_eq(a, b) -> bool:
    return abs(float(a) - float(b)) < 1e-6


def term_status(cells: list, axis: str, term: str, required=REQUIRED_LEVELS) -> dict:
    """Multiplicity rule: POS needs POS (delta CI > 0 and margin CI > 0) at >= MIN_POS_LEVELS of the required levels INCLUDING t_ca = 0.5 and no NEG level."""
    cs = [c for c in cells if c.get("cell") == axis and c.get("term") == term and any(t_eq(c["t_ca"], r) for r in required)]
    have = [c["t_ca"] for c in cs]
    missing = [r for r in required if not any(t_eq(r, h) for h in have)]
    pos = sorted(c["t_ca"] for c in cs if c["verdict"] == "POS")
    neg = sorted(c["t_ca"] for c in cs if c["verdict"] == "NEG")
    weak = sorted(c["t_ca"] for c in cs if c["verdict"] == "WEAK")
    if not cs:
        st = "NA"
    elif pos and neg:
        st = "MIXED"
    elif neg:
        st = "NEG"
    elif len(pos) >= MIN_POS_LEVELS and any(t_eq(0.5, t) for t in pos):
        st = "POS"
    elif pos or weak:
        st = "WEAK"
    else:
        st = "NS"
    return {"status": st, "pos": pos, "neg": neg, "weak": weak, "missing": missing}


def interaction_status(cells: list) -> dict:
    ce = [c for c in cells if c.get("cell") == "interactions|structure" and c.get("term") == "ring_ce" and "t_lat" in c]
    auc = [c for c in cells if c.get("cell") == "interactions|structure" and c.get("term") == "auc" and "t_lat" in c]
    if not ce:
        return {"status": "NA", "pos": [], "neg": [], "weak": [], "missing": ["ring_ce at fixed t_lat"]}
    c = ce[0]
    st = "POS" if c["verdict"] == "POS" else ("NEG" if c["verdict"] == "NEG" else "NS")
    if st == "POS" and auc and auc[0]["verdict"] == "NEG":
        st = "NEG(auc)"
    return {"status": st, "pos": [c["t_ca"]] if c["verdict"] == "POS" else [], "neg": [c["t_ca"]] if c["verdict"] == "NEG" else [], "weak": [], "missing": []}


def axes_status(cells: list) -> dict:
    bb, ca = term_status(cells, "structure|interactions", "bb_ca"), term_status(cells, "structure|interactions", "ca_x0_rmsd")
    seq = term_status(cells, "sequence|interactions", "seq_ce")
    inter = interaction_status(cells)
    structure = bb["status"] if ca["status"] not in ("NEG", "MIXED") else "NEG(ca_x0_rmsd)"
    return {"structure": structure, "sequence": seq["status"], "interactions": inter["status"], "terms": {"bb_ca": bb, "ca_x0_rmsd": ca, "seq_ce": seq, "ring_ce": inter}}


def file_problems(o: dict, min_chains: int, min_units, check_cells: bool) -> list:
    """Admissibility of one oracle file (reasons; empty = admissible)."""
    r = []
    ld = o.get("load", {})
    if ld.get("evaluated", 0) < min_chains:
        r.append(f"evaluated {ld.get('evaluated', 0)} chains < {min_chains}")
    if min_units is not None and o.get("n_units", 0) < min_units:
        r.append(f"{o.get('n_units', 0)} sequence clusters < {min_units}")
    if o.get("checks", {}).get("crn_violations", 0):
        r.append(f"{o['checks']['crn_violations']} common-random-number violations")
    if check_cells:
        cells = o.get("cells", [])
        for axis, term in (("structure|interactions", "bb_ca"), ("structure|interactions", "ca_x0_rmsd"), ("sequence|interactions", "seq_ce")):
            miss = term_status(cells, axis, term)["missing"]
            if miss:
                r.append(f"missing {term} cells at t_ca {miss}")
        if not any(c.get("cell") == "interactions|structure" and c.get("term") == "ring_ce" and "t_lat" in c for c in cells):
            r.append("no interactions|structure ring_ce cell with a fixed t_lat (old confounded contrast or not evaluated)")
    return r


def check_control(c: dict, min_chains: int, tol: float) -> dict:
    """The step-0 control: the released model (ring_embed == 0) must give delta = shuffled = margin = 0 in every stream cell."""
    reasons = file_problems(c, min_chains, None, False)
    if c.get("model", {}).get("ring_embed_absmax") != 0.0:
        reasons.append("the control model has a nonzero ring_embed (not the released model)")
    mx = 0.0
    n = 0
    for cell in c.get("cells", []):
        if cell.get("cell") in STREAM_AXES:
            n += 1
            for k in ("delta", "shuffled", "margin"):
                v = (cell.get(k) or {}).get("value")
                mx = max(mx, abs(v) if v is not None else 0.0)
    if n == 0:
        reasons.append("the control has no stream cells")
    if mx > tol:
        reasons.append(f"max |delta| = {mx:.3g} > {tol:g}: the stream changes the released model")
    return {"ok": not reasons, "reasons": reasons, "max_abs_delta": mx, "n_cells": n}


def judge_oracle(o: dict, control: dict | None, min_chains: int = 200, min_units: int = 40, allow_no_step0: bool = False) -> dict:
    """-> {'status': PASS | NOT_YET | INVALID | CONTROL_OK, 'reasons', 'axes'}. INVALID blocks PASS."""
    if o.get("model", {}).get("ring_embed_absmax") == 0.0:       # a released-model run is a control, never an arm
        c = check_control(o, min_chains, 0.0 if control is None else control.get("tol", 0.0))
        return {"status": "CONTROL_OK" if c["ok"] else "INVALID", "reasons": c["reasons"], "axes": None, "control": c}
    reasons = file_problems(o, min_chains, min_units, True)
    if control is None:
        if not allow_no_step0:
            reasons.append("no step-0 control supplied (--step0 <released-model oracle json>)")
    elif not control["ok"]:
        reasons.append("step-0 control failed: " + "; ".join(control["reasons"]))
    ax = axes_status(o.get("cells", []))
    if reasons:
        status = "INVALID"
    else:
        status = "PASS" if (ax["structure"] == "POS" and ax["sequence"] == "POS" and ax["interactions"] == "POS") else "NOT_YET"
    return {"status": status, "reasons": reasons, "axes": ax}


def ci_s(s, spec="%+.4g"):
    if not s or s.get("value") is None:
        return "n/a"
    lo, hi = (s.get("ci95") or [None, None])
    return (spec % s["value"]) + (" [" + (spec % lo) + "," + (spec % hi) + "]" if lo is not None else "")


def oracle_report(label: str, o: dict, verdict: dict, all_terms: bool = False) -> list:
    lines = []
    ld, mdl = o.get("load", {}), o.get("model", {})
    lines.append(f"  [{label}] {mdl.get('arm', '?')} {str(mdl.get('weights', ''))[:60]}  chains: requested={ld.get('requested')} evaluated={ld.get('evaluated')} "
                 f"failed={ld.get('failed_to_load')} no_obs={ld.get('no_observed_pairs')} seq_clusters={o.get('n_units')} (raw {o.get('n_units_raw', '?')})  -> {verdict['status']}")
    for r in verdict["reasons"]:
        lines.append(f"      INVALID: {r}")
    if verdict["axes"] is None:
        return lines
    ax = verdict["axes"]
    lines.append(f"      axes: structure={ax['structure']} sequence={ax['sequence']} interactions={ax['interactions']}  (POS needs >= {MIN_POS_LEVELS} of 4 levels incl. 0.5, none NEG)")
    cells = o.get("cells", [])
    for axis, terms in SHOWN_TERMS.items():
        shown = sorted({c["term"] for c in cells if c.get("cell") == axis}) if all_terms else terms
        for term in shown:
            cs = sorted([c for c in cells if c.get("cell") == axis and c.get("term") == term], key=lambda c: c["t_ca"])
            if not cs:
                continue
            t = ax["terms"].get(term)
            tag = f" status={t['status']} POS@{t['pos'] or '-'} NEG@{t['neg'] or '-'}" if t else ""
            lines.append(f"  {axis:<24s} {term:<13s}{tag}")
            for c in cs:
                if axis == "interactions|structure":
                    lines.append(f"      t_ca {c['t_ca']:g}->{c['t_ca_ref']:g} @t_lat {c.get('t_lat', '?')}  delta {ci_s(c['delta'])}  {c['verdict']}")
                else:
                    lines.append(f"      t_ca {c['t_ca']:<4g} delta {ci_s(c['delta'])}  shuffled {ci_s(c['shuffled'])}  margin {ci_s(c['margin'])}  rel {100 * (c['delta_rel']['value'] or 0):+.2f}%  {c['verdict']}")
    return lines


# ------------------------------------------------------------------------------------------------------------------------ report
def fmt_row(label, p, bad=""):
    return (f"  {label:<22s} seeds={p.get('seeds', 1):<2d} n={p['n']:<4d} designable {p['designable']:>4d}/{p['n']:<4d}={100 * frac(p, 'designable'):5.1f}%  "
            f"d&natural {p['natural']:>4d}={100 * frac(p, 'natural'):5.1f}%  LC {p['lc']:>3d}  H={p['entropy']:.2f}b Glu={100 * p['glu']:.0f}%  medRMSD={p['med_scrmsd']:.2f}{bad}")


def gate_a_line(label: str, c: dict) -> str:
    if c["status"] in ("INCOMPLETE", "UNPAIRED"):
        extra = ""
        if "arm" in c and c["arm"]["n"]:
            extra = f" (numbers, not a verdict: designable {100 * frac(c['arm'], 'designable'):.1f}% d&natural {100 * frac(c['arm'], 'natural'):.1f}%)"
        return f"  {label:<22s} {c['status']}: " + "; ".join(c["reasons"][:3]) + extra
    g = c["gate"]
    return (f"  {label:<22s} designable {100 * g['designable']['arm']:5.1f}% (thr {100 * g['designable']['threshold']:.1f}%, margin {g['designable']['margin_per48']:+.1f}/48) "
            f"{'ok ' if g['designable']['pass'] else 'LOW'} | d&natural {100 * g['natural']['arm']:5.1f}% (thr {100 * g['natural']['threshold']:.1f}%, margin "
            f"{g['natural']['margin_per48']:+.1f}/48) {'ok ' if g['natural']['pass'] else 'LOW'} -> {c['status']}   [seeds {','.join(c['arm_seeds'])}; {c['band']['tag']}]")


def build_report(args) -> tuple:
    specs = []
    for r in args.run or []:
        g, rest = r.split("=", 1)
        sc, tar = rest.split(",", 1)
        specs.append((g, sc, tar))
    if args.manifest:
        for line in open(args.manifest):
            line = line.split("#", 1)[0].strip()
            if line:
                g, sc, tar = line.split()
                specs.append((g, sc, tar))
    groups = collect(specs, args.expect_n)
    pooled = {k: pool(list(v.values())) for k, v in groups.items()}
    js = {"groups": {}, "gate_a": {}, "gate_a_prime": {}, "gate_b": {}}
    lines = ["TRIAD-R S2 GATES  (judge: scRMSD<2A & pLDDT>70 own-sequence; natural = no residue >25%; complete runs only)"]
    base_runs = {}
    for (label, mode), runs in groups.items():
        if label == args.base:
            base_runs.update(runs)
    if base_runs:
        bsrc = f"label '{args.base}' seeds {','.join(sorted(base_runs))}"
    else:
        bsrc = REF_BASE["source"] + " (built-in; L=100 only; not seed-paired)"
    lines.append(f"base reference: {bsrc}")
    lines.append("")
    lines.append("runs")
    for (label, mode), p in sorted(pooled.items()):
        bad = ""
        probs = problems_of(groups[(label, mode)])
        if probs:
            bad = "  INCOMPLETE: " + "; ".join(probs[:2])
        lines.append(fmt_row(f"{label}/{mode}", p, bad))
        js["groups"][f"{label}/{mode}"] = {**p, "seed_list": sorted(groups[(label, mode)]), "problems": probs}
    lines.append("")
    lines.append(f"gate (a)  stream-MASKED designability vs base (pass: >= base - band for designable AND natural; require={args.require})")
    done = False
    for (label, mode), runs in sorted(groups.items()):
        if mode != "masked" or label == args.base:
            continue
        done = True
        c = compare_masked_to_base(runs, base_runs, args.sd, args.require) if base_runs else compare_builtin(runs, args.sd, args.require)
        ctrl_runs = groups.get((args.control_for(label), "masked")) if args.control else None
        if ctrl_runs and c["status"] in ("PASS", "FAIL"):
            d = frac(c["arm"], "designable") - frac(pool([ctrl_runs[s] for s in sorted(ctrl_runs) if s in runs] or list(ctrl_runs.values())), "designable")
            c["vs_control_pts"] = 100 * d
        lines.append(gate_a_line(label + "/masked", c) + (f"  vs control {args.control_for(label)}: {c['vs_control_pts']:+.1f} pts" if "vs_control_pts" in c else ""))
        js["gate_a"][label] = c
    if not done:
        lines.append("  (no <label>/masked runs given)")
    lines.append("")
    lines.append("gate (a') generated vs masked, SAME seeds only (pass: not worse than masked - band)")
    done = False
    for label in sorted({k[0] for k in groups}):
        if (label, "generated") in groups and (label, "masked") in groups:
            done = True
            band = band_from(base_runs, args.sd)
            c = compare_gen_masked(groups[(label, "generated")], groups[(label, "masked")], band)
            if c["status"] in ("INCOMPLETE", "UNPAIRED"):
                lines.append(f"  {label:<22s} {c['status']}: " + "; ".join(c["reasons"][:3]))
            else:
                g = c["gate"]
                lines.append(f"  {label:<22s} designable {100 * g['designable']['gen']:5.1f}% vs {100 * g['designable']['masked']:5.1f}% ({100 * g['designable']['diff']:+.1f} pts) {g['designable']['status']} | "
                             f"d&natural {100 * g['natural']['gen']:5.1f}% vs {100 * g['natural']['masked']:5.1f}% ({100 * g['natural']['diff']:+.1f}) {g['natural']['status']}  -> {c['status']}  "
                             f"per-seed gen-masked designable {c['per_seed_designable_diff']}  [{band['tag']}]")
            js["gate_a_prime"][label] = c
    if not done:
        lines.append("  (needs <label>/generated and <label>/masked)")
    lines.append("")
    lines.append("gate (b)  oracle matrix (delta = loss(all-MASK) - loss(map): + = the map helps; margin = shuffled - true: + = the TRUE map beats a wrong one)")
    oracles = []
    for spec in args.oracle or []:
        path, _, label = spec.partition("=")
        o = json.load(open(path))
        oracles.append((label or o.get("tag") or os.path.basename(path), o))
    control_o = None
    if args.step0:
        control_o = json.load(open(args.step0))
    else:
        control_o = next((o for _, o in oracles if o.get("model", {}).get("ring_embed_absmax") == 0.0), None)
    control = None
    if control_o is not None:
        control = check_control(control_o, args.min_chains, args.step0_tol)
        control["tol"] = args.step0_tol
        lines.append(f"  step-0 control (released model): evaluated={control_o.get('load', {}).get('evaluated')} max |delta| over {control['n_cells']} stream cells = {control['max_abs_delta']:.3g} -> "
                     f"{'OK' if control['ok'] else 'FAILED: ' + '; '.join(control['reasons'])}")
        js["gate_b"]["_step0_control"] = control
    verdicts = {}
    for label, o in oracles:
        v = judge_oracle(o, control, args.min_chains, args.min_units, args.allow_no_step0)
        verdicts[label] = v
        if v["status"] == "CONTROL_OK" and o is control_o:
            continue
        lines += oracle_report(label, o, v, args.all_terms)
        js["gate_b"][label] = v
    if oracles:
        lines.append("  gate (b) overall (PASS = valid file, all three axes POS):")
        for label, v in verdicts.items():
            if v["status"] == "CONTROL_OK":
                continue
            ax = v["axes"] or {}
            lines.append(f"    {label:<22s} {v['status']:<8s} " + "  ".join(f"{k}={ax.get(k)}" for k in ("structure", "sequence", "interactions")) if ax else f"    {label:<22s} {v['status']}")
    else:
        lines.append("  (no --oracle given)")
    lines.append("")
    lines.append("gate (c)  generated vs RING-measured labels of the decoded designs: not computed here (needs RING on the designs). Generated-label density of the runs:")
    n_ring = 0
    for f in args.eval_log or []:
        for line in open(f, errors="replace"):
            if line.startswith("EVAL ring ") or line.startswith("EVAL RUN "):
                lines.append("  " + line.strip()[:170])
                n_ring += line.startswith("EVAL ring ")
    if not n_ring:
        lines.append("  (no --eval-log given)")
    if getattr(args, "gate_c", None):
        g = json.load(open(args.gate_c))
        lines += gate_c_lines(g)
        js["gate_c"] = {k: g.get(k) for k in ("schema", "params", "natural", "validate_converter")}
        js["gate_c"]["runs"] = {n: {k: v for k, v in r.items() if k not in ("per_design",)} for n, r in (g.get("runs") or {}).items()}
    return lines, js


def _gc(d, nd=3):
    """'v [lo,hi]' of a gate_c.py {value, ci95} dict (stdlib only)."""
    if not isinstance(d, dict) or d.get("value") is None:
        return "n/a"
    ci = d.get("ci95") or [None, None]
    f = lambda x: "nan" if x is None else f"{x:.{nd}f}"  # noqa: E731
    return f"{f(d['value'])} [{f(ci[0])},{f(ci[1])}]"


def gate_c_lines(g: dict) -> list:
    """Gate (c) section from the json of ``s2/gate_c.py`` (schema gate_c/v1): LENGTH-MATCHED densities vs natural, generated-vs-RING agreement (binary and 23-state / type level), controls,
    completeness and the provisional verdict (INVALID(join) > INCOMPLETE > INVALID_STATES > PASS / WEAK / FAIL)."""
    pr = g.get("params", {})
    out = ["  RING-measured on the decoded designs (gate_c.py; strata gap3 = |i-j|>=%s PRIMARY, lr = |i-j|>=%s; F1 micro over designs, 95%% CI = bootstrap over designs):" % (pr.get("min_sep"), pr.get("long_sep"))]
    nat = g.get("natural") or {}
    nd = nat.get("density") or {}
    if nd:
        ld = nat.get("matched_length_distribution") or {}
        out.append(f"  natural reference ({'LENGTH-MATCHED +-%d%%' % round(100 * pr.get('nat_window', 0)) if nat.get('length_matched') else 'NOT length matched (--nat-range)'}): {nd['n']} val-split chains, "
                   f"{nd['density'].get('n_clusters')} sequence clusters, L {ld.get('min')}-{ld.get('max')} (mean {ld.get('mean', float('nan')):.1f}): positives/residue {nd['pos_per_res']['mean']:.3f} "
                   f"[{nd['pos_per_res']['ci95'][0]:.3f},{nd['pos_per_res']['ci95'][1]:.3f}]; density/C(L,2) {nd['density']['mean']:.4f} (length dependent)")
    else:
        out.append(f"  natural reference: UNAVAILABLE ({nat.get('error', 'not computed')})")
    for name, r in sorted((g.get("runs") or {}).items()):
        if "density" not in r:
            out.append(f"  {name:<26s} FAILED: {r.get('error')} ({len(r.get('ring_failures') or {})} RING failures)")
            continue
        d, vn = r["density"]["pos_per_res"], r.get("vs_natural") or {}
        cm = r.get("completeness") or {}
        bad = f"  [RING ok {r['n_ring_ok']}/{r.get('n_archived', r['n_designs'])}]" if r["n_ring_ok"] != r.get("n_archived", r["n_designs"]) else ""
        out.append(f"  {name:<26s} {r['mode']:<9s} RING positives/residue {d['mean']:.3f} [{d['ci95'][0]:.3f},{d['ci95'][1]:.3f}]"
                   + (f" = {vn['pos_per_res']['ratio']:.2f}x length-matched natural [{vn['pos_per_res']['ci95'][0]:.2f},{vn['pos_per_res']['ci95'][1]:.2f}]" if "pos_per_res" in vn else (f" (natural: {vn['error']})" if vn.get("error") else "")) + bad)
        v0 = r.get("gate_c_verdict")
        a = r.get("agreement")
        if v0 and not a:
            out.append(f"      -> {v0['verdict']}: {'; '.join(v0.get('reasons') or [])}")
            continue
        if not a:
            continue
        gd = (r.get("generated") or {}).get("density", {}).get("pos_per_res")
        p, lr = a["gap3"], a["lr"]
        out.append(f"      generated labels positives/residue {gd['mean']:.3f}  | gap3: P {_gc(p['precision'])} R {_gc(p['recall'])} F1 {_gc(p['f1'])}" if gd else "      (generated density n/a)")
        out.append(f"      controls gap3: shuffled-design F1 {_gc(p['f1_shuffled'])} -> dF1 {_gc(p['dF1_shuf'])} | distance: d<={pr['dist_cut']:g}A F1 {_gc(p['f1_dist8'])}, "
                   f"closest-K F1 {_gc(p['f1_distK'])}, gen - best {_gc(p['dF1_dist_best'])}; AUC(-d) {_gc(p['auc_dist'], 2)}")
        out.append(f"      long range: F1 {_gc(lr['f1'])} shuffled {_gc(lr['f1_shuffled'])} dF1 {_gc(lr['dF1_shuf'])} | d<=cut F1 {_gc(lr['f1_dist8'])}; "
                   f"in-contact lift P(ring|gen)-P(ring|no gen) {_gc(p['lift_in_contact'])}")
        out.append(f"      23-state level gap3: exact-state F1 {_gc(p['f1_state23'])}; type agreement of position-matched positives {_gc(p['type_agree'], 2)} vs majority-class {_gc(p['type_majority_baseline'], 2)} "
                   f"(d {_gc(p['dtype_vs_majority'])}, kappa {_gc(p['type_kappa'], 2)}); lr: {_gc(lr['type_agree'], 2)} vs {_gc(lr['type_majority_baseline'], 2)} (d {_gc(lr['dtype_vs_majority'])})")
        cls = [f"{k} F1 {c['f1']['value']:.2f} (n={c['support_ring']})" for k, c in p["classes"].items() if c.get("meaningful")]
        v, st = a["verdict"], a.get("states") or {}
        out.append(f"      classes: {'; '.join(cls)}")
        out.append(f"      completeness: evaluated {cm.get('n_evaluated')}/{cm.get('n_archived')} (RING failures {cm.get('n_ring_failed')}, join missing {cm.get('n_join_missing')}, length mismatch {cm.get('n_length_mismatch')}); "
                   f"informative designs gap3 {v.get('n_informative_gap3')} lr {v.get('n_informative_lr')} (bootstrap draws lacking evidence: max {100 * v.get('max_dropped_draw_frac', 0):.1f}%); "
                   f"impossible states {st.get('invalid_pairs')} pairs (inter-chain {st.get('inter_state_pairs')}, MASK {st.get('mask_pairs')}; {st.get('designs_with_invalid')} designs)")
        out.append(f"      -> {v['verdict']} (if valid: {v.get('verdict_if_valid')}): specific gap3={v['specific_gap3']} lr={v['specific_lr']}; beats distance gap3={v['beats_distance_gap3']} lr={v['beats_distance_lr']}; "
                   f"type beats majority gap3={v.get('type_beats_majority_gap3')} lr={v.get('type_beats_majority_lr')}; adds beyond contact={v['adds_beyond_contact_gap3']}  [provisional rule, no F1 floor]"
                   + ("  " + "; ".join(v["reasons"]) if v.get("reasons") else ""))
    return out


def _clean(x):
    """NaN / inf -> None (strict JSON), tuples / sets -> lists."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", metavar="GROUP[@SEED]=SCORE,TAR")
    p.add_argument("--manifest")
    p.add_argument("--base", default="base")
    p.add_argument("--control", default="", help="label prefix of the control arm (B1000 -> Z1000 with --control Z)")
    p.add_argument("--oracle", action="append", metavar="FILE[=LABEL]")
    p.add_argument("--step0", default="", help="oracle json of the released model (step-0 control of gate (b))")
    p.add_argument("--step0-tol", type=float, default=0.0, help="largest |delta| the step-0 control may show (default: exactly 0)")
    p.add_argument("--allow-no-step0", action="store_true", help="exploration only: do not require a step-0 control for gate (b)")
    p.add_argument("--min-chains", type=int, default=200)
    p.add_argument("--min-units", type=int, default=40, help="minimum number of independent sequence clusters of an oracle file")
    p.add_argument("--expect-n", type=int, default=48, help="designs per run (archive, design_map and score log must all have exactly this many; 0 = use the archive's count)")
    p.add_argument("--eval-log", action="append")
    p.add_argument("--gate-c", default="", help="json of s2/gate_c.py: print its gate (c) section (RING-measured vs generated labels)")
    p.add_argument("--sd", type=float, default=REF_BASE["sd_per_48"], help="default seed SD of a 48-design run in designs (used when < 2 base seeds are supplied)")
    p.add_argument("--require", choices=("both", "designable"), default="both")
    p.add_argument("--all-terms", action="store_true")
    p.add_argument("--json", default="")
    a = p.parse_args(argv)
    a.control_for = lambda label: a.control + re.sub(r"^[A-Za-z]+", "", label)  # B1000 -> Z1000 with --control Z
    lines, js = build_report(a)
    print("\n".join(lines))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(_clean(js), fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
