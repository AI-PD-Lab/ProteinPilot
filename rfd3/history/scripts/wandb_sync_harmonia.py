#!/usr/bin/env python
"""Replay RFD3 SS/ADJ training metrics into W&B FROM THE LOGIN NODE.

Compute nodes have no internet, so training jobs cannot talk to wandb directly.
The jobs do write to shared storage, so this script reads what they wrote and
pushes it from a node that does have internet. Idempotent and incremental: it
records the last step pushed per run in a state file and only sends newer ones,
so it is safe to run from cron every few minutes while a job is still training.

Two metric sources, preferred in this order:
  1. lightning_logs/version_0/metrics.csv  -- structured, written by CSVLogger.
     Available for the restart phases and phase 5.
  2. the SLURM stdout log                  -- the rich-formatted loss tables.
     The only source for phase 3 / phase 4, which predate the CSV logger here.

One wandb run per TRAINING PHASE, not per SLURM job: a phase is the meaningful
unit, and the multiple job ids inside one phase are just 24h-walltime restarts
(plus, in a few cases, startup failures that trained nothing). Steps are keyed on
absolute epoch so restarts stitch together continuously instead of resetting.

Usage (login node, rfd3_ssadj_test env):
  python scripts/wandb_sync_harmonia.py --dry-run   # parse + report, no upload
  python scripts/wandb_sync_harmonia.py             # incremental push (cron-safe)
  python scripts/wandb_sync_harmonia.py --reset     # forget state, full backfill
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys, time
from glob import glob
from pathlib import Path

ENTITY = os.environ.get("WANDB_ENTITY", "Protein-Design-Tools")
PROJECT = os.environ.get("WANDB_PROJECT", "HARMONIA")
CKPT_ROOT = Path("/path/to/checkpoints")
HNVME = Path("/path/to/work_rfd3")
STATE = CKPT_ROOT / ".wandb_sync_harmonia_state.json"

# phase -> (wandb run id, slurm log dir, [job ids oldest->newest], csv run dirs, steps/epoch, notes)
# Jobs that died during startup and trained zero steps are deliberately omitted.
PHASES = [
    dict(
        rid="ss-adj-phase3", name="phase3-adjacency-loss-v2",
        logdir=CKPT_ROOT / "ss_adj_phase3_local/logs", jobs=["590351"],
        csv_globs=[], steps_per_epoch=2400,
        notes="First run of the block-pair AdjacencyContactLoss (from epoch-0645). "
              "Produced epoch-0651. Earlier job 578182 used the buggy token-pair loss "
              "that collapsed structures and is excluded.",
    ),
    dict(
        rid="ss-adj-phase4-motif", name="phase4-motif-recovery",
        logdir=CKPT_ROOT / "ss_adj_phase4_motif_recovery/logs",
        jobs=["608188", "610801", "615945", "632877"],
        csv_globs=[f"{HNVME}/ss_adj_phase4_motif_recovery/logs/train/ss-adj-phase4-motif-recovery/*/lightning_logs/version_0/metrics.csv"],
        steps_per_epoch=2400,
        notes="Restored island/tipatom/ppi motif conditioning. epoch-0651 -> epoch-0670. "
              "epoch-0670 is the checkpoint everything downstream is based on.",
    ),
    dict(
        rid="ss-adj-restart-phase1", name="restart-phase1-adapters",
        logdir=CKPT_ROOT / "ss_adj_restart/logs", jobs=["643653"],
        csv_globs=[f"{HNVME}/ss_adj_restart/logs/train/ss-adj-restart-phase1/*/lightning_logs/version_0/metrics.csv"],
        steps_per_epoch=2400,
        notes="Adapter-only warmup from rfd3_latest, trunk frozen (1,920 trainable params).",
    ),
    dict(
        rid="ss-adj-restart-phase2", name="restart-phase2-full-model",
        logdir=CKPT_ROOT / "ss_adj_restart/logs", jobs=["665381", "670044"],
        csv_globs=[f"{HNVME}/ss_adj_restart/logs/train/ss-adj-restart-phase2/*/lightning_logs/version_0/metrics.csv"],
        steps_per_epoch=3000,
        notes="Full 336M-param unfreeze. Abandoned: the restart premise turned out to be "
              "a config/checkpoint attention mismatch, not catastrophic forgetting.",
    ),
    dict(
        rid="ss-adj-phase6-typed", name="phase6-typed-adjacency",
        logdir=CKPT_ROOT / "ss_adj_phase6_typed/logs",
        jobs=["734963", "736021", "736893", "736897", "741108"],
        csv_globs=[f"{HNVME}/ss_adj_phase6_typed/logs/train/ss-adj-phase6-typed/*/lightning_logs/version_0/metrics.csv"],
        steps_per_epoch=2400,
        notes="From epoch-0684. First VALIDATED run in this lineage (phases 3-5 had "
              "validation disabled three ways) -- val is stratified protein / ligand / "
              "nucleic-acid. Activates use_cb (score C-beta, the atom the label is "
              "defined on), negative_weight=0.5 (native-matched repulsive arm for "
              "explicit non-contacts), detach_motif, and typed_blocks (ligand + NA "
              "adjacency). Label semantics also change: loop/ligand/NA pairs are masked "
              "instead of asserted non-adjacent, motif_region_mask_prob 0.5->0.2, and "
              "ADJ-only reveals negatives. NOT comparable with the 0645-0684 lineage, "
              "and several variables move at once by design.",
    ),
    dict(
        rid="ss-adj-phase5-geometry", name="phase5-contact-well",
        logdir=CKPT_ROOT / "ss_adj_phase5_geometry/logs", jobs=["688838"],
        csv_globs=[f"{HNVME}/ss_adj_phase5_geometry/logs/train/ss-adj-phase5-geometry/*/lightning_logs/version_0/metrics.csv"],
        steps_per_epoch=2400,
        notes="LIVE. From epoch-0670 with one change: AdjacencyContactLoss is a "
              "flat-bottomed well (zero force in 4.5-6.0 A) instead of a one-sided "
              "attractor that compressed contacts to 3.81 A and strained the backbone.",
    ),
    dict(
        rid="ss-adj-phase7-loop-ab", name="phase7-loop-adjacency-antibody",
        logdir=CKPT_ROOT / "ss_adj_phase7_loop_ab/logs",
        jobs=["743581", "750195", "752724"],
        csv_globs=[f"{HNVME}/ss_adj_phase7_loop_ab/logs/train/ss-adj-phase7-loop-ab/*/lightning_logs/version_0/metrics.csv"],
        steps_per_epoch=2400,
        notes="LIVE. From epoch-0694. First run in this lineage where LOOP pairs are "
              "real adjacency labels instead of masked (ss_adj_loop_adj: true), keeping "
              "phase 6's typed_blocks, plus mild antibody upsampling (5% of the train "
              "mix) for antibody-target interfaces. Expect the adjacency loss to START "
              "ABOVE phase 6 rather than continuing its curve -- loop pairs add labels "
              "to fit, so the task is strictly harder and the two are not comparable "
              "point-for-point. Job 743581 covered epochs 695-700 and hit the 24h wall "
              "mid-701, which is what motivated the --requeue/USR1 autorestart; 750195 resumes from epoch-0700 and requeues itself at each wall.",
    ),
    dict(
        rid="ss-adj-phase8-interface", name="phase8-interface-conditioning",
        logdir=CKPT_ROOT / "ss_adj_phase8_interface/logs",
        jobs=["755920", "755922", "755944", "756647", "756656", "756694", "756911", "756926", "756944"],
        csv_globs=[f"{HNVME}/ss_adj_phase8_interface/logs/train/ss-adj-phase8-interface/*/lightning_logs/version_0/metrics.csv"],
        steps_per_epoch=2400,
        notes="LIVE. From epoch-0709. Targets the FIXED-TARGET interface path, isolated "
              "by three controls as the only failing one: long monomers stay clean to 400 "
              "tokens (recovery 1.000, 0 clashes) and de novo dimers condition on "
              "cross-chain adjacency perfectly (recovery 1.000, violation 0.000 on a "
              "forbidden-interface control), while binder-vs-fixed-target degrades with "
              "target size (0.688 -> 0.312) and 100% of its clashes are binder<->target "
              "interpenetration. Changes: ppi 17.4% -> 41% of conditions and now admits "
              "protein-NA interfaces (previously rejected outright -- why NA non-contact "
              "never moved); target-internal block pairs excluded from the loss mean "
              "(detached, diluting the learnable gradient ~7x); cross-chain pairs pinned "
              "in CFG dropout (39% -> 62% survival); a new SS-only motif mask training the "
              "real binder call (target SS masked + cross-chain adjacency specified), which "
              "NO prior mode produced; a native-matched steric arm on designed<->motif "
              "pairs regardless of adjacency state; crop_size 256 -> 384 (256 was the "
              "pretrain value). Expect a transient loss bump: crop size and label mix "
              "both change at once.",
    ),
    dict(
        rid="ss-adj-phase9-spots", name="phase9-adjacency-spots",
        logdir=CKPT_ROOT / "ss_adj_phase9_spots/logs",
        jobs=["767352", "767359", "768594", "768638", "769788", "769824", "770132",
              "771062", "774584", "776478", "779462", "785314", "786564"],
        csv_globs=[f"{HNVME}/ss_adj_phase9_spots/logs/train/*/*/lightning_logs/version_0/metrics.csv"],
        steps_per_epoch=2400,
        notes="LIVE, resumed to target 745. Job 779462 exited cleanly at 734 because run_phase9_prod.sh hardcoded max_epochs=734 while the config declared 745; 785314 resumes from epoch-0733 with the runner corrected. From epoch-0717/0719. Adds four adjacency states so a caller can "
              "single out the pairs carrying the design intent -- 3/4 SUPER (same "
              "min-distance objective as 1/0, weighted up) and 5/6 HOTSPOT/COLDSPOT "
              "(objective CHANGES: a hotspot maximises interface AREA against the "
              "native contact count rather than being satisfied by one close C-beta; a "
              "coldspot repels EVERY pair in the block, not just its closest). Plus "
              "atom-level coldspots (target regions to avoid, not a CFG feature) and "
              "type-aware steric floors -- protein 3.5 / NA 5.0 / ligand 4.5 A, the NA "
              "value measured on 1LMB where a genuine vdW contact never puts the "
              "protein C-beta closer than 5.17 A to a nucleotide's representative "
              "atom. NEW METRICS: adjacency_n_{super,hotspot,coldspot}_pairs and "
              "adjacency_{hotspot_area,coldspot}_loss -- if the *_n_* series sit at "
              "zero the promotion policy is not firing and the whole feature is inert, "
              "which is exactly how the first two smoke tests failed. Not comparable "
              "with phase 8: the loss gains two terms and the label distribution "
              "changes.",
    ),
    dict(
        rid="ss-adj-phase8-fixes", name="phase8-steric-fixes",
        logdir=CKPT_ROOT / "ss_adj_phase8_fixes/logs",
        jobs=["785258", "786563"],
        csv_globs=[f"{HNVME}/ss_adj_phase8_fixes/logs/train/*/*/lightning_logs/version_0/metrics.csv"],
        steps_per_epoch=2400,
        notes="LIVE. From epoch-0717_phase8. The BUG-FIX-ONLY cut: phase 8 in every "
              "other respect -- same config, same conditions, same three adjacency "
              "states (0/1/2), none of phase 9's SUPER/HOTSPOT/COLDSPOT vocabulary and "
              "no atom-level coldspots. Isolates whether the steric fixes alone close "
              "the remaining clash gap, which phase 9 confounds with four new states. "
              "Three fixes to the steric arm in AdjacencyContactLoss: (1) type-aware "
              "floors, protein 3.5 / NA 5.0 / ligand 4.5 A, replacing a single 3.5 A "
              "floor below the closest approach a genuine protein-NA vdW contact ever "
              "makes to a nucleotide's representative atom, so NA interpenetration was "
              "scoring as clash-free; (2) normalisation by the designed-token count "
              "instead of the violating-pair count, which made the penalty "
              "non-monotonic -- adding a clash could LOWER the loss by raising the "
              "denominator; (3) NaN/inf safety on padded distances, which reached the "
              "clamp as NaN and silently zeroed the term for the whole batch. Requires "
              "the new token_mol_type feature (0 protein / 1 NA / 2 ligand) plumbed "
              "through design_transforms. Target epoch 723, ~6 epochs. Branch "
              "phase8-steric-fixes.",
    ),
]

# Everything is keyed on ABSOLUTE FRACTIONAL EPOCH, scaled by EPOCH_SCALE to give
# wandb the monotonic integer step it requires. Epoch is the only axis comparable
# across these phases: the raw optimizer step restarts per phase and the phases use
# different epoch sizes (9600 vs 12000 examples), whereas epoch numbers are
# continuous down the lineage and are what the checkpoints are named after.
EPOCH_SCALE = 1000

def _wstep(frac_epoch: float) -> int:
    return int(round(frac_epoch * EPOCH_SCALE))

# ---- source 1: CSVLogger ---------------------------------------------------
# The CSV mixes two step semantics in one column: `train/batch_mean/*` rows carry
# the global optimizer step, while `train/per_epoch_*` rows carry the epoch number.
# Logging both against the raw column would pile every epoch metric at x<1000 while
# batch metrics sit near x=400000. So we recover the epoch boundaries in file order
# and convert global steps to fractional epochs.
def from_csv(paths: list[Path], default_spe: int) -> dict[int, dict]:
    out: dict[int, dict] = {}
    per_file = []
    for p in sorted(paths):
        try:
            per_file.append((p, list(csv.DictReader(open(p)))))
        except FileNotFoundError:
            continue

    # Pass 1 collects epoch boundaries -- (epoch, global step where it ended) --
    # across EVERY file in the phase, not per file. Lightning restores global_step
    # on resume (verified: epoch 676 ended at 409425 and the next job's first row
    # is 409425), so one anchor is valid for the whole phase. Anchoring per file
    # meant a freshly-started job had no boundary yet and got skipped entirely,
    # hiding its first epoch of metrics until it happened to finish one.
    bounds = []
    for _, rows in per_file:
        last_g = None
        for r in rows:
            s = int(float(r["step"]))
            if any(k.startswith("train/batch_mean/") and v not in ("", None)
                   for k, v in r.items()):
                last_g = s
            if any(k.startswith("train/per_epoch_") and v not in ("", None)
                   for k, v in r.items()) and last_g is not None:
                bounds.append((s, last_g))
    if not bounds:
        return out
    bounds.sort()
    deltas = [bounds[i + 1][1] - bounds[i][1] for i in range(len(bounds) - 1)]
    spe = deltas[0] if deltas and len(set(deltas)) == 1 else default_spe
    e_ref, g_ref = bounds[0]

    # Pass 2: emit. Epoch E occupies [E, E+1], so its final batch lands on E+1.
    # Classify each ROW rather than allow-listing key prefixes: a row carrying any
    # `train/batch_mean/*` value is step-indexed, anything else is epoch-indexed.
    # Validation metrics (`val/*`, first produced by phase 6 -- phases 3-5 never
    # validated) are then picked up automatically. A prefix allow-list would have
    # dropped them silently, which is exactly the failure mode worth avoiding on
    # the first run that actually has a val set.
    for _, rows in per_file:
        for r in rows:
            s = int(float(r["step"]))
            vals = {k: float(v) for k, v in r.items()
                    if k != "step" and v not in ("", None)}
            if not vals:
                continue
            # Epoch-indexed rows are the ones Lightning writes at epoch end:
            # `train/per_epoch_*` and (from phase 6) validation metrics. Everything
            # else -- batch means AND standalone `train/learning_rate` rows -- is
            # step-indexed. Classifying by "absence of batch_mean" instead was wrong:
            # learning-rate-only rows carry a global step (e.g. 395000) which was
            # then read as an epoch number, pushing ranges to epoch 395001.
            is_epoch_row = any(
                k.startswith("train/per_epoch_") or k.startswith("val")
                for k in vals
            )
            fe = (s + 1.0) if is_epoch_row else ((e_ref + 1) + (s - g_ref) / spe)
            out.setdefault(_wstep(fe), {}).update({**vals, "epoch": fe})
    return out

# ---- source 2: the rich loss tables in SLURM stdout ------------------------
HDR = re.compile(r"Epoch\s+(\d+)\s+Batch\s+(\d+)\s+\[")
ROW = re.compile(r"^│\s*([A-Za-z0-9 /_.\-]+?)\s*│\s*(-?[\d.]+)\s*│")
SKIP = ("rank", "train loss name", "loss name", "value")

def _key(label: str) -> str:
    return "train/batch_mean/" + label.strip().lower().replace(" ", "_")

def from_stdout(logs: list[Path], steps_per_epoch: int) -> dict[int, dict]:
    """Parse `Epoch N Batch M` headers and the loss table following each."""
    out: dict[int, dict] = {}
    for lp in logs:
        if not lp.exists():
            continue
        cur = None
        for line in lp.read_text(errors="ignore").splitlines():
            m = HDR.search(line)
            if m:
                ep, ba = int(m.group(1)), int(m.group(2))
                fe = ep + ba / steps_per_epoch
                cur = _wstep(fe)
                out.setdefault(cur, {})["epoch"] = fe
                continue
            if cur is None:
                continue
            r = ROW.match(line)
            if r:
                label, val = r.group(1), r.group(2)
                if label.strip().lower().startswith(SKIP):
                    continue
                try:
                    out[cur][_key(label)] = float(val)
                except ValueError:
                    pass
    # drop entries that only carry the epoch marker (header with no parsed table)
    return {k: v for k, v in out.items() if len(v) > 1}

# ---- main ------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--only", help="sync just this run id")
    args = ap.parse_args()

    state = {} if args.reset or not STATE.exists() else json.loads(STATE.read_text())

    for ph in PHASES:
        if args.only and ph["rid"] != args.only:
            continue
        csv_paths: list[Path] = []
        for g in ph["csv_globs"]:
            csv_paths += [Path(p) for p in sorted(glob(g))]
        pts = from_csv(csv_paths, ph["steps_per_epoch"])
        src = "metrics.csv"
        if not pts:
            pts = from_stdout([ph["logdir"] / f"train-{j}.out" for j in ph["jobs"]],
                              ph["steps_per_epoch"])
            src = "slurm stdout"

        last = state.get(ph["rid"], -1)
        new = sorted(s for s in pts if s > last)
        eps = [v["epoch"] for v in pts.values() if "epoch" in v]
        print(f"{ph['rid']:26s} src={src:13s} parsed={len(pts):5d}  new={len(new):5d}"
              f"  epoch {min(eps):.2f}..{max(eps):.2f}" if eps else "  (no data)")
        if args.dry_run or not new:
            continue

        import wandb
        # The login node's link to wandb is slow enough that the 90 s default
        # init timeout trips intermittently; retry rather than lose the tick.
        run = None
        for attempt in range(3):
            try:
                run = wandb.init(
                    entity=ENTITY, project=PROJECT, id=ph["rid"], name=ph["name"],
                    resume="allow", notes=ph["notes"],
                    config=dict(phase=ph["rid"], jobs=ph["jobs"], source=src,
                                epoch_scale=EPOCH_SCALE),
                    settings=wandb.Settings(silent=True, init_timeout=300))
                break
            except Exception as e:
                # A timed-out init still claims the run id server-side, so an
                # immediate retry fails with "run ID is in use". The claim clears
                # once the stale heartbeat expires -- back off well past that
                # rather than burning all three attempts on the same lock.
                print(f"    init attempt {attempt + 1}/3 failed: {type(e).__name__}: {e}")
                if attempt < 2:
                    time.sleep(90)
        if run is None:
            print(f"    SKIP {ph['rid']} -- could not reach wandb; will retry next tick")
            continue
        # x-axis: every metric plotted against absolute epoch rather than the
        # scaled integer step, so charts read in checkpoint numbering.
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        for s in new:
            run.log(pts[s], step=s)
        run.finish()
        state[ph["rid"]] = new[-1]
        STATE.write_text(json.dumps(state, indent=1))
        print(f"    pushed -> {ENTITY}/{PROJECT}/{ph['rid']} (through step {new[-1]})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
