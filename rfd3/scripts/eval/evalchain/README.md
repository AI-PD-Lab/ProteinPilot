# evalchain - one command, one verdict per checkpoint (HARMONIA-v2, 2026-11-07)

`evalchain.sh` submits the whole pre-registered evaluation of a checkpoint as one SLURM chain and ends with a CPU job that
writes `summary.md` + `summary.json` with every N028 / N030 kill criterion evaluated PASS / FAIL and the GPU-h used. Both live
instances of the session call it the same way, so M1, S1, K1, S2 (and G1) are judged identically.

```
scripts/eval/evalchain/evalchain.sh CKPT TAG --item N0xx [--weights ema|raw] [--seed S] [--full] [--extra "overrides"]
        [--recipe n019|none] [--cfgs "1.5 2.0"] [--n 32] [--skip heldout,ross,design] [--after JOBIDS] [--max-gpuh 6] [--yes]
        [--dry-run] [--overwrite]
```

* `CKPT` any v2 checkpoint (EMA shadow + raw weights inside). `TAG` = tar name `v2/eval/<TAG>.tar.gz`; convention
  `v2_<stage>_e<epoch>[_raw]` (e.g. `v2_M1_e0597`, `v2_M1_e0597_raw`, `v2_S1_e0598`, K1 arms `v2_E2_e0596_k1gamma03`).
* `--item N0xx` = the notebook item that owns the eval; records land in `results/N0xx-*/eval_<TAG>/`.
* `--weights raw`: the sampler always uses the EMA `shadow.*` tensors (foundry `BaseInferenceEngine` -> `trainer.load_checkpoint`,
  `EMA.forward` in eval mode; no engine flag). `raw` makes the GPU job derive, node-locally, a copy with `shadow.* := model.*`
  (`derive_ckpt.py`, verified after writing) and run the unchanged harness on it; nothing is written to /hnvme. M1's plan (N030
  F7) is "e597 EMA and RAW" = two chains, `v2_M1_e0597` and `v2_M1_e0597_raw`.
* `--extra` appends hydra overrides to the N019 recipe for every sampling job (K1 sampler arms on the E2 checkpoint, e.g.
  `--extra "inference_sampler.gamma_0=0.3"`); `--recipe none` drops the N019 potentials.
* `--full` adds seeds 20261015 + 20261016 (parallel, after the seed-1 job) and compares each with the matching N019 seed tag.
* `--after 1257548` makes the summarize job also wait for the baseline held-out job (its tar is needed for the held-out criterion;
  without it the row is UNVERIFIED and `summarize.py --chain ...` can simply be re-run once the tar exists).
* `--dry-run` prints every sbatch line (job ids as `$J_*`) and the estimate; nothing is submitted or written.

## The chain (all GPU jobs: preempt, 2 x H100, 16 CPUs, `--exclude=h14-21,h13-10,h13-21`, walltime <= 2 h)

| stage | script | what | measured (N019) | walltime | expected GPU-h |
|---|---|---|---|---|---|
| eval | `eval_stage.sbatch` -> `scripts/slurm/eval_checkpoint.sbatch` | specs_v2, CFGS "1.5 2.0", N=32, seed 20261014, N019 recipe | 66-68 min (jobs 1209800/1214842/1214843/1216805) | 2:00 | 2.24 |
| eval_sX | same | `--full` seeds 20261015/16, afterok eval | same | 2:00 | 2.24 each |
| heldout | same, SPECS = N029 `probe/specs_beta_heldout.tar.gz` | 4 held-out beta targets x 4 arms x 16, CFG 1.5 | scaled: 256 of 1376 designs + env restore | 1:00 | 0.52 |
| ross | `ross_panel.sbatch` | old-Rossmann stress (full/ssonly/adjonly), seeds 7101/7102, CFG 2.0, 64 per case | 20.6 min (1214844) | 0:45 | 0.68 |
| design | `scripts/eval/designability.sbatch` | eval tar, CFG 1.5 + 2.0, goal arms, 8 per cell, LigandMPNN K=8 -> ESMFold2 | 7.7 min for 120 designs at one CFG (1214841) | 0:45 | 0.60 |
| summarize | `summarize.sbatch` -> `summarize.py` (preempt_cpu, 16 cores) | scoring + report, afterany of everything | ~5-10 min CPU | 1:30 | 0 |

Screen = **~4.0 GPU-h expected** (walltime cap 9.0 GPU-h); `--full` = ~8.5 expected (cap 17). The gate `--max-gpuh` (default 6)
is on the expected figure; a cap > 25 GPU-h needs the user's approval and is refused without `--yes`. Dependencies are
`afterok` (a failed stage cancels its successors via `--kill-on-invalid-dep=yes`), except summarize (`afterany`: a failed stage
still gives a report with UNVERIFIED rows and verdict INCOMPLETE). Walltimes <= 2 h sit inside Helma's PreemptExemptTime, so the
(non-resumable) sampling jobs are never preempted; if one is cancelled, resubmit the chain with `--overwrite` or the single
stage by its line in `submit.log`.

## Criteria (`criteria.json`; evaluated on the seed-1 screen, specs_v2 goal arms full/sparse8/dense0_true, CFG 1.5)

| id | kind | rule | baseline (re-scored each run) |
|---|---|---|---|
| valid | kill | every tar valid (harness meta), job COMPLETED | - |
| sfar15 | kill | SFAR@1.5 >= 0.754 | v2_E2_e0596_h02r1cen2m0 0.754 |
| sandwich_pass | kill | sandwich_1tit pass (goal arms pooled) >= 0.50 | 0.417 |
| sandwich_target | goal | same >= 0.55 (the caller's brief lists 0.55; N028/N030 pre-register the kill at 0.50) | |
| hairpin12_cut | kill | 1 - p12/p12_baseline >= 0.40 | p12 0.292 |
| paired14 | kill | p14 >= 0.70 | 0.594 |
| split | kill | kit split3 SPLIT fraction <= 5 % in every goal arm x CFG (1.5, 2.0) | max 4.4 % |
| breaks | kill | mean ca_breaks <= 0.005 per CFG | 0.000 |
| designability | kill | designable >= 55 % at CFG 2.0 (goal arms x 5 targets x 8) | 59.2 % (71/120) |
| heldout | kill | held-out SFAR@1.5 delta vs v2_E2_e0596_heldout_s1: FAIL only if delta < -0.03 AND the bootstrap 95 % CI excludes 0 | job 1257548 (pending at writing) |
| ross | guardrail | full_rossmann_like SPLIT % (strict kit rule, mean of 2 seeds) <= 30 | 21.1 (E2 29.7, p0694 28.9) |
| sfar15_3seed | goal | `--full`: 3-seed mean >= 0.745 | N019 0.745 |

Verdict: KILL (a kill criterion FAIL) / INCOMPLETE (a kill criterion UNVERIFIED) / PASS. Baselines are always re-scored by the
same code in the same run (N007 kit `score_all.sh` into the shared cache `v2/eval/cache/`, then the N029 `score_anatomy.py`
with `--compare`); the first chain on a fresh cache scores the three baseline tags (~5 min extra).

## Outputs

`results/N0xx-*/eval_<TAG>/`: `chain.json` (args, checkpoint size/mtime, estimate, every stage's job id + sbatch line),
`jobs.json`, `submit.log`, `summary.md`, `summary.json` (criteria rows, metrics, sacct per job, GPU-h total), `anatomy_cfg1.5.csv`,
`anatomy_cfg2.0.csv`, `heldout_cfg1.5.csv`, `guardrails.csv`, `ross.csv`, `<TAG>.cells.csv`, the designability md.
Tars: `v2/eval/<TAG>.tar.gz`, `<TAG>_heldout.tar.gz`, `<TAG>.ross.tar.gz`, `<TAG>_cfg1.5_2.0.designability.{json,md}`
(+ `<TAG>_s2026101x.tar.gz` with `--full`). Logs: `v2/eval/logs/ec_<TAG>_<stage>-<job>.out`. /hnvme inode cost per chain:
~12 files + ~5 per tag in the cache.

## Who calls what (session split of 2026-11-07, N029 item)

* helma5 instance (owner of N030 M1, S1 eval, N032 K1, the held-out pack): `evalchain.sh <M1 e597> v2_M1_e0597 --item N030
  --after 1257548`, `... v2_M1_e0597_raw --item N030 --weights raw --after 1257548`, `... <S1 e598> v2_S1_e0598 --item N028
  --after 1257548`, K1 arms: `evalchain.sh <E2 e596> v2_E2_e0596_k1gamma03 --item N032 --extra "inference_sampler.gamma_0=0.3"`.
* helma3 instance (owner of N033 S2, N031 G1): `evalchain.sh <S2 ckpt> v2_S2_e0597 --item N033`, G1 via `--extra` / a pack of its own
  (the specs_v2 part is what the criteria read).
* Rules: one chain per checkpoint and weight choice; never two chains with the same TAG; each instance submits only its own
  TAGs and waits with `~/.claude/skills/work-log/bin/wait-job.sh <summarize job>`; the record is `eval_<TAG>/summary.md`, which
  `effect.sh` cites. `--full` only after a screen passes (it is 2.1x the cost).

## Validation (CPU, `selftest.sh`, ~5 min; `test_summarize.py`)

`bash selftest.sh [scratch]`: synthetic spec pack + synthetic EMA checkpoint -> `eval_stage.sbatch` with `WEIGHTS=raw DRY=1`
(derive_ckpt verified, harness DRY path with the fake inference, tar valid, meta records the `_raw.ckpt`) -> `ross_panel.sbatch`
DRY=1 (2 cells x 3 cases scored) -> `evalchain.sh --dry-run` (5 lines; `--full` refused without `--yes`; nothing written) ->
unit tests: seed-1 tag vs itself (all deltas 0, CIs cover 0, hairpin cut 0), seed 2 vs seed 1 (within CI, no flag), guardrails
= the N019 record (split 3/480 @1.5, 13/480 @2.0, breaks 0), designability 71/120, Rossmann scorer = the N019 table (25.0 /
17.2 %), derive_ckpt on a synthetic ckpt, sacct parsing, criteria on synthetic metrics, and `summarize()` end to end on a chain
assembled from existing outputs (verdict KILL for the seed-2 baseline, held-out UNVERIFIED).

Not validated (needs a GPU job): `derive_ckpt.py` on a real 2.7 GB checkpoint inside the rfd3 env (the esmfold2 env cannot
unpickle `train_cfg`; the stub-pickle read of E2 e596 shows 814 `model.*` / 814 `shadow.*` keys, 804 pairs differ), the real
walltimes of the derived-checkpoint path (+ one env restore, ~2-3 min), `sacct` from a preempt_cpu node.
