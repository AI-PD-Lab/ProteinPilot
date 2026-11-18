#!/bin/bash
# selftest.sh - CPU-only validation of the evalchain plumbing (no SLURM, no GPU, no real checkpoint), ~5 min:
#   1. synthetic spec pack + synthetic EMA checkpoint (the harness test's own construction);
#   2. eval_stage.sbatch with WEIGHTS=raw DRY=1 (derive_ckpt on the synthetic ckpt + the harness' DRY path with fake inference)
#      -> one tar, meta.valid, ckpt_path *_raw.ckpt;
#   3. ross_panel.sbatch with DRY=1 -> <TAG>.ross.tar.gz with 2 cells x 3 cases, scored by summarize.ross_scores;
#   4. evalchain.sh --dry-run (screen: prints 5 sbatch lines; --full without --yes: refused rc 3);
#   5. python -m unittest test_summarize (cache tags: seed 1 vs itself / seed 2, guardrails, designability, Rossmann, criteria).
# usage: bash selftest.sh [scratch dir]
set -uo pipefail
ES=/path/to/env/bin/python
W=/path/to/work_rfd3; REPO=$W/harmonia_study; EC=$REPO/scripts/eval/evalchain
S=${1:-${TMPDIR:-/tmp}/evalchain_selftest_$$}; mkdir -p "$S/out/logs"
FAKE=$REPO/scripts/eval/tests/fake_run_inference.py
fail=0; ok() { echo "OK   $*"; }; bad() { echo "FAIL $*"; fail=1; }
# the harness tests' conftest.py / fake_run_inference.py import pytest (only @pytest.fixture); the esmfold2 env has none -> stub
mkdir -p "$S/stub"; printf 'def fixture(f=None, **kw):\n    return f if f is not None else (lambda g: g)\n' > "$S/stub/pytest.py"
export PYTHONPATH="$S/stub${PYTHONPATH:+:$PYTHONPATH}"
echo "### 1. synthetic pack + checkpoint in $S"
$ES - "$S" <<'PYEOF' || exit 1
import os, sys
S = sys.argv[1]
sys.path.insert(0, "/path/to/work_rfd3/harmonia_study/scripts/eval")
sys.path.insert(0, "/path/to/work_rfd3/harmonia_study/scripts/eval/tests")
import torch, conftest as cf, build_specs as bs, eval_common as ec, adapter_diag as ad
import biotite.structure.io.pdb as pdb
f = pdb.PDBFile(); f.set_structure(cf.atoms_from_ca(cf.helix_bundle_ca())); p = f"{S}/syn.pdb"; f.write(p)
bs.build(f"{S}/specs_w", refs=[dict(name="bundle_syn", pdb="syn", chain="A", fold="synthetic")], seed=1, ref_files={"bundle_syn": p}, uncond_lengths=(30,))
bs.pack_dir(f"{S}/specs_w", f"{S}/out/specs_syn.tar.gz")
rows = ad.expected_rows(ec.REPO_DEFAULT)["adj"]
sd = {}
for src in ("shadow", "model"):
    sd[f"{src}.{ad.ADJ_KEY}"] = torch.randn(rows, 8); sd[f"{src}.{ad.SS_KEY}"] = torch.randn(4, 8); sd[f"{src}.x.weight"] = torch.randn(2, 2)
torch.save({"model": sd, "train_cfg": {"synthetic": True}}, f"{S}/syn.ckpt")
import json; json.dump({"full_rossmann_like": {"length": "40"}, "ssonly_rossmann_like": {"length": "40"}, "adjonly_rossmann_like": {"length": "40"}}, open(f"{S}/ross_inputs.json", "w"))
print("pack", f"{S}/out/specs_syn.tar.gz", "ckpt rows", rows)
PYEOF
echo "### 2. eval_stage.sbatch WEIGHTS=raw DRY=1"
env CKPT=$S/syn.ckpt TAG=syn_raw WEIGHTS=raw DRY=1 DRY_PY=$ES DRY_INFER=$FAKE SPECS=$S/out/specs_syn.tar.gz OUT_DIR=$S/out TMPBASE=$S \
    N=2 BATCH=2 CFGS="1.5 2.0" NGPU=2 REPO=$REPO bash "$EC/eval_stage.sbatch" > "$S/stage.log" 2>&1; rc=$?
if [ $rc -eq 0 ] && [ -f "$S/out/syn_raw.tar.gz" ]; then
  META=$(tar -xzOf "$S/out/syn_raw.tar.gz" syn_raw/meta.json)
  echo "$META" | grep -q '"valid": true' && echo "$META" | grep -q '_raw.ckpt' && ok "eval stage: tar valid, harness ran on $(echo "$META" | grep -o '[^"]*_raw.ckpt')" || bad "eval stage meta: $META"
  grep -q "verified: 3 shadow/model pairs identical" "$S/stage.log" && ok "derive_ckpt verified the raw copy" || bad "derive_ckpt verification missing (see $S/stage.log)"
else bad "eval stage rc=$rc (see $S/stage.log)"; tail -20 "$S/stage.log"; fi
[ -d "$S/evalchain_local"* ] 2>/dev/null && bad "wrapper temp dir not removed" || ok "wrapper temp dir removed"
echo "### 3. ross_panel.sbatch DRY=1"
env CKPT=$S/syn.ckpt TAG=syn_raw WEIGHTS=raw DRY=1 DRY_PY=$ES DRY_INFER=$FAKE OUT_DIR=$S/out TMPBASE=$S IN=$S/ross_inputs.json NB=2 BS=2 NGPU=2 REPO=$REPO \
    bash "$EC/ross_panel.sbatch" > "$S/ross.log" 2>&1; rc=$?
if [ $rc -eq 0 ] && [ -f "$S/out/syn_raw.ross.tar.gz" ]; then
  $ES - "$S" <<'PYEOF' && ok "ross panel: 2 cells x 3 cases x 4 designs scored" || bad "ross panel scoring"
import sys, importlib.util
S = sys.argv[1]
spec = importlib.util.spec_from_file_location("summarize", "/path/to/work_rfd3/harmonia_study/scripts/eval/evalchain/summarize.py")
sm = importlib.util.module_from_spec(spec); sys.modules["summarize"] = sm; spec.loader.exec_module(sm)
sc = sm.ross_scores(f"{S}/out/syn_raw.ross.tar.gz", procs=4)
cells = {c: v for c, v in sc.items() if not c.startswith("_")}
assert sorted(cells) == ["syn_raw_s7101_cfg2.0", "syn_raw_s7102_cfg2.0"], sorted(cells)
assert all(sorted(v) == ["adjonly_rossmann_like", "full_rossmann_like", "ssonly_rossmann_like"] and all(x["n"] == 4 for x in v.values()) for v in cells.values()), cells
assert sc["_meta"]["valid"] and sc["_meta"]["weights"] == "raw" and sc["_meta"]["run_ckpt"].endswith("_raw.ckpt"), sc["_meta"]
print("ross meta", {k: sc["_meta"][k] for k in ("tag", "weights", "valid", "n_designs_expected_per_cell")}, "full split %", sm.ross_case_mean(sc))
PYEOF
else bad "ross panel rc=$rc (see $S/ross.log)"; tail -20 "$S/ross.log"; fi
unset PYTHONPATH
echo "### 4. evalchain.sh --dry-run"
E2=$(ls $W/v2/E2/train/ss-adj-v2-stageE2/*/ckpt/epoch-0596.ckpt | tail -1)
"$EC/evalchain.sh" "$E2" ec_selftest --item N029 --dry-run > "$S/dry.log" 2>&1; rc=$?
[ $rc -eq 0 ] && [ "$(grep -c '^\[dry-run\]' "$S/dry.log")" -eq 5 ] && ok "screen dry-run: 5 sbatch lines, rc 0, $(grep -o 'EXPECTED [0-9.]*' "$S/dry.log")" || { bad "screen dry-run rc=$rc"; cat "$S/dry.log"; }
"$EC/evalchain.sh" "$E2" ec_selftest --item N029 --dry-run --full > "$S/dry_full.log" 2>&1; rc=$?
[ $rc -eq 3 ] && grep -q REFUSED "$S/dry_full.log" && ok "--full without --yes refused (rc 3, $(grep -o 'EXPECTED [0-9.]*' "$S/dry_full.log"))" || bad "--full refusal rc=$rc"
"$EC/evalchain.sh" "$E2" ec_selftest --item N029 --dry-run --full --yes --weights raw --skip design > "$S/dry_full_yes.log" 2>&1; rc=$?
[ $rc -eq 0 ] && [ "$(grep -c '^\[dry-run\]' "$S/dry_full_yes.log")" -eq 6 ] && grep -q 'afterany:$J_EVAL:$J_S20261015:$J_S20261016:$J_HELDOUT:$J_ROSS ' "$S/dry_full_yes.log" && ok "--full --yes --skip design: 6 lines, summarize afterany of all" || { bad "--full --yes dry-run rc=$rc"; tail -3 "$S/dry_full_yes.log"; }
[ -e "$REPO/.agent-research/results/N029-"*/eval_ec_selftest ] && bad "dry-run wrote a results dir" || ok "dry-run wrote nothing"
echo "### 5. unit tests"
export TMPDIR="$S"     # the unit test's temp cache/results land in the selftest scratch, not on the login node's /scratch root
( cd "$EC" && $ES -m unittest test_summarize > "$S/unittest.log" 2>&1; tail -4 "$S/unittest.log" ) ; grep -q "^OK" "$S/unittest.log" && ok "unit tests" || bad "unit tests (see $S/unittest.log)"
echo "### selftest $([ $fail = 0 ] && echo PASSED || echo FAILED) (scratch $S)"
exit $fail
