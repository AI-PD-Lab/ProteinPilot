"""HARMONIA-RFD4 generation on the RFD3 HARMONIA spec manifest (natural 5-target panel), written in the RFD3 eval-tar layout so the RFD3
scorer (scripts/eval/score_designs.py, fr.py, split2.py, designability.sbatch) runs unchanged:

    <tag>/designs/cfg<w>/<key>_<batch>_model_<m>.cif.gz        <tag>/meta.json

``w`` = HARMONIA guidance weight: pred = pred_masked + w (pred_cond - pred_masked), where the masked pass sees NO conditioning at all
(adj_state all MASKED, C_SEC all NONE), i.e. the RFD3 meaning of CFG over SS+ADJ jointly. w = 1: one pass, no guidance.

Spec translation (manifest = RFD3 dialect v2): SS 0 helix / 1 strand / 2 loop / 3 masked -> RFD4 C_SEC 1 / 2 / 3 / 0; block pairs
pos -> CONTACT (1), neg -> NON_CONTACT (0), everything else MASKED (2) (same-block pairs, loops, unspecified; the RFD3 'intra_zero' / diag 0
dialect is NOT copied: training here never shows 0 on same-block pairs). Optional: --hot k promotes the k pos pairs with the largest native
n6 to HOT (3) (only for the HOT/COLD effect eval).

  python -m harmonia_ext.generate --ckpt <chk_*-ema.ckpt> | --bundle /w/harmonia_rfd4/base  --tag <tag> --out-tar <path> \
      [--cfgs 1.0,2.0] [--n 16] [--batch 8] [--keys all|<comma list>] [--uncond] [--seed 20261028]
"""
from __future__ import annotations

import argparse
import ast
import gzip
import io
import json
import os
import tarfile
import time

import numpy as np
import torch

from harmonia_ext import adj_labels as L
from harmonia_ext.adj_model import register

register()

MANIFEST = os.environ.get("HARMONIA_MANIFEST", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rfd3", "scripts", "eval", "kit", "manifest.json"))
RFD3_SS_TO_RFD4 = {0: L.SS_HELIX, 1: L.SS_SHEET, 2: L.SS_LOOP, 3: L.SS_NONE}


def _lit(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def promoted(target: dict, arm: dict, hot_k: int = 0, cold_k: int = 0) -> tuple[set, set]:
    """HOT = the hot_k listed contacts with the largest native n6; COLD = the cold_k listed non-contacts with the SMALLEST native min CB
    distance (near-misses: the hardest pairs to keep apart). Same selection in hotcold_score.py."""
    n6 = {k: int(v) for k, v in _lit(target["pair_n6"]).items()}
    dm = {k: float(v) for k, v in _lit(target["pair_min"]).items()}
    key = lambda p: f"{min(p)}-{max(p)}"
    pos = [tuple(p) for p in _lit(arm["pos_pairs"])]
    neg = [tuple(p) for p in _lit(arm["neg_pairs"])]
    hot = set(sorted(pos, key=lambda p: -n6.get(key(p), 0))[:hot_k]) if hot_k else set()
    cold = set(sorted(neg, key=lambda p: dm.get(key(p), 1e9))[:cold_k]) if cold_k else set()
    return hot, cold


def arm_inputs(target: dict | None, arm: dict, hot_k: int = 0, cold_k: int = 0) -> tuple[int, np.ndarray, np.ndarray]:
    """(length, C_SEC [L], adj_state [L,L]) of one manifest arm (or an unconditional key: target None)."""
    n = int(arm["length"])
    sec = np.zeros(n, dtype=np.int64)
    adj = np.full((n, n), L.ADJ_MASKED, dtype=np.int64)
    if "adj_tok" in arm:                         # raw token-space spec (RFD3 benchmark tensors, e.g. the dense-zero old Rossmann stress test):
        sec = np.array([RFD3_SS_TO_RFD4[int(s)] for s in arm["ss_tok"]], dtype=np.int64)   # RFD3 ADJ 0/1/2 = ours, copied verbatim
        return n, sec, np.asarray(arm["adj_tok"], dtype=np.int64)
    if target is None:
        return n, sec, adj
    if arm.get("arm") != "unconditional":
        sec = np.array([RFD3_SS_TO_RFD4[int(s)] for s in _lit(arm["ss"])], dtype=np.int64)
    if str(arm.get("has_adj")) == "True":
        blocks = {int(b["id"]): (int(b["start"]), int(b["stop"])) for b in _lit(target["blocks"])}
        pos = [tuple(p) for p in _lit(arm["pos_pairs"])]
        hot, cold = promoted(target, arm, hot_k, cold_k)
        for pairs, st in ((pos, L.ADJ_CONTACT), (_lit(arm["neg_pairs"]), L.ADJ_NON_CONTACT)):
            for a, b in pairs:
                (a0, a1), (b0, b1) = blocks[int(a)], blocks[int(b)]
                s = L.ADJ_HOT if (a, b) in hot else L.ADJ_COLD if (a, b) in cold else st
                adj[a0:a1, b0:b1] = s                  # block start / stop = half-open residue indices (verified on ferredoxin block 0)
                adj[b0:b1, a0:a1] = s
    return n, sec, adj


def load_model(ckpt: str | None, bundle: str | None, device):
    from rfproteina.proteina import Proteina

    if bundle:
        model = Proteina.from_pretrained(bundle)
        what = f"bundle:{bundle}"
    else:
        import rfproteina.generate as G
        from rfproteina.utils.weights import inference_state_dict

        cfg_exp = G._inference_cfg_exp(ckpt, {})
        model = Proteina.load_from_checkpoint(ckpt, strict=True, autoencoder_ckpt_path=None, load_training_metrics=False, weights_only=False,
                                              map_location="cpu", **({} if cfg_exp is None else {"cfg_exp": cfg_exp}))
        if not ckpt.endswith("-ema.ckpt"):
            raw = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(inference_state_dict(raw, ckpt), strict=True)
            del raw
        what = f"ckpt:{ckpt}"
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"weights": what, "nn_class": type(model.nn).__name__, "adj_embed_absmax": float(model.nn.adj_embed.weight.abs().max())}


def _rng(dev):
    st = {"cpu": torch.get_rng_state()}
    if dev.type == "cuda":
        st["cuda"] = torch.cuda.get_rng_state(dev)
    return st


def _set_rng(st, dev):
    torch.set_rng_state(st["cpu"])
    if "cuda" in st:
        torch.cuda.set_rng_state(st["cuda"], dev)


def install_guidance(fm, sec_key: str, w_ref: dict, model=None, ref_nn=None):
    """Wrap fm.get_clean_pred_n_guided_vector: conditional pass + fully-masked pass (same RNG draws), combined on v / v_guided / x_1 and
    the sequence logits. w_ref['w'] is read per call (1.0 = untouched upstream path).
    ref_nn given = autoguidance (Karras et al. 2024): the reference pass keeps the FULL conditioning but runs ref_nn (e.g. stock RFD4)
    in place of model.nn, so guidance pushes away from the reference model's errors instead of away from 'no conditioning'."""
    orig = fm.get_clean_pred_n_guided_vector

    def guided(batch, predict_for_sampling, guidance_w):
        w = float(w_ref["w"])
        if w == 1.0:
            return orig(batch, predict_for_sampling, guidance_w)
        adj, sec = batch["adj_state"], batch.get(sec_key)
        dev = adj.device
        r0 = _rng(dev)
        c = orig(batch, predict_for_sampling, guidance_w)
        r1 = _rng(dev)
        _set_rng(r0, dev)
        if ref_nn is not None:
            own_nn, model.nn = model.nn, ref_nn
            try:
                u = orig(batch, predict_for_sampling, guidance_w)
            finally:
                model.nn = own_nn
                _set_rng(r1, dev)
        else:
            pm = batch["padding_mask"].bool()
            batch["adj_state"] = torch.where(pm[:, :, None] & pm[:, None, :], torch.full_like(adj, L.ADJ_MASKED), torch.full_like(adj, L.ADJ_PAD))
            if sec is not None:
                batch[sec_key] = torch.zeros_like(sec)
            try:
                u = orig(batch, predict_for_sampling, guidance_w)
            finally:
                batch["adj_state"] = adj
                if sec is not None:
                    batch[sec_key] = sec
                _set_rng(r1, dev)
        for dm, d in c.items():
            if not isinstance(d, dict) or dm not in u:
                continue
            for k in ("v", "v_guided", "x_1", "logits"):
                if k in d and k in u[dm] and torch.is_tensor(d[k]):
                    d[k] = u[dm][k] + w * (d[k] - u[dm][k])
        return c

    fm.get_clean_pred_n_guided_vector = guided


def cif_gz(atoms) -> bytes:
    from biotite.structure.io import pdbx

    f = pdbx.CIFFile()
    pdbx.set_structure(f, atoms)
    buf = io.StringIO()
    f.write(buf)
    return gzip.compress(buf.getvalue().encode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--bundle")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-tar", required=True)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--cfgs", default="1.0,2.0")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--keys", default="all")
    ap.add_argument("--uncond", action="store_true", help="also the unconditional L64/120/200 keys (first cfg only)")
    ap.add_argument("--hot", type=int, default=0)
    ap.add_argument("--cold", type=int, default=0)
    ap.add_argument("--cath", default="", help="JSON {target: CATH code}: RFD4's native C_CATH (C.A.T levels) on every residue of that target's arms")
    ap.add_argument("--ref-bundle", default="", help="autoguidance: reference model (e.g. stock RFD4 bundle) for the guidance pass, same conditioning")
    ap.add_argument("--ref-ckpt", default="", help="autoguidance: reference EMA ckpt (alternative to --ref-bundle)")
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--tsr-k", type=float, default=1.5)
    ap.add_argument("--configs-dir", default="/fork/configs")
    ap.add_argument("--monomer-cif", default="/fork/data/benchmarks/monomer/monomer-short.cif.zst")
    a = ap.parse_args()
    assert bool(a.ckpt) != bool(a.bundle), "exactly one of --ckpt / --bundle"

    from rfproteina.conditions import C_CATH, C_SEC
    from rfproteina.search.vanilla import VanillaStrategy

    from triadj_ext.s2 import cond_gen as CG

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    model, meta = load_model(a.ckpt, a.bundle, dev)
    model.inf_cfg = CG.compose_inference_cfg(a.configs_dir, [f"generation.model.local_latents.simulation_step_params.tsr_k={a.tsr_k}"])
    model.guidance_model = None
    w_ref = {"w": 1.0}
    ref_nn = None
    if a.ref_bundle or a.ref_ckpt:
        assert not (a.ref_bundle and a.ref_ckpt), "at most one of --ref-bundle / --ref-ckpt"
        ref, meta["ref"] = load_model(a.ref_ckpt or None, a.ref_bundle or None, dev)
        ref_nn = ref.nn
        del ref
    install_guidance(model.fm, C_SEC.full_name, w_ref, model, ref_nn)
    src = CG.DeNovoBatches(a.monomer_cif)
    M = json.load(open(a.manifest))
    cath = {t: cath_index(c) for t, c in json.loads(a.cath).items()} if a.cath else {}
    keys = list(M["arms"]) if a.keys == "all" else a.keys.split(",")
    cfgs = [float(x) for x in a.cfgs.split(",")]
    assert all(abs(float(f"{c:.1f}") - c) < 1e-9 for c in cfgs) and len({f"{c:.1f}" for c in cfgs}) == len(cfgs), \
        f"guidance weights must be distinct at 1 decimal (tar names are cfg<w:.1f>): {cfgs}"   # red-team N001-2 #7
    jobs = [(c, k) for c in cfgs for k in keys] + ([(cfgs[0], k) for k in M["unconditional"]] if a.uncond else [])
    print(f"H4GEN start {meta} jobs={len(jobs)} n={a.n} load_s={time.time() - t0:.0f}", flush=True)
    members = {f"{a.tag}/meta.json": json.dumps({"tag": a.tag, "model": meta, "manifest": a.manifest, "cfgs": cfgs, "n": a.n, "batch": a.batch, "keys": keys, "hot": a.hot, "cold": a.cold,
                                                   "seed": a.seed, "tsr_k": a.tsr_k, "cath": a.cath, "guidance": "autoguidance: reference model, same conditioning" if ref_nn is not None else "masked pass = adj all MASKED + C_SEC all NONE"}, indent=1).encode()}
    for ji, (cfg, key) in enumerate(jobs):
        arm = M["arms"].get(key) or M["unconditional"][key]
        tgt = M["targets"].get(arm.get("target")) if key in M["arms"] else None
        n_res, sec, adj = arm_inputs(tgt, arm, a.hot, a.cold)
        w_ref["w"] = cfg
        k = 0
        for b in range(0, a.n, a.batch):
            rows = min(a.batch, a.n - b)
            batch = src.batch([f"{key}_{b + r}" for r in range(rows)], [n_res] * rows, keep_atom_array=True)
            batch = CG_move(batch, dev)
            batch["adj_state"] = torch.as_tensor(adj, dtype=torch.int16, device=dev)[None].expand(rows, n_res, n_res).contiguous()
            batch[C_SEC.full_name] = torch.as_tensor(sec, dtype=torch.long, device=dev)[None].expand(rows, n_res).contiguous()
            if arm.get("arm") != "unconditional" and arm.get("target") in cath:   # same in the guided and the masked pass (guidance is over SS+ADJ only)
                batch[C_CATH.full_name] = torch.tensor(cath[arm["target"]], dtype=torch.int32, device=dev).expand(rows, n_res, 3).contiguous()
            torch.manual_seed(a.seed + 7919 * ji + b)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out = VanillaStrategy().generate(model, batch)
            for r, res in enumerate(out):
                members[f"{a.tag}/designs/cfg{cfg:.1f}/{key}_{(b + r) // 4}_model_{(b + r) % 4}.cif.gz"] = cif_gz(res["generated_atom_array"])
                k += 1
        print(f"H4GEN {ji + 1}/{len(jobs)} cfg={cfg} key={key} L={n_res} designs={k} el_s={time.time() - t0:.0f}", flush=True)
    tmp = a.out_tar + ".tmp"
    with tarfile.open(tmp, "w:gz") as tf:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    os.replace(tmp, a.out_tar)
    print(f"H4GEN DONE {a.out_tar} members={len(members)} s={time.time() - t0:.0f}", flush=True)


def cath_index(code: str) -> list[int]:
    """CATH code -> [C, A, T] class indices of RFD4's cath_label_mapping.pkl (as datapipes/pipeline.py; -1 = level not given/unknown)."""
    import pickle

    from rfproteina.datapipes.pipeline import CATH_MAPPING_PATH
    from rfproteina.utils.fold import extract_cath_code_by_level

    m = pickle.load(open(CATH_MAPPING_PATH, "rb"))
    out = [m[lv].get(extract_cath_code_by_level(code, lv), -1) for lv in ("C", "A", "T")]
    assert all(x >= 0 for x in out), (code, out)
    return out


def CG_move(batch, dev):
    from triadj_ext.s2.design_with_contacts import move_batch

    return move_batch(batch, dev)


if __name__ == "__main__":
    main()
