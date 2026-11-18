"""HARMONIA guidance for RFD4-Proteina.

``install_guidance`` wraps ``fm.get_clean_pred_n_guided_vector`` so every denoising step runs a conditional pass and a pass with the
conditioning removed (adj_state all MASKED, C_SEC all NONE) on the same random draws, and combines them as

    pred = pred_masked + w * (pred_cond - pred_masked)          # w = 1: untouched upstream path

over SS and ADJ jointly (the RFD3 meaning of classifier-free guidance). Release operating point: w = 1.5.
With ``ref_nn`` the reference pass instead keeps the full conditioning and runs ``ref_nn`` (e.g. stock RFD4) in place of the
model trunk (autoguidance, Karras et al. 2024).

``rfd4_generate.py`` (run it with ``python -m harmonia_ext.rfd4_generate``) puts this on top of ``rfd4-generate``.
"""
from __future__ import annotations

import torch

from harmonia_ext import adj_labels as L


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
