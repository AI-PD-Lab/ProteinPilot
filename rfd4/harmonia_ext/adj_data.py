"""HARMONIA training rows for RFD4: PDB structures through RFD4's own parse + design pipeline, plus SS (C_SEC) and ADJ labels.

The structure source is the PDB manifest / CIF tar loader of ``triadj_ext.ring_data.RingDataset`` used with ``ring_labels=False``:
no RING record is ever opened or used. What this class adds:

* BEFORE the pipeline (full, uncropped chains): P-SEA SS (RFD4's ``annotate_secondary_structure``, the same annotator as its C_SEC
  channel; atom annotation ``harm_sse``), the curriculum mode, and the C_SEC annotation of the shown SS (``adj_labels.reveal_ss``).
  Annotations survive RFD4's crop / atomisation (verified by test).
* AFTER the pipeline (token space): SS blocks on the FINAL tokens (runs of one H/E code, same chain, consecutive res_id), reconstructed-CB
  (RFD1 rule, always the ideal CB from N/CA/C) block contacts -> ``feats['adj_state']`` [I, I] (``adj_labels`` states; atomised tokens,
  unindexed copies and non-protein tokens are MASKED).

``forced_mode`` fixes the curriculum mode (validation sets: 'full' vs 'uncond' measures how much the model uses the labels).
"""
from __future__ import annotations

import numpy as np
import torch

from harmonia_ext import adj_labels as L
from harmonia_ext.adj_model import register_specs
from triadj_ext.ring_data import TOKEN_TYPE_PROTEIN, RingDataset

register_specs()


class HarmoniaDataset(RingDataset):
    def __init__(self, *args, forced_mode: str | None = None, p_hot: float = 0.5, p_cold: float = 0.5, gate_compact: bool = False,
                 p_dense_zero: float = 0.0, dense_modes: tuple = ("full",), **kwargs):
        kwargs["ring_labels"] = False                 # structure loader only; RING is never read
        kwargs.setdefault("name", "harmonia")
        super().__init__(*args, **kwargs)
        assert forced_mode in (None,) + L.MODES, forced_mode
        self.forced_mode, self.p_hot, self.p_cold = forced_mode, float(p_hot), float(p_cold)
        self.gate_compact, self.p_dense_zero = bool(gate_compact), float(p_dense_zero)
        self.dense_modes = tuple(dense_modes)          # R2: ("full",); R3 adds "adj_only" (the Rossmann adj-only stress arm is that dialect)
        assert set(self.dense_modes) <= {"adj_only", "full", "sparse"}, self.dense_modes
        self._keep_aa = bool(self.return_atom_array)
        self._pending = None

    def __getstate__(self):
        d = super().__getstate__()
        d["_pending"] = None
        return d

    # ---- pre-pipeline hook --------------------------------------------------------------------------------------------
    def _pipeline(self):
        inner = super()._pipeline()

        def run(data):
            data = self._annotate(data)
            return inner(data)

        return run

    def _annotate(self, data: dict) -> dict:
        from atomworks.ml.utils.token import get_token_starts  # noqa: F401  (import check only)
        from biotite.structure import get_residue_starts, spread_residue_wise

        from rfproteina.conditions import C_SEC
        from rfproteina.metrics.sse_metrics import annotate_secondary_structure

        aa = annotate_secondary_structure(data["atom_array"], annotation_name="harm_sse", is_valid_annotation_name="harm_sse_valid")
        rs = get_residue_starts(aa)
        ss = np.asarray(aa.harm_sse[rs]).astype(np.int64)      # SS on the full chain (P-SEA needs context); blocks are built AFTER the crop
        rng = np.random.default_rng([int(data.get("seed", 0)), 0x4A12])
        mode = L.sample_mode(rng, self.forced_mode)
        shown = L.reveal_ss(ss, mode, rng)
        C_SEC.set_annotation(aa, spread_residue_wise(aa, shown).astype(int))
        data["atom_array"] = aa
        self._pending = {"mode": mode, "rng": rng, "n_ss_shown": int((shown > 0).sum()), "n_res": int(len(ss))}
        return data

    # ---- post-pipeline: token-space ADJ ----------------------------------------------------------------------------------
    def build_sample(self, i: int, seed: int | None = None) -> dict:
        self.return_atom_array = True
        try:
            sample = super().build_sample(i, seed)
        finally:
            self.return_atom_array = self._keep_aa
        oa = sample["atom_array"] if self._keep_aa else sample.pop("atom_array")
        pend, self._pending = self._pending, None
        assert pend is not None, "pre-pipeline hook did not run"
        feats = sample["feats"]
        adj, st = token_adj(oa, feats, pend["mode"], pend["rng"], self.p_hot, self.p_cold, self.gate_compact, self.p_dense_zero,
                            self.dense_modes)
        if st["gated"]:                                   # non-compact row: no SS and no ADJ (it trains the unconditional task)
            from rfproteina.conditions import C_SEC

            if C_SEC.full_name in feats:
                feats[C_SEC.full_name] = torch.zeros_like(torch.as_tensor(feats[C_SEC.full_name]))
            C_SEC.set_annotation(oa, np.zeros(oa.array_length(), dtype=int))   # returned atom arrays agree (red-team N003 #5)
        feats["adj_state"] = torch.from_numpy(adj.astype(np.int16))
        sample["harm_stats"] = {**st, "mode": pend["mode"], "n_ss_shown": 0 if st["gated"] else pend["n_ss_shown"], "n_res": pend["n_res"]}
        return sample


def token_cb(oa, ts: np.ndarray) -> np.ndarray:
    """[I,3] reconstructed CB per token: ALWAYS the ideal virtual CB from N/CA/C (RFdiffusion1 / RFD3 HARMONIA rule; red-team N001 #2: the real
    side-chain CB flipped contacts at the 6 A edge); NaN if the backbone is incomplete."""
    names = np.asarray(oa.atom_name).astype(str)
    xyz = np.asarray(oa.coord, dtype=np.float64)
    if "occupancy" in oa.get_annotation_categories():
        xyz = np.where((np.asarray(oa.occupancy) > 0)[:, None], xyz, np.nan)
    ends = np.append(ts[1:], len(names))
    out = np.full((len(ts), 3), np.nan)
    for k, (a, b) in enumerate(zip(ts, ends)):
        nm = names[a:b]
        get = {n: xyz[a + j] for j, n in enumerate(nm)}
        if all(x in get for x in ("N", "CA", "C")):
            out[k] = L.virtual_cb(get["N"], get["CA"], get["C"])
    return out


def token_backbone(oa, ts: np.ndarray) -> np.ndarray:
    """[I,3,3] N / CA / C per token (NaN if absent or zero occupancy)."""
    names = np.asarray(oa.atom_name).astype(str)
    occ = np.asarray(oa.occupancy) > 0 if "occupancy" in oa.get_annotation_categories() else np.ones(len(names), bool)
    ends = np.append(ts[1:], len(names))
    out = np.full((len(ts), 3, 3), np.nan)
    for k, (a, b) in enumerate(zip(ts, ends)):
        for q, atom in enumerate(("N", "CA", "C")):
            j = np.flatnonzero((names[a:b] == atom) & occ[a:b])
            if j.size:
                out[k, q] = oa.coord[a + j[0]]
    return out


def row_compact(oa, ts: np.ndarray, elig: np.ndarray) -> tuple[bool, list]:
    """Compactness gate: EVERY eligible protein chain must have >= GATE_MIN_CHAIN resolved tokens and pass the panel's universal split
    detector (scorer CB convention). A row with any shorter eligible chain is rejected (red-team N003 #1: an unchecked short / unresolved
    chain must not receive SS / ADJ next to a compact one); a row without eligible chains is rejected."""
    bb = token_backbone(oa, ts)
    ca = bb[:, 1]
    cb = L.scorer_cb(bb[:, 0], bb[:, 1], bb[:, 2])
    unit = np.asarray(oa.pn_unit_iid)[ts].astype(str)
    ok = elig & np.isfinite(bb).all((1, 2))
    ms = []
    for u in np.unique(unit[elig]):
        m = ok & (unit == u)
        if m.sum() < L.GATE_MIN_CHAIN:
            return False, ms + [{"short_chain": int(m.sum())}]
        ms.append(L.compactness(ca[m], cb[m]))
    return bool(ms) and all(L.is_compact(m) for m in ms), ms


def token_adj(oa, feats: dict, mode: str, rng: np.random.Generator, p_hot: float = 0.5, p_cold: float = 0.5, gate_compact: bool = False,
              p_dense_zero: float = 0.0, dense_modes: tuple = ("full",)):
    from atomworks.ml.utils.token import get_token_starts

    ts = get_token_starts(oa, add_exclusive_stop=False).astype(np.int64)
    I = int(feats["residue_type"].shape[0])
    assert len(ts) == I, (len(ts), I)
    ttype = feats["token_type"].numpy()
    atomized = np.asarray(oa.atomize)[ts].astype(bool) if "atomize" in oa.get_annotation_categories() else np.zeros(I, bool)
    copy = feats["insertion_source_idx"].numpy() >= 0 if "insertion_source_idx" in feats else np.zeros(I, bool)
    elig = (ttype == TOKEN_TYPE_PROTEIN) & ~atomized & ~copy
    sse = np.where(elig, np.asarray(oa.harm_sse)[ts], L.SS_NONE).astype(np.int64)
    blk = L.ss_blocks(sse, np.asarray(oa.pn_unit_iid)[ts].astype(str), np.asarray(oa.res_id)[ts])
    cb = token_cb(oa, ts)
    compact, cm = row_compact(oa, ts, elig) if gate_compact else (True, [])
    gated = gate_compact and not compact and mode != "uncond"
    ids, dmin, n6 = L.block_pair_stats(blk, cb)
    gt = L.block_states(dmin, n6)
    bst = L.reveal_block_pairs(gt, dmin, n6, "uncond" if gated else mode, rng, p_hot=p_hot, p_cold=p_cold)
    adj = L.tokens_from_blocks(blk, ids, bst)
    dense = (not gated) and mode in dense_modes and p_dense_zero > 0 and rng.random() < p_dense_zero
    if dense:
        adj = L.dense_zero(adj, elig)
    iu = np.triu_indices(len(ids), 1)
    stats = {"n_blocks": int(len(ids)), "n_contact_gt": int((gt[iu] == L.ADJ_CONTACT).sum()),
             "n_noncontact_gt": int((gt[iu] == L.ADJ_NON_CONTACT).sum()),
             "n_shown": int(np.isin(bst[iu], (0, 1, 3, 4)).sum()), "n_hot": int((bst[iu] == L.ADJ_HOT).sum()),
             "n_cold": int((bst[iu] == L.ADJ_COLD).sum()), "n_tokens": I, "n_elig": int(elig.sum()),
             "gated": int(gated), "compact": int(compact), "dense_zero": int(dense),
             "compact_checked": int(gate_compact), "mode_eff": "uncond" if gated else mode,
             "min_lr_cb": min((m["lr_cb"] for m in cm if "lr_cb" in m), default=float("nan")),
             "max_rg_n04": max((m["rg_n04"] for m in cm if "rg_n04" in m), default=float("nan"))}
    return adj, stats
