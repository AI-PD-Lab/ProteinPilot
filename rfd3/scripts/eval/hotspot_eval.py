#!/usr/bin/env python
"""Hotspot / coldspot specificity evaluation (WP-EVAL step 5, stage C of the retrain). CPU-testable on synthetic complexes.

Why a swap matrix: "binder atom within X A of the hotspot residue" is gamed by a binder that wraps the whole target and
by a model that always docks on its favourite patch (A5 4.3). So every target gets TWO disjoint epitopes A and B
(>= 12 A apart) and specs {hotspot A, hotspot B, no hotspot}; a model that ignores the spec or always docks on one patch
scores ~0 diagonal dominance:
    S_A = P(hit A | spec A) - P(hit A | spec B)      S_B = P(hit B | spec B) - P(hit B | spec A)
    spec_gain_E = P(hit E | spec E) - P(hit E | no spec)
    diag_dominance = (S_A + S_B) / 2   with bootstrap CI over designs
hit(E) = at least `hit_frac` (0.5) of the epitope residues have a heavy atom < 4.5 A from a binder heavy atom.
Matched-decoy AUC (per design, spec A or B): positives = the hotspot residues, negatives = non-hotspot target residues
within 15 A of the hotspot centroid whose relative SASA is within +-0.15 of a hotspot residue's (equally exposed);
score = -(min heavy-atom distance to the binder); invariant to the total interface size. Blanket flags: number of contacted
target residues and the fraction of the target surface covered, binder Rg, binder CA clashes, inter-chain heavy clashes.
Coldspot credit: contact rate of the cold epitope with the cold spec vs with no spec (an avoid instruction is only credited
where the baseline would have touched it).

Spec dialect: target FIRST in the contig (final chain A = target kept with residue numbers 1..n, B = designed binder):
  contig "A1-<n>,/0,<Lb>-<Lb>", select_fixed_atoms true, infer_ori_strategy com (all arms alike), is_non_loopy true
  1d arms   select_hotspots {"A<r>": "CB"}                (stock atom-level hotspot channel)
  adj arms  adj_pairs ["A<r-ranges> :: B1-<Lb>"]  (+ "A<r-ranges> !!: B1-<Lb>" for cold); needs the v2 inference operators
The refold-based ("independent judge") version of this score needs a multimer refold step (not part of designability.py,
which folds monomers); score_complex() accepts any complex structure, so refolded complexes can be scored by pointing
--complex-dir at them. TODO: wire an ESMFold2/AF3 multimer refold; the numbers from the generated coordinates are a first screen.

  python hotspot_eval.py build --targets 1ycr:A:B 4zqk:A:B --pack /hnvme/.../v2/eval/hotspot_specs_v2.tar.gz
  python hotspot_eval.py score --tar <tag>.tar.gz --specs-tar hotspot_specs_v2.tar.gz --out <dir>
(the generation step is the same eval_checkpoint.sbatch with SPECS=<hotspot pack>, CFGS="1.0 1.5").
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import build_specs as bs  # noqa: E402
import eval_common as ec  # noqa: E402

MAX_ASA = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225, "GLU": 223, "GLY": 104, "HIS": 224,
           "ILE": 197, "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159, "SER": 155, "THR": 172, "TRP": 285,
           "TYR": 263, "VAL": 174}          # Tien et al. 2013 (theoretical), A^2
CONTACT_CUT = 4.5       # heavy-atom contact between target residue and binder
INTER_CLASH_CUT = 2.2
HIT_FRAC = 0.5
MIN_SEP = 12.0          # A, minimum CB-CB distance between the two epitopes
PATCH_RADIUS = 10.0
RASA_TOL = 0.15
DECOY_RADIUS = 15.0


# --------------------------------------------------------------------------- geometry of the target
def residue_rasa(atoms) -> np.ndarray:
    """Relative SASA per residue (SASA / Tien max-ASA) of a protein AtomArray (target alone)."""
    import biotite.structure as struc

    atoms = atoms[atoms.element != "H"]
    try:
        sasa = struc.sasa(atoms, vdw_radii="ProtOr", point_number=200)
    except Exception:  # noqa: BLE001  (unknown atom names): element-based radii
        sasa = struc.sasa(atoms, vdw_radii="Single", point_number=200)
    res = struc.apply_residue_wise(atoms, np.nan_to_num(sasa), np.sum)
    names = atoms.res_name[struc.get_residue_starts(atoms)]
    return np.array([r / MAX_ASA.get(str(n), 200.0) for r, n in zip(res, names)])


def contacted_residues(target: "ec.Chain", partner_heavy: np.ndarray, cut: float = CONTACT_CUT):
    """(bool [n_res] contact, float [n_res] min heavy-atom distance to the partner) per target residue."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(partner_heavy, float))
    d, _ = tree.query(target.heavy)
    dmin = np.full(len(target), np.inf)
    np.minimum.at(dmin, target.heavy_res, d)
    return dmin < cut, dmin


def patch_from_seed(cb: np.ndarray, surface: np.ndarray, seed: int, n: int, radius: float = PATCH_RADIUS) -> list[int]:
    """The seed plus its n-1 nearest SURFACE residues (CB distance), all within `radius` of the seed."""
    d = np.linalg.norm(cb - cb[seed], axis=1)
    cand = [i for i in np.argsort(d) if surface[i] and d[i] <= radius]
    if seed not in cand:
        cand = [seed] + cand
    return sorted(int(i) for i in cand[:n])


def define_epitopes(target: "ec.Chain", rasa: np.ndarray, partner_heavy: np.ndarray | None = None, n_epi: int = 8,
                    min_sep: float = MIN_SEP, rasa_min: float = 0.25) -> dict:
    """Two disjoint epitopes of `target`: A = the native interface patch (if a partner is given) or the densest surface patch,
    B = a surface patch >= min_sep (CB-CB, every residue pair) away from A. Residue indices are 0-based."""
    cb = target.cb_leg
    surface = rasa >= rasa_min
    if surface.sum() < 2 * n_epi:
        raise ValueError(f"only {int(surface.sum())} surface residues for two epitopes of {n_epi}")
    # ---- epitope A
    if partner_heavy is not None:
        contact, dmin = contacted_residues(target, partner_heavy)
        cand = np.nonzero(contact & surface)[0]
        if len(cand) >= 3:
            cen = cb[cand].mean(0)
            seed = int(cand[np.argmin(np.linalg.norm(cb[cand] - cen, axis=1))])
            # restrict the patch to interface residues first, then fill with the nearest surface residues
            order = [int(i) for i in cand[np.argsort(np.linalg.norm(cb[cand] - cb[seed], axis=1))]]
            A = sorted(order[:n_epi])
            if len(A) < n_epi:
                A = patch_from_seed(cb, surface, seed, n_epi)
        else:
            partner_heavy = None
    if partner_heavy is None:
        dens = [(np.linalg.norm(cb[surface] - cb[i], axis=1) < PATCH_RADIUS).sum() if surface[i] else -1 for i in range(len(cb))]
        A = patch_from_seed(cb, surface, int(np.argmax(dens)), n_epi)
    # ---- epitope B: farthest-first seeds, accept the first patch that is >= min_sep from every A residue
    cenA = cb[A].mean(0)
    seeds = [int(i) for i in np.argsort(-np.linalg.norm(cb - cenA, axis=1)) if surface[i] and i not in A]
    B = None
    for s in seeds:
        patch = patch_from_seed(cb, surface, s, n_epi)
        if len(patch) < max(3, n_epi - 2) or set(patch) & set(A):
            continue
        sep = np.linalg.norm(cb[A][:, None] - cb[patch][None], axis=-1).min()
        if sep >= min_sep:
            B = patch
            break
    if B is None:
        raise ValueError(f"no epitope B >= {min_sep} A from A found")
    sep = float(np.linalg.norm(cb[A][:, None] - cb[B][None], axis=-1).min())
    return {"A": A, "B": B, "min_sep": sep}


# --------------------------------------------------------------------------- spec building
def _ranges(res_1based: list[int]) -> str:
    """[3,4,5,9] -> 'A3-5,A9' (target chain A, residue numbers 1..n)."""
    res = sorted(res_1based)
    out, i = [], 0
    while i < len(res):
        j = i
        while j + 1 < len(res) and res[j + 1] == res[j] + 1:
            j += 1
        out.append(f"A{res[i]}" if i == j else f"A{res[i]}-{res[j]}")
        i = j + 1
    return ",".join(out)


def make_arms(tname: str, n_target: int, binder_len: int, epi: dict, gly: set[int], ori_1d: str = "com") -> tuple[dict, dict]:
    """(inputs {key: spec}, arm records {key: rec}) for one target; epitope residues are 0-based in `epi`, 1-based in specs.

    Origin placement: ALL arms use infer_ori_strategy `com` by default so the swap matrix compares like with like (the stock
    recipe `hotspots` places the ORI token at the hotspots, which itself encodes the epitope and exists only for the 1d arms;
    the rfd3 parser asserts that `hotspots` needs select_hotspots, so it cannot be used for the none/adj arms anyway).
    ori_1d='hotspots' switches the 1d arms to the stock recipe (a separate, confounded comparison)."""
    E = {k: [i + 1 for i in v] for k, v in epi.items() if k in ("A", "B")}
    base = {"dialect": 2, "input": f"{ec.SPEC_ROOT_TOKEN}/targets/{tname}.pdb", "contig": f"A1-{n_target},/0,{binder_len}-{binder_len}",
            "select_fixed_atoms": True, "infer_ori_strategy": "com", "is_non_loopy": True}
    Lb = binder_len

    def hs(which):
        return {f"A{r}": ("CA" if (r - 1) in gly else "CB") for r in E[which]}
    arms = {
        "none": (None, None, None),
        "hotA_1d": ("A", None, "1d"), "hotB_1d": ("B", None, "1d"),
        "hotA_adj": ("A", None, "adj"), "hotB_adj": ("B", None, "adj"),
        "hotA_adj_coldB": ("A", "B", "adj"), "hotB_adj_coldA": ("B", "A", "adj"),
    }
    inputs, recs = {}, {}
    for arm, (hot, cold, mode) in arms.items():
        key = f"{tname}__{arm}"
        spec = dict(base, extra={"harmonia_eval": {"target": tname, "arm": arm, "hot": hot, "cold": cold}})
        if mode == "1d":
            spec["select_hotspots"] = hs(hot)
            spec["infer_ori_strategy"] = ori_1d
        elif mode == "adj":
            pairs = [f"{_ranges(E[hot])} :: B1-{Lb}"]
            if cold:
                pairs.append(f"{_ranges(E[cold])} !!: B1-{Lb}")
            spec["adj_pairs"] = pairs
        inputs[key] = spec
        recs[key] = {"target": tname, "arm": arm, "key": key, "hot": hot, "cold": cold, "mode": mode, "length": n_target + binder_len,
                     "binder_len": binder_len}
    return inputs, recs


def build(out_dir: str, targets: list[str], binder_len: int = 70, n_epi: int = 8, mirror_tar=ec.PDB_MIRROR_TAR,
          mirror_index=ec.PDB_MIRROR_INDEX) -> dict:
    """targets: 'pdb:targetchain[:partnerchain]'. Writes targets/<name>.pdb, inputs_template.json, manifest.json."""
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb

    os.makedirs(os.path.join(out_dir, "targets"), exist_ok=True)
    manifest = {"spec_version": 2, "kind": "hotspot", "targets": {}, "arms": {}, "unconditional": {}, "binder_len": binder_len}
    inputs: dict = {}
    for spec in targets:
        parts = spec.split(":")
        pid, tchain = parts[0], parts[1]
        pchain = parts[2] if len(parts) > 2 else None
        atoms = ec.read_structure(bs.mirror_bytes(pid, mirror_tar, mirror_index), pid + ".cif.gz")
        atoms = atoms[struc.filter_amino_acids(atoms) & (atoms.element != "H")]
        tgt_atoms = atoms[atoms.chain_id == tchain]
        chains = ec.protein_chains(tgt_atoms)
        lo, hi = bs.gap_free_segment(chains[0])
        rid_lo, rid_hi = int(chains[0].res_ids[lo]), int(chains[0].res_ids[hi - 1])
        tgt_atoms = bs.select_residue_range(tgt_atoms, lo, hi)
        tchain_obj = ec.protein_chains(tgt_atoms)[0]
        partner = atoms[atoms.chain_id == pchain].coord if pchain else None
        rasa = residue_rasa(tgt_atoms)
        epi = define_epitopes(tchain_obj, rasa, partner, n_epi)
        name = f"hot_{pid}"
        # renumber 1..n and write the fixed target
        t2 = tgt_atoms.copy()
        starts = struc.get_residue_starts(t2)
        t2.res_id = np.repeat(np.arange(1, len(starts) + 1), np.diff(list(starts) + [t2.array_length()]))
        t2.chain_id[:] = "A"
        f = pdb.PDBFile()
        f.set_structure(t2)
        f.write(os.path.join(out_dir, "targets", f"{name}.pdb"))
        gly = {i for i, r in enumerate(tchain_obj.res_names) if r == "GLY"}
        ins, recs = make_arms(name, len(tchain_obj), binder_len, epi, gly)
        inputs.update(ins)
        manifest["arms"].update(recs)
        manifest["targets"][name] = {
            "name": name, "pdb_id": pid, "chain": tchain, "partner_chain": pchain, "n_res": len(tchain_obj),
            "res_range": [rid_lo, rid_hi], "epitope_A": epi["A"], "epitope_B": epi["B"], "min_sep": epi["min_sep"],
            "rasa": [round(float(x), 3) for x in rasa], "surface": [int(i) for i in np.nonzero(rasa >= 0.25)[0]],
            "cb": np.round(tchain_obj.cb_leg, 2).tolist()}
    ec.write_json(os.path.join(out_dir, "inputs_template.json"), inputs, indent=1)
    ec.write_json(os.path.join(out_dir, "manifest.json"), manifest, indent=None)
    return manifest


# --------------------------------------------------------------------------- scoring
def score_complex(chains: list["ec.Chain"], tinfo: dict, rec: dict) -> dict:
    """Metrics of ONE complex (target = chains[0], binder = chains[1]) for the spec `rec` of target `tinfo` (manifest dicts)."""
    row: dict = {"ok": False}
    if len(chains) < 2:
        row["error"] = "no binder chain"
        return row
    tgt, bnd = chains[0], chains[1]
    if len(tgt) != tinfo["n_res"]:
        row["error"] = f"target length {len(tgt)} != {tinfo['n_res']}"
        return row
    contact, dmin = contacted_residues(tgt, bnd.heavy)
    A, B = tinfo["epitope_A"], tinfo["epitope_B"]
    surface = np.zeros(len(tgt), bool)
    surface[tinfo["surface"]] = True
    for e, idx in (("A", A), ("B", B)):
        frac = float(contact[idx].mean())
        row[f"frac_{e}"] = frac
        row[f"hit_{e}"] = float(frac >= HIT_FRAC)
        row[f"dmin_{e}"] = float(np.median(dmin[idx]))
    row["n_contact_res"] = int(contact.sum())
    row["frac_surface_contacted"] = float((contact & surface).sum() / max(surface.sum(), 1))
    epi_union = set(A) | set(B)
    row["contacts_outside_epitopes"] = int(sum(1 for i in np.nonzero(contact)[0] if i not in epi_union))
    row["binder_rg"] = float(np.sqrt(np.nanmean(((bnd.ca - np.nanmean(bnd.ca, 0)) ** 2).sum(1))))
    D = ec.pairwise(bnd.ca)
    iu = np.triu_indices(len(bnd), 3)
    row["binder_ca_clash"] = int((D[iu] < ec.CA_CLASH_CUT).sum())
    from scipy.spatial import cKDTree

    row["interface_heavy_clash"] = int(sum(len(x) for x in cKDTree(tgt.heavy).query_ball_tree(cKDTree(bnd.heavy), INTER_CLASH_CUT)))
    # ---- matched-decoy AUC for the specified hotspot
    hot = rec.get("hot")
    if hot:
        H = tinfo["epitope_A" if hot == "A" else "epitope_B"]
        row["auc_matched"], row["auc_matched_binary"] = matched_decoy_auc(
            np.asarray(tinfo["cb"]), np.asarray(tinfo["rasa"]), H, dmin)
    row["ok"] = True
    return row


def matched_decoy_auc(cb: np.ndarray, rasa: np.ndarray, hot: list[int], dmin: np.ndarray,
                      radius: float = DECOY_RADIUS, tol: float = RASA_TOL, contact_cut: float = CONTACT_CUT):
    """AUC (continuous -dmin, and binary contact) of hotspot residues vs equally exposed non-hotspot residues near them.
    Per hotspot residue h the decoys are non-hotspot residues within `radius` of the hotspot centroid with |RASA - RASA_h| <= tol."""
    hot = list(hot)
    cen = cb[hot].mean(0)
    pool = [i for i in range(len(cb)) if i not in set(hot) and np.linalg.norm(cb[i] - cen) <= radius and rasa[i] >= 0.05]
    wins = ties = n = 0.0
    wins_b = ties_b = 0.0
    for h in hot:
        dec = [d for d in pool if abs(rasa[d] - rasa[h]) <= tol]
        for d in dec:
            n += 1
            s_h, s_d = -dmin[h], -dmin[d]
            wins += s_h > s_d
            ties += s_h == s_d
            b_h, b_d = float(dmin[h] < contact_cut), float(dmin[d] < contact_cut)
            wins_b += b_h > b_d
            ties_b += b_h == b_d
    if n == 0:
        return float("nan"), float("nan")
    return float((wins + 0.5 * ties) / n), float((wins_b + 0.5 * ties_b) / n)


def swap_matrix(rows: list[dict], target: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """rows = score_complex rows with keys target, arm, hot, cold, mode (one cfg). Returns the matrix + dominance with CIs.
    Uses the `adj` arms (and 1d separately): mode 'adj'/'1d' are separate matrices; 'none' is shared."""
    out = {}
    for mode in ("1d", "adj"):
        sel = lambda hot, mode=mode: [r for r in rows if r["target"] == target and r.get("ok") and r.get("hot") == hot and r.get("cold") is None and r.get("mode") == mode]
        none = [r for r in rows if r["target"] == target and r.get("ok") and r.get("mode") is None]
        A, B = sel("A"), sel("B")
        if not (A and B):
            continue
        rate = lambda rs, e: np.array([r[f"hit_{e}"] for r in rs], float)
        mat = {"specA": {e: float(rate(A, e).mean()) for e in "AB"}, "specB": {e: float(rate(B, e).mean()) for e in "AB"},
               "none": {e: float(rate(none, e).mean()) for e in "AB"} if none else {}, "n": [len(A), len(B), len(none)]}
        rng = np.random.RandomState(seed)

        def boot(f):
            vals = []
            for _ in range(n_boot):
                ra = rng.randint(0, len(A), len(A))
                rb = rng.randint(0, len(B), len(B))
                rn = rng.randint(0, len(none), len(none)) if none else None
                vals.append(f(rate(A, "A")[ra], rate(A, "B")[ra], rate(B, "A")[rb], rate(B, "B")[rb],
                              rate(none, "A")[rn] if none else None, rate(none, "B")[rn] if none else None))
            vals = np.array(vals, float)
            return float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))
        sA = rate(A, "A").mean() - rate(B, "A").mean()
        sB = rate(B, "B").mean() - rate(A, "B").mean()
        mat["S_A"], mat["S_B"] = float(sA), float(sB)
        mat["diag_dominance"] = float((sA + sB) / 2)
        mat["diag_dominance_ci"] = boot(lambda aa, ab, ba, bb, na, nb: ((aa.mean() - ba.mean()) + (bb.mean() - ab.mean())) / 2)
        if none:
            mat["gain_A"] = float(rate(A, "A").mean() - rate(none, "A").mean())
            mat["gain_B"] = float(rate(B, "B").mean() - rate(none, "B").mean())
        aucs = [r["auc_matched"] for r in A + B if math.isfinite(r.get("auc_matched", float("nan")))]
        mat["auc_matched_mean"] = float(np.mean(aucs)) if aucs else float("nan")
        out[mode] = mat
    # coldspot credit (adj mode): cold-epitope contact rate with the cold spec vs the no-spec baseline
    cred = {}
    for hot, cold in (("A", "B"), ("B", "A")):
        cs = [r for r in rows if r["target"] == target and r.get("ok") and r.get("hot") == hot and r.get("cold") == cold]
        nn = [r for r in rows if r["target"] == target and r.get("ok") and r.get("mode") is None]
        if cs and nn:
            with_cold = float(np.mean([r[f"hit_{cold}"] for r in cs]))
            base = float(np.mean([r[f"hit_{cold}"] for r in nn]))
            nohot = [r for r in rows if r["target"] == target and r.get("ok") and r.get("hot") == hot and r.get("cold") is None and r.get("mode") == "adj"]
            ref = float(np.mean([r[f"hit_{cold}"] for r in nohot])) if nohot else float("nan")
            cred[f"hot{hot}_cold{cold}"] = {"cold_hit_with_cold_spec": with_cold, "cold_hit_hot_only": ref, "cold_hit_no_spec": base}
    out["coldspot"] = cred
    return out


# --------------------------------------------------------------------------- CLI
def cmd_build(a):
    tmp = None if a.work_dir else tempfile.TemporaryDirectory(prefix="harmonia_hot_")
    work = a.work_dir or tmp.name
    try:
        man = build(work, a.targets, a.binder_len, a.n_epi)
        bs.pack_dir(work, a.pack)
    finally:
        if tmp:
            tmp.cleanup()
    ec.write_json(a.pack.replace(".tar.gz", "") + "_manifest.json", man, indent=None)
    for t in man["targets"].values():
        print(f"{t['name']}: {t['pdb_id'].upper()} chain {t['chain']} (partner {t['partner_chain']}), {t['n_res']} res, epitope A {len(t['epitope_A'])} res, "
              f"B {len(t['epitope_B'])} res, min A-B CB distance {t['min_sep']:.1f} A, surface residues {len(t['surface'])}")
    print(f"{len(man['arms'])} specs, packed {a.pack}")


def cmd_score(a):
    import score_designs as sd

    manifest = bs.read_manifest_from_tar(a.specs_tar)
    rows = []
    for name, data in sd.iter_tar_designs(a.tar):
        mt = ec.DESIGN_RE.match(os.path.basename(name))
        key = mt.group("key")
        rec = manifest["arms"].get(key)
        if rec is None:
            continue
        cm = ec.CFG_DIR_RE.search(name.split("/")[-2])
        try:
            r = score_complex(ec.protein_chains(ec.read_structure(data, name)), manifest["targets"][rec["target"]], rec)
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        r.update(target=rec["target"], arm=rec["arm"], hot=rec["hot"], cold=rec["cold"], mode=rec["mode"],
                 cfg=float(cm.group("cfg")) if cm else float("nan"), key=key)
        rows.append(r)
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for cfg in sorted({r["cfg"] for r in rows}):
        for t in manifest["targets"]:
            res[f"{t}|cfg{cfg:.1f}"] = swap_matrix([r for r in rows if r["cfg"] == cfg], t)
    ec.write_json(os.path.join(a.out, "hotspot_scores.json"), {"matrices": res, "n_designs": len(rows)}, indent=1)
    for k, v in res.items():
        for mode in ("1d", "adj"):
            m = v.get(mode)
            if m:
                print(f"{k} [{mode}] n={m['n']} spec A -> hit A/B {m['specA']['A']:.2f}/{m['specA']['B']:.2f}; spec B -> {m['specB']['A']:.2f}/{m['specB']['B']:.2f}; "
                      f"none {m['none'].get('A', float('nan')):.2f}/{m['none'].get('B', float('nan')):.2f}; diag dominance {m['diag_dominance']:.2f} "
                      f"[{m['diag_dominance_ci'][0]:.2f},{m['diag_dominance_ci'][1]:.2f}]; matched-decoy AUC {m['auc_matched_mean']:.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--targets", nargs="+", required=True, metavar="PDB:TARGETCHAIN[:PARTNERCHAIN]")
    b.add_argument("--pack", default=ec.EVAL_ROOT + "/hotspot_specs_v2.tar.gz")
    b.add_argument("--work-dir", default=None)
    b.add_argument("--binder-len", type=int, default=70)
    b.add_argument("--n-epi", type=int, default=8)
    s = sub.add_parser("score")
    s.add_argument("--tar", required=True)
    s.add_argument("--specs-tar", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    (cmd_build if a.cmd == "build" else cmd_score)(a)


if __name__ == "__main__":
    main()
