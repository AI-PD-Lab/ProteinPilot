#!/usr/bin/env python
"""Per-key report of a built sketch pack against the native references it ships.  SCORING ONLY.

Nothing here is on the build path: the planner, the sketch builder and the acceptance gate never read
``refs.npz`` (red-team b0r5, section 1); this script is where the natives are opened, after the pack exists.
Per conditioned parent key ``<target>__<arm>`` it reports

* ``n_layouts`` / ``n_requested``: accepted layouts (with a shipped sketch) over attempted layout slots
  (``sketch_checks.json``), ``map_mean`` / ``map_min``: map agreement of the accepted layouts;
* ``tm_mean/min/max``, ``mirror_tm_mean``, ``rmsd_mean``: TM-score of every sketch CA trace against the native
  CA with the fixed residue correspondence (``tm_score.tm_score``; sketch and native have the same length and
  block layout), the TM-score of the x-mirrored sketch (``ca * [-1, 1, 1]``, as in N031
  ``probe/sketch_tm_vs_native.py``: TM == mirror-TM means no chirality signal) and the superposed CA RMSD;
* strand ORDER vs native for every target with >= 2 strands.  Native sheets are the connected components of
  the native geometric strand pairing (``extract_native_layout``: >= 2 CA pairs < 5.5 A, no map).  Inside each
  native sheet the strands are ordered by the projection of their centroids on the principal direction across
  the sheet (perpendicular to the mean, sign-aligned strand axis -- the N031 notebook / R3 test method), and the
  sketch order of the same strands must equal the native order or its reverse; a native ring (every strand has
  two partners inside the component) is compared as a cyclic sequence.  The sketch's own sheet partition
  (components of ITS geometric pairing, measured on the shipped CIF, never the planner's labels) is compared
  with the native partition (``partition_native``; a native strand without a CA-level partner, e.g. strand 2 of
  sandwich_1tit, is its own component).  ``order_native`` counts layouts where EVERY native sheet with >= 2
  strands is in native order; ``order_and_partition_native`` additionally requires the partition.  The manifest
  ``half1_blocks`` / ``half2_blocks`` are sequence halves (len // 2), not sheets; they are only echoed in the JSON;
* ``geometric_pair_polarity_*``: polarity of ALL native geometric E-E pairs, irrespective of map masking;
  ``packing_pair_polarity_*``: polarity of required E-E map contacts without a native geometric ladder;
  ``geometric_pairs_retained``: how many native ladders are also geometrically paired in the sketch;
* polarity: for the arm's required pairs (``manifest['arms'][key]['pos_pairs']``, same-type pairs only) and
  for all same-type axis pairs, whether sign(axis_i . axis_j) in the sketch equals the native sign.

Usage::

    python sketch_report.py --pack specs.tar.gz [--refs refs.npz] --out per_key.csv [--json per_layout.json]
                            [--keys ferredoxin_1urn__full ...] [--label specsv2]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import eval_common as ec  # noqa: E402
from layout_common import extract_native_layout  # noqa: E402
from tm_score import rmsd_superposed, tm_score  # noqa: E402

MIRROR = np.array([-1., 1., 1.])
CSV_FIELDS = ('pack', 'key', 'target', 'arm', 'n_layouts', 'n_requested', 'map_mean', 'map_min',
              'tm_mean', 'tm_min', 'tm_max', 'mirror_tm_mean', 'rmsd_mean', 'n_strands', 'native_sheets',
              'partition_native', 'order_native', 'order_and_partition_native', 'sketch_orders',
              'pos_pairs', 'pos_pair_polarity_matches', 'pos_pair_polarity_comparisons', 'pos_pair_polarity_detail',
              'all_pair_polarity_matches', 'all_pair_polarity_comparisons',
              'geometric_pair_polarity_matches', 'geometric_pair_polarity_comparisons', 'geometric_pair_polarity_detail',
              'packing_pair_polarity_matches', 'packing_pair_polarity_comparisons', 'packing_pair_polarity_detail',
              'geometric_pairs_retained', 'native_geometric_pairs', 'note')


# --------------------------------------------------------------------------- geometry helpers
def ca_from_cif(path):
    chains = ec.protein_chains(ec.read_structure(str(path), str(path)))
    if len(chains) != 1:
        raise ValueError(f'{path}: expected one chain, found {len(chains)}')
    ca = np.asarray(chains[0].ca, float)
    if not np.isfinite(ca).all():
        raise ValueError(f'{path}: missing CA atoms')
    return ca


def components(layout, strands):
    """Connected components of the geometric pairing graph restricted to `strands`, each sorted, by first id."""
    strands = [int(s) for s in strands]
    remaining, out = set(strands), []
    while remaining:
        seed = min(remaining)
        comp, stack = {seed}, [seed]
        while stack:
            i = stack.pop()
            for j in strands:
                if j in remaining and j not in comp and layout.pairing[i, j]:
                    comp.add(j)
                    stack.append(j)
        remaining -= comp
        out.append(sorted(comp))
    return sorted(out, key=lambda c: c[0])


def is_ring(layout, comp):
    return len(comp) >= 3 and all(sum(bool(layout.pairing[i, j]) for j in comp if j != i) == 2 for i in comp)


def across_sheet_positions(layout, comp):
    """Centroids of `comp` strands projected on the plane perpendicular to their mean (sign-aligned) axis."""
    idx = np.asarray(comp)
    axes = layout.axes[idx]
    axis = np.mean(axes * np.where(axes @ axes[0] >= 0, 1., -1.)[:, None], axis=0)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    pts = layout.centroids[idx] - layout.centroids[idx].mean(0)
    return pts - np.outer(pts @ axis, axis)


def sheet_order(layout, comp, cyclic=False):
    """Strand ids of `comp` along the principal across-sheet direction (or around the ring), canonical orientation."""
    comp = [int(c) for c in comp]
    if len(comp) < 2:
        return comp
    pts = across_sheet_positions(layout, comp)
    _, _, basis = np.linalg.svd(pts)
    if cyclic:
        angle = np.arctan2(pts @ basis[1], pts @ basis[0])
        order = [comp[k] for k in np.argsort(angle)]
    else:
        order = [comp[k] for k in np.argsort(pts @ basis[0])]
    return canonical(order, cyclic)


def canonical(order, cyclic=False):
    """Orientation-free representation: linear -> first id < last id; ring -> starts at min id, smaller neighbour next."""
    order = [int(o) for o in order]
    if len(order) < 2:
        return order
    if not cyclic:
        return order if order[0] <= order[-1] else order[::-1]
    k = order.index(min(order))
    order = order[k:] + order[:k]
    return order if len(order) < 3 or order[1] <= order[-1] else [order[0]] + order[1:][::-1]


def same_order(a, b, cyclic=False):
    return canonical(a, cyclic) == canonical(b, cyclic)


def order_str(order, cyclic=False):
    s = '-'.join(map(str, order))
    return f'({s})' if cyclic else f'[{s}]'


# --------------------------------------------------------------------------- scoring
def native_info(target_rec, nat_ca):
    ss = np.asarray(target_rec['ss'], int)
    nat_ca = np.asarray(nat_ca, float)
    if nat_ca.shape != (len(ss), 3):
        raise ValueError(f'native CA {nat_ca.shape} does not match ss length {len(ss)}')
    nat = extract_native_layout(nat_ca, ss, None)
    strands = [int(i) for i in np.flatnonzero(nat.types == 'E')]
    sheets = components(nat, strands) if strands else []
    rings = [is_ring(nat, c) for c in sheets]
    orders = [sheet_order(nat, c, r) for c, r in zip(sheets, rings)]
    return dict(layout=nat, ca=nat_ca, ss=ss, types=nat.types, strands=strands, sheets=sheets, rings=rings,
                orders=orders, halves=[target_rec.get('half1_blocks'), target_rec.get('half2_blocks')])


def score_layout(info, sk_ca, pos_pairs):
    nat, nat_ca, types = info['layout'], info['ca'], info['types']
    sk_ca = np.asarray(sk_ca, float)
    if sk_ca.shape != nat_ca.shape:
        raise ValueError(f'sketch CA {sk_ca.shape} does not match native {nat_ca.shape}')
    sk = extract_native_layout(sk_ca, info['ss'], None)   # same geometric extraction as for the native
    out = dict(tm=float(tm_score(sk_ca, nat_ca)), mirror_tm=float(tm_score(sk_ca * MIRROR, nat_ca)),
               rmsd=float(rmsd_superposed(sk_ca, nat_ca)))
    n = len(types)

    def match(i, j):
        return bool((sk.axes[i] @ sk.axes[j] > 0) == (nat.axes[i] @ nat.axes[j] > 0))

    all_pairs = {f'{i}-{j}': match(i, j) for i in range(n) for j in range(i + 1, n) if types[i] == types[j]}
    required = sorted({(min(int(i), int(j)), max(int(i), int(j))) for i, j in (pos_pairs or [])
                       if 0 <= int(i) < n and 0 <= int(j) < n and int(i) != int(j) and types[int(i)] == types[int(j)]})
    out['all_pairs'] = all_pairs
    out['required_pairs'] = {f'{i}-{j}': all_pairs[f'{i}-{j}'] for i, j in required}
    # Native geometric ladders and unsigned-map packing contacts are different
    # populations. Membership is scored from refs, never the proposed sheets.
    geometric = [(i,j) for i in info['strands'] for j in info['strands'] if i<j and nat.pairing[i,j]]
    packing = [(i,j) for i,j in required if types[i] == 'E' and not nat.pairing[i,j]]
    out['geometric_pairs'] = {f'{i}-{j}': match(i,j) for i,j in geometric}
    out['packing_pairs'] = {f'{i}-{j}': match(i,j) for i,j in packing}
    out['geometric_pairs_retained'] = sum(bool(sk.pairing[i,j]) for i,j in geometric)
    out['native_geometric_pairs'] = len(geometric)
    strands = info['strands']
    if strands:
        sk_sheets = components(sk, strands)
        out['sketch_sheets'] = sk_sheets
        out['partition_match'] = {frozenset(c) for c in sk_sheets} == {frozenset(c) for c in info['sheets']}
        orders = []
        for comp, ring, nat_order in zip(info['sheets'], info['rings'], info['orders']):
            if len(comp) < 2:
                continue
            o = sheet_order(sk, comp, ring)
            orders.append(dict(strands=comp, cyclic=ring, native=nat_order, sketch=o,
                               match=same_order(o, nat_order, ring)))
        out['orders'] = orders
        out['order_match'] = all(o['match'] for o in orders) if orders else True
        out['order_and_partition_match'] = bool(out['partition_match'] and out['order_match'])
    else:
        out.update(sketch_sheets=[], partition_match=None, orders=[], order_match=None, order_and_partition_match=None)
    return out


def _mean(x):
    return float(np.mean(x)) if len(x) else float('nan')


def _fmt(x, nd=3):
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))


def report_pack(pack, refs=None, keys=None, label=None, log=print):
    """Return (rows, detail): per-parent-key CSV rows and per-layout detail of the pack (tar.gz or directory)."""
    with tempfile.TemporaryDirectory(prefix='sketch_report_') as work:
        root = Path(pack)
        if not root.is_dir():
            ec.safe_extract(str(pack), work)
            root = Path(work)
        manifest = ec.read_json(str(root / 'manifest.json'))
        checks = ec.read_json(str(root / 'sketch_checks.json'))
        refs_path = Path(refs) if refs else root / 'refs.npz'
        natives = {}
        if refs_path.exists():
            with np.load(refs_path, allow_pickle=False) as z:
                natives = {k[:-3]: z[k].copy() for k in z.files if k.endswith('_ca')}
        label = label or Path(pack).name.replace('.tar.gz', '').replace('.tgz', '')
        arms = manifest.get('arms', {})
        parents = {}
        for subkey in sorted(checks):
            parent = subkey.rsplit('__', 1)[0]
            if keys and parent not in keys:
                continue
            parents.setdefault(parent, []).append(subkey)
        rows, detail, infos = [], {}, {}
        for parent, subkeys in parents.items():
            rec0 = next((arms[k] for k in subkeys if k in arms), {})
            target = rec0.get('target') or parent.split('__')[0]
            arm = rec0.get('arm') or parent.split('__', 1)[-1]
            accepted = [k for k in subkeys if checks[k].get('accepted') and (root / f'sketch/{k}.cif').exists()]
            maps = [float(checks[k]['map_agreement']) for k in accepted if checks[k].get('map_agreement') is not None]
            row = dict.fromkeys(CSV_FIELDS, '')
            row.update(pack=label, key=parent, target=target, arm=arm, n_layouts=len(accepted),
                       n_requested=len(subkeys), map_mean=_fmt(_mean(maps)), map_min=_fmt(min(maps) if maps else float('nan')))
            note = []
            info = None
            if target in natives and target in manifest.get('targets', {}):
                if target not in infos:
                    infos[target] = native_info(manifest['targets'][target], natives[target])
                info = infos[target]
            else:
                note.append('no native reference')
            scored = []
            for k in accepted:
                if info is None:
                    continue
                try:
                    s = score_layout(info, ca_from_cif(root / f'sketch/{k}.cif'), arms.get(k, {}).get('pos_pairs'))
                except ValueError as exc:
                    note.append(f'{k}: {exc}')
                    continue
                s['map_agreement'] = checks[k].get('map_agreement')
                detail[k] = s
                scored.append(s)
            if info is not None:
                row.update(n_strands=len(info['strands']),
                           native_sheets=''.join(order_str(o, r) for o, r in zip(info['orders'], info['rings'])))
            if scored:
                row.update(tm_mean=_fmt(_mean([s['tm'] for s in scored])), tm_min=_fmt(min(s['tm'] for s in scored)),
                           tm_max=_fmt(max(s['tm'] for s in scored)),
                           mirror_tm_mean=_fmt(_mean([s['mirror_tm'] for s in scored])),
                           rmsd_mean=_fmt(_mean([s['rmsd'] for s in scored]), 2))
                req_m = sum(sum(s['required_pairs'].values()) for s in scored)
                req_n = sum(len(s['required_pairs']) for s in scored)
                all_m = sum(sum(s['all_pairs'].values()) for s in scored)
                all_n = sum(len(s['all_pairs']) for s in scored)
                per_pair = Counter()
                for s in scored:
                    for pair, ok in s['required_pairs'].items():
                        per_pair[pair] += int(ok)
                pairs = sorted(per_pair, key=lambda p: tuple(map(int, p.split('-'))))
                row.update(pos_pairs=';'.join(pairs), pos_pair_polarity_matches=req_m, pos_pair_polarity_comparisons=req_n,
                           pos_pair_polarity_detail=';'.join(f'{p}:{per_pair[p]}/{len(scored)}' for p in pairs),
                           all_pair_polarity_matches=all_m, all_pair_polarity_comparisons=all_n)
                for category in ('geometric', 'packing'):
                    pair_counts = Counter()
                    for item in scored:
                        pair_counts.update({pair:int(ok) for pair,ok in item[category+'_pairs'].items()})
                    pair_ids = sorted(pair_counts, key=lambda p:tuple(map(int,p.split('-'))))
                    row[category+'_pair_polarity_matches'] = sum(pair_counts.values())
                    row[category+'_pair_polarity_comparisons'] = sum(len(item[category+'_pairs']) for item in scored)
                    row[category+'_pair_polarity_detail'] = ';'.join(f'{p}:{pair_counts[p]}/{len(scored)}' for p in pair_ids)
                row['geometric_pairs_retained'] = sum(item['geometric_pairs_retained'] for item in scored)
                row['native_geometric_pairs'] = sum(item['native_geometric_pairs'] for item in scored)
                if info['strands']:
                    orders = Counter(''.join(order_str(o['sketch'], o['cyclic']) for o in s['orders']) for s in scored)
                    row.update(partition_native=sum(bool(s['partition_match']) for s in scored),
                               order_native=sum(bool(s['order_match']) for s in scored),
                               order_and_partition_native=sum(bool(s['order_and_partition_match']) for s in scored),
                               sketch_orders='|'.join(f'{o}:{c}' for o, c in orders.most_common()))
            row['note'] = '; '.join(note)
            rows.append(row)
            log(f"{label:10s} {parent:34s} n {row['n_layouts']}/{row['n_requested']} map {row['map_mean']}({row['map_min']}) "
                f"TM {row['tm_mean']} [{row['tm_min']}-{row['tm_max']}] mirror {row['mirror_tm_mean']} "
                f"order {row['order_native']}/{row['n_layouts']} (partition {row['partition_native']}) native {row['native_sheets']} sketch {row['sketch_orders']} "
                f"pos-pol {row['pos_pair_polarity_matches']}/{row['pos_pair_polarity_comparisons']} "
                f"all-pol {row['all_pair_polarity_matches']}/{row['all_pair_polarity_comparisons']} "
                f"geometric-pol {row['geometric_pair_polarity_matches']}/{row['geometric_pair_polarity_comparisons']} "
                f"packing-pol {row['packing_pair_polarity_matches']}/{row['packing_pair_polarity_comparisons']} {row['note']}")
        for target, info in infos.items():
            detail[f'__native__{target}'] = dict(strands=info['strands'], sheets=info['sheets'], rings=info['rings'],
                                                 orders=info['orders'], manifest_halves=info['halves'])
    return rows, detail


def write_csv(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--pack', required=True, help='built sketch pack (tar.gz) or an extracted pack directory')
    p.add_argument('--refs', default=None, help='native refs.npz (default: the refs.npz inside the pack)')
    p.add_argument('--out', required=True, help='per-key CSV')
    p.add_argument('--json', default=None, help='optional per-layout detail JSON')
    p.add_argument('--keys', nargs='*', default=None, help='restrict to these parent keys (<target>__<arm>)')
    p.add_argument('--label', default=None, help='pack label for the CSV (default: pack file name)')
    args = p.parse_args(argv)
    rows, detail = report_pack(args.pack, refs=args.refs, keys=args.keys, label=args.label)
    write_csv(rows, args.out)
    if args.json:
        ec.write_json(args.json, detail)
    print(f'{len(rows)} keys -> {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
