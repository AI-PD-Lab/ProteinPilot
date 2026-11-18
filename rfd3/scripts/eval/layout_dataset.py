#!/usr/bin/env python
"""Native block-layout dataset. No torch, pickle, per-chain output files or network.

extract: disjoint weighted candidate shards; merge: source/beta quotas and split.
Breaks produce separate examples (metadata seg_key), never fictitious loop links.
All sequence leakage checks happen before either train/validation assignment.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
import io
import json
import multiprocessing as mp
from pathlib import Path
import time
import zlib
import zipfile

import numpy as np
import eval_common as ec
from layout_common import Layout, extract_native_layout

ROOT = Path(ec.WORKSPACE)
DEFAULT_OUT = ROOT / 'v2/eval/layout_planner'
AFDB_MANIFEST = ROOT / 'v2/data/afdb_replay_manifest.parquet'
AFDB_TAR = ROOT / 'v2/data/swissprot_cif_v6.tar'
PDB_MANIFEST = Path('/path/to/work/data/metadata/splits/pn_units_df_train.parquet')
TARGETS = ('1tit', '1urn', '3chy', '2hmq', '2ccy', '1fna', '1ulr', '1f41', '1stp')
MAX_BLOCKS, MAX_LENGTH = 24, 300


def kmer_identity(a, b, k=3):
    """Multiset k-mer containment / smaller chain's k-mer count (conservative).

    This is an operational leakage screen, not aligned sequence identity. Using
    the shorter denominator also excludes close domain fragments of a target.
    """
    def counts(s):
        return Counter(s[i:i+k] for i in range(max(0, len(s)-k+1)) if 'X' not in s[i:i+k])
    x, y = counts(a), counts(b)
    return sum((x & y).values()) / max(1, min(sum(x.values()), sum(y.values())))


def excluded(sequence, pdb_id, superfamily, targets, cath=None):
    if str(pdb_id).lower() in TARGETS:
        return True
    if cath and superfamily and superfamily in cath:
        return True
    return any(kmer_identity(sequence, seq) >= .25 for seq in targets)


def extract_segments(ca, ss, adjacency, *, sequence='', seg_keys=None, metadata=None,
                     min_length=50, max_length=300, min_blocks=3):
    """Array API shared by extraction and local refs.npz regression tests.

    Callers split at incomplete backbones, numbering gaps and CA breaks. Each
    contiguous seg_key run becomes its own Layout, centered independently.
    """
    ca, ss, adjacency = np.asarray(ca), np.asarray(ss), np.asarray(adjacency)
    if ca.shape != (len(ss), 3) or adjacency.shape != (len(ss), len(ss)):
        raise ValueError('CA/SS/ADJ length mismatch')
    keys = np.zeros(len(ss), int) if seg_keys is None else np.asarray(seg_keys)
    if keys.shape != ss.shape:
        raise ValueError('seg_keys must have one entry per residue')
    cuts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1])+1, len(ss)]
    out = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if not min_length <= hi-lo <= max_length or not np.isfinite(ca[lo:hi]).all():
            continue
        nb = len(ec.he_blocks_from_ss(ss[lo:hi]))
        if not min_blocks <= nb <= MAX_BLOCKS:
            continue
        lay = extract_native_layout(ca[lo:hi], ss[lo:hi], adjacency[lo:hi, lo:hi])
        lay.metadata.update(metadata or {})
        lay.metadata.update(seg_key=str(keys[lo]), start=int(lo), stop=int(hi),
                            sequence=sequence[lo:hi], split='candidate')
        out.append(lay)
    return out


def write_dataset(path, layouts):
    """Single versioned, padded NPZ, allow_pickle=False; accompanying CSV index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(layouts)
    data = dict(version=np.array(1), ss=np.full((n, MAX_LENGTH), 2, np.int8),
                length=np.zeros(n, np.int16), n_blocks=np.zeros(n, np.int8),
                n_res=np.zeros((n, MAX_BLOCKS), np.int16), types=np.full((n, MAX_BLOCKS), 2, np.int8),
                loop_before=np.zeros((n, MAX_BLOCKS), np.int16), loop_after=np.zeros((n, MAX_BLOCKS), np.int16),
                centroids=np.zeros((n, MAX_BLOCKS, 3), np.float32),
                axes=np.zeros((n, MAX_BLOCKS, 3), np.float32),
                normals=np.zeros((n, MAX_BLOCKS, 3), np.float32))
    for key, dtype, fill in [('adjacency', np.int8, 2), ('orientation', np.int8, 0),
                              ('register', np.int16, 0), ('pairing', bool, False)]:
        data[key] = np.full((n, MAX_BLOCKS, MAX_BLOCKS), fill, dtype)
    metadata, rows = [], []
    for i, lay in enumerate(layouts):
        b, l = len(lay.blocks), len(lay.ss)
        if b > MAX_BLOCKS or l > MAX_LENGTH:
            raise ValueError('layout exceeds dataset bounds')
        data['length'][i], data['n_blocks'][i] = l, b
        data['ss'][i, :l] = lay.ss
        starts = np.array([x['start'] for x in lay.blocks]); stops = np.array([x['stop'] for x in lay.blocks])
        data['n_res'][i, :b] = lay.n_res
        data['types'][i, :b] = np.where(lay.types == 'H', 0, 1)
        data['loop_before'][i, :b] = starts-np.r_[0, stops[:-1]]
        data['loop_after'][i, :b] = np.r_[starts[1:], l]-stops
        for key in ('centroids', 'axes', 'normals'):
            data[key][i, :b] = getattr(lay, key)
        for key in ('adjacency', 'orientation', 'register', 'pairing'):
            data[key][i, :b, :b] = getattr(lay, key)
        metadata.append(json.dumps(lay.metadata, sort_keys=True))
        rows.append(dict(index=i, source=lay.metadata.get('source', ''),
                         chain_id=lay.metadata.get('chain_id', ''), seg_key=lay.metadata.get('seg_key', ''),
                         split=lay.metadata.get('split', ''), length=l, n_blocks=b,
                         n_strands=int((lay.types == 'E').sum()),
                         mean_plddt=lay.metadata.get('mean_plddt', ''), sequence=lay.metadata.get('sequence', '')))
    data['metadata'] = np.asarray(metadata, dtype=str)
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **data)
    temp.replace(path)
    with path.with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else
                                ['index', 'source', 'chain_id', 'seg_key', 'split', 'length', 'n_blocks', 'n_strands', 'mean_plddt', 'sequence'])
        writer.writeheader(); writer.writerows(rows)


def read_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        # NpzFile decompresses an entire member on every indexing operation.
        # Cache each padded array once, rather than once per example.
        f = {key: archive[key] for key in archive.files}
        if int(f['version']) != 1:
            raise ValueError('unsupported dataset format')
        for i, (l, b) in enumerate(zip(f['length'], f['n_blocks'])):
            yield Layout(f['ss'][i, :l], f['centroids'][i, :b], f['axes'][i, :b],
                         f['normals'][i, :b], **{k: f[k][i, :b, :b] for k in
                         ('adjacency', 'orientation', 'register', 'pairing')},
                         metadata=json.loads(str(f['metadata'][i])))


def statistics(layouts):
    states, pairing = [], []
    for lay in layouts:
        strands = np.flatnonzero(lay.types == 'E')
        for i, j in zip(strands[:-1], strands[1:]):
            if lay.adjacency[i, j] != 2:
                states.append(int(lay.adjacency[i, j]))
                pairing.append(bool(lay.pairing[i, j]))
    s = np.asarray(states)
    return dict(n=len(layouts), block_histogram=dict(sorted(Counter(len(x.blocks) for x in layouts).items())),
                strand_block_fraction=float(sum((x.types == 'E').sum() for x in layouts)/max(1, sum(len(x.blocks) for x in layouts))),
                beta_enriched_fraction=float(np.mean([(x.types == 'E').sum() >= 4 for x in layouts])) if layouts else 0.,
                adjacent_strand_pairs=len(states),
                hairpin_contact_fraction=float(np.mean(s == 1)) if len(s) else None,
                jump_state0_fraction=float(np.mean(s == 0)) if len(s) else None,
                adjacent_geometric_pair_fraction=float(np.mean(pairing)) if pairing else None,
                labelstats_state0_reference=.51)


def _cluster_key(metadata):
    value = metadata.get('cluster_rep')
    return str(value) if value is not None and str(value) not in ('', 'nan', 'None') else None


def screen_global(layouts, targets, *, clusters=None, candidate_hits=()):
    """Exclude direct hits and their entire AFDB cluster before quotas/splitting.

    targets maps target names to all their protein sequences. candidate_hits
    includes full-chain hits absent from old pieces (screened during extraction).
    Counts overlap across targets; each count is retained candidate segments.
    """
    clusters = {} if clusters is None else clusters
    blocked = {name: set() for name in targets}
    direct = {name: set() for name in targets}
    for cluster, names in candidate_hits:
        for name in names:
            blocked[name].add(str(cluster))
    for i, lay in enumerate(layouts):
        meta = lay.metadata
        if meta['source'] == 'afdb':
            cluster = clusters.get(meta['chain_id'], _cluster_key(meta))
            if not _cluster_key(dict(cluster_rep=cluster)):
                raise ValueError(f"missing AFDB cluster_rep for {meta['chain_id']}")
            meta['cluster_rep'] = str(cluster)
        for name, seqs in targets.items():
            if (meta['chain_id'].split(':')[0].lower() == name or
                    any(kmer_identity(meta.get('sequence', ''), seq) >= .25 for seq in seqs)):
                direct[name].add(i)
                if meta['source'] == 'afdb':
                    blocked[name].add(meta['cluster_rep'])
    excluded_ids, counts = set(), {}
    for name in targets:
        direct[name].update(i for i, x in enumerate(layouts)
            if x.metadata['source'] == 'pdb' and 'pdbchain:'+x.metadata['chain_id'] in blocked[name])
        ids = direct[name] | {i for i, x in enumerate(layouts)
            if x.metadata['source'] == 'afdb' and _cluster_key(x.metadata) in blocked[name]}
        excluded_ids.update(ids)
        counts[name] = dict(direct=len(direct[name]), clusters=sum(not c.startswith('pdbchain:') for c in blocked[name]), excluded=len(ids))
    return [x for i, x in enumerate(layouts) if i not in excluded_ids], counts


def _scan_cluster(task):
    """Full-chain scan, including candidates rejected by the original extractor."""
    cluster, row, tar_path, targets = task
    with open(tar_path, 'rb') as fh:
        fh.seek(int(row['offset_data']))
        st = parse_atoms(fh.read(int(row['size'])))
    if row.get('chain'):
        st = st[st.chain_id == row['chain']]
    chains = ec.protein_chains(st)
    if not chains:
        raise ValueError(f'no protein chain while screening {cluster}')
    hits = set()
    for chain in chains:
        sequence = chain_sequence(chain)
        for name, seqs in targets.items():
            if any(kmer_identity(sequence, seq) >= .25 for seq in seqs):
                hits.add(name)
    return cluster, sorted(hits)


def merge_screen_inputs(layouts, args):
    """Recover legacy-piece clusters and scan every member of retained clusters.

    Only clusters represented in the pieces can affect the merged data. Scanning
    ALL their manifest members also catches hits removed by the old .4 screen,
    rejected by layout filters, or assigned to a different piece. No re-extraction.
    """
    import pandas as pd
    index = pd.read_parquet(args.mirror_index)
    targets = target_sequences(index, args.mirror_tar, named=True)
    manifest = pd.read_parquet(args.afdb_manifest)
    cluster_map = dict(zip(manifest.uniprot_id.astype(str), manifest.cluster_rep.astype(str)))
    ids = {x.metadata['chain_id'] for x in layouts if x.metadata['source'] == 'afdb'}
    missing = ids - cluster_map.keys()
    if missing:
        raise ValueError(f'AFDB ids absent from manifest: {sorted(missing)[:5]}')
    relevant = {cluster_map[i] for i in ids}
    members = manifest[manifest.cluster_rep.astype(str).isin(relevant)]
    tasks = [(str(r['cluster_rep']), r, args.afdb_tar, targets) for r in members.to_dict('records')]
    # Old segment metadata lacks the whole-chain sequence. Recover PDB full-chain
    # hits as well; a low-identity hit can be outside the retained segment.
    pdb_ids = {x.metadata['chain_id'] for x in layouts if x.metadata['source'] == 'pdb'}
    wanted_pids = {chain_id.split(':', 1)[0].lower() for chain_id in pdb_ids}
    mirror_subset = index[index.pdb_id.str.lower().isin(wanted_pids)]
    mirror_rows = {str(row['pdb_id']).lower(): row for row in mirror_subset.to_dict('records')}
    for chain_id in sorted(pdb_ids):
        pid, chain = chain_id.split(':', 1)
        if pid.lower() not in mirror_rows:
            raise ValueError(f'missing PDB candidate {pid} in mirror index')
        row = dict(mirror_rows[pid.lower()]); row['chain'] = chain
        tasks.append(('pdbchain:'+chain_id, row, args.mirror_tar, targets))
    # Independent tar offsets / CIF parsing / kmer comparisons parallelize by chain.
    print(json.dumps(dict(stage='global_screen', afdb_members=len(members), pdb_chains=len(pdb_ids),
                          workers=args.workers)), flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        hits = list(pool.map(_scan_cluster, tasks, chunksize=32))
    if getattr(args, 'cath_csv', None):
        with open(args.cath_csv) as handle:
            rows = list(csv.DictReader(handle))
        families = {r['superfamily'] for r in rows if r['pdb_id'].lower() in TARGETS}
        excluded_chains = {r['pdb_id'].lower()+':'+r['chain'] for r in rows if r['superfamily'] in families}
        # Family-only exclusions get a separate counter, not attributed to one target.
        targets['cath_superfamily'] = []
        hits.extend(('pdbchain:'+chain_id, ['cath_superfamily']) for chain_id in excluded_chains & pdb_ids)
    return targets, cluster_map, hits


def merge(paths, out, *, afdb=40000, pdb=10000, holdout=2000, seed=20261014, allow_shortfall=False,
          targets=None, clusters=None, candidate_hits=(), screen_args=None):
    if not paths:
        raise ValueError('merge requires at least one piece')
    if any(Path(p).resolve() == Path(out).resolve() for p in paths):
        raise ValueError('merge output must differ from pieces')
    layouts, seen = [], set()
    for path in paths:
        for lay in read_dataset(path):
            # Whole sequences group both segments and duplicated pn_units at split time.
            key = (lay.metadata['source'], lay.metadata['chain_id'], lay.metadata['seg_key'])
            if key not in seen:
                layouts.append(lay); seen.add(key)
    if screen_args is not None:
        targets, clusters, candidate_hits = merge_screen_inputs(layouts, screen_args)
    # The CLI always screens. Array-API callers must explicitly supply fixtures.
    if targets is None:
        raise ValueError('merge requires named target sequences or screen_args')
    layouts, exclusion_counts = screen_global(layouts, targets, clusters=clusters, candidate_hits=candidate_hits)
    rng = np.random.default_rng(seed)
    layouts.sort(key=lambda x: (x.metadata['source'], x.metadata['chain_id'], x.metadata['seg_key']))
    rng.shuffle(layouts)
    print(json.dumps(dict(excluded_per_target=exclusion_counts, after_exclusion=len(layouts)), sort_keys=True), flush=True)
    selected = []
    counts = {}
    for source, quota in [('afdb', afdb), ('pdb', pdb)]:
        for beta, need in [(True, quota//2), (False, quota-quota//2)]:
            group = [x for x in layouts if x.metadata['source'] == source and ((x.types == 'E').sum() >= 4) == beta]
            counts[f'{source}_beta{int(beta)}'] = len(group)
            if len(group) < need and not allow_shortfall:
                raise ValueError(f'quota shortfall {source} beta={beta}: {len(group)} < {need}; rerun with more time/pieces or explicitly --allow-shortfall')
            selected.extend(group[:need])
    rng.shuffle(selected)
    if holdout < 0 or holdout >= len(selected):
        raise ValueError('holdout must leave training examples')
    # Union chain, exact sequence and cluster, including transitive links.
    parent = list(range(len(selected)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    groups = {}
    for i, lay in enumerate(selected):
        keys = [('chain', lay.metadata['source'], lay.metadata['chain_id'])]
        if lay.metadata.get('sequence'):
            keys.append(('sequence', lay.metadata['sequence']))
        if _cluster_key(lay.metadata):
            keys.append(('cluster', lay.metadata['source'], _cluster_key(lay.metadata)))
        for key in keys:
            if key in groups:
                parent[find(i)] = find(groups[key])
            else:
                groups[key] = i
    group_sizes = Counter(find(i) for i in range(len(selected)))
    val_groups, nval = set(), 0
    for i in rng.permutation(len(selected)):
        g = find(i)
        if nval >= holdout:
            break
        if g not in val_groups:
            val_groups.add(g); nval += group_sizes[g]
    if nval >= len(selected):
        raise ValueError('holdout leaves no training examples')
    for i, lay in enumerate(selected):
        lay.metadata['split'] = 'val' if find(i) in val_groups else 'train'
    write_dataset(out, selected)
    stats = dict(statistics(selected), available=counts, validation=nval, requested_validation=holdout, excluded_per_target=exclusion_counts)
    print(json.dumps(stats, sort_keys=True), flush=True)
    return stats


def parse_atoms(raw, *, author=False):
    import biotite.structure.io.pdbx as pdbx
    if raw[:2] == b'\x1f\x8b':
        raw = zlib.decompress(raw, 47)
    return pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(raw.decode())), model=1,
                              use_author_fields=author, altloc='first')


def chain_sequence(chain):
    from biotite.sequence import ProteinSequence
    letters = []
    for name in chain.res_names:
        try:
            letters.append(ProteinSequence.convert_letter_3to1(name))
        except KeyError:
            letters.append('X')
    return ''.join(letters)


def target_sequences(index, tar_path, *, named=False):
    """All protein chains of each target; requiring all 9 avoids partial screens."""
    import biotite.structure as struc
    seqs, by_target = [], {}
    with open(tar_path, 'rb') as fh:
        for pid in TARGETS:
            rows = index[index.pdb_id.str.lower() == pid]
            if rows.empty:
                raise ValueError(f'missing leakage target {pid} in mirror index')
            row = rows.iloc[0]; fh.seek(int(row.offset_data))
            st = parse_atoms(fh.read(int(row['size'])))
            st = st[struc.filter_amino_acids(st)]
            by_target[pid] = [chain_sequence(c) for c in ec.protein_chains(st)]
            seqs.extend(by_target[pid])
    if not seqs or any(not s for s in seqs):
        raise ValueError('empty target sequence')
    return by_target if named else seqs


_WORK = {}


def _init_worker(targets, cath, tar_paths):
    # Unix fork shares already-loaded label module; each process has private file offsets.
    _WORK.update(targets=targets, cath=cath, files={s: open(p, 'rb') for s, p in tar_paths.items()},
                 lbl=ec.load_label_module())


def _extract_one(task):
    import biotite.structure as struc
    import build_specs as bs
    source, row = task
    try:
        fh = _WORK['files'][source]; fh.seek(int(row['offset_data']))
        st = parse_atoms(fh.read(int(row['size'])))
        st = st[struc.filter_amino_acids(st)]
        if source == 'pdb':
            st = st[st.chain_id == row['chain']]
        if len(np.unique(st.chain_id)) != 1:
            return [], 'multiple_or_missing_chain'
        chains = ec.protein_chains(st)
        if len(chains) != 1 or not 50 <= len(chains[0]) <= 300:
            return [], 'chain_or_length'
        ch = chains[0]
        sequence = chain_sequence(ch)
        # Global leakage/cluster screening is done by merge, including full chains.
        # Missing coordinates become isolated rejected segments. Numbering jumps,
        # missing residues and physical CA breaks all split the segmentation.
        good = np.isfinite(ch.n).all(1) & np.isfinite(ch.ca).all(1) & np.isfinite(ch.c).all(1)
        breaks = (np.diff(ch.res_ids) > 1) | (np.diff(ch.res_ids) < 0) | (np.linalg.norm(np.diff(ch.ca, axis=0), axis=1) > 4.5)
        breaks |= ~good[1:] | ~good[:-1]
        cuts = np.r_[0, np.flatnonzero(breaks)+1, len(ch)]
        out = []
        for seg, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
            if hi-lo < 50 or not good[lo:hi].all():
                continue
            atoms = bs.select_residue_range(st, int(lo), int(hi))
            ss, adj = _WORK['lbl'].get_ss_adj(atoms, loop_adj=False)
            meta = dict(source=source, chain_id=row['id'], seg_key=str(seg),
                        mean_plddt=row.get('mean_plddt', ''), cluster_rep=row.get('cluster_rep', ''),
                        superfamily=row.get('superfamily', ''),
                        original_start=int(lo), original_stop=int(hi))
            out.extend(extract_segments(ch.ca[lo:hi], ss, adj, sequence=sequence[lo:hi],
                                       seg_keys=np.full(hi-lo, seg), metadata=meta))
        return out, 'ok' if out else 'block_or_segment_filter'
    except Exception as exc:
        return [], f'{type(exc).__name__}: {str(exc)[:120]}'


def candidates(args, index):
    import pandas as pd
    af = pd.read_parquet(args.afdb_manifest)
    af = af[af.n_residues.between(50, 300) & (af.mean_plddt >= 80)].copy()
    rng = np.random.default_rng(args.seed)
    # Weighted permutation, partitioned AFTER shuffle: disjoint candidate rows.
    weights = np.maximum(af.weight.to_numpy(float), 1e-12)
    af = af.iloc[np.argsort(-np.log(np.maximum(rng.random(len(af)), 1e-12))/weights)]
    af = af.iloc[args.piece::args.pieces]
    afrows = []
    for r in af.to_dict('records'):
        r['id'] = str(r['uniprot_id']); afrows.append(('afdb', r))
    cols = ['pdb_id', 'q_pn_unit_iid', 'q_pn_unit_type', 'q_pn_unit_sequence_length', 'q_pn_unit_is_polymer']
    pdb = pd.read_parquet(args.pdb_manifest, columns=cols)
    pdb = pdb[pdb.q_pn_unit_is_polymer & (pdb.q_pn_unit_type.astype(int) == 6) & pdb.q_pn_unit_sequence_length.between(50, 300)].copy()
    pdb['pdb_id'] = pdb.pdb_id.str.lower()
    pdb['chain'] = pdb.q_pn_unit_iid.astype(str).str.split('_').str[0]
    pdb = pdb.drop_duplicates(['pdb_id', 'chain'])
    pdb = pdb.iloc[rng.permutation(len(pdb))].iloc[args.piece::args.pieces]
    pdb = pdb.merge(index[['pdb_id', 'offset_data', 'size']], on='pdb_id', how='inner')
    pdrows = []
    for r in pdb.to_dict('records'):
        r['id'] = f"{r['pdb_id']}:{r['chain']}"
        pdrows.append(('pdb', r))
    # Interleave 4:1 so a time-bounded piece doesn't starve PDB.
    for i in range(max((len(afrows)+3)//4, len(pdrows))):
        yield from afrows[4*i:4*i+4]
        yield from pdrows[i:i+1]


def piece_complete(path):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        # Exhaust the generator so truncated metadata/arrays cannot pass.
        list(read_dataset(path))
        return True
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return False


def extract(args):
    out = Path(args.out_dir) / f'piece_{args.piece:02d}.npz'
    if piece_complete(out):
        print(json.dumps(dict(piece=args.piece, skipped=True, output=str(out))), flush=True)
        return
    import pandas as pd
    started = time.monotonic()
    index = pd.read_parquet(args.mirror_index)
    targets = []  # E1 screening is a global merge operation, never piece-local.
    cath, lookup = set(), {}
    if args.cath_csv:
        for r in csv.DictReader(open(args.cath_csv)):
            lookup[(r['pdb_id'].lower(), r['chain'])] = r['superfamily']
            if r['pdb_id'].lower() in TARGETS:
                cath.add(r['superfamily'])
    def tasks():
        for source, row in candidates(args, index):
            row['superfamily'] = lookup.get((row.get('pdb_id', ''), row.get('chain', '')), '')
            yield source, row
    iterator = iter(tasks())
    kept, reasons, quota = [], Counter(), Counter()
    needs = {('afdb', b): (args.afdb+args.pieces-1)//args.pieces//2+1 for b in (False, True)}
    needs.update({('pdb', b): (args.pdb+args.pieces-1)//args.pieces//2+1 for b in (False, True)})
    # Load before fork to share module memory. Workers use one BLAS thread each.
    ec.load_label_module()
    pool = ProcessPoolExecutor(args.workers, mp_context=mp.get_context('fork'), initializer=_init_worker,
                               initargs=(targets, cath, {'afdb': args.afdb_tar, 'pdb': args.mirror_tar}))
    pending, exhausted = set(), False
    try:
        while time.monotonic()-started < args.budget and not all(quota[k] >= n for k, n in needs.items()):
            while len(pending) < args.workers*2 and not exhausted:
                try:
                    pending.add(pool.submit(_extract_one, next(iterator)))
                except StopIteration:
                    exhausted = True
            if not pending:
                break
            done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                layouts, reason = future.result(); reasons[reason] += 1
                for lay in layouts:
                    key = (lay.metadata['source'], (lay.types == 'E').sum() >= 4)
                    if quota[key] < needs[key]:
                        kept.append(lay); quota[key] += 1
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    out = Path(args.out_dir) / f'piece_{args.piece:02d}.npz'
    write_dataset(out, kept)
    print(json.dumps(dict(statistics(kept), piece=args.piece, elapsed=time.monotonic()-started,
                          filter_counts=dict(reasons)), sort_keys=True), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    x = sub.add_parser('extract')
    x.add_argument('--piece', type=int, required=True); x.add_argument('--pieces', type=int, default=12)
    x.add_argument('--workers', type=int, default=48); x.add_argument('--budget', type=float, default=420.)
    x.add_argument('--out-dir', default=str(DEFAULT_OUT))
    for name, default in [('afdb-manifest', AFDB_MANIFEST), ('afdb-tar', AFDB_TAR),
                          ('pdb-manifest', PDB_MANIFEST), ('mirror-index', ec.PDB_MIRROR_INDEX), ('mirror-tar', ec.PDB_MIRROR_TAR)]:
        x.add_argument('--'+name, default=str(default))
    x.add_argument('--cath-csv', help='legacy argument; repeat on merge for global superfamily exclusion')
    x.add_argument('--seed', type=int, default=20261014)
    m = sub.add_parser('merge'); m.add_argument('--pieces', nargs='+', required=True)
    m.add_argument('--out', default=str(DEFAULT_OUT/'layouts.npz'))
    m.add_argument('--holdout', type=int, default=2000); m.add_argument('--seed', type=int, default=20261014)
    m.add_argument('--allow-shortfall', action='store_true')
    m.add_argument('--remove-pieces', action='store_true', help='explicit destructive cleanup after merge; avoid for reruns')
    m.add_argument('--workers', type=int, default=48)
    m.add_argument('--cath-csv', help='optional pdb_id,chain,superfamily CSV for global exclusion')
    for name, default in [('afdb-manifest', AFDB_MANIFEST), ('afdb-tar', AFDB_TAR),
                          ('mirror-index', ec.PDB_MIRROR_INDEX), ('mirror-tar', ec.PDB_MIRROR_TAR)]:
        m.add_argument('--'+name, default=str(default))
    for parser in (x, m):
        parser.add_argument('--afdb', type=int, default=40000); parser.add_argument('--pdb', type=int, default=10000)
    a = p.parse_args(argv)
    if a.workers < 1:
        p.error('positive workers required')
    if min(a.afdb, a.pdb) < 0:
        p.error('source quotas must be nonnegative')
    if a.cmd == 'extract':
        if a.workers < 1 or not 0 <= a.piece < a.pieces or a.budget <= 0:
            p.error('invalid piece/workers/budget')
        extract(a)
    else:
        merge(a.pieces, a.out, afdb=a.afdb, pdb=a.pdb, holdout=a.holdout, seed=a.seed, allow_shortfall=a.allow_shortfall, screen_args=a)
        if a.remove_pieces:
            for path in a.pieces:
                Path(path).unlink()
                Path(path).with_suffix('.csv').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
