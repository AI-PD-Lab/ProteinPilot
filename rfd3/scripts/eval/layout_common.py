"""CPU block-layout geometry shared by the B0 planner and ideal-sketch oracle.

Indices are zero based, stops exclusive.  Layouts contain frames, never native
per-residue coordinates.  State 2 is unknown; hot/cold states 3/4 mean 1/0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
import math
from pathlib import Path
from functools import cached_property, lru_cache
from scipy.spatial.distance import cdist, pdist, squareform

import numpy as np
import eval_common as ec


def unit(v, fallback=None):
    v = np.asarray(v, float)
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        if fallback is None:
            raise ValueError("undefined direction")
        return unit(fallback)
    return v / norm


def perpendicular(u):
    u = unit(u)
    return unit(np.cross(u, np.eye(3)[np.argmin(abs(u))]))


def principal_axis(ca):
    ca = np.asarray(ca, float)
    if len(ca) < 2:
        return np.array([0., 0., 1.])
    u = np.linalg.svd(ca - ca.mean(0), full_matrices=False)[2][0]
    return u if np.dot(u, ca[-1] - ca[0]) >= 0 else -u


@dataclass
class Layout:
    ss: np.ndarray
    centroids: np.ndarray
    axes: np.ndarray
    normals: np.ndarray | None = None
    adjacency: np.ndarray | None = None  # block matrix [B,B], not residue tensor
    orientation: np.ndarray | None = None  # +1/-1 only for geometrically paired strands
    register: np.ndarray | None = None
    pairing: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        raw = np.asarray(self.ss)
        if raw.ndim != 1 or not np.isin(raw, [0, 1, 2, 3]).all() or not len(raw):
            raise ValueError("ss must be a nonempty 1-D 0/1/2/3 array")
        self.ss = raw.astype(np.int8)
        b = len(self.blocks)
        self.centroids = np.asarray(self.centroids, float).reshape(b, 3).copy()
        self.axes = np.asarray(self.axes, float).reshape(b, 3).copy()
        if not np.isfinite(self.centroids).all() or not np.isfinite(self.axes).all():
            raise ValueError("nonfinite frame")
        self.axes = np.array([unit(u) for u in self.axes]).reshape(b, 3)
        if self.normals is None:
            self.normals = np.array([perpendicular(u) for u in self.axes]).reshape(b, 3)
        else:
            self.normals = np.asarray(self.normals, float).reshape(b, 3).copy()
            if not np.isfinite(self.normals).all():
                raise ValueError("nonfinite normal")
            self.normals = np.array([unit(n - np.dot(n, u) * u, perpendicular(u))
                                     for n, u in zip(self.normals, self.axes)]).reshape(b, 3)
        self.adjacency = (np.full((b, b), 2, np.int8) if self.adjacency is None
                          else block_adjacency(self.ss, self.adjacency))
        self.orientation = self._matrix(self.orientation, b, np.int8)
        self.register = self._matrix(self.register, b, np.int32)
        self.pairing = self._matrix(self.pairing, b, bool)
        self.metadata = copy.deepcopy(self.metadata)

    @staticmethod
    def _matrix(a, b, dtype):
        return np.zeros((b, b), dtype=dtype) if a is None else np.asarray(a, dtype=dtype).reshape(b, b).copy()

    @cached_property
    def blocks(self):
        return ec.he_blocks_from_ss(self.ss)

    @cached_property
    def types(self):
        return np.asarray([b['type'] for b in self.blocks])

    @property
    def pair_normals(self):
        """Oriented sheet-plane normal for each paired strand pair."""
        out = np.zeros((len(self.blocks), len(self.blocks), 3))
        for i, j in zip(*np.where(self.pairing)):
            out[i, j] = unit(np.cross(self.axes[i], self.centroids[j] - self.centroids[i]),
                             self.normals[i])
        return out

    @cached_property
    def n_res(self):
        return np.asarray([b['stop'] - b['start'] for b in self.blocks], int)

    @cached_property
    def lengths(self):
        return np.where(self.types == 'H', 1.5, 3.3) * (self.n_res - 1)

    def copy(self):
        return copy.deepcopy(self)

    def centered(self):
        out = self.copy()
        if len(out.centroids):
            out.centroids -= out.centroids.mean(0)
        return out

    def save(self, path):
        np.savez_compressed(path, version=np.array(1), ss=self.ss,
                            centroids=self.centroids, axes=self.axes, normals=self.normals,
                            adjacency=self.adjacency, orientation=self.orientation,
                            register=self.register, pairing=self.pairing,
                            metadata=np.array(json.dumps(self.metadata)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            if int(f['version']) != 1:
                raise ValueError("unsupported layout format")
            data = {k: f[k] for k in ('ss', 'centroids', 'axes', 'normals', 'adjacency',
                                     'orientation', 'register', 'pairing')}
            return cls(**data, metadata=json.loads(str(f['metadata'])))


def block_adjacency(ss, adj):
    """Collapse uniform (ignoring masks) residue rectangles, or accept a block map.

    Signed orientations are separate from states: -1 here is rejected to avoid
    silently treating an antiparallel contact as a negative contact.
    """
    blocks = ec.he_blocks_from_ss(ss)
    b = len(blocks)
    adj = np.asarray(adj)
    if not np.isin(adj, [0, 1, 2, 3, 4]).all():
        raise ValueError("adjacency states must be 0..4; pass signs separately")
    adj = np.where(adj == 3, 1, np.where(adj == 4, 0, adj)).astype(np.int8)
    if adj.shape == (b, b):
        out = adj.copy()
    elif adj.shape == (len(ss), len(ss)):
        out = np.full((b, b), 2, np.int8)
        for i in range(b):
            for j in range(i + 1, b):
                values = np.unique(adj[np.ix_(ec.block_idx(blocks, i), ec.block_idx(blocks, j))])
                values = values[values != 2]
                if len(values) > 1:
                    raise ValueError(f"conflicting states in block pair {i},{j}")
                out[i, j] = out[j, i] = values[0] if len(values) else 2
        # Check the reverse rectangles too (do not silently symmetrise bad input).
        if not np.array_equal(adj, adj.T):
            raise ValueError("adjacency must be symmetric")
    else:
        raise ValueError("adjacency must be [B,B] or [L,L]")
    if not np.array_equal(out, out.T):
        raise ValueError("adjacency must be symmetric")
    np.fill_diagonal(out, 2)
    return out


def split_signed_adjacency(ss, signed_adj):
    """Explicit signed dialect: +1/-1 paired polarity, 0 no contact, 2 unknown.

    Ordinary state-1 maps must NOT be passed here: they have unknown polarity.
    Accepts block or uniform residue rectangles and returns (states, signs).
    """
    raw = np.asarray(signed_adj)
    if not np.isin(raw, [-1, 0, 1, 2]).all():
        raise ValueError("signed maps allow only -1,0,1,2")
    positive = block_adjacency(ss, np.where(raw == -1, 1, raw))
    # Encode the sign as a temporary state, so block_adjacency validates
    # uniformity before collapsing (mixed +/- contacts are an input error).
    encoded = block_adjacency(ss, np.where(raw == -1, 0, np.where(raw == 0, 2, raw)))
    signs = np.where(positive == 1, np.where(encoded == 0, -1, 1), 0).astype(np.int8)
    types = np.asarray([b['type'] for b in ec.he_blocks_from_ss(ss)])
    allowed = (types[:, None] == 'E') & (types[None, :] == 'E')
    if np.any((signs != 0) & ~allowed):
        raise ValueError("signed contacts must be strand-strand")
    return positive, signs


def strand_pair(ca, blocks, i, j):
    """N029 probe pairing: at least two CA pairs strictly below 5.5 A."""
    a, b = ca[ec.block_idx(blocks, i)], ca[ec.block_idx(blocks, j)]
    d = ec.pairwise(a, b)
    p, q = np.unravel_index(np.argmin(d), d.shape)
    return dict(paired=bool((d < 5.5).sum() >= 2), n5p5=int((d < 5.5).sum()),
                orientation=1 if np.dot(principal_axis(a), principal_axis(b)) >= 0 else -1,
                register=int(p - q), minimum=float(d.min()))


def extract_native_layout(ca, ss, adjacency=None):
    ca = np.asarray(ca, float)
    if ca.shape != (len(ss), 3) or not np.isfinite(ca).all():
        raise ValueError("complete finite CA [L,3] required")
    blocks = ec.he_blocks_from_ss(ss)
    centers, axes, normals = [], [], []
    for b in blocks:
        a = ca[b['start']:b['stop']]
        c, u = a.mean(0), principal_axis(a)
        residual = a - c - np.outer((a - c) @ u, u)
        # Retain only a frame roll (one direction), never residue-level shape.
        n = ((residual * (-1.) ** np.arange(len(a))[:, None]).sum(0)
             if b['type'] == 'E' else residual[0])
        centers.append(c); axes.append(u); normals.append(unit(n, perpendicular(u)))
    layout = Layout(ss, centers, axes, normals, adjacency, metadata={'native_layout': True})
    for i, a in enumerate(blocks):
        for j in range(i + 1, len(blocks)):
            if a['type'] != 'E' or blocks[j]['type'] != 'E':
                continue
            p = strand_pair(ca, blocks, i, j)
            if p['paired'] and (adjacency is None or layout.adjacency[i, j] == 1):
                layout.pairing[i, j] = layout.pairing[j, i] = True
                layout.orientation[i, j] = layout.orientation[j, i] = p['orientation']
                layout.register[i, j], layout.register[j, i] = p['register'], -p['register']
    return layout.centered()


def refresh_strand_pairs(layout, ca):
    """Refresh derived pairing/orientation/register after frame optimisation."""
    layout.pairing[:] = False
    layout.orientation[:] = 0
    layout.register[:] = 0
    for i, a in enumerate(layout.blocks):
        for j in range(i + 1, len(layout.blocks)):
            if a['type'] != 'E' or layout.types[j] != 'E' or layout.adjacency[i, j] != 1:
                continue
            p = strand_pair(ca, layout.blocks, i, j)
            if p['paired']:
                layout.pairing[i, j] = layout.pairing[j, i] = True
                layout.orientation[i, j] = layout.orientation[j, i] = p['orientation']
                layout.register[i, j], layout.register[j, i] = p['register'], -p['register']
    return layout


def crossover_values(layout):
    """Signed scalar triple products of consecutive E-H-E units.

    The formula is from planner_design section 1. Negative is the measured
    right-handed convention (1URN); the native-unit test guards it.
    """
    out = []
    for i in range(len(layout.blocks) - 2):
        if (tuple(layout.types[i:i + 3]) == ('E', 'H', 'E') and
                layout.blocks[i+1]['start']>layout.blocks[i]['stop'] and
                layout.blocks[i+2]['start']>layout.blocks[i+1]['stop']):
            a, h, b = layout.centroids[i:i + 3]
            out.append((i, i + 1, i + 2, float(np.dot(h - (a + b) / 2,
                        np.cross(layout.axes[i], b - a)))))
    return out


def handedness(layout):
    vals = [v for _, _, _, v in crossover_values(layout)]
    return {'n': len(vals), 'right': sum(v < -1e-8 for v in vals),
            'left': sum(v > 1e-8 for v in vals), 'values': vals}


def sheet_twist(layout, i, j):
    """Signed angle between N->C axes, aligned in polarity, about interstrand direction."""
    u, v = layout.axes[[i, j]]
    v = v * (1 if u @ v >= 0 else -1)
    d = layout.centroids[j] - layout.centroids[i]
    d = unit(d - (d @ u) * u)
    return float(np.arctan2(np.dot(np.cross(u, v), d), np.dot(u, v)))


def koga_rules(layout):
    """Loop-length rules; P/A describe terminal CA->CB, NOT helix polarity.

    Candidates only: inactive unless calibration on this label pipeline reaches
    >=90% with >=50 native observations. No rule controls strand roll.
    Koga et al. Nature 2012, doi:10.1038/nature11600, Fig. 1.
    L/R likewise describe strand pleating, separately from antiparallel axes.
    """
    rules = []
    for i, (a, b) in enumerate(zip(layout.blocks, layout.blocks[1:])):
        gap = b['start'] - a['stop']
        kinds = a['type'] + b['type']
        label = ({2: 'L', 3: 'L', 5: 'R'}.get(gap) if kinds == 'EE' else
                 {2: 'P', 3: 'A'}.get(gap) if kinds == 'EH' else
                 'P' if kinds == 'HE' and gap in (2, 3) else None)
        if label:
            rules.append(dict(i=i, j=i + 1, kind=kinds, gap=gap, expected=label))
    return rules


def koga_values(layout):
    from sketch_builder import placed_sse
    values = []
    for rule in koga_rules(layout):
        i, j, kind = rule['i'], rule['j'], rule['kind']
        strand = j if kind == 'HE' else i
        xyz = placed_sse(layout, strand)[0 if kind == 'HE' else -1]
        cue = unit(ec.reconstruct_cb(xyz[None, 0], xyz[None, 1], xyz[None, 2])[0] - xyz[1])
        delta = layout.centroids[j] - layout.centroids[i]
        direction = np.cross(layout.axes[i], delta) if kind == 'EE' else delta
        value = float(cue @ unit(direction, perpendicular(layout.axes[strand])))
        expected_sign = -1 if rule['expected'] in ('L', 'A') else 1
        key = f"{kind}:{rule['gap']}"
        calibration = layout.metadata.get('koga_calibration', {}).get(key, {})
        enabled = (calibration.get('n', 0) >= 50 and calibration.get('accuracy', 0.) >= .9)
        sign = int(calibration.get('sign', 1))
        values.append(dict(rule, value=value, calibrated_value=sign*value, enabled=enabled,
                           satisfied=bool(expected_sign * sign * value > .02)))
    return values


def polarity_constraints(layout):
    """Only true hairpins (2..5 coil residues) constrain inferred polarity.

    A 0/1-residue E split is one inference strand, so its pieces have the
    same sign. E-H-E polarity is probabilistic, including looped crossovers.
    Explicit signed inputs remain supported outside the pack-build path.
    """
    out = {}
    for i, (a, b) in enumerate(zip(layout.blocks, layout.blocks[1:])):
        gap = b['start'] - a['stop']
        if a['type'] == b['type'] == 'E' and gap <= 5:
            out[i, i+1] = 1 if gap <= 1 else -1
    for i, j, sign in layout.metadata.get('signed_pairs', []):
        if (i, j) in out and out[i, j] != sign:
            raise ValueError('signed polarity contradicts a hairpin constraint')
        out[i, j] = sign
    return out


def constraints_ok(layout):
    if any(sign * (layout.axes[i] @ layout.axes[j]) < .5
           for (i, j), sign in polarity_constraints(layout).items()):
        return False
    return all(v < -1e-4 for _, _, _, v in crossover_values(layout))


@lru_cache(maxsize=256)
def _strand_bridges(strands,flat,size):
    adj=np.asarray(flat).reshape(size,size);result=[]
    for at,i in enumerate(strands):
        for j in strands[at+1:]:
            if adj[i,j]!=1:continue
            seen={i};stack=[i]
            while stack:
                k=stack.pop()
                for other in strands:
                    if (k==i and other==j) or (k==j and other==i):continue
                    if other not in seen and adj[k,other]==1:
                        seen.add(other);stack.append(other)
            if j not in seen:result.append((i,j))
    return tuple(result)


@lru_cache(maxsize=1)
def layout_priors():
    """Frozen numpy/stdlib models and population geometry (no target records)."""
    with open(Path(__file__).with_name('layout_priors.json')) as fh:
        return json.load(fh)


LOOP_EDGES = [2, 3, 4, 5, 6, 9, 13, 21]
LOOP_LABELS = ['0-1', '2', '3', '4', '5', '6-8', '9-12', '13-20', '21+']


def loop_bin(x):
    return LOOP_LABELS[sum(x >= edge for edge in LOOP_EDGES)]


def pair_features(types, start, stop, adj):
    """One row per map-adjacent E-E pair (i<j). types 0=H/1=E, start/stop exclusive, adj 1 = contact.

    Every feature is a function of (types, start, stop, adj) only.
    """
    types = np.asarray(types, int); start = np.asarray(start, int); stop = np.asarray(stop, int)
    b = len(types); isE = types == 1; isH = types == 0
    A = np.asarray(adj) == 1
    strands = np.flatnonzero(isE); nE = len(strands); nH = int(isH.sum())
    degE = (A & isE[None, :]).sum(1); degH = (A & isH[None, :]).sum(1)
    nres = stop - start
    rows = []
    for a in range(nE):
        for c in range(a + 1, nE):
            i, j = int(strands[a]), int(strands[c])
            if not A[i, j]:
                continue
            between = np.arange(i + 1, j)
            tb = ''.join('HE'[int(types[k])] for k in between)
            nHb, nEb = tb.count('H'), tb.count('E')
            pattern = tb if len(tb) <= 2 else ('H3+' if nEb == 0 else 'E3+' if nHb == 0 else 'mixed3+')
            sep = int(start[j] - stop[i]); coil = int(sep - nres[between].sum())
            hb = between[isH[between]]; eb = between[isE[between]]
            rows.append(dict(i=i, j=j, pattern=pattern, n_between=len(between), nH_between=nHb, nE_between=nEb,
                             sep=sep, coil=coil, len_i=int(nres[i]), len_j=int(nres[j]), rank_i=a, rank_j=c,
                             n_strands=nE, n_helices=nH, n_blocks=b, degE_i=int(degE[i]), degE_j=int(degE[j]),
                             degH_i=int(degH[i]), degH_j=int(degH[j]),
                             common_E=int((A[i] & A[j] & isE).sum()), common_H=int((A[i] & A[j] & isH).sum()),
                             h_between_adj_both=int((A[i, hb] & A[j, hb]).sum()), h_between_adj_any=int((A[i, hb] | A[j, hb]).sum()),
                             e_between_adj_i=int(A[i, eb].sum()), e_between_adj_j=int(A[j, eb].sum()),
                             e_between_adj_both=int((A[i, eb] & A[j, eb]).sum()),
                             first_i=int(a == 0), last_j=int(c == nE - 1)))
    return rows


def derive_one(f):
    """Derived columns from the raw pair features (mirror of build_polarity_prior.derive)."""
    d = {k: (int(v) if isinstance(v, (bool, int)) or (isinstance(v, float) and float(v).is_integer()) else v) for k, v in f.items()}
    d['first_last'] = d['first_i'] * d['last_j']
    d['len_min'] = min(d['len_i'], d['len_j']); d['len_max'] = max(d['len_i'], d['len_j'])
    d['log_sep'] = math.log1p(max(d['sep'], 0)); d['log_coil'] = math.log1p(max(d['coil'], 0))
    d['nH_between_c'] = min(d['nH_between'], 4); d['nE_between_c'] = min(d['nE_between'], 6)
    den = max(d['n_strands'] - 1, 1)
    d['rank_i_frac'] = d['rank_i'] / den; d['rank_j_frac'] = d['rank_j'] / den; d['rank_gap'] = d['rank_j'] - d['rank_i']
    d['frac_E_blocks'] = d['n_strands'] / d['n_blocks']
    d['hairpin'] = int(d['n_between'] == 0); d['hairpin_le8'] = int(d['n_between'] == 0 and d['sep'] <= 8)
    d['loop_bin'] = loop_bin(d['coil']); d['hp_loop_bin'] = d['loop_bin'] if d['hairpin'] else 'none'
    d['hb01'] = int(d['h_between_adj_both'] > 0); d['commonE01'] = int(d['common_E'] > 0)
    d['degE_min_c'] = min(d['degE_i'], d['degE_j'], 4); d['degE_max_c'] = min(max(d['degE_i'], d['degE_j']), 4)
    d['sep_bin'] = loop_bin(d['sep'])
    return d


def p_parallel(features, which='paired', prior=None):
    """P(parallel) for one map-adjacent strand pair. `features`: raw dict (see module doc); `which`: model key."""
    m = (prior or layout_priors())['models'][which]
    d = derive_one(features)
    z = m['intercept']
    for t in m['numeric']:
        z += t['coef'] * (float(d[t['name']]) - t['mean']) / t['std']
    for c in m['crosses']:
        z += c['coef'].get('|'.join(str(d[k]) for k in c['cols']), 0.0)
    return 1.0 / (1.0 + math.exp(-max(-700., min(700., z))))


def merged_strand_inputs(blocks, adjacency):
    """Inference view only: fuse consecutive E runs separated by <=1 residue.

    A positive rectangle wins when merging; a known negative wins over an
    unknown rectangle. Pack/manifest block ids and SS are never rewritten.
    Return original-id groups and the feature inputs for the merged view.
    """
    groups = []
    for i, b in enumerate(blocks):
        if (groups and b['type'] == blocks[groups[-1][-1]]['type'] == 'E'
                and b['start'] - blocks[groups[-1][-1]]['stop'] <= 1):
            groups[-1].append(i)
        else:
            groups.append([i])
    types = np.array([0 if blocks[g[0]]['type'] == 'H' else 1 for g in groups])
    start = np.array([blocks[g[0]]['start'] for g in groups])
    stop = np.array([blocks[g[-1]]['stop'] for g in groups])
    adj = np.full((len(groups), len(groups)), 2, dtype=np.int8)
    for a, gi in enumerate(groups):
        for b in range(a+1, len(groups)):
            states = adjacency[np.ix_(gi, groups[b])]
            adj[a,b] = adj[b,a] = 1 if np.any(states == 1) else 0 if np.any(states == 0) else 2
    return groups, types, start, stop, adj


@lru_cache(maxsize=2048)
def _polarity_evidence(boundaries, flat):
    blocks = [dict(type=t, start=a, stop=b) for t,a,b in boundaries]
    adj = np.asarray(flat).reshape(len(blocks), len(blocks))
    groups, types, start, stop, adj = merged_strand_inputs(blocks, adj)
    rows = pair_features(types, start, stop, adj)
    return tuple((groups[r['i']][0], groups[r['j']][0],
                  float(np.clip(p_parallel(r, 'paired'), 1e-6, 1-1e-6)))
                 for r in rows)


def polarity_evidence(layout):
    boundaries = tuple((b['type'], b['start'], b['stop']) for b in layout.blocks)
    return _polarity_evidence(boundaries, tuple(layout.adjacency.ravel()))


def polarity_energy(layout):
    weight = layout.metadata.get('polarity_weight', layout_priors()['polarity_weight'])
    return float(-weight * sum(math.log(p if layout.axes[i] @ layout.axes[j] >= 0 else 1-p)
                               for i,j,p in polarity_evidence(layout)))


def strand_sign_assignments(layout):
    """Exact MAP, one global gauge fixed, <=12 merged inference strands.

    Retain exact ties (not random marginal draws), allowing unobserved
    component gauges to be resolved by geometry. Larger layouts use SA.
    The hard mask is independent of the probability model.
    """
    groups, types, _, _, _ = merged_strand_inputs(layout.blocks, layout.adjacency)
    strand_groups = [g for g,t in zip(groups,types) if t == 1]
    n = len(strand_groups)
    if not n or n > 12:
        return None
    signs = np.ones((2**(n-1), n), dtype=np.int8)
    if n > 1:
        signs[:,1:] = 1 - 2*((np.arange(len(signs))[:,None] >> np.arange(n-1)) & 1)
    ids = {i:k for k,g in enumerate(strand_groups) for i in g}
    valid = np.ones(len(signs), bool)
    for (i,j), sign in polarity_constraints(layout).items():
        valid &= signs[:,ids[i]] * signs[:,ids[j]] == sign
    scores = np.zeros(len(signs))
    for i,j,p in polarity_evidence(layout):
        scores += np.where(signs[:,ids[i]] * signs[:,ids[j]] > 0, np.log(p), np.log1p(-p))
    scores[~valid] = -np.inf
    if not valid.any():
        raise ValueError('inconsistent strand polarity constraints')
    best = np.max(scores)
    return [({i:int(row[k]) for i,k in ids.items()}, float(best))
            for row in signs[np.isclose(scores, best, rtol=0, atol=1e-12)]]


def register_prior(layout, i, j):
    """Population axial offset in residues; absolute offsets have no gauge."""
    lo, hi = sorted((i,j))
    between = ''.join(layout.types[lo+1:hi])
    parallel = layout.axes[i] @ layout.axes[j] >= 0
    table = layout_priors()['geometry']['register']
    if parallel and between == 'H':
        return table['parallel_beta_alpha_beta']
    # Loop-bin offsets were fitted to consecutive blocks only; applying them
    # to a long nonlocal sheet closure would change the population being used.
    if hi != lo+1:
        return None
    key = ('parallel:' if parallel else 'antiparallel:') + loop_bin(
        layout.blocks[hi]['start']-layout.blocks[lo]['stop'])
    return table.get(key)


def order_prior_energy(layout):
    """Soft population spacing, register and BAB handedness likelihoods.

    Second neighbours are defined by each proposed sheet order; none of
    these terms reads native pairing or a target's native order.
    """
    g = layout_priors()['geometry']
    spacing = g['sheet_second_neighbour']['perp']
    scale = (spacing['p75']-spacing['p25'])/1.349
    energy = 0.
    for group, ring in zip(layout.metadata.get('sheets', []), layout.metadata.get('barrels', [])):
        neighbours = {tuple(sorted(p)) for p in list(zip(group,group[1:]))+
                      ([(group[-1],group[0])] if ring and group else [])}
        second = ([(group[k],group[(k+2)%len(group)]) for k in range(len(group))]
                  if ring else list(zip(group,group[2:])))
        for i,j in sorted({tuple(sorted(p)) for p in second}-neighbours):
            delta = layout.centroids[j]-layout.centroids[i]
            u = layout.axes[i]
            distance = np.linalg.norm(delta-(delta@u)*u)
            energy += .5*((distance-spacing['median'])/scale)**2
    for i,j in layout.metadata.get('sheet_pairs', []):
        prior = register_prior(layout,i,j)
        if prior:
            z = abs((layout.centroids[j]-layout.centroids[i])@layout.axes[i])/3.3
            energy += .5*((z-prior['axial_abs_median']/3.3)/max(prior['axial_res_std'],.5))**2
    paired = {tuple(sorted(p)) for p in layout.metadata.get('sheet_pairs', [])}
    for i,h,j,value in crossover_values(layout):
        key = ('paired_parallel' if layout.axes[i]@layout.axes[j]>=0 else 'paired_antiparallel') if (i,j) in paired else 'unpaired'
        p = g['crossover'][key]['frac_right']
        energy -= math.log(p if value < 0 else 1-p)
    return float(energy)


def axis_distance(layout, i, j):
    """Shortest distance between finite block-axis segments (not infinite lines)."""
    a, b = layout.centroids[[i, j]]
    u, v = layout.axes[i] * layout.lengths[i] / 2, layout.axes[j] * layout.lengths[j] / 2
    p, q, d, e = a - u, b - v, 2 * u, 2 * v
    def endpoint(x, origin, direction):
        t = np.clip((x - origin) @ direction / max(direction @ direction, 1e-12), 0., 1.)
        return np.linalg.norm(x - origin - t * direction)
    distances = [endpoint(x, q, e) for x in (p, p + d)]
    distances += [endpoint(x, p, d) for x in (q, q + e)]
    mat = np.array([[d @ d, -d @ e], [-d @ e, e @ e]])
    if np.linalg.det(mat) > 1e-10:
        t, z = np.linalg.solve(mat, [(q - p) @ d, -(q - p) @ e])
        if 0 <= t <= 1 and 0 <= z <= 1:
            distances.append(np.linalg.norm(p + t * d - q - z * e))
    return float(min(distances))


def axis_floor(layout, i, j):
    kind = layout.types[i] + layout.types[j]
    if kind == 'HH':
        return 8.
    if kind in ('HE','EH') and abs(i-j)==1:
        a,b=layout.blocks[min(i,j)],layout.blocks[max(i,j)]
        gap=b['start']-a['stop']
        if gap<=1:
            return 3.5 if gap==0 else 5.5
    if kind == 'EE':
        layers=layout.metadata.get('layer_ids', {})
        # Native frames carry pairing, not a predicted layer assignment.
        if not layers or layers.get(str(i)) == layers.get(str(j)):
            return 4.5
        return 8.
    return 8.


def axis_distances(layout):
    """All exact finite-segment distances, vectorised over block pairs."""
    count=len(layout.blocks)
    if count<2:
        return np.zeros((count,count))
    i,j=np.triu_indices(count,1)
    half=layout.axes*layout.lengths[:,None]/2
    p=layout.centroids[i]-half[i];q=layout.centroids[j]-half[j]
    d=2*half[i];e=2*half[j]
    def endpoint(x,o,v):
        t=np.clip(np.einsum('ij,ij->i',x-o,v)/np.maximum((v*v).sum(1),1e-12),0.,1.)
        return ((x-o-t[:,None]*v)**2).sum(1)
    distance=np.minimum.reduce([endpoint(p,q,e),endpoint(p+d,q,e),endpoint(q,p,d),endpoint(q+e,p,d)])
    aa=(d*d).sum(1);bb=(d*e).sum(1);cc=(e*e).sum(1)
    rr=((q-p)*d).sum(1);ss=-((q-p)*e).sum(1)
    det=aa*cc-bb*bb;safe=np.where(det>1e-10,det,1.)
    t=(cc*rr+bb*ss)/safe;z=(bb*rr+aa*ss)/safe
    valid=(det>1e-10)&(t>=0)&(t<=1)&(z>=0)&(z<=1)
    interior=((p+t[:,None]*d-q-z[:,None]*e)**2).sum(1)
    distance=np.where(valid,np.minimum(distance,interior),distance)
    out=np.zeros((count,count));out[i,j]=out[j,i]=np.sqrt(np.maximum(distance,0.))
    return out


def axis_repulsion(layout):
    distances=axis_distances(layout)
    i,j=np.triu_indices(len(layout.blocks),1)
    kinds=layout.types
    floors=np.full(len(i),8.)
    ee=(kinds[i]=='E')&(kinds[j]=='E')
    layers=layout.metadata.get('layer_ids',{})
    same=np.array([not layers or layers.get(str(a))==layers.get(str(b)) for a,b in zip(i,j)],dtype=bool)
    floors[ee&same]=4.5
    for k,(a,b) in enumerate(zip(i,j)):
        if b==a+1 and kinds[a]!=kinds[b] and layout.blocks[b]['start']-layout.blocks[a]['stop']<=1:
            floors[k]=3.5 if layout.blocks[b]['start']==layout.blocks[a]['stop'] else 5.5
    return float(100.*np.square(np.maximum(floors-distances[i,j],0.)).sum())


def project_sheet_bounds(layout):
    """Project a proposed sheet onto simultaneous axis floors and upper bounds.

    A floor-only projection can push a different ribbon edge above 5 A and
    strand the geometry annealer outside its feasible set. Fixed-frame SLSQP
    makes the smallest centroid correction satisfying both inequalities. This
    is geometric feasibility, independent of maps' signs and native labels.
    """
    from scipy.optimize import minimize
    pairs = layout.metadata.get('sheet_pairs', [])
    if not pairs or layout.metadata.get('native_layout'):
        return layout.copy()
    n = len(layout.blocks)
    i,j = np.triu_indices(n,1)
    floors = np.array([axis_floor(layout,a,b) for a,b in zip(i,j)])
    lookup = {(int(a),int(b)):k for k,(a,b) in enumerate(zip(i,j))}
    sheet = np.array([lookup[tuple(sorted(p))] for p in pairs])
    start = layout.centroids.copy()
    trial = layout.copy()
    def bounds(x):
        trial.centroids = x.reshape(n,3)
        dist = axis_distances(trial)[i,j]
        return np.r_[dist-floors, 5.-dist[sheet]]
    if bounds(start.ravel()).min() >= -1e-7:
        return layout.copy()
    def distance(x):
        delta=x-start.ravel()
        return .5*(delta@delta),delta
    result=minimize(distance,start.ravel(),jac=True,method='SLSQP',
                    constraints={'type':'ineq','fun':bounds},
                    options={'maxiter':80,'ftol':1e-10})
    out=layout.copy()
    if bounds(result.x).min() >= -1e-5:
        out.centroids=result.x.reshape(n,3)
    return out


def project_axis_floors(layout, iterations=100):
    """Deterministic feasibility projection for the short-range axis wall."""
    out=layout.copy()
    for _ in range(iterations):
        worst=0.
        for i in range(len(out.blocks)):
            for j in range(i+1,len(out.blocks)):
                distance=axis_distance(out,i,j); floor=axis_floor(out,i,j)
                if distance >= floor-1e-5:
                    continue
                worst=max(worst,floor-distance)
                direction=np.empty(3)
                for k in range(3):
                    out.centroids[j,k]+=.0001
                    plus=axis_distance(out,i,j)
                    out.centroids[j,k]-=.0002
                    minus=axis_distance(out,i,j)
                    out.centroids[j,k]+=.0001
                    direction[k]=(plus-minus)/.0002
                direction=unit(direction, perpendicular(out.axes[i]))
                shift=.51*(floor-distance+1e-4)*direction
                out.centroids[i]-=shift;out.centroids[j]+=shift
        if worst < 1e-5:
            break
    return sheet_normals(out)


def sse_clashes(layout, ca=None):
    """Residue sterics remain necessary even when the block axes clear."""
    from sketch_builder import placed_sse
    ids=np.concatenate([ec.block_idx(layout.blocks,i) for i in range(len(layout.blocks))]) if layout.blocks else np.array([],int)
    if len(ids)<2:
        return 0
    points=(np.concatenate([placed_sse(layout,i)[:,1] for i in range(len(layout.blocks))])
            if ca is None else ca[ids])
    mask=np.triu(abs(ids[:,None]-ids[None,:])>=2,1)
    return int((ec.pairwise(points)[mask]<2.8).sum())


def shape_metrics(layout, ca, target=None):
    """Harness half_spans definition, using real-CB-else-CA (all sketches GLY).

    Default halves are contiguous spans of the first/last half of H/E blocks,
    including intervening loops; not sequence N//2 and not block centroids.
    """
    from design_metrics import half_spans
    blocks = layout.blocks
    if not target or not target.get('blocks'):
        middle = len(blocks) // 2
        target = dict(blocks=blocks, half1_blocks=list(range(middle)),
                      half2_blocks=list(range(middle, len(blocks))))
    spans = half_spans(target)
    half_sep = (float(np.linalg.norm(ca[slice(*spans[0])].mean(0) - ca[slice(*spans[1])].mean(0)))
                if spans else None)
    rg = float(np.sqrt(np.square(ca - ca.mean(0)).sum(1).mean()))
    return dict(half_sep=half_sep, rg=rg, rg_n04=rg / len(ca) ** .4, n_res=len(ca))


# Fitted on 470 eligible labelstats natives (3..24 blocks, 50..300 residues)
# from the 789 complete native chains; the nine evaluation targets are excluded.
# Rows, fitting protocol and the independent recalibration test accompany N031.
RG_CALIBRATION = dict(n=470, sse_fit=[-0.02192173,2.31428935],
    full_fit=[-0.05187700,2.86929725],sse_residual_upper=1.99851567,
    ratio_fit=[-0.01878836,1.33215096],ratio_residual=[-0.24619017,0.53276587])


def calibrate_layout_rg(rows):
    """Fit native full/SSE Rg/N^.4 vs block count and loop/SSE Rg ratios.

    rows contains (n_blocks,n_res,full_rg_n04,sse_rg_n04,full_rg/sse_rg).
    Central 95% empirical residuals define tolerances, never eval-native Rg.
    """
    data=np.asarray(rows,float)
    use=np.isfinite(data).all(1)&(data[:,0]>=3)&(data[:,0]<=24)&(data[:,1]>=50)&(data[:,1]<=300)
    data=data[use]
    if len(data)<50:raise ValueError('at least 50 eligible native chains required')
    fits=[np.polyfit(data[:,0],data[:,col],1) for col in (2,3,4)]
    residual=[data[:,col]-np.polyval(fit,data[:,0]) for col,fit in zip((2,3,4),fits)]
    return dict(n=len(data),full_fit=fits[0].tolist(),sse_fit=fits[1].tolist(),
                sse_residual_upper=float(np.quantile(residual[1],.975)),
                ratio_fit=fits[2].tolist(),ratio_residual=np.quantile(residual[2],[.025,.975]).tolist())


def layout_rg_limits(n_blocks,calibration=None):
    fit=RG_CALIBRATION if calibration is None else calibration
    count=float(np.clip(n_blocks,3,24))
    sse=float(np.polyval(fit['sse_fit'],count))
    ratio=float(np.polyval(fit['ratio_fit'],count))
    return dict(sse_rg_n04_limit=sse+fit['sse_residual_upper'],
                rg_layout_ratio_min=ratio+fit['ratio_residual'][0],
                rg_layout_ratio_max=ratio+fit['ratio_residual'][1],
                rg_calibration_n=fit['n'])


@lru_cache(maxsize=256)
def _objective_sse_template(kind,n,twist):
    """Canonical SSE transform shared by repeated SA objective calls."""
    from sketch_builder import ideal_sse, _roll_normal
    xyz=ideal_sse(kind,n,twist);ca=xyz[:,1];axis=principal_axis(ca)
    normal=_roll_normal(ca,axis,kind)
    source=np.column_stack((axis,normal,np.cross(axis,normal)))
    canonical=(xyz-ca.mean(0))@source
    canonical.setflags(write=False)
    return canonical


def objective_sse(layout,i):
    block=layout.blocks[i]
    canonical=_objective_sse_template(block['type'],block['stop']-block['start'],layout.metadata.get('strand_twist',0.))
    u,n=layout.axes[i],layout.normals[i]
    return canonical@np.column_stack((u,n,np.cross(u,n))).T+layout.centroids[i]


def objective(layout, backbone=None, return_terms=False):
    """B0(d) energy. Distances use reconstructed CB of ideal, placed SSEs.

    The default has SSE-only sterics and endpoint feasibility. When a full
    backbone is supplied, loop sterics and chain bond errors are included too.
    Weights fix units explicitly; state-2 pairs never enter the map term.
    """
    from sketch_builder import placed_sse
    blocks = layout.blocks
    if backbone is None:
        xyz, ids = [], []
        for i, b in enumerate(blocks):
            xyz.append(objective_sse(layout, i)); ids.extend(range(b['start'], b['stop']))
        xyz = np.concatenate(xyz) if xyz else np.empty((0, 4, 3))
        ids = np.asarray(ids, int)
    else:
        xyz = np.asarray(backbone, float)
        ids = np.arange(len(xyz))
    ca = xyz[:, 1]
    cb = ec.reconstruct_cb(xyz[:, 0], ca, xyz[:, 2])
    chunks = [np.flatnonzero((ids >= b['start']) & (ids < b['stop'])) for b in blocks]
    terms = dict(map=0., clash=0., loop=0., compactness=0., handedness=0., bonds=0., axes=axis_repulsion(layout), koga=0.)
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            state = layout.adjacency[i, j]
            if state == 2:
                continue
            d = float(cdist(cb[chunks[i]], cb[chunks[j]]).min())
            terms['map'] += float(10.*max(d - 6. if state == 1 else 7. - d, 0.) ** 2)
    if len(ca):
        d = squareform(pdist(ca))
        eligible = np.triu(abs(ids[:, None] - ids[None, :]) >= 2, 1)
        terms['clash'] = float(5 * np.square(np.maximum(4. - d[eligible], 0)).sum())
        sse_ca = np.concatenate([ca[c] for c in chunks]) if chunks else ca
        rg = np.sqrt(np.square(sse_ca - sse_ca.mean(0)).sum(1).mean())
        terms['compactness'] = float(10 * max(rg / len(layout.ss) ** .4 - 2., 0.) ** 2)
    for i in range(len(blocks) - 1):
        gap = blocks[i + 1]['start'] - blocks[i]['stop']
        dist = np.linalg.norm(ca[chunks[i][-1]] - ca[chunks[i + 1][0]])
        reach = (7.0 if gap == 1 else 3.65*(gap+1)) if layout.metadata.get('native_layout') else (6.0 if gap==1 else 3.1*(gap+1))
        terms['loop'] += float(5000 * max(dist-reach,0.)**2)
        if not layout.metadata.get('native_layout') and gap>0:
            tangent_out=unit(ca[chunks[i][-1]]-ca[chunks[i][-2]])
            tangent_in=unit(ca[chunks[i+1][1]]-ca[chunks[i+1][0]])
            cap_delta=ca[chunks[i+1][0]]-ca[chunks[i][-1]]-1.9*(tangent_out+tangent_in)
            cap_reach=3.8 if gap==1 else 3.1*gap
            terms['loop']+=500.*max(np.linalg.norm(cap_delta)-cap_reach,0.)**2
        if (not layout.metadata.get('native_layout') and 0<gap<=8 and
                blocks[i]['type']==blocks[i+1]['type']=='E' and layout.axes[i]@layout.axes[i+1]<-.5):
            # A short antiparallel turn joins the same sheet cap. Register
            # alignment gives its exterior route slack without squeezing CA.
            longitudinal=(ca[chunks[i+1][0]]-ca[chunks[i][-1]])@layout.axes[i]
            terms['loop']+=25.*longitudinal**2
        if gap == 0:
            terms['loop'] += float(10000*(dist-3.8)**2)
    atom_ids=np.repeat(ids,4);heavy=xyz.reshape(-1,3)
    rows,cols=np.triu_indices(len(heavy),1)
    use=atom_ids[cols]-atom_ids[rows]>=2
    distances=pdist(heavy)[use]
    terms['heavy']=float(2000.*np.square(np.maximum(2.75-distances,0.)).sum())
    terms['orientation_prior'] = 0. if layout.metadata.get('native_layout') else polarity_energy(layout)
    terms['order_prior'] = 0. if layout.metadata.get('native_layout') else order_prior_energy(layout)
    terms['handedness'] = float(sum(1000 * max(v + .1, 0.) ** 2 / (1 + abs(v))
                                      for _, _, _, v in crossover_values(layout)))
    # Neighbouring EHE units separated by a short EE turn can share a
    # helix face. The unsigned H-H noncontact overrides this weak packing
    # prior. Right-handedness then discriminates split sheet crossovers.
    terms['crossover_face']=0.
    if not layout.metadata.get('native_layout'):
        crosses=crossover_values(layout)
        for (a,h,b,_),(c,g,d,_) in zip(crosses,crosses[1:]):
            if c==b+1 and layout.blocks[c]['start']-layout.blocks[b]['stop']<=8 and layout.adjacency[h,g]!=0:
                normal=unit(layout.normals[a])
                side_h=(layout.centroids[h]-(layout.centroids[a]+layout.centroids[b])/2)@normal
                side_g=(layout.centroids[g]-(layout.centroids[c]+layout.centroids[d])/2)@normal
                terms['crossover_face']+=10.*max(-side_h*side_g,0.)
                # Compare the faces implied by right-handed strand-to-strand
                # crossovers, independently of helix tip/register motion.
                # A supplied H-H contact favours the same sheet face.
                cross_a=unit(np.cross(layout.axes[a],layout.centroids[b]-layout.centroids[a]))
                cross_c=unit(np.cross(layout.axes[c],layout.centroids[d]-layout.centroids[c]))
                face_weight=100. if layout.adjacency[h,g]==1 else 30.
                terms['crossover_face']+=face_weight*max(.5-cross_a@cross_c,0.)**2

                for group in layout.metadata.get('sheets',[]):
                    if b in group and c in group and layout.adjacency[b,c]!=0:
                        terms['crossover_face']+=50.*max(abs(group.index(b)-group.index(c))-1,0.)**2
    layers=layout.metadata.get('sheets',[])
    terms['tight_turn_order']=0.
    if not layout.metadata.get('native_layout'):
        for group in layers:
            for i,j in zip(range(len(blocks)-1),range(1,len(blocks))):
                if blocks[i]['type']==blocks[j]['type']=='E' and layout.adjacency[i,j]==1 and 0<blocks[j]['start']-blocks[i]['stop']<=4 and i in group and j in group:
                    terms['tight_turn_order']+=50.*max(abs(group.index(i)-group.index(j))-1,0.)**2
    if len(layers)==2 and not any(layout.metadata.get('barrels',[])):
        group_a,group_b=layers
        # Plane spacing is a layout variable, independent of sequence halves.
        normal=unit(layout.normals[group_a[0]])
        separation=abs((layout.centroids[group_b].mean(0)-layout.centroids[group_a].mean(0))@normal)
        terms['layer_spacing']=2.*(separation-9.5)**2
        # This counts strands per ribbon; it never compares sequence halves.
        if not layout.metadata.get('native_layout'):
            terms['sheet_width_prior']=3.*(len(group_a)-len(group_b))**2
    terms['sheet_spacing'] = float(sum(100*max(axis_distance(layout,i,j)-5.,0.)**2
                                             for i,j in layout.metadata.get('sheet_pairs',[])))
    terms['sheet_pairing']=0.
    for i,j in layout.metadata.get('sheet_pairs',[]):
        distances=cdist(ca[chunks[i]],ca[chunks[j]]).ravel()
        closest=np.partition(distances,1)[:2]
        terms['sheet_pairing']+=1000.*np.square(np.maximum(closest-5.25,0.)).sum()

    terms['koga'] = (0. if layout.metadata.get('native_layout') or not layout.metadata.get('koga_calibration') else float(sum(
        2. * max(.1 - (1 if r['expected'] in ('P', 'R') else -1) * r['calibrated_value'], 0.) ** 2
        for r in koga_values(layout) if r['enabled'])))
    if backbone is not None and len(ca) > 1:
        error = np.maximum(abs(np.linalg.norm(np.diff(ca, axis=0), axis=1) - 3.8) - .3, 0)
        terms['bonds'] = float(10 * (error ** 2).sum())
    return terms if return_terms else float(sum(terms.values()))


def calibrate_koga_rules(layouts, backbones=None, min_count=50):
    """Measure each P-SEA gap/type cue; insufficient or <90% rules stay OFF.

    backbones, when available, are native N/CA/C/O, never generated sketches.
    A calibration is opt-in metadata, not a rule learned from eval target names.
    """
    from sketch_builder import placed_sse
    counts = {}
    for k, layout in enumerate(layouts):
        for rule in koga_rules(layout):
            i, j, kind = rule['i'], rule['j'], rule['kind']
            strand = j if kind == 'HE' else i
            b = layout.blocks[strand]
            xyz = (placed_sse(layout, strand)[0 if kind == 'HE' else -1] if backbones is None
                   else backbones[k][b['start'] if kind == 'HE' else b['stop']-1])
            cue = unit(ec.reconstruct_cb(xyz[None,0],xyz[None,1],xyz[None,2])[0]-xyz[1])
            delta = layout.centroids[j]-layout.centroids[i]
            direction = np.cross(layout.axes[i],delta) if kind == 'EE' else delta
            value = cue @ unit(direction, perpendicular(layout.axes[strand]))
            expected = -1 if rule['expected'] in ('L','A') else 1
            key = f"{kind}:{rule['gap']}"
            counts.setdefault(key, []).append(expected*value)
    result = {}
    for key, values in counts.items():
        values = np.asarray(values)
        plus, minus = float(np.mean(values > .02)), float(np.mean(values < -.02))
        accuracy, sign = (plus, 1) if plus >= minus else (minus, -1)
        result[key] = dict(n=len(values), sign=sign, accuracy=accuracy,
                           enabled=len(values)>=min_count and accuracy>=.9)
    return result


def sheet_normals(layout):
    """Derive strand pleats from the local sheet, retaining their discrete phase.

    Native frames already contain a measured roll; a normal spring protects it.
    Predicted sheets use their current neighbour directions after every move.
    """
    if layout.metadata.get('native_layout'):
        return layout
    for i in np.flatnonzero(layout.types == 'E'):
        neighbours = [b if a == i else a for a,b in layout.metadata.get('sheet_pairs', [])
                      if i in (a,b)]
        if not neighbours:
            group=next((g for g in layout.metadata.get('sheets',[]) if i in g and len(g)>1),[])
            if not group:
                continue
            neighbours=[min((j for j in group if j!=i),
                            key=lambda j: np.linalg.norm(layout.centroids[j]-layout.centroids[i]))]
        normal = np.zeros(3)
        for j in neighbours:
            candidate = unit(np.cross(layout.axes[i], layout.centroids[j]-layout.centroids[i]),
                             layout.normals[i])
            normal += candidate * (1 if candidate@layout.normals[i]>=0 else -1)
        layout.normals[i] = unit(normal, layout.normals[i])
    return layout
