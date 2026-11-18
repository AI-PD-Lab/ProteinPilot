#!/usr/bin/env python
"""Conditional R6 block flow. Centered SO(3) augmentation, never reflections.

The transformer is not equivariant by construction; invariant losses are.
Checkpoint format contains tensors and primitive config only (weights_only=True).
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import itertools
import io
import json
import math
import os
import signal
import hashlib
from pathlib import Path
import time
import tarfile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import eval_common as ec
from layout_common import (Layout, block_adjacency, split_signed_adjacency, handedness,
                           perpendicular, refresh_strand_pairs, constraints_ok)
from layout_dataset import read_dataset, DEFAULT_OUT


def conditioning(layout, rng=None, dropout=False):
    """50% full, 25% sparse-k, 25% dense0; independent p=.5 signed pairs."""
    rng = np.random.default_rng() if rng is None else rng
    adj = layout.adjacency.copy()
    if dropout:
        mode = rng.choice(3, p=[.5, .25, .25])
        if mode:
            pairs = np.argwhere(np.triu(adj == 1, 1))
            count = min(8, math.ceil(len(pairs)/2))
            keep = pairs[rng.choice(len(pairs), count, replace=False)] if count else []
            sparse = np.full_like(adj, 2)
            if mode == 2:
                sparse[adj == 0] = 0
            for i, j in keep:
                sparse[i, j] = sparse[j, i] = 1
            adj = sparse
    signs = layout.orientation.copy()
    if dropout:
        reveal = np.triu(rng.random(signs.shape) < .5, 1)
        signs *= reveal | reveal.T
    signs[adj != 1] = 0
    np.fill_diagonal(adj, 2)
    return adj, signs


@lru_cache(maxsize=512)
def _local_cb(kind, n):
    from sketch_builder import ideal_sse, _roll_normal
    from layout_common import principal_axis
    xyz = ideal_sse(kind, n)
    ca = xyz[:, 1]
    u = principal_axis(ca); normal = _roll_normal(ca, u, kind)
    cb = ec.reconstruct_cb(xyz[:, 0], ca, xyz[:, 2])
    return ((cb-ca.mean(0)) @ np.column_stack((u, normal, np.cross(u, normal)))).astype(np.float32)


def collate(layouts, *, rng=None, dropout=False, augment=False, device='cpu'):
    rng = np.random.default_rng() if rng is None else rng
    n, b = len(layouts), max(len(x.blocks) for x in layouts)
    r = max(int(x.n_res.sum()) for x in layouts)
    batch = dict(x1=np.zeros((n, b, 6), np.float32), token=np.zeros((n, b, 6), np.float32),
                 mask=np.zeros((n, b), bool), adj=np.full((n, b, b), 2, np.int64),
                 signs=np.zeros((n, b, b), np.int64), sep=np.zeros((n, b, b), np.int64),
                 between=np.zeros((n, b, b), np.float32), local_cb=np.zeros((n, r, 3), np.float32),
                 atom_block=np.zeros((n, r), np.int64), atom_mask=np.zeros((n, r), bool))
    for row, lay in enumerate(layouts):
        nb = len(lay.blocks)
        if nb == 0:
            raise ValueError('flow requires at least one block')
        centers = lay.centroids-lay.centroids.mean(0)
        axes = lay.axes.copy()
        if augment:
            # QR plus determinant correction gives proper SO(3), never O(3).
            q, upper = np.linalg.qr(rng.normal(size=(3, 3)))
            q *= np.sign(np.diag(upper))[None, :]
            q[:, -1] *= np.linalg.det(q)
            centers, axes = centers @ q, axes @ q
        batch['x1'][row, :nb] = np.concatenate((centers/10, axes), axis=1)
        batch['mask'][row, :nb] = True
        adj, signs = conditioning(lay, rng, dropout)
        batch['adj'][row, :nb, :nb], batch['signs'][row, :nb, :nb] = adj, signs
        blocks = lay.blocks
        starts = np.array([v['start'] for v in blocks]); stops = np.array([v['stop'] for v in blocks])
        before = starts-np.r_[0, stops[:-1]]; after = np.r_[starts[1:], len(lay.ss)]-stops
        batch['token'][row, :nb] = np.column_stack((lay.types == 'H', lay.types == 'E', lay.n_res/30.,
                                                   (starts+stops)/2/len(lay.ss), np.minimum(before, 30)/30.,
                                                   np.minimum(after, 30)/30.))
        sep = abs(np.arange(nb)[:, None]-np.arange(nb)[None, :])
        batch['sep'][row, :nb, :nb] = np.minimum(np.floor(np.log2(sep+1)).astype(int), 5)
        between = np.maximum(np.maximum(starts[:, None]-stops[None, :], starts[None, :]-stops[:, None]), 0)
        batch['between'][row, :nb, :nb] = np.minimum(between, 300)/100.
        offset = 0
        for i, (kind, count) in enumerate(zip(lay.types, lay.n_res)):
            batch['local_cb'][row, offset:offset+count] = _local_cb(str(kind), int(count))
            batch['atom_block'][row, offset:offset+count] = i
            batch['atom_mask'][row, offset:offset+count] = True
            offset += count
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


class GraphLayer(nn.Module):
    def __init__(self, d, heads, pair_dim):
        super().__init__()
        self.heads, self.width = heads, d//heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, d*3)
        self.bias = nn.Linear(pair_dim, heads)
        self.output = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.SiLU(), nn.Linear(4*d, d))

    def forward(self, h, pair, mask):
        n, b, d = h.shape
        q, k, v = self.qkv(self.norm1(h)).reshape(n, b, 3, self.heads, self.width).permute(2, 0, 3, 1, 4)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.width)
        logits = logits+self.bias(pair).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~mask[:, None, None, :], -torch.inf)
        attention = logits.softmax(-1)
        h = h+self.output((attention @ v).transpose(1, 2).reshape(n, b, d))
        return (h+self.ff(self.norm2(h))) * mask[..., None]


class LayoutFlow(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, pair_dim=32):
        super().__init__()
        if d % heads or min(d, layers, heads, pair_dim) < 1:
            raise ValueError('invalid model dimensions')
        self.config = dict(d=d, layers=layers, heads=heads, pair_dim=pair_dim)
        self.input = nn.Linear(12, d)
        self.time = nn.Sequential(nn.Linear(17, d), nn.SiLU(), nn.Linear(d, d))
        self.state, self.sign, self.separation = nn.Embedding(3, pair_dim), nn.Embedding(3, pair_dim), nn.Embedding(6, pair_dim)
        self.between = nn.Linear(1, pair_dim)
        self.pair = nn.Sequential(nn.SiLU(), nn.Linear(pair_dim, pair_dim))
        self.layers = nn.ModuleList(GraphLayer(d, heads, pair_dim) for _ in range(layers))
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 6))
        if sum(p.numel() for p in self.parameters()) > 5_000_000:
            raise ValueError('model exceeds 5M parameter cap')

    def forward(self, x, t, batch):
        frequencies = torch.arange(8, dtype=x.dtype, device=x.device)*math.pi
        phase = t[:, None]*frequencies
        te = torch.cat((t[:, None], phase.sin(), phase.cos()), -1)
        h = self.input(torch.cat((x, batch['token']), -1))+self.time(te)[:, None]
        pair = self.pair(self.state(batch['adj'])+self.sign(batch['signs']+1)+
                         self.separation(batch['sep'])+self.between(batch['between'][..., None]))
        for layer in self.layers:
            h = layer(h, pair, batch['mask'])
        return self.output(h)*batch['mask'][..., None]


def _average(values, mask):
    return (values*mask).sum()/mask.sum().clamp_min(1)


def invariant_pair_loss(predicted, native, mask):
    """Squared centroid-distance (10 A units) and unit-axis dot errors."""
    valid = mask[:, :, None] & mask[:, None, :]
    valid &= torch.triu(torch.ones_like(valid), diagonal=1)
    pd, nd = torch.cdist(predicted[..., :3], predicted[..., :3]), torch.cdist(native[..., :3], native[..., :3])
    pu, nu = F.normalize(predicted[..., 3:], dim=-1), F.normalize(native[..., 3:], dim=-1)
    return _average((pd-nd).square()+(pu @ pu.transpose(1, 2)-nu @ nu.transpose(1, 2)).square(), valid)


def frame_normals(x, mask):
    """Roll prior from weighted transverse neighbour directions, not native roll.

    SO(3)-covariant except at geometrically degenerate collinear frames, where
    roll is undefined and a coordinate-basis fallback is necessary.
    """
    c, u = x[..., :3]*10., F.normalize(x[..., 3:], dim=-1, eps=1e-6)
    delta = c[:, None]-c[:, :, None]  # [N,i,j,3] = c_j-c_i
    residual = delta-(delta*u[:, :, None]).sum(-1, keepdim=True)*u[:, :, None]
    norm = residual.norm(dim=-1)
    valid = (norm >= 1e-5) & mask[:, None]
    weight = torch.exp(-delta.norm(dim=-1)/10.)*valid
    normal = (residual*weight[..., None]).sum(2)
    basis = F.one_hot(u.abs().argmin(-1), 3).to(x.dtype)
    fallback = torch.linalg.cross(u, basis)
    normal = torch.where((normal.norm(dim=-1) < 1e-5)[..., None], fallback, normal)
    return F.normalize(normal, dim=-1, eps=1e-6)


def map_hinge(x, batch, *, return_distances=False, normalized=False):
    """Differentiable exact MAP term of layout_common.objective E.

    Same ideal-SSE reconstructed CBs, minima, cutoffs 6/7 A, weight 10,
    squared one-sided hinges and unknown masks. Default: raw B0 E units.
    normalized=True: mean squared violation per known pair in (10 A)^2 units.
    """
    u = F.normalize(x[..., 3:], dim=-1, eps=1e-6)
    normal = frame_normals(x, batch['mask'])
    frames = torch.stack((u, normal, torch.linalg.cross(u, normal)), -1)
    ids = batch['atom_block']
    frame = frames.gather(1, ids[..., None, None].expand(-1, -1, 3, 3))
    centers = (x[..., :3]*10).gather(1, ids[..., None].expand(-1, -1, 3))
    cb = (frame @ batch['local_cb'][..., None]).squeeze(-1)+centers
    d = torch.cdist(cb, cb)
    valid = batch['atom_mask'][:, :, None] & batch['atom_mask'][:, None, :]
    d = d.masked_fill(~valid, torch.inf)
    n, b = batch['mask'].shape
    pair_ids = ids[:, :, None]*b+ids[:, None, :]
    dmin = torch.full((n, b*b), torch.inf, device=x.device, dtype=x.dtype)
    dmin = dmin.scatter_reduce(1, pair_ids.flatten(1), d.flatten(1), reduce='amin').reshape(n, b, b)
    adj = batch['adj']
    known = (adj != 2) & batch['mask'][:, :, None] & batch['mask'][:, None, :]
    known &= torch.triu(torch.ones_like(known), diagonal=1)
    # Replace infinities BEFORE squaring: masked inf*0 still gives NaN gradients.
    dmin = torch.where(known, dmin, torch.zeros_like(dmin))
    penalty = torch.where(adj == 1, F.relu(dmin-6), F.relu(7-dmin)).square()*10.
    per_layout = (penalty*known).sum((1, 2))
    if normalized:
        # E_B0 = 10 * sum(A^2). c/10 velocities use (10 A)^2 units.
        # Normalize EACH layout before averaging (padding/dropout change counts).
        per_layout = per_layout / (1000. * known.sum((1, 2)).clamp_min(1))
    loss = per_layout.mean()
    return (loss, dmin) if return_distances else loss


def center_state(x, mask):
    """Centroid translations are absent from both endpoints and vector fields."""
    weight = mask[..., None]
    centers = x[..., :3] - (x[..., :3]*weight).sum(1, keepdim=True)/weight.sum(1, keepdim=True).clamp_min(1)
    return torch.cat((centers, x[..., 3:]), -1)*weight


def flow_loss(model, batch, *, generator=None):
    x1, mask = batch['x1'], batch['mask']
    x0 = center_state(torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator), mask)
    t = torch.rand(len(x1), device=x1.device, generator=generator)
    xt = (1-t[:, None, None])*x0+t[:, None, None]*x1
    velocity = center_state(model(xt, t, batch), mask)
    predicted = xt+(1-t[:, None, None])*velocity
    error = (velocity-(x1-x0)).square()
    centroid = _average(error[..., :3].mean(-1), mask)
    axis = _average(error[..., 3:].mean(-1), mask)
    pair = invariant_pair_loss(predicted, x1, mask)
    hinge = map_hinge(predicted, batch, normalized=True)
    return centroid+axis+.5*pair+.2*hinge, dict(centroid=centroid, axis=axis, pair=pair, map=hinge)


def save_checkpoint(path, model, *, optimizer=None, step=0, training=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(version=1, config=model.config, state_dict=model.state_dict(), step=int(step))
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    if training is not None:
        payload['training'] = training
    tmp = path.with_suffix('.tmp.pt'); torch.save(payload, tmp); tmp.replace(path)


def load_checkpoint(path, device='cpu'):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get('version') != 1:
        raise ValueError('unsupported flow checkpoint')
    model = LayoutFlow(**payload['config']).to(device)
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model


@torch.no_grad()
def integrate(model, batch, *, steps=40, temperature=.85, seed=0):
    if steps < 1 or not .7 <= temperature <= 1.:
        raise ValueError('steps >= 1 and temperature in [0.7,1.0] required')
    model.eval()
    generator = torch.Generator(device=batch['x1'].device).manual_seed(int(seed))
    x = torch.randn(batch['x1'].shape, device=batch['x1'].device, generator=generator)*temperature
    mask = batch['mask'][..., None]
    x = center_state(x, batch['mask'])
    for k in range(steps):
        t = torch.full((len(x),), k/steps, device=x.device)
        x = (x+center_state(model(x, t, batch), batch['mask'])/steps)*mask
    u = x[..., 3:]
    fallback = torch.zeros_like(u); fallback[..., 2] = 1
    x[..., 3:] = torch.where((u.norm(dim=-1) > 1e-8)[..., None], F.normalize(u, dim=-1), fallback)*mask
    return x


@torch.enable_grad()
def project_map(x, batch, *, steps=80):
    """Softly anchored centroid repair using the same known-pair hinge.

    Axes stay fixed. This is deterministic optimization inside one draw; no
    native coordinates, extra random samples or acceptance/rejection are used.
    Unknown pairs exert no force. The spring limits movement away from the flow.
    """
    if steps < 0:
        raise ValueError('projection steps must be nonnegative')
    if not steps or not torch.triu(batch['adj'] != 2, 1).any():
        return x.detach()
    initial = x.detach()
    centers = initial[..., :3].clone().requires_grad_()
    optimizer = torch.optim.Adam([centers], lr=.03)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        state = torch.cat((centers, initial[..., 3:]), -1)
        hinge = map_hinge(state, batch, normalized=True)
        if float(hinge.detach()) <= 1e-8:
            break
        spring = _average((centers-initial[..., :3]).square().mean(-1), batch['mask'])
        loss = hinge + .02*spring
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite map projection')
        loss.backward(); optimizer.step()
        with torch.no_grad():
            weight = batch['mask'][..., None]
            centers -= (centers*weight).sum(1, keepdim=True)/weight.sum(1, keepdim=True).clamp_min(1)
            centers *= weight
    return torch.cat((centers.detach(), initial[..., 3:]), -1)


def project_loops(layout, iterations=160, tolerance=1e-4):
    """Small rigid centroid moves to satisfy the B0 ideal-axis endpoint bound."""
    out = layout.copy()
    for _ in range(iterations):
        worst = 0.
        for i, (a, b) in enumerate(zip(out.blocks, out.blocks[1:])):
            endpoint = out.centroids[i]+out.axes[i]*out.lengths[i]/2
            start = out.centroids[i+1]-out.axes[i+1]*out.lengths[i+1]/2
            delta = start-endpoint; distance = np.linalg.norm(delta)
            limit = 3.4*(b['start']-a['stop']+1)
            excess = distance-limit
            if excess > tolerance:
                shift = .5*excess*delta/max(distance, 1e-8)
                out.centroids[i] += shift; out.centroids[i+1] -= shift
                worst = max(worst, excess)
        if worst <= tolerance:
            break
    return out.centered()


def loop_excess(layout):
    excess = 0.
    for i, (a, b) in enumerate(zip(layout.blocks, layout.blocks[1:])):
        end = layout.centroids[i]+layout.axes[i]*layout.lengths[i]/2
        start = layout.centroids[i+1]-layout.axes[i+1]*layout.lengths[i+1]/2
        excess = max(excess, np.linalg.norm(end-start)-3.4*(b['start']-a['stop']+1))
    return float(excess)


def apply_rules(layout):
    from layout_planner_sa import enforce_crossovers
    import layout_common as common
    from layout_common import polarity_constraints, unit
    out = layout.copy()
    # Resolve polarity graph without the SA initializer's global-z alignment:
    # project directions onto one signed component mean; preserves global SO(3).
    graph = {i: [] for i in range(len(out.blocks))}
    try:
        polarities = polarity_constraints(out)
    except ValueError as error:
        # User signs outrank a statistical loop prior. Conflicts are diagnostics,
        # never a reason to discard a sampled draw.
        out.metadata['polarity_conflict'] = str(error)
        unsigned = out.copy(); unsigned.metadata.pop('signed_pairs', None)
        polarities = polarity_constraints(unsigned)
        polarities.update({(i, j): sign for i, j, sign in out.metadata.get('signed_pairs', [])})
    def rules_ok(value):
        try:
            return bool(constraints_ok(value))
        except ValueError:
            return False
    for (i, j), sign in polarities.items():
        graph[i].append((j, sign)); graph[j].append((i, sign))
    visited = set()
    for start in graph:
        if start in visited:
            continue
        signs, queue = {start: 1}, [start]
        while queue:
            i = queue.pop()
            for j, sign in graph[i]:
                desired = signs[i]*sign
                if j in signs and signs[j] != desired:
                    out.metadata['polarity_conflict'] = True
                    continue
                if j not in signs:
                    signs[j] = desired; queue.append(j)
        visited.update(signs)
        if all(sign*(out.axes[i] @ out.axes[j]) >= .5
               for i in signs for j, sign in graph[i]):
            continue  # Retain learned twist and roll when the rule already holds.
        direction = unit(sum(s*out.axes[i] for i, s in signs.items()), out.axes[start])
        for i, s in signs.items():
            out.axes[i] = s*direction
            normal = out.normals[i] - (out.normals[i] @ out.axes[i])*out.axes[i]
            out.normals[i] = unit(normal, perpendicular(out.axes[i]))
    # Global mirror is deterministic and internal to this draw. Mixed-handed
    # units then use part 2's local helix-plane reflection, no best-of selection.
    before = handedness(out)
    if before['left'] > before['right']:
        out.centroids[:, 0] *= -1; out.axes[:, 0] *= -1; out.normals[:, 0] *= -1
    for _ in range(8):
        enforce_crossovers(out)
        out = project_loops(out)
        if rules_ok(out):
            break
    # Query the shared calibrated table at CALL time. Old tables omit the flag:
    # those rules are uncalibrated by default. Never roll or reject on a cue.
    table = {(r['i'], r['j'], r['kind'], r['gap']): r for r in common.koga_rules(out)}
    rules = common.koga_values(out)
    def calibrated(rule):
        entry = table.get((rule.get('i'), rule.get('j'), rule.get('kind'), rule.get('gap')), {})
        flags = [record[key] for record in (entry, rule) for key in ('calibrated', 'enabled') if key in record]
        if rule.get('uncalibrated', False) or entry.get('uncalibrated', False):
            return False
        return bool(flags) and all(flags)
    active = [r for r in rules if calibrated(r)]
    out.metadata['rule_diagnostics'] = dict(handedness=handedness(out), constraints_ok=rules_ok(out),
                                           koga=active, koga_skipped=len(rules)-len(active),
                                           koga_satisfied=all(v['satisfied'] for v in active),
                                           loop_excess=loop_excess(out))
    return out


def sample_layouts(model, requests, *, seed=0, steps=40, temperature=.85, rules=True, return_exceptions=False):
    """One padded model integration per chunk; geometric repair remains per draw.

    Requests are (SS, adjacency, optional signs). Unsigned requests expose no
    native orientation. A chunk has a deterministic RNG stream for its seed.
    """
    conditions = []
    for ss, adjacency, signs in requests:
        ss = np.asarray(ss)
        nb = len(ec.he_blocks_from_ss(ss))
        if not nb or nb > 24 or len(ss) > 300:
            raise ValueError('flow supports 1-24 blocks, at most 300 residues')
        cond = Layout(ss, np.zeros((nb, 3)), np.tile([0., 0., 1.], (nb, 1)), adjacency=adjacency)
        if signs is not None:
            signs = np.asarray(signs)
            if signs.shape != (nb, nb) or not np.isin(signs, [-1, 0, 1]).all() or not np.array_equal(signs, signs.T):
                raise ValueError('signs must be symmetric [B,B] -1/0/+1')
            allowed = (cond.types[:, None] == 'E') & (cond.types[None, :] == 'E') & (cond.adjacency == 1)
            if np.any((signs != 0) & ~allowed):
                raise ValueError('signs only allowed on strand contacts')
            cond.orientation = signs.copy()
            cond.metadata['signed_pairs'] = [[int(i), int(j), int(signs[i, j])] for i, j in zip(*np.where(np.triu(signs != 0, 1)))]
        conditions.append(cond)
    if not conditions:
        return []
    batch = collate(conditions, device=next(model.parameters()).device)
    x = integrate(model, batch, steps=steps, temperature=temperature, seed=seed)
    if rules:
        try:
            x = project_map(x, batch)
        except (ValueError, FloatingPointError) as error:
            if not return_exceptions:
                raise
            return [error for _ in conditions]
    values = x.cpu().numpy()
    normals = frame_normals(x, batch['mask']).cpu().numpy()
    from sketch_builder import assemble
    outputs = []
    for row, cond in enumerate(conditions):
        nb = len(cond.blocks); value = values[row, :nb]
        try:
            out = Layout(cond.ss, value[:, :3]*10., value[:, 3:], adjacency=cond.adjacency,
                         metadata=dict(cond.metadata, planner='flow', seed=int(seed), batch_row=row,
                                       steps=int(steps), temperature=float(temperature), map_projection_steps=80 if rules else 0))
            out.normals = normals[row, :nb]
            out = apply_rules(out) if rules else out
            outputs.append(refresh_strand_pairs(out, assemble(out)[:, 1]))
        except (ValueError, FloatingPointError) as error:
            if not return_exceptions:
                raise
            outputs.append(error)
    return outputs


def sample_layout(model, ss, adjacency, *, signs=None, seed=0, steps=40, temperature=.85, rules=True):
    return sample_layouts(model, [(ss, adjacency, signs)], seed=seed, steps=steps,
                          temperature=temperature, rules=rules)[0]


def kabsch_rmsd(a, b):
    a, b = np.asarray(a)-np.mean(a, 0), np.asarray(b)-np.mean(b, 0)
    u, _, vt = np.linalg.svd(a.T @ b)
    fix = np.eye(3); fix[-1, -1] = np.linalg.det(u @ vt)
    return float(np.sqrt(np.mean(np.sum((a @ u @ fix @ vt-b)**2, axis=-1))))


def native_pack_layouts(path):
    """Optional nine-target validation source: local spec packs, no mirror."""
    from layout_common import extract_native_layout
    with tarfile.open(path) as tf:
        manifest = json.load(tf.extractfile('manifest.json'))
        with np.load(io.BytesIO(tf.extractfile('refs.npz').read()), allow_pickle=False) as refs:
            for name, target in manifest['targets'].items():
                if name+'_ca' not in refs.files:
                    continue
                ss = np.asarray(target['ss'])
                b = len(ec.he_blocks_from_ss(ss))
                adj = np.full((b, b), 2, np.int8)
                for key, value in [('pos_pairs', 1), ('neg_pairs', 0)]:
                    for i, j in target[key]:
                        adj[i, j] = adj[j, i] = value
                lay = extract_native_layout(refs[name+'_ca'], ss, adj)
                lay.metadata.update(split='val', target=name)
                yield lay


def validate(model, layouts, *, draws=8, seed=0, sweeps=0, repo=None, batch_size=64):
    from sketch_builder import build_sketch
    if min(draws, batch_size) < 1:
        raise ValueError('draws and batch_size must be positive')
    layouts = list(layouts)
    tasks = [(i, d, native) for i, native in enumerate(layouts) for d in range(draws)]
    sampled_by_key = {}
    for lo in range(0, len(tasks), batch_size):
        chunk = tasks[lo:lo+batch_size]
        samples = sample_layouts(model, [(n.ss, n.adjacency, None) for _, _, n in chunk], seed=seed+lo, return_exceptions=True)
        sampled_by_key.update({(i, d): sample for (i, d, _), sample in zip(chunk, samples)})
    rows = []
    for index, native in enumerate(layouts):
        samples = []
        for draw in range(draws):
            try:
                sampled = sampled_by_key.pop((index, draw))
                if isinstance(sampled, Exception):
                    raise sampled
                sampled, xyz, report = build_sketch(sampled, sweeps=sweeps, seed=seed+draw, repo=repo)
                samples.append(sampled)
                ee = np.triu((native.types[:, None] == 'E') & (native.types[None, :] == 'E'), 1)
                native_paired = ee & native.pairing  # Orientation is open on unsigned contacts too.
                ee &= native.adjacency == 2  # Pair presence is open only on state 2.
                paired = ee & native.pairing
                union = ee & (native.pairing | sampled.pairing)
                h = handedness(sampled)
                rows.append(dict(index=index, target=native.metadata.get('target', native.metadata.get('chain_id')), draw=draw, valid=True, centroid_rmsd=kabsch_rmsd(sampled.centroids, native.centroids),
                                 pairing_accuracy=float(np.mean(sampled.pairing[ee] == native.pairing[ee])) if ee.any() else None,
                                 pairing_pairs=int(ee.sum()),
                                 pairing_recall=float(np.mean(sampled.pairing[paired])) if paired.any() else None,
                                 pairing_jaccard=float((ee & native.pairing & sampled.pairing).sum()/union.sum()) if union.any() else None,
                                 polarity_accuracy=float(np.mean(sampled.orientation[native_paired] == native.orientation[native_paired])) if native_paired.any() else None,
                                 handedness_right=h['right'], handedness_total=h['n'],
                                 handedness_rate=h['right']/h['n'] if h['n'] else None,
                                 geometry_ok=report['geometry_ok'], fixed_map_agreement=report['map_agreement'],
                                 label_map_agreement=report['label_map_agreement'], sketch_recovery=report['sketch_recovery'],
                                 sketch_violation=report['sketch_violation']))
            except (ValueError, FloatingPointError) as exc:
                rows.append(dict(index=index, target=native.metadata.get('target', native.metadata.get('chain_id')), draw=draw, valid=False, error=str(exc)))
        diversity = [kabsch_rmsd(a.centroids, b.centroids) for a, b in itertools.combinations(samples, 2)]
        rows.append(dict(index=index, diversity=float(np.mean(diversity)) if diversity else None,
                         diversity_draws=len(samples), requested_draws=draws))
    return rows


def train(args):
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    layouts = [x for x in read_dataset(args.dataset) if x.metadata.get('split') == 'train']
    if not layouts:
        raise ValueError('dataset has no train split')
    if min(args.epochs, args.batch_size) < 1 or args.max_steps < 0 or args.lr <= 0:
        raise ValueError('positive epochs, batch size, learning rate and nonnegative max steps required')
    run_id = args.wandb_run_id or os.environ.get('WANDB_RUN_ID')
    if os.environ.get('SLURM_JOB_ID') and not run_id:
        raise ValueError('scheduled training requires registered WANDB_RUN_ID and login-node sync cron')
    run = None
    if run_id:
        import wandb
        directory = Path(args.wandb_dir); directory.mkdir(parents=True, exist_ok=True)
        run = wandb.init(project='harmonia-layout-planner', id=run_id, mode='offline',
                         dir=str(directory), config=dict(slurm_job_id=os.environ.get('SLURM_JOB_ID', 'local'),
                                                        dataset=str(args.dataset), seed=args.seed))
        lineage_path = directory / ('lineage_'+run_id+'.json')
        lineage = json.loads(lineage_path.read_text()) if lineage_path.exists() else dict(run_id=run_id, jobs=[])
        job = os.environ.get('SLURM_JOB_ID')
        if job not in lineage.setdefault('jobs', []):
            lineage['jobs'].append(job)
        lineage.update(slurm_job_id=job, checkpoint=str(args.ckpt))
        lineage_path.write_text(json.dumps(lineage))
    rng = np.random.default_rng(args.seed)
    model = LayoutFlow().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps, epoch, cursor, order = 0, 0, 0, None
    signature = dict(dataset_sha256=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
                     batch_size=args.batch_size, seed=args.seed, lr=args.lr)
    if args.resume:
        resume = args.ckpt if args.resume == 'latest' else args.resume
        if Path(resume).exists():
            payload = torch.load(resume, map_location=args.device, weights_only=True)
            state = payload.get('training')
            if state is None or state['signature'] != signature:
                raise ValueError('resume requires matching dataset, batch size, seed, lr and training RNG state')
            model.load_state_dict(payload['state_dict']); optimizer.load_state_dict(payload['optimizer'])
            steps, epoch, cursor, order = payload['step'], state['epoch'], state['cursor'], state['order']
            rng.bit_generator.state = json.loads(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'].cpu())
            if state['cuda_rng'] and args.device == 'cuda':
                torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        elif args.resume != 'latest':
            raise FileNotFoundError(resume)
    start, stopping = time.monotonic(), False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGUSR1)}
    def checkpoint():
        save_checkpoint(args.ckpt, model, optimizer=optimizer, step=steps, training=dict(
            signature=signature, epoch=epoch, cursor=cursor, order=order,
            numpy_rng=json.dumps(rng.bit_generator.state), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else []))
    print(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()), train_examples=len(layouts),
                          device=args.device, threads=args.threads, resumed_step=steps)), flush=True)
    try:
        while epoch < args.epochs:
            if stopping or (args.max_minutes and time.monotonic()-start >= args.max_minutes*60):
                stopping = True; break
            if args.max_steps and steps >= args.max_steps:
                break
            if order is None:
                order = rng.permutation(len(layouts)).tolist(); cursor = 0
            indices = order[cursor:cursor+args.batch_size]
            batch = collate([layouts[i] for i in indices], rng=rng, dropout=True, augment=True, device=args.device)
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss, terms = flow_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite flow loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            steps += 1; cursor += len(indices)
            metrics = dict(epoch=epoch+1, step=steps, loss=float(loss.detach()), seconds=time.monotonic()-start,
                           **{k: float(v.detach()) for k, v in terms.items()})
            if cursor >= len(order):
                epoch += 1; order = None; cursor = 0
            if steps % args.save_every == 0 or order is None:
                checkpoint()
            if steps == 1 or steps % 100 == 0 or order is None:
                if run is not None:
                    run.log(metrics, step=steps)
                print(json.dumps(metrics), flush=True)
    finally:
        checkpoint()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if run is not None:
            run.finish()
    # Budgets and preemption reserve the remaining wall time for saving, not validation.
    if not stopping:
        heldout = [x for x in read_dataset(args.dataset) if x.metadata.get('split') == 'val'][:args.val_limit]
        metrics = validate(model, heldout, draws=args.val_draws, seed=args.seed)
        Path(str(args.ckpt)+'.validation.json').write_text(json.dumps(metrics, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--dataset', default=str(DEFAULT_OUT/'layouts.npz'))
    t.add_argument('--epochs', type=int, default=40); t.add_argument('--batch-size', type=int, default=256)
    t.add_argument('--lr', type=float, default=2e-4); t.add_argument('--max-steps', type=int, default=0)
    t.add_argument('--resume', nargs='?', const='latest', help='checkpoint path, or latest --ckpt if present')
    t.add_argument('--max-minutes', type=float, default=0., help='save and exit after this training budget; zero is unlimited')
    t.add_argument('--save-every', type=int, default=100)
    t.add_argument('--val-limit', type=int, default=8); t.add_argument('--val-draws', type=int, default=8)
    t.add_argument('--ckpt', default=str(DEFAULT_OUT/'flow.pt'))
    t.add_argument('--wandb-run-id', help='registered ID; required for scheduled training; offline logging')
    t.add_argument('--wandb-dir', default=str(DEFAULT_OUT/'wandb_offline'))
    s = sub.add_parser('sample')
    s.add_argument('--ss', required=True); s.add_argument('--adj', required=True)
    g = s.add_mutually_exclusive_group(); g.add_argument('--signs'); g.add_argument('--signed-adj', action='store_true')
    s.add_argument('--out', required=True, help='Layout NPZ; companion .cif is written')
    s.add_argument('--steps', type=int, default=40); s.add_argument('--temperature', type=float, default=.85)
    s.add_argument('--sweeps', type=int, default=50); s.add_argument('--repo')
    v = sub.add_parser('validate')
    v.add_argument('--dataset', default=str(DEFAULT_OUT/'layouts.npz')); v.add_argument('--out', required=True)
    v.add_argument('--specs-pack', action='append', help='validate native targets from local pack(s) instead of dataset')
    v.add_argument('--batch-size', type=int, default=64)
    v.add_argument('--limit', type=int, default=200); v.add_argument('--draws', type=int, default=8)
    v.add_argument('--sweeps', type=int, default=0); v.add_argument('--repo')
    for parser in (s, v):
        parser.add_argument('--ckpt', required=True)
    for parser in (t, s, v):
        parser.add_argument('--seed', type=int, default=20261014)
        parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
        parser.add_argument('--threads', type=int, default=48)
        parser.add_argument('--dry-run', action='store_true', help='print configuration without reading or writing data')
    a = p.parse_args(argv)
    if a.cmd == 'train' and (a.max_minutes < 0 or not math.isfinite(a.max_minutes) or a.save_every < 1):
        p.error('nonnegative finite budget and positive save interval required')
    if a.threads < 1:
        p.error('positive threads required')
    if a.dry_run:
        print(json.dumps(vars(a), sort_keys=True)); return
    torch.set_num_threads(a.threads)
    if a.cmd == 'train':
        train(a); return
    model = load_checkpoint(a.ckpt, a.device)
    if a.cmd == 'sample':
        ss, adj = np.load(a.ss), np.load(a.adj)
        signs = np.load(a.signs) if a.signs else None
        if a.signed_adj:
            adj, signs = split_signed_adjacency(ss, adj)
        layout = sample_layout(model, ss, adj, signs=signs, seed=a.seed, steps=a.steps, temperature=a.temperature)
        from sketch_builder import build_sketch, write_cif
        layout, xyz, report = build_sketch(layout, sweeps=a.sweeps, seed=a.seed, repo=a.repo)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        layout.save(a.out); write_cif(Path(a.out).with_suffix('.cif'), xyz)
        print(json.dumps(report, sort_keys=True))
    else:
        natives = ([x for path in a.specs_pack for x in native_pack_layouts(path)] if a.specs_pack else
                   [x for x in read_dataset(a.dataset) if x.metadata.get('split') == 'val'])[:a.limit]
        if not natives:
            raise ValueError('no held-out examples')
        rows = validate(model, natives, draws=a.draws, seed=a.seed, sweeps=a.sweeps, repo=a.repo, batch_size=a.batch_size)
        Path(a.out).write_text(json.dumps(rows, indent=2))
        print(json.dumps(dict(natives=len(natives), draws=a.draws, valid=sum(r.get('valid', False) for r in rows))))


if __name__ == '__main__':
    main()
