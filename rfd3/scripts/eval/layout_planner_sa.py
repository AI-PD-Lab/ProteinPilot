#!/usr/bin/env python
"""B0: seeded, CPU rule-based map-to-layout planner (no native coordinates).

A CB contact graph includes both H-bond neighbours and cross-sheet packing.
Degree-two sheet hypotheses and near-cyclic barrels initialise rigid frames;
SA preserves their neighbours, supplied signs and right-handed crossovers.
"""
from __future__ import annotations

import argparse
from itertools import permutations, product
from functools import lru_cache
import json

import numpy as np
from scipy.spatial.transform import Rotation

import eval_common as ec
from layout_common import (Layout, block_adjacency, unit, perpendicular, objective,
                           crossover_values, strand_pair, refresh_strand_pairs, split_signed_adjacency,
                           polarity_constraints, polarity_evidence, strand_sign_assignments, layout_priors, register_prior, constraints_ok, koga_values, project_axis_floors, project_sheet_bounds, axis_repulsion, axis_distance, sse_clashes, sheet_normals)

# Signed triple-product convention measured from 1TIT CA axes and nearest
# paired strands (0-5, 1-4, 3-4). The native regression test guards the sign.
SHEET_TWIST = -np.deg2rad(12.)


def strand_components(types, adj):
    unseen = set(np.flatnonzero(np.asarray(types) == 'E').tolist())
    out = []
    while unseen:
        stack = [min(unseen)]
        comp = []
        while stack:
            i = stack.pop()
            if i not in unseen:
                continue
            unseen.remove(i); comp.append(i)
            stack.extend(sorted(j for j in unseen if adj[i, j] == 1))
        out.append(sorted(comp))
    return out


def _order_score(order, adj, cyclic=False):
    pairs = list(zip(order, order[1:]))
    if cyclic:
        pairs.append((order[-1], order[0]))
    return sum(3. if adj[i, j] == 1 else -4. if adj[i, j] == 0 else 0. for i, j in pairs)


def sheet_order(component, adj, rng):
    """Exact maximum-weight Hamiltonian path through <=8 strands, else greedy+SA.

    Reservoir sampling resolves optimal ties reproducibly without storing n!.
    """
    component = list(component)
    if len(component) <= 1:
        return component
    if len(component) <= 8:
        best, score, ties = None, -np.inf, 0
        for order in permutations(component):
            if order[0] > order[-1]:
                continue  # path reversals are equivalent here
            value = _order_score(order, adj)
            if value > score:
                best, score, ties = list(order), value, 1
            elif value == score:
                ties += 1
                if rng.integers(ties) == 0:
                    best = list(order)
        return best
    degrees = np.array([sum(adj[i, j] == 1 for j in component if j != i) for i in component])
    # Begin at a graph boundary when available. Starting in the middle of a
    # simple path needlessly makes greedy ordering miss a known contact.
    starts = np.flatnonzero(degrees == degrees.min())
    order = [component.pop(int(rng.choice(starts)))]
    while component:
        scores = np.array([3 if adj[order[-1], j] == 1 else -4 if adj[order[-1], j] == 0 else 0
                           for j in component])
        choices = np.flatnonzero(scores == scores.max())
        order.append(component.pop(int(rng.choice(choices))))
    score = _order_score(order, adj)
    best, best_score = order[:], score
    for step in range(1000):
        trial = order[:]
        a, b = rng.choice(len(order), 2, replace=False)
        trial[a], trial[b] = trial[b], trial[a]
        value = _order_score(trial, adj)
        t = max(.05, 2 * (1 - step / 1000))
        if value > score or rng.random() < np.exp(min(0., (value - score) / t)):
            order, score = trial, value
        if value > best_score:
            best, best_score = trial[:], value
    return best


@lru_cache(maxsize=128)
def _path_bipartitions(strands, flat_adjacency, size):
    """Enumerate path covers of a small unsigned strand graph, no native input.

    Each layer is an ordered H-bond path; unused positive edges are packing
    contacts. Path reversal/layer exchange symmetries are removed. Rank by
    lateral map feasibility, keeping alternatives for the geometric objective.
    """
    adj=np.asarray(flat_adjacency).reshape(size,size)
    found={}
    for order in permutations(strands):
        for cut in range(2,len(order)-1):
            a,b=order[:cut],order[cut:]
            if min(strands) not in a or a[0]>a[-1] or b[0]>b[-1]:
                continue
            if any(adj[i,j]==0 for group in (a,b) for i,j in zip(group,group[1:])):
                continue
            score_best=np.inf
            for reverse in (False,True):
                other=b[::-1] if reverse else b
                coords={i:4.85*(k-(len(g)-1)/2) for g in (a,other) for k,i in enumerate(g)}
                contacts=[(i,j) for i in a for j in other if adj[i,j]==1]
                shift=np.mean([coords[i]-coords[j] for i,j in contacts]) if contacts else 0.
                score=sum(max(abs(coords[i]-coords[j]-shift)-3.,0.)**2 for i,j in contacts)
                score+=sum(10. for g in (a,other) for k,i in enumerate(g) for j in g[k+2:] if adj[i,j]==1)
                score+=sum(max(3.-abs(coords[i]-coords[j]-shift),0.)**2 for i in a for j in other if adj[i,j]==0)
                score_best=min(score_best,score)
            found[(a,b)]=score_best
    return tuple(key for key,_ in sorted(found.items(),key=lambda kv:(kv[1],kv[0])))


def alpha_beta_orders(types, adj):
    """Infer a ribbon from helix footprints when E-E observations are sparse.

    A helix packs against a contiguous interval on a sheet face. Its positive
    footprint must not contain a known noncontact. This includes split BAB
    crossovers: sequence neighbours need not be sheet neighbours.
    """
    strands=np.flatnonzero(types=='E').tolist()
    helices=np.flatnonzero(types=='H').tolist()
    if not helices or not 3<=len(strands)<=8:
        return []
    def score(order):
        value=_order_score(order,adj)
        priors=layout_priors()['geometry']
        right=priors['crossover']
        for h in helices:
            hits=[k for k,i in enumerate(order) if adj[h,i]==1]
            if len(hits)>1:
                interval=order[min(hits):max(hits)+1]
                value-=5.*sum(adj[h,i]==0 for i in interval)
                value-=.3*(len(interval)-len(hits))
            if h and h+1<len(types) and types[h-1]==types[h+1]=='E':
                a,b=sorted((order.index(h-1),order.index(h+1)))
                # A right-handed BAB is more common for paired parallel
                # neighbours (0.960) than for split crossovers (0.700).
                subset='paired_parallel' if b-a==1 else 'unpaired'
                value+=np.log(right[subset]['frac_right'])
                # A split crossover spans a second neighbour most often;
                # the empirical perpendicular spacing has a broad upper tail.
                # Use its p5/p95 envelope so a third strand can lie underneath.
                if b-a>1:
                    spacing=priors['sheet_second_neighbour']['perp']
                    distance=4.85*(b-a)
                    value-=.5*(max(spacing['p5']-distance,0.,distance-spacing['p95'])/
                               ((spacing['p95']-spacing['p5'])/3.29))**2
                # A contacting third strand lies under a split crossover,
                # between the two EHE endpoints rather than beyond a cap.
                value-=(-np.log(right['unpaired']['frac_right']))*sum(
                    adj[h,i]==1 for k,i in enumerate(order) if k<a or k>b)
        return value
    orders=[o for o in permutations(strands) if o[0]<o[-1]]
    ranked=sorted(orders,key=lambda o:(-score(o),o))
    return [o for o in ranked if score(o)>=score(ranked[0])-2.][:6]


def sheet_hypotheses(types,adj,rng,limit=3):
    original=partition_sheets(types,adj,rng)
    strands=tuple(np.flatnonzero(types=='E').tolist())
    branched=any(sum(adj[i,j]==1 for j in strands if j!=i)>2 for i in strands)
    if any(ring for _,ring in original):
        return [original]
    if len(strands)<5 or not branched:
        orders=alpha_beta_orders(types,adj)
        return ([[(list(o),False)] for o in orders] if orders else [original])
    if len(strands)<=8:
        covers=_path_bipartitions(strands,tuple(adj.ravel()),len(types))
        candidates=[original]+[[(list(a),False),(list(b),False)] for a,b in covers[:limit]]
        # A non-leaf graph bridge cannot be explained by a packing triangle.
        # Keep that ribbon edge paired when proposing alternative layer covers;
        # a CB tip contact across layers is not an equivalent sheet topology.
        from layout_common import _strand_bridges
        bridges=_strand_bridges(strands,tuple(adj.ravel()),len(types))
        degree={i:sum(adj[i,j]==1 for j in strands if i!=j) for i in strands}
        required={tuple(sorted((i,j))) for i,j in bridges if min(degree[i],degree[j])>=2}
        def keeps_bridges(hypothesis):
            neighbours={tuple(sorted((i,j))) for order,_ in hypothesis for i,j in zip(order,order[1:])}
            return required<=neighbours
        return [h for h in candidates if keeps_bridges(h)] or [original]
    # Large graphs: SA over the bipartition and order, bounded independently
    # of factorial graph size. Unknown edges may bridge a path; zeros may not.
    best={};groups=[list(o) for o,_ in original[:2]]
    if len(groups)!=2 or set(sum(groups,[]))!=set(strands):
        groups=[list(strands[:len(strands)//2]),list(strands[len(strands)//2:])]
    def score(gs):
        return sum(_order_score(g,adj) for g in gs)-.2*abs(len(gs[0])-len(gs[1]))
    value=score(groups)
    for step in range(400):
        trial=[g[:] for g in groups]
        face=int(rng.integers(2))
        if len(trial[face])>2 and rng.random()<.5:
            trial[1-face].insert(int(rng.integers(len(trial[1-face])+1)),trial[face].pop(int(rng.integers(len(trial[face])))))
        else:
            rng.shuffle(trial[face])
        new=score(trial);temperature=max(.05,2*(1-step/400))
        if new>value or rng.random()<np.exp(min(0.,(new-value)/temperature)):
            groups,value=trial,new
        if all(adj[i,j]!=0 for g in groups for i,j in zip(g,g[1:])):
            best[tuple(tuple(g) for g in groups)]=value
    return [original]+[[(list(a),False),(list(b),False)] for (a,b),_ in sorted(best.items(),key=lambda kv:-kv[1])[:limit]]


def component_flip(layout,group):
    """Reverse a connected component and its register about its own centre."""
    group=list(group)
    # A directly attached helix must follow the reversed strand cap; an
    # isolated strand flip would otherwise stretch a zero-residue junction.
    group+= [h for h in np.flatnonzero(layout.types=='H') if h and h-1 in group
             and layout.blocks[h]['start']==layout.blocks[h-1]['stop']]
    axis=unit(layout.axes[group[0]])
    center=layout.centroids[group].mean(0)
    relative=layout.centroids[group]-center
    layout.centroids[group]-=2*(relative@axis)[:,None]*axis
    layout.axes[group]*=-1
    return sheet_normals(layout)


def orientation_components(layout):
    # Hairpin-connected components can span layers. Treating single strands
    # alone made their relative polarities effectively unreachable.
    graph={i:set() for i in np.flatnonzero(layout.types=='E')}
    for i,(a,b) in enumerate(zip(layout.blocks,layout.blocks[1:])):
        if a['type']==b['type']=='E' and b['start']-a['stop']<=8:
            graph[i].add(i+1);graph[i+1].add(i)
    for (i,j),_ in polarity_constraints(layout).items():
        graph[i].add(j);graph[j].add(i)
    groups=[];unseen=set(graph)
    while unseen:
        todo=[min(unseen)];group=[]
        while todo:
            i=todo.pop()
            if i not in unseen:continue
            unseen.remove(i);group.append(i);todo.extend(graph[i]&unseen)
        groups.append(group)
    return groups


def partition_sheets(types, adj, rng):
    """Degree-two H-bond neighbour hypothesis, separate from packing contacts.

    CB-min maps do not identify H bonds uniquely. Triangles/branches therefore
    generate sheet hypotheses, not a Hamiltonian sheet. Pure-beta branched
    components are split into balanced layers; contacts across layers remain
    packing constraints. Lone strands extend the nearest available sheet edge.
    A near-complete sequence cycle may have a non-contact, staggered seam.
    """
    strands = np.flatnonzero(types == 'E').tolist()
    if len(strands) >= 6:
        ring_edges = list(zip(strands, strands[1:] + strands[:1]))
        if sum(adj[i, j] == 1 for i, j in ring_edges) >= len(strands) - 1:
            return [(strands, True)]
    neighbours = adj.copy()
    if len(strands) >= 4:
        # In a triangle the two edges to a vertex with an external neighbour
        # cannot all occupy distinct H-bond faces. Keep the outside bridge
        # and the opposite edge; classify the competing triangle edges as
        # inter-sheet packing. This is a topology prior, not a native lookup.
        from itertools import combinations
        for triangle in combinations(strands, 3):
            if not all(adj[i,j] == 1 for i,j in combinations(triangle,2)):
                continue
            branches = [i for i in triangle if any(adj[i,j] == 1 for j in strands if j not in triangle)]
            if len(branches) == 1:
                for j in triangle:
                    if j != branches[0]:
                        neighbours[branches[0],j] = neighbours[j,branches[0]] = 0
        # Prune to the two available H-bond faces. Short turns have priority;
        # otherwise retain bridging long-range neighbours over local packing.
        for i in strands:
            contacts = [j for j in strands if neighbours[i,j] == 1]
            contacts.sort(key=lambda j: (abs(i-j) == 1, abs(i-j)), reverse=True)
            for j in contacts[2:]:
                neighbours[i,j] = neighbours[j,i] = 0
    sheets, singles = [], []
    for comp in strand_components(types, neighbours):
        order = sheet_order(comp, neighbours, rng)
        cyclic = len(comp) >= 4 and all(sum(adj[i, j] == 1 for j in comp if j != i) == 2 for i in comp)
        if cyclic:
            sheets.append((order, True)); continue
        segments, piece = [], [order[0]]
        for i, j in zip(order, order[1:]):
            if adj[i, j] == 0:
                segments.append(piece); piece = []
            piece.append(j)
        segments.append(piece)
        branched = any(sum(adj[i, j] == 1 for j in comp if j != i) > 2 for i in comp)
        width = int(np.ceil(len(comp) / 2))
        for piece in segments:
            if branched and len(comp) >= 5 and len(piece) > width:
                cuts = range(max(1, len(piece) - width), min(width, len(piece) - 1) + 1)
                # Triangle edges compete for the same H-bond face. Prefer
                # classifying one as packing while retaining bridge contacts.
                cut = max(cuts, key=lambda k: (any(adj[piece[k-1], z] == adj[piece[k], z] == 1
                                                   for z in comp if z not in piece[k-1:k+1]),
                                              -abs(len(piece) - 2*k)))
                sheets.extend([(piece[:cut], False), (piece[cut:], False)])
            elif len(piece) == 1:
                singles.append(piece[0])
            else:
                sheets.append((piece, False))
    for i in singles:
        if not sheets:
            sheets.append(([i], False)); continue
        candidates = [(int(adj[i,order[0]] == 0), -int(adj[i,order[0]] == 1), len(order), k, 0) for k, (order, ring) in enumerate(sheets) if not ring]
        candidates += [(int(adj[i,order[-1]] == 0), -int(adj[i,order[-1]] == 1), len(order), k, 1) for k, (order, ring) in enumerate(sheets) if not ring]
        if candidates:
            _, _, _, k, end = min(candidates)
            order = sheets[k][0]
            order.insert(len(order) if end else 0, i)
        else:
            sheets.append(([i], False))
    return sheets


def initialise_polarities(layout):
    constraints = polarity_constraints(layout)
    graph = {i: [] for i in range(len(layout.blocks))}
    for (i, j), sign in constraints.items():
        graph[i].append((j, sign)); graph[j].append((i, sign))
    polarity = {}
    for start in graph:
        if start in polarity:
            continue
        polarity[start] = 1
        queue = [start]
        while queue:
            i = queue.pop()
            for j, sign in graph[i]:
                wanted = polarity[i] * sign
                if j in polarity and polarity[j] != wanted:
                    raise ValueError('inconsistent strand polarity constraints')
                if j not in polarity:
                    polarity[j] = wanted; queue.append(j)
    # Align constraint components to the already proposed sheet axes.
    seen = set()
    for start in graph:
        if start in seen:
            continue
        queue, comp = [start], []
        while queue:
            i = queue.pop()
            if i in seen:
                continue
            seen.add(i); comp.append(i); queue += [j for j, _ in graph[i]]
        anchor_sign = 1 if layout.axes[start, 2] >= 0 else -1
        for i in comp:
            layout.axes[i] *= anchor_sign * polarity[i] * (1 if layout.axes[i, 2] >= 0 else -1)


def enforce_koga_roll(layout):
    """Compatibility wrapper: pleats come only from the local sheet normal.

    Loop-length cues are calibrated soft priors, never instructions to roll a
    strand into its sheet plane. Discrete pleat phases are sampled in SA.
    """
    return sheet_normals(layout)


def helix_face_flip(layout, i):
    """Reflect a helix through the plane of its nearest/contacting sheet."""
    strands = [j for j in range(len(layout.blocks)) if layout.types[j] == 'E']
    if layout.types[i] != 'H' or not strands:
        return False
    contacts = [j for j in strands if layout.adjacency[i,j] == 1] or strands
    # The plane is determined by the sheet, so a flip is an involution.
    anchor = min(contacts)
    normal = layout.normals[anchor]
    origin = layout.centroids[contacts].mean(0)
    layout.centroids[i] -= 2*np.dot(layout.centroids[i]-origin,normal)*normal
    return True


def parallel_sheet_hypothesis(types, adj):
    """Alpha/beta fallback when the map supplies no E-E contact.

    Shared H contacts provide weak strand-neighbour evidence; sequence E-H-E
    motifs supply parallel polarity. No native names or coordinates enter.
    """
    strands = np.flatnonzero(types == 'E').tolist()
    helices = np.flatnonzero(types == 'H').tolist()
    if len(strands)<3 or not helices or any(adj[i,j] == 1 for i in strands for j in strands if i!=j):
        return None
    def score(order):
        return sum(sum(adj[i,h] == adj[j,h] == 1 for h in helices)
                   -.05*abs(i-j) for i,j in zip(order,order[1:]))
    # A shared-helix graph does not uniquely determine sheet order. This is a
    # proposal, with the map hinge still deciding whether it is satisfiable.
    candidates = permutations(strands) if len(strands)<=8 else [strands]
    return list(max(candidates,key=score))


def _hairpin(blocks, i, j):
    lo, hi = sorted((i, j))
    return hi == lo + 1 and 0 <= blocks[hi]['start'] - blocks[lo]['stop'] <= 8


def orient_sheet_layers(sheets, adjacency):
    """Resolve relative sheet reversal and lateral register from packing edges.

    Path reversal is equivalent within an isolated sheet, but not when another
    layer packs against it. Enumerate this missing discrete degree of freedom.
    """
    from itertools import product
    if len(sheets)!=2 or any(cyclic for _,cyclic in sheets):
        return sheets, [0.]*len(sheets)
    best=None;best_score=np.inf
    for flips in product((False,True),repeat=len(sheets)-1):
        orders=[list(sheets[0][0])]+[list(reversed(order)) if flip else list(order)
                                 for (order,_),flip in zip(sheets[1:],flips)]
        coords={};layers={}
        for face,order in enumerate(orders):
            x=np.r_[0.,np.cumsum([7.2 if adjacency[i,j]==0 else 4.85 for i,j in zip(order,order[1:])])]
            x-=x.mean()
            for i,value in zip(order,x):coords[i]=value;layers[i]=face
        pairs=[(i,j) for i in orders[0] for j in orders[1] if adjacency[i,j]==1]
        shift=np.mean([coords[i]-coords[j] for i,j in pairs]) if pairs else 0.
        score=sum((coords[i]-coords[j]-shift)**2 for i,j in pairs)
        if score<best_score:
            best_score=score;best=([(o,False) for o in orders],[0.,float(shift)])
    return best


def initial_layout(ss, adjacency, rng, signs=None, orders=None, sheet_spec=None, antiparallel=False):
    blocks = ec.he_blocks_from_ss(ss)
    types = np.array([b['type'] for b in blocks])
    b = len(blocks)
    adj = block_adjacency(ss, adjacency)
    signs = np.zeros((b, b), int) if signs is None else np.asarray(signs, int)
    if signs.shape != (b, b) or not np.isin(signs, [-1, 0, 1]).all() or not np.array_equal(signs, signs.T):
        raise ValueError("signs must be symmetric [B,B], 0 unknown / +1 parallel / -1 antiparallel")
    if np.any((signs != 0) & ((adj != 1) | (types[:, None] != 'E') | (types[None, :] != 'E'))):
        raise ValueError("orientation signs require state-1 strand pairs")
    centers = np.zeros((b, 3))
    axes = np.tile([0., 0., 1.], (b, 1))
    normals = np.tile([0., 1., 0.], (b, 1))
    orientation = np.zeros((b, b), np.int8)
    fallback = parallel_sheet_hypothesis(types, adj) if orders is None else None
    sheets = ([(fallback, False)] if fallback is not None else partition_sheets(types, adj, rng)) if orders is None else [(list(o), False) for o in orders]
    if sheet_spec is not None:
        sheets=[(list(o),bool(ring)) for o,ring in sheet_spec]
    sheets,lateral_offsets=orient_sheet_layers(sheets,adj)
    for face, (order, cyclic) in enumerate(sheets):
        polarity, x = 1, 0.
        for k, i in enumerate(order):
            if k:
                j = order[k - 1]
                relation = signs[i, j] or (-1 if cyclic or _hairpin(blocks, i, j) else 1)
                polarity *= relation
                orientation[i, j] = orientation[j, i] = relation
                x += 7.2 if adj[i,j] == 0 and fallback is None else 4.85
            theta = (SHEET_TWIST*.25 if len(sheets)==2 else max(SHEET_TWIST, -np.deg2rad(80)/max(len(order)-1,1))) * (k - (len(order) - 1) / 2)
            axes[i] = polarity * np.array([0., -np.sin(theta), np.cos(theta)])
            normals[i] = np.array([0., np.cos(theta), np.sin(theta)])
            if not antiparallel:normals[i]*=rng.choice([-1,1])
            centers[i] = [x, 9.5 * face, 0.]
            if cyclic:
                angle = 2 * np.pi * k / len(order)
                radius = 4.85 / (2 * np.sin(np.pi / len(order)))
                centers[i] = [radius * np.cos(angle), radius * np.sin(angle) + 10 * face, 0.]
                normals[i] = [np.cos(angle), np.sin(angle), 0.]
                tangent=np.array([-np.sin(angle),np.cos(angle),0.])
                axes[i]=polarity*(np.cos(SHEET_TWIST)*np.array([0.,0.,1.])+np.sin(SHEET_TWIST)*tangent)
        if not cyclic:
            centers[order, 0] += lateral_offsets[face]-np.mean(centers[order, 0])
    strands = np.flatnonzero(types == 'E')
    helices = np.flatnonzero(types == 'H')
    for k, i in enumerate(helices):
        contacts = [j for j in strands if adj[i, j] == 1]
        if contacts:
            base = centers[contacts].mean(0)
            flank=(i-1 if i>0 and i+1<b and types[i-1]==types[i+1]=='E' else contacts[0])
            u = unit(axes[contacts[0]])
            if u@axes[flank]<0:u=-u
            n = unit(normals[contacts[0]])
            # Choose the face with the lowest H contact/noncontact mismatch.
            def face_cost(sign):
                point=base+10.*sign*n
                strand_cost=sum(max(np.linalg.norm(point-centers[j])-11.,0.)**2 if adj[i,j]==1
                                else max(9.-np.linalg.norm(point-centers[j]),0.)**2
                                for j in strands if adj[i,j]!=2)
                return strand_cost+sum(max(np.linalg.norm(point-centers[h])-12.,0.)**2
                           if adj[i,h] == 1 else max(9.-np.linalg.norm(point-centers[h]),0.)**2
                           for h in helices[:k] if adj[i,h] != 2)
            sign = min((-1,1),key=lambda side: face_cost(side)+rng.uniform(0,.01))
            centers[i] = base + 10. * sign * n
            axes[i] = Rotation.from_rotvec(rng.normal(0, .2, 3)).apply(-u)
            normals[i] = perpendicular(axes[i])
        elif len(strands):
            centers[i] = [9. * k, -10., 0.]
        else:
            angle = 2 * np.pi * k / max(len(helices), 1)
            radius = 5. if len(helices) > 1 else 0.
            centers[i] = [radius * np.cos(angle), radius * np.sin(angle), 0.]
            axes[i] *= (-1) ** k
    layout = Layout(ss, centers, axes, normals, adj, orientation=orientation)
    layout.metadata['signed_pairs'] = [[i,j,int(signs[i,j])] for i in range(b) for j in range(i+1,b) if signs[i,j]]
    initialise_polarities(layout)
    enforce_crossovers(layout)
    enforce_koga_roll(layout)
    layout.metadata.update(planner='sa', sheets=[o for o, _ in sheets],
                           barrels=[cyclic for _, cyclic in sheets], sheet_twist=float(SHEET_TWIST),
                           sheet_pairs=[list(pair) for order, cyclic in sheets
                                        for pair in list(zip(order, order[1:])) +
                                        ([(order[-1], order[0])] if cyclic else []) if adj[pair[0],pair[1]] == 1 or (fallback is not None and adj[pair[0],pair[1]] == 2)],
                           layer_ids={str(i): face for face,(order,_) in enumerate(sheets) for i in order})
    # Contiguous P-SEA E/H blocks are a bent SSE junction, not a loop that
    # can span an arbitrary sheet-face offset. Aim the helix N cap at the
    # preceding SSE cap before the finite-axis feasibility projection.
    from sketch_builder import placed_sse
    for h in helices:
        if h and blocks[h]['start']==blocks[h-1]['stop']:
            anchor=placed_sse(layout,h-1)[-1,1]
            side=layout.centroids[h]-anchor
            side-=np.dot(side,layout.axes[h-1])*layout.axes[h-1]
            direction=unit(layout.axes[h-1]+.4*unit(side,layout.normals[h-1]))
            layout.axes[h]=direction;layout.normals[h]=perpendicular(direction)
            offset=placed_sse(layout,h)[0,1]-layout.centroids[h]
            layout.centroids[h]=anchor+3.8*direction-offset
    layout.metadata['packing_pairs'] = [[i,j] for i in range(b) for j in range(i+1,b)
                                         if types[i] == types[j] == 'E' and adj[i,j] == 1
                                         and [i,j] not in layout.metadata['sheet_pairs']
                                         and [j,i] not in layout.metadata['sheet_pairs']]
    layout.metadata['parallel_sheet_fallback'] = fallback is not None
    sheet_normals(layout)
    return project_axis_floors(layout).centered()


def enforce_crossovers(layout):
    # With the design's triple-product formula the native right-handed sign
    # is NEGATIVE (1URN); reflecting the helix across its strand plane fixes it.
    for i, h, j, value in crossover_values(layout):
        if value > 0:
            normal = unit(np.cross(layout.axes[i], layout.centroids[j] - layout.centroids[i]),
                          perpendicular(layout.axes[i]))
            mid = (layout.centroids[i] + layout.centroids[j]) / 2
            layout.centroids[h] -= 2 * np.dot(layout.centroids[h] - mid, normal) * normal


def _topology_ok(layout):
    """Keep graph-derived sheet neighbours paired during rigid SA.

    The CB minimum alone can be satisfied by one tip contact. This constraint
    preserves the sheet constructed by B0 instead of allowing a paired ladder
    to dissolve into isolated tip contacts while lowering the map energy.
    """
    if not constraints_ok(layout) or sse_clashes(layout):
        return False
    if any(axis_distance(layout,i,j) > 5.0001 for i,j in layout.metadata.get('sheet_pairs',[])):
        return False
    for group,cyclic in zip(layout.metadata.get('sheets',[]),layout.metadata.get('barrels',[])):
        if cyclic and any(np.linalg.norm(layout.centroids[i]-layout.centroids[j])>=12.
                          for i,j in zip(group,group[1:]+group[:1])):
            return False
    if not layout.metadata.get('sheet_pairs'):
        return True
    from sketch_builder import placed_sse
    ca = np.zeros((len(layout.ss), 3))
    for i, block in enumerate(layout.blocks):
        ca[block['start']:block['stop']] = placed_sse(layout, i)[:, 1]
    return all(strand_pair(ca, layout.blocks, i, j)['paired']
               for i, j in layout.metadata.get('sheet_pairs', []))


def _sign_penalty(layout, signs):
    if signs is None:
        return 0.
    return float(sum(20 * max(.5 - signs[i, j] * (layout.axes[i] @ layout.axes[j]), 0.) ** 2
                     for i in range(len(layout.blocks)) for j in range(i + 1, len(layout.blocks)) if signs[i, j]))


def layout_map_agreement(layout):
    from sketch_builder import placed_sse
    from scipy.spatial.distance import cdist
    xyz=[placed_sse(layout,i) for i in range(len(layout.blocks))]
    cb=[ec.reconstruct_cb(x[:,0],x[:,1],x[:,2]) for x in xyz]
    results=[(cdist(cb[i],cb[j]).min()<6.)==bool(layout.adjacency[i,j])
             for i in range(len(cb)) for j in range(i+1,len(cb)) if layout.adjacency[i,j]!=2]
    return float(np.mean(results)) if results else 1.


def optimise_centroids(layout, iterations=80):
    """Analytic fixed-frame map/steric/endpoint minimisation for topology starts.

    This lets an entire discrete component move through the register barrier
    before comparison. It uses only SSE geometry and the supplied unsigned map.
    """
    from scipy.optimize import minimize
    from scipy.spatial.distance import pdist, cdist
    from sketch_builder import placed_sse
    blocks=layout.blocks;n=len(blocks)
    if n<2:return layout.copy()
    if any(layout.metadata.get('barrels',[])):iterations=max(iterations,150)
    xyz=[placed_sse(layout,i)-layout.centroids[i] for i in range(n)]
    cb=[ec.reconstruct_cb(x[:,0],x[:,1],x[:,2]) for x in xyz]
    pairs=list(zip(*np.triu_indices(n,1)))
    starts=layout.centroids.copy();lengths=layout.lengths
    # Axis minimisers on each segment give the envelope derivative w.r.t.
    # translation. Cache directions, lengths and pair constants per solve.
    ii,jj=np.triu_indices(n,1)
    d=layout.axes[ii]*lengths[ii,None];e=layout.axes[jj]*lengths[jj,None]
    aa=(d*d).sum(1);bb=(d*e).sum(1);cc=(e*e).sum(1)
    determinant=aa*cc-bb*bb
    floors=np.array([__import__('layout_common').axis_floor(layout,i,j) for i,j in pairs])
    sheet=set(tuple(sorted(p)) for p in layout.metadata.get('sheet_pairs',[]))
    def value(x):
        centers=x.reshape(n,3);grad=np.zeros((n,3));total=0.
        for i,j in pairs:
            delta=centers[i]-centers[j]
            state=layout.adjacency[i,j]
            if state!=2:
                vec=cb[i][:,None]+delta-cb[j][None];dist=np.linalg.norm(vec,axis=2)
                a,b=np.unravel_index(dist.argmin(),dist.shape);r=dist[a,b]
                error=max(r-5.9,0.) if state==1 else min(r-7.1,0.)
                total+=100.*error**2
                g=200.*error*vec[a,b]/max(r,1e-9);grad[i]+=g;grad[j]-=g
            if (i,j) in sheet:
                ca_vec=xyz[i][:,1,None,:]+delta-xyz[j][None,:,1,:]
                ca_dist=np.linalg.norm(ca_vec,axis=2)
                closest=np.argpartition(ca_dist.ravel(),1)[:2]
                vec=ca_vec.reshape(-1,3)[closest];distance=ca_dist.ravel()[closest]
                error=np.maximum(distance-5.25,0.)
                total+=1000.*np.square(error).sum()
                g=(2000.*error/np.maximum(distance,1e-9))[:,None]*vec
                grad[i]+=g.sum(0);grad[j]-=g.sum(0)
            # Atom-level sterics, fixed frames. Exclude true near neighbours.
            vec=(xyz[i].reshape(-1,3)[:,None]+delta-xyz[j].reshape(-1,3)[None])
            distances=np.linalg.norm(vec,axis=2)
            ids_i=np.repeat(np.arange(blocks[i]['start'],blocks[i]['stop']),4)
            ids_j=np.repeat(np.arange(blocks[j]['start'],blocks[j]['stop']),4)
            err=np.minimum(distances-2.75,0.)*(ids_j[None]-ids_i[:,None]>=2)
            total+=2000.*(err**2).sum()
            g=(4000.*err/np.maximum(distances,1e-9))[:,:,None]*vec
            grad[i]+=g.sum((0,1));grad[j]-=g.sum((0,1))
        # Finite-axis wall and H-bond neighbour spacing with exact envelopes.
        p=centers[ii]-d/2;q=centers[jj]-e/2
        diff=q-p;rr=(diff*d).sum(1);ss=-(diff*e).sum(1)
        safe=np.where(determinant>1e-10,determinant,1.)
        t=(cc*rr+bb*ss)/safe;z=(bb*rr+aa*ss)/safe
        candidates=[]
        for tt in (0.,1.):
            pp=p+tt*d;zz=np.clip(((pp-q)*e).sum(1)/np.maximum(cc,1e-12),0.,1.)
            candidates.append(pp-q-zz[:,None]*e)
        for zz in (0.,1.):
            qq=q+zz*e;tt=np.clip(((qq-p)*d).sum(1)/np.maximum(aa,1e-12),0.,1.)
            candidates.append(p+tt[:,None]*d-qq)
        interior=p+t[:,None]*d-q-z[:,None]*e
        valid=(determinant>1e-10)&(t>=0)&(t<=1)&(z>=0)&(z<=1)
        interior[~valid]=1e6;candidates.append(interior)
        vectors=np.stack(candidates);squared=(vectors*vectors).sum(2)
        chosen=squared.argmin(0);vector=vectors[chosen,np.arange(len(ii))];dist=np.sqrt(squared[chosen,np.arange(len(ii))])
        err=np.minimum(dist-floors,0.)
        weight=np.full(len(ii),3000.)
        spacing=np.array([tuple(pair) in sheet for pair in pairs])
        over=np.maximum(dist-4.9,0.)*spacing
        total+=np.sum(weight*err**2+3000.*over**2)
        g=(2*weight*err+6000.*over)[:,None]*vector/np.maximum(dist,1e-9)[:,None]
        np.add.at(grad,ii,g);np.add.at(grad,jj,-g)
        for i in range(n-1):
            gap=blocks[i+1]['start']-blocks[i]['stop']
            delta=centers[i+1]+xyz[i+1][0,1]-centers[i]-xyz[i][-1,1]
            r=np.linalg.norm(delta);reach=6. if gap==1 else 3.1*(gap+1)
            error=r-3.8 if gap==0 else max(r-reach,0.)
            total+=5000.*error**2
            g=10000.*error*delta/max(r,1e-9);grad[i]-=g;grad[i+1]+=g
            if gap>0:
                tangent_out=unit(xyz[i][-1,1]-xyz[i][-2,1])
                tangent_in=unit(xyz[i+1][1,1]-xyz[i+1][0,1])
                cap_delta=delta-1.9*(tangent_out+tangent_in)
                cap_distance=np.linalg.norm(cap_delta)
                cap_reach=3.8 if gap==1 else 3.1*gap
                error=max(cap_distance-cap_reach,0.)
                total+=500.*error**2
                g=1000.*error*cap_delta/max(cap_distance,1e-9);grad[i]-=g;grad[i+1]+=g
            if 0<gap<=8 and blocks[i]['type']==blocks[i+1]['type']=='E' and layout.axes[i]@layout.axes[i+1]<-.5:
                longitudinal=delta@layout.axes[i];total+=25.*longitudinal**2
                g=50.*longitudinal*layout.axes[i];grad[i]-=g;grad[i+1]+=g
        for group,cyclic in zip(layout.metadata.get('sheets',[]),layout.metadata.get('barrels',[])):
            if not cyclic:continue
            for i,j in zip(group,group[1:]+group[:1]):
                delta=centers[i]-centers[j];distance=np.linalg.norm(delta)
                error=max(distance-11.5,0.)
                total+=2000.*error**2
                g=4000.*error*delta/max(distance,1e-9);grad[i]+=g;grad[j]-=g
        layers=layout.metadata.get('sheets',[])
        if len(layers)==2 and not any(layout.metadata.get('barrels',[])):
            a,b=layers;normal=layout.normals[a[0]]
            delta=(centers[b].mean(0)-centers[a].mean(0))@normal
            error=delta-np.sign(delta)*9.5;total+=2*error**2
            grad[b]+=4*error*normal/len(b);grad[a]-=4*error*normal/len(a)
        total+=.01*np.square(centers-starts).sum();grad+=.02*(centers-starts)
        return float(total),grad.ravel()
    result=minimize(value,starts.ravel(),jac=True,method='L-BFGS-B',options=dict(maxiter=iterations,ftol=1e-8,maxls=20))
    out=layout.copy();out.centroids=result.x.reshape(n,3)
    enforce_crossovers(out);sheet_normals(out)
    return out


def aim_short_helix_caps(layout):
    """Choose a rigid helix roll to ease short E-H / H-E connections.

    Rotation about the helix axis preserves its geometry. The weighted radial
    endpoint least-squares problem has an exact one-angle solution; no loop
    coordinates or native frames enter.
    """
    from sketch_builder import placed_sse
    for h in np.flatnonzero(layout.types=='H'):
        xyz=placed_sse(layout,h)[:,1]
        axis=layout.axes[h];cosine=sine=0.
        neighbours=[]
        if h and layout.types[h-1]=='E':
            gap=layout.blocks[h]['start']-layout.blocks[h-1]['stop']
            if 0<gap<=3:neighbours.append((0,h-1,-1,gap))
        if h+1<len(layout.blocks) and layout.types[h+1]=='E':
            gap=layout.blocks[h+1]['start']-layout.blocks[h]['stop']
            if 0<gap<=3:neighbours.append((-1,h+1,0,gap))
        for end,i,other,gap in neighbours:
            radial=xyz[end]-layout.centroids[h]
            radial-=radial@axis*axis
            target=placed_sse(layout,i)[other,1]-layout.centroids[h]
            weight=1./(gap+1)**2
            cosine+=weight*(radial@target)
            sine+=weight*(np.cross(axis,radial)@target)
        if neighbours:
            rotation=Rotation.from_rotvec(axis*np.arctan2(sine,cosine))
            layout.normals[h]=rotation.apply(layout.normals[h])
    return layout


def layer_tilt_hypotheses(layout):
    """Population-relative sheet-axis angles, with both tilt handednesses.

    Flat, mutually parallel sheets are only one initialisation. A rigid tilt
    preserves a ribbon's own distances and gives its packing contacts axial
    freedom before fixed-frame centroid fitting. Constants come from the
    target-excluded layout population, never from a requested structure.
    """
    yield layout
    sheets=layout.metadata.get('sheets', [])
    if len(sheets)!=2 or any(layout.metadata.get('barrels', [])):
        return
    group=list(sheets[1])
    members=group+[h for h in np.flatnonzero(layout.types=='H')
                   if h and h-1 in group]
    center=layout.centroids[group].mean(0)
    normal=unit(layout.normals[group[0]])
    angle=layout_priors()['geometry']['sheet_axis_angle']['median']
    for value in (-angle,angle):
        out=layout.copy()
        rotation=Rotation.from_rotvec(normal*np.deg2rad(value)).as_matrix()
        out.centroids[members]=(out.centroids[members]-center)@rotation.T+center
        out.axes[members]=out.axes[members]@rotation.T
        out.normals[members]=out.normals[members]@rotation.T
        out.metadata['layer_tilt_degrees']=float(value)
        yield out


def contiguous_helix_hypotheses(layout):
    """Reverse a contiguous E-H continuation about its N-cap, if needed.

    Changing the second strand's MAP sign can require the helix to run back
    along the first strand. Its N-cap stays at the same junction; fixing its
    old axis while changing strand signs creates an artificial loop barrier.
    Physical fitting and gates select among these rigid alternatives.
    """
    from sketch_builder import placed_sse
    helices=[h for h in np.flatnonzero(layout.types=='H') if h and
             layout.types[h-1]=='E' and layout.blocks[h]['start']==layout.blocks[h-1]['stop']][:3]
    for flips in product((False,True),repeat=len(helices)):
        out=layout.copy()
        for h,flip in zip(helices,flips):
            if flip:
                anchor=placed_sse(out,h)[0,1]
                out.axes[h]*=-1
                out.centroids[h]+=anchor-placed_sse(out,h)[0,1]
        yield out


def global_orientations(layout):
    """Enumerate relative layer orders and exact MAP signs for <=12 strands.

    One absolute sign is fixed by rotational symmetry. Hairpin/signed
    constraints prune the binary search; E-H-E polarity stays probabilistic. Unlike single-strand SA moves, joint
    changes can traverse a tight hairpin's register barrier.
    """
    strands=np.flatnonzero(layout.types=='E').tolist()
    sheets=layout.metadata.get('sheets',[])
    assignments=strand_sign_assignments(layout)
    if assignments is None:
        return [layout]
    # Exact tied MAP solutions differ only in unobserved component gauges.
    # Bound geometry refinements, after evaluating every discrete sign vector.
    assignments=assignments[:8]
    outputs=[]
    for flips in product((False,True),repeat=max(0,len(sheets)-1)):
        base=layout.copy()
        for group,flip in zip(sheets[1:],flips):
            if flip:
                base.centroids[group,0]=base.centroids[group[::-1],0]
                base.metadata['sheets']=[g[::-1] if g==group else g for g in base.metadata['sheets']]
        for tilted in layer_tilt_hypotheses(base):
            for signs,log_probability in assignments:
                trial=tilted.copy()
                trial.metadata.update(polarity_method='exact_map_initialisation', initial_polarity_log_probability=log_probability,
                                      initial_map_polarities=[[i,signs[i]] for i in strands])
                changes={i:signs[i]*(1 if trial.axes[i,2]>=0 else -1) for i in strands}
                for i in strands:
                    trial.axes[i]*=changes[i]
                # A strand reversal also reverses the antiparallel helix in its
                # EHE unit. At a contiguous EH cap the entire continuation moves
                # to the reversed C end. Otherwise the binary search ranks an
                # artificial stretched junction instead of the intended state.
                for h in np.flatnonzero(trial.types=='H'):
                    contacts=[i for i in strands if trial.adjacency[h,i]==1]
                    anchor=h-1 if h and h-1 in changes else (contacts[0] if contacts else None)
                    if anchor is not None and changes[anchor]<0:
                        trial.axes[h]*=-1
                        if anchor==h-1 and trial.blocks[h]['start']==trial.blocks[h-1]['stop']:
                            axis=tilted.axes[h-1]
                            delta=trial.centroids[h]-trial.centroids[h-1]
                            trial.centroids[h]-=2.*(delta@axis)*axis
                # Fast axial register fit accounts for every sequence loop, not
                # just the H-bond edges. A two/three-residue EE turn is a tight
                # cap-to-cap hairpin; long loops retain their contour slack.
                constrained={h for _,h,_,_ in crossover_values(trial)}
                free=[h for h in np.flatnonzero(trial.types=='H') if h not in constrained
                      and not (h and trial.blocks[h]['start']==trial.blocks[h-1]['stop'])]
                # Terminal helices have no BAB handedness constraint. Their face
                # must be reconsidered after the crossover helices have moved.
                for hinged in contiguous_helix_hypotheses(trial):
                    for faces in product((False,True),repeat=min(len(free),3)):
                        variant=hinged.copy()
                        for h,flip in zip(free,faces):
                            if flip:helix_face_flip(variant,h)
                        phase_graph=np.full_like(variant.adjacency,2)
                        for i,j in variant.metadata.get('sheet_pairs',[]):phase_graph[i,j]=phase_graph[j,i]=1
                        # Absolute pleat phases matter for cross-sheet packing, and
                        # an isolated strand still has a phase. A phase gauge per
                        # ladder was incorrectly discarding both freedoms.
                        phase_ids = (strands if len(strands)<=6 else
                                     [group[0] for group in sheets[:3]])
                        # Every ribbon has relative pleat phases, for both parallel
                        # and antiparallel states. Comparing one polarity with more
                        # roll freedom would bias the discrete orientation search.
                        phase_candidates=[]
                        for phases in product((False,True),repeat=len(phase_ids)):
                            candidate=variant.copy()
                            for i,flip in zip(phase_ids,phases):
                                if flip:candidate.normals[i]*=-1
                            enforce_crossovers(candidate);aim_short_helix_caps(candidate)
                            candidate=fit_registers(candidate)
                            enforce_crossovers(candidate);sheet_normals(candidate)
                            value=objective(candidate)
                            phase_candidates.append((value,candidate))
                        # Registers change again in the centroid solve. A single
                        # pre-fit phase winner can discard the feasible packing state.
                        outputs.extend(candidate for _,candidate in
                                       sorted(phase_candidates,key=lambda row:row[0])[:4])
    return outputs


def fit_registers(layout):
    from scipy.optimize import least_squares
    from sketch_builder import placed_sse
    blocks=layout.blocks;n=len(blocks)
    xyz=[placed_sse(layout,i)-layout.centroids[i] for i in range(n)]
    centers=layout.centroids.copy()
    def residual(z):
        points=centers.copy();points[:,2]+=z
        result=[]
        for i in range(n-1):
            gap=blocks[i+1]['start']-blocks[i]['stop']
            delta=points[i+1]+xyz[i+1][0,1]-points[i]-xyz[i][-1,1]
            dist=np.linalg.norm(delta)
            reach=3.8 if gap==0 else (5.8 if gap==1 else 3.0*(gap+1))
            result.append(20.*(dist-reach if gap==0 else max(dist-reach,0.)))
            if blocks[i]['type']==blocks[i+1]['type']=='E' and 0<gap<=4:
                result.append(2.*delta[2])
            result.append(.05*dist/(gap+1))
        for i,j in layout.metadata.get('sheet_pairs',[]):
            prior=register_prior(layout,i,j)
            if prior:
                axial=abs((points[j]-points[i])@layout.axes[i])/3.3
                result.append(.2*(axial-prior['axial_abs_median']/3.3)/max(prior['axial_res_std'],.5))
            else:
                result.append(.2*(points[i,2]-points[j,2]))
        result.extend(.02*z)
        return result
    result=least_squares(residual,np.zeros(n),max_nfev=15)
    out=layout.copy();out.centroids[:,2]+=result.x
    return out


def relax_polarity_geometry(layout, iterations=25):
    """Geometry proposals after exact MAP initialisation use the soft prior.

    A coupled component/cap/register move can escape the endpoint barrier of
    a single axis reversal. Only observed short hairpins and supplied signs
    constrain the proposal. Model preferences remain part of the energy.
    """
    current=layout.copy()
    has_cap=any(h and current.types[h-1]=='E' and
                current.blocks[h]['start']==current.blocks[h-1]['stop']
                for h in np.flatnonzero(current.types=='H'))
    if current.metadata.get('polarity_method')!='exact_map_initialisation' or not has_cap:
        return current
    for _ in range(2):
        changed=False
        for group in orientation_components(current):
            trial=current.copy();component_flip(trial,group)
            if not constraints_ok(trial):
                continue
            trial=fit_registers(trial)
            trial=project_sheet_bounds(project_axis_floors(optimise_centroids(trial,iterations=iterations)))
            if (_topology_ok(trial) and
                    layout_map_agreement(trial)>=min(.9,layout_map_agreement(current)) and
                    objective(trial)<objective(current)):
                current=trial;changed=True
        if not changed:
            break
    return current


def plan_layout(ss, adjacency, *, seed=0, steps=600, restarts=6, signs=None):
    """One seeded draw, including discrete initialisations and annealed choices.

    No native lookup, length-specific exception, or best-of-diffusion selection.
    Joint layer/sign/register searches precede local rigid annealing.
    """
    if steps < 0 or restarts < 1:
        raise ValueError("steps >= 0 and restarts >= 1 required")
    rng = np.random.default_rng(seed)
    if not ec.he_blocks_from_ss(ss):
        return Layout(ss, np.empty((0, 3)), np.empty((0, 3)), metadata={'planner': 'sa', 'seed': int(seed)})
    signs = None if signs is None else np.asarray(signs, int)
    block_types=np.array([b['type'] for b in ec.he_blocks_from_ss(ss)])
    hypotheses=sheet_hypotheses(block_types,block_adjacency(ss,adjacency),rng)
    def energy(lay):
        return objective(lay) + _sign_penalty(lay, signs)
    candidates=[]
    for hypothesis in hypotheses:
        base=initial_layout(ss,adjacency,rng,signs,sheet_spec=hypothesis,antiparallel=True)
        pool=global_orientations(base)
        # Map/loop/handedness scores choose joint states before the expensive
        # atom solve. Tight-turn priors order the bounded refinement queue.
        ranked=sorted(pool,key=energy)
        refinement_limit=8 if len(hypotheses)==1 or len(np.flatnonzero(block_types=='E'))>6 else 4
        refinement_budget=min(refinement_limit,max(2,2*int(restarts)))
        for lay in ranked[:refinement_budget]:
            # Sub-Angstrom continuous restarts explore flat hinge plateaus
            # without changing the jointly enumerated discrete hypotheses.
            lay=lay.copy();lay.centroids+=rng.normal(0,.1,lay.centroids.shape)
            candidates.append(optimise_centroids(lay,iterations=70))
    energies = [energy(lay) for lay in candidates]
    candidates=[project_sheet_bounds(project_axis_floors(lay)) for lay in candidates]
    for lay in candidates:enforce_crossovers(lay)
    # Compare geometry-relaxed polarities for every proposed sheet cover,
    # rather than annealing only the best strained MAP cover.
    by_topology={}
    for lay in candidates:
        key=tuple(tuple(sorted(g)) for g in lay.metadata.get('sheets', []))
        if key not in by_topology or energy(lay)<energy(by_topology[key]):
            by_topology[key]=lay
    candidates.extend(relax_polarity_geometry(lay) for lay in by_topology.values())
    candidates=[lay for lay in candidates if constraints_ok(lay)]
    if not candidates:
        raise ValueError('no restart satisfies supplied polarity/crossover constraints')
    energies=[energy(lay) for lay in candidates]
    # Endpoint energy is advisory; it cannot discard a map-consistent start
    # in favour of a map-inconsistent one. The final backbone gates decide
    # whether a proposed loop is actually constructible.
    selected=min(range(len(candidates)),key=lambda k:(
        -int(layout_map_agreement(candidates[k])>=.9),
        -int(objective(candidates[k],return_terms=True)['loop']<=1000.),energies[k]))
    current = candidates[selected]
    e = energies[selected]
    best, best_e = current.copy(), e
    best_agreement=layout_map_agreement(best)
    b = len(current.blocks)
    counts = dict(rigid=0, order=0, orientation=0, face=0, register=0, mirror=0, helix_face=0, layer_distance=0, phase=0, component=0, layer_order=0, lone_layer=0)
    for step in range(steps):
        trial = current.copy()
        frac = 1 - step / max(steps, 1)
        temperature = .08 + 2.5 * frac ** 2
        i = int(rng.integers(b))
        choice = rng.random()
        if choice < .025 and b > 1:
            counts['order'] += 1
            if trial.metadata.get('polarity_method')=='exact_map_initialisation':
                trial=candidates[int(rng.integers(len(candidates)))].copy()
            else:
                trial=initial_layout(ss,adjacency,rng,signs,sheet_spec=hypotheses[int(rng.integers(len(hypotheses)))])
        elif choice < .065:
            counts['component']+=1
            groups=orientation_components(trial)+trial.metadata.get('sheets',[])
            if groups:component_flip(trial,groups[int(rng.integers(len(groups)))])
        elif choice < .085:
            counts['layer_order']+=1
            sheets=trial.metadata.get('sheets',[])
            if sheets and not any(trial.metadata.get('barrels',[])):
                group=sheets[int(rng.integers(len(sheets)))]
                trial.centroids[group]=trial.centroids[group[::-1]].copy()
                trial.metadata['sheets']=[g[::-1] if g==group else g for g in sheets]
        elif choice < .105:
            counts['lone_layer']+=1
            sheets=trial.metadata.get('sheets',[])
            paired=set(k for pair in trial.metadata.get('sheet_pairs',[]) for k in pair)
            lone=[i for g in sheets for i in g if i not in paired]
            if len(sheets)==2 and lone and current.metadata.get('polarity_method')!='exact_map_initialisation':
                strand=int(rng.choice(lone));groups=[g[:] for g in sheets]
                face=next(k for k,g in enumerate(groups) if strand in g)
                groups[face].remove(strand);groups[1-face].append(strand)
                if groups[face]:
                    trial=initial_layout(ss,adjacency,rng,signs,orders=groups)
        elif choice < .13:
            counts['orientation'] += 1
            trial.axes[i] *= -1
        elif choice < .15:
            counts['phase']+=1
            trial.normals[i]*=-1
        elif choice < .19:
            helices=np.flatnonzero(trial.types == 'H')
            if len(helices):
                counts['helix_face'] += 1
                helix_face_flip(trial,int(rng.choice(helices)))
            else:
                counts['phase'] += 1
                trial.normals[i] *= -1
        elif choice < .21:
            counts['layer_distance'] += 1
            sheets=trial.metadata.get('sheets',[])
            if len(sheets)>1 and not any(trial.metadata.get('barrels',[])):
                group=sheets[int(rng.integers(len(sheets)))]
                normal=unit(trial.normals[group].mean(0),trial.normals[group[0]])
                trial.centroids[group] += rng.normal(0,.5)*normal
            else:
                trial.centroids[i] += rng.normal(0,.5,3)
        elif choice < .24:
            counts['register'] += 1
            trial.centroids[i] += rng.choice([-1, 1]) * 3.3 * trial.axes[i]
        elif choice < .26:
            counts['mirror'] += 1
            trial.centroids[:, 0] *= -1
            trial.axes[:, 0] *= -1
            trial.normals[:, 0] *= -1
        else:
            counts['rigid'] += 1
            trial.centroids[i] += rng.normal(0, .2 + .8 * frac, 3)
            rot = Rotation.from_rotvec(rng.normal(0, .025 + .1 * frac, 3)).as_matrix()
            trial.axes[i] = rot @ trial.axes[i]
            trial.normals[i] = rot @ trial.normals[i]
        enforce_crossovers(trial)
        sheet_normals(trial)
        if axis_repulsion(trial) > 1e-5 or not _topology_ok(trial):
            continue
        value = energy(trial)
        if value < e or rng.random() < np.exp(min(0., (e - value) / temperature)):
            current, e = trial, value
        agreement=layout_map_agreement(current)
        if (int(agreement>=.9),-e)>(int(best_agreement>=.9),-best_e):
            best,best_e,best_agreement=current.copy(),e,agreement
    if objective(best,return_terms=True)['loop']>10. or best_agreement<.9:
        candidate=project_sheet_bounds(project_axis_floors(optimise_centroids(best,iterations=180)))
        enforce_crossovers(candidate);sheet_normals(candidate)
        if (_topology_ok(candidate) and layout_map_agreement(candidate)>=min(.9,best_agreement)
                and energy(candidate)<best_e):
            best=candidate;best_e=energy(best);best_agreement=layout_map_agreement(best)
    best.metadata.update(polarity_geometry='soft_likelihood', seed=int(seed), steps=int(steps), restarts=int(restarts),
                         energy=float(best_e), proposals=counts, hypothesis_count=len(hypotheses),
                         layout_map_agreement=best_agreement,
                         layout_loop_feasible=bool(objective(best,return_terms=True)['loop']<=1000.))
    enforce_crossovers(best)
    best = best.centered()
    from sketch_builder import placed_sse
    ca = np.zeros((len(best.ss), 3))
    for i, block in enumerate(best.blocks):
        ca[block['start']:block['stop']] = placed_sse(best, i)[:, 1]
    return refresh_strand_pairs(best, ca)


def main(argv=None):
    from sketch_builder import build_sketch, write_cif
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ss', required=True, help='SS .npy')
    p.add_argument('--adj', required=True, help='residue or block adjacency .npy')
    p.add_argument('--signed-adj', action='store_true', help='interpret +/-1 as known strand polarity')
    p.add_argument('--signs', help='optional block orientation .npy: 0 unknown, +/-1 known')
    p.add_argument('--out', required=True, help='sketch .cif')
    p.add_argument('--layout-out', required=True, help='layout .npz')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--sweeps', type=int, default=50)
    p.add_argument('--repo')
    a = p.parse_args(argv)
    ss, adj = np.load(a.ss), np.load(a.adj)
    if a.signs and a.signed_adj:
        p.error('--signs and --signed-adj are mutually exclusive')
    if a.signed_adj:
        adj, signs = split_signed_adjacency(ss, adj)
    else:
        signs = np.load(a.signs) if a.signs else None
    layout = plan_layout(ss, adj, seed=a.seed, steps=a.steps, signs=signs)
    layout, xyz, report = build_sketch(layout, seed=a.seed, sweeps=a.sweeps, repo=a.repo)
    layout.save(a.layout_out); write_cif(a.out, xyz)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
