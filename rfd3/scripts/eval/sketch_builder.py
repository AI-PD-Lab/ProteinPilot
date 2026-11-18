#!/usr/bin/env python
"""Build poly-GLY N/CA/C/O sketches from Layout objects or --layout NPZ.

Loops are approximate backbones, intended for partial diffusion at t >= 60;
SSE atoms use standard peptide internal coordinates, never native atoms.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.spatial.transform import Rotation

import eval_common as ec
from layout_common import (Layout, unit, perpendicular, principal_axis, objective,
                           handedness, strand_pair, refresh_strand_pairs, shape_metrics,
                           constraints_ok, koga_values, axis_repulsion, project_axis_floors, project_sheet_bounds, axis_distance, sse_clashes, sheet_normals, layout_rg_limits)


def nerf(a, b, c, length, angle, dihedral):
    """Next atom from three preceding atoms; angles in degrees."""
    bc = unit(c - b)
    normal = unit(np.cross(b - a, bc))
    cross = np.cross(normal, bc)
    theta, phi = np.deg2rad([angle, dihedral])
    return c + length * (-np.cos(theta) * bc + np.sin(theta) *
                         (np.cos(phi) * cross + np.sin(phi) * normal))


@lru_cache(maxsize=256)
def _ideal_sse(kind, n, twist):
    if kind not in ('H', 'E') or n < 1:
        raise ValueError("SSE requires H/E and positive residue count")
    phi, psi = (-57., -47.) if kind == 'H' else (-120., 130.)
    # Atom order N, CA, C; trans omega. Standard Engh-Huber geometry.
    xyz = [np.array([0., 0., 0.]), np.array([1.458, 0., 0.])]
    xyz.append(xyz[1] + 1.525 * np.array([-np.cos(np.deg2rad(111.2)),
                                        np.sin(np.deg2rad(111.2)), 0.]))
    for i in range(1, n):
        mod = twist * np.sin(2 * np.pi * i / max(n, 2)) if kind == 'E' else 0.
        xyz.append(nerf(*xyz[-3:], 1.329, 116.2, psi + mod))
        xyz.append(nerf(*xyz[-3:], 1.458, 121.7, 180.))
        xyz.append(nerf(*xyz[-3:], 1.525, 111.2, phi + mod))
    ncac = np.asarray(xyz).reshape(n, 3, 3)
    out = np.empty((n, 4, 3))
    out[:, :3] = ncac
    for i in range(n):
        out[i, 3] = nerf(*ncac[i], 1.231, 120.8, psi + 180.)
    return out


def ideal_sse(kind, n, twist=0.):
    """Independent copy, shape [n,4,3], ordered N/CA/C/O."""
    return _ideal_sse(str(kind).upper(), int(n), float(twist)).copy()


def _roll_normal(ca, u, kind):
    rel = ca - ca.mean(0)
    residual = rel - np.outer(rel @ u, u)
    v = ((residual * (-1.) ** np.arange(len(ca))[:, None]).sum(0)
         if kind == 'E' else residual[0])
    return unit(v, perpendicular(u))


def placed_sse(layout, i):
    b = layout.blocks[i]
    xyz = ideal_sse(b['type'], b['stop'] - b['start'], layout.metadata.get('strand_twist', 0.))
    ca = xyz[:, 1]
    axis = principal_axis(ca)
    normal = _roll_normal(ca, axis, b['type'])
    source = np.column_stack((axis, normal, np.cross(axis, normal)))
    u, n = layout.axes[i], layout.normals[i]
    target = np.column_stack((u, n, np.cross(u, n)))
    return (xyz - ca.mean(0)) @ (target @ source.T).T + layout.centroids[i]


def loop_ca(start, stop, n_loop, center, *, refine=True):
    """Quadratic Bezier arc, resampled by length; endpoints excluded.

    A stretched gap is returned as a straight trace and flagged by self_check;
    inventing residues or silently moving SSE endpoints would change the spec.
    """
    if n_loop <= 0:
        return np.empty((0, 3))
    edges = n_loop + 1
    delta = stop - start
    gap = np.linalg.norm(delta)
    if gap >= 3.8 * edges:
        return np.linspace(start, stop, edges + 1)[1:-1]
    tangent = unit(delta, [1., 0., 0.])
    inward = center - (start + stop) / 2
    inward -= np.dot(inward, tangent) * tangent
    inward = unit(inward, perpendicular(tangent))
    side = np.cross(tangent, inward)
    t = np.linspace(0., 1., edges + 1)
    # Bounded surface detour: use waves rather than a length-proportional
    # antenna. Global loop refinement below supplies steric feasibility.
    # Start with a single exterior arc. Repeated helical waves introduce
    # winding traps before steric optimisation has even begun.
    def curve(radius):
        dense_t=np.linspace(0.,1.,1001)
        dense=start+dense_t[:,None]*delta-radius*np.sin(np.pi*dense_t)[:,None]*inward
        length=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(dense,axis=0),axis=1))]
        sample=np.linspace(0.,length[-1],edges+1)
        return np.column_stack([np.interp(sample,length,dense[:,k]) for k in range(3)])
    radius = brentq(lambda h: np.linalg.norm(np.diff(curve(h),axis=0),axis=1).mean()-3.8,
                    0.,3.8*edges,xtol=1e-7)
    pts=curve(radius)
    if refine:
        from scipy.optimize import least_squares
        initial = pts[1:-1].copy()
        def residual(x):
            p = np.vstack((start, x.reshape(-1,3), stop))
            return np.r_[np.linalg.norm(np.diff(p,axis=0),axis=1)-3.8, .001*(x-initial.ravel())]
        pts[1:-1] = least_squares(residual, initial.ravel(), max_nfev=100).x.reshape(-1,3)
    return pts[1:-1]


def backbone_from_ca(ca):
    """Peptide atoms in the trace's local frame; final routing refines planes."""
    ca = np.asarray(ca, float)
    if len(ca) == 0:
        return np.empty((0,4,3))
    return _peptide_from_ca(ca)


def assemble(layout):
    n = len(layout.ss)
    xyz = np.full((n, 4, 3), np.nan)
    blocks = layout.blocks
    center = layout.centroids.mean(0) if len(blocks) else np.zeros(3)
    if not blocks:
        return backbone_from_ca(np.column_stack((3.8 * np.arange(n), np.zeros((n, 2)))))
    for i, b in enumerate(blocks):
        xyz[b['start']:b['stop']] = placed_sse(layout, i)
    for a, b in zip(blocks, blocks[1:]):
        lo, hi = a['stop'], b['start']
        xyz[lo:hi] = backbone_from_ca(loop_ca(xyz[lo - 1, 1], xyz[hi, 1], hi - lo, center, refine=False))
    # Fold terminal tails around the body; never extrapolate an end-block axis.
    for first in (True, False):
        block = blocks[0 if first else -1]
        anchor = block['start'] if first else block['stop'] - 1
        positions = list(range(anchor-1, -1, -1) if first else range(anchor+1, n))
        if not positions:
            continue
        start = xyz[anchor,1]
        # Extend locally beyond the SSE cap, then curl along the exterior.
        # Centroid-centred terminal circles can start deep inside the body.
        neighbour = anchor+1 if first else anchor-1
        tangent=unit(start-xyz[neighbour,1],start-center)
        outward=unit(start-center,perpendicular(tangent))
        outward-=np.dot(outward,tangent)*tangent
        outward=unit(outward,perpendicular(tangent))
        point=start.copy()
        for k,pos in enumerate(positions,1):
            angle=min((k-1)*.55,2.)
            direction=np.cos(angle)*tangent+np.sin(angle)*outward
            point=point+3.8*direction
            xyz[pos]=backbone_from_ca(point[None])[0]
    if not np.isfinite(xyz).all():
        raise ValueError("incomplete sketch")
    return xyz


def sse_mask(layout):
    mask = np.zeros(len(layout.ss), bool)
    for block in layout.blocks:
        mask[block['start']:block['stop']] = True
    return mask


def hull_planes(points):
    """Unit outward hull planes; degenerate bodies have no enclosed volume."""
    from scipy.spatial import ConvexHull, QhullError
    points = np.asarray(points)
    if len(points) < 4 or np.linalg.matrix_rank(points-points.mean(0)) < 3:
        return np.empty((0, 4))
    try:
        return ConvexHull(points).equations
    except QhullError:
        return np.empty((0, 4))


def sheet_obstacles(layout, ca):
    """Finite 2 A sheet slabs, including sheets derived from native pairing."""
    sheets = layout.metadata.get('sheets')
    if sheets is None:
        unseen = set(np.flatnonzero(layout.types == 'E').tolist()); sheets = []
        while unseen:
            stack = [min(unseen)]; group = []
            while stack:
                i = stack.pop()
                if i not in unseen:
                    continue
                unseen.remove(i); group.append(i)
                stack.extend(j for j in unseen if layout.pairing[i,j])
            sheets.append(group)
    result = []
    from scipy.spatial import ConvexHull
    for group_index,group in enumerate(sheets):
        # A cylindrical sheet cannot be tested as a planar obstacle.
        cyclic=layout.metadata.get('barrels',[])
        if group_index<len(cyclic) and cyclic[group_index]:
            continue
        if len(group) < 2:
            continue
        pts = ca[np.concatenate([ec.block_idx(layout.blocks,i) for i in group])]
        center = pts.mean(0); _,singular,basis = np.linalg.svd(pts-center,full_matrices=False)
        if singular[2] > .4*singular[1]:
            continue  # Curved/cylindrical sheets have no single slab.
        planes = ConvexHull((pts-center)@basis[:2].T).equations
        result.append((center, basis, planes))
    return result


def threading_report(layout, ca):
    loops = np.flatnonzero(~sse_mask(layout)); residues = set()
    for center,basis,planes in sheet_obstacles(layout,ca):
        local = (ca[loops]-center)@basis.T
        inside = ((local[:,:2]@planes[:,:2].T+planes[:,2]).max(1) < -1e-6)
        residues.update(loops[inside & (abs(local[:,2]) < 2.)].tolist())
        for i in range(len(ca)-1):
            if sse_mask(layout)[i] or sse_mask(layout)[i+1]:
                continue
            a,b=(ca[i:i+2]-center)@basis.T
            if a[2]*b[2] < 0:
                t=a[2]/(a[2]-b[2]);p=a[:2]+t*(b[:2]-a[:2])
                if (p@planes[:,:2].T+planes[:,2]).max() < -1e-6:
                    residues.add(i if i in loops else i+1)
    return sorted(residues)


def _peptide_from_ca(ca, xp=np, roll=None):
    """Trans peptide planes in local CA frames (no world-frame offsets).

    CA_i-C_i-N_i+1-CA_i+1 is the same rigid peptide unit used by NeRF.
    CA virtual angles remain free. The local bend selects the plane phase.
    End atoms extrapolate the nearest peptide plane.
    """
    def norm(x):
        return x / xp.sqrt((x*x).sum(-1,keepdims=True)+1e-16)
    if len(ca) < 2:
        # A single residue has no peptide frame.
        return ca[:,None,:] + xp.asarray([[-1.2,.8,0.],[0.,0.,0.],[1.3,.8,0.],[1.6,1.9,0.]])
    tangent = norm(ca[1:]-ca[:-1])
    bends = xp.concatenate((tangent[1:]-tangent[:-1], tangent[-1:]-tangent[-2:-1]),0) if len(ca)>2 else tangent*0
    # Plane bends are local. Non-collinear deterministic fallback is used only
    # for straight/degenerate traces, which the angle gate subsequently rejects.
    fallback = xp.asarray([.371,.529,.763]) + tangent*0
    bend = bends - (bends*tangent).sum(-1,keepdims=True)*tangent
    fallback = fallback-(fallback*tangent).sum(-1,keepdims=True)*tangent
    plane = norm(bend+1e-6*fallback)
    if roll is not None:
        cross = xp.stack((tangent[:,1]*plane[:,2]-tangent[:,2]*plane[:,1],
                          tangent[:,2]*plane[:,0]-tangent[:,0]*plane[:,2],
                          tangent[:,0]*plane[:,1]-tangent[:,1]*plane[:,0]),-1)
        plane = xp.cos(roll)[:,None]*plane + xp.sin(roll)[:,None]*cross
    c = ca[:-1] + 1.428421134*tangent + .534076833*plane
    n = ca[:-1] + 2.395637878*tangent - .377368596*plane
    o = c + .220096559*tangent + 1.211164111*plane
    # Trans unit endpoints are within 0.02 A of standard bond geometry.
    c = xp.concatenate((c, ca[-1:]+1.428421134*tangent[-1:]+.534076833*plane[-1:]),0)
    o = xp.concatenate((o, c[-1:]+.220096559*tangent[-1:]+1.211164111*plane[-1:]),0)
    n = xp.concatenate((ca[:1]-1.408317060*tangent[:1]-.377368596*plane[:1], n),0)
    return xp.stack((n,ca,c,o),1)


def compact_loops(layout, backbone, seed=0):
    """Fixed-SSE routing with bonds, virtual angles and atom sterics.

    No radius of gyration, half separation or attraction to a body centre is
    used. Finite sheet slabs protect topology; hull depth is diagnostic only.
    The route is permitted a 6 A surface shell. Failed routes are reported by
    self_check; steric/angle walls are never weakened to obtain a passing key.
    """
    ca = backbone[:,1].copy(); fixed = sse_mask(layout); variable = np.flatnonzero(~fixed)
    if not len(variable):
        return backbone
    body = ca[fixed]
    planes = hull_planes(body)
    slabs = sheet_obstacles(layout,ca)
    rows,cols = np.triu_indices(len(ca),2)
    active = ~fixed[rows] | ~fixed[cols]; rows,cols = rows[active],cols[active]
    tri = ~(fixed[:-2] & fixed[1:-1] & fixed[2:])
    tri_ids = np.flatnonzero(tri)
    rng = np.random.default_rng(seed)
    initial = ca[variable].copy()
    def energy(x):
        ca[variable] = x.reshape(-1,3)
        grad = np.zeros_like(ca)
        delta = np.diff(ca,axis=0); dist = np.linalg.norm(delta,axis=1)
        errors = dist-3.8
        value = 2000.*(errors**2).sum()
        g = 4000.*(errors/np.maximum(dist,1e-9))[:,None]*delta
        grad[:-1]-=g;grad[1:]+=g
        delta=ca[rows]-ca[cols];dist=np.linalg.norm(delta,axis=1)
        error=np.minimum(dist-3.85,0.)
        value+=4000.*(error**2).sum()
        g=8000.*(error/np.maximum(dist,1e-9))[:,None]*delta
        np.add.at(grad,rows,g);np.add.at(grad,cols,-g)
        v=ca[:-2]-ca[1:-1];w=ca[2:]-ca[1:-1]
        nv=np.linalg.norm(v,axis=1);nw=np.linalg.norm(w,axis=1)
        cos=(v*w).sum(1)/np.maximum(nv*nw,1e-12)
        # Leave numerical margin inside the requested 85..145 degree gate.
        lower,upper=(87.,143.) if layout.metadata.get('native_layout') else (77.,153.)
        err=cos-np.clip(cos,np.cos(np.deg2rad(upper)),np.cos(np.deg2rad(lower)))
        err[~tri]=0
        value+=250000.*(err**2).sum()
        gv=500000.*err[:,None]*(w/np.maximum(nv*nw,1e-12)[:,None]-cos[:,None]*v/np.maximum(nv**2,1e-12)[:,None])
        gw=500000.*err[:,None]*(v/np.maximum(nv*nw,1e-12)[:,None]-cos[:,None]*w/np.maximum(nw**2,1e-12)[:,None])
        grad[:-2]+=gv;grad[2:]+=gw;grad[1:-1]-=gv+gw
        for center,basis,poly in slabs:
            local=(ca[variable]-center)@basis.T
            distances=local[:,:2]@poly[:,:2].T+poly[:,2]
            face=distances.argmax(1);dxy=distances[np.arange(len(variable)),face]
            dz=abs(local[:,2])-2.15
            # Distance to nearest escape face of a convex sheet prism.
            escape=np.maximum(dxy,dz);err=np.minimum(escape,0.)
            value+=4000.*(err**2).sum()
            normals=np.where((dxy>=dz)[:,None],poly[face,:2]@basis[:2],np.sign(local[:,2,None])*basis[2])
            grad[variable]+=8000.*err[:,None]*normals
        if len(body):
            distances=ec.pairwise(ca[variable],body);near=distances.argmin(1)
            dist=distances[np.arange(len(variable)),near]
            err=np.maximum(dist-6.,0.)
            value+=.2*(err**2).sum()
            grad[variable]+=.4*(err/np.maximum(dist,1e-9))[:,None]*(ca[variable]-body[near])
        return float(value),grad[variable].ravel()
    best=None;best_e=np.inf
    for attempt in range(2):
        start=initial.copy() if attempt==0 else initial+rng.normal(0,2.,initial.shape)
        result=minimize(energy,start.ravel(),jac=True,method='L-BFGS-B',
                        options=dict(maxiter=1600 if layout.metadata.get('native_layout') else 800,ftol=1e-11,gtol=1e-5,maxls=40))
        if result.fun<best_e:
            best=result.x.copy();best_e=result.fun
        if best_e < 1.:
            break
    ca[variable]=best.reshape(-1,3)
    out=backbone.copy();out[variable]=_peptide_from_ca(ca)[variable]
    # Refine CA positions with atom sterics differentiating the peptide frames.
    out = refine_loop_atoms(layout,out,variable,planes,slabs,energy)
    layout.metadata['loop_route_energy'] = float(energy(out[variable,1].ravel())[0])
    return out


def refine_loop_atoms(layout, backbone, variable, planes, slabs, ca_energy):
    import torch
    # Small CPU tensors: BLAS thread teams cost more than these vector ops.
    previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        fixed_xyz=torch.tensor(backbone,dtype=torch.float64)
        fixed_ca=fixed_xyz[:,1].clone();n=len(backbone)
        mask=np.zeros(n,bool);mask[variable]=True
        ids=np.repeat(np.arange(n),4)
        rows,cols=np.triu_indices(4*n,1)
        use=(ids[cols]-ids[rows]>=2)&(mask[ids[rows]]|mask[ids[cols]])
        rows,cols=rows[use],cols[use]
        ncoords=3*len(variable)
        def value(x):
            base,base_grad=ca_energy(x[:ncoords])
            parameters=torch.tensor(x,requires_grad=True,dtype=torch.float64)
            points=parameters[:ncoords].reshape(-1,3)
            ca=fixed_ca.index_copy(0,torch.as_tensor(variable),points)
            class XP:
                sqrt=staticmethod(torch.sqrt); cos=staticmethod(torch.cos); sin=staticmethod(torch.sin)
                concatenate=staticmethod(torch.cat); stack=staticmethod(torch.stack)
                @staticmethod
                def asarray(a):return torch.as_tensor(a,dtype=torch.float64)
            loop=_peptide_from_ca(ca,XP,parameters[ncoords:])
            xyz=fixed_xyz.index_copy(0,torch.as_tensor(variable),loop[variable]).reshape(-1,3)
            distances=torch.sqrt(((xyz[rows]-xyz[cols])**2).sum(1)+1e-16)
            error=torch.clamp(2.75-distances,min=0.)
            loss=5000.*(error**2).sum()
            loss.backward()
            gradient=parameters.grad.numpy().copy();gradient[:ncoords]+=base_grad
            return base+float(loss.detach()),gradient
        start=np.r_[backbone[variable,1].ravel(),np.zeros(n-1)]
        result=minimize(value,start,jac=True,method='L-BFGS-B',
                        options=dict(maxiter=2000 if layout.metadata.get('native_layout') else 600,ftol=1e-11,gtol=1e-5,maxls=40))
        ca=backbone[:,1].copy();ca[variable]=result.x[:ncoords].reshape(-1,3)
        out=backbone.copy();out[variable]=_peptide_from_ca(ca,roll=result.x[ncoords:])[variable]
        return out
    finally:
        torch.set_num_threads(previous_threads)


def relax_layout(layout, sweeps=50, seed=0):
    """Seeded rigid-block Monte Carlo on SSE sterics and endpoint feasibility.

    Loop routing follows once, with SSEs fixed. Compactness uses SSE CA in
    both SA and relaxation, so loop detours cannot pull block axes together.
    """
    rng = np.random.default_rng(seed)
    from layout_planner_sa import enforce_crossovers, layout_map_agreement
    initial_agreement=layout_map_agreement(layout)
    current = project_sheet_bounds(project_axis_floors(layout))
    enforce_crossovers(current)
    energy = objective(current)
    best, best_e = current.copy(), energy
    # Native layouts retain all strand-pair topology, including absent pairs.
    required_pairs = [(i, j) for i in range(len(layout.blocks))
                      for j in range(i + 1, len(layout.blocks)) if layout.pairing[i, j]]
    def contact_states(lay):
        xyz = [placed_sse(lay,k) for k in range(len(lay.blocks))]
        cb = [ec.reconstruct_cb(x[:,0],x[:,1],x[:,2]) for x in xyz]
        return {(i,j): bool(ec.pairwise(cb[i],cb[j]).min()<6.)
                for i in range(len(cb)) for j in range(i+1,len(cb))}
    initial_states = contact_states(layout)
    for sweep in range(sweeps):
        progress = 1 - sweep / max(sweeps, 1)
        temperature = .05 + .25 * progress
        for i in rng.permutation(len(current.blocks)):
            trial = current.copy()
            if rng.random() < .25 and trial.types[i] == 'H':
                # Helix roll search also reduces backbone contacts at a face.
                angle = rng.choice([-1, 1]) * (np.pi / 6 if trial.types[i] == 'H' else .12)
                rot = Rotation.from_rotvec(trial.axes[i] * angle).as_matrix()
            else:
                trial.centroids[i] += rng.normal(0, .35 * progress + .05, 3)
                rot = Rotation.from_rotvec(rng.normal(0, .035 * progress + .005, 3)).as_matrix()
            trial.axes[i] = rot @ trial.axes[i]
            trial.normals[i] = rot @ trial.normals[i]
            sheet_normals(trial)
            if (axis_repulsion(trial) > 1e-5 or not constraints_ok(trial) or
                    sse_clashes(trial) > sse_clashes(current) or
                    any(axis_distance(trial,a,b) > 5.0001 for a,b in trial.metadata.get('sheet_pairs',[]))):
                continue
            trial_xyz = np.zeros((len(trial.ss),4,3))
            for k, block in enumerate(trial.blocks):
                trial_xyz[block['start']:block['stop']] = placed_sse(trial,k)
            if any(not strand_pair(trial_xyz[:, 1], trial.blocks, i, j)['paired']
                   for i, j in required_pairs):
                continue
            if layout.metadata.get('native_layout'):
                if any(strand_pair(trial_xyz[:,1],trial.blocks,a,b)['paired'] != bool(layout.pairing[a,b])
                       for a in range(len(trial.blocks)) for b in range(a+1,len(trial.blocks))
                       if trial.types[a] == trial.types[b] == 'E'):
                    continue
                states = contact_states(trial)
                if any(states[i,j] != bool(layout.adjacency[i,j])
                       for (i,j), state in initial_states.items()
                       if layout.adjacency[i,j] != 2 and state == bool(layout.adjacency[i,j])):
                    continue
            if not layout.metadata.get('native_layout') and initial_agreement>=.9 and layout_map_agreement(trial)<.9:
                continue
            e = objective(trial)
            # Keep the oracle close to its supplied frames; feasible endpoints
            # and repulsion may repair small defects without map overpacking.
            e += 2.*np.square(trial.centroids-layout.centroids).sum()
            e += 20.*np.square(trial.axes-layout.axes).sum()
            e += 80.*np.square(trial.normals-layout.normals).sum()
            if e < energy or rng.random() < np.exp(min(0., (energy - e) / temperature)):
                current, energy = trial, e
            if energy < best_e:
                best, best_e = current.copy(), energy
    best.metadata['relaxation'] = dict(sweeps=int(sweeps), seed=int(seed), energy=float(best_e))
    return best


def to_atoms(backbone):
    import biotite.structure as struc
    backbone = np.asarray(backbone, float)
    n = len(backbone)
    atoms = struc.AtomArray(4 * n)
    atoms.coord = backbone.reshape(-1, 3).astype(np.float32)
    atoms.atom_name = np.tile(['N', 'CA', 'C', 'O'], n)
    atoms.element = np.tile(['N', 'C', 'C', 'O'], n)
    atoms.res_name[:] = 'GLY'
    atoms.res_id = np.repeat(np.arange(1, n + 1), 4)
    atoms.chain_id[:] = 'A'
    atoms.hetero[:] = False
    return atoms


def write_cif(path, backbone):
    import biotite.structure.io.pdbx as pdbx
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    f = pdbx.CIFFile()
    pdbx.set_structure(f, to_atoms(backbone))
    f.write(str(path))


# KMT / Alexander closure diagnostics adapted from the independent N031 RT2 probe.
def _seg_tri(p0, p1, a, b, c):
    d = p1 - p0; e1 = b - a; e2 = c - a; h = np.cross(d, e2); det = e1 @ h
    if abs(det) < 1e-12: return False
    f = 1. / det; s = p0 - a; u = f * (s @ h)
    if u < 0 or u > 1: return False
    q = np.cross(s, e1); v = f * (d @ q)
    if v < 0 or u + v > 1: return False
    t = f * (e2 @ q)
    return 0 < t < 1

def _radial_closure(ca):
    ca = np.asarray(ca, float); c = ca.mean(0)
    far = 100. * max(np.linalg.norm(ca - c, axis=1).max(), 1.)
    e0 = c + unit(ca[0] - c, [1, 0, 0]) * far; e1 = c + unit(ca[-1] - c, [0, 1, 0]) * far
    mid = c + unit(unit(e0 - c) + unit(e1 - c), perpendicular(unit(e0 - c))) * 2 * far
    return np.vstack((ca, e1[None], mid[None], e0[None]))  # closed polygon (last connects to first)

def alexander_det(poly, rng):
    """|Delta(-1)| of the closed polygon via a random projection. 1 = unknot-like, 3 = trefoil."""
    R = _rand_rot(rng); P = poly @ R.T; n = len(P)
    segs = [(P[i], P[(i + 1) % n]) for i in range(n)]
    crossings = []  # (seg_over, t_over, seg_under, t_under)
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1): continue
            p, p2 = segs[i]; q, q2 = segs[j]
            r = p2[:2] - p[:2]; s = q2[:2] - q[:2]; den = r[0] * s[1] - r[1] * s[0]
            if abs(den) < 1e-12: continue
            qp = q[:2] - p[:2]; t = (qp[0] * s[1] - qp[1] * s[0]) / den; u = (qp[0] * r[1] - qp[1] * r[0]) / den
            if 0 < t < 1 and 0 < u < 1:
                zi = p[2] + t * (p2[2] - p[2]); zj = q[2] + u * (q2[2] - q[2])
                if zi > zj: crossings.append((i, t, j, u))
                else: crossings.append((j, u, i, t))
    m = len(crossings)
    if m == 0: return 1, 0
    unders = sorted([(c[2], c[3], k) for k, c in enumerate(crossings)])
    # arcs: arc id increments at each under-crossing along the curve; arc 0 starts after the last under-crossing (wraps)
    def arc_at(seg, t):
        # number of under-crossings strictly before (seg,t) along the curve
        cnt = sum(1 for (s_, t_, _) in unders if (s_ < seg) or (s_ == seg and t_ < t))
        return cnt % m
    M = np.zeros((m, m))
    for k, (so, to, su, tu) in enumerate(crossings):
        over = arc_at(so, to); under_in = (arc_at(su, tu) - 0) % m  # arc before the under point
        # arc_at counts unders before the point; at the under point itself the arc before it is cnt (wrap) and after is cnt+1
        under_out = (under_in + 1) % m
        M[k, over] += 2.; M[k, under_in] += -1.; M[k, under_out] += -1.
    sub = M[1:, 1:]
    return int(round(abs(np.linalg.det(sub)))) if m > 1 else 1, m

def _rand_rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)], [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)], [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])

def kmt_cyclic(points):
    pts = [np.asarray(p, float) for p in points]
    changed = True
    while changed and len(pts) > 3:
        changed = False; i = 0
        while i < len(pts) and len(pts) > 3:
            n = len(pts); a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
            skip = {(i - 2) % n, (i - 1) % n, i, (i + 1) % n}
            hit = False
            for k in range(n):
                if k in skip: continue
                if _seg_tri(pts[k], pts[(k + 1) % n], a, b, c): hit = True; break
            if not hit: del pts[i]; changed = True
            else: i += 1
    return len(pts)

def knot_report(ca, seed=0):
    rng = np.random.default_rng(seed)
    radial = _radial_closure(ca); direct = np.asarray(ca, float)
    dr = [alexander_det(radial, rng)[0] for _ in range(5)]; dd = [alexander_det(direct, rng)[0] for _ in range(5)]
    return dict(det_radial=max(set(dr), key=dr.count), det_direct=max(set(dd), key=dd.count), kmt_radial=kmt_cyclic(radial), kmt_direct=kmt_cyclic(direct))


def self_check(layout, backbone, repo=None):
    """Return measured failures; never replace SS with the conditioning SS.

    Fixed-block adherence is the harness metric; label_* separately invokes
    get_ss_adj with P-SEA-derived blocks and compares inter-block rectangles.
    """
    ca = backbone[:, 1]
    cb = ec.reconstruct_cb(backbone[:, 0], ca, backbone[:, 2])
    blocks = layout.blocks
    pos, neg, correct = [], [], []
    lbl = ec.load_label_module(repo)
    actual_ss, actual_adj = lbl.get_ss_adj(to_atoms(backbone), loop_adj=False)
    label_agree = []
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            state = layout.adjacency[i, j]
            if state == 2:
                continue
            idx = np.ix_(ec.block_idx(blocks, i), ec.block_idx(blocks, j))
            contact = bool(ec.pairwise(cb[ec.block_idx(blocks, i)], cb[ec.block_idx(blocks, j)]).min() < 6.)
            (pos if state == 1 else neg).append(contact)
            correct.append(contact == bool(state))
            # Unknown labels count as disagreement, rather than being dropped.
            label_agree.append(float(np.mean(actual_adj[idx] == state)))
    d = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    dist = ec.pairwise(ca)
    nonlocal_mask = np.triu(abs(np.arange(len(ca))[:, None] - np.arange(len(ca))) >= 2, 1)
    # Physical gates include rare native cis peptides and tightly packed turns.
    # Preserve stricter sketch-quality counts separately from hard violations.
    clashes = int((dist[nonlocal_mask] < 2.8).sum())
    u=ca[:-1]-backbone[:-1,2];v=backbone[1:,0]-backbone[:-1,2]
    w=ca[1:]-backbone[1:,0]
    normal1=np.cross(u,v);normal2=np.cross(-v,w)
    cosine=(normal1*normal2).sum(1)/np.maximum(np.linalg.norm(normal1,axis=1)*np.linalg.norm(normal2,axis=1),1e-12)
    # cis has the two CA on the same side of the peptide C-N axis.
    cis=(cosine>np.cos(np.deg2rad(30.))) & (np.linalg.norm(v,axis=1)<1.5)
    bad_bonds = int(((abs(d - 3.8) > .3) & ~(cis & (abs(d-2.95)<.3))).sum())
    report = dict(sketch_recovery=float(np.mean(pos)) if pos else None,
                  sketch_violation=float(np.mean(neg)) if neg else None,
                  map_agreement=float(np.mean(correct)) if correct else None,
                  label_map_agreement=float(np.mean(label_agree)) if label_agree else None,
                  ss_accuracy=float(np.mean(np.asarray(actual_ss) == layout.ss)),
                  bad_ca_bonds=bad_bonds, ca_clashes=clashes,
                  ca_distance_min=float(d.min()) if len(d) else None,
                  ca_distance_max=float(d.max()) if len(d) else None,
                  geometry_ok=not bad_bonds and not clashes, handedness=handedness(layout),
                  koga=koga_values(layout), axis_repulsion=axis_repulsion(layout))
    report.update(shape_metrics(layout, ca))
    sse = np.zeros(len(ca), bool)
    for block in blocks:
        sse[block['start']:block['stop']] = True
    sse_ca=ca[sse]
    sse_rg=float(np.sqrt(np.square(sse_ca-sse_ca.mean(0)).sum(1).mean())) if len(sse_ca) else report['rg']
    report.update(sse_rg_n04=sse_rg/len(ca)**.4,rg_layout_ratio=report['rg']/max(sse_rg,1e-12),n_blocks=len(blocks))
    report.update(layout_rg_limits(len(blocks)))
    loop_bonds = d[~(sse[:-1] & sse[1:])]
    report['loop_ca_distance_max'] = float(loop_bonds.max()) if len(loop_bonds) else None
    atom_ids=np.repeat(np.arange(len(ca)),4)
    heavy=backbone.reshape(-1,3)
    hmask=np.triu(abs(atom_ids[:,None]-atom_ids[None,:])>=2,1)
    hd=ec.pairwise(heavy)[hmask]
    report['heavy_clashes']=int((hd<2.6).sum())
    # Local i,i+2 O...N contacts include native tight turns down to 1.78 A;
    # long-range backbone contacts must still clear 2.2 A.
    hrows,hcols=np.triu_indices(len(heavy),1)
    hsep=atom_ids[hcols]-atom_ids[hrows]
    hdist=np.linalg.norm(heavy[hrows]-heavy[hcols],axis=1)
    report['hard_heavy_clashes']=int(((hsep==2)&(hdist<1.7) | (hsep>=3)&(hdist<2.2)).sum())
    report['ca_contacts_below_3p6']=int((dist[nonlocal_mask]<3.6).sum())
    report['cis_peptides']=np.flatnonzero(cis).tolist()
    report['heavy_distance_min']=float(hd.min()) if len(hd) else None
    peptide_cn=np.linalg.norm(backbone[:-1,2]-backbone[1:,0],axis=1)
    report['peptide_cn_min']=float(peptide_cn.min()) if len(peptide_cn) else None
    report['peptide_cn_max']=float(peptide_cn.max()) if len(peptide_cn) else None
    tri=~(sse[:-2]&sse[1:-1]&sse[2:])
    v=ca[:-2]-ca[1:-1];w=ca[2:]-ca[1:-1]
    cos=(v*w).sum(1)/np.maximum(np.linalg.norm(v,axis=1)*np.linalg.norm(w,axis=1),1e-12)
    angles=np.rad2deg(np.arccos(np.clip(cos,-1.,1.)))[tri]
    report['loop_angle_min']=float(angles.min()) if len(angles) else None
    report['loop_angle_max']=float(angles.max()) if len(angles) else None
    report['bad_loop_angles']=int(((angles<75.)|(angles>155.)).sum())
    planes=hull_planes(ca[sse])
    depth=((ca[~sse]@planes[:,:3].T+planes[:,3]).max(1) if len(planes) else np.empty(0))
    report['loop_hull_penetrations']=int((depth < -1e-3).sum())
    report['loop_hull_depth_max']=float(max(0.,-depth.min())) if len(depth) else 0.
    report['threading_residues']=threading_report(layout,ca)
    report['threading']=len(report['threading_residues'])
    report['knot']=knot_report(ca)
    # Either closure detecting a nontrivial knot is a conservative rejection;
    # no open-chain closure is a proof of absence of every possible knot.
    report['knotted']=any(report['knot'][k]!=1 for k in ('det_radial','det_direct'))
    report['geometry_ok']=bool(report['geometry_ok'] and not report['hard_heavy_clashes'] and
                               not report['bad_loop_angles'] and
                               not report['threading'] and not report['knotted'])
    # SSE-only half-separation is a diagnostic, never a loop-routing target.
    mid=len(blocks)//2
    report['sse_half_sep']=(float(np.linalg.norm(np.concatenate([ca[ec.block_idx(blocks,i)] for i in range(mid)]).mean(0)-
                          np.concatenate([ca[ec.block_idx(blocks,i)] for i in range(mid,len(blocks))]).mean(0))) if mid else None)
    report['strand_polarity'] = {f'{i}-{j}': (1 if layout.axes[i] @ layout.axes[j] >= 0 else -1)
                                  for i in range(len(blocks)) for j in range(i+1,len(blocks))
                                  if layout.types[i] == layout.types[j] == 'E'}
    report['sheets'] = layout.metadata.get('sheets')
    report['n_sheets'] = len(report['sheets']) if report['sheets'] is not None else None
    report['barrels'] = layout.metadata.get('barrels')
    layers=layout.metadata.get('sheets',[])
    report['layer_plane_distance']=(float(abs((layout.centroids[layers[1]].mean(0)-
        layout.centroids[layers[0]].mean(0))@layout.normals[layers[0][0]]))
        if len(layers)==2 and not any(layout.metadata.get('barrels',[])) else None)
    report['strand_pairs'] = {f'{i}-{j}': strand_pair(ca, blocks, i, j)
                              for i in range(len(blocks)) for j in range(i + 1, len(blocks))
                              if blocks[i]['type'] == blocks[j]['type'] == 'E'}
    return report


def sketch_acceptance(report, target=None):
    """Ship only fixed-block agreement, geometry and compactness successes."""
    native = (target or {}).get('native', {})
    half_limit = (1.15*float(native['half_sep']) if native.get('half_sep') is not None
                  else 2.15*float((target or {}).get('n_res', report['n_res']))**.4 if target else None)
    reasons = []
    if not report['geometry_ok']:
        reasons.append('geometry')
    for gate in ('bad_ca_bonds','ca_clashes','bad_loop_angles'):
        if report.get(gate,0):reasons.append(gate)
    agreement=report['map_agreement']
    if agreement is None or not np.isfinite(agreement) or agreement < .9:
        reasons.append('fixed_block_agreement')
    if not np.isfinite(report['rg_n04']):
        reasons.append('rg_nonfinite')
    sse_rg=report.get('sse_rg_n04')
    if sse_rg is not None and (not np.isfinite(sse_rg) or sse_rg>report['sse_rg_n04_limit']):
        reasons.append('layout_compactness')
    ratio=(report['rg_n04']/sse_rg if sse_rg is not None and sse_rg>0 else report.get('rg_layout_ratio'))
    if ratio is not None:report['rg_layout_ratio']=ratio
    if ratio is not None and (not np.isfinite(ratio) or
            not report['rg_layout_ratio_min']<=ratio<=report['rg_layout_ratio_max']):
        reasons.append('loop_rg_relative_to_layout')
    # Native full-chain Rg is diagnostic. Layout compactness and loop expansion
    # have different, population-calibrated gates and distinct failure reasons.
    if report.get('hard_heavy_clashes',report.get('heavy_clashes',0)):
        reasons.append('heavy_clashes')
    if report.get('knotted',False):
        reasons.append('knot')
    if report.get('threading',0):
        reasons.append('threading')
    if report['handedness']['left'] or report['handedness']['right'] != report['handedness']['n']:
        reasons.append('handedness')
    # The SSE-compactness limit is already in the report as 'sse_rg_n04_limit' (set by the caller from
    # layout_rg_limits and tested above); the 'rg_n04_limit' entry was a misleadingly named copy of it
    # (red-team b0r5 section 7.5) and is no longer written.  Nothing in the repo reads 'rg_n04_limit'.
    report.update(accepted=not reasons, failure_reasons=reasons, half_sep_limit=half_limit)
    return not reasons


def repair_packing(layout):
    """Optimise CB contacts with physical planes, registers and pleat phases.

    Normal phase flips stay on the local sheet normal. The layout objective
    controls layer spacing; no loop coordinates or evaluation metrics enter.
    """
    current=layout.copy()
    def cost(lay):
        terms=objective(lay,return_terms=True)
        return (100.*terms['map']+100.*terms['axes']+100.*terms.get('sheet_spacing',0.)+
                terms['loop']+terms['clash']+terms.get('heavy',0.)+terms.get('layer_spacing',0.))
    for _ in range(2):
        for i in np.flatnonzero(current.types=='E'):
            trial=current.copy();trial.normals[i]*=-1
            if cost(trial)<cost(current) and not sse_clashes(trial):current=trial
        start=current.centroids.copy()
        def energy(x):
            trial=current.copy();trial.centroids=x.reshape(-1,3)
            sheet_normals(trial)
            return cost(trial)+.05*np.square(trial.centroids-start).sum()
        result=minimize(energy,start.ravel(),method='L-BFGS-B',
                        options=dict(maxiter=30,ftol=1e-9,maxls=30))
        trial=current.copy();trial.centroids=result.x.reshape(-1,3);sheet_normals(trial)
        trial=project_sheet_bounds(project_axis_floors(trial))
        if cost(trial)<cost(current) and constraints_ok(trial):current=trial
    return current


def project_contacts(layout, iterations=80):
    """Repair violated contact inequalities; no force inside the 6 A cut.

    A spring-to-input plus a hinge can stop just outside a strict inequality.
    Small rigid translations project those residuals, while native pair
    topology, polarity and steric floors remain protected by the caller.
    """
    current=layout.copy()
    for _ in range(iterations):
        worst=0.
        for i in range(len(current.blocks)):
            for j in range(i+1,len(current.blocks)):
                if current.adjacency[i,j] != 1:
                    continue
                x,y=placed_sse(current,i),placed_sse(current,j)
                cbx=ec.reconstruct_cb(x[:,0],x[:,1],x[:,2]); cby=ec.reconstruct_cb(y[:,0],y[:,1],y[:,2])
                distances=ec.pairwise(cbx,cby); a,b=np.unravel_index(distances.argmin(),distances.shape)
                distance=distances[a,b]
                if distance >= 6.:
                    error=min(distance-5.98,.25); worst=max(worst,error)
                    delta=unit(cby[b]-cbx[a])*error/2
                    current.centroids[i]+=delta; current.centroids[j]-=delta
        current=project_sheet_bounds(project_axis_floors(current))
        if worst == 0:
            break
    return current


def build_sketch(layout, *, sweeps=50, seed=0, repo=None, loop_attempts=4):
    relaxed = relax_layout(layout, sweeps=sweeps, seed=seed) if sweeps else layout.copy()
    if sweeps and not layout.metadata.get('native_layout'):
        candidate=repair_packing(relaxed)
        from layout_planner_sa import _topology_ok, layout_map_agreement
        minimum_agreement=min(.9,layout_map_agreement(relaxed))
        if _topology_ok(candidate) and layout_map_agreement(candidate)>=minimum_agreement:relaxed=candidate
        candidate=project_contacts(relaxed)
        if _topology_ok(candidate) and layout_map_agreement(candidate)>=minimum_agreement and objective(candidate,return_terms=True)['heavy']<1e-4:
            relaxed=candidate
    if sweeps and layout.metadata.get('native_layout'):
        projected=project_contacts(relaxed)
        ca=np.zeros((len(projected.ss),3))
        for k, block in enumerate(projected.blocks):
            ca[block['start']:block['stop']]=placed_sse(projected,k)[:,1]
        same_pairs=all(strand_pair(ca,projected.blocks,i,j)['paired'] == bool(layout.pairing[i,j])
                       for i in range(len(projected.blocks)) for j in range(i+1,len(projected.blocks))
                       if projected.types[i] == projected.types[j] == 'E')
        if constraints_ok(projected) and same_pairs and not sse_clashes(projected,ca):
            relaxed=projected
    if loop_attempts<1:
        raise ValueError('positive loop_attempts required')
    initial=assemble(relaxed)
    history=[];best=None;best_score=np.inf
    for attempt in range(loop_attempts):
        route_seed=int(seed)+104729*attempt
        xyz=compact_loops(relaxed,initial,seed=route_seed)
        report=self_check(relaxed,xyz,repo=repo)
        history.append(dict(seed=route_seed,geometry_ok=report['geometry_ok'],
                            bad_ca_bonds=report['bad_ca_bonds'],heavy_clashes=report['heavy_clashes'],
                            bad_loop_angles=report['bad_loop_angles'],threading=report['threading']))
        # Retry only geometric failures. Never rank routes by Rg, native shape,
        # half-separation or a map/evaluation score; the SSE layout stays fixed.
        score=(100*(report['bad_ca_bonds']+report['ca_clashes']+report.get('hard_heavy_clashes',report['heavy_clashes'])+
                    report['threading']+int(report['knotted']))+
               report['bad_loop_angles'])
        if score<best_score:
            best_score=score;best=(xyz,report)
        if report['geometry_ok']:
            break
    xyz,report=best
    report['loop_routing_attempts']=history
    relaxed.metadata['loop_routing_attempts']=history
    refresh_strand_pairs(relaxed,xyz[:,1])
    return relaxed,xyz,report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layout', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--layout-out')
    p.add_argument('--report')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--sweeps', type=int, default=50)
    p.add_argument('--repo')
    p.add_argument('--strict', action='store_true', help='fail on geometry or map disagreement < .9')
    args = p.parse_args(argv)
    layout, xyz, report = build_sketch(Layout.load(args.layout), sweeps=args.sweeps, seed=args.seed, repo=args.repo)
    write_cif(args.out, xyz)
    if args.layout_out:
        layout.save(args.layout_out)
    if args.report:
        ec.write_json(args.report, report)
    print(json.dumps(report))
    if args.strict and not sketch_acceptance(report):
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
