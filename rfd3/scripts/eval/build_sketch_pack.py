#!/usr/bin/env python
"""Build a B0 sketched evaluation pack; existing sampler/scorer need no changes.

Example: build_sketch_pack.py --base-pack specs_v2.tar.gz --out sketches.tar.gz
         --n-layouts 8 --t-sketch 80 --seed 20261014
Use N=4 BATCH=4 to obtain 32 designs per parent arm with eight layout keys.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
from pathlib import Path
import re
import shutil
import tempfile
import zlib

import numpy as np

import build_specs as bs
import eval_common as ec
from layout_planner_sa import plan_layout
from sketch_builder import build_sketch, write_cif, sketch_acceptance
from layout_common import shape_metrics


@contextmanager
def unpack_base(source):
    """Private workspace: never modify the supplied unpacked pack."""
    with tempfile.TemporaryDirectory(prefix='harmonia_layout_') as work:
        root = Path(work)
        source = Path(source)
        if source.is_dir():
            # Copy files, not symlinks that could escape the pack on writing.
            for path in source.rglob('*'):
                if path.is_symlink():
                    raise ValueError(f'symlink in base pack: {path}')
            shutil.copytree(source, root, dirs_exist_ok=True)
        else:
            ec.safe_extract(str(source), str(root))
        # A prior render must not leak machine-local absolute paths into the tar.
        (root / 'inputs.json').unlink(missing_ok=True)
        yield root


def tensor_path(root, value):
    if not isinstance(value, str) or not value.startswith(ec.SPEC_ROOT_TOKEN + '/'):
        raise ValueError('pack tensor paths must start with @SPEC_ROOT@/')
    path = (root / value[len(ec.SPEC_ROOT_TOKEN) + 1:]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'unsafe or missing pack file: {value}')
    return path


def check_key(key):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', key) or key in ('.', '..'):
        raise ValueError(f'unsafe pack key: {key!r}')


def partial_spec(parent, cif_rel, t_sketch, metadata):
    spec = copy.deepcopy(parent)
    spec.pop('length', None)  # input CIF sets token count (N029 oracle convention)
    spec.update(dialect=2, input=f'{ec.SPEC_ROOT_TOKEN}/{cif_rel}', partial_t=float(t_sketch),
                select_fixed_atoms=False, select_unfixed_sequence=True)
    he = spec.setdefault('extra', {}).setdefault('harmonia_eval', {})
    he['layout_sketch'] = copy.deepcopy(metadata)
    return spec


def _build_layout_key(task):
    """Independent CPU work unit; the parent alone writes the output archive."""
    ss,adj,target,seeds,steps,sweeps,repo,planner,ckpt,flow_steps,temperature,flow_signs=task
    flow=None
    if planner=='flow':
        from layout_planner_flow import load_checkpoint, sample_layout
        flow=load_checkpoint(ckpt)
    attempts=[]
    for attempt,planner_seed in enumerate(seeds):
        try:
            layout=(plan_layout(ss,adj,seed=planner_seed,steps=steps) if flow is None else
                    sample_layout(flow,ss,adj,seed=planner_seed,steps=flow_steps,temperature=temperature,signs=flow_signs))
            layout,xyz,report=build_sketch(layout,sweeps=sweeps,seed=planner_seed,repo=repo)
            report.update(shape_metrics(layout,xyz[:,1],target=target))
            accepted=sketch_acceptance(report,target=target)
        except (ValueError,FloatingPointError) as error:
            layout=xyz=None
            report=dict(accepted=False,failure_reasons=['construction'],error=str(error))
            accepted=False
        attempts.append(dict(attempt=attempt+1,planner_seed=planner_seed,report=copy.deepcopy(report)))
        if accepted:break
    return layout,xyz,report,attempts,planner_seed


def build_pack(base_pack, out, *, n_layouts=8, t_sketch=80., seed=20261014,
               steps=600, sweeps=50, repo=None, strict=False, max_attempts=5, planner='sa', ckpt=None, workers=1,
               flow_steps=40, temperature=.85, signs=None, signed_adj=False):
    flow = None
    checkpoint_info = None
    if planner == 'flow':
        if ckpt is None:
            raise ValueError('--ckpt required with --planner flow')
        from layout_planner_flow import load_checkpoint, sample_layout
        if signs is not None and signed_adj:
            raise ValueError('choose signs or signed_adj')
        if signs is not None and not isinstance(signs, dict):
            raise ValueError('flow signs must map parent keys to block matrices')
        if flow_steps < 1 or not .7 <= temperature <= 1.:
            raise ValueError('positive flow_steps and temperature in [0.7, 1.0] required')
        flow = load_checkpoint(ckpt)
        checkpoint_info = dict(name=Path(ckpt).name, sha256=hashlib.sha256(Path(ckpt).read_bytes()).hexdigest())
    elif planner != 'sa':
        raise ValueError('unknown planner')
    if workers<1:
        raise ValueError('positive workers required')
    if max_attempts < 1:
        raise ValueError('positive max_attempts required')
    if n_layouts < 1 or not np.isfinite(t_sketch) or t_sketch <= 0:
        raise ValueError('positive n_layouts and t_sketch required')
    with unpack_base(base_pack) as root:
        manifest = ec.read_json(str(root / 'manifest.json'))
        templates = ec.read_json(str(root / 'inputs_template.json'))
        new_manifest = copy.deepcopy(manifest)
        new_manifest['arms'], new_manifest['unconditional'] = {}, {}
        new_templates, diagnostics, failures = {}, {}, {}
        tasks=[];task_keys=[]
        for key,rec in manifest.get('arms',{}).items():
            parent=templates[key]
            if 'adj_file' not in parent:continue
            check_key(key)
            if 'ss_file' not in parent:
                raise ValueError(f'{key}: adjacency without SS cannot define block layout')
            ss=np.load(tensor_path(root,parent['ss_file']));adj=np.load(tensor_path(root,parent['adj_file']))
            flow_signs=None
            if planner=='flow':
                from layout_common import split_signed_adjacency
                if signed_adj:adj,flow_signs=split_signed_adjacency(ss,adj)
                elif signs is not None:flow_signs=signs.get(key)
            target=manifest.get('targets',{}).get(rec.get('target'))
            for index in range(n_layouts):
                seeds=[int(seed)+zlib.crc32(key.encode())+index+attempt*n_layouts for attempt in range(max_attempts)]
                tasks.append((ss,adj,target,seeds,steps,sweeps,repo,planner,ckpt,flow_steps,temperature,flow_signs))
                task_keys.append((key,index))
        if workers==1:
            results=list(map(_build_layout_key,tasks))
        else:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                results=list(pool.map(_build_layout_key,tasks))
        built=dict(zip(task_keys,results))
        for group in ('arms', 'unconditional'):
            for key, rec in manifest.get(group, {}).items():
                check_key(key)
                parent = templates[key]
                conditioned = group == 'arms' and 'adj_file' in parent
                if conditioned and 'ss_file' not in parent:
                    raise ValueError(f'{key}: adjacency without SS cannot define block layout')
                ss = np.load(tensor_path(root, parent['ss_file'])) if conditioned else None
                adj = np.load(tensor_path(root, parent['adj_file'])) if conditioned else None
                for index in range(n_layouts):
                    subkey = f'{key}__{"l" if conditioned else "r"}{index:02d}'
                    newrec = copy.deepcopy(rec)
                    newrec['key'] = subkey
                    spec = copy.deepcopy(parent)
                    if conditioned:
                        layout,xyz,report,attempts,planner_seed=built[key,index]
                        accepted=report['accepted']
                        diagnostics[subkey] = dict(report, attempts=attempts)
                        if not accepted:
                            failures[subkey] = dict(parent_key=key, attempts=attempts)
                            continue
                        rel = f'sketch/{subkey}.cif'
                        write_cif(root / rel, xyz)
                        layout.save(root / f'sketch/{subkey}.npz')
                        metadata = dict(planner=planner, t_sketch=float(t_sketch), layout_id=index,
                                        planner_seed=planner_seed, sketch_recovery=report['sketch_recovery'],
                                        sketch_violation=report['sketch_violation'], handedness=report['handedness'],
                                        half_sep=report['half_sep'], rg_n04=report['rg_n04'],
                                        loop_ca_distance_max=report['loop_ca_distance_max'],
                                        map_agreement=report['map_agreement'], geometry_ok=report['geometry_ok'],
                                        sse_half_sep=report.get('sse_half_sep'), n_sheets=report.get('n_sheets'),
                                        heavy_clashes=report.get('heavy_clashes'), knotted=report.get('knotted'),
                                        threading=report.get('threading'), bad_ca_bonds=report.get('bad_ca_bonds'),
                                        loop_hull_penetrations=report.get('loop_hull_penetrations'),
                                        strand_polarity=report.get('strand_polarity'),
                                        sse_rg_n04=report.get('sse_rg_n04'),rg_layout_ratio=report.get('rg_layout_ratio'))
                        if checkpoint_info is not None:
                            metadata['flow_checkpoint'] = checkpoint_info
                            metadata.update(flow_steps=flow_steps, temperature=temperature,
                                            signed_pairs=layout.metadata.get('signed_pairs', []),
                                            map_projection_steps=layout.metadata.get('map_projection_steps', 0))
                        newrec['layout_sketch'] = copy.deepcopy(metadata)
                        spec = partial_spec(parent, rel, t_sketch, metadata)
                    new_templates[subkey] = spec
                    new_manifest[group][subkey] = newrec
        new_manifest['layout_failures'] = {k: dict(attempts=len(v['attempts']),
                                                  failure_reasons=v['attempts'][-1]['report']['failure_reasons'])
                                           for k,v in failures.items()}
        fields=('accepted','failure_reasons','bad_ca_bonds','loop_ca_distance_max','ca_clashes',
                'heavy_clashes','hard_heavy_clashes','bad_loop_angles','threading','knotted',
                'loop_hull_penetrations','map_agreement','strand_polarity','n_sheets',
                'half_sep','sse_half_sep','rg_n04','sse_rg_n04','rg_layout_ratio','sheets')
        gate_report={key:dict({field:report.get(field) for field in fields},attempts=len(report['attempts']))
                     for key,report in diagnostics.items()}
        ec.write_json(str(root/'layout_gate_report.json'),gate_report)
        new_manifest['layout_gate_report']='layout_gate_report.json'
        new_manifest['layout_arm_counts']={key:sum(k.startswith(key+'__') for k in new_templates)
                                           for key in templates}
        new_manifest['layout_complete']=not failures
        ec.write_json(str(root / 'layout_failures.json'), failures)
        ec.write_json(str(root / 'manifest.json'), new_manifest)
        ec.write_json(str(root / 'inputs_template.json'), new_templates)
        ec.write_json(str(root / 'sketch_checks.json'), diagnostics)
        bs.pack_dir(str(root), str(out))
    return new_manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-pack', required=True, help='tar.gz or unpacked directory')
    p.add_argument('--out', required=True)
    p.add_argument('--planner', choices=['sa', 'flow'], default='sa', help='block layout planner')
    p.add_argument('--ckpt', help='required flow checkpoint')
    p.add_argument('--flow-steps', type=int, default=40)
    p.add_argument('--temperature', type=float, default=.85, help='flow initial-noise temperature')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--signs', help='flow: JSON mapping parent keys to symmetric block sign matrices')
    group.add_argument('--signed-adj', action='store_true', help='flow: interpret signed strand adjacency')
    p.add_argument('--n-layouts' , type=int, default=8)
    p.add_argument('--t-sketch', type=float, default=80.)
    p.add_argument('--seed', type=int, default=20261014)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--sweeps', type=int, default=50)
    p.add_argument('--repo')
    p.add_argument('--strict', action='store_true', help='write diagnostics and exit nonzero if any requested key is exhausted')
    p.add_argument('--workers',type=int,default=8,help='independent CPU processes; one per layout key')
    p.add_argument('--max-attempts', type=int, default=5, help='draws per layout key before reporting failure')
    a = p.parse_args(argv)
    manifest = build_pack(a.base_pack, a.out, n_layouts=a.n_layouts, t_sketch=a.t_sketch,
                          seed=a.seed, steps=a.steps, sweeps=a.sweeps, repo=a.repo, strict=a.strict, max_attempts=a.max_attempts, planner=a.planner, ckpt=a.ckpt,workers=a.workers,flow_steps=a.flow_steps,temperature=a.temperature,
                          signs=ec.read_json(a.signs) if a.signs else None,signed_adj=a.signed_adj)
    print(f'wrote {a.out}: {len(manifest["arms"])} conditioned/SS keys, '
          f'{len(manifest["unconditional"])} unconditional keys')
    return 2 if a.strict and manifest['layout_failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
