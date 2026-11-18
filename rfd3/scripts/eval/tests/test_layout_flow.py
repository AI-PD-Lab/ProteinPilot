"""CPU writer checks; no mirrors, parquet, scheduler or pretrained weights.

Learning checks use distinct native layouts plus perturbations. A separate
constant-triangle fixture only checks plumbing; production convergence is measured separately.
"""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest
import torch

# Tiny token graphs: set before the first parallel kernel (runtime pool changes
# after backward can stall on the shared login node). Production uses --threads.
torch.set_num_threads(1)
from scipy.spatial.transform import Rotation

import eval_common as ec
import layout_dataset as ds
import layout_planner_flow as flow
from layout_common import Layout, extract_native_layout, objective, handedness, constraints_ok
from sketch_builder import build_sketch, write_cif


@pytest.fixture(scope='module')
def natives():
    path = Path(os.environ.get('HARMONIA_LAYOUT_SPECS', ec.SPECS_TAR_DEFAULT))
    if not path.exists():
        pytest.skip(f'local native fixtures unavailable: {path}')
    with tarfile.open(path) as tf:
        manifest = json.load(tf.extractfile('manifest.json'))
        with np.load(io.BytesIO(tf.extractfile('refs.npz').read())) as refs:
            result = {}
            for name in ('sandwich_1tit', 'ferredoxin_1urn', 'bundle_2ccy'):
                ca = refs[name+'_ca'].copy()
                ss = np.array(manifest['targets'][name]['ss'])
                adj = np.load(io.BytesIO(tf.extractfile(f'tensors/{name}__full.adj.npy').read()))
                result[name] = (ca, ss, adj)
    return result


def synthetic(n=16):
    layouts = []
    for i in range(n):
        # All-alpha capacity test: conditioning varies across 16 examples.
        ss = np.r_[np.zeros(8+i%4, int), np.full(6, 2), np.zeros(9+i//4, int),
                   np.full(6, 2), np.zeros(8, int)]
        centers = np.array([[-7., -4., 0.], [7., -4., 0.], [0., 8., 0.]])*(1+i*.005)
        lay = Layout(ss, centers, [[0, 0, 1], [0, 0, -1], [0, 0, 1]])
        lay.metadata.update(source='afdb' if i < n//2 else 'pdb', chain_id=f'c{i}', seg_key='0',
                            sequence='ACDEFGHIKLMNPQRSTVWY'[i%20]*len(ss), split='train', cluster_rep=f'cluster{i}')
        layouts.append(lay)
    return layouts


@pytest.mark.parametrize('name', ['sandwich_1tit', 'ferredoxin_1urn', 'bundle_2ccy'])
def test_extract_local_native(name, natives, tmp_path):
    ca, ss, adj = natives[name]
    layouts = ds.extract_segments(ca, ss, adj, sequence='A'*len(ss), metadata=dict(source='local', chain_id=name))
    assert len(layouts) == 1
    lay = layouts[0]
    np.testing.assert_allclose(lay.centroids.mean(0), 0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(lay.axes, axis=-1), 1)
    assert np.array_equal(lay.n_res, [b['stop']-b['start'] for b in ec.he_blocks_from_ss(ss)])
    reference = extract_native_layout(ca, ss, adj)
    np.testing.assert_array_equal(lay.pairing, reference.pairing)
    np.testing.assert_array_equal(lay.orientation, reference.orientation)
    np.testing.assert_array_equal(lay.register, reference.register)
    ds.write_dataset(tmp_path/'layouts.npz', layouts)
    with np.load(tmp_path/'layouts.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['n_res'][0, :len(lay.blocks)], lay.n_res)
        assert (saved['loop_before'] >= 0).all() and (saved['loop_after'] >= 0).all()
    loaded = list(ds.read_dataset(tmp_path/'layouts.npz'))[0]
    np.testing.assert_allclose(loaded.centroids, lay.centroids, atol=1e-6)
    assert loaded.metadata == lay.metadata
    assert (tmp_path/'layouts.csv').exists()


def test_seg_keys_do_not_bridge_same_ss_run():
    ss = np.zeros(60, int)
    ca = np.c_[np.zeros(60), np.zeros(60), np.arange(60)]
    layouts = ds.extract_segments(ca, ss, np.full((60, 60), 2), seg_keys=np.r_[np.zeros(30), np.ones(30)],
                                 min_length=10, min_blocks=1)
    assert len(layouts) == 2
    assert [len(x.blocks) for x in layouts] == [1, 1]
    assert layouts[0].metadata['seg_key'] != layouts[1].metadata['seg_key']
    assert [len(x.ss) for x in layouts] == [30, 30]


def test_kmer_leakage():
    a = 'ACDEFGHIKLMNPQRSTVWY'*5
    assert ds.excluded(a, '', '', [a])
    assert ds.excluded(a[10:70], '', '', [a])
    assert ds.excluded('YYYYY', '1TIT', '', [])
    assert ds.excluded('YYYYY', '', 'sf1', [], {'sf1'})
    assert not ds.excluded('A'*80, '', '', ['G'*80])
    assert ds.kmer_identity('XXX', 'XXX') == 0


def test_merge_quotas_grouped_holdout(tmp_path, natives):
    low = synthetic(8)
    ca, ss, adj = natives['sandwich_1tit']
    beta = ds.extract_segments(ca, ss, adj)[0]
    items = []
    for source in ('afdb', 'pdb'):
        for i in range(4):
            for is_beta in (False, True):
                lay = beta.copy() if is_beta else low[i].copy()
                lay.metadata.update(source=source, chain_id=f'{source}{i}{is_beta}', seg_key='0',
                                    sequence=f'{source}{i}{is_beta}', split='candidate', cluster_rep=f'{source}{i}{is_beta}')
                items.append(lay)
    ds.write_dataset(tmp_path/'piece_00.npz', items)
    stats = ds.merge([tmp_path/'piece_00.npz'], tmp_path/'layouts.npz', afdb=8, pdb=8, holdout=2, targets={})
    assert stats['n'] == 16 and stats['beta_enriched_fraction'] == .5
    result = list(ds.read_dataset(tmp_path/'layouts.npz'))
    assert sum(x.metadata['split'] == 'val' for x in result) == 2
    with pytest.raises(ValueError, match='quota shortfall'):
        ds.merge([tmp_path/'piece_00.npz'], tmp_path/'bad.npz', afdb=80, pdb=8, holdout=2, targets={})


def test_merge_duplicate_segments_stay_in_one_split(tmp_path):
    items = synthetic(6)
    duplicate = items[0].copy(); duplicate.metadata['seg_key'] = '1'
    items.append(duplicate)
    ds.write_dataset(tmp_path/'piece_00.npz', items)
    ds.merge([tmp_path/'piece_00.npz'], tmp_path/'layouts.npz', afdb=8, pdb=6, holdout=1, allow_shortfall=True, targets={})
    result = list(ds.read_dataset(tmp_path/'layouts.npz'))
    siblings = [x for x in result if x.metadata['chain_id'] == 'c0']
    assert len(siblings) == 2 and len({x.metadata['split'] for x in siblings}) == 1


def test_forward_backward_padding_and_cap():
    layouts = synthetic(2)
    # Different block counts and residue counts exercise padding masks.
    layouts[1] = Layout(np.zeros(12, int), [[0, 0, 0]], [[0, 0, 1]])
    batch = flow.collate(layouts)
    model = flow.LayoutFlow()
    count = sum(p.numel() for p in model.parameters())
    assert 500_000 < count < 5_000_000
    predicted = model(batch['x1'], torch.ones(2)*.4, batch)
    assert predicted.shape == (2, 3, 6)
    assert (predicted[1, 1:] == 0).all()
    loss, terms = flow.flow_loss(model, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert set(terms) == {'centroid', 'axis', 'pair', 'map'}
    with pytest.raises(ValueError, match='cap'):
        flow.LayoutFlow(d=512)


def test_so3_augmentation_and_invariant_losses():
    layouts = synthetic(3)
    for lay in layouts:
        lay.adjacency[:] = 1; np.fill_diagonal(lay.adjacency, 2)
    batch = flow.collate(layouts)
    x = batch['x1'].clone(); x[..., 0] *= 1.2
    # Avoid collinear/degenerate roll fallback; all blocks have noncollinear frames.
    q = torch.tensor(Rotation.from_rotvec([.3, -.7, .4]).as_matrix(), dtype=x.dtype)
    y = x.clone(); y[..., :3] = x[..., :3] @ q; y[..., 3:] = x[..., 3:] @ q
    ref = batch['x1'].clone(); ref[..., :3] = ref[..., :3] @ q; ref[..., 3:] = ref[..., 3:] @ q
    torch.testing.assert_close(flow.invariant_pair_loss(x, batch['x1'], batch['mask']),
                               flow.invariant_pair_loss(y, ref, batch['mask']), atol=1e-6, rtol=1e-5)
    assert flow.map_hinge(x, batch).item() > 1.
    torch.testing.assert_close(flow.map_hinge(x, batch), flow.map_hinge(y, batch), atol=1e-3, rtol=1e-5)
    augmented = flow.collate(layouts, rng=np.random.default_rng(7), augment=True)
    torch.testing.assert_close(torch.cdist(augmented['x1'][..., :3], augmented['x1'][..., :3]),
                               torch.cdist(batch['x1'][..., :3], batch['x1'][..., :3]))
    # Proper rotations preserve a signed volume: reflections would reverse it.
    x = batch['x1'][0]
    y = augmented['x1'][0]
    def chirality(a):
        return torch.dot(a[1, :3]-a[0, :3], torch.linalg.cross(a[0, 3:], a[2, :3]-a[0, :3]))
    torch.testing.assert_close(chirality(x), chirality(y))


def test_map_hinge_matches_b0_exactly():
    lay = synthetic(1)[0]
    lay.adjacency[:] = 1; np.fill_diagonal(lay.adjacency, 2)
    batch = flow.collate([lay])
    normals = flow.frame_normals(batch['x1'], batch['mask'])[0].numpy()
    lay.normals = normals
    x = batch['x1'].clone().requires_grad_()
    actual = flow.map_hinge(x, batch)
    actual.backward()
    assert torch.isfinite(x.grad).all()
    assert actual.detach().item() == pytest.approx(objective(lay, return_terms=True)['map'], rel=2e-5, abs=1e-3)
    batch['adj'].fill_(2)
    assert flow.map_hinge(batch['x1'], batch).item() == 0


def test_dropout_policy():
    ca, ss = np.zeros((50, 3)), np.r_[np.ones(8), np.full(6, 2), np.ones(8), np.full(6, 2), np.ones(8), np.full(6, 2), np.ones(8)]
    lay = Layout(ss, np.zeros((4, 3)), [[0, 0, 1]]*4, adjacency=np.array([[2,1,1,0],[1,2,0,1],[1,0,2,1],[0,1,1,2]]),
                 orientation=np.array([[0,-1,1,0],[-1,0,0,1],[1,0,0,-1],[0,1,-1,0]]))
    rng = np.random.default_rng(10); counts = [0, 0, 0]; signs_seen = []
    for _ in range(2000):
        adj, sign = flow.conditioning(lay, rng, dropout=True)
        if np.array_equal(adj, lay.adjacency): counts[0] += 1
        elif (adj == 0).any(): counts[2] += 1
        else: counts[1] += 1
        assert np.all(adj[adj != 2] == lay.adjacency[adj != 2])
        assert not np.any(sign[adj != 1])
        signs_seen.extend(sign[adj == 1] != 0)
    np.testing.assert_allclose(np.array(counts)/2000, [.5, .25, .25], atol=.04)
    assert .46 < np.mean(signs_seen) < .54


@pytest.fixture(scope='module')
def plumbing_model():
    torch.manual_seed(7)
    layouts = synthetic(16)
    batch = flow.collate(layouts)
    model = flow.LayoutFlow()
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=0)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=.0002)
    generator = torch.Generator().manual_seed(23)
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        loss, _ = flow.flow_loss(model, batch, generator=generator)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.); opt.step(); schedule.step()
    predicted = flow.integrate(model, batch, seed=19, steps=40)
    rmsd = torch.sqrt(((predicted[..., :3]-batch['x1'][..., :3])*10).square().sum(-1).mean()).item()
    print(f'plumbing triangle centroid RMSD {rmsd:.6f} A')
    return model, layouts, rmsd


def test_plumbing_constant_layout(plumbing_model):
    assert np.isfinite(plumbing_model[2])


def test_sample_checkpoint_and_roundtrip(plumbing_model, tmp_path):
    model, layouts, _ = plumbing_model
    flow.save_checkpoint(tmp_path/'tiny.pt', model)
    restored = flow.load_checkpoint(tmp_path/'tiny.pt')
    lay = layouts[0]
    sample = flow.sample_layout(restored, lay.ss, lay.adjacency, seed=3)
    again = flow.sample_layout(restored, lay.ss, lay.adjacency, seed=3)
    np.testing.assert_allclose(sample.centroids, again.centroids)
    np.testing.assert_allclose(np.linalg.norm(sample.axes, axis=-1), 1)
    assert constraints_ok(sample)
    relaxed, xyz, report = build_sketch(sample, sweeps=0)
    assert len(xyz) == len(lay.ss) and np.isfinite(xyz).all()
    assert 'label_map_agreement' in report
    sample.save(tmp_path/'sample.npz'); write_cif(tmp_path/'sample.cif', xyz)
    np.testing.assert_allclose(Layout.load(tmp_path/'sample.npz').axes, sample.axes)
    assert len(ec.protein_chains(ec.read_structure(str(tmp_path/'sample.cif'), 'sample.cif'))[0]) == len(lay.ss)


def test_rule_layer_mirror_and_loop_feasibility(natives):
    ca, ss, adj = natives['ferredoxin_1urn']
    lay = extract_native_layout(ca, ss, adj)
    lay.centroids[:, 0] *= -1; lay.axes[:, 0] *= -1; lay.normals[:, 0] *= -1
    assert handedness(lay)['left'] > 0
    repaired = flow.apply_rules(lay)
    assert constraints_ok(repaired)
    assert handedness(repaired)['right'] == handedness(repaired)['n']
    projected = flow.project_loops(repaired)
    for i, (a, b) in enumerate(zip(projected.blocks, projected.blocks[1:])):
        end = projected.centroids[i]+projected.axes[i]*projected.lengths[i]/2
        start = projected.centroids[i+1]-projected.axes[i+1]*projected.lengths[i+1]/2
        assert np.linalg.norm(end-start) <= 3.4*(b['start']-a['stop']+1)+.001


@pytest.mark.parametrize('command,arguments', [('train', []), ('sample', ['--ss', 'missing', '--adj', 'missing', '--out', 'missing', '--ckpt', 'missing']),
                                              ('validate', ['--out', 'missing', '--ckpt', 'missing'])])
def test_cli_dry_run(command, arguments):
    script = Path(flow.__file__)
    result = subprocess.run([sys.executable, str(script), command, *arguments, '--dry-run'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['cmd'] == command


def test_cli_train_sample_validate_and_flow_pack(tmp_path, plumbing_model):
    import build_sketch_pack as packer
    model, layouts, _ = plumbing_model
    flow.save_checkpoint(tmp_path/'tiny.pt', model)
    # An all-alpha map is intentionally unconstrained by chirality/Koga and exercises
    # the real pack gate, including acceptance, checkpoint identity and retained CIF/NPZ output.
    lay = layouts[0]
    np.save(tmp_path/'ss.npy', lay.ss); np.save(tmp_path/'adj.npy', lay.adjacency)
    root = tmp_path/'pack'; (root/'tensors').mkdir(parents=True)
    known_zero = np.zeros_like(lay.adjacency); np.fill_diagonal(known_zero, 2)
    np.save(root/'tensors/a.ss.npy', lay.ss); np.save(root/'tensors/a.adj.npy', known_zero)
    ec.write_json(str(root/'inputs_template.json'), {'a': dict(length=len(lay.ss), ss_file='@SPEC_ROOT@/tensors/a.ss.npy', adj_file='@SPEC_ROOT@/tensors/a.adj.npy')})
    ec.write_json(str(root/'manifest.json'), dict(arms={'a': dict(key='a', target='synthetic', arm='full')}, targets={}))
    result = subprocess.run([sys.executable, str(Path(packer.__file__)), '--base-pack', str(root), '--out', str(tmp_path/'flow.tgz'),
                             '--planner', 'flow', '--ckpt', str(tmp_path/'tiny.pt'), '--n-layouts', '1', '--max-attempts', '1', '--sweeps', '0'],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    with tarfile.open(tmp_path/'flow.tgz') as tf:
        checks = json.load(tf.extractfile('sketch_checks.json'))
        assert 'a__l00' in checks and len(checks['a__l00']['attempts']) == 1
        assert checks['a__l00']['accepted'] and checks['a__l00']['geometry_ok']
        assert checks['a__l00']['map_agreement'] == 1.
        manifest = json.load(tf.extractfile('manifest.json'))
        metadata = manifest['arms']['a__l00']['layout_sketch']
        assert metadata['planner'] == 'flow'
        assert len(metadata['flow_checkpoint']['sha256']) == 64
        assert metadata['map_projection_steps'] == 80
        assert not manifest['layout_failures']
        assert tf.extractfile('sketch/a__l00.cif').read()
    val = layouts[1].copy(); val.metadata['split'] = 'val'
    ds.write_dataset(tmp_path/'dataset.npz', [layouts[0], val])
    commands = [
        ['train', '--dataset', str(tmp_path/'dataset.npz'), '--ckpt', str(tmp_path/'cli.pt'), '--epochs', '1', '--max-steps', '1', '--batch-size', '1', '--val-limit', '0'],
        ['sample', '--ss', str(tmp_path/'ss.npy'), '--adj', str(tmp_path/'adj.npy'), '--ckpt', str(tmp_path/'tiny.pt'), '--out', str(tmp_path/'cli_sample.npz'), '--sweeps', '0'],
        ['validate', '--dataset', str(tmp_path/'dataset.npz'), '--ckpt', str(tmp_path/'tiny.pt'), '--out', str(tmp_path/'metrics.json'), '--limit', '1', '--draws', '2']]
    for command in commands:
        result = subprocess.run([sys.executable, str(Path(flow.__file__)), *command, '--threads', '1'], capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout+result.stderr
    assert (tmp_path/'cli_sample.cif').exists()
    metrics = json.loads((tmp_path/'metrics.json').read_text())
    assert metrics[-1]['requested_draws'] == 2
    assert len([r for r in metrics if 'draw' in r]) == 2
    assert all('handedness_rate' in r for r in metrics if r.get('valid'))


def test_dataset_cli_and_sbatch_syntax():
    script = Path(ds.__file__)
    for cmd in ('extract', 'merge'):
        result = subprocess.run([sys.executable, str(script), cmd, '--help'], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0
    result = subprocess.run(['bash', '-n', str(script.with_suffix('.sbatch'))], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_kabsch_no_reflection():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    rotation = Rotation.from_rotvec([.5, -.2, .7]).as_matrix()
    assert flow.kabsch_rmsd(points @ rotation+7, points) < 1e-10
    reflected = points.copy(); reflected[:, 0] *= -1
    assert flow.kabsch_rmsd(reflected, points) > .1


def test_candidate_shards_filters_and_no_overlap(monkeypatch):
    from types import SimpleNamespace
    import pandas as pd
    af = pd.DataFrame([dict(uniprot_id=f'u{i}', n_residues=100 if i != 0 else 400,
                            mean_plddt=90 if i != 1 else 70, weight=1/(i+1)**.5,
                            offset_data=i*512, size=100) for i in range(24)])
    pdb = pd.DataFrame([dict(pdb_id=f'p{i}', q_pn_unit_iid='A_1', q_pn_unit_type=6,
                             q_pn_unit_sequence_length=100, q_pn_unit_is_polymer=True) for i in range(12)])
    index = pd.DataFrame([dict(pdb_id=f'p{i}', offset_data=i*512, size=100) for i in range(12)])
    monkeypatch.setattr(pd, 'read_parquet', lambda path, **kw: af.copy() if path == 'afdb' else pdb.copy())
    seen = set()
    for piece in range(3):
        args = SimpleNamespace(seed=7, piece=piece, pieces=3, afdb_manifest='afdb', pdb_manifest='pdb')
        tasks = list(ds.candidates(args, index))
        ids = {(source, row['id']) for source, row in tasks}
        assert not seen.intersection(ids)
        seen.update(ids)
    assert len(seen) == 34
    assert ('afdb', 'u0') not in seen and ('afdb', 'u1') not in seen


def test_native_worker_cif_and_training_label_path(tmp_path, natives):
    from sketch_builder import backbone_from_ca
    ca, ss, adj = natives['bundle_2ccy']
    lay = extract_native_layout(ca, ss, adj)
    # Actual CIF parser, chain reader, training P-SEA/ADJ and frame extractor.
    # Labels are recomputed from native CA with approximate N/C/O, no mirror required.
    path = tmp_path/'native.cif'
    write_cif(path, backbone_from_ca(ca))
    raw = path.read_bytes()
    task = ('afdb', dict(offset_data=0, size=len(raw), id='local', mean_plddt=90))
    old = ds._WORK.copy()
    ds._WORK.update(files={'afdb': io.BytesIO(raw)}, targets=[], cath=set(), lbl=ec.load_label_module())
    try:
        layouts, reason = ds._extract_one(task)
    finally:
        ds._WORK.clear(); ds._WORK.update(old)
    assert layouts, reason
    assert layouts[0].metadata['source'] == 'afdb'
    assert len(layouts[0].metadata['sequence']) == len(layouts[0].ss)
    assert layouts[0].metadata['mean_plddt'] == 90


def test_target_pack_validation_source(natives):
    layouts = list(flow.native_pack_layouts(os.environ.get('HARMONIA_LAYOUT_SPECS', ec.SPECS_TAR_DEFAULT)))
    names = {x.metadata['target'] for x in layouts}
    assert {'sandwich_1tit', 'ferredoxin_1urn', 'bundle_2ccy'} <= names
    assert all(x.metadata['split'] == 'val' for x in layouts)


def test_rule_layer_retains_already_valid_strand_twist():
    angle = np.deg2rad(20.)
    lay = Layout(np.r_[np.ones(6, int), np.full(3, 2), np.ones(6, int)],
                 [[-2.4, 0, 0], [2.4, 0, 0]], [[0, 0, 1], [0, np.sin(angle), -np.cos(angle)]],
                 adjacency=np.array([[2, 1], [1, 2]]))
    repaired = flow.apply_rules(lay)
    np.testing.assert_allclose(repaired.axes, lay.axes, atol=1e-10)
    assert constraints_ok(repaired)


def real_variants(natives, count=16):
    """Three actual CA natives plus coordinate/loop-length perturbations.

    Topologies, SSE counts and contacts differ. Extra terminal loop residues
    identify each perturbation through real sequence context, never a sample ID.
    """
    rng = np.random.default_rng(511)
    base = [extract_native_layout(*natives[name]) for name in natives]
    result = []
    for i in range(count):
        native = base[i % 3]
        variant = i // 3
        ss = np.r_[np.full(variant*2, 2, dtype=int), native.ss]
        centers = native.centroids*(1 + .065*variant)
        if variant:
            centers = centers + rng.normal(0, .25*variant, centers.shape)
        centers -= centers.mean(0)
        out = Layout(ss, centers, native.axes, native.normals, adjacency=native.adjacency,
                     orientation=native.orientation, pairing=native.pairing)
        result.append(out)
    return result


def test_screen_60_percent_and_whole_cluster(tmp_path):
    rng = np.random.default_rng(10)
    alphabet = np.array(list('ACDEFGHIKLMNPQRSTVWY'))
    target = ''.join(rng.choice(alphabet, 200))
    candidate = list(target)
    mutate = rng.choice(200, 80, replace=False)
    for i in mutate:
        candidate[i] = rng.choice(alphabet[alphabet != candidate[i]])
    candidate = ''.join(candidate)
    assert sum(a == b for a, b in zip(candidate, target))/200 == .6
    score = ds.kmer_identity(candidate, target)
    assert .25 <= score < .4, score
    assert ds.excluded(candidate, '', '', [target])
    items = synthetic(8)
    items[0].metadata.update(sequence=candidate, cluster_rep='homologues')
    items[1].metadata.update(sequence='W'*200, cluster_rep='homologues')
    # A hit absent from legacy pieces still blocks its whole represented cluster.
    items[2].metadata['cluster_rep'] = 'old-discarded'
    kept, counts = ds.screen_global(items, {'eval': [target]}, candidate_hits=[('old-discarded', ['eval'])])
    assert not {x.metadata['chain_id'] for x in kept}.intersection({'c0', 'c1', 'c2'})
    assert counts['eval'] == dict(direct=1, clusters=2, excluded=3)
    print('60%-identity containment', score, 'excluded', counts)


def test_cluster_grouped_split_and_repeat_merge(tmp_path):
    items = synthetic(20)
    for i, item in enumerate(items):
        item.metadata['cluster_rep'] = f'cluster{i//3}'
    piece, out = tmp_path/'piece_00.npz', tmp_path/'merged.npz'
    ds.write_dataset(piece, items)
    first = ds.merge([piece], out, afdb=20, pdb=20, holdout=4, allow_shortfall=True, targets={})
    before = out.read_bytes()
    second = ds.merge([piece], out, afdb=20, pdb=20, holdout=4, allow_shortfall=True, targets={})
    assert first == second and before == out.read_bytes() and piece.exists()
    groups = {}
    for lay in ds.read_dataset(out):
        key = (lay.metadata['source'], lay.metadata['cluster_rep'])
        groups.setdefault(key, set()).add(lay.metadata['split'])
    assert all(len(splits) == 1 for splits in groups.values())


def test_piece_skip_and_incomplete_resume(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    piece = tmp_path/'piece_00.npz'
    ds.write_dataset(piece, synthetic(1))
    args = SimpleNamespace(out_dir=tmp_path, piece=0)
    # No manifests/env/imports are accessed on a completed-piece rerun.
    ds.extract(args)
    assert json.loads(capsys.readouterr().out)['skipped']
    assert ds.piece_complete(piece)
    piece.write_bytes(b'truncated')
    assert not ds.piece_complete(piece)
    # Atomic writer overwrites corrupt final and ignores a leftover temp.
    piece.with_suffix('.tmp.npz').write_bytes(b'partial')
    ds.write_dataset(piece, synthetic(2))
    assert ds.piece_complete(piece) and len(list(ds.read_dataset(piece))) == 2


def test_runtime_uncalibrated_rules_advisory(natives, monkeypatch):
    import layout_common as common
    lay = extract_native_layout(*natives['sandwich_1tit'])
    calls = []
    def values(layout):
        calls.append(1)
        return [dict(kind='EE', calibrated=False, satisfied=False),
                dict(kind='EH', calibrated=True, satisfied=False)]
    monkeypatch.setattr(common, 'koga_values', values)
    out = flow.apply_rules(lay)
    assert calls == [1]
    assert out.metadata['rule_diagnostics']['koga_skipped'] == 1
    assert not out.metadata['rule_diagnostics']['koga_satisfied']
    assert out.metadata['rule_diagnostics']['koga'] == [dict(kind='EH', calibrated=True, satisfied=False)]
    # Neither calibrated failure nor uncalibrated failure rejects or changes roll.
    np.testing.assert_allclose(out.normals, lay.normals)


def test_centering_and_per_layout_hinge_units(natives):
    layouts = real_variants(natives, 3)
    batch = flow.collate(layouts)
    noise = flow.center_state(torch.randn_like(batch['x1']), batch['mask'])
    torch.testing.assert_close(noise[..., :3].sum(1), torch.zeros(len(layouts), 3), atol=5e-7, rtol=0)
    class CenteredProbe(torch.nn.Module):
        def forward(self, state, time, features):
            torch.testing.assert_close(state[..., :3].sum(1), torch.zeros(len(layouts),3), atol=1e-6, rtol=0)
            return torch.zeros_like(state)
    flow.flow_loss(CenteredProbe(), batch, generator=torch.Generator().manual_seed(42))
    raw = []
    for layout in layouts:
        b = flow.collate([layout]); count = torch.triu(b['adj'] != 2, 1).sum().clamp_min(1)
        raw.append(flow.map_hinge(b['x1'], b)/(1000*count))
    torch.testing.assert_close(flow.map_hinge(batch['x1'], batch, normalized=True), torch.stack(raw).mean())


def test_validate_batches_and_open_pair_metrics(natives, monkeypatch):
    ca, ss, adj = natives['sandwich_1tit']
    lay = extract_native_layout(ca, ss, adj)
    # One open native-positive decision, others known. Predict no pairing.
    i, j = np.argwhere(np.triu(lay.pairing, 1))[0]
    lay.adjacency[i, j] = lay.adjacency[j, i] = 2
    sample = lay.copy(); sample.pairing[:] = False
    model = flow.LayoutFlow(d=16, layers=1, heads=2, pair_dim=4)
    calls = []
    def sampler(model, requests, **kwargs):
        calls.append(len(requests)); return [sample.copy() for _ in requests]
    monkeypatch.setattr(flow, 'sample_layouts', sampler)
    import sketch_builder
    monkeypatch.setattr(sketch_builder, 'build_sketch', lambda x, **kw: (x, None, dict(
        geometry_ok=True, map_agreement=1., label_map_agreement=1., sketch_recovery=1., sketch_violation=0.)))
    rows = flow.validate(model, [lay]*3, draws=5, batch_size=4)
    assert calls == [4, 4, 4, 3]
    scored = [r for r in rows if r.get('valid')]
    assert len(scored) == 15
    assert all(r['pairing_pairs'] == 1 and r['pairing_accuracy'] == 0 and r['pairing_jaccard'] == 0 for r in scored)


def test_resume_roundtrip_and_budget(tmp_path):
    ds.write_dataset(tmp_path/'data.npz', synthetic(7))
    script = str(Path(flow.__file__))
    def run(name, steps, *extra):
        cmd = [sys.executable, script, 'train', '--dataset', str(tmp_path/'data.npz'),
               '--ckpt', str(tmp_path/name), '--epochs', '8', '--max-steps', str(steps),
               '--batch-size', '2', '--threads', '1', '--val-limit', '0', '--save-every', '2', *extra]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout+result.stderr
        return torch.load(tmp_path/name, weights_only=True)
    uninterrupted = run('full.pt', 7)
    partial = run('part.pt', 3)
    assert partial['step'] == 3
    resumed = run('part.pt', 7, '--resume')
    assert resumed['step'] == uninterrupted['step'] == 7
    for key, value in uninterrupted['state_dict'].items():
        torch.testing.assert_close(value, resumed['state_dict'][key], atol=0, rtol=0)
    assert resumed['training']['numpy_rng'] == uninterrupted['training']['numpy_rng']
    torch.testing.assert_close(resumed['training']['torch_rng'], uninterrupted['training']['torch_rng'])
    budget = run('budget.pt', 0, '--max-minutes', '.000001')
    assert budget['step'] == 0 and not (tmp_path/'budget.pt.validation.json').exists()


def test_legacy_cluster_recovery_and_discarded_hit(tmp_path, monkeypatch):
    """The old piece is usable: no cluster metadata or exclusion records needed."""
    from types import SimpleNamespace
    import pandas as pd
    layouts = synthetic(4)
    for layout in layouts:
        layout.metadata.update(source='afdb')
        layout.metadata.pop('cluster_rep')
    manifest = pd.DataFrame([dict(uniprot_id=f'c{i}', cluster_rep=f'g{i//2}', offset_data=i, size=1)
                             for i in range(4)] + [dict(uniprot_id='old_hit', cluster_rep='g0', offset_data=4, size=1)])
    monkeypatch.setattr(pd, 'read_parquet', lambda path: manifest if path == 'manifest' else pd.DataFrame(columns=['pdb_id']))
    monkeypatch.setattr(ds, 'target_sequences', lambda *a, **kw: {'eval': ['ACDEFGHIK']})
    scanned = []
    def scan(task):
        cluster, row, path, targets = task
        scanned.append(row['uniprot_id'])
        return cluster, ['eval'] if row['uniprot_id'] == 'old_hit' else []
    class SerialPool:
        def __init__(self, workers): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def map(self, fn, tasks, **kwargs): return map(scan, tasks)
    monkeypatch.setattr(ds, 'ProcessPoolExecutor', SerialPool)
    args = SimpleNamespace(mirror_index='idx', mirror_tar='mirror', afdb_manifest='manifest', afdb_tar='tar', workers=48)
    targets, clusters, hits = ds.merge_screen_inputs(layouts, args)
    kept, counts = ds.screen_global(layouts, targets, clusters=clusters, candidate_hits=hits)
    assert 'old_hit' in scanned
    assert 'c0' not in [x.metadata['chain_id'] for x in kept]
    assert counts['eval']['excluded'] == 2


def test_distinct_native_overfit_and_tiny_generalization(natives):
    """Real known-pair learning, fresh Gaussian draws, no rule/SA assistance."""
    torch.manual_seed(7)
    layouts = real_variants(natives)
    assert len({x.ss.tobytes() for x in layouts}) == 16
    batch = flow.collate(layouts)
    assert torch.triu(batch['adj'] != 2, 1).sum() > 100
    model = flow.LayoutFlow()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=0)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1600, eta_min=.0002)
    def terms():
        with torch.no_grad():
            _, values = flow.flow_loss(model, batch, generator=torch.Generator().manual_seed(22))
        return {k: float(v) for k, v in values.items()}
    initial = terms()
    for step in range(1600):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = flow.flow_loss(model, batch)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step(); schedule.step()
        if step == 99:
            after100 = terms()
    assert all(after100[k] < initial[k] for k in initial), (initial, after100)
    value = flow.integrate(model, batch, seed=19, steps=40)
    rmsds = [flow.kabsch_rmsd(value[i, :len(l.blocks), :3].numpy()*10, l.centroids) for i, l in enumerate(layouts)]
    constants = []
    for kind in range(3):
        group = layouts[kind::3]
        mean_shape = np.mean([x.centroids for x in group], axis=0)
        constants.extend(flow.kabsch_rmsd(mean_shape, x.centroids) for x in group)
    assert np.mean(rmsds) < 1. and np.mean(rmsds) < .8*np.mean(constants), (rmsds, constants)
    # Fresh perturbations at interleaved scales / different loop lengths. These
    # are interpolation within the three native folds, not a fold-transfer claim.
    heldout = []
    for i in (0, 1, 2):
        l = layouts[i]
        out = Layout(np.r_[np.full(3, 2), l.ss], l.centroids*1.0325, l.axes, l.normals,
                     adjacency=l.adjacency, pairing=l.pairing, orientation=l.orientation)
        heldout.append(out)
    predictions = flow.integrate(model, flow.collate(heldout), seed=77)
    errors = [flow.kabsch_rmsd(predictions[i, :len(l.blocks), :3].numpy()*10, l.centroids) for i,l in enumerate(heldout)]
    gaussian = []
    rng = np.random.default_rng(77)
    for l in heldout:
        cloud = rng.normal(size=l.centroids.shape); cloud -= cloud.mean(0)
        cloud *= np.linalg.norm(l.centroids)/np.linalg.norm(cloud)
        gaussian.append(flow.kabsch_rmsd(cloud, l.centroids))
    assert np.mean(errors) < 2. and np.mean(errors) < np.mean(gaussian), (errors, gaussian)
    print('distinct native: initial', initial, '100 steps', after100,
          'train RMSD', float(np.mean(rmsds)), 'interpolation', errors, 'Gaussian', gaussian)


def test_pack_flow_controls(tmp_path, monkeypatch, natives):
    import build_sketch_pack as packer
    import sketch_builder
    ca, ss, adj = natives['sandwich_1tit']
    lay = extract_native_layout(ca, ss, adj)
    root = tmp_path/'base'; (root/'tensors').mkdir(parents=True)
    np.save(root/'tensors/ss.npy', ss); np.save(root/'tensors/adj.npy', adj)
    ec.write_json(root/'inputs_template.json', {'a': dict(ss_file='@SPEC_ROOT@/tensors/ss.npy', adj_file='@SPEC_ROOT@/tensors/adj.npy')})
    ec.write_json(root/'manifest.json', dict(arms={'a': dict(key='a', target='local')}, targets={}))
    (tmp_path/'fake.pt').write_bytes(b'checkpoint')
    monkeypatch.setattr(flow, 'load_checkpoint', lambda path: object())
    seen = []
    def sample(model, ss, adj, **kwargs):
        seen.append(kwargs); return lay
    monkeypatch.setattr(flow, 'sample_layout', sample)
    monkeypatch.setattr(packer, 'build_sketch', lambda *a, **kw: (lay, np.zeros((len(ss),4,3)), dict(
        sketch_recovery=1., sketch_violation=0., handedness={}, half_sep=0., rg_n04=1.,
        loop_ca_distance_max=3.8, map_agreement=1., geometry_ok=True)))
    monkeypatch.setattr(packer, 'shape_metrics', lambda *a, **kw: {})
    monkeypatch.setattr(packer, 'sketch_acceptance', lambda *a, **kw: True)
    signs = np.zeros_like(adj).tolist()
    result = packer.build_pack(root, tmp_path/'out.tgz', planner='flow', ckpt=tmp_path/'fake.pt',
                              flow_steps=17, temperature=.73, signs={'a': signs}, n_layouts=1)
    assert seen[0]['steps'] == 17 and seen[0]['temperature'] == .73 and seen[0]['signs'] == signs
    assert result['arms']['a__l00']['layout_sketch']['flow_steps'] == 17


def test_quick_cpu_benchmark(natives):
    """Red-team mini_train protocol, default 1200 steps; no best-of selection.

    Set HARMONIA_LAYOUT_BENCH_DATASET to the red-team's 3450-train NPZ for the
    full reproduction; absent that file, use the distinct local-native pool.
    """
    path = os.environ.get('HARMONIA_LAYOUT_BENCH_DATASET')
    train = ([x for x in ds.read_dataset(path) if x.metadata['split'] == 'train'] if path
             else real_variants(natives))
    assert train
    updates = int(os.environ.get('HARMONIA_LAYOUT_BENCH_STEPS', '1200'))
    learning_rate = float(os.environ.get('HARMONIA_LAYOUT_BENCH_LR', '2e-4'))
    assert updates >= 100 and learning_rate > 0
    torch.manual_seed(0); rng = np.random.default_rng(0)
    model = flow.LayoutFlow()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    probe = flow.collate(train[:64], rng=np.random.default_rng(22), dropout=True, augment=True)
    def measure():
        with torch.no_grad():
            _, terms = flow.flow_loss(model, probe, generator=torch.Generator().manual_seed(22))
        return {k: float(v) for k, v in terms.items()}
    initial = measure()
    logs = [dict(step=0, **initial)]
    for step in range(1, updates+1):
        batch = flow.collate([train[i] for i in rng.choice(len(train), 64, replace=len(train)<64)],
                             rng=rng, dropout=True, augment=True)
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss, _ = flow.flow_loss(model, batch)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        if step in (100, 400, 800, 1200, updates):
            logs.append(dict(step=step, **measure()))
    assert all(logs[1][k] < initial[k] for k in initial), logs[:2]
    targets = ([x for x in flow.native_pack_layouts(ec.SPECS_TAR_DEFAULT)] if path
               else [extract_native_layout(*natives[name]) for name in natives])
    rmsd, gaussian, raw_rmsd = [], [], []
    evaluation_rng = np.random.default_rng(442)
    raw_gate = os.environ.get('HARMONIA_LAYOUT_BENCH_RAW_GATE') == '1'
    for native in targets:
        raw = flow.sample_layouts(model, [(native.ss, native.adjacency, None)]*8, seed=100, rules=False)
        raw_rmsd.append(float(np.mean([flow.kabsch_rmsd(x.centroids, native.centroids) for x in raw])))
        samples = raw if raw_gate else flow.sample_layouts(model, [(native.ss, native.adjacency, None)]*8, seed=100, rules=True)
        rmsd.append(float(np.mean([flow.kabsch_rmsd(x.centroids, native.centroids) for x in samples])))
        clouds = []
        for _ in range(8):
            cloud = evaluation_rng.normal(size=native.centroids.shape); cloud -= cloud.mean(0)
            cloud *= np.linalg.norm(native.centroids)/np.linalg.norm(cloud)
            clouds.append(flow.kabsch_rmsd(cloud, native.centroids))
        gaussian.append(float(np.mean(clouds)))
    print('CPU benchmark', json.dumps(dict(train=len(train), updates=updates, lr=learning_rate, logs=logs, native_rmsd=rmsd, raw_rmsd=raw_rmsd, gaussian=gaussian)))
    # B0 red-team range is 1.5-6 A; approaching that range is a smoke gate,
    # not a claim that learned planning matches or surpasses B0 per target.
    assert np.mean(rmsd) < np.mean(gaussian) and np.mean(rmsd) < 6., (rmsd, gaussian)
    if raw_gate:
        assert np.mean(raw_rmsd) < np.mean(gaussian) and np.mean(raw_rmsd) < 6., (raw_rmsd, gaussian)


def test_sigterm_saves_and_resumes(tmp_path):
    import signal
    ds.write_dataset(tmp_path/'data.npz', synthetic(7))
    cmd = [sys.executable, str(Path(flow.__file__)), 'train', '--dataset', str(tmp_path/'data.npz'),
           '--ckpt', str(tmp_path/'term.pt'), '--epochs', '100', '--batch-size', '2',
           '--threads', '1', '--val-limit', '0']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        first = json.loads(process.stdout.readline())
        assert first['resumed_step'] == 0
        # Handler is installed before the flushed startup record.
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout+stderr
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
    checkpoint = torch.load(tmp_path/'term.pt', weights_only=True)
    assert 'optimizer' in checkpoint and 'training' in checkpoint
    result = subprocess.run([*cmd, '--resume', '--max-steps', str(checkpoint['step']+1)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    assert torch.load(tmp_path/'term.pt', weights_only=True)['step'] == checkpoint['step']+1


def test_rule_conflict_is_diagnostic_and_enabled_table_is_dynamic(natives, monkeypatch):
    import layout_common as common
    lay = extract_native_layout(*natives['sandwich_1tit'])
    # An explicit sign conflicting with the short-hairpin prior must survive.
    lay.metadata['signed_pairs'] = [[0, 1, 1]]
    monkeypatch.setattr(common, 'koga_values', lambda x: [dict(kind='EE', enabled=False, satisfied=False),
                                                        dict(kind='EH', enabled=True, satisfied=False)])
    out = flow.apply_rules(lay)
    assert out.metadata['polarity_conflict']
    assert out.metadata['rule_diagnostics']['koga_skipped'] == 1
    assert out.metadata['rule_diagnostics']['koga'][0]['kind'] == 'EH'


def test_dataset_arrays_are_decompressed_once(tmp_path, monkeypatch):
    ds.write_dataset(tmp_path/'data.npz', synthetic(5))
    # Numpy's public NpzFile class moves modules between releases.
    with np.load(tmp_path/'data.npz') as archive:
        cls = type(archive); members = set(archive.files)
    original = cls.__getitem__; counts = {}
    def read(self, key):
        counts[key] = counts.get(key, 0)+1
        return original(self, key)
    monkeypatch.setattr(cls, '__getitem__', read)
    assert len(list(ds.read_dataset(tmp_path/'data.npz'))) == 5
    assert set(counts) == members and all(n == 1 for n in counts.values())


def test_batch_sampling_unsigned_features_and_error_rows(natives, monkeypatch):
    ca, ss, adj = natives['sandwich_1tit']
    model = flow.LayoutFlow(d=16, layers=1, heads=2, pair_dim=4)
    calls = []
    original = model.forward
    def forward(x, t, batch):
        calls.append(len(x)); assert not batch['signs'].any()
        torch.testing.assert_close(x[..., :3].sum(1), torch.zeros(len(x),3), atol=2e-6, rtol=0)
        return original(x, t, batch)
    monkeypatch.setattr(model, 'forward', forward)
    outputs = flow.sample_layouts(model, [(ss, adj, None)]*3, steps=7, rules=False)
    assert len(outputs) == 3 and calls == [3]*7
    # A per-draw geometric failure is returned when validation requests it.
    def fail(layout): raise ValueError('rule diagnostic failure')
    monkeypatch.setattr(flow, 'apply_rules', fail)
    outputs = flow.sample_layouts(model, [(ss, adj, None)]*2, steps=1, return_exceptions=True)
    assert len(outputs) == 2 and all(isinstance(x, ValueError) for x in outputs)
    def failed_projection(*args, **kwargs): raise FloatingPointError('nonfinite projection')
    monkeypatch.setattr(flow, 'project_map', failed_projection)
    outputs = flow.sample_layouts(model, [(ss, adj, None)]*2, steps=1, return_exceptions=True)
    assert len(outputs) == 2 and all(isinstance(x, FloatingPointError) for x in outputs)


def test_sampling_map_projection_is_masked_and_anchored():
    lay = synthetic(1)[0]
    lay.adjacency[:] = 0; np.fill_diagonal(lay.adjacency, 2)
    batch = flow.collate([lay]); x = batch['x1'].clone(); x[..., :3] *= .1
    before = flow.map_hinge(x, batch, normalized=True)
    projected = flow.project_map(x, batch)
    after = flow.map_hinge(projected, batch, normalized=True)
    assert after < before, (before, after)
    torch.testing.assert_close(projected[..., 3:], x[..., 3:])
    torch.testing.assert_close(projected[..., :3].sum(1), torch.zeros(1,3), atol=1e-6, rtol=0)
    assert torch.linalg.vector_norm(projected[..., :3]-x[..., :3], dim=-1).max() < 3.
    batch['adj'].fill_(2)
    torch.testing.assert_close(flow.project_map(x, batch), x)
