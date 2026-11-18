"""sketch_report.py: scoring-only per-key pack report.

A synthetic three-strand meander pack with known answers (order, partition, polarity, TM-score), plus a smoke
test on two parent keys of the N031 G2 pack whose strand orders were checked first hand (skipped if absent).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import build_specs as bs
import eval_common as ec
import sketch_report as sr
from layout_common import Layout
from sketch_builder import assemble, write_cif

N_RES, LOOP, SPACING = 7, 3, 4.8
POS_PAIRS = [[0, 1], [1, 2]]


def meander(centroids, axes):
    ss = [2, 2]
    for k in range(len(centroids)):
        ss += [1] * N_RES + ([2] * LOOP if k < len(centroids) - 1 else [2, 2])
    return Layout(np.array(ss), np.asarray(centroids, float), np.asarray(axes, float), [[0., 0., 1.]] * len(centroids))


# Native: antiparallel meander 0-1-2 across x.  Swapped: strands 0 and 1 exchange places (order 1-0-2) and strand 2
# is flipped, so pair 1-2 (required) and pair 0-2 lose their native polarity while 0-1 keeps it.
NATIVE = meander([[0, 0, 0], [SPACING, 0, 0], [2 * SPACING, 0, 0]], [[0, 1, 0], [0, -1, 0], [0, 1, 0]])
SWAPPED = meander([[SPACING, 0, 0], [0, 0, 0], [2 * SPACING, 0, 0]], [[0, 1, 0], [0, -1, 0], [0, -1, 0]])


@pytest.fixture
def synthetic_pack(tmp_path):
    root = tmp_path / 'pack'
    (root / 'sketch').mkdir(parents=True)
    native_xyz = assemble(NATIVE)
    ss = NATIVE.ss
    target = dict(name='meander', n_res=int(len(ss)), ss=ss.tolist(), blocks=ec.he_blocks_from_ss(ss),
                  half1_blocks=[0, 1], half2_blocks=[2], pos_pairs=POS_PAIRS, neg_pairs=[])
    arms, checks = {}, {}
    for lid, xyz, lay, agreement in (('l00', native_xyz, NATIVE, 1.), ('l01', assemble(SWAPPED), SWAPPED, .95)):
        key = f'meander__full__{lid}'
        write_cif(root / f'sketch/{key}.cif', xyz)
        lay.save(root / f'sketch/{key}.npz')
        arms[key] = dict(target='meander', arm='full', key=key, pos_pairs=POS_PAIRS, layout_sketch={})
        checks[key] = dict(accepted=True, map_agreement=agreement, failure_reasons=[], attempts=[])
    checks['meander__full__l02'] = dict(accepted=False, map_agreement=.8, failure_reasons=['geometry'], attempts=[])
    manifest = dict(spec_version=2, targets={'meander': target}, arms=arms, unconditional={},
                    layout_failures={'meander__full__l02': {}}, layout_complete=False)
    ec.write_json(str(root / 'manifest.json'), manifest)
    ec.write_json(str(root / 'sketch_checks.json'), checks)
    np.savez(root / 'refs.npz', meander_ca=native_xyz[:, 1])
    out = tmp_path / 'pack.tar.gz'
    bs.pack_dir(str(root), str(out))
    return out


def test_synthetic_pack_report(tmp_path, synthetic_pack):
    out, detail = tmp_path / 'report.csv', tmp_path / 'report.json'
    assert sr.main(['--pack', str(synthetic_pack), '--out', str(out), '--json', str(detail), '--label', 'syn']) == 0
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 1 and list(rows[0]) == list(sr.CSV_FIELDS)
    row = rows[0]
    assert (row['pack'], row['key'], row['target'], row['arm']) == ('syn', 'meander__full', 'meander', 'full')
    assert (row['n_layouts'], row['n_requested']) == ('2', '3')          # the failed l02 slot has no sketch
    assert float(row['map_min']) == .95 and abs(float(row['map_mean']) - .975) < 1e-9
    assert row['n_strands'] == '3' and row['native_sheets'] == '[0-1-2]'
    assert (row['partition_native'], row['order_native'], row['order_and_partition_native']) == ('2', '1', '1')
    assert row['sketch_orders'] == '[0-1-2]:1|[1-0-2]:1'
    assert row['pos_pairs'] == '0-1;1-2'
    assert (row['pos_pair_polarity_matches'], row['pos_pair_polarity_comparisons']) == ('3', '4')
    assert row['pos_pair_polarity_detail'] == '0-1:2/2;1-2:1/2'
    assert (row['all_pair_polarity_matches'], row['all_pair_polarity_comparisons']) == ('4', '6')
    assert float(row['tm_max']) > .99 and float(row['tm_min']) < float(row['tm_max'])   # l00 is the native itself
    assert 0. <= float(row['mirror_tm_mean']) <= 1. and float(row['rmsd_mean']) >= 0.
    assert row['note'] == ''
    data = json.loads(detail.read_text())
    assert data['meander__full__l00']['orders'][0]['match'] and data['meander__full__l00']['tm'] > .99
    assert data['meander__full__l01']['orders'][0]['sketch'] == [1, 0, 2]
    assert data['meander__full__l01']['required_pairs'] == {'0-1': True, '1-2': False}
    assert data['__native__meander']['manifest_halves'] == [[0, 1], [2]]


def test_canonical_orders():
    assert sr.canonical([3, 2, 1]) == [1, 2, 3] and sr.canonical([0, 6, 4, 2]) == [0, 6, 4, 2]
    assert sr.same_order([5, 0, 3, 2], [2, 3, 0, 5]) and not sr.same_order([0, 6, 4, 2], [2, 0, 4, 6])
    assert sr.canonical([3, 0, 1, 2], cyclic=True) == [0, 1, 2, 3] == sr.canonical([2, 1, 0, 3], cyclic=True)
    assert sr.same_order([1, 2, 3, 0], [3, 2, 1, 0], cyclic=True) and not sr.same_order([0, 2, 1, 3], [0, 1, 2, 3], cyclic=True)


G2 = Path('.agent-research/results/N031-v2-p1-layout-first-generation-native-blo/codex_b0r5/specs_n031_g2_specsv2.tar.gz')


def test_g2_pack_known_orders(tmp_path):
    pack = Path(ec.REPO_DEFAULT) / G2
    if not pack.exists():
        pytest.skip(f'G2 pack unavailable: {pack}')
    out = tmp_path / 'g2.csv'
    assert sr.main(['--pack', str(pack), '--out', str(out),
                    '--keys', 'ferredoxin_1urn__full', 'rossmann_3chy__sparse8']) == 0
    rows = {r['key']: r for r in csv.DictReader(open(out))}
    assert set(rows) == {'ferredoxin_1urn__full', 'rossmann_3chy__sparse8'}
    fer, ros = rows['ferredoxin_1urn__full'], rows['rossmann_3chy__sparse8']
    # First-hand N031 checks (notebook item, red-team b0r5 section 6b): the ferredoxin full-map sketches carry the
    # native 2-3-0-5 order in 8/8; the Rossmann sparse8 sketches are the parallel-sheet fallback 0-6-4-2 (7/8) or
    # 0-4-2-6 (1/8) although every one of them has map agreement 1.0.
    assert fer['native_sheets'] == '[2-3-0-5]'
    assert (fer['n_layouts'], fer['order_native'], fer['order_and_partition_native']) == ('8', '8', '8')
    assert fer['pos_pair_polarity_matches'] == fer['pos_pair_polarity_comparisons'] == '32'
    assert ros['native_sheets'] == '[2-0-4-6]'
    assert (ros['n_layouts'], ros['order_native'], ros['map_min']) == ('8', '0', '1.000')
    assert ros['sketch_orders'] == '[0-6-4-2]:7|[0-4-2-6]:1'
