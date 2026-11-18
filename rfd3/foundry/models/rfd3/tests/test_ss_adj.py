"""
Integration tests for SS/ADJ fold conditioning.

Tests verify:
  1. ss_adj.py: correct SS labels, ADJ shape/dtype/symmetry/diagonal
  2. ss_adj_adapters.py: zero-init, masked output zeroing, shape
  3. FeaturizeSSADJ: transform produces correct feats keys and shapes
  4. RFD1 compatibility: loop rows/cols=0, values in {0,1,2}, no SS_MASKED
  5. LMDB round-trip: pack/unpack roundtrip preserves values
  6. CFG dropout: correct modes with new 40/20/10/10/10/10 distribution
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Synthetic structure builder (no PDB required)
# ---------------------------------------------------------------------------


def _make_helix_strand_loop_structure(
    n_helix: int = 10,
    n_strand: int = 8,
    n_loop: int = 12,
) -> "biotite.structure.AtomArray":
    """Build a minimal biotite AtomArray with known SS composition."""
    from biotite.structure import AtomArray, BondList

    L = n_helix + n_strand + n_loop
    n_atoms = L  # one Cα per residue for simplicity

    aa = AtomArray(n_atoms)
    aa.atom_name = np.array(["CA"] * n_atoms)
    aa.element = np.array(["C"] * n_atoms)
    aa.chain_id = np.array(["A"] * n_atoms)
    aa.res_id = np.arange(1, n_atoms + 1)
    aa.res_name = np.array(["ALA"] * n_atoms)
    aa.hetero = np.array([False] * n_atoms)

    # Helix coords: compact ~3.8Å spacing along x
    coords = np.zeros((n_atoms, 3), dtype=np.float32)
    for i in range(n_helix):
        coords[i] = [i * 3.8, 0, 0]
    # Strand: shifted but close enough for inter-block contact
    for i in range(n_strand):
        coords[n_helix + i] = [i * 3.8 + 1.0, 5.0, 0]
    # Loop: far away
    for i in range(n_loop):
        coords[n_helix + n_strand + i] = [i * 3.8 + 100.0, 100.0, 0]
    aa.coord = coords

    aa.bonds = BondList(n_atoms)
    return aa


def _make_ss_labels_np(n_helix: int, n_strand: int, n_loop: int) -> np.ndarray:
    """Build ground-truth SS labels array."""
    from foundry.data.ss_adj import SS_HELIX, SS_LOOP, SS_STRAND

    ss = np.array(
        [SS_HELIX] * n_helix + [SS_STRAND] * n_strand + [SS_LOOP] * n_loop,
        dtype=np.float32,
    )
    return ss


# ---------------------------------------------------------------------------
# Tests: foundry.data.ss_adj
# ---------------------------------------------------------------------------


class TestSSAdjCore:
    @pytest.fixture(autouse=True)
    def imports(self):
        from foundry.data.ss_adj import (
            SS_HELIX,
            SS_LOOP,
            SS_MASKED,
            SS_STRAND,
            _adj_blocks_from_coords,
            _segment_blocks,
            get_adj_matrix_standard,
            pack_adj,
            pack_ss,
            unpack_adj,
            unpack_ss,
        )

        self.SS_HELIX = SS_HELIX
        self.SS_STRAND = SS_STRAND
        self.SS_LOOP = SS_LOOP
        self.SS_MASKED = SS_MASKED
        self._segment_blocks = _segment_blocks
        self._adj_blocks_from_coords = _adj_blocks_from_coords
        self.get_adj_matrix_standard = get_adj_matrix_standard
        self.pack_ss = pack_ss
        self.unpack_ss = unpack_ss
        self.pack_adj = pack_adj
        self.unpack_adj = unpack_adj

    def test_ss_constants(self):
        assert self.SS_HELIX == 0.0
        assert self.SS_STRAND == 1.0
        assert self.SS_LOOP == 2.0
        assert self.SS_MASKED == 3.0

    def test_segment_blocks_single(self):
        ss = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 2.0], dtype=np.float32)
        blocks = self._segment_blocks(ss)
        assert len(blocks) == 3
        np.testing.assert_array_equal(blocks[0], [0, 1, 2])
        np.testing.assert_array_equal(blocks[1], [3, 4])
        np.testing.assert_array_equal(blocks[2], [5])

    def test_segment_blocks_empty(self):
        assert self._segment_blocks(np.array([], dtype=np.float32)) == []

    def test_adj_standard_shape_dtype(self):
        n_h, n_e, n_l = 8, 6, 10
        ss = _make_ss_labels_np(n_h, n_e, n_l)
        L = n_h + n_e + n_l
        ca_coords = np.zeros((L, 3), dtype=np.float32)
        adj = self._adj_blocks_from_coords(ca_coords, ss, loop_adj=False)
        assert adj.shape == (L, L)
        assert adj.dtype == np.float32

    def test_adj_standard_rfd1_compatible(self):
        """Loop rows and columns must all be 0.0 in standard mode."""
        n_h, n_e, n_l = 8, 6, 10
        ss = _make_ss_labels_np(n_h, n_e, n_l)
        L = n_h + n_e + n_l
        ca_coords = np.random.default_rng(0).random((L, 3)).astype(np.float32) * 5
        adj = self._adj_blocks_from_coords(ca_coords, ss, loop_adj=False)

        loop_mask = ss == self.SS_LOOP
        assert adj[loop_mask].sum() == 0.0, "Loop rows must be all 0.0"
        assert adj[:, loop_mask].sum() == 0.0, "Loop cols must be all 0.0"

    def test_adj_symmetric(self):
        n_h, n_e, n_l = 8, 6, 10
        ss = _make_ss_labels_np(n_h, n_e, n_l)
        L = n_h + n_e + n_l
        ca_coords = np.random.default_rng(1).random((L, 3)).astype(np.float32) * 10
        adj = self._adj_blocks_from_coords(ca_coords, ss, loop_adj=False)
        np.testing.assert_allclose(adj, adj.T)

    def test_adj_diagonal_zero(self):
        n_h, n_e, n_l = 8, 6, 10
        ss = _make_ss_labels_np(n_h, n_e, n_l)
        L = n_h + n_e + n_l
        ca_coords = np.zeros((L, 3), dtype=np.float32)
        adj = self._adj_blocks_from_coords(ca_coords, ss, loop_adj=False)
        np.testing.assert_array_equal(np.diag(adj), 0.0)

    def test_adj_binary_values(self):
        n_h, n_e, n_l = 8, 6, 10
        ss = _make_ss_labels_np(n_h, n_e, n_l)
        L = n_h + n_e + n_l
        ca_coords = np.zeros((L, 3), dtype=np.float32)
        adj = self._adj_blocks_from_coords(ca_coords, ss, loop_adj=False)
        unique_vals = set(np.unique(adj).tolist())
        assert unique_vals <= {0.0, 1.0}

    def test_rfd1_does_not_fill_within_block(self):
        ss = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        coords = np.zeros((4, 3), dtype=np.float32)
        adj = self._adj_blocks_from_coords(coords, ss, cb_coords=coords)
        assert adj[:2, :2].sum() == 0.0
        assert adj[2:, 2:].sum() == 0.0

    def test_rfd1_single_cb_contact_fills_interblock_rectangle(self):
        ss = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        cb = np.array(
            [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [5.9, 0.0, 0.0], [100.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        adj = self._adj_blocks_from_coords(cb, ss, cb_coords=cb)
        np.testing.assert_array_equal(adj[:2, 2:], np.ones((2, 2)))
        np.testing.assert_array_equal(adj[2:, :2], np.ones((2, 2)))

    def test_rfd1_cutoff_is_strictly_less_than_six(self):
        ss = np.array([0.0, 1.0], dtype=np.float32)
        cb = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=np.float32)
        adj = self._adj_blocks_from_coords(cb, ss, cb_coords=cb)
        assert adj.sum() == 0.0

    def test_loop_extension_requires_three_ca_pairs(self):
        ss = np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float32)
        ca = np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 5.0], [0.0, 1.0, 5.0]],
            dtype=np.float32,
        )
        adj = self._adj_blocks_from_coords(
            ca, ss, loop_adj=True, cb_coords=ca, min_loop_contact_pairs=3
        )
        assert adj[:2, 2:].sum() == 4.0
        adj = self._adj_blocks_from_coords(
            ca, ss, loop_adj=True, cb_coords=ca, min_loop_contact_pairs=5
        )
        assert adj.sum() == 0.0

    def test_rfd1_cb_reconstruction_formula(self):
        from foundry.data.ss_adj import _reconstruct_cb

        backbone = np.array(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]],
            dtype=np.float32,
        )
        expected = np.array([[1.5689693, -0.5441217, -0.57910144]], dtype=np.float32)
        np.testing.assert_allclose(_reconstruct_cb(backbone), expected, atol=1e-7)

    def test_ss_lmdb_roundtrip(self):
        ss = _make_ss_labels_np(8, 6, 10)
        packed = self.pack_ss(ss)
        recovered = self.unpack_ss(packed)
        np.testing.assert_array_equal(ss, recovered)
        assert recovered.dtype == np.float32

    def test_adj_lmdb_roundtrip(self):
        n_h, n_e, n_l = 8, 6, 10
        ss = _make_ss_labels_np(n_h, n_e, n_l)
        L = n_h + n_e + n_l
        ca_coords = np.zeros((L, 3), dtype=np.float32)
        adj = self._adj_blocks_from_coords(ca_coords, ss, loop_adj=False)
        packed = self.pack_adj(adj)
        recovered = self.unpack_adj(packed)
        np.testing.assert_array_equal(adj, recovered)
        assert recovered.dtype == np.float32

    def test_adj_lmdb_float16_no_loss(self):
        """Binary {0,1} ADJ stored as float16 has zero precision loss."""
        adj_np = np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        packed = self.pack_adj(adj_np)
        recovered = self.unpack_adj(packed)
        np.testing.assert_array_equal(adj_np, recovered)

    def test_ss_masked_not_in_packed_ss(self):
        """SS_MASKED (3.0) must never appear in pack_ss (only runtime sentinel)."""
        ss = _make_ss_labels_np(8, 6, 10)  # only 0,1,2
        assert (
            self.SS_MASKED not in ss
        ), "ss labels from _make_ss_labels_np should not contain SS_MASKED"
        packed = self.pack_ss(ss)
        recovered = self.unpack_ss(packed)
        assert self.SS_MASKED not in recovered


# ---------------------------------------------------------------------------
# Tests: ss_adj_adapters.py
# ---------------------------------------------------------------------------


class TestSSAdjAdapters:
    def test_ss_adapter_zero_init(self):
        from rfd3.model.layers.ss_adj_adapters import SSAdapter

        adapter = SSAdapter(c_s=64)
        weights = adapter.embed.weight.data
        assert (weights == 0).all(), "SSAdapter must be zero-initialized"

    def test_ss_adapter_output_shape(self):
        from rfd3.model.layers.ss_adj_adapters import SSAdapter

        adapter = SSAdapter(c_s=64)
        labels = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)
        out = adapter(labels)
        assert out.shape == (5, 64)

    def test_ss_adapter_masked_is_zero(self):
        """Label==3 (SS_MASKED) must produce zero output regardless of weights."""
        from rfd3.model.layers.ss_adj_adapters import SSAdapter

        adapter = SSAdapter(c_s=64)
        # Manually set non-zero weights
        with torch.no_grad():
            adapter.embed.weight.data.fill_(1.0)

        labels = torch.tensor([3, 3, 3], dtype=torch.long)
        out = adapter(labels)
        assert (out == 0).all(), "Masked tokens must produce zero output"

    def test_ss_adapter_non_masked_nonzero_after_training(self):
        """After setting non-zero weights, non-masked labels produce non-zero output."""
        from rfd3.model.layers.ss_adj_adapters import SSAdapter

        adapter = SSAdapter(c_s=64)
        with torch.no_grad():
            adapter.embed.weight.data.fill_(1.0)

        labels = torch.tensor([0, 1, 2], dtype=torch.long)
        out = adapter(labels)
        assert (out != 0).all()

    def test_adj_adapter_zero_init(self):
        from rfd3.model.layers.ss_adj_adapters import ADJAdapter

        adapter = ADJAdapter(c_z=32)
        weights = adapter.embed.weight.data
        assert (weights == 0).all(), "ADJAdapter must be zero-initialized"

    def test_adj_adapter_output_shape(self):
        from rfd3.model.layers.ss_adj_adapters import ADJAdapter

        adapter = ADJAdapter(c_z=32)
        adj = torch.zeros(10, 10, dtype=torch.float32)
        out = adapter(adj)
        assert out.shape == (10, 10, 32)

    def test_adj_adapter_masked_state_is_zero(self):
        """State 2 (unspecified/masked) must produce zero output."""
        from rfd3.model.layers.ss_adj_adapters import ADJAdapter

        adapter = ADJAdapter(c_z=32)
        with torch.no_grad():
            adapter.embed.weight.data.fill_(1.0)

        adj = torch.full((4, 4), 2, dtype=torch.float32)
        out = adapter(adj)
        assert (out == 0).all(), "Masked (state=2) positions must produce zero output"

    def test_adapters_receive_gradients(self):
        from rfd3.model.layers.ss_adj_adapters import ADJAdapter, SSAdapter

        ss_adapter = SSAdapter(c_s=8)
        adj_adapter = ADJAdapter(c_z=4)
        ss = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        adj = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

        loss = ss_adapter(ss).sum() + adj_adapter(adj).sum()
        loss.backward()

        assert ss_adapter.embed.weight.grad is not None
        assert ss_adapter.embed.weight.grad[:3].abs().sum() > 0
        assert ss_adapter.embed.weight.grad[3].abs().sum() == 0
        assert adj_adapter.embed.weight.grad is not None
        assert adj_adapter.embed.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Tests: FeaturizeSSADJ transform
# ---------------------------------------------------------------------------


class TestFeaturizeSSADJ:
    @pytest.fixture
    def atom_array_with_ss(self):
        """AtomArray with is_helix_gt / is_sheet_gt / is_loop_gt annotations."""

        struct = _make_helix_strand_loop_structure(n_helix=8, n_strand=6, n_loop=10)
        L = struct.array_length()

        # Annotate like Add1DSSFeature would
        helix_gt = np.array([True] * 8 + [False] * 6 + [False] * 10)
        sheet_gt = np.array([False] * 8 + [True] * 6 + [False] * 10)
        loop_gt = np.array([False] * 8 + [False] * 6 + [True] * 10)

        struct.set_annotation("is_helix_gt", helix_gt)
        struct.set_annotation("is_sheet_gt", sheet_gt)
        struct.set_annotation("is_loop_gt", loop_gt)
        # Required by get_token_starts:
        struct.set_annotation("token_id", np.arange(L))
        # Required by get_token_starts internals:
        struct.set_annotation("label_entity_id", np.ones(L, dtype=int))
        return struct

    def test_produces_ss_labels_feats(self, atom_array_with_ss):
        from rfd3.transforms.ss_adj_transforms import FeaturizeSSADJ

        transform = FeaturizeSSADJ(loop_adj=False)
        data = {"atom_array": atom_array_with_ss, "is_inference": False, "feats": {}}

        # Patch CFG dropout to be deterministic (no drop)
        with _mock_random_no_drop():
            out = transform(data)

        assert "ss_labels" in out["feats"]
        assert "adj_matrix" in out["feats"]

    def test_ss_labels_shape_dtype(self, atom_array_with_ss):
        from rfd3.transforms.ss_adj_transforms import FeaturizeSSADJ

        transform = FeaturizeSSADJ(loop_adj=False)
        data = {"atom_array": atom_array_with_ss, "is_inference": False, "feats": {}}

        with _mock_random_no_drop():
            out = transform(data)

        ss = out["feats"]["ss_labels"]
        assert isinstance(ss, torch.Tensor)
        assert ss.dtype == torch.long
        assert ss.ndim == 1

    def test_adj_matrix_shape_dtype(self, atom_array_with_ss):
        from rfd3.transforms.ss_adj_transforms import FeaturizeSSADJ

        transform = FeaturizeSSADJ(loop_adj=False)
        data = {"atom_array": atom_array_with_ss, "is_inference": False, "feats": {}}

        with _mock_random_no_drop():
            out = transform(data)

        adj = out["feats"]["adj_matrix"]
        assert isinstance(adj, torch.Tensor)
        assert adj.dtype == torch.float32
        assert adj.ndim == 2
        assert adj.shape[0] == adj.shape[1]

    def test_online_transform_reconstructs_cb_for_rfd1_contact(self):
        from biotite.structure import AtomArray
        from rfd3.transforms.ss_adj_transforms import FeaturizeSSADJ

        aa = AtomArray(6)
        aa.atom_name = np.array(["N", "CA", "C", "N", "CA", "C"])
        aa.element = np.array(["N", "C", "C", "N", "C", "C"])
        aa.chain_id = np.array(["A"] * 6)
        aa.res_id = np.array([1, 1, 1, 2, 2, 2])
        aa.res_name = np.array(["ALA"] * 6)
        aa.coord = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [5, 0, 0], [6, 0, 0], [6, 1, 0]],
            dtype=np.float32,
        )
        aa.set_annotation("token_id", np.array([0, 0, 0, 1, 1, 1]))
        aa.set_annotation("label_entity_id", np.ones(6, dtype=int))
        aa.set_annotation("is_helix_gt", np.array([1, 1, 1, 0, 0, 0], dtype=bool))
        aa.set_annotation("is_sheet_gt", np.array([0, 0, 0, 1, 1, 1], dtype=bool))

        with _mock_random_no_drop():
            out = FeaturizeSSADJ(loop_adj=False)(
                {"atom_array": aa, "is_inference": False, "feats": {}}
            )
        adj = out["feats"]["adj_matrix"]
        assert adj[0, 1] == 1
        assert adj[1, 0] == 1
        assert torch.diag(adj).sum() == 0

    def test_cfg_dropout_masks_all_ss(self, atom_array_with_ss):
        """In unconditional mode, all labels become SS_MASKED=3."""
        from rfd3.transforms.ss_adj_transforms import FeaturizeSSADJ

        transform = FeaturizeSSADJ()
        data = {"atom_array": atom_array_with_ss, "is_inference": False, "feats": {}}

        with _mock_random_drop_unconditional():
            out = transform(data)

        ss = out["feats"]["ss_labels"]
        assert (ss == 3).all(), "After unconditional drop, all labels must be SS_MASKED=3"

    def test_conditioning_mode_probabilities_sum_to_one(self):
        from rfd3.transforms.ss_adj_transforms import FeaturizeSSADJ

        total = (
            FeaturizeSSADJ.UNCONDITIONAL_PROB
            + FeaturizeSSADJ.SS_ONLY_PROB
            + FeaturizeSSADJ.ADJ_ONLY_PROB
            + FeaturizeSSADJ.FULL_CONDITIONING_PROB
            + FeaturizeSSADJ.INTERACTION_SITE_PROB
            + FeaturizeSSADJ.REMODELING_PROB
        )
        assert total == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("draw", "expect_ss_masked", "expect_adj_masked"),
        [
            (0.05, True, True),    # unconditional: SS=3, ADJ=2
            (0.15, False, True),   # SS-only: SS kept, ADJ=2
            (0.25, True, False),   # ADJ-only: SS=3, ADJ sparse/kept
            (0.40, False, False),  # full: both unchanged
        ],
    )
    def test_categorical_conditioning_modes(
        self, atom_array_with_ss, draw, expect_ss_masked, expect_adj_masked
    ):
        from rfd3.transforms.ss_adj_transforms import FeaturizeSSADJ

        transform = FeaturizeSSADJ()
        token_starts = np.arange(24)
        ss = torch.tensor([0] * 8 + [2] * 8 + [1] * 8, dtype=torch.long)
        adj = torch.ones((24, 24), dtype=torch.float32)
        feats = {"ss_ratio": torch.full((24, 1), 0.5)}
        with patch(
            "rfd3.transforms.ss_adj_transforms.np.random.random",
            return_value=draw,
        ):
            out_ss, out_adj = transform._apply_cfg_dropout(
                ss, adj, atom_array_with_ss, token_starts, feats
            )

        assert bool((out_ss == 3).all()) is expect_ss_masked
        assert bool((out_adj == 2).all()) is expect_adj_masked
        if draw < transform.UNCONDITIONAL_PROB:
            assert (feats["ss_ratio"] == -1).all()

    def test_remodeling_partial_masks_matching_adj(self, atom_array_with_ss):
        """Remodeling mode masks a contiguous segment; adj rows/cols for masked become 2."""
        from rfd3.transforms.ss_adj_transforms import FeaturizeSSADJ

        transform = FeaturizeSSADJ()
        token_starts = np.arange(24)
        ss = torch.tensor([0] * 8 + [2] * 8 + [1] * 8, dtype=torch.long)
        adj = torch.ones((24, 24), dtype=torch.float32)

        np.random.seed(7)
        with patch(
            "rfd3.transforms.ss_adj_transforms.np.random.random",
            return_value=0.95,  # remodeling: 0.90-1.00
        ):
            masked_ss, masked_adj = transform._apply_cfg_dropout(
                ss, adj, atom_array_with_ss, token_starts
            )

        masked = masked_ss == 3
        assert masked.any(), "Some residues should be masked"
        assert (~masked).any(), "Some residues should remain unmasked"
        assert (masked_adj[masked, :] == 2).all(), "Masked rows must be 2 (unspecified)"
        assert (masked_adj[:, masked] == 2).all(), "Masked cols must be 2 (unspecified)"
        assert torch.count_nonzero(masked_adj[~masked][:, ~masked] != 2) > 0, (
            "Unmasked block should retain contact values"
        )

    def test_interaction_site_exposes_contact_blocks(self, atom_array_with_ss):
        """Interaction-site mode reveals SS+ADJ for contact patches; masks the body."""
        from rfd3.transforms.ss_adj_transforms import FeaturizeSSADJ

        transform = FeaturizeSSADJ()
        token_starts = np.arange(24)
        # helix(8) + loop(8) + strand(8); helix-strand interact via adj
        ss = torch.tensor([0] * 8 + [2] * 8 + [1] * 8, dtype=torch.long)
        adj = torch.full((24, 24), 2, dtype=torch.float32)
        adj[:8, 16:] = 1   # helix ↔ strand contact
        adj[16:, :8] = 1

        np.random.seed(0)
        with patch(
            "rfd3.transforms.ss_adj_transforms.np.random.random",
            return_value=0.7,   # interaction-site: 0.50-0.90
        ):
            masked_ss, masked_adj = transform._apply_cfg_dropout(
                ss, adj, atom_array_with_ss, token_starts
            )

        exposed = masked_ss != 3
        body_masked = masked_ss == 3
        assert exposed.any(), "Contact-patch residues should be exposed"
        assert body_masked.any(), "Body residues (loop here) should be masked"
        # Masked residues' adj rows and cols must all be 2
        assert (masked_adj[body_masked, :] == 2).all()
        assert (masked_adj[:, body_masked] == 2).all()

    def test_interaction_site_fallback_for_no_contacts(self, atom_array_with_ss):
        """When no inter-block contacts exist, interaction-site falls back to full."""
        from rfd3.transforms.ss_adj_transforms import FeaturizeSSADJ

        transform = FeaturizeSSADJ()
        token_starts = np.arange(24)
        ss = torch.tensor([0] * 8 + [2] * 8 + [1] * 8, dtype=torch.long)
        adj = torch.zeros((24, 24), dtype=torch.float32)  # no contacts

        np.random.seed(1)
        with patch(
            "rfd3.transforms.ss_adj_transforms.np.random.random",
            return_value=0.7,
        ):
            out_ss, out_adj = transform._apply_cfg_dropout(
                ss, adj, atom_array_with_ss, token_starts
            )

        # Fallback to full conditioning: SS unchanged, adj unchanged
        assert torch.equal(out_ss, ss)
        assert torch.equal(out_adj, adj)

    def test_cfg_dropout_masks_adj_in_ss_only_mode(self, atom_array_with_ss):
        """In SS-only mode, all adj_matrix entries become 2 (unspecified)."""
        from rfd3.transforms.ss_adj_transforms import FeaturizeSSADJ

        transform = FeaturizeSSADJ()
        data = {"atom_array": atom_array_with_ss, "is_inference": False, "feats": {}}

        with _mock_random_drop_adj():
            out = transform(data)

        adj = out["feats"]["adj_matrix"]
        assert (adj == 2).all(), "In SS-only mode, adj must be all 2 (unspecified)"

    def test_inference_uses_ss_label_conditioning(self):
        """Inference mode reads ss_label_conditioning annotation."""
        from rfd3.transforms.ss_adj_transforms import FeaturizeSSADJ

        struct = _make_helix_strand_loop_structure()
        L = struct.array_length()
        struct.set_annotation("token_id", np.arange(L))
        struct.set_annotation("label_entity_id", np.ones(L, dtype=int))
        # Set inference conditioning: first 5 as helix (0), rest unconditioned (3)
        cond = np.full(L, 3, dtype=np.int64)
        cond[:5] = 0
        struct.set_annotation("ss_label_conditioning", cond)

        transform = FeaturizeSSADJ()
        data = {"atom_array": struct, "is_inference": True, "feats": {}}
        out = transform(data)

        ss = out["feats"]["ss_labels"]
        assert (ss[:5] == 0).all()
        assert (ss[5:] == 3).all()

    def test_interaction_site_blocks_broken_at_chain_boundary(self):
        """SS blocks must not merge across chains even when SS labels match."""
        from rfd3.transforms.ss_adj_transforms import FeaturizeSSADJ

        transform = FeaturizeSSADJ()

        # Two chains, each all-helix — without chain awareness these would merge
        # into one block and produce no inter-block contacts.
        ss = torch.tensor([0] * 8 + [0] * 8, dtype=torch.long)  # all helix
        # Set an inter-chain contact: residue 4 (chain A) ↔ residue 12 (chain B)
        adj = torch.zeros((16, 16), dtype=torch.float32)
        adj[:8, 8:] = 1
        adj[8:, :8] = 1

        from biotite.structure import AtomArray, BondList
        aa = AtomArray(16)
        aa.atom_name = np.array(["CA"] * 16)
        aa.element = np.array(["C"] * 16)
        aa.chain_id = np.array(["A"] * 8 + ["B"] * 8)
        aa.res_id = np.arange(1, 17)
        aa.res_name = np.array(["ALA"] * 16)
        aa.hetero = np.array([False] * 16)
        aa.coord = np.zeros((16, 3), dtype=np.float32)
        aa.bonds = BondList(16)
        aa.set_annotation("token_id", np.arange(16))
        aa.set_annotation("label_entity_id", np.ones(16, dtype=int))
        token_starts = np.arange(16)

        np.random.seed(0)
        with patch(
            "rfd3.transforms.ss_adj_transforms.np.random.random",
            return_value=0.7,  # interaction-site mode
        ):
            out_ss, out_adj = transform._apply_cfg_dropout(
                ss, adj, aa, token_starts
            )

        # With chain-aware blocks, the two helix runs are separate blocks and
        # their inter-chain contact is detected → some residues are exposed.
        exposed = out_ss != 3
        assert exposed.any(), (
            "Inter-chain contact patch must be detected when blocks are broken at "
            "chain boundary; without chain awareness both chains merge into one "
            "block and no contact pairs exist, causing fallback to full conditioning "
            "with all SS unchanged."
        )

    def test_no_inter_chain_adj_dropout(self, atom_array_with_ss):
        """Inter-chain ADJ dropout is disabled; multi-chain contacts are preserved."""
        from rfd3.transforms.ss_adj_transforms import FeaturizeSSADJ

        transform = FeaturizeSSADJ()
        token_starts = np.arange(24)
        ss = torch.tensor([0] * 8 + [2] * 8 + [1] * 8, dtype=torch.long)

        # Assign two chains to the atom_array fixture
        import copy
        aa2 = copy.copy(atom_array_with_ss)
        chain_ids = np.array(["A"] * 12 + ["B"] * 12)
        aa2.chain_id = chain_ids

        adj = torch.zeros((24, 24), dtype=torch.float32)
        adj[:12, 12:] = 1  # inter-chain contacts
        adj[12:, :12] = 1

        # Force full-conditioning mode — no dropout should touch inter-chain adj
        with patch(
            "rfd3.transforms.ss_adj_transforms.np.random.random",
            return_value=0.40,
        ):
            out_ss, out_adj = transform._apply_cfg_dropout(
                ss, adj, aa2, token_starts
            )

        assert (out_adj[:12, 12:] == 1).all(), "Inter-chain contacts must be preserved"
        assert (out_adj[12:, :12] == 1).all()


# ---------------------------------------------------------------------------
# Tests: RFD1 compatibility (loop rows/cols = 0)
# ---------------------------------------------------------------------------


class TestRFD1Compatibility:
    def test_standard_adj_has_zero_loop_rows_cols(self):
        from foundry.data.ss_adj import SS_LOOP, _adj_blocks_from_coords

        n_h, n_e, n_l = 10, 8, 15
        ss = _make_ss_labels_np(n_h, n_e, n_l)
        L = n_h + n_e + n_l
        rng = np.random.default_rng(42)
        ca_coords = rng.random((L, 3)).astype(np.float32) * 10
        adj = _adj_blocks_from_coords(ca_coords, ss, loop_adj=False)

        loop_mask = ss == SS_LOOP
        assert adj[loop_mask].sum() == 0.0
        assert adj[:, loop_mask].sum() == 0.0

    def test_ss_values_in_0_1_2(self):
        from foundry.data.ss_adj import SS_HELIX, SS_LOOP, SS_MASKED, SS_STRAND

        ss = _make_ss_labels_np(10, 8, 15)
        assert SS_MASKED not in ss
        valid = {SS_HELIX, SS_STRAND, SS_LOOP}
        assert set(np.unique(ss).tolist()) <= valid


# ---------------------------------------------------------------------------
# Context managers for deterministic CFG dropout testing
# ---------------------------------------------------------------------------


@contextmanager
def _mock_random_no_drop():
    """Force full conditioning mode (draw=0.40, in [0.30, 0.50))."""
    with patch("rfd3.transforms.ss_adj_transforms.np.random.random", return_value=0.40):
        with patch(
            "rfd3.transforms.ss_adj_transforms.np.random.uniform", return_value=0.3
        ):
            yield


@contextmanager
def _mock_random_drop_unconditional():
    """Force fully unconditional mode (draw=0.05, in [0.00, 0.10))."""
    with patch("rfd3.transforms.ss_adj_transforms.np.random.random", return_value=0.05):
        yield


@contextmanager
def _mock_random_drop_adj():
    """Force SS-only mode (draw=0.15, in [0.10, 0.20)) — adj becomes all-2."""
    with patch("rfd3.transforms.ss_adj_transforms.np.random.random", return_value=0.15):
        yield
