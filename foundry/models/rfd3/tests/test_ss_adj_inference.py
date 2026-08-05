import numpy as np
import pytest
import torch
from rfd3.inference.input_parsing import (
    DesignInputSpecification,
    _expand_ss_adj_tensors,
    _load_ss_adj_tensor,
    _validate_adj_tensor,
    _validate_ss_tensor,
)


def test_load_rfd1_pt_tensors(tmp_path):
    ss_path = tmp_path / "target_ss.pt"
    adj_path = tmp_path / "target_adj.pt"
    torch.save(torch.tensor([0.0, 1.0, 2.0]), ss_path)
    torch.save(
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), adj_path
    )

    ss = _validate_ss_tensor(_load_ss_adj_tensor(str(ss_path), "ss"), 3, str(ss_path))
    adj = _validate_adj_tensor(
        _load_ss_adj_tensor(str(adj_path), "adj"), 3, str(adj_path)
    )

    assert ss.dtype == np.int64
    assert adj.dtype == np.float32


def test_scaffold_expansion_masks_loops_and_preserves_topology():
    ss = np.array([0, 0, 2, 2, 1, 1], dtype=np.int64)
    adj = np.zeros((6, 6), dtype=np.float32)
    adj[:2, 4:] = 1
    adj[4:, :2] = 1

    expanded_ss, expanded_adj, metadata = _expand_ss_adj_tensors(
        ss,
        adj,
        sampled_insertion="2-2",
        sampled_n="1-1",
        sampled_c="1-1",
        mask_loops=True,
    )

    np.testing.assert_array_equal(expanded_ss, np.array([3, 0, 0, 3, 3, 3, 3, 1, 1, 3]))
    assert expanded_adj[np.ix_([1, 2], [7, 8])].sum() == 4
    assert expanded_adj[np.ix_([7, 8], [1, 2])].sum() == 4
    assert expanded_adj[[0, 3, 4, 5, 6, 9]].sum() == 0
    assert metadata["sampled_insertions"] == [2]


def test_scaffold_insertions_expand_manual_mask_segments():
    ss, adj, metadata = _expand_ss_adj_tensors(
        np.array([0, 3, 1]),
        np.zeros((3, 3), dtype=np.float32),
        sampled_insertion="2-2",
        mask_loops=False,
    )
    np.testing.assert_array_equal(ss, np.array([0, 3, 3, 3, 1]))
    assert adj.shape == (5, 5)
    assert metadata["sampled_insertions"] == [2]


def test_scaffold_insertions_require_a_masked_site():
    with pytest.raises(ValueError, match="SS=3 segment"):
        _expand_ss_adj_tensors(
            np.array([0, 2, 1]),
            np.zeros((3, 3), dtype=np.float32),
            sampled_insertion=1,
            mask_loops=False,
        )


def test_ss_boundary_mask_zeros_conditioning():
    ss = np.array([0, 0, 0, 0, 2, 1, 1, 1, 1], dtype=np.int64)
    adj = np.ones((9, 9), dtype=np.float32)
    np.fill_diagonal(adj, 0)
    expanded_ss, expanded_adj, _ = _expand_ss_adj_tensors(
        ss, adj, mask_loops=True, ss_mask=1
    )
    np.testing.assert_array_equal(expanded_ss, np.array([3, 0, 0, 3, 3, 3, 1, 1, 3]))
    masked = np.array([0, 3, 4, 5, 8])
    assert expanded_adj[masked].sum() == 0
    assert expanded_adj[:, masked].sum() == 0


def test_design_spec_builds_expanded_contig_length(tmp_path):
    ss_path = tmp_path / "scaffold_ss.pt"
    adj_path = tmp_path / "scaffold_adj.pt"
    torch.save(torch.tensor([0, 0, 2, 2, 1, 1]), ss_path)
    adj = torch.zeros((6, 6))
    adj[:2, 4:] = 1
    adj[4:, :2] = 1
    torch.save(adj, adj_path)

    spec = DesignInputSpecification(
        length="6",
        ss_file=str(ss_path),
        adj_file=str(adj_path),
        mask_loops=True,
        sampled_insertion="2-2",
        sampled_N="1-1",
        sampled_C="1-1",
    )
    data = spec.to_pipeline_input("expanded")

    from atomworks.ml.utils.token import get_token_starts

    token_starts = get_token_starts(data["atom_array"])
    assert len(token_starts) == 10
    labels = data["atom_array"].ss_label_conditioning[token_starts]
    np.testing.assert_array_equal(labels, np.array([3, 0, 0, 3, 3, 3, 3, 1, 1, 3]))
    assert data["feats"]["adj_matrix"].shape == (10, 10)
    expansion = data["specification"]["extra"]["scaffold_expansion"]
    assert expansion["expanded_length"] == 10


def test_ss_tensor_length_must_match_tokens():
    with pytest.raises(ValueError, match="final token count"):
        _validate_ss_tensor(np.array([0, 1]), 3, "target_ss.pt")


def test_adj_tensor_validation():
    with pytest.raises(ValueError, match="symmetric"):
        _validate_adj_tensor(
            np.array([[0, 1], [0, 0]], dtype=np.float32),
            2,
            "target_adj.pt",
        )
    with pytest.raises(ValueError, match="diagonal"):
        _validate_adj_tensor(np.eye(2, dtype=np.float32), 2, "target_adj.pt")
