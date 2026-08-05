import torch


def test_apply_cfg_defaults_masks_ss_adj_and_ratio():
    from rfd3.model.ss_adj_cfg import apply_cfg_defaults

    features = {
        "ss_ratio": torch.tensor([0.25]),
        "ss_labels": torch.tensor([0, 1, 2], dtype=torch.long),
        "adj_matrix": torch.ones(3, 3),
    }
    out = apply_cfg_defaults(features, ["ss_ratio", "ss_labels", "adj_matrix"])

    assert torch.equal(out["ss_ratio"], torch.tensor([-1.0]))
    assert torch.equal(out["ss_labels"], torch.full((3,), 3, dtype=torch.long))
    assert torch.count_nonzero(out["adj_matrix"]) == 0


def test_apply_cfg_defaults_leaves_unlisted_features_unchanged():
    from rfd3.model.ss_adj_cfg import apply_cfg_defaults

    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    adjacency = torch.ones(3, 3)
    out = apply_cfg_defaults(
        {"ss_labels": labels, "adj_matrix": adjacency},
        ["ref_atomwise_rasa"],
    )

    assert torch.equal(out["ss_labels"], labels)
    assert torch.equal(out["adj_matrix"], adjacency)
