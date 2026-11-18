import numpy as np
import pytest
from biotite.structure import AtomArray

from rfd3.transforms.ss_adj_transforms import RejectShortMultichainLocalTokenExamples


def _example(n_tokens: int, n_chains: int) -> dict:
    atom_array = AtomArray(n_tokens)
    atom_array.set_annotation("token_id", np.arange(n_tokens, dtype=np.int64))
    atom_array.set_annotation("res_id", np.arange(n_tokens, dtype=np.int64))
    atom_array.set_annotation(
        "chain_iid",
        np.asarray([chr(ord("A") + (i % n_chains)) for i in range(n_tokens)]),
    )
    atom_array.set_annotation(
        "pn_unit_iid",
        np.asarray([chr(ord("A") + (i % n_chains)) for i in range(n_tokens)]),
    )
    return {"atom_array": atom_array, "example_id": "test-example"}


def test_rejects_short_four_chain_example():
    transform = RejectShortMultichainLocalTokenExamples()

    with pytest.raises(ValueError, match="4 chains but only 16 tokens"):
        transform(_example(n_tokens=16, n_chains=4))


def test_accepts_four_chain_example_at_attention_floor():
    transform = RejectShortMultichainLocalTokenExamples()

    result = transform(_example(n_tokens=32, n_chains=4))

    assert len(result["atom_array"]) == 32


def test_accepts_short_example_with_three_chains():
    transform = RejectShortMultichainLocalTokenExamples()

    result = transform(_example(n_tokens=16, n_chains=3))

    assert len(result["atom_array"]) == 16
