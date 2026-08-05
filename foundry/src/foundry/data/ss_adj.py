"""
SS/ADJ preprocessing for RFD3 fold conditioning.

Produces tensors byte-for-byte compatible with the RFD1 fold-conditioned
checkpoint.  Standard H/E adjacency follows RFdiffusion1 exactly: reconstructed C-beta,
strict 6 A cutoff, distinct SS blocks only, loops excluded. The optional
loop-aware extension separately requires three C-alpha pairs within 6 A.

Ground truth tensor format verified against:
    insulin_target_ss.pt  → Size([150]), float32, values {0.,1.,2.}
    2KL8_ss.pt            → Size([79]),  float32, values {0.,1.,2.}
    insulin_target_adj.pt → Size([150,150]), float32, values {0.,1.}
    2KL8_adj.pt           → Size([79,79]),   float32, values {0.,1.}

LMDB field layout:
    {pdb_id}_ss              → pack_ss(arr)
    {pdb_id}_adj             → pack_adj(arr)   (stored as float16)
    {pdb_id}_loop_adj_active → b'\\x01' or b'\\x00'
"""

from __future__ import annotations

import struct

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

# RFD1-compatible SS encoding
SS_HELIX = 0.0  # biotite 'a': alpha-helix (DSSP H, G, I)
SS_STRAND = 1.0  # biotite 'b': beta-strand  (DSSP E, B)
SS_LOOP = 2.0  # biotite 'c': coil / loop   (DSSP T, S, C, ' ')
SS_MASKED = 3.0  # dropout sentinel — NEVER stored in LMDB or precomputed files


# ---------------------------------------------------------------------------
# DSSP label extraction
# ---------------------------------------------------------------------------


def get_ss_labels(
    structure: AtomArray,
    chain_order: list[str] | None = None,
) -> np.ndarray:
    """Per-residue SS labels matching RFD1 encoding exactly.

    Uses biotite.structure.annotate_sse (same invocation pattern as the
    existing get_secondary_structure_types in ppi_transforms.py).

    DSSP → label mapping:
        'a' (helix)  → 0.0
        'b' (strand) → 1.0
        'c' (coil)   → 2.0

    Args:
        structure:   biotite AtomArray (full structure, all atom types)
        chain_order: chain IDs to include, in order. If None, use appearance
                     order. Must match the AtomWorks residue ordering.

    Returns:
        np.ndarray [L] float32, values in {0., 1., 2.}
    """
    if chain_order is not None:
        parts = []
        for chain_id in chain_order:
            mask = structure.chain_id == chain_id
            if not mask.any():
                raise ValueError(f"Chain '{chain_id}' not found in structure")
            parts.append(structure[mask])
        if not parts:
            raise ValueError("chain_order produced no chains")
        structure = struc.concatenate(parts)

    # annotate_sse returns one label per residue: 'a', 'b', or 'c'
    sse_per_res = struc.annotate_sse(structure)  # [L_res]

    ss = np.full(len(sse_per_res), SS_LOOP, dtype=np.float32)
    ss[sse_per_res == "a"] = SS_HELIX
    ss[sse_per_res == "b"] = SS_STRAND
    # 'c' and anything else stays SS_LOOP (already filled)

    return ss


# ---------------------------------------------------------------------------
# Backbone coordinate helpers
# ---------------------------------------------------------------------------


def _get_ca_coords(structure: AtomArray, ss_labels: np.ndarray) -> np.ndarray:
    """One Cα coordinate per residue, NaN where Cα is absent.

    Returns np.ndarray [L, 3] float32  (L == len(ss_labels)).
    """
    res_starts = struc.get_residue_starts(structure, add_exclusive_stop=True)
    n_res = len(res_starts) - 1

    if n_res != len(ss_labels):
        raise ValueError(
            f"Residue count from structure ({n_res}) != ss_labels length "
            f"({len(ss_labels)}). Ensure chain_order matches what was used "
            f"to compute ss_labels."
        )

    ca_coords = np.full((n_res, 3), np.nan, dtype=np.float32)
    for i, (start, end) in enumerate(zip(res_starts[:-1], res_starts[1:])):
        residue = structure[start:end]
        ca_mask = residue.atom_name == "CA"
        if ca_mask.any():
            ca_coords[i] = residue.coord[ca_mask][0]

    return ca_coords


def _get_backbone_coords(structure: AtomArray, ss_labels: np.ndarray) -> np.ndarray:
    """Return N, C-alpha, and C coordinates per residue as [L,3,3]."""
    res_starts = struc.get_residue_starts(structure, add_exclusive_stop=True)
    n_res = len(res_starts) - 1
    if n_res != len(ss_labels):
        raise ValueError(
            f"Residue count from structure ({n_res}) != ss_labels length "
            f"({len(ss_labels)})."
        )

    backbone = np.full((n_res, 3, 3), np.nan, dtype=np.float32)
    for i, (start, end) in enumerate(zip(res_starts[:-1], res_starts[1:])):
        residue = structure[start:end]
        for atom_i, atom_name in enumerate(("N", "CA", "C")):
            atom_mask = residue.atom_name == atom_name
            if atom_mask.any():
                backbone[i, atom_i] = residue.coord[atom_mask][0]
    return backbone


def _reconstruct_cb(backbone: np.ndarray) -> np.ndarray:
    """Reconstruct C-beta using the exact RFdiffusion1 formula."""
    n, ca, c = backbone[:, 0], backbone[:, 1], backbone[:, 2]
    b = ca - n
    c_vec = c - ca
    a = np.cross(b, c_vec)
    return (-0.57910144 * a + 0.5689693 * b - 0.5441217 * c_vec + ca).astype(np.float32)


def _segment_blocks(ss_labels: np.ndarray) -> list[np.ndarray]:
    """Partition per-residue SS labels into contiguous runs of equal type.

    Returns a list of index arrays (one per block), in order.
    """
    L = len(ss_labels)
    if L == 0:
        return []
    blocks: list[np.ndarray] = []
    i = 0
    while i < L:
        j = i + 1
        while j < L and ss_labels[j] == ss_labels[i]:
            j += 1
        blocks.append(np.arange(i, j))
        i = j
    return blocks


# ---------------------------------------------------------------------------
# ADJ matrix — standard pipeline  (Pipeline 1, RFD1 compatible)
# ---------------------------------------------------------------------------


def get_adj_matrix_standard(
    structure: AtomArray,
    ss_labels: np.ndarray,
    cb_threshold: float = 6.0,
) -> np.ndarray:
    """Binary residue-residue ADJ matrix — RFD1 compatible.

    Loop rows and columns are always 0.0.
    Returns [L, L] float32, values in {0., 1.}
    """
    ca_coords = _get_ca_coords(structure, ss_labels)
    cb_coords = _reconstruct_cb(_get_backbone_coords(structure, ss_labels))
    return _adj_blocks_from_coords(
        ca_coords,
        ss_labels,
        loop_adj=False,
        cb_coords=cb_coords,
        cb_threshold=cb_threshold,
    )


# ---------------------------------------------------------------------------
# ADJ matrix — loop submodule  (Pipeline 2)
# ---------------------------------------------------------------------------


def get_adj_matrix_loop(
    structure: AtomArray,
    ss_labels: np.ndarray,
    cb_threshold: float = 6.0,
    ca_threshold_loop: float = 6.0,
    min_loop_contact_pairs: int = 3,
) -> np.ndarray:
    """Binary ADJ matrix including loop-mediated contacts (Pipeline 2).

    Loop intra-block: 0.0. Loop inter-block: tighter threshold + min pair count.
    Returns [L, L] float32, values in {0., 1.}
    """
    ca_coords = _get_ca_coords(structure, ss_labels)
    cb_coords = _reconstruct_cb(_get_backbone_coords(structure, ss_labels))
    return _adj_blocks_from_coords(
        ca_coords,
        ss_labels,
        loop_adj=True,
        cb_coords=cb_coords,
        cb_threshold=cb_threshold,
        ca_threshold_loop=ca_threshold_loop,
        min_loop_contact_pairs=min_loop_contact_pairs,
    )


# ---------------------------------------------------------------------------
# Coord-based ADJ helpers (used when AtomArray residue counting is unreliable,
# e.g. after PadTokensWithVirtualAtoms inserts extra residues)
# ---------------------------------------------------------------------------


def _adj_blocks_from_coords(
    ca_coords: np.ndarray,
    ss_labels: np.ndarray,
    loop_adj: bool = False,
    cb_coords: np.ndarray | None = None,
    cb_threshold: float = 6.0,
    ca_threshold_loop: float = 6.0,
    min_loop_contact_pairs: int = 3,
) -> np.ndarray:
    """Build RFD1 H/E adjacency plus the optional loop-aware extension.

    H/E blocks use reconstructed C-beta coordinates and the RFD1 6 A rule.
    Loop-involved block pairs use C-alpha coordinates and require several
    close residue pairs. No block is marked adjacent to itself.
    """
    if cb_coords is None:
        cb_coords = ca_coords

    I = len(ss_labels)
    adj = np.zeros((I, I), dtype=np.float32)
    blocks = _segment_blocks(ss_labels)

    for b_idx, b1 in enumerate(blocks):
        is_loop1 = ss_labels[b1[0]] == SS_LOOP
        for b2 in blocks[b_idx + 1 :]:
            is_loop2 = ss_labels[b2[0]] == SS_LOOP

            if is_loop1 or is_loop2:
                if not loop_adj:
                    continue
                c1 = ca_coords[b1].astype(np.float32)
                c2 = ca_coords[b2].astype(np.float32)
                valid1 = ~np.isnan(c1).any(axis=1)
                valid2 = ~np.isnan(c2).any(axis=1)
                if not valid1.any() or not valid2.any():
                    continue
                distances = np.linalg.norm(
                    c1[valid1][:, None, :] - c2[valid2][None, :, :], axis=-1
                )
                if int((distances < ca_threshold_loop).sum()) < min_loop_contact_pairs:
                    continue
            else:
                c1 = cb_coords[b1].astype(np.float32)
                c2 = cb_coords[b2].astype(np.float32)
                valid1 = ~np.isnan(c1).any(axis=1)
                valid2 = ~np.isnan(c2).any(axis=1)
                if not valid1.any() or not valid2.any():
                    continue
                distances = np.linalg.norm(
                    c1[valid1][:, None, :] - c2[valid2][None, :, :], axis=-1
                )
                if not np.any(distances < cb_threshold):
                    continue

            adj[np.ix_(b1, b2)] = 1.0
            adj[np.ix_(b2, b1)] = 1.0

    np.fill_diagonal(adj, 0.0)
    adj = np.maximum(adj, adj.T)
    return adj


def get_adj_from_token_array(
    atom_array: AtomArray,
    token_starts: np.ndarray,
    ss_labels: np.ndarray,
    loop_adj: bool = False,
    **kwargs,
) -> np.ndarray:
    """Compute ADJ via token_id mapping — avoids residue-counting mismatch.

    Used by FeaturizeSSADJ in the transform pipeline after virtual atoms have
    been inserted.  token_starts is the output of get_token_starts(atom_array).

    Reconstructs C-beta from N/C-alpha/C for RFD1-compatible H/E contacts.
    Returns [I,I] float32.
    """
    I = len(token_starts)
    token_ids = atom_array.token_id[token_starts]
    token_id_to_idx: dict[int, int] = {int(tid): i for i, tid in enumerate(token_ids)}

    backbone = np.full((I, 3, 3), np.nan, dtype=np.float32)
    for atom_i, atom_name in enumerate(("N", "CA", "C")):
        atom_subset = atom_array[atom_array.atom_name == atom_name]
        for tid, coord in zip(
            atom_subset.token_id.tolist(), atom_subset.coord.tolist()
        ):
            idx = token_id_to_idx.get(int(tid))
            if idx is not None:
                backbone[idx, atom_i] = coord

    ca_coords = backbone[:, 1]
    cb_coords = _reconstruct_cb(backbone)
    return _adj_blocks_from_coords(
        ca_coords,
        ss_labels,
        loop_adj=loop_adj,
        cb_coords=cb_coords,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def get_ss_adj(
    structure: AtomArray,
    chain_order: list[str] | None = None,
    loop_adj: bool = False,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ss_labels [L] float32, adj_matrix [L,L] float32).

    loop_adj=False → standard pipeline (H/E contacts only, RFD1 compatible)
    loop_adj=True  → loop submodule (includes L contacts)
    **kwargs forwarded to the relevant get_adj_matrix_* function.
    """
    ss = get_ss_labels(structure, chain_order)
    if loop_adj:
        adj = get_adj_matrix_loop(structure, ss, **kwargs)
    else:
        adj = get_adj_matrix_standard(structure, ss, **kwargs)
    return ss, adj


# ---------------------------------------------------------------------------
# LMDB serialization
# ---------------------------------------------------------------------------


def pack_ss(arr: np.ndarray) -> bytes:
    """Serialize [L] float32 SS array → bytes.

    Format: struct.pack('<I', L) + arr.astype(float32).tobytes()
    """
    arr = np.asarray(arr, dtype=np.float32)
    L = arr.shape[0]
    return struct.pack("<I", L) + arr.tobytes()


def unpack_ss(data: bytes) -> np.ndarray:
    """Deserialize bytes → [L] float32 SS array."""
    (L,) = struct.unpack_from("<I", data, 0)
    return np.frombuffer(data[4:], dtype=np.float32).copy()


def pack_adj(arr: np.ndarray) -> bytes:
    """Serialize [L,L] float32 ADJ → bytes (stored as float16 to halve size).

    Format: struct.pack('<I', L) + arr.astype(float16).tobytes()
    Values are binary {0,1}, so float16 has no precision loss.
    """
    arr = np.asarray(arr, dtype=np.float16)
    assert arr.ndim == 2 and arr.shape[0] == arr.shape[1], "adj must be [L,L]"
    L = arr.shape[0]
    return struct.pack("<I", L) + arr.tobytes()


def unpack_adj(data: bytes) -> np.ndarray:
    """Deserialize bytes → [L,L] float32 ADJ array (cast from float16 on load)."""
    (L,) = struct.unpack_from("<I", data, 0)
    arr16 = np.frombuffer(data[4:], dtype=np.float16).copy()
    return arr16.reshape(L, L).astype(np.float32)
