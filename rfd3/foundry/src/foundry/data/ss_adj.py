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
# Block types.
#
# Adjacency was originally protein-only: a "block" was a run of DSSP H/E/L and a
# contact was Cbeta < 6 A. Ligands and nucleic acids have no secondary structure,
# and _get_ss_training defaults every token to SS_LOOP -- so before block types
# existed, every ligand atom and nucleotide was treated as a loop, which routed
# them into the loop branch and (with loop_adj=False) left the pair at 0, i.e.
# an ACTIVE "explicitly non-adjacent" embedding. Protein-ligand is ~30% of the
# interface training set, so that mislabelled a third of the data.
#
# These are block types, not SS labels: they never reach SSAdapter (which stays a
# 4-state helix/strand/loop/masked embedding). They only select the segmentation
# and contact rule below.
BLOCK_HELIX = 0
BLOCK_STRAND = 1
BLOCK_LOOP = 2
BLOCK_NA = 3       # nucleic acid, before ds/ss is resolved (see na_duplex_blocks)
BLOCK_LIGAND = 4   # non-polymer, branched (glycans), macrolide
# Nucleic acid split by LOCAL pairing state rather than by secondary-structure
# class. Classifying stems / hairpins / bulges / junctions needs a global SS graph,
# which pseudoknots break by construction -- and pseudoloops can span vast
# distances, so the class label says little about local geometry. A per-nucleotide
# "is it base-paired right now" test needs no graph, and a pseudoknotted helix is
# simply another stem that gets its own helical-turn blocks.
BLOCK_NA_DS = 5    # base-paired: rigid duplex, blocked by helical turn, spans both strands
BLOCK_NA_SS = 6    # unpaired: flexible (hairpin / bulge / junction / ss overhang)

# Contact rule per unordered pair of block types:
#     (atoms, cutoff_A, min_contact_pairs)
#
# `cbeta` is only meaningful between protein blocks -- it is a side-chain proxy
# for "these two elements pack", and RFD1's 6 A is calibrated for exactly that.
# Ligand and nucleic-acid contacts are direct van der Waals interactions, which
# are conventionally called at 4-5 A between heavy atoms; using 6 A there would
# mark mere neighbours as contacts and make the conditioning meaningless.
#
# Multiplicity mirrors the existing loop rule: a rigid element touching at one
# point is a contact, a floppy one brushing past is not.
_PROTEIN_BLOCKS = (BLOCK_HELIX, BLOCK_STRAND, BLOCK_LOOP)

DEFAULT_CONTACT_RULES: dict[tuple[int, int], tuple[str, float, int]] = {}
for _a in _PROTEIN_BLOCKS:
    for _b in _PROTEIN_BLOCKS:
        _loopish = BLOCK_LOOP in (_a, _b)
        DEFAULT_CONTACT_RULES[tuple(sorted((_a, _b)))] = (
            "cbeta", 6.0, 3 if _loopish else 1
        )
_NA_BLOCKS = (BLOCK_NA, BLOCK_NA_DS, BLOCK_NA_SS)
for _p in _PROTEIN_BLOCKS:
    DEFAULT_CONTACT_RULES[tuple(sorted((_p, BLOCK_LIGAND)))] = ("heavy", 4.5, 1)
for _n in _NA_BLOCKS:
    # ss is flexible, so it gets the same treatment as a protein loop: a single
    # nucleotide brushing past is not an interface, require several contacts.
    _mp = 3 if _n == BLOCK_NA_SS else 2
    for _p in _PROTEIN_BLOCKS:
        DEFAULT_CONTACT_RULES[tuple(sorted((_p, _n)))] = ("heavy", 5.0, _mp)
    DEFAULT_CONTACT_RULES[tuple(sorted((_n, BLOCK_LIGAND)))] = ("heavy", 4.5, 1)
    for _m in _NA_BLOCKS:
        _mp2 = 3 if BLOCK_NA_SS in (_n, _m) else 2
        DEFAULT_CONTACT_RULES[tuple(sorted((_n, _m)))] = ("heavy", 5.0, _mp2)
DEFAULT_CONTACT_RULES[(BLOCK_LIGAND, BLOCK_LIGAND)] = ("heavy", 4.5, 1)

# Max residues per loop sub-block. A contiguous coil run can be 25+ residues; as a
# single block one incidental contact marks the whole thing adjacent and the
# conditioning becomes uselessly coarse. H/E need no such split -- their length is
# naturally bounded.
LOOP_SUBBLOCK_MAX = 8
# Nucleotides per block for UNPAIRED nucleic acid (single strands, hairpin loops,
# bulges, junctions). Paired regions are blocked by helical turn instead -- see
# BP_PER_HELICAL_TURN.
NA_BLOCK_SIZE = 4

# Base pairs per block in a duplex. One helical turn is the natural recognition
# unit: a protein recognition helix inserts into ONE major groove, and a turn runs
# from one major groove to the next. B-DNA is ~10.5 bp/turn, A-form RNA ~11.
#
# Crucially a duplex block spans BOTH strands. A protein reads a base pair, not a
# single-strand segment, so per-chain windows would make "this helix contacts NA
# block X" refer to only half the groove.
BP_PER_HELICAL_TURN = 10


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



def block_types_from_token_array(
    atom_array, token_starts: np.ndarray, ss_labels: np.ndarray
) -> np.ndarray:
    """Per-token block type (BLOCK_*), from molecule type + secondary structure.

    Protein tokens keep their H/E/L identity; everything else is typed by molecule
    class rather than inheriting SS_LOOP from _get_ss_training's default fill.
    """
    from atomworks.enums import ChainType

    n = len(token_starts)
    out = np.empty(n, dtype=np.int64)
    ss_tok = np.asarray(ss_labels)[:n]
    out[:] = BLOCK_LOOP
    out[ss_tok == SS_HELIX] = BLOCK_HELIX
    out[ss_tok == SS_STRAND] = BLOCK_STRAND

    try:
        ctype = np.asarray(atom_array.chain_type)[token_starts]
    except AttributeError:
        return out  # no chain_type annotation -> protein-only behaviour

    na_types = {
        int(ChainType.DNA), int(ChainType.RNA),
        int(ChainType.DNA_RNA_HYBRID), int(ChainType.PEPTIDE_NUCLEIC_ACID),
    }
    lig_types = {
        int(ChainType.NON_POLYMER), int(ChainType.BRANCHED),
        int(ChainType.MACROLIDE),
    }
    out[np.isin(ctype, list(na_types))] = BLOCK_NA
    out[np.isin(ctype, list(lig_types))] = BLOCK_LIGAND
    return out


def _segment_blocks_typed(
    block_types: np.ndarray,
    chain_ids: np.ndarray | None = None,
    loop_subblock_max: int = LOOP_SUBBLOCK_MAX,
    na_block_size: int = NA_BLOCK_SIZE,
    exclude_terminal_loops: bool = True,
    skip_na: bool = False,
) -> tuple[list[np.ndarray], list[int]]:
    """Segment tokens into blocks with per-type granularity.

    Returns (blocks, types). Splits on block-type change and on chain change, then
    applies per-type sub-blocking:
      - loop: chunks of <= loop_subblock_max (a 25-residue coil as one block makes
        one incidental contact mark the whole thing adjacent)
      - NA:   chunks of na_block_size (one block per chain would mark a whole
        20-mer adjacent from a single contact)
      - ligand: left whole -- one block per entity is the right granularity
      - H/E:  left whole -- length is naturally bounded
    """
    n = len(block_types)
    if n == 0:
        return [], []
    runs: list[np.ndarray] = []
    start = 0
    for i in range(1, n + 1):
        same = (
            i < n
            and block_types[i] == block_types[i - 1]
            and (chain_ids is None or chain_ids[i] == chain_ids[i - 1])
        )
        if not same:
            runs.append(np.arange(start, i))
            start = i

    blocks: list[np.ndarray] = []
    types: list[int] = []
    for r_i, run in enumerate(runs):
        t = int(block_types[run[0]])
        if t == BLOCK_LOOP:
            if exclude_terminal_loops and (r_i == 0 or r_i == len(runs) - 1):
                continue  # disordered tails: their "contacts" are coordinate noise
            step = max(1, loop_subblock_max)
        elif t == BLOCK_NA:
            if skip_na:
                continue  # handled by na_duplex_blocks (helical turns, both strands)
            step = max(1, na_block_size)
        else:
            step = len(run)
        for j in range(0, len(run), step):
            chunk = run[j : j + step]
            if len(chunk):
                blocks.append(chunk)
                types.append(t)
    return blocks, types


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
    mask_loop_pairs: bool = True,
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
    mask_loop_pairs: bool = True,
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
                    # RFD1 MASKS these, it does not call them non-adjacent.
                    # BlockAdjacency.mask_ss_adj sets adj[~expanded_mask] = 2 for
                    # loops/insertions ("adj: block adjacency with equivalent
                    # masking as ss"). Leaving them at 0 declares "these blocks are
                    # explicitly NOT in contact" for ~half the residues in the
                    # structure, while they are frequently packed against their
                    # neighbours -- which is both wrong as a label and dangerous for
                    # any loss term that treats 0 as a supervised negative.
                    if mask_loop_pairs:
                        adj[np.ix_(b1, b2)] = 2.0
                        adj[np.ix_(b2, b1)] = 2.0
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




def na_duplex_blocks(
    atom_array,
    token_starts: np.ndarray,
    block_types: np.ndarray,
    bp_per_turn: int = BP_PER_HELICAL_TURN,
    na_block_size: int = NA_BLOCK_SIZE,
) -> tuple[list[np.ndarray], list[int]]:
    """Blocks for nucleic acid: helical turns for duplex, windows for unpaired.

    Duplex regions are segmented by HELICAL TURN and each block spans both strands,
    because the recognition unit a protein sees is one major groove face, not a
    single-strand stretch. Unpaired regions (hairpin loops, bulges, junctions,
    ssDNA/ssRNA) fall back to contiguous windows -- they have no helical geometry
    to turn-segment.

    Returns (blocks, types) with every block typed BLOCK_NA.
    """
    na_tok = np.flatnonzero(block_types == BLOCK_NA)
    if na_tok.size == 0:
        return [], []

    # residue index (into atom_array residues) -> token index
    try:
        import biotite.structure as struc

        res_starts = struc.get_residue_starts(atom_array)
        pairs = struc.base_pairs(atom_array)
    except Exception:
        pairs = None

    partner: dict[int, int] = {}
    if pairs is not None and len(pairs):
        # NOTE: biotite's docstring says base_pairs returns "indices of the
        # residues", but it actually returns ATOM indices (verified on 1BNA:
        # values up to 464 for a 24-residue, 486-atom duplex). Map atom -> token
        # by locating the enclosing token span.
        ts = np.asarray(token_starts)
        for a1, a2 in np.asarray(pairs, dtype=np.int64):
            t1 = int(np.searchsorted(ts, a1, side="right") - 1)
            t2 = int(np.searchsorted(ts, a2, side="right") - 1)
            if not (0 <= t1 < len(ts) and 0 <= t2 < len(ts)):
                continue
            if block_types[t1] == BLOCK_NA and block_types[t2] == BLOCK_NA:
                partner[t1] = t2
                partner[t2] = t1

    blocks: list[np.ndarray] = []
    btypes: list[int] = []
    paired = set(partner)

    # --- duplex: walk stems (i ascending while its partner descends) ---
    consumed: set[int] = set()
    for t in sorted(paired):
        if t in consumed:
            continue
        stem = []
        cur = t
        while (
            cur in partner
            and cur not in consumed
            and (not stem or partner.get(cur, -1) == partner[stem[-1]] - 1)
            and (not stem or cur == stem[-1] + 1)
        ):
            stem.append(cur)
            consumed.add(cur)
            consumed.add(partner[cur])
            cur += 1
        if not stem:
            continue
        for j in range(0, len(stem), max(1, bp_per_turn)):
            turn = stem[j : j + bp_per_turn]
            both = sorted(set(turn) | {partner[x] for x in turn})
            blocks.append(np.asarray(both, dtype=np.int64))
            btypes.append(BLOCK_NA_DS)

    # --- unpaired: contiguous runs, windowed ---
    unpaired = [t for t in na_tok if t not in paired]
    if unpaired:
        run = [unpaired[0]]
        for t in unpaired[1:]:
            if t == run[-1] + 1:
                run.append(t)
            else:
                for j in range(0, len(run), max(1, na_block_size)):
                    blocks.append(np.asarray(run[j : j + na_block_size], dtype=np.int64))
                    btypes.append(BLOCK_NA_SS)
                run = [t]
        for j in range(0, len(run), max(1, na_block_size)):
            blocks.append(np.asarray(run[j : j + na_block_size], dtype=np.int64))
            btypes.append(BLOCK_NA_SS)

    return blocks, btypes


def adj_from_block_types(
    blocks: list[np.ndarray],
    types: list[int],
    cb_coords: np.ndarray,
    token_atom_coords: list[np.ndarray] | None,
    n_tokens: int,
    contact_rules: dict | None = None,
    mask_unruled: bool = True,
    loop_adj: bool = False,
) -> np.ndarray:
    """Build an {0,1,2} adjacency matrix using per-pair-type contact rules.

    cb_coords          (I,3) per-token C-beta (virtual); used for 'cbeta' rules.
    token_atom_coords  list of (n_atoms_i, 3) heavy-atom coords per token; used for
                       'heavy' rules. May be None -> those pairs are masked.
    mask_unruled       pairs with no rule (or missing coords) become 2 (masked)
                       rather than 0. Masked contributes nothing through ADJAdapter;
                       0 is an ACTIVE learned "not in contact" signal, so guessing
                       0 is never safe.
    """
    rules = DEFAULT_CONTACT_RULES if contact_rules is None else contact_rules
    adj = np.zeros((n_tokens, n_tokens), dtype=np.float32)

    for i, b1 in enumerate(blocks):
        t1 = types[i]
        for j in range(i + 1, len(blocks)):
            b2, t2 = blocks[j], types[j]
            # Respect loop_adj even in the typed path. The rule table HAS loop
            # entries, so without this check turning on typed_blocks (for ligands
            # and nucleic acids) would silently also enable loop adjacency -- a much
            # larger label change, staged separately. Masked, never 0.
            if not loop_adj and BLOCK_LOOP in (t1, t2):
                if mask_unruled:
                    adj[np.ix_(b1, b2)] = 2.0
                    adj[np.ix_(b2, b1)] = 2.0
                continue
            rule = rules.get(tuple(sorted((t1, t2))))
            if rule is None:
                if mask_unruled:
                    adj[np.ix_(b1, b2)] = 2.0
                    adj[np.ix_(b2, b1)] = 2.0
                continue
            atoms, cutoff, min_pairs = rule

            if atoms == "cbeta":
                c1, c2 = cb_coords[b1], cb_coords[b2]
                v1 = ~np.isnan(c1).any(axis=1)
                v2 = ~np.isnan(c2).any(axis=1)
                if not v1.any() or not v2.any():
                    if mask_unruled:
                        adj[np.ix_(b1, b2)] = 2.0
                        adj[np.ix_(b2, b1)] = 2.0
                    continue
                d = np.linalg.norm(
                    c1[v1][:, None, :] - c2[v2][None, :, :], axis=-1
                )
                n_close = int((d < cutoff).sum())
            else:  # heavy-atom minimum over all atoms of both blocks
                if token_atom_coords is None:
                    if mask_unruled:
                        adj[np.ix_(b1, b2)] = 2.0
                        adj[np.ix_(b2, b1)] = 2.0
                    continue
                a1 = [token_atom_coords[k] for k in b1 if len(token_atom_coords[k])]
                a2 = [token_atom_coords[k] for k in b2 if len(token_atom_coords[k])]
                if not a1 or not a2:
                    if mask_unruled:
                        adj[np.ix_(b1, b2)] = 2.0
                        adj[np.ix_(b2, b1)] = 2.0
                    continue
                p1 = np.concatenate(a1, axis=0)
                p2 = np.concatenate(a2, axis=0)
                d = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=-1)
                n_close = int((d < cutoff).sum())

            if n_close >= min_pairs:
                adj[np.ix_(b1, b2)] = 1.0
                adj[np.ix_(b2, b1)] = 1.0

    # Tokens in no block at all (e.g. excluded terminal loops) are unspecified,
    # not non-adjacent.
    in_block = np.zeros(n_tokens, dtype=bool)
    for b in blocks:
        in_block[b] = True
    if mask_unruled and (~in_block).any():
        adj[~in_block, :] = 2.0
        adj[:, ~in_block] = 2.0

    np.fill_diagonal(adj, 0.0)
    return adj


def get_adj_from_token_array(
    atom_array: AtomArray,
    token_starts: np.ndarray,
    ss_labels: np.ndarray,
    loop_adj: bool = False,
    typed_blocks: bool = False,
    contact_rules: dict | None = None,
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

    if typed_blocks:
        # Molecule-type-aware path: protein keeps RFD1 C-beta/6 A, ligands and
        # nucleic acids get heavy-atom rules (see DEFAULT_CONTACT_RULES). Without
        # this everything non-protein inherits SS_LOOP and is routed through the
        # protein loop rule.
        block_types = block_types_from_token_array(atom_array, token_starts, ss_labels)
        try:
            chain_ids = atom_array.chain_iid[token_starts]
        except AttributeError:
            chain_ids = None
        blocks, types = _segment_blocks_typed(block_types, chain_ids, skip_na=True)
        na_blocks, na_types = na_duplex_blocks(atom_array, token_starts, block_types)
        blocks = blocks + na_blocks
        types = types + na_types

        # Heavy-atom coordinates grouped per token, for the 'heavy' rules.
        # Vectorised: the obvious form (a dict loop assigning tok_of_atom, then a
        # boolean mask per token) is two O(n_tokens * n_atoms) passes and measured
        # 1.29 ms/example at crop scale -- ~1.7 min per epoch of pure Python at
        # 2400 steps x 32. Sorting once and slicing is 0.07 ms, output identical.
        tok_of_atom = np.array(
            [token_id_to_idx.get(int(t), -1) for t in atom_array.token_id],
            dtype=np.int64,
        )
        is_heavy = (atom_array.element != "H") & (tok_of_atom >= 0)
        order = np.argsort(tok_of_atom, kind="stable")
        keep = order[is_heavy[order]]
        tok_keep = tok_of_atom[keep]
        bounds = np.searchsorted(tok_keep, np.arange(I + 1))
        coords_sorted = atom_array.coord[keep]
        token_atom_coords: list[np.ndarray] = [
            coords_sorted[bounds[i] : bounds[i + 1]] for i in range(I)
        ]
        return adj_from_block_types(
            blocks, types, cb_coords, token_atom_coords, I,
            contact_rules=contact_rules, loop_adj=loop_adj,
        )

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
