"""
FeaturizeSSADJ: transform that writes feats["ss_labels"] and feats["adj_matrix"].

Training path
  SS:  reads is_helix_gt / is_sheet_gt from atom_array (set by Add1DSSFeature)
  ADJ: computed from reconstructed Cβ coordinates via token_id mapping (robust to virtual atoms)
  CFG dropout (6 modes, summing to 1.0):
    10% unconditional       — all SS masked (3), all ADJ unspecified (2)
    10% SS-only             — full SS, all ADJ unspecified (2)
    10% ADJ-only            — all SS masked (3), sparse or full ADJ
    20% full conditioning   — complete SS + complete ADJ
    40% interaction-site    — expose SS+ADJ for structurally-selected contact patches
    10% remodeling          — mask 1–3 contiguous segments at SS boundaries

Inference path
  SS:  reads ss_label_conditioning annotation (integer per token, 3 = unconditioned)
  ADJ: reads feats["adj_matrix"] if present (set by input_parsing.py from
       adj_file / adj_pairs)

  If the user specified nothing, the corresponding key is OMITTED rather than
  filled with a sentinel. TokenInitializer guards with `"adj_matrix" in f`, so an
  absent key means no adapter contribution at all -- no 0, no 1, no 2 -- and
  RFD3._cfg_is_noop then skips the classifier-free-guidance reference pass, whose
  delta would be exactly zero. Emitting an all-ZERO matrix (the behaviour before)
  was actively wrong: ADJAdapter embeds state 0 with |out| ~30 vs 0 for masked, so
  it asserted "nothing touches anything" on every pair.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from atomworks.ml.transforms._checks import check_contains_keys, check_is_instance
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray

_SS_HELIX = 0
_SS_STRAND = 1
_SS_LOOP = 2
_SS_MASKED = 3


class RejectShortMultichainLocalTokenExamples(Transform):
    """Reject crops unsupported by the baseline local-token attention path."""

    def __init__(self, min_tokens: int = 32, min_chains: int = 4):
        self.min_tokens = min_tokens
        self.min_chains = min_chains

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        token_starts = get_token_starts(atom_array)
        if len(token_starts) >= self.min_tokens:
            return data

        token_array = atom_array[token_starts]
        n_chains = len(np.unique(token_array.pn_unit_iid))
        if n_chains >= self.min_chains:
            example_id = data.get("example_id", "<unknown>")
            raise ValueError(
                f"{example_id} has {n_chains} chains but only {len(token_starts)} "
                f"tokens after conditioning transforms; baseline local-token "
                f"attention requires at least {self.min_tokens} tokens for "
                f"examples with {self.min_chains}+ chains"
            )
        return data


class FeaturizeSSADJ(Transform):
    """Compute or load SS labels + ADJ matrix, apply CFG dropout, write to feats.

    Must run AFTER Add1DSSFeature (training) or after _assign_types_to_input sets
    the ss_label_conditioning annotation (inference).
    """

    UNCONDITIONAL_PROB = 0.10
    SS_ONLY_PROB = 0.10
    ADJ_ONLY_PROB = 0.10
    FULL_CONDITIONING_PROB = 0.20
    INTERACTION_SITE_PROB = 0.40
    REMODELING_PROB = 0.10

    def __init__(
        self,
        loop_adj: bool = False,
        typed_blocks: bool = False,
        cb_threshold: float = 6.0,
        ca_threshold_loop: float = 6.0,
        min_loop_contact_pairs: int = 3,
        mask_loop_pairs: bool = True,
        adj_only_negative_frac: float = 0.3,
        motif_region_mask_prob: float = 0.5,
        cross_chain_seed_prob: float = 0.0,
    ):
        self.loop_adj = loop_adj
        self.typed_blocks = typed_blocks
        self.cb_threshold = cb_threshold
        self.ca_threshold_loop = ca_threshold_loop
        self.min_loop_contact_pairs = min_loop_contact_pairs
        self.mask_loop_pairs = mask_loop_pairs
        self.adj_only_negative_frac = adj_only_negative_frac
        self.motif_region_mask_prob = motif_region_mask_prob
        self.cross_chain_seed_prob = cross_chain_seed_prob

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        feats = data.setdefault("feats", {})
        data.setdefault("specification", {})
        if "src_component" not in atom_array.get_annotation_categories():
            atom_array.set_annotation(
                "src_component",
                np.full(atom_array.array_length(), "", dtype=atom_array.res_name.dtype),
            )
        is_inference = data.get("is_inference", False)

        token_starts = get_token_starts(atom_array)
        I = len(token_starts)

        if is_inference:
            ss_labels = self._get_ss_inference(atom_array, token_starts, I)
        else:
            ss_labels = self._get_ss_training(atom_array, token_starts, I)

        if is_inference:
            adj_matrix = feats["adj_matrix"].float() if "adj_matrix" in feats else None
            _pairs = (data.get("specification") or {}).get("adj_pairs")
            if _pairs:
                adj_matrix = self._apply_adj_pairs(
                    atom_array, token_starts, I, _pairs, adj_matrix
                )
        else:
            adj_matrix = self._compute_adj(atom_array, token_starts, ss_labels.numpy())

        if not is_inference:
            ss_labels, adj_matrix = self._apply_cfg_dropout(
                ss_labels, adj_matrix, atom_array, token_starts, feats
            )
            feats["ss_labels"] = ss_labels
            feats["adj_matrix"] = adj_matrix
            return data

        if ss_labels is not None and not bool((ss_labels == _SS_MASKED).all()):
            feats["ss_labels"] = ss_labels
        else:
            feats.pop("ss_labels", None)
        if adj_matrix is not None:
            feats["adj_matrix"] = adj_matrix
        else:
            feats.pop("adj_matrix", None)
        return data


    @staticmethod
    def _apply_adj_pairs(
        atom_array: AtomArray,
        token_starts: np.ndarray,
        I: int,
        pairs: list,
        adj_matrix: torch.Tensor | None,
    ) -> torch.Tensor:
        """Resolve `adj_pairs` selections against the FINAL token layout.

        Runs here rather than in input_parsing because token identity is not
        settled until after AtomizeByCCDName (one token per ligand ATOM) and
        PadTokensWithVirtualAtoms. Resolving early produced a residue-sized matrix
        (61) against an atomised model input (76) for a 16-atom biotin.

        A ligand therefore resolves to all of its atom tokens, i.e. contact means
        "close to any atom of the entity". That matches the training label, where
        _segment_blocks_typed groups a ligand's atom tokens into ONE block.
        """
        from rfd3.inference.parsing import InputSelection

        if adj_matrix is None:
            adj_matrix = torch.full((I, I), 2, dtype=torch.float32)
        for entry in pairs:
            if "!:" in entry:
                lhs, rhs = entry.split("!:", 1)
                value = 0.0
            elif ":" in entry:
                lhs, rhs = entry.split(":", 1)
                value = 1.0
            else:
                raise ValueError(
                    f"adj_pairs entry {entry!r} must contain ':' (contact) or "
                    f"'!:' (non-contact), e.g. 'A20-35 : B10-25'."
                )
            idxs = []
            for side, raw in (("left", lhs), ("right", rhs)):
                sel = raw.strip()
                if not sel:
                    raise ValueError(
                        f"adj_pairs entry {entry!r} has an empty {side} selection."
                    )
                m = InputSelection.from_any(
                    sel, atom_array=atom_array
                ).get_mask()[token_starts]
                if not m.any():
                    raise ValueError(
                        f"adj_pairs selection {sel!r} (in {entry!r}) matched no "
                        f"final tokens."
                    )
                idxs.append(np.where(m)[0])
            a, b = idxs
            if np.intersect1d(a, b).size:
                raise ValueError(
                    f"adj_pairs entry {entry!r} selects overlapping regions; a "
                    f"block cannot be adjacent to itself."
                )
            ta, tb = torch.from_numpy(a), torch.from_numpy(b)
            adj_matrix[ta[:, None], tb[None, :]] = value
            adj_matrix[tb[:, None], ta[None, :]] = value
        adj_matrix.fill_diagonal_(0.0)
        return adj_matrix


    def _get_ss_inference(
        self, atom_array: AtomArray, token_starts: np.ndarray, I: int
    ) -> torch.Tensor:
        if "ss_label_conditioning" in atom_array.get_annotation_categories():
            labels = atom_array.get_annotation("ss_label_conditioning")[token_starts]
            return torch.tensor(np.asarray(labels), dtype=torch.long)
        return torch.full((I,), _SS_MASKED, dtype=torch.long)

    def _get_ss_training(
        self, atom_array: AtomArray, token_starts: np.ndarray, I: int
    ) -> torch.Tensor:
        cats = atom_array.get_annotation_categories()
        if "is_helix_gt" not in cats:
            return torch.full((I,), _SS_MASKED, dtype=torch.long)

        helix_tok = atom_array.is_helix_gt[token_starts].astype(bool)
        sheet_tok = atom_array.is_sheet_gt[token_starts].astype(bool)
        ss = np.full(I, _SS_LOOP, dtype=np.int64)
        ss[helix_tok] = _SS_HELIX
        ss[sheet_tok] = _SS_STRAND

        try:
            from atomworks.enums import ChainType

            ctype = np.asarray(atom_array.chain_type)[token_starts]
            non_protein = ~np.isin(
                ctype,
                [int(ChainType.POLYPEPTIDE_L), int(ChainType.POLYPEPTIDE_D)],
            )
            ss[non_protein] = _SS_MASKED
        except AttributeError:
            pass

        return torch.tensor(ss, dtype=torch.long)


    def _compute_adj(
        self,
        atom_array: AtomArray,
        token_starts: np.ndarray,
        ss_labels_np: np.ndarray,
    ) -> torch.Tensor:
        from foundry.data.ss_adj import get_adj_from_token_array

        adj = get_adj_from_token_array(
            atom_array,
            token_starts,
            ss_labels_np.astype(np.float32),
            loop_adj=self.loop_adj,
            typed_blocks=self.typed_blocks,
            cb_threshold=self.cb_threshold,
            ca_threshold_loop=self.ca_threshold_loop,
            min_loop_contact_pairs=self.min_loop_contact_pairs,
            mask_loop_pairs=self.mask_loop_pairs,
        )
        return torch.from_numpy(adj)


    @staticmethod
    def _find_ss_blocks(
        labels: np.ndarray, chain_ids: np.ndarray | None = None
    ) -> list[np.ndarray]:
        """Return index arrays for each contiguous run of same SS label.

        Blocks are also broken at chain boundaries so that SS elements from
        different chains are never merged (important for multimer contact-pair
        detection and interaction-site masking).
        """
        blocks: list[np.ndarray] = []
        start = 0
        while start < len(labels):
            end = start + 1
            while end < len(labels):
                same_ss = labels[end] == labels[start]
                same_chain = chain_ids is None or chain_ids[end] == chain_ids[start]
                if same_ss and same_chain:
                    end += 1
                else:
                    break
            blocks.append(np.arange(start, end))
            start = end
        return blocks

    @staticmethod
    def _select_connected_pairs(
        contact_pairs: list[tuple[np.ndarray, np.ndarray]],
        n_keep: int,
        seed_candidates: list[int] | None = None,
        always_include: list[int] | None = None,
    ) -> list[int]:
        """Select n_keep contact-pair indices biased toward a connected subgraph.

        Pure uniform sampling over independent block-pairs tends to produce
        scattered, non-contiguous exposures (e.g. revealing pair (1,2) and pair
        (5,6) while leaving 3,4 untouched) -- not representative of how a real
        partial fold specification would look (a coherent region, not scattered
        unrelated facts). This grows the selection one pair at a time, preferring
        pairs that share a block with something already selected; falls back to
        uniform selection when no connected candidate remains.
        """
        n_total = len(contact_pairs)
        if n_keep >= n_total:
            return list(range(n_total))

        forced = [i for i in (always_include or []) if 0 <= i < n_total]
        if forced and len(forced) >= n_keep:
            return forced[:max(n_keep, 1)]

        def block_key(b: np.ndarray) -> tuple[int, int]:
            return (int(b[0]), int(b[-1]))

        remaining = list(range(n_total))
        if forced:
            selected = list(forced)
            for i in forced:
                if i in remaining:
                    remaining.remove(i)
            involved = {block_key(b) for i in forced for b in contact_pairs[i]}
        else:
            pool = [i for i in (seed_candidates or []) if i in remaining] or remaining
            first = int(np.random.choice(pool))
            selected = [first]
            remaining.remove(first)
            involved = {block_key(b) for b in contact_pairs[first]}

        while len(selected) < n_keep and remaining:
            candidates = [
                idx
                for idx in remaining
                if involved & {block_key(b) for b in contact_pairs[idx]}
            ]
            if not candidates:
                candidates = remaining
            pick = int(np.random.choice(candidates))
            selected.append(pick)
            remaining.remove(pick)
            involved |= {block_key(b) for b in contact_pairs[pick]}

        return selected


    def _sample_sparse_contacts(
        self,
        adj_matrix: torch.Tensor,
        ss_labels: torch.Tensor,
        chain_ids: np.ndarray | None = None,
        max_contacts: int = 3,
        negative_frac: float = 0.0,
    ) -> torch.Tensor:
        """Keep a few positive SS-block contacts and mask all other pairs with 2.

        The completeness ceiling (`max_contacts`) scales with how many true
        contact pairs the structure actually has, so complex topologies (many
        blocks/contacts) aren't structurally capped at a lower max exposure than
        simple ones -- a flat constant caps a 3-contact helix bundle at 100%
        reachable but a 5-contact beta sandwich at 60%, for no principled reason.
        """
        labels = ss_labels.cpu().numpy()
        blocks = self._find_ss_blocks(labels, chain_ids)

        adj_np = adj_matrix.cpu().numpy()
        contact_pairs = []
        for block_i_idx, block_i in enumerate(blocks):
            for block_j in blocks[block_i_idx + 1:]:
                if np.any(adj_np[np.ix_(block_i, block_j)] == 1):
                    contact_pairs.append((block_i, block_j))

        sparse = torch.full_like(adj_matrix, 2)
        if not contact_pairs:
            return sparse
        effective_cap = max(max_contacts, math.ceil(0.8 * len(contact_pairs)))
        n_keep = int(np.random.randint(1, min(effective_cap, len(contact_pairs)) + 1))
        cross_idxs: list[int] = []
        if chain_ids is not None:
            for pi, (bi, bj) in enumerate(contact_pairs):
                try:
                    if chain_ids[bi[0]] != chain_ids[bj[0]]:
                        cross_idxs.append(pi)
                except (IndexError, TypeError):
                    continue
        keep_cross = bool(cross_idxs) and np.random.random() < getattr(
            self, "cross_chain_seed_prob", 0.0
        )
        selected = self._select_connected_pairs(
            contact_pairs,
            max(n_keep, len(cross_idxs) if keep_cross else 0),
            seed_candidates=cross_idxs if keep_cross else None,
            always_include=cross_idxs if keep_cross else None,
        )
        for pair_idx in selected:
            first, second = contact_pairs[int(pair_idx)]
            first_t = torch.from_numpy(first).to(adj_matrix.device)
            second_t = torch.from_numpy(second).to(adj_matrix.device)
            sparse[first_t[:, None], second_t[None, :]] = adj_matrix[
                first_t[:, None], second_t[None, :]
            ]
            sparse[second_t[:, None], first_t[None, :]] = adj_matrix[
                second_t[:, None], first_t[None, :]
            ]

        if negative_frac > 0.0 and len(blocks) > 1:
            non_pairs = []
            for bi in range(len(blocks)):
                for bj in range(bi + 1, len(blocks)):
                    b1, b2 = blocks[bi], blocks[bj]
                    if np.all(adj_np[np.ix_(b1, b2)] == 0):
                        non_pairs.append((b1, b2))
            if non_pairs:
                n_neg = int(np.random.binomial(len(non_pairs), min(1.0, negative_frac)))
                if n_neg:
                    for k in np.random.choice(len(non_pairs), size=n_neg, replace=False):
                        b1, b2 = non_pairs[int(k)]
                        t1 = torch.from_numpy(b1).to(adj_matrix.device)
                        t2 = torch.from_numpy(b2).to(adj_matrix.device)
                        sparse[t1[:, None], t2[None, :]] = 0
                        sparse[t2[:, None], t1[None, :]] = 0
        return sparse

    def _sample_interaction_site_partial(
        self,
        adj_matrix: torch.Tensor,
        ss_labels: torch.Tensor,
        chain_ids: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reveal SS + ADJ for structurally-selected contact patches; mask the body.

        Identifies contacting SS-element block pairs, selects 1–3+ patches
        weighted toward interface-sized exposures (single site 30%, 2–3 patches
        40%, larger/complex 30%), then reveals only those residues and their
        adjacency relationships. Exposure fraction is a structural consequence,
        not a random target. Multi-patch selections are biased toward a
        connected subgraph (see _select_connected_pairs) rather than scattered
        independent pairs.

        Blocks are broken at chain boundaries so cross-chain SS merging cannot
        happen; inter-chain contacts (PPI interfaces) are first-class patches.

        Falls back to full conditioning if the structure has no inter-block
        contacts (e.g., single-element proteins).
        """
        labels = ss_labels.cpu().numpy()
        adj_np = adj_matrix.cpu().numpy()
        blocks = self._find_ss_blocks(labels, chain_ids)

        contact_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        for i, block_i in enumerate(blocks):
            for block_j in blocks[i + 1:]:
                if np.any(adj_np[np.ix_(block_i, block_j)] == 1):
                    contact_pairs.append((block_i, block_j))

        if not contact_pairs:
            return ss_labels, adj_matrix

        n_max = len(contact_pairs)
        r = np.random.random()
        if r < 0.30 or n_max < 2:
            n_patches = 1
        elif r < 0.70:
            n_patches = int(np.random.randint(2, min(4, n_max) + 1))
        else:
            n_patches = int(np.random.randint(2, n_max + 1))
        n_patches = max(1, min(n_patches, n_max))

        cross_idxs: list[int] = []
        if chain_ids is not None:
            for pi, (bi, bj) in enumerate(contact_pairs):
                try:
                    if chain_ids[bi[0]] != chain_ids[bj[0]]:
                        cross_idxs.append(pi)
                except (IndexError, TypeError):
                    continue
        keep_cross = bool(cross_idxs) and np.random.random() < self.cross_chain_seed_prob
        selected_idxs = self._select_connected_pairs(
            contact_pairs,
            max(n_patches, len(cross_idxs) if keep_cross else 0),
            seed_candidates=cross_idxs if keep_cross else None,
            always_include=cross_idxs if keep_cross else None,
        )

        exposed_set: set[int] = set()
        for idx in selected_idxs:
            block_i, block_j = contact_pairs[int(idx)]
            exposed_set.update(block_i.tolist())
            exposed_set.update(block_j.tolist())
        exposed = np.array(sorted(exposed_set), dtype=np.int64)

        new_ss = torch.full_like(ss_labels, _SS_MASKED)
        new_ss[exposed] = ss_labels[exposed]

        new_adj = torch.full_like(adj_matrix, 2)
        e = torch.from_numpy(exposed)
        new_adj[e[:, None], e[None, :]] = adj_matrix[e[:, None], e[None, :]]

        return new_ss, new_adj

    def _sample_remodeling_partial(
        self,
        adj_matrix: torch.Tensor,
        ss_labels: torch.Tensor,
        chain_ids: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask 10–50% of residues in 1–3 contiguous segments at SS transitions.

        Anchors segments at secondary-structure boundary positions (where SS
        type changes), keeping the unmasked region fully conditioned. Chain
        boundaries are treated as additional transition points so segments do
        not span multiple chains. Suitable for loop/region remodeling where the
        overall fold topology is preserved.
        """
        I = ss_labels.shape[0]
        labels = ss_labels.numpy()

        transitions = np.flatnonzero(labels[:-1] != labels[1:]) + 1
        if chain_ids is not None:
            chain_boundaries = np.flatnonzero(chain_ids[:-1] != chain_ids[1:]) + 1
            transitions = np.unique(np.concatenate([transitions, chain_boundaries]))

        frac = np.random.uniform(0.10, 0.50)
        n_mask = max(1, int(round(I * frac)))

        n_segs = int(np.random.randint(1, 4)) if I >= 6 else 1

        if len(transitions) >= n_segs:
            anchors = np.random.choice(transitions, size=n_segs, replace=False)
        elif len(transitions) > 0:
            extra = np.random.randint(0, I, size=n_segs - len(transitions))
            anchors = np.concatenate([transitions, extra])
        else:
            anchors = np.random.randint(0, I, size=n_segs)

        per_seg = n_mask // n_segs
        remainder = n_mask - per_seg * n_segs

        mask_np = np.zeros(I, dtype=bool)
        for i, anchor in enumerate(anchors):
            budget = per_seg + (1 if i < remainder else 0)
            half = budget // 2
            start = max(0, int(anchor) - half)
            end = min(I, start + budget)
            start = max(0, end - budget)
            mask_np[start:end] = True

        partial_mask = torch.from_numpy(mask_np)
        new_ss = ss_labels.clone()
        new_ss[partial_mask] = _SS_MASKED
        new_adj = adj_matrix.clone()
        new_adj[partial_mask, :] = 2
        new_adj[:, partial_mask] = 2
        return new_ss, new_adj


    def _apply_cfg_dropout(
        self,
        ss_labels: torch.Tensor,
        adj_matrix: torch.Tensor,
        atom_array: AtomArray,
        token_starts: np.ndarray,
        feats: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a controlled mixture of fold-conditioning modes.

        Distribution (sums to 1.0):
          10% unconditional        — all SS = 3 (masked), all ADJ = 2 (unspecified)
          10% SS-only              — full SS, ADJ = 2 everywhere
          10% ADJ-only             — SS = 3, sparse or kept ADJ
          20% full conditioning    — complete SS + complete ADJ unchanged
          40% interaction-site     — expose SS+ADJ for structural contact patches
          10% remodeling           — mask contiguous segments at SS boundaries
        """
        I = ss_labels.shape[0]
        draw = np.random.random()
        chain_ids = atom_array.chain_id[token_starts]

        unconditional_end    = self.UNCONDITIONAL_PROB
        ss_only_end          = unconditional_end + self.SS_ONLY_PROB
        adj_only_end         = ss_only_end + self.ADJ_ONLY_PROB
        full_end             = adj_only_end + self.FULL_CONDITIONING_PROB
        interaction_site_end = full_end + self.INTERACTION_SITE_PROB

        if draw < unconditional_end:
            ss_labels = torch.full((I,), _SS_MASKED, dtype=torch.long)
            adj_matrix = torch.full_like(adj_matrix, 2)
            if feats is not None and "ss_ratio" in feats:
                feats["ss_ratio"] = torch.full_like(feats["ss_ratio"], -1.0)
        elif draw < ss_only_end:
            adj_matrix = torch.full_like(adj_matrix, 2)
        elif draw < adj_only_end:
            r = np.random.random()
            if r < 0.50:
                adj_matrix = self._sample_sparse_contacts(
                    adj_matrix, ss_labels, chain_ids,
                    negative_frac=self.adj_only_negative_frac,
                )
            elif r < 0.70:
                pass
            else:
                adj_matrix = torch.full_like(adj_matrix, 2)
            ss_labels = torch.full((I,), _SS_MASKED, dtype=torch.long)
        elif draw < full_end:
            pass
        elif draw < interaction_site_end:
            ss_labels, adj_matrix = self._sample_interaction_site_partial(
                adj_matrix, ss_labels, chain_ids
            )
        else:
            ss_labels, adj_matrix = self._sample_remodeling_partial(
                adj_matrix, ss_labels, chain_ids
            )

        ss_labels, adj_matrix = self._maybe_mask_motif_region(
            ss_labels, adj_matrix, atom_array, token_starts
        )

        return ss_labels, adj_matrix

    def _maybe_mask_motif_region(
        self,
        ss_labels: torch.Tensor,
        adj_matrix: torch.Tensor,
        atom_array: AtomArray,
        token_starts: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sometimes strip SS/ADJ from motif tokens, matching `ss_adj_contig` inference.

        The six CFG-dropout modes above are motif-blind: they mask SS blocks and
        contact pairs chosen at random, with no regard for which tokens are a fixed
        motif. But the real "fold conditioning onto a fixed target" inference call
        (`input` + `contig` + `ss_file`/`adj_file` + `ss_adj_contig`, see
        DesignInputSpecification.to_pipeline_input) produces a very specific pattern:
        the *diffused* region carries real SS/ADJ while every motif token is
        ss=3 / adj=2, including all motif x diffused cross terms. Random block
        dropout essentially never reproduces that motif-aligned split, so without
        this the model would only ever meet it at inference time.

        Applied probabilistically (not always) because the other valid inference
        pattern -- omitting `ss_adj_contig` and supplying a full-length tensor that
        covers motif and diffused tokens alike -- is what the unmasked path already
        trains. Pure de novo examples have no motif tokens and are untouched.
        """
        if self.motif_region_mask_prob <= 0.0:
            return ss_labels, adj_matrix
        if "is_motif_token" not in atom_array.get_annotation_categories():
            return ss_labels, adj_matrix

        is_motif = np.asarray(
            atom_array.get_annotation("is_motif_token")[token_starts]
        ).astype(bool)
        if not is_motif.any() or is_motif.all():
            return ss_labels, adj_matrix
        draw = np.random.random()
        motif_idx = torch.from_numpy(np.nonzero(is_motif)[0])

        if draw >= self.motif_region_mask_prob:
            return ss_labels, adj_matrix

        ss_labels = ss_labels.clone()
        adj_matrix = adj_matrix.clone()
        ss_labels[motif_idx] = _SS_MASKED
        adj_matrix[motif_idx, :] = 2
        adj_matrix[:, motif_idx] = 2
        return ss_labels, adj_matrix
