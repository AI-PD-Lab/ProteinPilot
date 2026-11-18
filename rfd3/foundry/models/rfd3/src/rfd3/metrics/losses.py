import torch
import torch.nn as nn

from foundry.training.checkpoint import activation_checkpointing


class SequenceLoss(nn.Module):
    def __init__(self, weight, min_t=0, max_t=torch.inf):
        super().__init__()
        self.weight = weight
        self.min_t = min_t
        self.max_t = max_t
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, network_input, network_output, loss_input):
        t = network_input["t"]  # (B,)
        valid_t = (self.min_t <= t) & (t < self.max_t)  # bool mask over batch
        n_valid_t = valid_t.sum()

        # Grab network outputs
        sequence_logits_I = network_output["sequence_logits_I"]  # (B, L, 32)
        sequence_indices_I = network_output["sequence_indices_I"]  # (B, L)

        if n_valid_t == 0:
            zero = sequence_logits_I.sum() * 0.0
            return zero, {
                "valid_t_fraction": torch.tensor([0.0]),
                "n_valid_t": torch.tensor([0.0]),
            }

        pred_seq = sequence_logits_I[valid_t]  # (V, L, 32)
        gt_seq = loss_input["seq_token_lvl"]  # [L,]
        gt_seq = gt_seq.unsqueeze(0).expand(n_valid_t, -1)  # (V, L)
        w_seq = loss_input["sequence_valid_mask"]  # [L,]

        # Cross‑entropy token loss
        token_loss = self.loss_fn(pred_seq.permute(0, 2, 1), gt_seq)  # (V, L)
        token_loss = token_loss * w_seq[None]  # (V, L)
        token_loss = token_loss.mean(dim=-1)  # (V,)

        _, order = torch.sort(t[valid_t])  # low‑t first
        sequence_indices_I = sequence_indices_I[valid_t]
        recovery = (sequence_indices_I == gt_seq).float()  # (V, L)
        recovery = recovery[order]  # reorder by t
        recovery = recovery[..., (w_seq > 0).bool()]  # [V, L_valid]
        lowest_t_rec = recovery[0].mean()  # scalar

        outs = {
            "token_lvl_sequence_loss": token_loss.mean().detach(),
            "seq_recovery": recovery.mean().detach(),
            "lowest_t_seq_recovery": lowest_t_rec.detach(),
            "valid_t_fraction": valid_t.float().mean().detach(),
            "n_valid_t": n_valid_t.float(),
        }
        token_loss = torch.clamp(token_loss.mean(), max=4)
        return self.weight * token_loss, outs


class DiffusionLoss(nn.Module):
    def __init__(
        self,
        *,
        weight,
        sigma_data,
        lddt_weight,
        alpha_virtual_atom=1.0,
        alpha_unindexed_diffused=1.0,
        alpha_polar_residues=1.0,
        alpha_ligand=2.0,
        unindexed_t_alpha=1.0,
        unindexed_norm_p=1.0,
        lp_weight=0.0,
        **_,  # dump args from old configs
    ):
        super().__init__()
        self.weight = weight
        self.lddt_weight = lddt_weight
        self.sigma_data = sigma_data

        self.alpha_unindexed_diffused = alpha_unindexed_diffused
        self.alpha_virtual_atom = alpha_virtual_atom
        self.unindexed_norm_p = unindexed_norm_p
        self.unindexed_t_alpha = unindexed_t_alpha
        self.lp_weight = lp_weight
        self.alpha_ligand = alpha_ligand
        self.alpha_polar_residues = alpha_polar_residues

        self.get_lambda = (
            lambda sigma: (sigma**2 + self.sigma_data**2)
            / (sigma * self.sigma_data) ** 2
        )

    def forward(self, network_input, network_output, loss_input):
        X_L = network_output["X_L"]  # D, L, 3
        D = X_L.shape[0]
        crd_mask_L = loss_input["crd_mask_L"]  # (D, L)
        crd_mask_L = crd_mask_L.unsqueeze(0).expand(D, -1)
        tok_idx = network_input["f"]["atom_to_token_map"]
        t = network_input["t"]  # (D,)
        is_original_unindexed_token = loss_input["is_original_unindexed_token"][tok_idx]
        is_polar_atom = network_input["f"]["is_polar"][tok_idx]
        is_ligand = network_input["f"]["is_ligand"][tok_idx]
        is_virtual_atom = network_input["f"]["is_virtual"]  # L
        is_sidechain_atom = network_input["f"]["is_sidechain"]  # L
        is_sidechain_atom = is_sidechain_atom & ~is_virtual_atom

        w_L = torch.ones_like(tok_idx, dtype=X_L.dtype)
        w_L[is_original_unindexed_token] = (
            w_L[is_original_unindexed_token] * self.alpha_unindexed_diffused
        )
        w_L[is_virtual_atom] *= self.alpha_virtual_atom
        w_L[is_ligand] *= self.alpha_ligand

        # Upweight polar residues
        w_L[is_polar_atom] *= self.alpha_polar_residues
        w_L = w_L[None].expand(D, -1) * crd_mask_L

        X_gt_L = torch.nan_to_num(loss_input["X_gt_L_in_input_frame"])
        l_mse_L = w_L * torch.sum((X_L - X_gt_L) ** 2, dim=-1)
        l_mse_L = torch.div(l_mse_L, 3 * torch.sum(crd_mask_L[0]) + 1e-4)  # D, L

        if torch.any(is_original_unindexed_token):
            t_exp = t[:, None].expand(-1, X_L.shape[1])  # [D, L]
            t_exp = (
                t_exp * (~is_original_unindexed_token)
                + self.unindexed_t_alpha * t_exp * is_original_unindexed_token
            )

            l_global = (self.get_lambda(t_exp) * l_mse_L).sum(-1)

            # Get renormalization factor to equalize expectation of the loss
            r = self.get_lambda(t * self.unindexed_t_alpha) / self.get_lambda(t)
            t_factor = crd_mask_L.sum(-1) / (
                r * crd_mask_L[:, is_original_unindexed_token].sum(-1)
                + crd_mask_L[:, ~is_original_unindexed_token].sum(-1)
            )
            assert t_factor.shape == (D,), t_factor.shape
            l_global = l_global * t_factor
        else:
            l_global = self.get_lambda(t) * l_mse_L.sum(-1)

        assert l_global.shape == (D,), l_global.shape

        if torch.any(is_original_unindexed_token):
            lp_norm_L = w_L * torch.linalg.norm(
                X_L - X_gt_L, ord=self.unindexed_norm_p, dim=-1
            )  # [D, L]
            lp_norm_unindexed_diffused = lp_norm_L * is_original_unindexed_token[None]
            lp_norm_unindexed_diffused = torch.div(
                lp_norm_unindexed_diffused,
                self.alpha_unindexed_diffused
                * 3
                * torch.sum(is_original_unindexed_token)
                + 1e-4,
            )  # D, L
            lp_norm_unindexed_diffused = lp_norm_unindexed_diffused.sum(
                -1
            ) * self.get_lambda(self.unindexed_t_alpha * t)

            l_total = l_global + self.lp_weight * lp_norm_unindexed_diffused
        else:
            lp_norm_unindexed_diffused = None
            lp_norm_L = None
            l_total = l_global

        # ... Aggregate
        l_mse_total = torch.clamp(l_total, max=2)
        assert l_mse_total.shape == (
            D,
        ), f"Expected l_total to be of shape (D,), got {l_total.shape}"
        l_mse_total = torch.mean(l_mse_total)  # D, -> scalar

        # ... Return
        if self.lddt_weight > 0:
            # ... Calculate LDDT loss at the beginning
            smoothed_lddt_loss_, lddt_loss_dict = smoothed_lddt_loss(
                X_L,
                X_gt_L,
                crd_mask_L,
                network_input["f"]["is_dna"],
                network_input["f"]["is_rna"],
                tok_idx,
                return_extras=True,
            )  # D,
            l_total = l_mse_total + self.lddt_weight * smoothed_lddt_loss_.mean()
        else:
            lddt_loss_dict = {}
            l_total = l_mse_total
        # ... Return additional losses
        t, indices = torch.sort(t)
        l_mse_low, l_mse_high = torch.split(l_global[indices], [D // 2, D - D // 2])
        loss_dict = {
            "mse_loss_mean": l_mse_total,
            "mse_loss_low_t": l_mse_low,
            "mse_loss_high_t": l_mse_high,
            "lp_norm": lp_norm_L,
            "lp_norm_unindexed_diffused": lp_norm_unindexed_diffused,
        } | lddt_loss_dict
        loss_dict = {
            k: torch.mean(v).detach() for k, v in loss_dict.items() if v is not None
        }

        return self.weight * l_total, loss_dict


def smoothed_lddt_loss(
    X_L,
    X_gt_L,
    crd_mask_L,
    is_dna,
    is_rna,
    tok_idx,
    is_virtual=None,
    alpha_virtual=1.0,
    return_extras=False,
    eps=1e-6,
):
    @activation_checkpointing
    def _dolddt(X_L, X_gt_L, crd_mask_L, is_dna, is_rna, tok_idx, eps, use_amp=True):
        B, L = X_L.shape[:2]
        first_index, second_index = torch.triu_indices(L, L, 1, device=X_L.device)

        # compute the unique distances between all pairs of atoms
        X_gt_L = X_gt_L.nan_to_num()

        # only use native 1 (assumes dist map identical btwn all copies)
        ground_truth_distances = torch.linalg.norm(
            X_gt_L[0:1, first_index] - X_gt_L[0:1, second_index], dim=-1
        )

        # only score pairs that are close enough in the ground truth
        is_na_L = is_dna[tok_idx][first_index] | is_rna[tok_idx][first_index]
        pair_mask = torch.logical_and(
            ground_truth_distances > 0,
            ground_truth_distances < torch.where(is_na_L, 30.0, 15.0),
        )
        del is_na_L

        # only score pairs that are resolved in the ground truth
        pair_mask *= crd_mask_L[0:1, first_index] * crd_mask_L[0:1, second_index]

        # don't score pairs that are in the same token
        pair_mask *= tok_idx[None, first_index] != tok_idx[None, second_index]

        _, valid_pairs = pair_mask.nonzero(as_tuple=True)
        pair_mask = pair_mask[:, valid_pairs].to(X_L.dtype)
        ground_truth_distances = ground_truth_distances[:, valid_pairs]
        first_index, second_index = first_index[valid_pairs], second_index[valid_pairs]

        predicted_distances = torch.linalg.norm(
            X_L[:, first_index] - X_L[:, second_index], dim=-1
        )

        delta_distances = torch.abs(predicted_distances - ground_truth_distances + eps)
        del predicted_distances, ground_truth_distances

        if is_virtual is not None:
            pair_mask[:, (is_virtual[first_index] * is_virtual[second_index])] *= (
                alpha_virtual
            )

        # I assume gradients flow better if we sum first rather than keeping everything in D, L...
        lddt = (
            0.25
            * (
                torch.sum(torch.sigmoid(0.5 - delta_distances) * pair_mask, dim=(1))
                + torch.sum(torch.sigmoid(1.0 - delta_distances) * pair_mask, dim=(1))
                + torch.sum(torch.sigmoid(2.0 - delta_distances) * pair_mask, dim=(1))
                + torch.sum(torch.sigmoid(4.0 - delta_distances) * pair_mask, dim=(1))
            )
            / (torch.sum(pair_mask, dim=(1)) + eps)
        )

        if not return_extras:
            return 1 - lddt

        # ...Hence we recalculate the losses here and pick out the parts of interest
        with torch.no_grad():
            lddt_ = (
                0.25
                * (
                    torch.sigmoid(0.5 - delta_distances)
                    + torch.sigmoid(1.0 - delta_distances)
                    + torch.sigmoid(2.0 - delta_distances)
                    + torch.sigmoid(4.0 - delta_distances)
                )
                * pair_mask
                / (torch.sum(pair_mask, dim=(1)) + eps)
            )

            def filter_lddt(mask, scale=1.0):
                mask = mask.to(pair_mask.dtype)
                if mask.ndim > 1:
                    mask = mask[0]
                mask = (mask[first_index] * mask[second_index])[None].expand(
                    pair_mask.shape[0], -1
                )
                mask = (mask * pair_mask).to(bool)
                return (
                    (1 - torch.sum(lddt_[:, mask[0]] * scale, dim=(1)))
                    .mean()
                    .detach()
                    .cpu()
                )

            extra_lddts = {}
            extra_lddts["mean_lddt"] = filter_lddt(
                torch.full_like(crd_mask_L, 1.0, device=X_L.device)
            )
            extra_lddts["mean_lddt_dna"] = filter_lddt(is_dna[tok_idx])
            extra_lddts["mean_lddt_rna"] = filter_lddt(is_rna[tok_idx])
            extra_lddts["mean_lddt_protein"] = filter_lddt(
                ~is_dna[tok_idx] & ~is_rna[tok_idx]
            )
            # NOTE: This also seems to have issues at epoch level, as with n_valid_t
            # before. Will leave as-is for now but may want to spoof as 0 in the future.
            if is_virtual is not None:
                extra_lddts["mean_lddt_virtual"] = filter_lddt(
                    is_virtual, scale=1 / alpha_virtual
                )
                extra_lddts["mean_lddt_non_virtual"] = filter_lddt(~is_virtual)

        return 1 - lddt, extra_lddts

    return _dolddt(X_L, X_gt_L, crd_mask_L, is_dna, is_rna, tok_idx, eps)


class AdjacencyContactLoss(nn.Module):
    """Soft penalty for violating conditioned positive SS-block adjacency.

    Only scores block-pairs where `f["adj_matrix"] == 1` in the tensor actually
    fed to ADJAdapter this step. Masked (2) and explicit non-contact (0) entries
    are never penalized -- this is leak-free by construction, because every
    masking path in FeaturizeSSADJ._apply_cfg_dropout stamps 2, so a `1` can only
    appear where the pair was genuinely revealed to the model this step (see
    docs/ss_adj_training_plan.md and ss_adj_transforms.py for the six CFG modes).

    IMPORTANT -- operates at the BLOCK-PAIR level (one loss term per conditioned
    SS-block pair, using the minimum cross-block distance), not per individual
    token-pair. adj_matrix is block-filled by ss_adj.py: when block A and block B
    have *any* residue-pair within 6A, the *entire* A x B rectangle is stamped 1
    (see _adj_blocks_from_coords). Scoring every individual (i, j) in that
    rectangle independently -- the v1 implementation -- asks not just "these two
    blocks touch somewhere" but "every residue in A is near every residue in B",
    which for large blocks with several simultaneous contacts (e.g. full
    conditioning revealing all of a structure's true contacts at once) is only
    jointly satisfiable by collapsing the whole structure into a dense blob. This
    was confirmed empirically: a Phase 3 run with the v1 loss hit adj_recovery=1.0
    on full-conditioning cases alongside 500+ CA-CA clashes/design and ~95% loop
    fraction (i.e. no real secondary structure) -- the metric was satisfied by
    collapse, not by learning correct topology. Taking the min distance per block
    pair (mirroring the *at least one contact* ground-truth criterion exactly)
    removes the incentive to move any residue beyond what's needed to satisfy
    the single closest-approach pair.

    Block boundaries are recovered from adj_matrix's own row-uniform structure
    (contiguous tokens belonging to one original SS block always have identical
    rows, since every fill/mask operation upstream operates on whole blocks),
    not from f["ss_labels"] -- ss_labels is *fully masked* in adj_only mode
    (all 3) even though adj_matrix retains real revealed values there, so keying
    off ss_labels would silently lose the loss entirely for that CFG mode.

    Uses a bounded sigmoid "soft contact" score (same pattern as smoothed_lddt_loss)
    rather than a raw hinge. X_L at every t is EDM's x0-estimate, not "structure at
    noise level t" -- but at high t that estimate is inherently blurry, since SS
    elements haven't taken shape yet in the trajectory. An unbounded hinge on a
    wildly-wrong high-t prediction would inject a huge, noisy gradient exactly when
    the network has the least ability to act on it. The sigmoid score instead
    saturates in its flat tails (near 0 far above threshold, near 1 well below it),
    so gradient mass concentrates only where the current prediction sits near the
    6A decision boundary, at whatever t that happens to occur -- no hand-tuned
    t-weighting schedule needed.
    """

    def __init__(
        self,
        *,
        weight: float,
        threshold: float = 6.0,
        softness: float = 2.0,
        min_distance: float = 4.5,
        negative_weight: float = 0.0,
        negative_threshold: float = 8.0,
        negative_margin: float = 1.0,
        use_cb: bool = False,
        detach_motif: bool = False,
        steric_weight: float = 0.0,
        steric_floor: float = 3.5,
        steric_margin: float = 0.5,
    ):
        super().__init__()
        self.weight = weight
        self.threshold = threshold
        self.softness = softness  # Angstrom scale of the sigmoid transition
        # Repulsive arm for block pairs marked as EXPLICIT non-contact (adj == 0).
        #
        # Without it this loss is purely attractive: only `adj == 1` pairs are
        # scored, so nothing in the objective keeps explicitly-non-adjacent blocks
        # apart, and the attractive term biases the model toward association in
        # general. Masked pairs (2) stay indeterminate and are never scored here --
        # only values the featurizer actually revealed as 0.
        #
        # Self-balancing despite 0-pairs vastly outnumbering 1-pairs: the penalty is
        # flat-zero once a pair is beyond `negative_threshold`, so only violators
        # contribute gradient. Left at 0.0 the loss is bit-identical to before.
        self.negative_weight = negative_weight
        # Ceiling on the separation we ask for: beyond this a pair is plainly
        # non-adjacent and we stop caring. Never exceeded even if the native
        # distance is much larger.
        self.negative_threshold = negative_threshold
        # Slack below the native distance before a 0-pair counts as collapsed.
        self.negative_margin = negative_margin
        # Score C-beta (RFD1's virtual CB, rebuilt from predicted N/CA/C) instead of
        # C-alpha, matching the atom the ADJ label is actually defined on. Off by
        # default so existing checkpoints/configs are unaffected.
        self.use_cb = use_cb
        # Freeze fixed-motif coordinates inside this loss (see forward).
        self.detach_motif = detach_motif
        # --- steric arm ---------------------------------------------------------
        # A MASKED adjacency pair (2) means "I do not care whether these are in
        # contact". It does NOT mean the designed chain may occupy the motif's
        # volume. Adjacency state and steric exclusion are orthogonal: the former is
        # a design intent, the latter is physics that holds unconditionally.
        #
        # The attractive well's `min_distance` floor only guards pairs revealed as 1,
        # and the repulsive arm only guards pairs revealed as 0, so a masked
        # designed<->motif pair is currently unconstrained by THIS loss. The
        # diffusion loss against native coordinates does teach clash-free geometry
        # (no native structure self-intersects), which is why this is a hardening
        # term rather than the sole defence -- but at inference the motif is pinned
        # at coordinates the scaffold never saw during its own trajectory, and that
        # is where interpenetration actually shows up.
        #
        # Scored over ALL designed x fixed-motif token pairs regardless of adjacency
        # state. Left at 0.0 the loss is bit-identical to before.
        self.steric_weight = steric_weight
        # Clash threshold on the contact atom (CB, or CA where no CB exists). Real
        # contacts sit at 4.5-6 A, so a 3.5 A floor never competes with a legitimate
        # interface -- it only forbids overlap.
        self.steric_floor = steric_floor
        # Slack below the NATIVE separation before a pair counts as interpenetrating.
        self.steric_margin = steric_margin
        # Lower wall of the flat-bottomed well. Paired beta-strands sit at CA-CA
        # ~4.8-5.5 A and the CA-clash metric flags <3.5 A, so 4.5 leaves the
        # physically correct range unpenalised while stopping the loss from
        # compressing contacts into clash territory. Set equal to `threshold` to
        # recover the old one-sided (monotonic, always-attracting) behaviour.
        self.min_distance = min_distance

    @staticmethod
    def _virtual_cb(X_L: torch.Tensor, bb_idx: torch.Tensor) -> torch.Tensor:
        """Rebuild virtual C-beta from predicted N/CA/C, differentiably.

        Byte-identical constants to foundry.data.ss_adj._reconstruct_cb, which is
        itself RFD1's generate_Cbeta (the Rosetta-parameter variant). Using the
        same construction on both sides makes the loss score exactly the quantity
        the ADJ label is defined on, including for glycine -- real CB does not
        exist there, and `central_atom="CB"` silently falls back to CA.
        """
        n = X_L[:, bb_idx[:, 0], :]
        ca = X_L[:, bb_idx[:, 1], :]
        c = X_L[:, bb_idx[:, 2], :]
        b = ca - n
        c_vec = c - ca
        a = torch.cross(b, c_vec, dim=-1)
        return -0.57910144 * a + 0.5689693 * b - 0.5441217 * c_vec + ca

    def _contact_coords(self, X_L, f, X_ca):
        """Per-token contact atom, in preference order.

        1. the REAL predicted C-beta -- RFD3 is all-atom, CB is an actual
           predicted-and-supervised atom, so score it directly and let gradient
           reach the atom that determines packing;
        2. virtual C-beta rebuilt from N/CA/C, for glycine (no CB exists, and
           RFD1 synthesises one there too);
        3. C-alpha, for anything that is not a standard protein residue.
        """
        if not self.use_cb:
            return X_ca
        bb_idx = f.get("token_bb_idx")
        bb_valid = f.get("token_bb_valid")
        cb_idx = f.get("token_cb_idx")
        has_cb = f.get("token_has_cb")
        if bb_idx is None or bb_valid is None:
            return X_ca
        out = X_ca
        if bb_valid.any():
            out = torch.where(
                bb_valid[None, :, None], self._virtual_cb(X_L, bb_idx), out
            )
        if cb_idx is not None and has_cb is not None and has_cb.any():
            out = torch.where(has_cb[None, :, None], X_L[:, cb_idx, :], out)
        return out

    @staticmethod
    def _blocks_from_adjacency_rows(
        adj: torch.Tensor, chain_idx: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        """Recover contiguous token groups from adj_matrix's block-uniform rows.

        Every ground-truth fill (ss_adj.py) and every CFG-dropout mask/copy
        (ss_adj_transforms.py) operates on whole cross-block rectangles, so two
        adjacent tokens in the same original SS block always have identical
        rows in adj_matrix -- this holds in every CFG mode, unlike ss_labels.
        """
        I = adj.shape[0]
        same_as_prev = (adj[1:] == adj[:-1]).all(dim=-1)  # (I-1,) bool
        if chain_idx is not None:
            # Never merge across a chain break, mirroring _find_ss_blocks on the
            # label side. Without this a multi-chain binder example can fuse a
            # target block onto a designed block whenever their rows coincide.
            same_as_prev = same_as_prev & (chain_idx[1:] == chain_idx[:-1])
        blocks = []
        start = 0
        for i in range(1, I):
            if not bool(same_as_prev[i - 1]):
                blocks.append(torch.arange(start, i, device=adj.device))
                start = i
        blocks.append(torch.arange(start, I, device=adj.device))
        return blocks

    def forward(self, network_input, network_output, loss_input):
        f = network_input["f"]
        X_L = network_output["X_L"]  # (D, L, 3)
        D = X_L.shape[0]

        if "adj_matrix" not in f:
            zero = X_L.new_zeros(())
            return self.weight * zero, {}

        adj = f["adj_matrix"]  # (I, I) in {0, 1, 2}; symmetric, zero diagonal
        is_ca = f["is_ca"]  # (L,) bool; exactly I True entries, in token order

        X_ca = X_L[:, is_ca, :]  # (D, I, 3)
        I = X_ca.shape[1]
        assert adj.shape == (I, I), (
            f"adj_matrix shape {tuple(adj.shape)} does not match token count "
            f"{I} implied by f['is_ca']"
        )

        blocks = self._blocks_from_adjacency_rows(adj, f.get("token_chain_idx"))
        # Block pairs in which BOTH sides are fully-fixed motif are unlearnable when
        # detach_motif is on: every coordinate feeding them is detached, so they
        # contribute a constant with zero gradient. They are NOT free, though -- the
        # per-example loss is a MEAN, so each one enlarges the denominator and shrinks
        # the gradient on the pairs that can actually be learned. In binder design the
        # target's own internal fold supplies most of the conditioned contacts (a
        # 200-residue target easily gives 10-20x more target-internal pairs than
        # cross-interface ones), so the cross-interface signal -- the entire point of
        # the task -- was being averaged down by an order of magnitude.
        _fixed_tok_early = f.get("is_motif_token_with_fully_fixed_coord")
        _fixed_np = _fixed_tok_early.bool() if _fixed_tok_early is not None else None

        def _both_fixed(b1, b2) -> bool:
            if _fixed_np is None or not self.detach_motif:
                return False
            return bool(_fixed_np[b1].all()) and bool(_fixed_np[b2].all())

        block_pairs = []
        neg_block_pairs = []
        for bi in range(len(blocks)):
            b1 = blocks[bi]
            for bj in range(bi + 1, len(blocks)):
                b2 = blocks[bj]
                if _both_fixed(b1, b2):
                    continue
                rect = adj[b1][:, b2]
                if bool((rect == 1).any()):
                    block_pairs.append((b1, b2))
                elif self.negative_weight > 0.0 and bool((rect == 0).all()):
                    # Explicit non-contact only. `.all()` (not `.any()`) so a
                    # rectangle holding any masked (2) entry is left indeterminate
                    # rather than being treated as a revealed negative.
                    neg_block_pairs.append((b1, b2))

        if not block_pairs and not neg_block_pairs:
            zero = X_L.new_zeros(())
            return self.weight * zero, {"adjacency_contact_loss": zero.detach()}

        crd_mask_L = loss_input["crd_mask_L"]  # (L,) or (D, L)
        if crd_mask_L.dim() == 1:
            crd_mask_L = crd_mask_L.unsqueeze(0).expand(D, -1)
        valid_ca = crd_mask_L[:, is_ca].bool()  # (D, I)
        # Score C-beta (label's atom) when enabled; falls back to C-alpha per
        # token where no full N/CA/C backbone exists.
        X_ct = self._contact_coords(X_L, f, X_ca)
        # Hoisted: the steric arm needs this mask whether or not detach_motif is on.
        fixed_tok = f.get("is_motif_token_with_fully_fixed_coord")
        if self.detach_motif:
            # For a designed<->fixed-target pair, gradient should move the binder
            # to the target, not drag the pinned target toward the binder.
            if fixed_tok is not None:
                # Token-level (length I): built by apply_token_wise, which reduces
                # over each token via apply_segment_wise. Guard anyway rather than
                # broadcasting a wrong-length mask into silence.
                assert fixed_tok.shape[-1] == X_ct.shape[1], (
                    f"is_motif_token_with_fully_fixed_coord has length "
                    f"{fixed_tok.shape[-1]}, expected {X_ct.shape[1]} tokens"
                )
                X_ct = torch.where(
                    fixed_tok.bool()[None, :, None], X_ct.detach(), X_ct
                )

        per_bp_loss = []
        per_bp_valid = []
        for b1, b2 in block_pairs:
            xa = X_ct[:, b1, :]  # (D, |b1|, 3)
            xb = X_ct[:, b2, :]  # (D, |b2|, 3)
            dist = torch.linalg.norm(xa.unsqueeze(2) - xb.unsqueeze(1), dim=-1)  # (D, |b1|, |b2|)

            va = valid_ca[:, b1]  # (D, |b1|)
            vb = valid_ca[:, b2]  # (D, |b2|)
            valid = va.unsqueeze(2) & vb.unsqueeze(1)  # (D, |b1|, |b2|)

            # Large finite sentinel (not inf) for invalid entries -- keeps the
            # sigmoid argument a normal finite number instead of routing through
            # an inf-adjacent computation.
            dist_masked = dist.masked_fill(~valid, 1.0e6)
            min_dist = dist_masked.flatten(1).min(dim=-1).values  # (D,)
            any_valid = valid.flatten(1).any(dim=-1)  # (D,)

            # Flat-bottomed well: zero penalty anywhere in [min_distance, threshold].
            # The previous form, 1 - sigmoid((threshold - d) / softness), is monotonic
            # in d with its optimum at d -> 0, so it kept pulling conditioned blocks
            # together long after the contact was physically satisfied (loss 0.50 at
            # 6.0 A, 0.35 at 4.8, 0.25 at 3.8, 0.12 at 2.0). Measured consequence on
            # epoch-0670: mean minimum cross-block distance 3.81 A, against the
            # ~4.8-5.5 A that paired beta-strands actually adopt -- i.e. contacts
            # compressed a full Angstrom into near-clash range, which is where the
            # chain breaks and CA clashes came from (epoch-0645, trained without this
            # loss, has 0.00 of both).
            #
            # Both arms are squashed through 2*sigmoid(x)-1, which is 0 at x=0 and
            # saturates at 1, preserving the bounded-gradient property the sigmoid was
            # chosen for: at high t the x0-estimate is meaningless, and an unbounded
            # hinge there would inject a huge gradient the network cannot act on.
            too_far = torch.clamp(min_dist - self.threshold, min=0.0)
            too_close = torch.clamp(self.min_distance - min_dist, min=0.0)
            penalty = (2.0 * torch.sigmoid(too_far / self.softness) - 1.0) + (
                2.0 * torch.sigmoid(too_close / self.softness) - 1.0
            )
            per_bp_loss.append(penalty * any_valid)
            per_bp_valid.append(any_valid)

        if per_bp_loss:
            losses_stacked = torch.stack(per_bp_loss, dim=0)  # (n_bp, D)
            valid_stacked = torch.stack(per_bp_valid, dim=0).float()  # (n_bp, D)
            denom = valid_stacked.sum(dim=0).clamp(min=1)  # (D,)
            per_example_loss = (losses_stacked * valid_stacked).sum(dim=0) / denom
        else:
            # Negative-only example (revealed 0s but no revealed contacts). Keep the
            # per-example vector defined so the low_t/high_t split below still works.
            per_example_loss = X_L.new_zeros(D)
        loss = per_example_loss.mean()

        # --- repulsive arm: explicit non-contacts must stay non-adjacent ---
        # Mirror image of the attractive well, one-sided: zero once the pair is
        # beyond `negative_threshold`, so satisfied pairs exert no force and only
        # violators produce gradient. Averaged over VIOLATING pairs rather than all
        # negative pairs, so the term's magnitude does not shrink as the structure
        # grows (negative pairs scale ~quadratically with block count, and dividing
        # by all of them would silently anneal the penalty away on large systems).
        neg_loss = X_L.new_zeros(())
        n_violating = 0.0
        if neg_block_pairs:
            # NATIVE-MATCHED separation floor. A fixed threshold is unusable here:
            # with `loop_adj=False` (the pn_unit / interface datasets), every
            # loop-involved block pair is skipped by _adj_blocks_from_coords and so
            # is left at 0 -- i.e. labelled "explicit non-contact" even when the loop
            # is physically packed against a helix. Loops are ~half of all residues,
            # so a flat "push all 0-pairs past 6-8 A" would try to peel most of the
            # protein apart from itself.
            #
            # Instead require only that a 0-pair stays as far apart as it natively
            # is, capped at `negative_threshold` (beyond that we do not care -- the
            # pair is plainly non-adjacent). A loop natively touching at 4 A gets a
            # floor of ~3 A (anti-clash only); a pair natively 15 A apart gets the
            # full 8 A floor. This also sidesteps the CA/CB question: we compare a
            # predicted CA distance against the native CA distance, like with like,
            # rather than against a CB-derived threshold.
            X_gt_L = torch.nan_to_num(loss_input["X_gt_L_in_input_frame"])
            if X_gt_L.dim() == 2:
                X_gt_L = X_gt_L.unsqueeze(0)
            X_gt_ca = X_gt_L[..., is_ca, :].expand(D, -1, -1)
            X_gt_ct = self._contact_coords(X_gt_L.expand(D, -1, -1), f, X_gt_ca)
            neg_penalties = []
            neg_valids = []
            for b1, b2 in neg_block_pairs:
                xa = X_ct[:, b1, :]
                xb = X_ct[:, b2, :]
                dist = torch.linalg.norm(xa.unsqueeze(2) - xb.unsqueeze(1), dim=-1)
                va = valid_ca[:, b1]
                vb = valid_ca[:, b2]
                valid = va.unsqueeze(2) & vb.unsqueeze(1)
                dist_masked = dist.masked_fill(~valid, 1.0e6)
                min_dist = dist_masked.flatten(1).min(dim=-1).values  # (D,)
                any_valid = valid.flatten(1).any(dim=-1)

                with torch.no_grad():
                    ga = X_gt_ct[:, b1, :]
                    gb = X_gt_ct[:, b2, :]
                    gdist = torch.linalg.norm(ga.unsqueeze(2) - gb.unsqueeze(1), dim=-1)
                    gmin = gdist.masked_fill(~valid, 1.0e6).flatten(1).min(dim=-1).values
                    floor = torch.clamp(gmin - self.negative_margin, min=0.0)
                    floor = torch.minimum(
                        floor, torch.full_like(floor, self.negative_threshold)
                    )

                too_near = torch.clamp(floor - min_dist, min=0.0)
                neg_penalties.append(
                    (2.0 * torch.sigmoid(too_near / self.softness) - 1.0) * any_valid
                )
                neg_valids.append(any_valid & (too_near > 0))
            neg_stacked = torch.stack(neg_penalties, dim=0)  # (n_neg, D)
            active = torch.stack(neg_valids, dim=0).float()  # (n_neg, D)
            neg_denom = active.sum(dim=0).clamp(min=1)
            neg_loss = ((neg_stacked * active).sum(dim=0) / neg_denom).mean()
            n_violating = float(active.sum().item())

        # --- steric arm: the designed chain may not occupy the motif's volume ----
        # Independent of adjacency state, INCLUDING masked (2) pairs. "Masked" means
        # the adjacency is unspecified, not that overlap is permitted.
        #
        # Native-matched, like the repulsive arm and for the same reason: a designed
        # residue that natively packs against the motif at 3.8 A must not be pushed
        # to 3.5 A+margin. The floor is min(steric_floor, native - margin), so we
        # never demand MORE separation than the native structure itself has, and the
        # term can only ever forbid going closer than nature does.
        steric_loss = X_L.new_zeros(())
        n_clashing = 0.0
        if self.steric_weight > 0.0 and fixed_tok is not None:
            fixed_m = fixed_tok.bool()
            des_m = ~fixed_m
            if bool(fixed_m.any()) and bool(des_m.any()):
                X_gt_s = torch.nan_to_num(loss_input["X_gt_L_in_input_frame"])
                if X_gt_s.dim() == 2:
                    X_gt_s = X_gt_s.unsqueeze(0)
                X_gt_s = X_gt_s.expand(D, -1, -1)
                X_gt_ct_s = self._contact_coords(X_gt_s, f, X_gt_s[..., is_ca, :])

                xd = X_ct[:, des_m, :]                       # (D, n_des, 3)
                xf = X_ct[:, fixed_m, :].detach()            # pinned: never dragged
                d_sf = torch.cdist(xd, xf)                   # (D, n_des, n_fix)
                vd = valid_ca[:, des_m]
                vf = valid_ca[:, fixed_m]
                valid_sf = vd.unsqueeze(2) & vf.unsqueeze(1)

                with torch.no_grad():
                    g_sf = torch.cdist(
                        X_gt_ct_s[:, des_m, :], X_gt_ct_s[:, fixed_m, :]
                    )
                    floor_sf = torch.minimum(
                        torch.clamp(g_sf - self.steric_margin, min=0.0),
                        torch.full_like(g_sf, self.steric_floor),
                    )

                # NaN safety. Padded / absent tokens can carry NaN coordinates, and
                # `clamp(...) * valid_sf` does NOT sanitise them: NaN * 0 is NaN, so a
                # single padded pair poisons the whole sum. Use torch.where, and map
                # non-finite distances to a large finite value so they can never look
                # like a clash.
                floor_sf = torch.nan_to_num(floor_sf, nan=0.0)
                d_sf = torch.nan_to_num(d_sf, nan=1.0e6, posinf=1.0e6)
                overlap = torch.where(
                    valid_sf,
                    torch.clamp(floor_sf - d_sf, min=0.0),
                    torch.zeros_like(d_sf),
                )
                clashing = overlap > 0
                n_clashing = float(clashing.sum().item())
                # Normalise by the DESIGNED-TOKEN count, not the violating-pair count.
                # Dividing by violators makes the loss non-monotonic in the very thing
                # it punishes: an extra clash raises the numerator AND the denominator,
                # so more overlap can score LOWER. One clash of overlap 1.0 gives
                # 1.0/1 = 1.0; adding a second of overlap 0.2 gives (1.0+0.04)/2 = 0.52,
                # i.e. the model is rewarded for clashing more. Measured on a real pair:
                # 2.250 at 3.5 A separation but 1.516 at 3.0 A -- the penalty got WEAKER
                # as the clash got worse. The designed-token count does not move when
                # clashes appear, and still does not scale with system size.
                n_des = float(max(int(des_m.sum().item()), 1))
                steric_loss = overlap.pow(2).sum() / n_des

        # `weight` is applied once to the whole term at return, so express the
        # repulsive arm relative to it rather than scaling it twice.
        if self.negative_weight > 0.0:
            loss = loss + (self.negative_weight / max(self.weight, 1e-8)) * neg_loss
        if self.steric_weight > 0.0:
            loss = loss + (self.steric_weight / max(self.weight, 1e-8)) * steric_loss

        with torch.no_grad():
            t = network_input["t"]
            order = torch.argsort(t)
            half = D // 2
            sorted_loss = per_example_loss[order]
            low_t_loss = (
                sorted_loss[:half].mean() if half > 0 else sorted_loss.new_zeros(())
            )
            high_t_loss = sorted_loss[half:].mean()

        loss_dict = {
            "adjacency_contact_loss": loss.detach(),
            "adjacency_contact_loss_low_t": low_t_loss,
            "adjacency_contact_loss_high_t": high_t_loss,
            "adjacency_n_conditioned_block_pairs": torch.tensor(
                float(len(block_pairs)), device=X_L.device
            ),
            "adjacency_steric_loss": steric_loss.detach(),
            "adjacency_n_clashing_pairs": torch.tensor(
                n_clashing, device=X_L.device
            ),
        }
        return self.weight * loss, loss_dict
