"""
Chunked pairwise embedding implementation for memory-efficient large structure processing.

This module provides memory-optimized versions of pairwise embedders that compute
only the pairs needed for sparse attention, reducing memory usage from O(L²) to O(L×k).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from rfd3.model.layers.blocks import PositionPairDistEmbedder, SinusoidalDistEmbed
from rfd3.model.layers.layer_utils import RMSNorm, linearNoBias


class ChunkedPositionPairDistEmbedder:
    """
    Memory-efficient version of PositionPairDistEmbedder that computes pairs on-demand.
    Uses a trained PositionPairDistEmbedder instance and shares the forward method.
    """

    def __init__(self, embedder_instance: PositionPairDistEmbedder):
        """
        Initialize the ChunkedPositionPairDistEmbedder from a parent PositionPairDistEmbedder instance.
        """
        self.embed_frame = embedder_instance.embed_frame
        if embedder_instance.embed_frame:
            self.process_d = embedder_instance.process_d
        self.process_inverse_dist = embedder_instance.process_inverse_dist
        self.process_valid_mask = embedder_instance.process_valid_mask
        self.forward = embedder_instance.forward

    def compute_pairs_chunked(
        self,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise embeddings for specific query-key pairs.

        Args:
            query_pos: Query positions [B, 3]
            key_pos: Key positions [B, k, 3]
            valid_mask: Valid pair mask [B, k, 1]

        Returns:
            P_sparse: Pairwise embeddings [B, k, c_atompair]
        """
        D_pairs = query_pos.unsqueeze(1) - key_pos

        if self.embed_frame:
            P_pairs = self.process_d(D_pairs) * valid_mask

            norm_sq = torch.linalg.norm(D_pairs, dim=-1, keepdim=True) ** 2
            inv_dist = 1 / (1 + norm_sq)
            P_pairs = P_pairs + self.process_inverse_dist(inv_dist) * valid_mask

            P_pairs = (
                P_pairs
                + self.process_valid_mask(valid_mask.to(P_pairs.dtype)) * valid_mask
            )
        else:
            norm_sq = torch.linalg.norm(D_pairs, dim=-1, keepdim=True) ** 2
            norm_sq = torch.clamp(norm_sq, min=1e-6)
            inv_dist = 1 / (1 + norm_sq)
            P_pairs = self.process_inverse_dist(inv_dist) * valid_mask
            P_pairs = (
                P_pairs
                + self.process_valid_mask(valid_mask.to(P_pairs.dtype)) * valid_mask
            )

        return P_pairs


class ChunkedSinusoidalDistEmbed:
    """
    Memory-efficient version of SinusoidalDistEmbed.
    Uses a trained SinusoidalDistEmbed instance and shares the forward method.
    """

    def __init__(self, embedder_instance: SinusoidalDistEmbed):
        """
        Initialize the ChunkedSinusoidalDistEmbed from a parent SinusoidalDistEmbed instance.
        """
        assert (
            embedder_instance.c_atompair % 2 == 0
        ), "Output embedding dim must be even"

        self.n_freqs = embedder_instance.n_freqs
        self.c_atompair = embedder_instance.c_atompair
        self.output_proj = embedder_instance.output_proj
        self.process_valid_mask = embedder_instance.process_valid_mask
        self.forward = embedder_instance.forward

    def compute_pairs_chunked(
        self,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sinusoidal distance embeddings for specific query-key pairs.
        """
        device = query_pos.device

        D_pairs = query_pos.unsqueeze(1) - key_pos
        dist_matrix = torch.linalg.norm(D_pairs, dim=-1)

        half_dim = self.n_freqs
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half_dim, dtype=torch.float32, device=device)
            / half_dim
        )

        angles = dist_matrix.unsqueeze(-1) * freq
        sin_embed = torch.sin(angles)
        cos_embed = torch.cos(angles)
        sincos_embed = torch.cat([sin_embed, cos_embed], dim=-1)

        P_pairs = self.output_proj(sincos_embed)
        P_pairs = P_pairs * valid_mask

        P_pairs = (
            P_pairs + self.process_valid_mask(valid_mask.to(P_pairs.dtype)) * valid_mask
        )

        return P_pairs


class ChunkedPairwiseEmbedder:
    """
    Main chunked pairwise embedder that combines all embedding types.
    This replaces the full P_LL computation with sparse computation.

    Not an nn.Module: all sub-components are shared references from TokenInitializer
    and already registered there. Inheriting nn.Module would cause them to appear
    under duplicate paths (chunked_pairwise_embedder.*) in state_dict, which don't
    exist in checkpoints trained without use_chunked_pll.
    """

    def __init__(
        self,
        c_atompair: int,
        motif_pos_embedder: ChunkedSinusoidalDistEmbed,
        ref_pos_embedder: ChunkedPositionPairDistEmbedder,
        process_single_l: Optional[nn.Module] = None,
        process_single_m: Optional[nn.Module] = None,
        process_z: Optional[nn.Module] = None,
        pair_mlp: Optional[nn.Module] = None,
        **kwargs,
    ):
        self.c_atompair = c_atompair
        self.motif_pos_embedder = motif_pos_embedder
        self.ref_pos_embedder = ref_pos_embedder

        if process_single_l is not None:
            self.process_single_l = process_single_l
        else:
            self.process_single_l = nn.Sequential(
                nn.ReLU(), linearNoBias(128, c_atompair)
            )

        if process_single_m is not None:
            self.process_single_m = process_single_m
        else:
            self.process_single_m = nn.Sequential(
                nn.ReLU(), linearNoBias(128, c_atompair)
            )

        if process_z is not None:
            self.process_z = process_z
        else:
            self.process_z = nn.Sequential(RMSNorm(128), linearNoBias(128, c_atompair))

        if pair_mlp is not None:
            self.pair_mlp = pair_mlp
        else:
            self.pair_mlp = nn.Sequential(
                nn.ReLU(),
                linearNoBias(c_atompair, c_atompair),
                nn.ReLU(),
                linearNoBias(c_atompair, c_atompair),
                nn.ReLU(),
                linearNoBias(c_atompair, c_atompair),
            )

        self._sl_cached: Optional[torch.Tensor] = None
        self._sm_cached: Optional[torch.Tensor] = None
        self._Z_proc_cached: Optional[torch.Tensor] = None

    def cache_static_projections(
        self, C_L: torch.Tensor, Z_init_II: torch.Tensor
    ) -> None:
        """
        Precompute and cache the three MLP projections that are identical across
        all diffusion steps (they depend only on the static tokenization outputs).

        Call this once after tokenization, before the diffusion loop.
        forward_chunked will then replace those MLP calls with free tensor indexing.

        Args:
            C_L:       Atom features [L, c_token]
            Z_init_II: Token-pair features [I, I, c_z]
        """
        self._sl_cached = self.process_single_l(C_L)
        self._sm_cached = self.process_single_m(C_L)
        self._Z_proc_cached = self.process_z(Z_init_II)

    def forward_chunked(
        self,
        f: dict,
        indices: torch.Tensor,
        C_L: torch.Tensor,
        Z_init_II: torch.Tensor,
        tok_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute P_LL only for the pairs specified by attention indices.

        When cache_static_projections() has been called beforehand, the three MLP
        terms (process_single_l, process_single_m, process_z) are replaced with
        free tensor index operations, since those projections are identical across
        all diffusion steps.

        Args:
            f:         Feature dictionary (motif_pos, ref_pos, etc.)
            indices:   Sparse attention indices [B, L, k]
            C_L:       Atom-level features [L, c_token] or [B, L, c_token]
            Z_init_II: Token-level pair features [I, I, c_z]
            tok_idx:   Atom-to-token mapping [L]

        Returns:
            P_LL_sparse: Sparse pairwise features [B, L, k, c_atompair]
        """
        B, L, k = indices.shape
        device = indices.device

        P_LL_sparse = torch.zeros(
            B, L, k, self.c_atompair, device=device, dtype=C_L.dtype
        )

        if C_L.dim() == 2:
            C_L = C_L.unsqueeze(0)
        L_max = C_L.shape[1]
        valid_indices = torch.clamp(
            indices, 0, L_max - 1
        )

        if valid_indices.dim() == 2:
            valid_indices = valid_indices.unsqueeze(0).expand(
                C_L.shape[0], -1, -1
            )

        if self.motif_pos_embedder is not None and "motif_pos" in f:
            motif_pos = f["motif_pos"]
            is_motif = f["is_motif_atom_with_fixed_coord"]
            is_motif_idx = torch.where(is_motif)[0]
            for l in is_motif_idx:
                key_indices = valid_indices[:, l, :]
                key_pos = motif_pos[key_indices]
                query_pos = motif_pos[l].unsqueeze(0).expand(B, -1)

                key_is_motif = is_motif[key_indices]
                valid_mask = key_is_motif.unsqueeze(-1).float()

                if valid_mask.sum() > 0:
                    motif_pairs = self.motif_pos_embedder.compute_pairs_chunked(
                        query_pos, key_pos, valid_mask
                    )
                    P_LL_sparse[:, l, :, :] += motif_pairs

        if self.ref_pos_embedder is not None and "ref_pos" in f:
            ref_pos = f["ref_pos"]
            ref_space_uid = f["ref_space_uid"]
            is_motif_seq = f["is_motif_atom_with_fixed_seq"]
            is_motif_seq_idx = torch.where(is_motif_seq)[0]
            for l in is_motif_seq_idx:
                key_indices = valid_indices[:, l, :]
                key_pos = ref_pos[key_indices]
                query_pos = ref_pos[l].unsqueeze(0).expand(B, -1)

                key_space_uid = ref_space_uid[key_indices]
                key_is_motif_seq = is_motif_seq[key_indices]

                same_token = key_space_uid == ref_space_uid[l]
                valid_mask = (
                    (same_token & key_is_motif_seq).unsqueeze(-1).float()
                )

                if valid_mask.sum() > 0:
                    ref_pairs = self.ref_pos_embedder.compute_pairs_chunked(
                        query_pos, key_pos, valid_mask
                    )
                    P_LL_sparse[:, l, :, :] += ref_pairs

        if self._sl_cached is not None:
            single_l = self._sl_cached.unsqueeze(0).unsqueeze(2).expand(B, -1, k, -1)
            single_m = self._sm_cached[valid_indices]
        else:
            if C_L.shape[0] != B:
                C_L = C_L.expand(B, -1, -1)
            C_L_queries = C_L.unsqueeze(2).expand(-1, -1, k, -1)
            C_L_keys = torch.gather(
                C_L_queries,
                1,
                valid_indices.unsqueeze(-1).expand(-1, -1, -1, C_L.shape[-1]),
            )
            single_l = self.process_single_l(C_L_queries)
            single_m = self.process_single_m(C_L_keys)
        P_LL_sparse += single_l + single_m

        if tok_idx.dim() == 1:
            tok_idx_expanded = tok_idx.unsqueeze(0)
        else:
            tok_idx_expanded = tok_idx

        if tok_idx_expanded.shape[0] != B:
            tok_idx_expanded = tok_idx_expanded.expand(B, -1)
        tok_queries = tok_idx_expanded.unsqueeze(2).expand(-1, -1, k)
        tok_keys = torch.gather(tok_queries, 1, valid_indices)

        if self._Z_proc_cached is not None:
            Z_processed = self._Z_proc_cached
        else:
            Z_processed = self.process_z(Z_init_II)

        I_z, I_z2 = Z_processed.shape[:2]
        Z_pairs_processed = torch.zeros(
            B, L, k, self.c_atompair, device=device, dtype=Z_processed.dtype
        )
        for b in range(B):
            tq = torch.clamp(tok_queries[b], 0, I_z - 1)
            tk = torch.clamp(tok_keys[b], 0, I_z2 - 1)
            Z_pairs_processed[b] = Z_processed[tq, tk]

        P_LL_sparse += Z_pairs_processed

        P_LL_sparse = P_LL_sparse + self.pair_mlp(P_LL_sparse)

        return P_LL_sparse.contiguous()


def create_chunked_embedders(
    c_atompair: int, embed_frame: bool = True
) -> ChunkedPairwiseEmbedder:
    """
    Factory function to create chunked pairwise embedder with standard components.
    """
    motif_pos_embedder = ChunkedSinusoidalDistEmbed(
        embedder_instance=SinusoidalDistEmbed(c_atompair, embed_frame)
    )
    ref_pos_embedder = ChunkedPositionPairDistEmbedder(
        embedder_instance=PositionPairDistEmbedder(c_atompair, embed_frame)
    )

    return ChunkedPairwiseEmbedder(
        c_atompair=c_atompair,
        motif_pos_embedder=motif_pos_embedder,
        ref_pos_embedder=ref_pos_embedder,
    )
