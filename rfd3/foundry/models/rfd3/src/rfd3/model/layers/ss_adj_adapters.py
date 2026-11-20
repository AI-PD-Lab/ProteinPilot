"""
Zero-initialized additive adapters for SS/ADJ fold conditioning (RFD3).

Both modules initialize all weights to zero, making checkpoint warm-start safe:
loading a pre-trained RFD3 checkpoint without these keys present works because
the adapters contribute nothing until trained.

SSAdapter  → injects per-residue secondary structure into S_I   [I, c_s]
ADJAdapter → injects pairwise adjacency matrix into Z_init_II   [I, I, c_z]
"""

from __future__ import annotations

import torch
import torch.nn as nn

_SS_HELIX = 0
_SS_STRAND = 1
_SS_LOOP = 2
_SS_MASKED = 3


class SSAdapter(nn.Module):
    """Per-residue SS embedding, additively injected into S_I.

    Architecture:
      - Embedding table: 4 entries (helix=0, strand=1, loop=2, masked=3)
      - All weights zero-initialized → safe pretrained checkpoint warm-start
      - Masked positions (label==3) always produce zero output regardless of
        embedding weight values, enforced by explicit zeroing post-lookup.

    Usage in TokenInitializer.init_tokens():
        if self.ss_adapter is not None and "ss_labels" in f:
            S_I = S_I + self.ss_adapter(f["ss_labels"])
    """

    def __init__(self, c_s: int):
        super().__init__()
        self.embed = nn.Embedding(4, c_s)
        nn.init.zeros_(self.embed.weight)

    def forward(self, ss_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ss_labels: [I] long, values in {0,1,2,3}

        Returns:
            [I, c_s] float — zero for masked positions (label==3)
        """
        safe_labels = ss_labels.clamp(0, 3)
        out = self.embed(safe_labels)
        not_masked = (ss_labels != _SS_MASKED).float().unsqueeze(-1)
        return out * not_masked


class ADJAdapter(nn.Module):
    """Three-state pairwise adjacency embedding for Z_init_II.

    States match RFdiffusion1: 0=explicit non-contact, 1=contact,
    2=masked/unspecified. The masked state always contributes zero.
    """

    def __init__(self, c_z: int):
        super().__init__()
        self.embed = nn.Embedding(3, c_z)
        nn.init.zeros_(self.embed.weight)

    def forward(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        safe_states = adj_matrix.long().clamp(0, 2)
        out = self.embed(safe_states)
        not_masked = (safe_states != 2).float().unsqueeze(-1)
        return out * not_masked
