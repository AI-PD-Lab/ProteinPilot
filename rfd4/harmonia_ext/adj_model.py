"""HARMONIA ADJ input for the released RFD4-Proteina denoiser (no upstream file edited).

``adj_embed`` = nn.Embedding(6, pair_repr_dim), ZERO-initialised. ``batch['adj_state']`` ([b,n,n] int, adj_labels states) is embedded
and ADDED to the pair input z0 inside ``_run_trunk`` (which re-adds z_init on every recycle pass), exactly like RFD3's ADJAdapter adds
into Z_init_II. States MASKED (2) and PAD (5) are multiplied by 0, so an unspecified map is the released model bit-for-bit, and a
missing ``adj_state`` key is the released model too. SS needs no new module: RFD4's own C_SEC channel (SecondaryStructureSeqFeat,
0 none / 1 helix / 2 sheet / 3 loop) carries it; it is unfrozen during the fine-tune (see train_entry.UNFREEZE_PATTERNS).

Registry: ``Proteina.__init__`` builds ``LocalLatentsTransformer`` through the module global of rfproteina.proteina, so ``register()``
rebinds it (every entry point: from_pretrained, load_from_checkpoint, train.py then builds the subclass).
"""
from __future__ import annotations

import torch
from torch import nn

from rfproteina.nn.local_latents_transformer import LocalLatentsTransformer

from harmonia_ext.adj_labels import ADJ_PAD, N_ADJ_ROWS, ZERO_STATES


class HarmoniaLocalLatentsTransformer(LocalLatentsTransformer):
    """Released denoiser + zero-init block-adjacency pair input."""

    def __init__(self, **kwargs):
        kwargs.pop("adj", None)                          # optional nn.adj config block; not forwarded to the base class
        super().__init__(**kwargs)
        self.adj_embed = nn.Embedding(N_ADJ_ROWS, self.pair_repr_dim)
        nn.init.zeros_(self.adj_embed.weight)            # step 0 reproduces the released model exactly

    def _run_trunk(self, input, s_init, z_init, c, seq_mask):
        a = input.get("adj_state")
        if a is not None:
            a = a.long()
            keep = seq_mask[:, :, None] & seq_mask[:, None, :]
            for zs in ZERO_STATES:
                keep = keep & (a != zs)
            e = self.adj_embed(a.clamp(0, N_ADJ_ROWS - 1))
            z_init = z_init + (e * keep[..., None]).to(z_init.dtype)
        return super()._run_trunk(input, s_init, z_init, c, seq_mask)


def register_specs() -> None:
    """Teach RFD4's collate the [I, I] key (padded with ADJ_PAD). Idempotent; runs at import (spawned loader workers)."""
    from rfproteina.datapipes.schema import proteina as schema

    schema.SPECS.setdefault("adj_state", (torch.int16, ADJ_PAD, "I I"))


def register() -> None:
    import rfproteina.proteina as P

    if not (isinstance(P.LocalLatentsTransformer, type) and issubclass(P.LocalLatentsTransformer, HarmoniaLocalLatentsTransformer)):
        P.LocalLatentsTransformer = HarmoniaLocalLatentsTransformer
    register_specs()


register_specs()
