"""TRIAD-J pair-label stream for the released RFD4-Proteina denoiser (spike S4).

Adds to ``rfproteina.nn.local_latents_transformer.LocalLatentsTransformer`` WITHOUT editing any upstream file:

  * ``ring_embed``  = nn.Embedding(25, pair_repr_dim), ZERO-initialised. ``batch['ring_state']`` ([b,n,n] int) is
    embedded and ADDED to the pair input z0 (right after ``PairReprBuilder``; implemented in ``_run_trunk``, which
    receives z0 as ``z_init`` and re-adds it to every recycle pass).
  * ``ring_head_*`` = RMSNorm -> Linear(pair_repr_dim, 23) on the FINAL pair state z of the last recycle pass,
    symmetrised over (i, j). Result: ``nn_out['interaction']['logits']`` [b,n,n,23].

State ids: 0..22 = the 23 interaction states, 23 = MASK (absorbing state of a masked-diffusion modality),
24 = PAD / not applicable. A missing ``batch['ring_state']`` key leaves the model exactly the released one.

Registry: ``Proteina.__init__`` hard-codes ``self.nn = LocalLatentsTransformer(**cfg_exp.nn, latent_dim=...)``
(rfproteina/proteina.py:421, module-global name), so ``register()`` rebinds that name; every entry point
(Proteina.from_pretrained, load_from_checkpoint, train.py:260 ``Proteina(cfg_exp)``) then builds the subclass.
"""
from __future__ import annotations

import torch
from torch import nn

from rfproteina.nn.local_latents_transformer import LocalLatentsTransformer
from rfproteina.nn.modules.norm import make_norm

RING_N_STATES = 23
RING_MASK = 23
RING_PAD = 24
RING_EMBED_ROWS = 25
NEW_PARAM_SUFFIXES = ("ring_embed.weight", "ring_head_norm.weight", "ring_head_proj.weight", "ring_head_proj.bias")


class TriadJLocalLatentsTransformer(LocalLatentsTransformer):
    """Released denoiser + zero-init pair-label input + symmetric pair-label head."""

    def __init__(self, **kwargs):
        ring = dict(kwargs.pop("ring", None) or {})  # optional nn.ring config block; not forwarded to the base class
        super().__init__(**kwargs)
        d = self.pair_repr_dim
        self.ring_n_states = int(ring.get("n_states", RING_N_STATES))
        self.ring_embed = nn.Embedding(int(ring.get("embed_rows", RING_EMBED_ROWS)), d)
        nn.init.zeros_(self.ring_embed.weight)  # <- step 0 reproduces the released model exactly
        self.ring_head_norm = make_norm(d)
        self.ring_head_proj = nn.Linear(d, self.ring_n_states)
        self._z_final: torch.Tensor | None = None

    # -- injection: z0 += E[ring_state] --------------------------------------------------------------------
    def _run_trunk(self, input, s_init, z_init, c, seq_mask):
        rs = input.get("ring_state")
        if rs is not None:
            pair_mask = seq_mask[:, :, None] & seq_mask[:, None, :]  # [b,n,n]
            e = self.ring_embed(rs.long())  # [b,n,n,d]
            z_init = z_init + (e * pair_mask[..., None]).to(z_init.dtype)
        s, z = super()._run_trunk(input, s_init, z_init, c, seq_mask)
        self._z_final = z  # base _forward_impl discards it (`s, _z = ...`, local_latents_transformer.py:296)
        return s, z

    # -- head: final z -> symmetric logits -----------------------------------------------------------------
    def _forward_impl(self, input):
        self._z_final = None
        nn_out = super()._forward_impl(input)
        z, self._z_final = self._z_final, None
        h = self.ring_head_proj(self.ring_head_norm(z))  # [b,n,n,23]
        nn_out["interaction"] = {"logits": 0.5 * (h + h.transpose(1, 2))}
        return nn_out


def register_specs() -> None:
    """Teach the collate the new keys (``SPECS`` pads ``[I, I]`` maps; a missing key is zero-filled per row). Idempotent;
    also runs at import so spawned dataloader workers that merely import this module are covered."""
    from rfproteina.datapipes.schema import proteina as schema

    schema.SPECS.setdefault("ring_state", (torch.int16, RING_PAD, "I I"))
    schema.SPECS.setdefault("ring_state_gt", (torch.int16, RING_PAD, "I I"))  # clean labels (loss target); pad 24 = unobserved
    schema.SPECS.setdefault("ring_obs", (torch.bool, False, "I I"))
    # validation-only per-row switch read by ring_flow.corrupt_batch: 0 tube (= training), 1 all-MASK, 2 full reveal. A scalar per
    # sample (dim string "" -> 0-dim tensor -> [b] after the stack); collate fills 0 for rows/datasets that do not set it.
    schema.SPECS.setdefault("ring_val_mode", (torch.int64, 0, ""))  # same spec as ring_data.register_ring_specs (whichever imports first wins)


def register() -> None:
    """Make Proteina build the subclass and let the collate know the new [I, I] key. Idempotent; never DOWNGRADES a more
    specific registration (``model_s2.register_arm('C')`` binds a subclass of ``TriadJLocalLatentsTransformer``)."""
    import rfproteina.proteina as P

    cur = P.LocalLatentsTransformer
    if not (isinstance(cur, type) and issubclass(cur, TriadJLocalLatentsTransformer)):
        P.LocalLatentsTransformer = TriadJLocalLatentsTransformer
    register_specs()


def new_param_names(model: nn.Module) -> list[str]:
    """Names (as in ``model.named_parameters()``) of every parameter this extension added: all start with ``ring_``
    (``ring_embed``, ``ring_head_*``; arm C adds ``ring_blk.*`` and ``ring_fb.*``)."""
    return [n for n, _ in model.named_parameters() if n.split("nn.", 1)[-1].startswith("ring_")]


def load_triadj_proteina(bundle: str, nn_overrides: dict | None = None, device: str = "cpu"):
    """``Proteina.from_pretrained`` for the subclass: identical config/weights path, but ``strict=False`` and then an
    EXACT check that the only missing keys are the new TRIAD-J ones (nothing unexpected)."""
    from omegaconf import OmegaConf

    import rfproteina.proteina as P
    from rfproteina.utils.hub import load_pretrained_bundle

    register()
    payload, state = load_pretrained_bundle(bundle)
    cfg = payload["config"]
    cfg.update({"run_name_": "pretrained", "autoencoder_ckpt_path": None, "precision": "32", "metrics": None,
                "datasets": {"train": {}, "val": {}}, "log": {"step_log_frequency": 100}})  # as Proteina.from_pretrained:333-343
    cfg["nn"] = {**cfg.get("nn", {}), **(nn_overrides or {})}
    model = P.Proteina(OmegaConf.create(cfg), load_training_metrics=False)
    res = model.load_state_dict(state, strict=False, assign=True)
    expected = {k for k in model.state_dict() if k.startswith("nn.ring_")}  # whatever the registered class added (arm A: 4 keys)
    assert not any(k.startswith("nn.ring_") for k in state), "the bundle already carries ring_ keys: use Proteina.from_pretrained"
    assert set(res.missing_keys) == expected, (sorted(set(res.missing_keys) ^ expected))
    assert not res.unexpected_keys, res.unexpected_keys
    return model.to(device).eval(), res


register_specs()  # import side effect, see register_specs
