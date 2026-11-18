"""TRIAD-R S2 model side: arm registry (A/B/C/Z) and the arm-C network with per-block interaction heads and feedback.

Upstream (``src/rfproteina``) stays UNMODIFIED; everything here is a subclass + registry rebinding, like ``pair_stream``.

Arms (see ``results/N002-*/S2_harness_spec.md`` sec. 2.1, 4):
  A   zero-init ``ring_embed`` + final head ``ring_head_*`` (``pair_stream.TriadJLocalLatentsTransformer``), only these train
  B   A + LoRA on the trunk attention (LoRA itself is trainer glue: ``rfproteina.utils.lora``; ``install_lora_unfreeze``)
  C   B + ``TriadRTransformer``: heads at trunk blocks 4/8/12 and zero-init feedback at 4/8 ("interaction reads AND writes z")
  Z   control: A-class network with every ring parameter frozen, stream forced all-MASK, no ring loss (``TriadJControlFlowMatcher``)

New parameter names all start with ``ring_`` (``ring_embed``, ``ring_head_norm/proj``, arm C: ``ring_blk.<i>.{0,1}``, ``ring_fb.<i>``), so
the pre-existing optimiser pattern ``^nn\\.ring_`` (Adam instead of Muon) and ``UNFREEZE_PATTERNS`` cover every arm without a second
pattern. (The spec's ``rfb`` / ``^nn\\.(ring_|rfb_)`` would NOT have matched ``nn.rfb.4.weight``: the module is called ``ring_fb``.)

Config (under ``nn.ring``; survives ``train.py:_adopt_bundle_architecture`` because ``nn`` is one of the adopted fields and
``hf_bundle_overrides.nn.ring`` is merged into it, and survives a checkpoint round trip because ``cfg_exp`` is in the hparams):
  n_states 23, embed_rows 25                  all arms
  head_blocks [4, 8, 12]                      arm C: pair-update blocks after which a symmetric 23-state head reads z
  feedback_blocks [4, 8]                      arm C: heads whose softmax is projected back into z on still-MASK pairs (subset of head_blocks)
  feedback_detach true                        arm C: stop-gradient on the soft prediction fed back (ablation flag)
``arm_nn_overrides(arm)`` returns exactly this block per arm; ``arm_hf_bundle_overrides(arm)`` wraps it for Hydra.
Loss side (``ring_flow``): ``loss.ring.{block_weight 0.5, block_weights, final_weight 1.0, loss_scale}``.

CLI:  python -m triadj_ext.model_s2 expand --arm C --src /w/rfd4_weights --dst /w/triadr_s2_base_C [--seed 0]
      python -m triadj_ext.model_s2 counts      (parameter counts per arm, needs the bundle)
"""
from __future__ import annotations

import os
import re
import sys

import torch
from torch import nn

from rfproteina.nn.modules.norm import make_norm
from triadj_ext.pair_stream import (RING_MASK, RING_N_STATES, TriadJLocalLatentsTransformer, load_triadj_proteina,
                                    new_param_names, register_specs)

ARMS = ("A", "B", "C", "Z")
DEFAULT_HEAD_BLOCKS = (4, 8, 12)
DEFAULT_FEEDBACK_BLOCKS = (4, 8)

# every new parameter of every arm; covers ring_embed, ring_head_*, ring_blk.*, ring_fb.*
UNFREEZE_PATTERNS = [r"^nn\.ring_"]
UNFREEZE_PATTERNS_BY_ARM = {"A": UNFREEZE_PATTERNS, "B": UNFREEZE_PATTERNS, "C": UNFREEZE_PATTERNS, "Z": []}

_CURRENT_ARM: str | None = None


def _int_tuple(x) -> tuple[int, ...]:
    return tuple(sorted({int(i) for i in (x or ())}))


class TriadRTransformer(TriadJLocalLatentsTransformer):
    """Arm C: ``TriadJLocalLatentsTransformer`` + per-block interaction heads with zero-init feedback.

    After ``z = pair_layers[i](s, z, seq_mask)`` for i in ``head_blocks`` (default 4, 8, 12):
      h_i = sym(Linear(RMSNorm(z)))                                  [b,n,n,23], stashed in ``out['interaction']['blocks'][i]``
      i in ``feedback_blocks``: z += ring_fb[i](softmax(h_i)) * (still-MASK pair)      (zero-init weights => z + 0 exactly)
    The hook sits in the eager block loop (``_apply_trunk``), outside every compiled block (``_setup_compiled_trunk`` compiles
    each ``transformer_layers[i]`` / ``pair_update_layers[i]`` separately; the python loop stays eager), implemented by wrapping
    the entries of the ``pair_layers`` list that ``_apply_trunk`` receives, so the upstream loop is not copied.
    Every recycle pass runs the hooks (no-grad passes too, because feedback changes z); only the last pass's logits are stashed.
    Invariant for ``nn.checkpointing: medium``: ``self._ms`` (the MASK map) is re-read when the checkpointed trunk is recomputed in
    backward, so backward must run before the next forward through this module (true for ``Proteina.training_step``).
    """

    def __init__(self, **kwargs):
        ring = dict(kwargs.get("ring") or {})  # the base class pops its own copy of this block
        heads = _int_tuple(ring.get("head_blocks", DEFAULT_HEAD_BLOCKS))
        feedback = _int_tuple(ring.get("feedback_blocks", DEFAULT_FEEDBACK_BLOCKS))
        detach = bool(ring.get("feedback_detach", True))
        super().__init__(**kwargs)
        if not set(feedback) <= set(heads):
            raise ValueError(f"nn.ring.feedback_blocks {feedback} must be a subset of head_blocks {heads}")
        if heads and not self.update_pair_repr:
            raise ValueError("nn.ring.head_blocks needs nn.update_pair_repr (the heads read the updated pair state)")
        for i in heads:
            if not (0 <= i < self.nlayers - 1) or self.pair_update_layers[i] is None:
                valid = [j for j in range(self.nlayers - 1) if self.pair_update_layers[j] is not None]
                raise ValueError(f"nn.ring.head_blocks: block {i} has no pair update; valid blocks {valid}")
        d, k = self.pair_repr_dim, self.ring_n_states
        self.head_blocks, self.feedback_blocks, self.feedback_detach = heads, feedback, detach
        self.ring_blk = nn.ModuleDict({str(i): nn.Sequential(make_norm(d), nn.Linear(d, k)) for i in heads})
        self.ring_fb = nn.ModuleDict({str(i): nn.Linear(k, d, bias=False) for i in feedback})
        for m in self.ring_fb.values():
            nn.init.zeros_(m.weight)  # <- step 0 reproduces the released model exactly (z + 0)
        self._ms: torch.Tensor | None = None
        self._blk: dict[int, torch.Tensor] = {}
        self._blk_open = False

    # -- per-pass state ------------------------------------------------------------------------------------------
    def _run_trunk(self, input, s_init, z_init, c, seq_mask):
        rs = input.get("ring_state")
        self._ms = None if rs is None else (rs == RING_MASK)  # kept after forward on purpose (checkpoint recompute reads it)
        return super()._run_trunk(input, s_init, z_init, c, seq_mask)

    def _forward_impl(self, input):
        self._blk, self._blk_open = {}, True
        try:
            out = super()._forward_impl(input)
        finally:
            self._blk_open = False  # a checkpoint recompute (in backward) must not overwrite the stash
        blocks, self._blk = self._blk, {}
        out["interaction"]["blocks"] = blocks
        return out

    # -- the hook ------------------------------------------------------------------------------------------------
    def _apply_trunk(self, s, z, c, seq_mask, attn_layers, pair_layers):
        if not self.head_blocks:
            return super()._apply_trunk(s, z, c, seq_mask, attn_layers, pair_layers)
        wrapped = list(pair_layers)
        for i in self.head_blocks:
            wrapped[i] = self._hooked(i, wrapped[i])
        return super()._apply_trunk(s, z, c, seq_mask, attn_layers, wrapped)

    def _hooked(self, i, layer):
        def run(s, z, seq_mask):
            return self._after_pair_block(i, layer(s, z, seq_mask), seq_mask)

        return run

    def _after_pair_block(self, i, z, seq_mask):
        key = str(i)
        h = self.ring_blk[key](z)
        h = 0.5 * (h + h.transpose(1, 2))  # symmetric logits, like the final head
        if self._blk_open:
            self._blk[i] = h
        if key in self.ring_fb and self._ms is not None:
            p = h.float().softmax(-1)
            if self.feedback_detach:
                p = p.detach()
            gate = (self._ms & (seq_mask[:, :, None] & seq_mask[:, None, :]))[..., None]  # still-MASK real pairs only
            fb = self.ring_fb[key]
            z = z + (fb(p.to(fb.weight.dtype)) * gate).to(z.dtype)
        return z


# ---- arm configuration ---------------------------------------------------------------------------------------------
def _norm_arm(arm: str | None) -> str:
    arm = (arm or os.environ.get("TRIADR_ARM") or "C").strip().upper()
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    return arm


def current_arm() -> str | None:
    """The arm last registered in THIS process (``TRIADR_ARM`` is also exported so spawned workers inherit it)."""
    return _CURRENT_ARM


def arm_nn_overrides(arm: str, **ring_kw) -> dict:
    """``nn`` overrides that build the arm's modules (for ``load_triadj_proteina(nn_overrides=...)``)."""
    arm = _norm_arm(arm)
    ring: dict = {"n_states": RING_N_STATES, "embed_rows": 25}
    if arm == "C":
        ring.update(head_blocks=list(DEFAULT_HEAD_BLOCKS), feedback_blocks=list(DEFAULT_FEEDBACK_BLOCKS), feedback_detach=True)
    ring.update(ring_kw)
    return {"ring": ring}


def arm_hf_bundle_overrides(arm: str, **ring_kw) -> dict:
    """The same block as ``hf_bundle_overrides`` (``train.py`` merges it into the bundle's ``nn``)."""
    return {"nn": arm_nn_overrides(arm, **ring_kw)}


def unfreeze_patterns(arm: str | None = None) -> list[str]:
    return list(UNFREEZE_PATTERNS_BY_ARM[_norm_arm(arm or _CURRENT_ARM)])


def unfreeze_new_params(model: nn.Module, arm: str | None = None) -> list[str]:
    """``requires_grad_(True)`` on every parameter matching the arm's patterns (LoRA's ``_inject`` freezes everything first).
    Returns the names switched on. Arm Z: nothing."""
    pats = [re.compile(p) for p in unfreeze_patterns(arm)]
    on = []
    for n, p in model.named_parameters():
        if any(r.search(n) for r in pats):
            p.requires_grad_(True)
            on.append(n)
    return on


def install_lora_unfreeze() -> None:
    """Wrap ``add_lora_adapter`` so the new modules stay trainable under LoRA (spec 2.1: ``_inject`` runs ``requires_grad_(False)``
    on the whole model). Rebinds the SOURCE (``rfproteina.utils.lora``, bound by ``train.py`` at its import), ``rfproteina.proteina``
    (``on_load_checkpoint`` re-adds adapters on resume/generate) and ``rfproteina.train`` when already imported. Idempotent."""
    import rfproteina.proteina as P
    import rfproteina.utils.lora as L

    if getattr(L.add_lora_adapter, "_triadr_wrapped", False):
        orig = L.add_lora_adapter._triadr_orig
    else:
        orig = L.add_lora_adapter

        def wrapped(model, config):
            orig(model, config)
            unfreeze_new_params(model)

        wrapped._triadr_wrapped, wrapped._triadr_orig = True, orig
        L.add_lora_adapter = wrapped
    P.add_lora_adapter = L.add_lora_adapter
    train = sys.modules.get("rfproteina.train")
    if train is not None:
        train.add_lora_adapter = L.add_lora_adapter


def register_arm(arm: str | None = None) -> str:
    """Bind the arm's network class and flow matcher into ``rfproteina.proteina`` (hard-coded module globals, proteina.py:418/421),
    add the collate SPECS and the LoRA-unfreeze wrapper. ``arm=None`` reads ``TRIADR_ARM`` (default ``C``) so a spawned worker that
    re-runs the entry module registers the same arm; the arm is exported to ``TRIADR_ARM``. Idempotent, safe to call in every process."""
    global _CURRENT_ARM
    arm = _norm_arm(arm)
    import rfproteina.proteina as P
    from triadj_ext.ring_flow import TriadJControlFlowMatcher, TriadJSamplerFlowMatcher

    register_specs()
    P.LocalLatentsTransformer = TriadRTransformer if arm == "C" else TriadJLocalLatentsTransformer
    P.ProductSpaceFlowMatcher = TriadJControlFlowMatcher if arm == "Z" else TriadJSamplerFlowMatcher
    install_lora_unfreeze()
    os.environ["TRIADR_ARM"] = arm
    _CURRENT_ARM = arm
    return arm


# ---- arm <-> checkpoint agreement (generation) ----------------------------------------------------------------------
def _ring_cfg_facts(nn_cfg) -> dict:
    ring = (nn_cfg or {}).get("ring") if hasattr(nn_cfg, "get") else None
    ring = dict(ring) if ring else None
    return {"has_ring": ring is not None, "has_blocks": bool(ring and ring.get("head_blocks"))}


def source_arm_info(cfg) -> dict:
    """What the model source of a generate config says about its arm, without loading tensors.
    ``cfg``: dict-like with ``hf_ckpt_path`` / ``ckpt_path`` / ``ckpt_name``. Returns
    ``{kind: 'ckpt'|'bundle'|'hub'|'unknown', has_ring, has_blocks, declared, declared_from, lora}``:
    ``declared`` = the arm label the trainer wrote into the checkpoint hparams (``cfg_exp.arm``, else the ``run_name_`` suffix
    ``..._<A|B|C|Z>``), None for bundles and for checkpoints without it; ``has_blocks`` = ``nn.ring.head_blocks`` present => arm C."""
    import json
    import re
    from pathlib import Path

    from omegaconf import OmegaConf

    info = {"kind": "unknown", "has_ring": False, "has_blocks": False, "declared": None, "declared_from": None, "lora": None, "path": None}
    hf = cfg.get("hf_ckpt_path") if hasattr(cfg, "get") else None
    if hf:
        d = Path(str(hf))
        if d.is_dir() and (d / "config.json").is_file():
            nn_cfg = json.loads((d / "config.json").read_text()).get("config", {}).get("nn", {})
            info.update(kind="bundle", path=str(d), **_ring_cfg_facts(nn_cfg))
        else:
            info.update(kind="hub", path=str(hf))  # a hub id is the stock release
        return info
    ckpt_path, ckpt_name = cfg.get("ckpt_path"), cfg.get("ckpt_name")
    if not ckpt_path or not ckpt_name:
        return info
    f = os.path.join(str(ckpt_path), str(ckpt_name))
    if not os.path.isfile(f):
        return info
    hp = torch.load(f, map_location="cpu", weights_only=False, mmap=True).get("hyper_parameters", {}) or {}
    c = hp.get("cfg_exp")
    if c is None:
        info.update(kind="ckpt", path=f)
        return info
    c = c if OmegaConf.is_config(c) else OmegaConf.create(c)
    nn_cfg = OmegaConf.to_container(c.get("nn"), resolve=True) if c.get("nn") is not None else {}
    info.update(kind="ckpt", path=f, **_ring_cfg_facts(nn_cfg), lora=bool((c.get("lora") or {}).get("r")))
    if c.get("arm"):
        info.update(declared=str(c.get("arm")).strip().upper(), declared_from="cfg_exp.arm")
    else:
        m = re.search(r"_([ABCZ])$", str(c.get("run_name_") or ""))
        if m:
            info.update(declared=m.group(1), declared_from="cfg_exp.run_name_")
    return info


def resolve_arm(requested: str | None, info: dict, expect: str | None = None) -> tuple[str, list[str]]:
    """Decide the arm to register for generation and check it against the source. -> (arm in ARMS or 'NONE', warnings).
    Raises ValueError on any disagreement. Rules:
      * ``nn.ring.head_blocks`` in the source <=> arm C (hard check both ways); no ring config <=> stock (``none``);
      * a trainer-written label (``declared``) must equal the arm; without a label A/B/Z cannot be told apart from tensors, so the arm
        MUST be given explicitly (``TRIADR_ARM``) and a loud ``ARM not verifiable`` warning is returned;
      * Z (control flow matcher, stream forced all-MASK) is never inferred: only an explicit request ``Z`` selects it;
      * ``expect`` (``TRIADR_EXPECT_ARM``, set by the eval sbatch from the run it scores) must equal the resolved arm."""
    warns: list[str] = []
    req = (requested or "").strip().upper() or None
    if req is not None and req not in ARMS + ("NONE",):
        raise ValueError(f"TRIADR_ARM must be one of {ARMS + ('none',)}, got {requested!r}")
    declared = info.get("declared")
    where = info.get("path") or info.get("kind")
    if req is None:  # derive, conservatively
        if info["kind"] in ("bundle", "hub", "ckpt") and not info["has_ring"]:
            req = "NONE"
        elif info["has_blocks"]:
            req = "C"
        elif declared in ("A", "B"):
            req = declared
        elif declared == "Z":
            raise ValueError(f"{where} declares arm Z (control): set TRIADR_ARM=Z explicitly (the control flow matcher is never inferred)")
        else:
            raise ValueError(f"ARM not verifiable for {where}: no arm label in the source and no nn.ring.head_blocks, so A/B/Z are "
                             f"indistinguishable; set TRIADR_ARM explicitly")
    if req == "NONE":
        if info["has_ring"]:
            raise ValueError(f"TRIADR_ARM=none (stock model) but {where} carries nn.ring modules: use the arm it was trained with")
    else:
        if not info["has_ring"]:
            raise ValueError(f"TRIADR_ARM={req} but {where} has no nn.ring config (stock release): use TRIADR_ARM=none")
        if req == "C" and not info["has_blocks"]:
            raise ValueError(f"TRIADR_ARM=C but {where} has no nn.ring.head_blocks (an A/B/Z model): the arm does not match the source")
        if req != "C" and info["has_blocks"]:
            raise ValueError(f"TRIADR_ARM={req} but {where} has nn.ring.head_blocks (arm C): the arm does not match the source")
        if declared is not None and declared != req:
            raise ValueError(f"TRIADR_ARM={req} but {where} was trained as arm {declared} ({info.get('declared_from')})")
        if declared is None and req != "C":
            warns.append(f"ARM not verifiable: {where} carries no arm label, A/B/Z cannot be told apart from its tensors; trusting "
                         f"TRIADR_ARM={req} (set TRIADR_EXPECT_ARM in the eval job to cross-check)")
    if expect is not None and expect.strip().upper() != req:
        raise ValueError(f"TRIADR_EXPECT_ARM={expect} but the resolved arm is {req} (source {where})")
    return req, warns


def guard_arm_for_args(arm: str, ckpt: str = "", bundle: str = "", expect: str | None = None) -> dict:
    """The arm check of ``register_for_generation`` for the offline evaluators that take ``--arm`` + ``--ckpt`` / ``--bundle`` (oracle_s2, cond_gen): the requested arm must
    agree with the checkpoint (trainer label in the hparams, ``nn.ring.head_blocks``); a Z request on an A checkpoint (or C on A, ...) raises ValueError BEFORE anything is built.
    Registers nothing. A STOCK bundle (no ``nn.ring``: the released model) is the legal step-0 control for any arm (the arm's modules are added at zero init) and is reported as such.
    Warnings (unlabelled A/B/Z source) are printed loudly. Returns {'arm', 'source', 'declared', 'verified'}."""
    import sys as _sys

    arm = _norm_arm(arm)
    if bool(ckpt) == bool(bundle):
        raise ValueError("guard_arm_for_args: give exactly one of ckpt / bundle")
    cfg = {"hf_ckpt_path": bundle} if bundle else {"hf_ckpt_path": None, "ckpt_path": os.path.dirname(os.path.abspath(ckpt)), "ckpt_name": os.path.basename(ckpt)}
    info = source_arm_info(cfg)
    if bundle and not info["has_ring"]:
        return {"arm": arm, "source": info.get("path") or bundle, "declared": None, "verified": "stock bundle: step-0 control, arm modules added at zero init"}
    if info["kind"] == "ckpt" and not info["has_ring"]:
        raise ValueError(f"--arm {arm} but {ckpt} carries no nn.ring config (stock model checkpoint)")
    got, warns = resolve_arm(arm, info, expect if expect is not None else os.environ.get("TRIADR_EXPECT_ARM"))
    for w in warns:
        print(f"[triadr] WARNING {w}", file=_sys.stderr, flush=True)
    return {"arm": got, "source": info.get("path"), "declared": info.get("declared"), "verified": "label+ring" if info.get("declared") else "ring config only (ARM not verifiable for A/B/Z)"}


def register_for_generation(cfg, requested: str | None = None, expect: str | None = None) -> str:
    """``resolve_arm`` on ``source_arm_info(cfg)`` (env ``TRIADR_ARM`` / ``TRIADR_EXPECT_ARM`` by default), print the warnings loudly,
    register the arm (nothing for ``none``) and export ``TRIADR_ARM``. Called by ``generate_entry`` before the model is built."""
    import sys as _sys

    requested = requested if requested is not None else os.environ.get("TRIADR_ARM")
    expect = expect if expect is not None else os.environ.get("TRIADR_EXPECT_ARM")
    info = source_arm_info(cfg)
    arm, warns = resolve_arm(requested, info, expect)
    for w in warns:
        print(f"[triadr] WARNING {w}", file=_sys.stderr, flush=True)
    if arm == "NONE":
        os.environ["TRIADR_ARM"] = "none"
        print("[triadr] arm: none (stock model)", flush=True)
    else:
        register_arm(arm)
        print(f"[triadr] arm: {arm} (source {info.get('kind')}, label={info.get('declared')}, ring blocks={info.get('has_blocks')})", flush=True)
    return arm


# ---- expanded bundle + counts ---------------------------------------------------------------------------------------
def _json_diff(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(set(a) | set(b)) for d in _json_diff(a.get(k, "<absent>"), b.get(k, "<absent>"), f"{path}/{k}")]
    return [] if a == b else [path]


def expand_bundle(arm: str, src: str, dst: str, seed: int = 0) -> dict:
    """Released bundle + the arm's new keys (seeded random heads, zero embedding / feedback) written with ``save_pretrained``;
    ``hf_ckpt_path=<dst>`` then loads strictly (spec decision 3) and the LoRA/``weight_loading`` ban never fires. Arms A, B, Z share
    one bundle (same seed => the shared keys are bit-equal across A and C). The written ``config.json`` must differ from the source's only
    in ``nn.ring`` (plus the forced ``nn.use_torch_compile=false`` and the dropped ``weights_sha256`` of the released file): checked here, the differing paths are returned."""
    import json
    from pathlib import Path

    arm = register_arm(arm)
    torch.manual_seed(seed)
    model, res = load_triadj_proteina(src, nn_overrides=arm_nn_overrides(arm))
    new = new_param_names(model)
    model.save_pretrained(dst)
    a = json.loads((Path(src) / "config.json").read_text())
    b = json.loads((Path(dst) / "config.json").read_text())
    diff = _json_diff(a, b)
    ok_paths = ("/config/nn/use_torch_compile", "/weights_sha256")  # the sha of the RELEASED weights is not valid for the expanded file
    bad = [d for d in diff if not (d.startswith("/config/nn/ring") or d in ok_paths)]
    if bad:
        raise RuntimeError(f"expanded bundle config differs from the source outside nn.ring: {bad}")
    return {"arm": arm, "dst": str(dst), "missing_filled": sorted(res.missing_keys), "config_diff": diff,
            "n_new_params": sum(dict(model.named_parameters())[n].numel() for n in new)}


def param_counts(model: nn.Module) -> dict:
    new = set(new_param_names(model))
    ps = dict(model.named_parameters())
    return {"nn_total": sum(p.numel() for n, p in ps.items() if n.startswith("nn.")),
            "new_total": sum(ps[n].numel() for n in new),
            "new_by_module": {m: sum(ps[n].numel() for n in new if n.split(".", 2)[1] == m) for m in sorted({n.split(".", 2)[1] for n in new})}}


def _main(argv: list[str]) -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("expand")
    e.add_argument("--arm", required=True, choices=ARMS)
    e.add_argument("--src", default="/w/rfd4_weights")
    e.add_argument("--dst", required=True)
    e.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("counts")
    c.add_argument("--src", default="/w/rfd4_weights")
    a = ap.parse_args(argv)
    if a.cmd == "expand":
        print(json.dumps(expand_bundle(a.arm, a.src, a.dst, a.seed), indent=1))
    else:
        for arm in ("A", "C"):
            register_arm(arm)
            m, _ = load_triadj_proteina(a.src, nn_overrides=arm_nn_overrides(arm))
            print(arm, json.dumps(param_counts(m)))


if __name__ == "__main__":
    _main(sys.argv[1:])
