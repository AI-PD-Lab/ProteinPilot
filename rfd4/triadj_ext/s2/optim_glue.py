"""Optimiser / LoRA glue of the S2 fine-tuning run (no upstream file is edited; module globals are rebound, as ``ring_flow`` does).

(a) ``install_lora_unfreeze``  ``rfproteina.utils.lora._inject`` runs ``model.requires_grad_(False)`` on the WHOLE model, which would freeze
    the new ``ring_*`` modules under LoRA. ``add_lora_adapter`` is looked up as a module global at call time in BOTH
    ``rfproteina.train`` (fresh start / resume, train.py:313) and ``rfproteina.proteina`` (``on_load_checkpoint``, proteina.py:510-514, used
    by ``load_from_checkpoint`` / generate): both are rebound to a wrapper that calls the original and re-enables ``requires_grad`` on every
    parameter matching the UNFREEZE patterns (``triadj_ext.model_s2.UNFREEZE_PATTERNS``; an arm without new trainable modules, i.e. Z,
    contributes none).
(b) ``install_two_group_adam``  ``build_optimizers`` (proteina.py, called from ``configure_optimizers``) uses ONE ``opt.lr`` for Muon and Adam.
    A fresh head and a zero-init embedding need O(0.1) movement, which lr 5e-5 cannot deliver in 2-6k steps, so the Adam optimiser is rebuilt
    with two parameter groups: the new modules at ``opt.new_module_lr_mult x lr``, everything else at ``lr``. The ``LambdaLR`` warmup
    (proteina.py:497-503) multiplies each group's own base lr, so the ratio is constant. The new-module patterns are also appended to
    ``opt.muon_exclude_param_patterns`` (they must stay on Adam: Muon on an embedding / zero-init matrix is wrong).
"""
from __future__ import annotations

import re
from typing import Callable, Iterable

import torch

DEFAULT_NEW_PATTERNS = (r"^nn\.ring_",)


def new_module_patterns() -> list[str]:
    """Regexes of the new modules: ``model_s2.UNFREEZE_PATTERNS`` when C2's module is importable, else ``DEFAULT_NEW_PATTERNS``."""
    try:
        from triadj_ext.model_s2 import UNFREEZE_PATTERNS

        return list(UNFREEZE_PATTERNS) or list(DEFAULT_NEW_PATTERNS)
    except Exception:  # noqa: BLE001  (a stub-only test environment)
        return list(DEFAULT_NEW_PATTERNS)


def _active_patterns() -> list[str]:
    """Patterns to RE-ENABLE after a LoRA injection: arm-aware when ``model_s2.unfreeze_patterns`` exists (arm Z -> none)."""
    try:
        from triadj_ext import model_s2

        return list(model_s2.unfreeze_patterns())
    except Exception:  # noqa: BLE001
        return new_module_patterns()


def unfreeze_matching(model: torch.nn.Module, patterns: Iterable[str]) -> list[str]:
    pats = [re.compile(p) for p in patterns]
    on = []
    for n, p in model.named_parameters():
        if any(r.search(n) for r in pats):
            p.requires_grad_(True)
            on.append(n)
    return on


# ---------------------------------------------------------------------------------------------------------------- (a) LoRA
def install_lora_unfreeze(patterns_fn: Callable[[], Iterable[str]] | None = None) -> None:
    """Rebind ``add_lora_adapter`` in ``rfproteina.train`` and ``rfproteina.proteina`` (and the source ``rfproteina.utils.lora`` binding of
    modules imported later) to a wrapper that re-enables the new modules. Idempotent per target; stacks harmlessly on top of
    ``model_s2.install_lora_unfreeze`` (both just set ``requires_grad`` True)."""
    import rfproteina.proteina as P
    import rfproteina.train as T

    patterns_fn = patterns_fn or _active_patterns
    for mod in (T, P):
        cur = mod.add_lora_adapter
        if getattr(cur, "_s2_wrapped", False):
            cur._s2_patterns_fn = patterns_fn                      # idempotent: only refresh the pattern source
            continue
        mod.add_lora_adapter = _wrap(cur, patterns_fn)


def _wrap(orig, patterns_fn):
    def add_lora_adapter(model, config):
        orig(model, config)
        on = unfreeze_matching(model, add_lora_adapter._s2_patterns_fn())
        print(f"S2 lora: re-enabled {len(on)} new-module parameters after the adapter injection", flush=True)

    add_lora_adapter._s2_wrapped = True
    add_lora_adapter._s2_orig = orig
    add_lora_adapter._s2_patterns_fn = patterns_fn
    return add_lora_adapter


# ---------------------------------------------------------------------------------------------------------- (b) two-group Adam
def split_adam(opt: torch.optim.Adam, new_ids: set[int], lr: float, mult: float) -> torch.optim.Adam:
    """Rebuild a freshly constructed Adam with two groups (old params at ``lr``, params in ``new_ids`` at ``lr * mult``); same defaults."""
    params = [p for g in opt.param_groups for p in g["params"]]
    new = [p for p in params if id(p) in new_ids]
    old = [p for p in params if id(p) not in new_ids]
    groups = ([{"params": old, "lr": lr}] if old else []) + ([{"params": new, "lr": lr * mult}] if new else [])
    kw = {k: v for k, v in opt.defaults.items() if k not in ("params", "lr")}
    return torch.optim.Adam(groups, lr=lr, **kw)


def build_two_group(inner, named_parameters, cfg_opt, patterns: list[str] | None = None):
    """``inner(named, cfg)`` (the upstream ``build_optimizers``) with the new-module patterns appended to the Muon exclusion list and the
    resulting Adam split into two lr groups (``cfg_opt.new_module_lr_mult``, default 1.0 = no split)."""
    from omegaconf import OmegaConf

    patterns = list(patterns if patterns is not None else new_module_patterns())
    named = list(named_parameters)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg_opt, resolve=True))
    excl = list(cfg.get("muon_exclude_param_patterns") or [])
    for pat in patterns:
        if pat not in excl:
            excl.append(pat)
    cfg["muon_exclude_param_patterns"] = excl
    mult = float(cfg.get("new_module_lr_mult") or 1.0)
    opts = inner(named, cfg)
    regs = [re.compile(p) for p in patterns]
    new_ids = {id(p) for n, p in named if p.requires_grad and any(r.search(n) for r in regs)}
    if mult == 1.0 or not new_ids:
        return opts
    out = []
    for opt in opts:
        if isinstance(opt, torch.optim.Adam):
            opt = split_adam(opt, new_ids, float(cfg.lr), mult)
            sizes = [(len(g["params"]), sum(p.numel() for p in g["params"]), g["lr"]) for g in opt.param_groups]
            print(f"S2 optim: Adam groups (n_params, numel, lr) = {sizes}", flush=True)
        out.append(opt)
    return out


def install_two_group_adam(patterns: list[str] | None = None) -> None:
    """Wrap ``rfproteina.proteina.build_optimizers`` (global looked up by ``Proteina.configure_optimizers``). Idempotent."""
    import rfproteina.proteina as P

    if getattr(P.build_optimizers, "_s2_two_group", False):
        return
    inner = P.build_optimizers

    def wrapped(named_parameters, cfg_opt):
        return build_two_group(inner, named_parameters, cfg_opt, patterns)

    wrapped._s2_two_group = True
    wrapped._s2_inner = inner
    P.build_optimizers = wrapped
