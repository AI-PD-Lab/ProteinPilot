import os

import hydra
import torch
from omegaconf import DictConfig
from rfd3.model.cfg_utils import (
    strip_f,
)
from rfd3.model.inference_sampler import ConditionalDiffusionSampler
from rfd3.model.layers.encoders import TokenInitializer
from rfd3.model.ss_adj_cfg import apply_cfg_defaults
from torch import nn

from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def _cfg_is_noop(f: dict, f_ref: dict) -> bool:
    """True when stripping produced an input identical to the conditional one.

    Guards against paying for a second forward pass whose guidance delta is
    provably zero. Compares only keys present in either dict; a key dropped by
    strip_f counts as a difference.
    """
    keys = set(f) | set(f_ref)
    for k in keys:
        a, b = f.get(k), f_ref.get(k)
        if (a is None) != (b is None):
            return False
        if a is None:
            continue
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape or not torch.equal(a, b):
                return False
        elif a is not b and a != b:
            return False
    return True


class RFD3(nn.Module):
    """
    Simplified model for generation
    This module level serves to wrap the diffusion module of AF3
    to be roughly equivalent to the AF3 model w/o trunk processing.

    Allows the same sampler to be used
    """

    def __init__(
        self,
        *,
        c_s: int,
        c_z: int,
        c_atom: int,
        c_atompair: int,
        token_initializer: DictConfig | dict,
        diffusion_module: DictConfig | dict,
        inference_sampler: DictConfig | dict,
        **_,
    ):
        super().__init__()
        use_chunked_pll = os.environ.get("RFD3_LOW_MEMORY_MODE", None) == "1"
        ranked_logger.info(f"RFD3 initialized with chunked_pll={use_chunked_pll}")

        self.token_initializer = TokenInitializer(
            c_s=c_s,
            c_z=c_z,
            c_atom=c_atom,
            c_atompair=c_atompair,
            use_chunked_pll=use_chunked_pll,
            **token_initializer,
        )

        self.diffusion_module = hydra.utils.instantiate(
            diffusion_module, c_atom=c_atom, c_atompair=c_atompair, c_s=c_s, c_z=c_z
        )

        self.use_classifier_free_guidance = (
            inference_sampler["use_classifier_free_guidance"]
            and inference_sampler["cfg_scale"] != 1.0
        )
        self.cfg_features = inference_sampler.pop("cfg_features", [])

        self.inference_sampler = ConditionalDiffusionSampler(**inference_sampler)

    def forward(
        self,
        input: dict,
        coord_atom_lvl_to_be_noised: torch.Tensor = None,
        n_cycle=None,
        **_,
    ) -> dict:
        initializer_outputs = self.token_initializer(input["f"])

        if self.training:
            return self.diffusion_module(
                X_noisy_L=input["X_noisy_L"],
                t=input["t"],
                f=input["f"],
                n_recycle=n_cycle,
                **initializer_outputs,
            )
        else:
            if self.use_classifier_free_guidance:
                f_ref = strip_f(input["f"], self.cfg_features)
                f_ref = apply_cfg_defaults(f_ref, self.cfg_features)
                if _cfg_is_noop(input["f"], f_ref):
                    f_ref = None
                    ref_initializer_outputs = None
                else:
                    ref_initializer_outputs = self.token_initializer(f_ref)
            else:
                f_ref = None
                ref_initializer_outputs = None

            return self.inference_sampler.sample_diffusion_like_af3(
                f=input["f"],
                f_ref=f_ref,
                diffusion_module=self.diffusion_module,
                diffusion_batch_size=coord_atom_lvl_to_be_noised.shape[0],
                coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
                initializer_outputs=initializer_outputs,
                ref_initializer_outputs=ref_initializer_outputs,
            )
