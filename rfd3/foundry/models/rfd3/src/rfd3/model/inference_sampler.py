import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

import torch
from jaxtyping import Float
from rfd3.inference.symmetry.symmetry_utils import apply_symmetry_to_xyz_atomwise
from rfd3.model.cfg_utils import strip_X

from foundry.common import exists
from foundry.utils.alignment import weighted_rigid_align
from foundry.utils.ddp import RankedLogger
from foundry.utils.rotation_augmentation import (
    rot_vec_mul,
    uniform_random_rotation,
)

logging.basicConfig(level=logging.INFO)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def _adjacency_guidance_grad(
    X_denoised_L: torch.Tensor,
    f: dict[str, Any],
    threshold: float = 6.0,
    softness: float = 2.0,
) -> torch.Tensor | None:
    """Gradient of a bounded soft-contact potential w.r.t. X_denoised_L.

    Scores only token pairs where f["adj_matrix"] == 1 -- the same masking
    invariant as AdjacencyContactLoss (metrics/losses.py): every CFG-dropout
    masking path stamps 2, so a `1` can only appear where genuinely revealed.
    Returns None if there's nothing to guide toward.

    Treats the model's own denoised (x0) estimate as a cheap standalone input to
    a geometric potential -- no backprop through the network, just a local
    autograd graph over the potential itself (same pattern as RFdiffusion1's
    potentials.py / classifier guidance on x0-estimates). Bounded sigmoid score
    (not a raw hinge) so guidance degrades gracefully instead of exploding when
    the current x0-estimate is far off, e.g. at high noise before SS elements
    have taken shape.
    """
    if "adj_matrix" not in f or "is_ca" not in f:
        return None
    adj = f["adj_matrix"]
    is_ca = f["is_ca"]

    X_leaf = X_denoised_L.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        X_ca = X_leaf[:, is_ca, :]
        I = X_ca.shape[1]
        i_idx, j_idx = torch.triu_indices(I, I, offset=1, device=X_ca.device)
        pos_pairs = adj[i_idx, j_idx] == 1
        if not torch.any(pos_pairs):
            return None
        i_sel, j_sel = i_idx[pos_pairs], j_idx[pos_pairs]
        dist = torch.linalg.norm(X_ca[:, i_sel] - X_ca[:, j_sel], dim=-1)
        contact_score = torch.sigmoid((threshold - dist) / softness)
        potential = (1.0 - contact_score).mean()
        (grad,) = torch.autograd.grad(potential, X_leaf)
    return grad.detach()


@dataclass(kw_only=True)
class SampleDiffusionConfig:
    kind: Literal["default", "symmetry"] = "default"

    num_timesteps: int = 200
    min_t: int = 0
    max_t: int = 1
    sigma_data: int = 16
    s_min: float = 4e-4
    s_max: int = 160
    p: int = 7
    gamma_0: float = 0.6
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    solver: Literal["af3"] = "af3"

    center_option: str = "all"
    s_trans: float = 1.0
    s_jitter_origin: float = 0.0
    fraction_of_steps_to_fix_motif: float = 0.0
    skip_few_diffusion_steps: bool = False
    allow_realignment: bool = False
    insert_motif_at_end: bool = True
    use_classifier_free_guidance: bool = False
    cfg_scale: float = 2.0
    cfg_t_max: float | None = None

    adjacency_potential_scale: float = 0.0
    adjacency_potential_threshold: float = 6.0
    adjacency_potential_softness: float = 2.0

    n_recycle: int | None = None


class SampleDiffusionWithMotif(SampleDiffusionConfig):
    """Diffusion sampler that supports optional motif alignment."""

    def _construct_inference_noise_schedule(
        self, device: torch.device, partial_t: float = None
    ) -> torch.Tensor:
        """Constructs a noise schedule for use during inference.

        The inference noise schedule is defined in the AF-3 supplement as:

            t_hat = sigma_data * (s_max**(1/p) + t * (s_min**(1/p) - s_max**(1/p)))**p

        Returns:
            torch.Tensor: A tensor representing the noise schedule `t_hat`.

        Reference:
            AlphaFold 3 Supplement, Section 3.7.1.
        """
        t = torch.linspace(self.min_t, self.max_t, self.num_timesteps, device=device)

        t_hat = (
            self.sigma_data
            * (
                (self.s_max) ** (1 / self.p)
                + t * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))
            )
            ** self.p
        )

        if partial_t is not None:
            partial_t = float(partial_t.mean())
            noise_schedule = t_hat
            ranked_logger.info("Using partial diffusion with t={}".format(partial_t))

            original_schedule_len = len(noise_schedule)
            original_max = noise_schedule.max().item()
            original_min = noise_schedule.min().item()

            noise_schedule = noise_schedule[noise_schedule <= partial_t]

            new_schedule_len = len(noise_schedule)
            if new_schedule_len > 0:
                new_max = noise_schedule.max().item()
                new_min = noise_schedule.min().item()
                ranked_logger.info(
                    f"Noise schedule: {original_schedule_len} → {new_schedule_len} steps"
                )
                ranked_logger.info(
                    f"Original range: [{original_min:.3f}, {original_max:.3f}]"
                )
                ranked_logger.info(f"Filtered range: [{new_min:.3f}, {new_max:.3f}]")
            else:
                ranked_logger.warning(
                    f"No noise schedule steps found with t <= {partial_t}!"
                )
                ranked_logger.info(
                    f"Original schedule range: [{original_min:.3f}, {original_max:.3f}]"
                )
                noise_schedule_original = self._construct_inference_noise_schedule(
                    device=device
                )
                noise_schedule = noise_schedule_original[-1:]
                ranked_logger.info(
                    f"Using fallback: final step with t={noise_schedule[0].item():.6f}"
                )
        else:
            noise_schedule = t_hat

        return noise_schedule

    def _get_initial_structure(
        self,
        c0: torch.Tensor,
        D: int,
        L: int,
        coord_atom_lvl_to_be_noised: torch.Tensor,
        is_motif_atom_with_fixed_coord,
    ) -> torch.Tensor:
        noise = c0 * torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=c0.device)
        noise[..., is_motif_atom_with_fixed_coord, :] = 0
        X_L = noise + coord_atom_lvl_to_be_noised
        return X_L

    def sample_diffusion_like_af3(
        self,
        *,
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coord_atom_lvl_to_be_noised: Float[torch.Tensor, "D L 3"],
        initializer_outputs,
        ref_initializer_outputs: dict[str, Any] | None,
        f_ref: dict[str, Any] | None,
    ) -> dict[str, Any]:
        is_motif_atom_with_fixed_coord = f["is_motif_atom_with_fixed_coord"]

        noise_schedule = self._construct_inference_noise_schedule(
            device=coord_atom_lvl_to_be_noised.device,
            partial_t=f.get("partial_t", None),
        )

        L = f["ref_element"].shape[0]
        D = diffusion_batch_size

        X_L = self._get_initial_structure(
            c0=noise_schedule[0],
            D=D,
            L=L,
            coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised.clone(),
            is_motif_atom_with_fixed_coord=is_motif_atom_with_fixed_coord,
        )

        if self.s_jitter_origin > 0.0:
            X_L[:, is_motif_atom_with_fixed_coord, :] += torch.normal(
                mean=0.0,
                std=self.s_jitter_origin,
                size=(D, 1, 3),
                device=X_L.device,
            )

        X_noisy_L_traj = []
        X_denoised_L_traj = []
        sequence_entropy_traj = []
        t_hats = []

        threshold_step = (len(noise_schedule) - 1) * self.fraction_of_steps_to_fix_motif

        for step_num, (c_t_minus_1, c_t) in enumerate(
            zip(noise_schedule, noise_schedule[1:])
        ):
            assert not torch.is_grad_enabled(), "Computation graph should not be active"
            assert not X_L.requires_grad, "X_L should not require gradients"

            if self.allow_realignment:
                X_L, _ = centre_random_augment_around_motif(
                    X_L,
                    coord_atom_lvl_to_be_noised,
                    is_motif_atom_with_fixed_coord,
                    center_option=self.center_option,
                    centering_affects_motif=(max(step_num - 1, 0)) >= threshold_step,
                    s_trans=self.s_trans if step_num >= threshold_step else 0.0,
                )

            gamma = self.gamma_0 if c_t > self.gamma_min else 0
            step_scale = self.step_scale

            t_hat = c_t_minus_1 * (gamma + 1)

            epsilon_L = (
                self.noise_scale
                * torch.sqrt(torch.square(t_hat) - torch.square(c_t_minus_1))
                * torch.normal(mean=0.0, std=1.0, size=X_L.shape, device=X_L.device)
            )
            epsilon_L[..., is_motif_atom_with_fixed_coord, :] = (
                0
            )
            X_noisy_L = X_L + epsilon_L

            if "chunked_pairwise_embedder" in initializer_outputs:
                tic = time.time()
                chunked_embedder = initializer_outputs[
                    "chunked_pairwise_embedder"
                ]
                other_outputs = {
                    k: v
                    for k, v in initializer_outputs.items()
                    if k != "chunked_pairwise_embedder"
                }
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,
                    t=t_hat.tile(D),
                    f=f,
                    P_LL=None,
                    chunked_pairwise_embedder=chunked_embedder,
                    initializer_outputs=other_outputs,
                    n_recycle=self.n_recycle,
                    **other_outputs,
                )
                toc = time.time()
                ranked_logger.info(
                    f"[chunked] step {step_num}: {(toc - tic)*1000:.1f} ms"
                )
            else:
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,
                    t=t_hat.tile(D),
                    f=f,
                    n_recycle=self.n_recycle,
                    **initializer_outputs,
                )

            X_denoised_L = outs["X_L"] if "X_L" in outs else outs

            if self.adjacency_potential_scale != 0.0:
                grad = _adjacency_guidance_grad(
                    X_denoised_L,
                    f,
                    threshold=self.adjacency_potential_threshold,
                    softness=self.adjacency_potential_softness,
                )
                if grad is not None:
                    X_denoised_L = X_denoised_L - self.adjacency_potential_scale * grad

            delta_L = (
                X_noisy_L - X_denoised_L
            ) / t_hat
            d_t = c_t - t_hat

            if (
                self.use_classifier_free_guidance
                and f_ref is not None
                and (self.cfg_t_max is None or c_t > self.cfg_t_max)
            ):
                X_noisy_L_stripped = strip_X(X_noisy_L, f_ref)

                outs_ref = diffusion_module(
                    X_noisy_L=X_noisy_L_stripped,
                    t=t_hat.tile(D),
                    f=f_ref,
                    n_recycle=self.n_recycle,
                    **ref_initializer_outputs,
                )

                X_denoised_L_stripped = outs_ref["X_L"]

                delta_L_ref = (
                    X_noisy_L_stripped - X_denoised_L_stripped
                ) / t_hat

                if delta_L_ref.shape[1] < delta_L.shape[1]:
                    delta_L_ref = torch.cat(
                        [
                            delta_L_ref,
                            torch.zeros_like(delta_L[:, delta_L_ref.shape[1] :, :]),
                        ],
                        dim=1,
                    )

                delta_L = delta_L + (self.cfg_scale - 1) * (delta_L - delta_L_ref)

            if exists(outs.get("sequence_logits_I")):
                p = torch.softmax(
                    outs["sequence_logits_I"], dim=-1
                ).cpu()
                seq_entropy = -torch.sum(
                    p * torch.log(p + 1e-10), dim=-1
                )
                sequence_entropy_traj.append(seq_entropy)

            X_L = X_noisy_L + step_scale * d_t * delta_L

            X_noisy_L_scaled = (
                self.sigma_data * X_noisy_L / torch.sqrt(t_hat**2 + self.sigma_data**2)
            )
            X_noisy_L_traj.append(X_noisy_L_scaled)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(t_hat)

        if torch.any(is_motif_atom_with_fixed_coord) and self.allow_realignment:
            X_L, _ = centre_random_augment_around_motif(
                X_L,
                coord_atom_lvl_to_be_noised,
                is_motif_atom_with_fixed_coord,
                reinsert_motif=self.insert_motif_at_end,
            )

            X_L = weighted_rigid_align(
                coord_atom_lvl_to_be_noised,
                X_L,
                X_exists_L=is_motif_atom_with_fixed_coord,
            )

        return dict(
            X_L=X_L,
            X_noisy_L_traj=X_noisy_L_traj,
            X_denoised_L_traj=X_denoised_L_traj,
            t_hats=t_hats,
            sequence_logits_I=outs.get("sequence_logits_I"),
            sequence_indices_I=outs.get("sequence_indices_I"),
            sequence_entropy_traj=sequence_entropy_traj,
        )


class SampleDiffusionWithSymmetry(SampleDiffusionWithMotif):
    """
    This class is a wrapper around the SampleDiffusionWithMotif class.
    It is used to sample diffusion with symmetry.
    """

    def __init__(self, sym_step_frac: float = 0.9, **kwargs):
        assert (
            kwargs.get("gamma_0") > 0.5
        ), "gamma_0 must be greater than 0.5 for symmetry sampling"
        self.sym_step_frac = sym_step_frac
        super().__init__(**kwargs)

    def apply_symmetry_to_X_L(self, X_L, f):

        assert "sym_transform" in f.keys(), "Symmetry transform not found in f"

        symmetry_feats = {k: v for k, v in f.items() if "sym" in k}

        X_L = apply_symmetry_to_xyz_atomwise(
            X_L, symmetry_feats, partial_diffusion=("partial_t" in f)
        )

        return X_L

    def sample_diffusion_like_af3(
        self,
        *,
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coord_atom_lvl_to_be_noised: Float[torch.Tensor, "D L 3"],
        initializer_outputs,
        ref_initializer_outputs: dict[str, Any] | None,
        f_ref: dict[str, Any] | None,
        **_,
    ) -> dict[str, Any]:
        is_motif_atom_with_fixed_coord = f["is_motif_atom_with_fixed_coord"]
        noise_schedule = self._construct_inference_noise_schedule(
            device=coord_atom_lvl_to_be_noised.device,
            partial_t=f.get("partial_t", None),
        )

        L = f["ref_element"].shape[0]
        D = diffusion_batch_size
        X_L = self._get_initial_structure(
            c0=noise_schedule[0],
            D=D,
            L=L,
            coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised.clone(),
            is_motif_atom_with_fixed_coord=is_motif_atom_with_fixed_coord,
        )

        X_noisy_L_traj = []
        X_denoised_L_traj = []
        sequence_entropy_traj = []
        t_hats = []

        gamma_min_sym_idx = min(
            int(len(noise_schedule) * self.sym_step_frac), len(noise_schedule) - 1
        )
        gamma_min_sym = noise_schedule[gamma_min_sym_idx]

        ranked_logger.info(f"gamma_min_sym: {gamma_min_sym}")
        ranked_logger.info(f"gamma_min: {self.gamma_min}")
        for step_num, (c_t_minus_1, c_t) in enumerate(
            zip(noise_schedule, noise_schedule[1:])
        ):
            assert not torch.is_grad_enabled(), "Computation graph should not be active"
            assert not X_L.requires_grad, "X_L should not require gradients"

            if self.allow_realignment:
                X_L, R = centre_random_augment_around_motif(
                    X_L,
                    coord_atom_lvl_to_be_noised,
                    is_motif_atom_with_fixed_coord,
                )

            gamma = self.gamma_0 if c_t > self.gamma_min else 0
            step_scale = self.step_scale

            t_hat = c_t_minus_1 * (gamma + 1)

            epsilon_L = (
                self.noise_scale
                * torch.sqrt(torch.square(t_hat) - torch.square(c_t_minus_1))
                * torch.normal(mean=0.0, std=1.0, size=X_L.shape, device=X_L.device)
            )
            epsilon_L[..., is_motif_atom_with_fixed_coord, :] = (
                0
            )

            X_noisy_L = X_L + epsilon_L

            if "chunked_pairwise_embedder" in initializer_outputs:
                tic = time.time()
                chunked_embedder = initializer_outputs[
                    "chunked_pairwise_embedder"
                ]
                other_outputs = {
                    k: v
                    for k, v in initializer_outputs.items()
                    if k != "chunked_pairwise_embedder"
                }
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,
                    t=t_hat.tile(D),
                    f=f,
                    P_LL=None,
                    chunked_pairwise_embedder=chunked_embedder,
                    initializer_outputs=other_outputs,
                    n_recycle=self.n_recycle,
                    **other_outputs,
                )
                toc = time.time()
                ranked_logger.info(
                    f"[chunked] step {step_num}: {(toc - tic)*1000:.1f} ms"
                )
            else:
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,
                    t=t_hat.tile(D),
                    f=f,
                    n_recycle=self.n_recycle,
                    **initializer_outputs,
                )
            if "X_L" in outs and c_t > gamma_min_sym:
                outs["X_L"] = self.apply_symmetry_to_X_L(outs["X_L"], f)

            X_denoised_L = outs["X_L"] if "X_L" in outs else outs

            if self.adjacency_potential_scale != 0.0:
                grad = _adjacency_guidance_grad(
                    X_denoised_L,
                    f,
                    threshold=self.adjacency_potential_threshold,
                    softness=self.adjacency_potential_softness,
                )
                if grad is not None:
                    X_denoised_L = X_denoised_L - self.adjacency_potential_scale * grad

            delta_L = (
                X_noisy_L - X_denoised_L
            ) / t_hat
            d_t = c_t - t_hat


            if exists(outs.get("sequence_logits_I")):
                p = torch.softmax(
                    outs["sequence_logits_I"], dim=-1
                ).cpu()
                seq_entropy = -torch.sum(
                    p * torch.log(p + 1e-10), dim=-1
                )
                sequence_entropy_traj.append(seq_entropy)

            X_L = X_noisy_L + step_scale * d_t * delta_L

            X_noisy_L_scaled = (
                self.sigma_data * X_noisy_L / torch.sqrt(t_hat**2 + self.sigma_data**2)
            )
            X_noisy_L_traj.append(X_noisy_L_scaled)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(t_hat)

        if torch.any(is_motif_atom_with_fixed_coord) and self.allow_realignment:
            X_L, R = centre_random_augment_around_motif(
                X_L,
                coord_atom_lvl_to_be_noised,
                is_motif_atom_with_fixed_coord,
                reinsert_motif=self.insert_motif_at_end,
            )

            X_L = self.apply_symmetry_to_X_L(X_L, f)

            X_L = weighted_rigid_align(
                coord_atom_lvl_to_be_noised,
                X_L,
                X_exists_L=is_motif_atom_with_fixed_coord,
            )

        return dict(
            X_L=X_L,
            X_noisy_L_traj=X_noisy_L_traj,
            X_denoised_L_traj=X_denoised_L_traj,
            t_hats=t_hats,
            sequence_logits_I=outs.get("sequence_logits_I"),
            sequence_indices_I=outs.get("sequence_indices_I"),
            sequence_entropy_traj=sequence_entropy_traj,
        )


class ConditionalDiffusionSampler:
    """
    Conditional diffusion sampler, chooses at construction time which sampler to use,
    then forwards `sample_diffusion_like_af3` to the chosen sampler.
    If you write a new sampler, you best add it to the registry below
    and inference_sampler.kind in inference_engine config.
    """

    _registry = {
        "default": SampleDiffusionWithMotif,
        "symmetry": SampleDiffusionWithSymmetry,
    }

    def __init__(self, kind="default", **kwargs):
        ranked_logger.info(
            f"Initializing ConditionalDiffusionSampler with kind: {kind}"
        )
        try:
            SamplerCls = self._registry[kind]
            init_args = self.get_class_init_args(SamplerCls)
            kwargs = {k: v for k, v in kwargs.items() if k in init_args}
        except KeyError:
            raise ValueError(
                f"Invalid sampler kind: {kind}, must be one of {list(self._registry.keys())}"
            )
        self.sampler = SamplerCls(**kwargs)

    def sample_diffusion_like_af3(self, **kwargs):
        return self.sampler.sample_diffusion_like_af3(**kwargs)

    def get_class_init_args(self, cls):
        arg_names = []
        if hasattr(cls, "__init__") and callable(cls.__init__):
            for p_cls in cls.__mro__:
                if "__init__" in p_cls.__dict__ and p_cls is not object:
                    signature = inspect.signature(p_cls.__init__)
                    arg_names.extend(
                        [param.name for param in signature.parameters.values()]
                    )
        return arg_names


def centre_random_augment_around_motif(
    X_L: torch.Tensor,
    coord_atom_lvl_to_be_noised: torch.Tensor,
    is_motif_atom_with_fixed_coord: torch.Tensor,
    s_trans: float = 1.0,
    center_option: str = "all",
    centering_affects_motif: bool = True,
    reinsert_motif=True,
):
    D, L, _ = X_L.shape

    if reinsert_motif and torch.any(is_motif_atom_with_fixed_coord):
        coords_with_gt_aligned = weighted_rigid_align(
            X_L[..., is_motif_atom_with_fixed_coord, :],
            coord_atom_lvl_to_be_noised[..., is_motif_atom_with_fixed_coord, :],
        )

        X_L[..., is_motif_atom_with_fixed_coord, :] = coords_with_gt_aligned

    if torch.any(is_motif_atom_with_fixed_coord):
        if center_option == "motif":
            center = torch.mean(
                X_L[..., is_motif_atom_with_fixed_coord, :], dim=-2, keepdim=True
            )
        elif center_option == "diffuse":
            center = torch.mean(
                X_L[..., ~is_motif_atom_with_fixed_coord, :], dim=-2, keepdim=True
            )

        else:
            center = torch.mean(X_L, dim=-2, keepdim=True)
    else:
        center = torch.mean(X_L, dim=-2, keepdim=True)

    if centering_affects_motif:
        X_L = X_L - center
    else:
        X_L[..., ~is_motif_atom_with_fixed_coord, :] = (
            X_L[..., ~is_motif_atom_with_fixed_coord, :] - center
        )

    R = uniform_random_rotation((D,)).to(X_L.device)
    noise = (
        torch.normal(mean=0, std=1, size=(D, 1, 3), device=X_L.device) * s_trans
    )
    X_L = rot_vec_mul(R[:, None], X_L) + noise

    return X_L, R
