"""
The Atom14 data pipeline for training and inference
"""

import warnings
from pathlib import Path
from typing import List

import numpy as np
from atomworks.constants import (
    AF3_EXCLUDED_LIGANDS,
    GAP,
    STANDARD_AA,
    STANDARD_DNA,
    STANDARD_RNA,
)
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from atomworks.ml.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddGlobalTokenIdAnnotation,
    AddProteinTerminiAnnotation,
    AddWithinChainInstanceResIdx,
    AddWithinPolyResIdxAnnotation,
    ComputeAtomToTokenMap,
    CopyAnnotation,
)
from atomworks.ml.transforms.atomize import (
    AtomizeByCCDName,
    FlagNonPolymersForAtomization,
)
from atomworks.ml.transforms.base import (
    AddData,
    Compose,
    ConditionalRoute,
    ConvertToTorch,
    Identity,
    RandomRoute,
    SubsetToKeys,
)
from atomworks.ml.transforms.bfactor_conditioned_transforms import SetOccToZeroOnBfactor
from atomworks.ml.transforms.bonds import AddAF3TokenBondFeatures
from atomworks.ml.transforms.cached_residue_data import LoadCachedResidueLevelData
from atomworks.ml.transforms.covalent_modifications import (
    FlagAndReassignCovalentModifications,
)
from atomworks.ml.transforms.crop import CropContiguousLikeAF3, CropSpatialLikeAF3
from atomworks.ml.transforms.diffusion.batch_structures import (
    BatchStructuresForDiffusionNoising,
)
from atomworks.ml.transforms.diffusion.edm import SampleEDMNoise
from atomworks.ml.transforms.featurize_unresolved_residues import (
    MaskPolymerResiduesWithUnresolvedFrameAtoms,
    PlaceUnresolvedTokenAtomsOnRepresentativeAtom,
    PlaceUnresolvedTokenOnClosestResolvedTokenInSequence,
)
from atomworks.ml.transforms.filters import (
    FilterToSpecifiedPNUnits,
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemoveNucleicAcidTerminalOxygen,
    RemovePolymersWithTooFewResolvedResidues,
    RemoveTerminalOxygen,
    RemoveUnresolvedLigandAtomsIfTooMany,
    RemoveUnresolvedPNUnits,
)
from atomworks.ml.utils.token import get_token_count
from rfd3.transforms.conditioning_base import (
    SampleConditioningFlags,
    SampleConditioningType,
    StrtoBoolforIsXFeatures,
    UnindexFlaggedTokens,
)
from rfd3.transforms.design_transforms import (
    AddAdditional1dFeaturesToFeats,
    AddGroundTruthSequence,
    AddIsXFeats,
    AssignTypes,
    AugmentNoise,
    CreateDesignReferenceFeatures,
    FeaturizeAtoms,
    FeaturizepLDDT,
    MotifCenterRandomAugmentation,
    SubsampleToTypes,
)
from rfd3.transforms.dna_crop import ProteinDNAContactContiguousCrop
from rfd3.transforms.hbonds_hbplus import CalculateHbondsPlus
from rfd3.transforms.ppi_transforms import (
    Add1DSSFeature,
    AddGlobalIsNonLoopyFeature,
    AddPPIHotspotFeature,
    PPIFullBinderCropSpatial,
)
from rfd3.transforms.ss_adj_transforms import (
    FeaturizeSSADJ,
    RejectShortMultichainLocalTokenExamples,
)
from rfd3.transforms.rasa import (
    CalculateRASA,
    SetZeroOccOnDeltaRASA,
)
from rfd3.transforms.symmetry import AddSymmetryFeats
from rfd3.transforms.util_transforms import (
    IPDB,
    AggregateFeaturesLikeAF3WithoutMSA,
    EncodeAF3TokenLevelFeatures,
    RemoveTokensWithoutCorrespondingCentralAtom,
)
from rfd3.transforms.virtual_atoms import PadTokensWithVirtualAtoms

from foundry.common import exists

af3_sequence_encoding = AF3SequenceEncoding()


IPDB


def TrainingRoute(transform):
    return ConditionalRoute(
        condition_func=lambda data: data["is_inference"],
        transform_map={True: Identity(), False: transform},
    )


def InferenceRoute(transform):
    return ConditionalRoute(
        condition_func=lambda data: data["is_inference"],
        transform_map={False: Identity(), True: transform},
    )


def TrainingConditionRoute(condition, transform):
    transform = TrainingRoute(
        ConditionalRoute(
            condition_func=lambda data: data["conditions"][condition],
            transform_map={
                True: transform,
                False: Identity(),
            },
        )
    )
    return transform


def get_pre_crop_transforms(
    central_atom: str,
    b_factor_min: float | None,
):
    return [
        InferenceRoute(StrtoBoolforIsXFeatures()),
        RemoveHydrogens(),
        FilterToSpecifiedPNUnits(
            extra_info_key_with_pn_unit_iids_to_keep="all_pn_unit_iids_after_processing"
        ),
        RemoveTerminalOxygen(),
        TrainingRoute(SetOccToZeroOnBfactor(b_factor_min, None)),
        RemoveUnresolvedPNUnits(),
        TrainingRoute(RemovePolymersWithTooFewResolvedResidues(min_residues=4)),
        MaskPolymerResiduesWithUnresolvedFrameAtoms(),
        TrainingRoute(HandleUndesiredResTokens(AF3_EXCLUDED_LIGANDS)),
        TrainingRoute(
            RemoveUnresolvedLigandAtomsIfTooMany(unresolved_ligand_atom_limit=5)
        ),
        TrainingRoute(
            RemoveTokensWithoutCorrespondingCentralAtom(central_atom=central_atom),
        ),
        FlagAndReassignCovalentModifications(),
        FlagNonPolymersForAtomization(),
        AddGlobalAtomIdAnnotation(),
        AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
            move_atomized_part_to_end=False,
            validate_atomize=False,
        ),
        RemoveNucleicAcidTerminalOxygen(),
        AddWithinChainInstanceResIdx(),
        AddWithinPolyResIdxAnnotation(),
        AddProteinTerminiAnnotation(),
    ]


def get_crop_transform(
    crop_size: int,
    crop_center_cutoff_distance: float,
    crop_contiguous_probability: float,
    crop_spatial_probability: float,
    dna_contact_crop_probability: float,
    keep_full_binder_in_spatial_crop: bool,
    max_binder_length: int,
    max_atoms_in_crop: int | None,
    allowed_types: List[str],
):
    if (
        crop_contiguous_probability > 0
        or crop_spatial_probability > 0
        or dna_contact_crop_probability > 0
    ):
        assert np.isclose(
            crop_contiguous_probability
            + crop_spatial_probability
            + dna_contact_crop_probability,
            1.0,
            atol=1e-6,
        ), "Crop probabilities must sum to 1.0"
        assert crop_size > 0, "Crop size must be greater than 0"
        assert (
            crop_center_cutoff_distance > 0
        ), "Crop center cutoff distance must be greater than 0"

    pre_crop_transforms = [
        SubsampleToTypes(allowed_types=allowed_types),
    ]

    cropping_transform = RandomRoute(
        transforms=[
            CropContiguousLikeAF3(
                crop_size=crop_size,
                keep_uncropped_atom_array=True,
                max_atoms_in_crop=max_atoms_in_crop,
            ),
            ConditionalRoute(
                condition_func=lambda data: (
                    keep_full_binder_in_spatial_crop
                    and data["sampled_condition_name"] == "ppi"
                    and get_token_count(
                        data["atom_array"][data["atom_array"].is_binder_pn_unit]
                    )
                    < max_binder_length
                    and data["conditions"]["full_binder_crop"]
                ),
                transform_map={
                    True: PPIFullBinderCropSpatial(
                        crop_size=crop_size,
                        crop_center_cutoff_distance=crop_center_cutoff_distance,
                        keep_uncropped_atom_array=True,
                        max_atoms_in_crop=max_atoms_in_crop,
                    ),
                    False: CropSpatialLikeAF3(
                        crop_size=crop_size,
                        crop_center_cutoff_distance=crop_center_cutoff_distance,
                        keep_uncropped_atom_array=True,
                        max_atoms_in_crop=max_atoms_in_crop,
                    ),
                },
            ),
            ProteinDNAContactContiguousCrop(
                protein_contact_type="all",
                dna_contact_type="base",
                max_atoms_in_crop=max_atoms_in_crop,
            ),
        ],
        probs=[
            crop_contiguous_probability,
            crop_spatial_probability,
            dna_contact_crop_probability,
        ],
    )

    post_crop_transforms = [
        TrainingRoute(
            PlaceUnresolvedTokenAtomsOnRepresentativeAtom(annotation_to_update="coord")
        ),
        TrainingRoute(
            PlaceUnresolvedTokenOnClosestResolvedTokenInSequence(
                annotation_to_update="coord",
                annotation_to_copy="coord",
            )
        ),
    ]

    transform = (
        pre_crop_transforms
        + [
            TrainingRoute(cropping_transform),
        ]
        + post_crop_transforms
    )
    return transform


def get_diffusion_transforms(
    *,
    sigma_data: float,
    diffusion_batch_size: int,
):
    return [
        ComputeAtomToTokenMap(),
        ConvertToTorch(keys=["encoded", "feats"]),
        CopyAnnotation(annotation_to_copy="coord", new_annotation="coord_to_be_noised"),
        AggregateFeaturesLikeAF3WithoutMSA(),
        BatchStructuresForDiffusionNoising(batch_size=diffusion_batch_size),
        SampleEDMNoise(
            sigma_data=sigma_data, diffusion_batch_size=diffusion_batch_size
        ),
    ]


def build_atom14_base_pipeline_(
    *,
    is_inference: bool,
    return_atom_array: bool,
    allowed_types: List[str],
    crop_size: int,
    crop_center_cutoff_distance: float,
    crop_contiguous_probability: float,
    crop_spatial_probability: float,
    dna_contact_crop_probability: float,
    keep_full_binder_in_spatial_crop: bool,
    max_binder_length: int,
    max_atoms_in_crop: int | None,
    b_factor_min: float | None,
    zero_occ_on_exposure_after_cropping: bool,
    sigma_data: float,
    diffusion_batch_size: int,
    generate_conformers: bool,
    generate_conformers_for_non_protein_only: bool,
    provide_reference_conformer_when_unmasked: bool,
    ground_truth_conformer_policy: str,
    provide_elements_for_unindexed_components: bool,
    use_element_for_atom_names_of_atomized_tokens: bool,
    residue_cache_dir: bool,
    train_conditions: dict,
    meta_conditioning_probabilities: dict,
    n_atoms_per_token: int,
    central_atom: str,
    sigma_perturb: float,
    sigma_perturb_com: float,
    association_scheme: str | None,
    center_option: str,
    atom_1d_features: dict | None,
    token_1d_features: dict | None,
    max_ppi_hotspots_frac_to_provide: float,
    ppi_hotspot_max_distance: float,
    max_ss_frac_to_provide: float,
    min_ss_island_len: int,
    max_ss_island_len: int,
    ss_adj_enabled: bool = False,
    ss_adj_loop_adj: bool = False,
    ss_adj_typed_blocks: bool = False,
    ss_adj_motif_region_mask_prob: float = 0.5,
    ss_adj_cross_chain_seed_prob: float = 0.0,
    ss_adj_cb_threshold: float = 6.0,
    ss_adj_ca_threshold_loop: float = 6.0,
    ss_adj_min_loop_contact_pairs: int = 3,
    ss_adj_mask_loop_pairs: bool = True,
    reject_short_multichain_local_tokens: bool = False,
    **_,
):
    """
    All-Atom design pipeline
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    transforms = [
        AddData(
            {
                "is_inference": is_inference,
                "sampled_condition_name": None,
                "conditions": {},
            }
        ),
        AssignTypes(),
    ]
    transforms += [
        TrainingRoute(
            SampleConditioningType(
                train_conditions=train_conditions,
                meta_conditioning_probabilities=meta_conditioning_probabilities,
                sequence_encoding=af3_sequence_encoding,
            ),
        ),
    ]

    transforms += get_pre_crop_transforms(
        central_atom=central_atom,
        b_factor_min=b_factor_min,
    )
    if zero_occ_on_exposure_after_cropping:
        transforms.append(TrainingRoute(CalculateRASA(requires_ligand=False)))

    transforms += get_crop_transform(
        crop_size=crop_size,
        crop_center_cutoff_distance=crop_center_cutoff_distance,
        crop_contiguous_probability=crop_contiguous_probability,
        crop_spatial_probability=crop_spatial_probability,
        dna_contact_crop_probability=dna_contact_crop_probability,
        keep_full_binder_in_spatial_crop=keep_full_binder_in_spatial_crop,
        max_binder_length=max_binder_length,
        max_atoms_in_crop=max_atoms_in_crop,
        allowed_types=allowed_types,
    )

    if zero_occ_on_exposure_after_cropping:
        transforms.append(TrainingRoute(SetZeroOccOnDeltaRASA()))
    else:
        transforms.append(
            TrainingConditionRoute(
                "calculate_rasa", CalculateRASA(requires_ligand=True)
            )
        )

    transforms.append(AddGlobalTokenIdAnnotation())
    transforms.append(TrainingRoute(SampleConditioningFlags()))

    transforms.append(
        TrainingConditionRoute(
            "calculate_hbonds",
            CalculateHbondsPlus(
                cutoff_HA_dist=3,
                cutoff_DA_distance=3.5,
            ),
        )
    )

    transforms += [
        LoadCachedResidueLevelData(
            dir=Path(residue_cache_dir) if exists(residue_cache_dir) else None,
            sharding_depth=1,
        ),
        UnindexFlaggedTokens(central_atom=central_atom),
        PadTokensWithVirtualAtoms(
            n_atoms_per_token=n_atoms_per_token,
            atom_to_pad_from=central_atom,
            association_scheme=association_scheme,
        ),
        TrainingRoute(RejectShortMultichainLocalTokenExamples())
        if reject_short_multichain_local_tokens
        else Identity(),
        TrainingRoute(
            ConditionalRoute(
                condition_func=lambda data: data["sampled_condition_name"] == "ppi"
                and data["conditions"]["add_ppi_hotspots"],
                transform_map={
                    True: AddPPIHotspotFeature(
                        max_hotspots_frac_to_provide=max_ppi_hotspots_frac_to_provide,
                        hotspot_max_distance=ppi_hotspot_max_distance,
                    ),
                    False: Identity(),
                },
            )
        ),
        TrainingRoute(
            ConditionalRoute(
                condition_func=lambda data: data["conditions"][
                    "add_global_is_non_loopy_feature"
                ],
                transform_map={
                    True: AddGlobalIsNonLoopyFeature(),
                    False: Identity(),
                },
            )
        ),
        TrainingRoute(
            Add1DSSFeature(
                max_secondary_structure_frac_to_provide=max_ss_frac_to_provide,
                min_ss_island_len=min_ss_island_len,
                max_ss_island_len=max_ss_island_len,
            ),
        ),
        FeaturizeSSADJ(
            loop_adj=ss_adj_loop_adj,
            typed_blocks=ss_adj_typed_blocks,
            cb_threshold=ss_adj_cb_threshold,
            ca_threshold_loop=ss_adj_ca_threshold_loop,
            min_loop_contact_pairs=ss_adj_min_loop_contact_pairs,
            mask_loop_pairs=ss_adj_mask_loop_pairs,
            motif_region_mask_prob=ss_adj_motif_region_mask_prob,
            cross_chain_seed_prob=ss_adj_cross_chain_seed_prob,
        )
        if ss_adj_enabled
        else Identity(),
        EncodeAF3TokenLevelFeatures(
            sequence_encoding=af3_sequence_encoding, encode_residues_to=GAP
        ),
        CreateDesignReferenceFeatures(
            generate_conformers=generate_conformers,
            generate_conformers_for_non_protein_only=generate_conformers_for_non_protein_only,
            provide_reference_conformer_when_unmasked=provide_reference_conformer_when_unmasked,
            ground_truth_conformer_policy=ground_truth_conformer_policy,
            provide_elements_for_unindexed_components=provide_elements_for_unindexed_components,
            use_element_for_atom_names_of_atomized_tokens=use_element_for_atom_names_of_atomized_tokens,
        ),
        AddIsXFeats(
            X=[
                "is_backbone",
                "is_sidechain",
                "is_virtual",
                "is_central",
                "is_ca",
                "is_motif_atom_with_fixed_coord",
                "is_motif_atom_unindexed",
                "is_motif_atom_with_fixed_seq",
                "is_motif_token_with_fully_fixed_coord",
            ],
            central_atom=central_atom,
        ),
        FeaturizeAtoms(),
        FeaturizepLDDT(skip=b_factor_min is not None),
        AddAdditional1dFeaturesToFeats(
            autofill_zeros_if_not_present_in_atomarray=True,
            token_1d_features=token_1d_features,
            atom_1d_features=atom_1d_features,
        ),
        AddAF3TokenBondFeatures(),
        AddGroundTruthSequence(sequence_encoding=af3_sequence_encoding),
        ConditionalRoute(
            condition_func=lambda data: "symmetry_id"
            in data["atom_array"].get_annotation_categories(),
            transform_map={
                True: AddSymmetryFeats(),
                False: Identity(),
            },
        ),
    ]

    transforms += get_diffusion_transforms(
        sigma_data=sigma_data,
        diffusion_batch_size=diffusion_batch_size,
    )

    transforms += [
        MotifCenterRandomAugmentation(
            batch_size=diffusion_batch_size,
            sigma_perturb=sigma_perturb,
            center_option=center_option,
        ),
        AugmentNoise(
            sigma_perturb_com=sigma_perturb_com,
            batch_size=diffusion_batch_size,
        ),
    ]

    keys_to_keep = [
        "example_id",
        "feats",
        "t",
        "noise",
        "ground_truth",
        "coord_atom_lvl_to_be_noised",
        "extra_info",
        "sampled_condition_name",
        "log_dict",
    ]
    if return_atom_array:
        keys_to_keep.extend(
            [
                "atom_array",
                "specification",
            ]
        )
        if not is_inference:
            keys_to_keep.append("conditions")
    transforms.append(SubsetToKeys(keys_to_keep))

    pipeline = Compose(transforms)
    return pipeline


def build_atom14_base_pipeline(
    is_inference: bool,
    *,
    protein_msa_dirs=None,
    rna_msa_dirs=None,
    n_recycles=None,
    n_msa=None,
    **kwargs,
):
    """
    Wrapper around pipeline construction to handle empty training args
    Sets default behaviour for inference to keep backward compatibility
    """

    if is_inference:
        kwargs.setdefault("crop_size", 512)
        kwargs.setdefault("crop_center_cutoff_distance", 10.0)
        kwargs.setdefault("crop_contiguous_probability", 1.0)
        kwargs.setdefault("crop_spatial_probability", 0.0)
        kwargs.setdefault("dna_contact_crop_probability", 0.0)
        kwargs.setdefault("max_atoms_in_crop", None)
        kwargs.setdefault("keep_full_binder_in_spatial_crop", True)
        kwargs.setdefault("max_ppi_hotspots_frac_to_provide", 0)
        kwargs.setdefault("ppi_hotspot_max_distance", 15)
        kwargs.setdefault("max_ss_frac_to_provide", 0.0)
        kwargs.setdefault("min_ss_island_len", 0)
        kwargs.setdefault("max_ss_island_len", 999)
        kwargs.setdefault("max_binder_length", 999)

        kwargs.setdefault("b_factor_min", None)
        kwargs.setdefault("zero_occ_on_exposure_after_cropping", False)
        kwargs.setdefault("meta_conditioning_probabilities", {})
        kwargs.setdefault("association_scheme", "dense")
        kwargs.setdefault("sigma_perturb", 0.0)
        kwargs.setdefault("sigma_perturb_com", 0.0)
        kwargs.setdefault("allowed_types", "ALL")
        kwargs.setdefault("train_conditions", {})
        kwargs.setdefault("residue_cache_dir", None)

        kwargs.setdefault("generate_conformers_for_non_protein_only", True)
        kwargs.setdefault("return_atom_array", True)
        kwargs.setdefault("provide_elements_for_unindexed_components", False)
        kwargs.setdefault("center_option", "all")

    return build_atom14_base_pipeline_(
        is_inference=is_inference,
        **kwargs,
    )
