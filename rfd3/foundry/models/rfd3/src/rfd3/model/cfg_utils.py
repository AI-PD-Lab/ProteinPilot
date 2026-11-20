import torch


def strip_f(
    f,
    cfg_features,
):
    """
    Strips conditioning features from 'f' for classifier-free guidance.

    Args:
        f (dict): Conditioning features
        cfg_features (list): List of features to be set to 0

    Returns:
        dict: Stripped conditioning features
    """
    token_dim = f["is_motif_token_unindexed"].shape[0]
    atom_dim = f["is_motif_atom_unindexed"].shape[0]

    crop = torch.any(f["is_motif_atom_unindexed"]).item()
    atom_crop_index = (
        torch.where(f["is_motif_atom_unindexed"])[0][0]
        if crop
        else f["is_motif_atom_unindexed"].shape[0]
    )
    token_crop_index = (
        torch.where(f["is_motif_token_unindexed"])[0][0]
        if crop
        else f["is_motif_token_unindexed"].shape[0]
    )

    f_stripped = f.copy()

    for k, v in f.items():
        v_cropped = v

        if token_dim in v.shape:
            if len(v.shape) == 2 and v.shape[0] == v.shape[1]:
                v_cropped = v[:token_crop_index, :token_crop_index]
            else:
                v_cropped = v[:token_crop_index]
        if atom_dim in v.shape:
            if len(v.shape) == 2 and v.shape[0] == v.shape[1]:
                v_cropped = v[:atom_crop_index, :atom_crop_index]
            else:
                v_cropped = v[:atom_crop_index]

        if k in cfg_features:
            v_cropped = torch.zeros_like(v_cropped).to(
                v_cropped.device, dtype=v_cropped.dtype
            )

        f_stripped[k] = v_cropped

    return f_stripped


def strip_X(X_L, f_stripped):
    """
    Strips X_L unindexed atoms from X_L

    Args:
        X_L (torch.Tensor): Atom coordinates
        f_stripped (dict): Stripped conditioning features

    Returns:
        torch.Tensor: Atom coordinates with unindexed atoms removed
    """
    return X_L[..., : f_stripped["is_motif_atom_unindexed"].shape[0], :]
