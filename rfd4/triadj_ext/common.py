"""Shared helpers for the TRIAD-J spike (CPU, released RFD4-Proteina bundle)."""
import os
import sys

os.environ.setdefault("CLUSTER", "local")
os.environ.setdefault("DATA_PATH", "/w")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BUNDLE = "/w/rfd4_weights"
# Runtime-only options (LocalLatentsTransformer.RUNTIME_OPTIONS); the bundle stores
# tri_mult_backend=cuequivariance (needs libcuda) and attention_type=torch (CPU einsum fallback exists).
CPU_NN_OVERRIDES = {"tri_mult_backend": "torch", "attention_type": "torch"}


def log(*a):
    print(*a, flush=True)


def load_model(nn_overrides=None, device="cpu"):
    """Proteina.from_pretrained on CPU; returns eval-mode model (fp32)."""
    import torch

    from rfproteina.proteina import Proteina

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    ov = {} if os.environ.get("TRIADJ_NO_OVERRIDES") else dict(CPU_NN_OVERRIDES)
    if nn_overrides:
        ov.update(nn_overrides)
    return Proteina.from_pretrained(BUNDLE, device=device, nn_overrides=ov)


def tree_shapes(d, prefix="", out=None):
    """Flatten a (nested) batch dict into [(key, shape, dtype)] rows."""
    import torch

    out = [] if out is None else out
    for k, v in d.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            tree_shapes(v, name + ".", out)
        elif isinstance(v, torch.Tensor):
            out.append((name, tuple(v.shape), str(v.dtype).replace("torch.", "")))
        else:
            out.append((name, type(v).__name__, repr(v)[:60]))
    return out


def get_train_batch(cif="/fork/tests/data/5vht_from_file.cif", crop=96, seed=0):
    """Training-style batch (plain dict of tensors). Cached in $TRIADJ_DATA/train_batch.pt when that dir has it."""
    import torch

    d = os.environ.get("TRIADJ_DATA")
    path = os.path.join(d, "train_batch.pt") if d else None
    if path and os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    from triadj_ext.s2_batch import build_training_batch

    tb, _ = build_training_batch(cif, crop, seed)
    out = {k: v for k, v in tb.items() if isinstance(v, torch.Tensor)}
    if path:
        os.makedirs(d, exist_ok=True)
        torch.save(out, path)
    return out


def prepare_noisy_batch(model, batch, t_ca=0.8, t_lat=0.3, seed=0):
    """Replicates the first half of Proteina.training_step (proteina.py:625-657) with FIXED clocks:
    coords_nm -> add_clean_samples (frozen AE encode -> x_1) -> fm.corrupt_batch (x_0, x_t, t)."""
    import torch

    torch.manual_seed(seed)
    batch = dict(batch)
    batch["coords_nm"] = batch["coords"] / 10.0  # proteina.py:625
    batch["coord_mask"] = batch["atom37_mask"]  # proteina.py:626
    with torch.no_grad():
        batch = model.add_clean_samples(batch)  # proteina.py:634/740: x_1 = {bb_ca [b,n,3], local_latents [b,n,32]}
    b = batch["padding_mask"].shape[0]
    t = {"bb_ca": torch.full((b,), float(t_ca)), "local_latents": torch.full((b,), float(t_lat))}
    batch = model.fm.corrupt_batch(batch, t=t)  # product_space_flow_matcher.py:222 -> x_0, x_1, x_t, t
    return batch
