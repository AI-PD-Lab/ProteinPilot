import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from opt_einsum import contract as einsum
from rfd3.model.layers.block_utils import (
    create_attention_indices,
    indices_to_mask,
)
from rfd3.model.layers.layer_utils import (
    AdaLN,
    LinearBiasInit,
    RMSNorm,
    linearNoBias,
)

from foundry.common import exists
from foundry.training.checkpoint import activation_checkpointing
from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)

try:
    from cuequivariance_torch import attention_pair_bias as cueq_attention_pair_bias

    _CUEQ_AVAILABLE = True
except Exception:
    _CUEQ_AVAILABLE = False


@torch.compiler.disable
def kernel_pairbias_attention(
    *,
    s: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor | None,
    num_heads: int,
    w_proj_z: torch.Tensor,
    w_proj_g: torch.Tensor,
    w_proj_o: torch.Tensor,
    w_ln_z: torch.Tensor,
    b_ln_z: torch.Tensor,
    b_proj_z: torch.Tensor | None = None,
    b_proj_g: torch.Tensor | None = None,
    b_proj_o: torch.Tensor | None = None,
    attn_scale: float | None = None,
    compute_pair_bias: bool = True,
    multiplicity: int = 1,
) -> torch.Tensor:
    """Thin wrapper around cuequivariance_torch.attention_pair_bias."""
    raise NotImplementedError("CUDA Kernel for attention pair bias not implemented")
    out, _proj_z = cueq_attention_pair_bias(
        s=s,
        q=q,
        k=k,
        v=v,
        z=z,
        mask=mask,
        num_heads=num_heads,
        w_proj_z=w_proj_z,
        w_proj_g=w_proj_g,
        w_proj_o=w_proj_o,
        w_ln_z=w_ln_z,
        b_ln_z=b_ln_z,
        b_proj_z=b_proj_z,
        b_proj_g=b_proj_g,
        b_proj_o=b_proj_o,
        attn_scale=attn_scale,
        compute_pair_bias=compute_pair_bias,
        multiplicity=multiplicity,
    )
    return out


class GatedCrossAttention(nn.Module):
    def __init__(
        self,
        c_query,
        c_kv,
        c_pair=None,
        c_model=128,
        n_head=4,
        kq_norm=True,
        dropout=0.0,
        **_,
    ):
        super().__init__()
        self.n_head = n_head
        self.scale = 1 / math.sqrt(c_model // n_head)
        assert c_model % n_head == 0, "c_model must be divisible by n_heads"

        self.ln_q = RMSNorm(c_query)
        self.ln_kv = RMSNorm(c_kv)

        self.to_q = linearNoBias(c_query, c_model)
        self.to_k = linearNoBias(c_kv, c_model)
        self.to_v = linearNoBias(c_kv, c_model)
        self.to_g = nn.Sequential(
            linearNoBias(c_query, c_model),
            nn.Sigmoid(),
        )
        self.to_out = nn.Sequential(nn.Linear(c_model, c_query), nn.Dropout(dropout))
        self.kq_norm = kq_norm
        if self.kq_norm:
            self.k_norm = RMSNorm(c_model)
            self.q_norm = RMSNorm(c_model)

        self.c_pair = c_pair
        if c_pair is not None:
            self.to_b = nn.Sequential(RMSNorm(c_pair), linearNoBias(c_pair, n_head))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_g[0].weight)
        nn.init.xavier_uniform_(self.to_out[0].weight)

    def forward(self, q, kv, attn_mask=None, pair_bias=None):
        """
        Args:
            q: [B, tok, n_q, c_query]
            kv: [B, tok, n_kv, c_kv]
            attn_mask: [n_q, n_kv]
        Returns:
            attn_out: [B, tok, n_q, c_query]
        """

        q = self.ln_q(q)
        kv = self.ln_kv(kv)

        q, k, v, g = self.to_q(q), self.to_k(kv), self.to_v(kv), self.to_g(q)

        if self.kq_norm:
            k = self.k_norm(k)
            q = self.q_norm(q)

        q, k, v, g = map(
            lambda t: rearrange(t, "b t n (h c) -> b h t n c", h=self.n_head),
            (q, k, v, g),
        )

        attn = einsum("bhtqc,bhtkc->bhtqk", q, k) * self.scale

        if pair_bias is not None:
            b = self.to_b(pair_bias)
            b = rearrange(b, "b t q k (h) -> b (h) t q k", h=self.n_head)
            attn = attn + b

        if attn_mask is not None:
            attn = attn.masked_fill(~attn_mask[None, None], float("-inf"))

            invalid_queries = torch.logical_not(
                torch.any(attn_mask, dim=-1, keepdim=False)
            )
            attn[:, :, invalid_queries, :] = 0.0

        attn = F.softmax(attn, dim=-1)
        attn_out = einsum("bhtqk,bhtkd->bhtqd", attn, v)
        attn_out = attn_out * g

        attn_out = rearrange(attn_out, "b h t n c -> b t n (h c)")
        attn_out = self.to_out(attn_out)
        return attn_out


class LocalAttentionPairBias(nn.Module):
    def __init__(
        self,
        c_a,
        c_s,
        c_pair,
        n_head,
        kq_norm=True,
        n_attn_seq_neighbours=2,
        n_attn_keys=128,
    ):
        super().__init__()
        self.c = c_a
        self.n_head = n_head

        self.to_q = linearNoBias(c_a, self.c)
        self.to_k = linearNoBias(c_a, self.c)
        self.to_v = linearNoBias(c_a, self.c)
        self.to_b = linearNoBias(c_pair, self.n_head)
        self.to_g = nn.Sequential(
            linearNoBias(c_a, self.c, bias=False),
            nn.Sigmoid(),
        )
        self.kq_norm = kq_norm
        if kq_norm:
            self.ln_q = RMSNorm(self.c)
            self.ln_k = RMSNorm(self.c)

        self.to_o = linearNoBias(self.c, c_a)

        if exists(c_s):
            self.ada_ln_1 = AdaLN(c_a=c_a, c_s=c_s)
            self.linear_output_project = nn.Sequential(
                LinearBiasInit(c_s, c_a, biasinit=-2.0),
                nn.Sigmoid(),
            )
        else:
            self.ln_1 = RMSNorm(c_a)

        self.n_attn_seq_neighbours = n_attn_seq_neighbours
        self.n_attn_keys = n_attn_keys
        self.use_checkpointing = True

    def forward(
        self,
        Q_L,
        C_L,
        P_LL,
        indices=None,
        f=None,
        X_L=None,
        full=False,
        chunked_pairwise_embedder=None,
        initializer_outputs=None,
    ):
        """
        Q_L: [D, L, c_a]
        C_L: [D, L, c_s]
        P_LL: [D, L, L, c_pair] or None (if using chunked mode)
        indices: [D, L, k] long
        chunked_pairwise_embedder: ChunkedPairwiseEmbedder for memory efficient computation
        initializer_outputs: Dict containing features for chunked computation
        """

        if not exists(indices):
            indices = create_attention_indices(
                f,
                n_attn_keys=self.n_attn_keys,
                n_attn_seq_neighbours=self.n_attn_seq_neighbours,
                X_L=X_L,
            )

        if chunked_pairwise_embedder is not None and P_LL is None:
            P_LL_sparse = chunked_pairwise_embedder.forward_chunked(
                f=f,
                indices=indices,
                C_L=initializer_outputs["C_L"],
                Z_init_II=initializer_outputs["Z_II"],
                tok_idx=f["atom_to_token_map"],
            )
            use_sparse_pll = True
        else:
            P_LL_sparse = None
            use_sparse_pll = False

        use_kernel = False

        def do_attention(Q_L, C_L, P_LL):
            if exists(C_L):
                Q_L = self.ada_ln_1(Q_L, C_L)
            else:
                Q_L = self.ln_1(Q_L)

            if use_kernel and not use_sparse_pll:
                q, k, v, g, b = (
                    self.to_q(Q_L),
                    self.to_k(Q_L),
                    self.to_v(Q_L),
                    self.to_g(Q_L),
                    self.to_b(P_LL),
                )
                q, k = (self.ln_q(q), self.ln_k(k)) if self.kq_norm else (q, k)
                attn_out = _fused_full_pairbias_attention(
                    Q_L=q,
                    K_L=k,
                    V_L=v,
                    P_LL=P_LL,
                    num_heads=self.n_head,
                    to_b=None,
                    to_g_linear=None,
                    to_o_linear=None,
                    w_ln_z_identity=None,
                    b_ln_z_identity=None,
                    attn_scale=1.0 / math.sqrt(self.c // self.n_head),
                )
            else:
                q, k, v, g = (
                    self.to_q(Q_L),
                    self.to_k(Q_L),
                    self.to_v(Q_L),
                    self.to_g(Q_L),
                )
                q, k = (self.ln_q(q), self.ln_k(k)) if self.kq_norm else (q, k)

                if use_sparse_pll:
                    b = self.to_b(P_LL_sparse)
                    attn_out = sparse_pairbias_attention(
                        Q=q,
                        K=k,
                        V=v,
                        B=b,
                        G=g,
                        gather_bias=False,
                        indices=indices,
                        H=self.n_head,
                        full=full,
                    )
                else:
                    b = self.to_b(P_LL)
                    attn_out = sparse_pairbias_attention(
                        Q=q,
                        K=k,
                        V=v,
                        B=b,
                        G=g,
                        gather_bias=True,
                        indices=indices,
                        H=self.n_head,
                        full=full,
                    )

            Q_L = self.to_o(attn_out)
            if exists(C_L):
                Q_L = self.linear_output_project(C_L) * Q_L

            return Q_L

        do_attention_ = (
            activation_checkpointing(do_attention)
            if self.use_checkpointing
            else do_attention
        )

        if use_sparse_pll:
            return do_attention_(Q_L, C_L, P_LL_sparse)
        else:
            return do_attention_(Q_L, C_L, P_LL)


def sparse_pairbias_attention(
    Q, K, V, B, indices, H, gather_bias=True, G=None, full=False
):
    """
    Computes attention with sparse pairwise bias, where indices specify which
    keys to attend to for each query token.
    Q: (D, L, c)  # query vectors
    K: (D, L, c)  # key vectors
    V: (D, L, c)  # value vectors
    B: (L, L, H)  # attention bias (unbatched or pre-gathered and [D, L, k, H])
    G: (D, L, c)  # Gate (optional)
    B2: (D, L, 14, 14, H)  # attention bias (batched and within token) (optional)
    indices: (D, L, k_neigh) long  # indices of neighbours to attend to
    Returns
    -------
    attn_out: (D, L, c)  # attention output
    """
    D, L, c = Q.shape
    k = indices.shape[-1]

    if full:
        return pairbias_attention_(
            Q=Q,
            K=K,
            V=V,
            B=B,
            H=H,
            valid_mask=indices_to_mask(indices),
            G=G,
        )

    batch_idx = torch.arange(D, device=Q.device).view(-1, 1, 1)
    K_gathered = K[batch_idx, indices].contiguous()
    V_gathered = V[batch_idx, indices].contiguous()

    if gather_bias:
        query_idx = torch.arange(L, device=Q.device).view(1, L, 1)
        query_idx = query_idx.expand(D, -1, k)
        if B.ndim == 3:
            B_gathered = B[query_idx, indices, :]
        elif B.ndim == 4:
            B_gathered = B[batch_idx, query_idx, indices, :]
    else:
        assert B.shape == (D, L, k, H), "B must be batched with shape (D, L, k, H)"
        B_gathered = B
    B_gathered = B_gathered.contiguous()

    Q = Q.reshape(D, L, H, c // H)
    K_gathered = K_gathered.reshape(D, L, k, H, c // H)
    V_gathered = V_gathered.reshape(D, L, k, H, c // H)
    B_gathered = B_gathered.reshape(D, L, k, H)
    Q = Q.permute(0, 2, 1, 3)
    K_gathered = K_gathered.permute(0, 3, 1, 2, 4)
    V_gathered = V_gathered.permute(0, 3, 1, 2, 4)
    B_gathered = B_gathered.permute(0, 3, 1, 2)

    attn = torch.einsum("...ld,...lkd->...lk", Q, K_gathered)
    attn = attn / sqrt(c // H)
    attn = attn + B_gathered
    attn = torch.softmax(attn, dim=-1)
    attn_out = torch.einsum(
        "...ij,...ijc->...ic", attn, V_gathered
    )

    if G is not None:
        G = G.reshape(D, L, H, c // H).permute(0, 2, 1, 3)
        attn_out = attn_out * G

    attn_out = attn_out.permute(0, 2, 1, 3)
    attn_out = attn_out.reshape(D, L, c).contiguous()

    return attn_out


def pairbias_attention_(Q, K, V, B, H, valid_mask=None, G=None):
    """
    Fully connected variant of pairbias attention with optional gating and valid mask.
    Equivalent to sparse attention but with all keys

    Attn_out: [batch_size, query_length, H * head_dim]
    """
    D, L, c = Q.shape
    k = L

    Q = Q.reshape(D, L, H, c // H)
    K = K.reshape(D, k, H, c // H)
    V = V.reshape(D, k, H, c // H)
    B = B.reshape(D, L, k, H)

    Q = Q.permute(0, 2, 1, 3)
    K = K.permute(0, 2, 1, 3)
    V = V.permute(0, 2, 1, 3)
    B = B.permute(0, 3, 1, 2)

    attn = torch.einsum("...ld,...kd->...lk", Q, K)
    attn = attn / sqrt(c // H)
    attn = attn + B
    if exists(valid_mask):
        attn = attn.masked_fill(~valid_mask.unsqueeze(1), float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    attn_out = torch.einsum("...ij,...jc->...ic", attn, V)

    if G is not None:
        G = G.reshape(D, L, H, c // H).permute(0, 2, 1, 3)
        attn_out = attn_out * G

    attn_out = attn_out.permute(0, 2, 1, 3)
    attn_out = attn_out.reshape(D, L, c).contiguous()

    return attn_out


def _fused_full_pairbias_attention(
    *,
    Q_L,
    K_L,
    V_L,
    P_LL,
    num_heads: int,
    to_b: nn.Linear,
    to_g_linear: nn.Linear,
    to_o_linear: nn.Linear,
    w_ln_z_identity: torch.torch.Tensor,
    b_ln_z_identity: torch.torch.Tensor,
    attn_scale: float | None = None,
):
    """
    Uses cuequivariance_torch.attention_pair_bias for dense (full) attention.
    Expects Q/K/V to be projected *before* calling this function.
    """
    B, L, c = Q_L.shape
    H = num_heads
    assert c % H == 0, "Model dim must be divisible by num_heads"
    DH = c // H

    q = Q_L.reshape(B, L, H, DH).permute(0, 2, 1, 3).contiguous()
    k = K_L.reshape(B, L, H, DH).permute(0, 2, 1, 3).contiguous()
    v = V_L.reshape(B, L, H, DH).permute(0, 2, 1, 3).contiguous()

    s = Q_L.contiguous()

    mask = None

    w_proj_z = to_b.weight
    b_proj_z = to_b.bias if hasattr(to_b, "bias") else None

    w_proj_g = to_g_linear.weight
    b_proj_g = to_g_linear.bias if hasattr(to_g_linear, "bias") else None

    w_proj_o = to_o_linear.weight
    b_proj_o = to_o_linear.bias if hasattr(to_o_linear, "bias") else None

    w_ln_z = w_ln_z_identity.to(dtype=P_LL.dtype, device=P_LL.device)
    b_ln_z = b_ln_z_identity.to(dtype=P_LL.dtype, device=P_LL.device)

    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(DH)

    out, _proj_z = cueq_attention_pair_bias(
        s=s,
        q=q,
        k=k,
        v=v,
        z=P_LL,
        mask=mask,
        num_heads=H,
        w_proj_z=w_proj_z,
        w_proj_g=w_proj_g,
        w_proj_o=w_proj_o,
        w_ln_z=w_ln_z,
        b_ln_z=b_ln_z,
        b_proj_z=b_proj_z,
        b_proj_g=b_proj_g,
        b_proj_o=b_proj_o,
        attn_scale=attn_scale,
        compute_pair_bias=True,
        multiplicity=1,
    )
    return out
