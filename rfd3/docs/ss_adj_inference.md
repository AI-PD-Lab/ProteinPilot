# SS/ADJ Inference

RFD3 accepts the same tensor conventions as RFD1:

- SS: `[L]` float or integer tensor with `0=helix`, `1=strand`, `2=loop`, `3=unconditioned`.
- ADJ: `[L,L]` binary, symmetric tensor with a zero diagonal.
- Files may be PyTorch `.pt` tensors or NumPy `.npy` arrays.

Example input JSON:

```json
{
  "example": {
    "length": "150",
    "ss_file": "/path/target_ss.pt",
    "adj_file": "/path/target_adj.pt"
  }
}
```

The tensor length must exactly match the final RFD3 token count and ordering. Standard H/E adjacency uses the RFdiffusion1 definition: reconstructed C-beta contacts below 6 A between distinct secondary-structure blocks. The optional loop-aware training mode retains the custom requirement of at least three C-alpha pairs below 6 A. For a protein-only unconditional design, this is the requested design length. For motifs, ligands, multiple chains, or atomized components, generate tensors in the final token order rather than assuming protein-residue order.

`select_ss` remains available for assigning labels inline by residue selection. The old `select_adj: true` mode is rejected because adjacency cannot be inferred from coordinates after regular-diffusion coordinates have been erased; use `adj_file` instead.

RFD1-style masked scaffold expansion is available in the same input entry:

```json
{
  "example": {
    "length": "150",
    "ss_file": "/path/target_ss.pt",
    "adj_file": "/path/target_adj.pt",
    "mask_loops": true,
    "sampled_insertion": "0-15",
    "sampled_N": "0-5",
    "sampled_C": "0-5",
    "ss_mask": 1
  }
}
```

An integer `N` samples inclusively from `0` to `N`; `A-B` samples from that explicit range. Each loop samples independently. Original loops, inserted residues, terminal additions, and `ss_mask` boundary residues receive SS label `3` and zero ADJ rows/columns. `sampled_insertion` expands existing SS label `3` segments; `mask_loops: true` additionally converts loop segments into insertion sites. N- and C-terminal additions do not require loop masking. For an explicit contig, the parser constrains its total sampled length to the expanded tensor length and rejects incompatible mappings. Expansion is not supported with partial diffusion.

Classifier-free guidance can be enabled with `inference_engine.inference_sampler.use_classifier_free_guidance=true`. The unconditional pass masks SS and zeros ADJ, while the conditional pass uses the supplied tensors.
