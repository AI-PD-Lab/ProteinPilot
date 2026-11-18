# SS-Ratio Conditioning for RFD3

Adds a continuous **global** secondary-structure conditioning token (`ss_ratio`)
to RFD3. One scalar per protein chain biases generation toward a target α/β
ratio without requiring per-residue topology knowledge at inference time.

---

## Concept

```
ss_ratio = E / (H + E)   ∈ [0, 1]
  0.0  →  pure α-helix
  1.0  →  pure β-sheet
 -1.0  →  unconditional sentinel (model uses its prior; used for CFG)

H = helix fraction  (biotite annotate_sse label 'a')
E = strand fraction (biotite annotate_sse label 'b')
```

**Global, not per-residue.** A single scalar is computed per chain and
broadcast to every token. The model learns to associate that global budget
with the overall SS composition of generated structures.

A **25 % conditioning dropout** is applied at training time: one quarter of
examples receive `ss_ratio = -1.0` (unconditional). This trains the model to
run both with and without the signal, which is required for
classifier-free guidance (CFG) at inference.

---

## Architecture

```
ss_ratio scalar  [I, 1]   (same value broadcast to all I tokens)
        │
        ▼
  SSRatioEncoder                    ← NEW module (ss_conditioning.py)
  ┌─────────────────────────────┐
  │  clamp to [0, 1]            │
  │  × fixed Fourier freqs      │   k·π for k = 1…16
  │  → [sin(·), cos(·)]  [I,32] │   fixed basis, not learned
  │  Linear(32 → 384)           │   learned
  │  SiLU                       │
  │  Linear(384 → 384)          │   learned, ZERO-INITIALISED
  │  × mask (0 if sentinel)     │   zeros out unconditional samples
  └─────────────────────────────┘
        │
        ▼  additive injection
       S_I  [I, 384]               token representation in TokenInitializer
        │
        ▼
  → rest of diffusion module
```

The final `Linear(384 → 384)` is zero-initialised so at the start of
fine-tuning the encoder contributes exactly zero to `S_I`. The pre-trained
backbone behaviour is preserved until gradients begin to update it.

**Why Fourier features and not a single linear layer?**
A `Linear(1, 384)` can only learn a monotone response to the ratio — one
global "direction" in representation space. The Fourier expansion lets the
model learn independent, frequency-specific responses (e.g. behave
differently near 0.0, 0.5, and 1.0 simultaneously).

---

## Implementation map

| File | Role |
|---|---|
| `foundry/models/rfd3/src/rfd3/model/layers/` (`TokenInitializer`) | Fourier encoder `ss_encoder`, zero-initialised, added to the token initialiser |
| `foundry/models/rfd3/src/rfd3/constants.py` | `"ss_ratio": -1.0` in `OPTIONAL_CONDITIONING_VALUES` (inference default = unconditional) |

The ss_ratio encoder is present in every Harmonia-RFD3 checkpoint. It was trained in the first fine-tune stage (frozen backbone,
learning rate 1e-4, five epochs) and then carried unchanged through the SS/ADJ stages. The original training launchers were
specific to one cluster and are not part of this release; the development history is on the `phase10` and `harmonia-v2` branches.
