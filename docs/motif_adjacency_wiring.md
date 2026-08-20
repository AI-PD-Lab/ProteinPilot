# Motif × Adjacency: explicit wiring and training plan

Companion to `ss_adj_training_plan.md` and `ss_adj_inference.md`.

**Status.**

Implemented and validated, flag-gated **off** by default unless noted:
- `use_cb` — score C-beta (real predicted CB -> virtual CB for glycine -> CA)
- `negative_weight` — native-matched repulsive arm for explicit non-contacts
- `detach_motif` — freeze fixed-motif coordinates inside the loss
- chain-aware block recovery — active whenever `token_chain_idx` is present
- `mask_loop_pairs` — default **ON**; changes label semantics vs the 0645-0684 lineage
- `typed_blocks` — ligand / nucleic-acid adjacency (§12), incl. helical-turn duplex
  blocks spanning both strands and a ds/ss split by local pairing state
- non-protein SS label -> MASKED instead of LOOP
- typed metrics: per-pair-kind recovery + **non-adjacency violation rate** (§13),
  with `non_contact_blocks` now emitted by the benchmark generator

Validated on real structures, not only synthetic geometry:
- 1BNA (B-DNA): helical-turn blocks span both strands, 24/24 coverage
- 1F1T (RNA aptamer): 5 ds + 7 ss blocks from intramolecular pairing, no SS graph
- 1LMB (lambda repressor-operator): protein-DNA contacts **0 -> 2584**
- manifest negatives cross-checked against the .pt tensors: 13 rectangles, 0 drift

Also implemented since:
- negatives revealed in the ADJ-only sampler, plus a complete-matrix sub-branch
  (50/20/30 sparse-with-negatives / complete / masked)
- `motif_region_mask_prob` 0.5 -> 0.2 (designed<->motif adjacency now survives in
  79% of motif examples, up from 48.9%)
- `adj_pairs` (§11): contig-relative binder/motif adjacency, verified on the real
  5OQV spec. Needs NO training counterpart -- see §11.
- inference with no `adj_file` now returns a fully-masked matrix instead of an
  all-zero one that asserted non-contact on every pair
- validation restored (§13) against a stratified protein / ligand / NA val set

**Not done: any training run using the above.** Every claim here is unit-level or
single-structure. Nothing has been exercised by a dataloader or a GPU.

Note that `mask_loop_pairs`, `motif_region_mask_prob` and the ADJ-only split each
change label semantics, so phase 6 is NOT comparable with the 0645-0684 lineage,
and it moves several variables at once -- a delta against 0684 will not be
attributable to any single change.

Plan and sequencing in §13; non-protein fix plan in §15.

## 1. The problem this addresses

RFD3 (epoch-0675 / 0684 lineage) **beats RFD1 on designability in most de novo
cases**, and beats it on adjacency-block recovery generally. The exception is
**motif / binder scaffolding**, where RFD1 is partially ahead on AlphaFold
metrics. Two further RFD1 advantages are reported:

- RFD1 keeps **explicitly non-adjacent** blocks non-adjacent; RFD3 associates them
  much more often. (Masked = indeterminate, so this is only about explicit `0`s.)
- RFD1 holds up better on **large multi-chain motif systems**.

So this is not a general quality gap. It is specific to the motif branch, and the
hypothesis of this document is that **motif and adjacency are not actually wired
together** — not in the inference spec, not in the training mixture, and not in
the loss.

## 2. Evidence

### 2.1 The motif benchmark is out of distribution on two axes at once

| | tokens | vs `crop_size=256` | conditioned block pairs | vs training max |
|---|---|---|---|---|
| de novo benchmark | 100–120 | 0.4× ✅ | 3–11 | ✅ (max seen 27) |
| motif (5OQV) | **456** | **1.8×** ❌ | **58** | **2.1×** ❌ |

Measured distribution of `adjacency_n_conditioned_block_pairs` over phase 5
(468 logged batches): min 1, median 5, p90 15, **max 27**, mean 6.4. The ceiling
is structural — a 256-token crop cannot hold more SS blocks than that.

This maps exactly onto where RFD3 wins (de novo, in-distribution) and loses
(motif, 2× outside on both size and specification density).

### 2.2 Designed↔motif adjacency is currently masked out at inference

The 5OQV input supplies `5_scaffold_adj_diffused288.pt`, which is **288×288**,
while the assembled complex has **456 tokens** (288 designed + 4×42 target).
Combined with `ss_adj_contig: "A1-288"`, every target token receives `adj=2`.

**The model is told nothing about which designed blocks should touch the target.**
It is not failing at designed↔motif adjacency; it was never given any.

### 2.3 Training does support it, but halves it on purpose

`_maybe_mask_motif_region` (`ss_adj_transforms.py:543-544`) strips SS/ADJ from
motif tokens with `motif_region_mask_prob = 0.5`:

```python
adj_matrix[motif_idx, :] = 2
adj_matrix[:, motif_idx] = 2
```

So half of all motif-bearing examples have exactly the binder↔target signal we
care about deleted. For a binder-focused model this is backwards.

### 2.4 The ADJ-only mode contains no negatives at all

`_sample_sparse_contacts` builds `torch.full_like(adj_matrix, 2)` and then copies
back **only selected contact rectangles**, which are all `1`s. ADJ-only is the
mode closest to how a binder is actually driven (adjacency, no SS), and in it the
model has **never seen an explicit non-adjacency**. This is an omission, not a
tuning choice, and is a plausible direct cause of the over-association above.

Where explicit `0`s actually reach the model:

| mode | prob | adj handling | explicit `0`s |
|---|---|---|---|
| unconditional | 10% | all → 2 | no |
| SS-only | 10% | all → 2 | no |
| ADJ-only | 10% | all → 2, only contact rectangles revealed | **no** |
| FULL | 20% | untouched | yes, all |
| interaction-site | 40% | all → 2, exposed submatrix copied | yes, in-patch |
| remodeling | 10% | clone, masked rows/cols → 2 | yes, outside masked segs |

### 2.5 Chain identity is implicit and positional

- Training: chain identity is used, but **only for block construction** —
  `_find_ss_blocks(labels, chain_ids)` breaks blocks at chain boundaries.
- The adjacency matrix itself carries **no chain information**. It is I×I of
  {0,1,2}; chain structure exists only in token ordering.
- Inference: alignment is purely positional against the contig's token order. In
  5OQV, rows 0–287 are the binder and 288–455 the target *by convention only* — a
  contig reorder silently misaligns the entire matrix.
- **Loss-side bug:** `_blocks_from_adjacency_rows` recovers blocks purely from row
  uniformity and has no chain information. Two tokens either side of a chain break
  with coincidentally identical rows are merged into one block. Live risk for
  multi-chain binder training.

## 3. What we enforce today vs RFD1

Verified against RFD1 source
(`RFdiffusion/helper_scripts/make_secstruc_adj.py`):

```python
def construct_block_adj_matrix(sstruct, xyz, cutoff=6, include_loops=False):
    # "By eye, Nate thinks 6A is a good Cb distance cutoff"
    Cb = generate_Cbeta(N, Ca, C)          # virtual CB for EVERY residue
    dist = get_pair_dist(Cb, Cb)
    if torch.any(dist[bi:ei, bj:ej] < cutoff): block_adj[...] = 1
```

| | atoms | threshold | comparator | multiplicity |
|---|---|---|---|---|
| RFD1 | virtual Cβ (all residues) | 6.0 | `<` | any 1 pair |
| our label, H/E↔H/E | reconstructed Cβ | 6.0 | `<` | any 1 pair |
| our label, loop-involved | Cα | 6.0 | `<` | **≥3 pairs**, and gated by `loop_adj` |
| **our loss** | **Cα** | 6.0 up / 4.5 down | soft | any 1 pair |
| **our recovery metric** | **real Cβ, Cα for Gly** | 6.0 | `<=` | any 1 pair |

Three consequences:

1. **Atom mismatch.** Label is Cβ<6; loss enforces Cα≤6. Cβ projects toward the
   partner, so a pair already satisfying the label at Cβ=6 sits at Cα≈6.5–8. The
   loss keeps pulling for another 0.5–2 Å past the label's condition. Consistent
   with the 3.81 Å mean cross-block Cα distance measured on epoch-0670.
2. **`loop_adj=false`** on the `pn_unit` / `interface` datasets means every
   loop-involved pair is skipped and left at `0` — labelled "explicitly
   non-adjacent" while physically packed. Loops are ~half of all residues.
3. **The loss was the sole outlier.** `analyze_ss_adj_conditioning.py` already
   reads real Cβ (Cα only for glycine), so label and metric were both on Cβ while
   only the loss used Cα. Switching the loss to Cβ therefore *aligns* it with the
   metric — it does not require a re-baseline. (An earlier draft of this document
   claimed the metric was Cα-based and that recovery would drop; that was wrong.)

## 4. Cβ in RFD3 — is a 1:1 port possible?

RFD1 is frame-based and must synthesise Cβ. **RFD3 is all-atom and already has
one**: `constants.py:186` gives designed residues
`["N", "CA", "C", "O", "CB"] + virtuals`, so a predicted Cβ exists at every
designed position.

A Cβ mask is also already reachable —
`get_af3_token_representative_masks(atom_array, central_atom="CB")`; the
`central_atom` parameter exists for exactly this and is applied to
`is_standard_aa_not_glycine`.

**But it is not automatically 1:1.** Open items that must be settled before
claiming fidelity:

- **Glycine.** RFD1 generates a virtual Cβ for *every* residue including Gly.
  `central_atom="CB"` falls back to **CA** for Gly (`glycine_representative_atom`
  is hardcoded to `"CA"`). Real difference of ~1.5 Å at every Gly position.
- **Real vs virtual Cβ.** Our label calls `_reconstruct_cb` (virtual, from GT
  backbone). A real-CB loss would compare *virtual* Cβ on the label side against
  *predicted real* Cβ on the loss side — two different constructions.
- **Designed-position Cβ meaning.** During diffusion the sequence is not fixed;
  the predicted CB is a 5-atom-representation slot, not a rotamer-resolved atom.
  Needs confirming that it is trained to sit at ideal-Cβ geometry.

**Resolution taken (implemented):** score the **real predicted Cβ**, because RFD3
is all-atom and CB is an actual MSE-supervised atom — reconstructing from N/CA/C
would discard the model's own CB and never supervise it. Virtual Cβ
(`_reconstruct_cb`'s formula, RFD1's constants, verified to 1.9e-6) is the
**fallback for glycine**, where no CB exists; Cα is the last resort for
non-protein tokens. This is the same preference order
`analyze_ss_adj_conditioning.py` already uses, so loss and metric agree.

The superseded alternative — reconstruct everywhere — needs N and C masks,
which do not exist as features today but are derivable — designed residues have a
fixed 5-atom layout.

**Alternative that sidesteps the whole question:** native-matched targets (§6.2).
Compare predicted Cα against *native* Cα for the same pair — like with like, no
threshold, no atom question, and automatically correct for helix (~8–10 Å) vs
strand (~5 Å) packing.

## 5. Design: wire motif and adjacency together explicitly

### 5.1 Inference — chain/residue-keyed adjacency pairs

Replace positional dependence with explicit selections, resolved against the
assembled structure the way `ss_adj_contig` already is:

```yaml
adj_pairs:
  - "A1-30  :  B10-40"     # adjacent
  - "A1-30  !: B60-80"     # explicitly NON-adjacent
```

Benefits: contig-order independent; expresses negatives (nobody hand-authors `0`s
in a 456×456 tensor); makes hotspot-style binder specs writable; and lets the
matrix span motif tokens, fixing §2.2 with no retraining.

Must still accept the existing `.pt` path for backwards compatibility.

### 5.2 A first-class "binder mode" in training

Today motif marking (`island`/`tipatom`/`ppi` via `SampleConditioningType`) and
fold conditioning (`FeaturizeSSADJ`) are independent systems that only meet in
`_maybe_mask_motif_region`. Note that **interaction-site mode does not mark
anything as a motif** — it only reveals SS/ADJ for contact patches; no atoms are
fixed.

Proposed explicit mode: given a motif-bearing example, mark the target as fixed,
then reveal adjacency **between designed blocks and selected target blocks**, plus
explicit non-adjacency to other target blocks. That is the actual inference task,
trained directly instead of approximated.

### 5.3 Chain-aware blocks in the loss

Pass a token chain index into `AdjacencyContactLoss` and split blocks on it,
mirroring `_find_ss_blocks`. Fixes §2.5.

### 5.4 Detach the motif side of a conditioned pair

For a designed↔fixed-target pair, gradient currently acts on both sides. Use
`is_motif_atom_with_fixed_coord` to detach the motif side so the binder moves to
the target rather than the two negotiating. Minor, but free and correct.

## 6. Loss changes

### 6.1 Repulsive arm for explicit non-contacts — IMPLEMENTED (flag-gated, default off)

Also implemented and flag-gated, all default-off: `use_cb` (score Cβ, §4),
`detach_motif` (§5.4), and chain-aware block recovery (§5.3, always on — it only
activates when `token_chain_idx` is present).

`negative_weight` / `negative_threshold` / `negative_margin` on
`AdjacencyContactLoss`. Only pairs whose rectangle is **entirely `0`** are scored
(`.all()`, not `.any()`, so any masked entry leaves the pair indeterminate).

Crucially the floor is **native-matched, not a fixed threshold**:

```
floor = min(native_min_dist - negative_margin, negative_threshold)
```

A flat threshold would be catastrophic here: with `loop_adj=false` most
loop-involved pairs are labelled `0` while physically touching, so "push all `0`
pairs past 6–8 Å" would try to peel the protein apart from itself.

Validated numerically:

| case | native | floor | penalty |
|---|---|---|---|
| loop packed at 4 Å, labelled 0 | 4.0 | 3.0 | 0.000 at 3.5/4/5 Å; 0.245 at 2 Å |
| genuine non-contact | 15.0 | 8.0 | 0.762 @4 Å, 0.245 @7 Å, 0.000 @8 Å+ |
| positive pairs | — | — | bit-identical to before |
| `negative_weight=0` | — | — | 0.0000 (default unchanged) |

Gradient on a violating negative pair pushes the blocks apart (−0.075 on the
separation axis).

### 6.2 Native-matched positive arm — PROPOSED

Same treatment for the attractive arm: flat bottom at `[d_gt − τ, d_gt + τ]`
instead of the constant `[4.5, 6.0]`. Removes the Cα/Cβ mismatch, gets helices and
strands right simultaneously, and leaves no invented constants.

Note this is *separate* from `use_cb`. The metric is already Cβ-based, so
`use_cb` needs no metric change. Native-matching the positive arm would change
the target from "within Cβ 6 Å" to "at native spacing", which the Cβ≤6 metric
would still score as recovered in the strand case but not necessarily for helices
sitting at native 8–10 Å Cβ. That one does warrant a re-baseline; `use_cb` alone
does not.

## 7. Training mixture — argued both ways

**For changing it:** ADJ-only has no negatives (§2.4); `motif_region_mask_prob`
deletes the binder signal (§2.3); the stated use case is adjacency-heavy.

**Against:** the 0645→0684 lineage shares one mixture, so changing it destroys
attribution for future benchmark deltas; interaction-site at 40% is already
binder-shaped; cutting SS-only weakens a supported inference mode (SS without
adjacency); and several simultaneous changes make an uninterpretable experiment.

**Recommendation: do not rebalance yet.** Fix only the two items that are defects
rather than preferences —

1. reveal negatives in `_sample_sparse_contacts` (ADJ-only)
2. `motif_region_mask_prob` 0.5 → ~0.2

— and leave the six mode probabilities alone. Keep `unconditional` at 10%
regardless: CFG needs an unconditional branch to have a guidance direction at all.

If over-association persists after that, rebalance with one variable moving:

| mode | now | candidate |
|---|---|---|
| unconditional | 10% | 10% (do not cut) |
| SS-only | 10% | 5% |
| ADJ-only | 10% | 15% (with negatives) |
| FULL | 20% | 20% |
| interaction-site | 40% | 40% |
| remodeling | 10% | 10% |

## 8. Ordering

Cheapest and most informative first; each step is independently testable.

1. **456×456 adjacency for 5OQV** (§2.2). Inference-only, no retraining. If AF3
   jumps, the motif gap is largely a specification bug.
2. **Run the motif benchmark on epoch-0645** (pre-adjacency-loss). Separates "we
   broke it" from "inherited from the base model". Pure inference.
3. Chain-aware blocks (§5.3) + config wiring for §6.1.
4. Sparse-mode negatives + `motif_region_mask_prob` (§7), short finetune from
   0684 with `negative_weight` on.
5. Explicit `adj_pairs` spec (§5.1).
6. Native-matched positive arm **and** metric change together (§6.2), as its own
   ablation with a full re-baseline.
7. Only if still short: `crop_size` 256→384 for a motif finetune. Pair memory is
   quadratic (2.25×) and this setup has OOM history — smoke-test first.

## 9. Open questions

- Does predicted Cβ at designed positions sit at ideal-Cβ geometry during
  diffusion, or is it effectively a free slot? Decides whether real-Cβ is usable
  or virtual reconstruction is required (§4).
- Is `loop_adj=false` deliberate? It makes most loop pairs explicit negatives,
  which interacts with everything in §6.
- Should the metric report Cβ<6 recovery *and* mean native-distance deviation?
  The latter is not gameable by over-tightening.

---

## 10. Loop pairs must be MASKED, not labelled non-adjacent — FIXED

RFD1 `BlockAdjacency.mask_ss_adj`:

```python
if self.mask_loops:
    adj[~expanded_mask, :] = 0
    adj[:, ~expanded_mask] = 0
adj[~expanded_mask] = 2      # <-- MASK
adj[:, ~expanded_mask] = 2
```

Its docstring: *"adj: block adjacency with equivalent masking as ss"*.

Our `_adj_blocks_from_coords` emitted a binary `{0,1}` array; with `loop_adj=false`
loop-involved pairs were `continue`d and left at **0 = explicit non-contact**,
for ~half the residues in every structure.

A masked state existed only on `ss_labels` (`_SS_MASKED = 3`); `2` reached `adj`
solely through CFG dropout. So where SS is supplied the model could in principle
infer that loop rows are uninformative — but in **ADJ-only mode `ss_labels` is
fully masked**, making a loop `0` indistinguishable from a real non-contact.

Fixed via `mask_loop_pairs: bool = True` (default = RFD1 behaviour): loop-involved
pairs are stamped `2`. Verified: loop×helix → `2`, distant helix×helix → `0`.

**This changes label semantics for all training.** Set `mask_loop_pairs=False` to
reproduce the 0645→0684 behaviour exactly.

## 11. Contig-driven binder/motif specification — NOT IMPLEMENTED

Required: CLI + inference must let a contig say which chain is the **binder**
(designed) and which is the **motif** (external PDB chain, atoms held fixed), with
adjacency expressible between them — and the same pattern must be **trained**, not
only supported at inference.

Sketch:

```
contig:   "A1-120/0 B1-84"           # A designed, B from external pdb
motif:    "B"                        # fixed-coordinate chain(s)
adj_pairs:
  - "A20-35  :  B10-25"              # binder helix must contact this target patch
  - "A60-75  !: B40-60"              # and must NOT contact that one
```

Resolved against the assembled structure like `ss_adj_contig` already is, so it is
order-independent and can span motif tokens (§2.2).

**No training counterpart is required, contrary to an earlier draft of this
document.** `adj_pairs` is not a new modality: it is the pattern training already
produces. `SampleConditioningType` runs BEFORE `FeaturizeSSADJ`, so adjacency is
computed over all tokens including the ones just marked as fixed motif, and the
CFG modes reveal a subset. Interaction-site mode (40%) in particular exposes
structurally-selected contact patches while masking the body -- with cross-chain
patches first-class, since blocks break at chain boundaries -- which is exactly
the adj_pairs shape.

Measured: designed<->motif adjacency survives `_maybe_mask_motif_region` in
**79%** of motif-bearing examples at `motif_region_mask_prob=0.2` (48.9% at the
old 0.5). So lowering that probability is not cosmetic -- it directly governs how
often this pattern is seen.

The gap was only ever that INFERENCE could not express it: a positionally-indexed
288x288 matrix on 456 tokens cannot say "designed block 5 contacts target chain
B". That is what adj_pairs fixes.

## 12. Adjacency for ligands and nucleic acids

### 12.1 Current state — worse than "unsupported"

`_get_ss_training` assigns secondary structure like this:

```python
ss = np.full(I, _SS_LOOP, dtype=np.int64)   # <-- default for EVERYTHING
ss[helix_tok] = _SS_HELIX
ss[sheet_tok] = _SS_STRAND
```

**Every ligand atom and every nucleotide is labelled `_SS_LOOP`.** They then hit
the loop branch of `_adj_blocks_from_coords`, which with `loop_adj=false` used to
`continue` and leave the pair at **0 — explicitly non-adjacent**. So in FULL mode
the model was being told, for every protein-ligand and protein-nucleic-acid
complex in the training set, that the ligand/NA is *definitively not in contact*
with the protein. The `mask_loop_pairs` fix (§10) incidentally repairs this: those
pairs are now masked rather than asserted-negative.

The data is present — the phase-5 pipeline resolves `allowed_types: ALL`, so
ligands and NA are not filtered out. This is a **labelling** problem, not a data
or architecture problem.

### 12.2 What needs no change

- **`ADJAdapter`** is `nn.Embedding(3, c_z)` over {0,1,2}. Token-type agnostic.
- **`SSAdapter`** is `nn.Embedding(4, ...)`; non-protein simply takes the masked
  state. The model already receives `is_dna` / `is_rna` / `is_ligand` separately,
  so it can distinguish "SS masked because non-protein" from "SS dropped by CFG".
- **`_contact_coords` already degrades correctly.** Ligand atoms and nucleotides
  have no `CB` and no atoms named exactly `N`/`CA`/`C`, so both the real-CB and
  virtual-CB branches fail their guards and it falls through to the token
  representative. Since **ligands are atomized (one token per atom)**, the token
  representative *is* the atom — so min-over-token-representatives for a ligand
  block is exactly heavy-atom minimum distance, for free. For nucleotides the
  representative is C2 (pyrimidine) / C4 (purine).

So no model surgery is required.

### 12.3 What must change

**1. Non-protein SS label: `LOOP` → `MASKED`.** Calling a ligand a loop is simply
false and routes it into a rule designed for flexible peptide segments.

**2. Type-aware contact rule.** Cβ<6 Å is a protein-packing proxy and does not
transfer. Proposed table, applied per block *pair*:

| pair | atoms | cutoff | multiplicity |
|---|---|---|---|
| protein ↔ protein (H/E) | virtual Cβ | 6.0 | any 1 |
| protein ↔ protein (loop) | Cβ (§14.3) | 6.0 | ≥3 |
| protein ↔ ligand | heavy atoms | **4.5** | ≥1 |
| protein ↔ NA | heavy atoms | **5.0** | ≥2 |
| NA ↔ NA | heavy atoms | 5.0 | ≥2 |
| ligand ↔ ligand / ligand ↔ NA | heavy atoms | 4.5 | ≥1 |

Rationale: protein Cβ 6 Å encodes "these two elements pack against each other"
through a side-chain proxy. Ligand and NA contacts are direct van-der-Waals
interactions, conventionally called at 4–5 Å heavy-atom. Using 6 Å there would
label near-neighbours as contacts and make the conditioning meaningless.

**3. Block definition.**
- *Ligand*: one block per ligand entity. Falls out of the existing machinery for
  free — all its tokens share the SS label and the chain, so `_find_ss_blocks`
  already groups them. Large flexible ligands would be better split per rigid
  fragment, but that needs a fragmenter; defer.
- *Nucleic acid*: one block per chain is too coarse (a 20-mer is one block, so a
  single contact marks the whole strand adjacent). Use **windows of ~4 nucleotides**
  — roughly comparable in size to a short SS element. The structurally principled
  version is base-paired **stems** (the duplex analogue of a β-strand), obtainable
  via `biotite.structure.base_pairs()`; worth doing as v2.

**4. The heavy-atom rule needs atoms, not representatives.** For protein↔ligand
this is free (atomized), but a nucleotide is one token with one representative, so
a genuine heavy-atom minimum needs the underlying atoms. Two options: (a) accept
the C2/C4 representative with a compensating larger cutoff, or (b) compute the
label from full atom coordinates in `get_adj_from_token_array`, which already
receives the `atom_array`. **(b) for the label** is correct and cheap; the loss can
keep using representatives, since the loss only needs a differentiable proxy that
is monotone in the true distance.

### 12.4 Honest uncertainties

- **Dataset composition is unverified.** `mean_lddt_dna` and `mean_lddt_rna` sit at
  exactly 1.0000 for every logged batch of phase 5, which is what an absent
  modality looks like. Before investing here, count how many training examples
  actually contain protein-NA and protein-ligand interfaces. If NA is effectively
  absent, NA adjacency cannot be trained no matter how it is labelled.
- **Cutoff values above are conventional, not tuned.** They should be checked
  against the distance distribution of real interfaces in the training set.
- **Conditioning-only vs supervised.** Adding these pairs to `AdjacencyContactLoss`
  means the well and the repulsive arm start acting on ligand geometry. Starting
  **conditioning-only** (label + adapter, no loss term) is cheaper and much less
  likely to damage ligand placement; add supervision once recovery is measurable.

## 13. Plan

Ordering principle: **diagnose before building**, then cheapest-and-most-certain
first, and never move two variables that the benchmark cannot separate.

### Phase 0 — two free experiments (inference only, no GPU-hours of training)

Neither needs a retrain and both change what is worth building next.

1. **456×456 adjacency on 5OQV.** Supply designed↔target contacts instead of a
   288×288 matrix on 456 tokens (§2.2). If AF3 jumps, the motif gap is largely a
   *specification* bug and Phase 2 becomes the priority.
2. **Motif benchmark on epoch-0645** (pre-adjacency-loss). Separates "the
   adjacency work broke motifs" from "inherited from the base model". If 0645 also
   trails RFD1, the adjacency loss is exonerated and Phase 1 will not fix it.

Decision gate: outcomes select between Phase 1 (loss/training defects) and Phase 2
(specification) as the next investment.

### Phase 1 — training-side defects + activate what is already implemented

All code exists and is flag-gated off. One short finetune from 0684:

- `use_cb: true` — score the atom the label is defined on
- `negative_weight: > 0` — the missing repulsive arm
- `detach_motif: true` — stop dragging the pinned target
- `mask_loop_pairs: true` — already the default; RFD1-faithful labels
- **reveal negatives in `_sample_sparse_contacts`** (not yet written) — the
  ADJ-only mode is the one closest to binder driving and contains no `0`s at all
- **`motif_region_mask_prob` 0.5 → 0.2** — stop deleting binder↔target adjacency
  in half the motif examples

Leave the six mode probabilities alone. Benchmark against 0684 on: chain breaks,
clashes, adjacency recovery, non-adjacency violation rate (new metric — see below),
and AF3.

**Caveat:** `mask_loop_pairs` and `use_cb` both change label/loss semantics, so
this phase is not comparable with the 0645→0684 lineage. Re-baseline 0684 under
the new metric set first.

### Phase 2 — explicit binder/motif specification (§11)

Only worth building after Phase 0 shows specification is the bottleneck.
Inference + CLI + **matching training mode**, in that order but shipped together —
an untrained inference capability reproduces the current 5OQV behaviour.

### Phase 3 — loop adjacency (§14)

Independent of Phases 1–2 and largely already built (`ss_adj_loop` dataset
exists). Own training phase because it shifts the label distribution.

### Phase 4 — ligand / nucleic-acid adjacency (§12)

Largest design surface, least certain payoff, and depends on decisions from
Phase 3 (block definition without secondary structure). Do last.

### New metric needed regardless

Adjacency recovery only measures `1` pairs. Add a **non-adjacency violation rate**:
fraction of explicit `0` pairs whose Cβ min distance falls below the contact
cutoff. Without it there is no number for the over-association problem, and Phase 1
cannot be evaluated.

## 14. Loop adjacency — full sketch

### 14.1 Why it is worth doing

- Loops are **~half of all residues**. Today they can carry no adjacency
  specification at all, so half the structure is unspecifiable.
- **Binder interfaces are frequently loop-mediated** (CDR loops being the extreme
  case). "This loop should contact that epitope" is the single most natural binder
  instruction and is currently inexpressible.
- It directly attacks the density-OOD problem (§2.1): more blocks per crop means
  more conditioned pairs available, raising the training-time ceiling of 27.

### 14.2 What already exists

`_adj_blocks_from_coords` has a complete loop branch, and a dataset config
`configs/datasets/train/ss_adj_loop.yaml` with `ss_adj_loop_adj: true`, its own
metadata path and crop settings. Phase 4/5 simply pass `datasets.train.loop=null`.
So this is **activation and refinement, not new construction**.

The existing rule, deliberately stricter than H/E:

```python
if is_loop1 or is_loop2:
    c1, c2 = ca_coords[b1], ca_coords[b2]          # C-ALPHA, not C-beta
    distances = ...
    if (distances < ca_threshold_loop).sum() < min_loop_contact_pairs:  # >= 3
        continue
```

H/E needs **one** Cβ pair under 6 Å; a loop pair needs **three** Cα pairs under
6 Å. That asymmetry is sound: a flexible extended loop brushing past an element is
not a packing interface, and demanding several contacts filters incidental
proximity.

### 14.3 Decisions required

1. **Atom: Cα or Cβ?** Currently Cα for loops, Cβ for H/E. The original rationale
   is defensible (loop side-chain orientation is less meaningful), but the loss and
   the metric are now both Cβ, so a Cα-defined label reintroduces exactly the
   atom mismatch we just removed. *Recommendation:* move loops to Cβ for
   consistency and keep `min_loop_contact_pairs` as the strictness knob.
2. **Long-loop sub-blocking.** A contiguous SS_LOOP run can be 25+ residues. As a
   single block it is not a useful unit — different parts contact different
   things, and one contact marks the whole loop adjacent. *Recommendation:* split
   loop runs into sub-blocks of ≤6–8 residues. H/E blocks need no such treatment
   because their length is naturally bounded.
3. **Termini.** N/C-terminal tails are often disordered and their "contacts" are
   coordinate noise. *Recommendation:* exclude terminal loop runs, or require
   resolved coordinates and a minimum length (≥3).
4. **Threshold.** Keep 6 Å for consistency; the ≥3-pair requirement already
   supplies the extra strictness.

### 14.4 Interactions and risks

- **Block-count growth.** Enabling loops roughly doubles block count and therefore
  ~quadruples candidate pairs. `_sample_sparse_contacts` scales automatically
  (`effective_cap = max(3, ceil(0.8 * N))`), so no code change is needed — but the
  conditioned-pair distribution shifts substantially and should be re-measured.
- **Repulsive arm.** With `loop_adj=true`, loop pairs become genuine `0`/`1`
  instead of masked, so the repulsive arm starts acting on them. The
  native-matched floor is what makes this safe — a loop natively packed at 4 Å
  gets a 3 Å floor, not an 8 Å one. **Do not enable loop adjacency together with a
  fixed-threshold repulsive term.**
- **`mask_loop_pairs` becomes moot** for loop pairs under `loop_adj=true`: they
  are properly labelled rather than skipped. The flag still matters for any
  dataset left at `loop_adj=false`.
- **Label distribution shift** is large (many more `1`s, many more blocks), so
  this needs its own phase and breaks comparability with everything before it.
- **Inference UX.** Loops are harder for a user to identify than helices, which
  argues for the chain/residue `adj_pairs` syntax (§11) over hand-authored
  matrices.

### 14.5 Staged rollout

1. Measure first: build labels with `loop_adj=true` on a sample and report block
   count, pair count, and the fraction of pairs that are loop-involved. If loop
   pairs swamp H/E pairs, sampling needs rebalancing before any training.
2. Implement sub-blocking + terminus exclusion; re-measure.
3. Decide Cα vs Cβ (recommend Cβ).
4. Train a phase with the `ss_adj_loop` dataset mixed in at modest probability
   (not as a replacement), `loop_adj=true`, negatives on.
5. Evaluate with a **loop-specific** recovery/violation breakdown — aggregate
   numbers will hide whether loop conditioning works, since H/E pairs dominate.

---

## 15. Fix plan: non-protein (and loop) adjacency

The insight that shapes this plan: **loops, ligands and nucleic acids are all the
same problem** — "how do I define a block and a contact for something that is not
an H/E element?". Fixing them as three separate special cases would mean three
divergent code paths in `_adj_blocks_from_coords`. Fixing them as one type-dispatch
table is a single coherent change that can then be enabled per-type by config.

### Stage A — stop the active harm ✅ DONE

`mask_loop_pairs=True` (§10). Loop-, ligand- and NA-involved pairs are now `2`
(zero contribution) instead of `0` (an active learned "non-adjacent" embedding,
measured at |out| = 30.5 vs 33.0 for a real contact).

This is defensive only: those pairs are now *ignorable*, not *specifiable*.

### Stage B — verify the data exists (cheap, no GPU)

Count, over the actual training manifest:
- examples containing a protein-ligand interface
- examples containing a protein-NA interface
- the distance distribution of those interfaces (to set cutoffs from data rather
  than convention)

**Gate:** `mean_lddt_dna` / `mean_lddt_rna` are exactly 1.0000 in every logged
phase-5 batch, which is what an absent modality looks like. If NA is effectively
absent from the training set, **do not build NA adjacency** — it cannot be
trained. Ligands are known present (`alpha_ligand: 2.0`).

### Stage C — generalize the label builder (the core work)

One change, in `foundry/src/foundry/data/ss_adj.py`.

**C1. Block-type assignment.** Replace the implicit "SS label == block type" with
an explicit per-token type derived from the atom array:
`HELIX | STRAND | LOOP | NA | LIGAND`. Non-protein currently inherits `SS_LOOP`
from `_get_ss_training`'s `np.full(I, _SS_LOOP)` default, which is simply false.

**C2. Block segmentation per type.**
| type | block |
|---|---|
| H / E | contiguous run (unchanged) |
| loop | contiguous run, **split at ≤6–8 residues**, termini excluded (§14.3) |
| ligand | one per entity — already falls out, since its tokens share label+chain |
| NA | windows of ~4 nt (v1); base-paired stems via `biotite.structure.base_pairs()` (v2) |

**C3. Contact rule as a pair-type table** (§12.3): Cβ<6 for protein↔protein,
heavy-atom 4.5 Å for anything involving a ligand, 5.0 Å for NA, with the
multiplicity requirement (`min_loop_contact_pairs`-style) per pair type.

**C4. Heavy-atom distances need atoms, not token representatives.**
`get_adj_from_token_array` already receives the full `atom_array`, so compute
label distances there. The *loss* keeps using token representatives — it only
needs a differentiable proxy that is monotone in the true distance, and for
atomized ligands the representative **is** the atom anyway.

**C5. Non-protein SS label → `MASKED`, not `LOOP`.** In
`ss_adj_transforms._get_ss_training`. No `SSAdapter` change: masked is already a
state, and the model separately receives `is_dna` / `is_rna` / `is_ligand`.

Everything above is label-side. **`ADJAdapter` and `SSAdapter` are unchanged**, and
`_contact_coords` already degrades correctly for non-protein (no CB, no exact
`N`/`CA`/`C` → falls through to token representative).

### Stage D — metrics, before any training

`analyze_ss_adj_conditioning.py` computes one aggregate recovery number over
protein Cβ. It cannot see any of this. Needs:
- recovery **broken out per pair type** (H/E, loop, ligand, NA) — an aggregate will
  be dominated by H/E and will hide whether ligand conditioning works at all
- the **non-adjacency violation rate** (§13)
- type-appropriate cutoffs matching C3

Without this, Stage E is unevaluable.

### Stage E — train, conditioning-only

Enable the new labels; **do not add non-protein pairs to `AdjacencyContactLoss`
yet**. Rationale: the well and the repulsive arm would start acting on ligand and
NA geometry, and ligand placement is currently good — risking it before recovery
is even measurable is a bad trade. Conditioning-only means the adapter learns from
the label and the existing structure losses keep geometry honest.

Enable per type via config so ligand can ship without waiting on NA.

### Stage F — supervise, only if warranted

If Stage D shows ligand/NA recovery is measurable but weak, add those pair types to
the loss with the type-aware cutoffs. Native-matched targets (§6.2) matter even
more here than for protein, since a fixed cutoff across pair types is meaningless.

### Risks

- **Label distribution shift is large.** Enabling loop + ligand + NA blocks
  multiplies block count and pair count several-fold. Re-measure the conditioned
  pair distribution before training; `_sample_sparse_contacts` scales
  automatically (`max(3, ceil(0.8*N))`) but the resulting density will be far from
  the current median of 5.
- **Not comparable** with the 0645→0684 lineage. Re-baseline.
- **Do not enable non-protein pairs together with a fixed-threshold repulsive
  arm.** Only the native-matched floor keeps a genuinely-contacting ligand from
  being pushed away.

### Suggested sequencing against the other work

B → C → D are independent of §11 (contig spec) and can proceed in parallel.
Stage C subsumes loop adjacency (§14), so **do not implement §14 separately** —
it is the `LOOP` row of the C2/C3 tables.
