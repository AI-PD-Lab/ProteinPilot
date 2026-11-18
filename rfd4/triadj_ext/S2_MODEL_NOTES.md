# TRIAD-R S2 model side (builder C2, 2026-10-22, CPU login node, fp32)

Files: `model_s2.py` (arm registry + `TriadRTransformer`), `generate_entry.py`, `ring_flow.py` (extended), `pair_stream.py` (small edits),
`tests/test_model_s2.py`, `tests/test_ring_flow.py` (extended). Upstream `src/rfproteina` untouched. All line numbers = fork `src/rfproteina/`.

## 1. Interface (for the trainer wiring)
```python
from triadj_ext.model_s2 import (register_arm, UNFREEZE_PATTERNS, unfreeze_patterns, unfreeze_new_params, install_lora_unfreeze,
                                 arm_nn_overrides, arm_hf_bundle_overrides, expand_bundle, param_counts, current_arm)
from triadj_ext.pair_stream import new_param_names            # [names of every new parameter], all start with ring_
register_arm("A"|"B"|"C"|"Z")   # or register_arm() = env TRIADR_ARM (default C); exports TRIADR_ARM so spawn workers inherit it
```
| arm | `P.LocalLatentsTransformer` | `P.ProductSpaceFlowMatcher` | trainable new params |
|---|---|---|---|
| A, B | `TriadJLocalLatentsTransformer` | `TriadJSamplerFlowMatcher` | `^nn\.ring_` (18,839) |
| C | `TriadRTransformer` | `TriadJSamplerFlowMatcher` | `^nn\.ring_` (64,220) |
| Z | `TriadJLocalLatentsTransformer` | `TriadJControlFlowMatcher` (stream all-MASK, ring losses x0) | none (`UNFREEZE_PATTERNS_BY_ARM['Z'] == []`) |

* `register_arm` is idempotent and does: SPECS (`ring_state`, `ring_state_gt`, `ring_obs`, `ring_val_mode`), class + flow-matcher rebinding in `rfproteina.proteina`,
  `install_lora_unfreeze()`, `os.environ['TRIADR_ARM']`. It does NOT call `ring_flow.register_optimizer_split()` (the trainer entry does; the default pattern `^nn\.ring_`
  already covers every arm because all new names start with `ring_`).
* `install_lora_unfreeze()` wraps `rfproteina.utils.lora.add_lora_adapter` (the source `train.py:54` binds at import, so `register_arm` must run BEFORE `import rfproteina.train`),
  `rfproteina.proteina.add_lora_adapter` and, if already imported, `rfproteina.train.add_lora_adapter`: after the original (which runs `model.requires_grad_(False)`,
  `utils/lora.py:19`) it re-enables `unfreeze_patterns(current arm)`. Verified on the real 178 M denoiser with r=4 (test_lora_toy_arm_C): trainable = 56 LoRA tensors (A and B of 2 modules x 14 layers) + 15 new (C) / + 0 (Z).
* Config block under `nn.ring` (merged by `train.py:_adopt_bundle_architecture` because `nn` is an adopted field; verified with the released bundle config, with and without
  `hf_bundle_overrides`, and from an expanded bundle that carries the block): `n_states 23, embed_rows 25` (all arms); arm C adds `head_blocks [4,8,12]`,
  `feedback_blocks [4,8]` (subset of heads), `feedback_detach true`. `arm_hf_bundle_overrides('C')` returns it as `{nn: {ring: {...}}}`.
* Loss config under `loss.ring.*` (read by `TriadJFlowMatcher._rc`, defaults in `DEFAULT_RING_CFG`): existing `loss_scale, class_w_nointer, w_clip, clock_mode, lag, sigma, p_indep` + new
  `final_weight 1.0`, `block_weight 0.5`, `block_weights {"4": w, ...}`. Logged per-sample [b] terms: `ring_prediction` (final), `ring_prediction_blk4/8/12`,
  `ring_ce` and `ring_ce_blk<i>` (mean CE over hidden pairs, `_justlog`); the objective weights are applied in `optimization_terms` (logs stay on one scale).
* `batch['ring_val_mode']` [b] int (0 tube, 1 all-MASK, 2 full reveal, 3 SHUFFLED full reveal = the label map of the nearest row in cyclic order (r-1, r-2, ...) that has observed pairs, a DIFFERENT source entry (4-char PDB id in `example_id_batch`, else the id string; only when ids exist) and a map that differs from the receiver's own on the overlap (so exploration replicates/duplicates are skipped); none in the micro-batch -> ValueError; needs batch >= 2; cropped/padded to the receiver, donor-unobserved pairs show 22, no ring loss on those rows; costs one scalar sync per corrupt_batch call (modes 0-2 output unchanged); per-row, validation only): `corrupt_batch` overrides the ring clock (0 / 1 / tube), writes the effective clock back into
  `batch['t']['ring']` (so MDLM weight and the val t-bins see it). SPECS entry `(int64, 0, "")` = same as `ring_data.register_ring_specs`; absent key = pure tube.
* `TRIADR_RING_SAMPLER=generated|masked|oracle` read at every `_arm` (`ring_flow.sampler_mode`); oracle raises unless every row has observed `ring_state_gt` (collate zero-fills 24 for absent keys: a
  silent all-22 stream would otherwise result). Control arm Z pins `masked`.
* Generated labels: `ring_flow.add_ring_sink(fn)`; `generate_entry` collects them per RUN into `<out_dir>/ring_states_job<J>[_rank<R>].npz` (`load_ring_states(path)`).
* `python -m triadj_ext.model_s2 expand --arm C --src /w/rfd4_weights --dst <dir> [--seed 0]` writes the expanded bundle (verified by `--heavy` test: config differs only in `nn.ring` + `weights_sha256`,
  all released tensors bit-equal, ring_embed/ring_fb zero, seeded `ring_head_*` equal between arms A and C). Arms A, B, Z share one bundle; C needs its own (extra keys).

## 2. Facts (measured on CPU, released weights, 5vht crop n=96, R=2 unless noted)
* Parameters: released nn 178,448,304. A +18,839 (`ring_embed` 9,600, `ring_head_norm` 384, `ring_head_proj` 8,855). C = A + 45,381 = 64,220 (`ring_blk.{4,8,12}`: 3 x 9,239; `ring_fb.{4,8}`: 2 x 8,832).
  LoRA r=32 on `[to_qkv, to_out_node]`: 14 x 147,456 = 2,064,384. Trainable per arm: A 18,839; B 2,083,223; C 2,128,604; Z 2,064,384 (+18,839 frozen).
* The 15 new keys of arm C: `ring_embed.weight, ring_head_norm.weight, ring_head_proj.{weight,bias}, ring_blk.{4,8,12}.0.weight, ring_blk.{4,8,12}.1.{weight,bias}, ring_fb.{4,8}.weight`.
* Latency (login node, load ~8, 6 threads, best of 3): eval forward base 2.11 s | A 2.10 | C 2.06; train fwd+bwd (R=2) 3.81 | 3.88 | 3.82 s. `FlopCounterMode` forward 566.8 GFLOP base vs 568.6 (+0.3 %).
  CPU cannot show the GPU cost: the added work is memory-bound elementwise ops on `[b,n,n,384]` (norm in, norm out, `z + ...`; ~3 sites) between compiled blocks; the +~10 % of the spec is unverified.
* Step-0 parity (fp32 CPU, assign=True loading): arms A and C bit-identical to the released model (max|diff| exactly 0.0 on bb_ca.v, local_latents.v, sequence.logits, insertion.logits) for
  eval/train x n_recycle 1/3 x ring_state absent/present (random labels + 50 % MASK); identical-input rerun floor 0.0. Also exact under CPU bf16 autocast (eval, R=2, ring_state present) and for the `feedback_blocks=[4,8,12]` variant.
* Alignment effect (new, matters for every parity claim): CPU fp32 outputs depend on the parameter memory alignment. A copy-loaded model (`load_state_dict` without `assign=True`, or `deepcopy`) differs from the
  safetensors-assigned released model by 1.9e-5 max-abs for the STOCK class too (assign-load: 0.0; the safetensors tensors are not 64-byte aligned). A bundle round trip (expanded bundle written then loaded) differs by 2.0e-5.
  So "max diff <= 3x identical-rerun floor (0.0)" only holds for same-storage models; across load routes use ~1e-4 (or rel-L2 1e-6). On GPU the parameters are copied to device memory (aligned): expected exact, unverified.
* Gradients (train mode, R=3, recycle.grad=last): with the full loss (structure + CE of final and 3 block heads) every one of the 15 new params has non-zero gradient; with the structure loss only at zero init
  `ring_fb.4/8` get gradient and the heads get none (detach=True); with non-zero feedback the heads get the structure gradient iff `feedback_detach=false`.
  Earlier no-grad recycle passes still run heads + feedback (call sequence 6 x grad-off then 3 x grad-on for R=3); the stash is the last pass (has a graph).
* Checkpointing: `nn.checkpointing` none / low / medium all run on CPU (non-reentrant `checkpoint` works here). Grad and logits equal to rel 1e-4 for 8 probe params + the three block logits. The recompute in backward
  must not re-stash (guard `_blk_open`) and re-reads `self._ms` (kept after forward on purpose) => backward before the next forward through the module (true for `Proteina.training_step`).
* torch.compile: confirmed from code (`_setup_compiled_trunk` compiles `transformer_layers[i]` / `pair_update_layers[i]` individually; `_apply_trunk` and `_trunk_step` are plain eager python), asserted by
  `test_apply_trunk_wrapper_relies_on_upstream_shape`. The hook wraps entries of the `pair_layers` list handed to the unmodified upstream loop, so no copy of the loop; the z handed to the next compiled block keeps
  shape/dtype/contiguity (`z + (...).to(z.dtype)`). Not exercised on CPU: compiled blocks, cudagraph reduce-overhead at inference, real GPU bf16-mixed, cuEquivariance tri-mult.
* One real forward/loss/backward through the registry-built `Proteina` (arm C, K=2 replicates, synthetic CA<8 A labels): per-block terms present, [K] shaped, winner gather works, weights 1.0/0.5 in the objective.
* 20-step sampler (arm C at zero init, real inference batch L=100): `generated` reveals everything (final labels all in 0..22), `masked` never reveals; sink + npz round trip OK.

## 3. Where the spec (S2_harness_spec.md) was wrong or stale
1. Names/patterns: spec `rfb` + `^nn\.(ring_|rfb_)` does not match `nn.rfb.4.weight`. Modules are `ring_blk` / `ring_fb`, so `^nn\.ring_` (also the existing Adam-split pattern) covers all.
2. Sec. 1.3 "BUG" (replay rows reveal random 22s): already fixed in the committed `ring_flow.py` (`has_ring`, N004 commit 6319524) with `test_unlabelled_rows_are_all_mask`.
3. Sec. 4: the head at block 12 reads exactly the z that the final head reads (no feedback after 12), so `ring_blk.12` duplicates `ring_head_*` (9,239 params, same target). Kept as specified; `feedback_blocks=[4,8,12]` is
   available and tested: block-12 feedback is the only way to write the soft label into the z that recycling reads (`z_cycle_proj`) and attention layer 13 ("feeds one attention layer only" overlooks recycling).
4. Sec. 6.1 "CPU fp32 floor is 0.0": only for identical parameter storage (see alignment effect above).
5. Minor: `checkpoint(trunk...)` is `:409`, the no-grad detach `:413` (not 412), `train.py` adds the adapter at `:314`.
6. LoRA rebinding: binding `rfproteina.train.add_lora_adapter` alone is not enough in processes where `train` was imported first; the source `rfproteina.utils.lora.add_lora_adapter` must be wrapped before `train` imports it (done).

## 4. What only the GPU smoke can verify
Throughput/peak memory of the three heads + two feedbacks at `[8,256,256,384]` bf16 (the +10 % estimate); bf16-mixed step-0 parity (exact-zero add should be exact); `use_torch_compile` with the eager hook
(no recompiles, no graph breaks inside blocks) and inference cudagraph mode `reduce-overhead` with `compile_lengths`; `nn.checkpointing: medium` under CUDA autocast; 2-GPU DDP without unused-parameter errors
(arm C: all 15 params are in the graph of every batch; arm Z has them frozen); Muon/Adam grouping of `ring_blk/ring_fb`; LoRA + expanded-bundle strict load via `train.py`; cuEquivariance backend; generate through
`generate_entry` with a real `chk_*-ema.ckpt` (`hf_ckpt_path=null ckpt_path=... ckpt_name=...`), including multi-GPU rank files.

## 5. Arm <-> checkpoint agreement at generation (Codex round 4 #7)
`generate_entry` calls `model_s2.register_for_generation(cfg)` (inside the `run_generation` wrapper, before any model is built): `source_arm_info` reads the checkpoint hparams with mmap (no tensors) or the bundle `config.json`;
`resolve_arm(TRIADR_ARM, info, TRIADR_EXPECT_ARM)` enforces: `nn.ring.head_blocks` <=> C, no `nn.ring` <=> `none`; the trainer's label (`hyper_parameters.cfg_exp.arm`, written by C1's config for every arm: A/B/C smoke ckpts checked; fallback: `run_name_` suffix)
must equal the arm; unlabelled A/B/Z sources need an explicit arm and log `[triadr] WARNING ARM not verifiable`; Z is never inferred (unset arm + Z label raises); `TRIADR_EXPECT_ARM` (to be set by the eval sbatch from the run it scores) must equal the resolved arm.
C checkpoint + `TRIADR_ARM=A` raises (verified on the real C smoke ckpt: "has nn.ring.head_blocks (arm C): the arm does not match the source").

## 6. Sparse maps (cond_gen, 2026-10-22)
Sampler mode `TRIADR_RING_SAMPLER=pinned` (ring_flow): only the pairs that carry a code < 23 in `batch['ring_state_gt']` are shown (labels 0..21 and 22), from step 0 to the end; every other valid pair stays MASK (never revealed,
never generated, never NO_INTERACTION). `cond_gen --conditions ...,sparse` uses it for `oracle_frac{5,10,25,50}` (nested random k % of the true positive pairs), `hotspot{3,8,20}` (longest-range true contacts),
`hotspot{3,8,20}_neg` (+ 5n random native non-contacts, CA-CA > 12 A, as NO_INTERACTION) and, with the oracle display, `edited` (true map minus its strongest long-range contact). A positives-only sparse reveal is outside the training
distribution (uniform random reveal over all pairs, ~95 % NO_INTERACTION): read a null result as "this reveal pattern is not usable", not as "the stream cannot be steered".


## 7. Stream CFG (TRIADR_STREAM_CFG=w, ring_flow + cond_gen, 2026-10-22)
At every sampler step the denoiser runs twice on the SAME noisy inputs: with the current stream (pinned / oracle / generated) and with the stream ALL-MASK; every predicted component that is integrated is combined as
`pred = pred_masked + w (pred_cond - pred_masked)` (`bb_ca` and `local_latents` velocity -> `v_guided`; sequence logits and, since Codex round 7, the clean prediction `x_1` of every guided mode likewise, so exported clean trajectories / metric guidance agree with the integrated velocity).
The second pass replays the torch RNG state of the first (random feature draws such as the `coord_group_seq` randperm are identical, only the stream differs) and the RNG is restored to the state ONE forward leaves behind, so later integration noise is identical across w. w must be finite and >= 0 (`TRIADR_STREAM_CFG`, `--cfg-w`); design_with_contacts requires > 0. w = 1 (default, unset) is the old single-pass path bit for bit; a stream that is already all-MASK
(mode masked) runs ONE pass (cond == masked, so any w is identical). Cost: 2x sampler time for w != 1. Upstream guidance (`guidance_w` != 1, its own `ucond` mode) is mutually exclusive and raises. `cond_gen`: env CFG_W="1,1.5,2,3"
(+ CFG_CONDS to restrict) adds `<cond>@cfg<w>` conditions with the same shared noise (cond_gen.sbatch header has the cost). The identity / linearity tests are in tests/test_ring_flow.py and tests/test_cond_gen.py.

## 8. design_with_contacts (user tool, 2026-10-22)
`s2/design_with_contacts.py` + `.sbatch` (1 H100): JSON spec -> pinned (only the requested pairs shown at every step) or hybrid (pinned labels re-written after every generated reveal; every other pair generated on the
`generated` schedule) sampler -> VanillaStrategy decode (all-atom + co-generated sequence, same path as generate_entry) -> ONE tar.gz (PDBs, FASTA, map.json, report.json, spec.json). Class names map to the intra-chain
states 0..10 through `RING_INTERACTION_TYPES` (the table gate_c uses; `any` = vdw state 7); negatives = state 22. Classes: hbond salt_bridge hydrophobic pi_stack pi_cation pi_hbond disulfide vdw any (`ionic`: RING IONIC edges come out of gate_c as salt_bridge, state 2 is never measurable; `metal` / `halogen`: no such atoms in the designed chain -> refused). `spec.json` in the tar is re-loadable. Exit codes: 0 ok, 2 bad spec/arguments/environment (arm Z, an inherited TRIADR_RING_SAMPLER), 3 RING failed for some design (tar kept; every `ring_*` rate counts ALL designs, unmeasured = unsatisfied, `*_measured_only` separately, n_requested / n_measured per contact, class and design), 4 pin verification failed; a failed run's tar is saved as OUT_TAR.FAILED. `report.json` carries, per contact, CA-CA <= 8 A, min heavy-atom distance <= 4.5 A, a class-specific geometric proxy,
and (host `ring-check` step, same RING 4 code as gate_c) the measured RING state; `pins_verified` counts the pinned pairs found in the denoiser input at every step. CPU: a 20-step sampler at the step-2000 smoke checkpoint gives
the plumbing, not satisfaction (contacts_ca_ok ~0-0.33). Only a GPU run with a trained checkpoint and the full 60-step sampler tells whether the requested contacts are actually obtained.

## 9. Two RING windows in design_with_contacts (2026-10-23)
`ring-check` measures every design at the binary's DEFAULT vdw window (strict: `ring_state` / `ring_ok` / `ring_none_ok`, unchanged semantics) and at `-w 0.25` (tolerant, the HEADLINE class satisfaction:
`ring_state_w0.25` / `ring_ok_w0.25` / `ring_none_ok_w0.25`, plus per-contact / per-class / per-design aggregates with the same suffix). Windows: `--ring-w` > env `DWC_RING_W` > spec `ring_w` > `none,0.25`; the strict window must be in the list;
`ring_check()` itself defaults to strict only (dwc_class_fidelity's call is unchanged). Any RING failure at any window = design unmeasured at every window (exit 3). `report.ring` records binary, version, flags, `-w` per window, the headline and why:
the default window is razor thin (AE re-decode of natives retains 45 % / 42 % vdw / hydrophobic at the default window, 89 % / 100 % at -w 0.25; designs realise vdw 0.22 -> 0.69, hydrophobic 0.26 -> 0.71).
