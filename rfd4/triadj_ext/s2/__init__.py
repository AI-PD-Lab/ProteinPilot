"""TRIAD-R S2 fine-tuning wiring (trainer side). Upstream ``src/rfproteina`` is untouched; everything here is configs + module-global rebinding.

Layout: ``configs/experiment/triadr_s2.yaml`` (+ ``configs/arm/{A,B,C,Z}.yaml``), ``glue.py`` (entry glue), ``optim_glue.py`` (LoRA unfreeze,
two-group Adam), ``callbacks.py`` (epoch / log / EMA), ``make_base_bundle.py`` (expanded base bundle), ``train_s2.sbatch``, ``wandb_sync_s2*``.
Model side (arms, flow matcher, new modules): ``triadj_ext/model_s2.py``, ``ring_flow.py``, ``pair_stream.py``. Data: ``ring_data.py``.
Tests (CPU, cheap logic + config + one real batch; no training): ``triadj_ext/tests/test_s2_{glue,config,sbatch}.py``, ``test_ring_data.py``.

Design in one paragraph. Two train leaves of ONE class (``RingDataset``) over the same PDB manifest rows (883 k monomer+interface rows, train split minus
every sequence cluster of the val manifest): ``ring_pdb`` (p = 0.45; RING labels, unconditional task, all conditioning off) and ``replay_pdb`` (p = 0.55; no
labels so the stream is all-MASK and no ring loss, task mix = preset ``unconditional_pretraining``: 90 % unconditional / 5 % folding / 5 % inverse folding, the
release's global conditioning probabilities and 20 % centroid conditioning). Replay preset decision: the S2 gate (a) measures UNCONDITIONAL de-novo monomer
designability, so 90 % of the replay rows train exactly that task; the 5 % + 5 % tail keeps the sequence-conditioned tasks (folding, inverse folding) alive so the
LoRA / new modules cannot drift on paths the gate never looks at. Rejected: ``unconditional`` (no folding / inverse-folding replay at all), ``motif_pretraining``
(80 % unindexed-motif scaffolding: unrelated to the gate and its tokens/losses would dominate the replay gradient), ``ppi_motif_scaffold_with_tip_atom`` (the release's
mix for PDB interfaces: 80 % of it needs ligands / a second chain, mostly absent in monomer rows), ``all`` / the stock 14-leaf mix (needs streaming data absent here).
Loss-only validation on 256 held-out rows (val manifest, disjoint sequence clusters): ``val_ring_masked`` (all-MASK stream), ``val_ring_full`` (full reveal), same rows
and crops, plus ``val_replay``; lines ``S2 val set=...`` in the log (EMA weights) with requested / evaluated / failed row counts (strict rows: a row that cannot be
built is dropped and counted, never substituted). Judge the interaction -> structure coupling on the STRUCTURE terms only: ``S2 val set=derived d_bb_ca d_local_latents
d_structure`` = masked - full (the total loss is confounded: full reveal removes the hidden-pair ring loss, so total(masked) > total(full) with identical structure).

Container prefix used below (``$APPT``); the PDB mirror / RING lmdb come from ``/w`` and ``/mir``::

  APPT='apptainer exec --nv --bind /path/to/work:/w --bind /path/to/work_rfd3:/mir:ro
        --bind /path/to/home/Training_Triad/RFD4_J:/fork --pwd /fork /path/to/work/rfd4-cu13.sif
        env CLUSTER=local DATA_PATH=/w HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/fork/src:/fork:/w/pylibs_lmdb'

(i) Expanded base bundle (login node, CPU, ~1 min, two files of ~1.7 GB; A, B and Z share family A, C has its own; written ONLY on request)::

  $APPT python -m triadj_ext.s2.make_base_bundle --arm A --dst /w/triadr_s2_base_A      # arms A, B, Z (4 new keys, 18,839 parameters)
  $APPT python -m triadj_ext.s2.make_base_bundle --arm C --dst /w/triadr_s2_base_C      # arm C (adds ring_blk.* / ring_fb.*, 64,220 parameters)
  emits <dst>/config.json + <dst>/model.safetensors and ONE stdout line ``BUNDLE {...}``: new keys, parameter counts, checks (released tensors bit-equal, only the
  expected keys added, strict reload with the registered class, zero-init tensors zero); an assertion fails otherwise. ``--seed`` fixes the random heads.

(ii) Smokes (~15-25 min each; STEPS <= 300 selects the tag automatically: ``smoke`` for 1 GPU, ``smoke<N>g`` for N > 1, i.e. an own $DEST per arm x GPU count, no
chain). The DONE marker records "steps=<n> gpus=<g>": the same steps again is skipped, a LARGER steps RESUMES from the newest $DEST/checkpoints/last*.ckpt, ``FORCE=1``
clears DONE/ABORT, ``RERUN=<suffix>`` starts a fresh $DEST (``<ARM>_<tag>_<suffix>``). A checkpoint that cannot be restored aborts (rc 4, ABORT marker), a home
file count above 538000 aborts the chain (rc 5)::

  S=/path/to/home/Training_Triad/RFD4_J/triadj_ext/s2/train_s2.sbatch
  ARM=B GPUS=1 STEPS=100 ACCUM=2 EXTRA="opt.val_every_n_epochs=1" sbatch --gres=gpu:h100:1 --ntasks-per-node=1 --time=01:00:00 $S     # 1-GPU smoke
  ARM=B GPUS=1 STEPS=200 ACCUM=2 EXTRA="opt.val_every_n_epochs=1" sbatch --gres=gpu:h100:1 --ntasks-per-node=1 --time=01:00:00 $S     # RESUME to 200 (same dest)
  ARM=C GPUS=2 STEPS=100 ACCUM=2 EXTRA="opt.val_every_n_epochs=1" sbatch --gres=gpu:h100:2 --ntasks-per-node=2 --time=01:00:00 $S     # 2-GPU DDP smoke (dest C_smoke2g)
  emits to /path/to/home/triad3_work/tj_s2_<job>.out: ``S2 job ...`` header (dest=..), ``S2 home files: ...``, ``S2 resume from ...`` (resume only), ``S2 optim:
  Adam groups ...``, ``S2 lora: re-enabled ...``, ``S2 epoch=<e> train set_epoch on RingDatasets [...]``, ``S2 step=10 rows_s= sec_step= peak_gib= data_wait= ce_ring=
  bb_ca_ring= bb_ca_replay= lat_ring= lat_replay= gn_new= gn_lora= loss= ...`` every 10 steps (``loss`` = the optimised objective), the validation sanity pass and
  (val_every_n_epochs=1) ``S2 val set=val_ring_masked|val_ring_full|val_replay step=100 requested= evaluated= failed= ...`` + ``S2 val set=derived ... d_bb_ca d_local_latents
  d_structure`` (masked - full, structure terms only), ``S2 DONE``, ``S2 exit rc=0``. Checkpoint at epoch 1: node-local /tmp/$USER/tr_s2_<ARM>_smoke_<job>/checkpoints/last.ckpt,
  mirrored to $WD/triadr_s2/<ARM>_smoke/checkpoints.
  Read: rows_s -> GPU-h per row (row cost); peak_gib -> largest BATCH with peak < ~65 (re-run); data_wait < 0.2 else NW up; gn_new and gn_lora finite and nonzero;
  ce_ring starts near ln 23 = 3.1; export S2_SMOKE_SEC_STEP=<sec_step at 64 rows/step> for the 4-GPU watchdog (it uses the median of the last three windows).

(iii) 4-GPU arm run (6 h jobs, successor submitted at the start of each job, auto-resume from the newest last*.ckpt, watchdog scoped to the job id)::

  ARM=A STEPS=2000 sbatch triadj_ext/s2/train_s2.sbatch                  # 4 x H100, 32 cpus per task, batch 8 x accumulate 2 x 4 GPUs = 64 rows per step
  ARM=B STEPS=4000 | ARM=C STEPS=6000 | ARM=Z STEPS=4000                  # B, C, Z: state the GPU-h estimate and get the >25 GPU-h approval BEFORE submitting
  emits the same lines; chk_<epoch>_<step>.ckpt + chk_*-ema.ckpt every 1000 steps (offline designability: ``python -m triadj_ext.generate_entry`` per chk_*-ema.ckpt);
  ``S2 DONE`` + $WD/triadr_s2/<ARM>/DONE at the end. Metrics: put every job id of an arm in ``triad3_work/tj_s2_arms.json`` ({"<jobid>": "<run name>"}) and run
  ``wandb_sync_s2_cron.sh`` from a login-node cron (not added by the builder); ``wandb_sync_s2.py --dry-run`` shows what it would send.
"""
