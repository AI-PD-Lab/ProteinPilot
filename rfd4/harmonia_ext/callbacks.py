"""HARMONIA training log line (stdout; compute nodes have no internet), every ``every_n`` optimizer steps on rank 0::

    H4 step=<int> sec_step= rows_s= peak_gib= loss= bb_adj= bb_free= lat_adj= lat_free= n_adj= n_free= gn_adj= gn_ss= gn_lora= lr=

bb_* / lat_* = mean per-row CA / latent flow loss over rows WITH revealed adjacency (``adj``: any adj_state in {0,1,3,4}) and rows without
(``free``: replay rows and the uncond / ss_only curriculum modes). These are DIFFERENT rows with different task mixes, so bb_adj vs bb_free
is a monitor, NOT a test of ADJ -> structure (red-team N001 #5); that test is generation on the natural panel (generate.py + RFD3 scorer).
gn_adj / gn_ss / gn_lora = gradient L2 norms (before clipping) of adj_embed, the SS embedding and the LoRA matrices (gn_lora is NaN in the
full fine-tune). Validation: ``H4 val <metric>=<value> ...`` from trainer.callback_metrics.
"""
from __future__ import annotations

import re
import time

import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback

REVEALED = (0, 1, 3, 4)


class HarmoniaLogCallback(Callback):
    NAMES = ("n_adj", "n_free", "bb_adj", "bb_free", "lat_adj", "lat_free", "loss", "loss_n")

    def __init__(self, every_n: int = 10, adj_pattern: str = r"^nn\.adj_embed\.", ss_pattern: str = r"ss_embedding", lora_pattern: str = r"lora_[AB]"):
        super().__init__()
        self.every_n = int(every_n)
        self.res = {"gn_adj": re.compile(adj_pattern), "gn_ss": re.compile(ss_pattern), "gn_lora": re.compile(lora_pattern)}
        self._acc = None
        self._t0 = None
        self._rows = 0
        self._steps = 0
        self._groups = None
        self._gn = {}
        self._want_gn = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if self._t0 is None:
            self._t0 = time.perf_counter()
        accum = int(pl_module.cfg_exp.opt.get("accumulate_grad_batches", 1))
        self._want_gn = (batch_idx + 1) % accum == 0 and (trainer.global_step + 1) % self.every_n == 0

    def on_after_backward(self, trainer, pl_module) -> None:
        if not self._want_gn:
            return
        if self._groups is None:
            ps = [(n, p) for n, p in pl_module.named_parameters() if p.requires_grad]
            self._groups = {k: [p for n, p in ps if r.search(n)] for k, r in self.res.items()}
        for k, ps in self._groups.items():
            g = [p.grad.detach().float().norm() for p in ps if p.grad is not None]
            self._gn[k] = float(torch.linalg.vector_norm(torch.stack(g))) if g else float("nan")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not isinstance(outputs, dict) or "losses_per_sample" not in outputs:
            return
        k = int(outputs.get("exploration_k", 1))
        loss = outputs["losses_per_sample"]
        dev = batch["padding_mask"].device
        a = batch.get("adj_state")
        if a is None:
            has = torch.zeros(batch["padding_mask"].shape[0], dtype=torch.bool, device=dev)
        else:
            has = torch.isin(a.long(), torch.tensor(REVEALED, device=a.device)).flatten(1).any(1)
        has = has[::k].to(dev)
        nb = int(has.shape[0])
        free = ~has

        def msum(key, m):
            x = loss.get(key)
            return (x.double() * m).sum() if x is not None and x.shape[0] == nb else torch.zeros((), dtype=torch.float64, device=dev)

        tot = sum((loss[t].double() for t in ("bb_ca", "local_latents", "sequence_prediction") if t in loss and loss[t].shape[0] == nb),
                  torch.zeros(nb, dtype=torch.float64, device=dev))
        v = torch.stack([has.sum().double(), free.sum().double(), msum("bb_ca", has), msum("bb_ca", free), msum("local_latents", has),
                         msum("local_latents", free), tot.sum(), torch.tensor(float(nb), dtype=torch.float64, device=dev)])
        self._acc = v if self._acc is None else self._acc + v
        self._rows += nb
        accum = int(pl_module.cfg_exp.opt.get("accumulate_grad_batches", 1))
        if (batch_idx + 1) % accum:
            return
        self._steps += 1
        step = trainer.global_step
        if step == 0 or step % self.every_n:
            return
        acc = self._acc
        rows = torch.tensor(float(self._rows), dtype=torch.float64, device=dev)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(acc)
            dist.all_reduce(rows)
        s = dict(zip(self.NAMES, acc.tolist()))
        dt = time.perf_counter() - self._t0
        div = lambda x, n: x / n if n > 0 else float("nan")
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else float("nan")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            lr = float(trainer.optimizers[-1].param_groups[0]["lr"])
        except Exception:  # noqa: BLE001
            lr = float("nan")
        f = [("step", step), ("sec_step", div(dt, self._steps)), ("rows_s", div(float(rows), dt)), ("peak_gib", peak),
             ("loss", div(s["loss"], s["loss_n"])), ("bb_adj", div(s["bb_adj"], s["n_adj"])), ("bb_free", div(s["bb_free"], s["n_free"])),
             ("lat_adj", div(s["lat_adj"], s["n_adj"])), ("lat_free", div(s["lat_free"], s["n_free"])), ("n_adj", int(s["n_adj"])),
             ("n_free", int(s["n_free"])), ("gn_adj", self._gn.get("gn_adj", float("nan"))), ("gn_ss", self._gn.get("gn_ss", float("nan"))), ("gn_lora", self._gn.get("gn_lora", float("nan"))), ("lr", lr)]
        if trainer.is_global_zero:
            print("H4 " + " ".join(f"{k}={v}" if isinstance(v, int) else f"{k}={v:.6g}" for k, v in f), flush=True)
        self._acc, self._rows, self._steps, self._t0, self._gn = None, 0, 0, time.perf_counter(), {}

    def on_validation_start(self, trainer, pl_module) -> None:
        self._t0 = None                                        # validation time is not step time

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        m = {k: float(v) for k, v in trainer.callback_metrics.items() if "val" in k and torch.is_tensor(v) and v.numel() == 1}
        if m:
            print(f"H4 val set=all step={trainer.global_step} " + " ".join(f"{k}={v:.6g}" for k, v in sorted(m.items())), flush=True)
