"""HARMONIA-RFD4: login-node replay of the harm4 SLURM logs to wandb (adapted copy of triadj_ext/s2/wandb_sync_s2.py; project HARMONIA)
Original doc (names adapted below: H4 lines, harm4_<J>.out, harm4_runs.json, project HARMONIA, run id r4-<name>):
Login-node replay of TRIAD-R S2 SLURM logs to wandb (compute nodes have no internet; same pattern as ``triadj_ext/wandb_sync_p0.py``).

Arms file ``<root>/tj_s2_arms.json`` = {"<jobid>": "<run name>"}; only listed jobs are synced. Jobs that share a run name (the chained jobs of one arm:
add every job id of the chain, the sbatch prints them) are ONE wandb run ``s2-<run name>`` (project TRIAD-R, entity Protein-Design-Tools, resume=allow): their
lines are replayed in job-id order and every step is logged once (wandb steps must increase; after a resume the lines between the last checkpoint and the
crash exist twice, the first copy wins). The log of job J is ``<root>/tj_s2_<J>.out`` (the sbatch ``--output``).

Lines understood (all other log lines are ignored):
  ``S2 step=<n> key=value ...``                    -> wandb ``s2/<key>`` with x axis ``s2/step`` = n (NaN dropped)
  ``S2 val set=<name> step=<n> metric=value ...``  -> wandb ``val/<name>/<metric>`` with x axis ``val/step`` = n (EMA weights; the counts requested /
                                                      evaluated / failed and the derived masked-minus-full structure terms are metrics too)
  Both streams own their step axis (``define_metric``, no explicit ``step=``): a validation record for step N arriving in a LATER cron invocation than
  train step N is kept (wandb drops explicit past steps after a resume).
  ``S2 job id=.. arm=.. gpus=.. ...``              -> run config (first header seen)
  ``S2 exit rc=<n> after <s>s`` / ``S2 DONE``      -> END MARKER of the job: the run summary (last values, rc, done) is written only after it
                                                      (a job that is still running is replayed incrementally and never finalised)

Run: /path/to/miniforge3/envs/triad/bin/python wandb_sync_s2.py [--dry-run]   (login node, cron every 10 min via wandb_sync_s2_cron.sh, flock'ed)
``S2_SYNC_ROOT`` overrides the root dir (default /path/to/home/triad3_work), ``WANDB_DIR`` the wandb dir (default on /hnvme: inode quota).
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

ROOT = os.environ.get("H4_SYNC_ROOT", "/path/to/home/logs")
KV = re.compile(r"(\w+)=([^\s]+)")
KV_ANY = re.compile(r"([^\s=]+)=([^\s]+)")


def num(v: str):
    try:
        x = float(v)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def parse(path: str) -> dict:
    """-> {steps: [dict], vals: [dict(set, step, metrics)], header: dict, end: bool, rc: int|None, done: bool}"""
    steps, vals, header, end, rc, done = [], [], {}, False, None, False
    for line in open(path, errors="replace"):
        if line.startswith("H4 step="):
            d = {k: num(v) for k, v in KV.findall(line)}
            if d.get("step") is not None:
                steps.append({k: v for k, v in d.items() if v is not None})
        elif line.startswith("H4 val "):
            raw = dict(KV_ANY.findall(line))
            name, step = raw.pop("set", None), num(raw.pop("step", "nan"))
            if name and step is not None:
                vals.append({"set": name, "step": int(step), "metrics": {k: num(v) for k, v in raw.items() if num(v) is not None}})
        elif line.startswith("S2 job ") and not header:
            header = dict(KV.findall(line))
        elif line.startswith("S2 exit "):
            end = True
            m = re.search(r"rc=(-?\d+)", line)
            rc = int(m.group(1)) if m else None
        elif line.startswith("S2 DONE"):
            done = True
    return {"steps": steps, "vals": vals, "header": header, "end": end or done, "rc": rc, "done": done}


def plan(parsed_jobs: list[tuple[str, dict]], last_step: int = -1, val_done: dict | None = None) -> dict:
    """Merge the parsed logs of the jobs of ONE run (in job-id order) into the events that are new. Train events: strictly increasing steps, first
    copy wins (``last_step`` = train cursor). Val events have their OWN cursor ``val_done`` = {set: last step sent}: a validation line for step N is
    new whenever N is above the cursor of ITS set, also when train step N (or later) was already synced by an earlier invocation (the
    cron tick may read the log between the end of train step N and the validation printout, or between two val lines of one pass).
    ``val_done=None`` (legacy state) keeps the old rule: drop val steps below the train cursor.
    -> {events: [(step, kind, payload)], last_step, val_done, finished_jobs, summary}"""
    events, last = [], last_step
    vd = dict(val_done) if val_done is not None else None
    summary: dict = {}
    finished = []
    for job, p in parsed_jobs:
        stream = [(s["step"], 0, s) for s in p["steps"]] + [(v["step"], 1, v) for v in p["vals"]]
        # within a job the file is chronological = step order, and the val lines of step n come after the train line of step n: (step, kind), stable
        for step, kind, payload in sorted(stream, key=lambda x: (x[0], x[1])):
            step = int(step)
            if kind == 0:
                if step <= last:
                    continue
                last = step
            elif vd is None:
                if step < last:
                    continue
            else:
                if step <= vd.get(payload["set"], -1):
                    continue
                vd[payload["set"]] = step
            events.append((step, "train" if kind == 0 else "val", payload))
        if p["end"]:
            finished.append(job)
            if p["steps"]:
                summary.update({f"last/{k}": v for k, v in p["steps"][-1].items()})
            summary.update({"rc": p["rc"] if p["rc"] is not None else 0, "done": p["done"], "last_job": job})
    return {"events": events, "last_step": last, "val_done": vd, "finished_jobs": finished, "summary": summary}


def load_state(path: str) -> tuple[int, dict | None]:
    """state file -> (train cursor, val cursors); an old state file holds just the train cursor (val cursors None = legacy rule)."""
    if not os.path.exists(path):
        return -1, {}
    txt = open(path).read().strip()
    try:
        d = json.loads(txt)
    except ValueError:
        return -1, {}
    return (int(d["train"]), {k: int(v) for k, v in d["val"].items()}) if isinstance(d, dict) else (int(d), None)


def define_axes(run) -> None:
    """Train and val metrics live on their OWN step axes (``s2/step``, ``val/step``) instead of wandb's global step. A resumed run starts at the step
    after its finalised history, so a validation record for step N that reaches the cron AFTER train step N was already synced (a later cron
    invocation) would be dropped as a past step; with metric-owned axes there is no monotonicity constraint between the two streams."""
    run.define_metric("s2/step")
    run.define_metric("s2/*", step_metric="s2/step")
    run.define_metric("val/step")
    run.define_metric("val/*", step_metric="val/step")


def emit(run, events) -> int:
    """Send planned events: no explicit ``step=`` argument anywhere, the x value is the metric ``s2/step`` / ``val/step``. Returns the count."""
    n = 0
    for step, kind, payload in events:
        if kind == "train":
            run.log({"s2/step": step, **{f"s2/{k}": v for k, v in payload.items() if k != "step"}})
        else:
            run.log({"val/step": step, **{f"val/{payload['set']}/{k}": v for k, v in payload["metrics"].items()}})
        n += 1
    return n


def runs_of(arms: dict) -> dict[str, list[str]]:
    by: dict[str, list[str]] = {}
    for job, name in arms.items():
        by.setdefault(str(name if isinstance(name, str) else name.get("name")), []).append(str(job))
    return {n: sorted(js, key=int) for n, js in by.items()}


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    dry = "--dry-run" in argv
    arms_f = f"{ROOT}/harm4_runs.json"
    arms = json.load(open(arms_f)) if os.path.exists(arms_f) else {}
    for name, jobs in runs_of(arms).items():
        parsed = [(j, parse(f"{ROOT}/harm4_{j}.out")) for j in jobs if os.path.exists(f"{ROOT}/harm4_{j}.out")]
        if not parsed:
            continue
        rid = f"r4-{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}"
        state_f, sent_f = f"{ROOT}/.wandb_{rid}.state", f"{ROOT}/.wandb_{rid}.final"
        last, val_done = load_state(state_f)
        sent = set(open(sent_f).read().split()) if os.path.exists(sent_f) else set()
        pl = plan(parsed, last, val_done)
        new_final = [j for j in pl["finished_jobs"] if j not in sent]
        if dry:
            print(rid, "jobs", [j for j, _ in parsed], "new events", len(pl["events"]), "of which val", sum(1 for e in pl["events"] if e[1] == "val"),
                  "last step", pl["last_step"], "finished", pl["finished_jobs"], "to finalise", new_final)
            continue
        if not pl["events"] and not new_final:
            continue
        import wandb

        hdr = parsed[0][1]["header"]
        run = wandb.init(project="HARMONIA", entity="Protein-Design-Tools", id=rid, name=name, resume="allow",
                         dir=os.environ.get("WANDB_DIR", "/path/to/work/wandb"),
                         config={"slurm_jobs": [j for j, _ in parsed], "header": hdr}, settings=wandb.Settings(silent=True))
        define_axes(run)
        emit(run, pl["events"])
        if new_final:
            for k, v in pl["summary"].items():
                run.summary[k] = v
            open(sent_f, "w").write(" ".join(sorted(sent | set(new_final))))
        run.finish()
        open(state_f, "w").write(json.dumps({"train": pl["last_step"], "val": pl["val_done"] or {}}))


if __name__ == "__main__":
    main()
