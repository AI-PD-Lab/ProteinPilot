#!/usr/bin/env python3
"""Plot SS/ADJ inference benchmark and Phase 2 training/validation metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(path)


def plot_benchmark(summary_csv: Path, out_dir: Path) -> None:
    if not summary_csv.exists():
        return
    df = pd.read_csv(summary_csv)
    if df.empty:
        return
    df["label"] = df["mode"] + "\n" + df["case"]

    metrics = [
        ("adj_block_recovery_mean", "ADJ Recovery At 6A", "adj_recovery_by_case.png", (0, 1.05)),
        ("ss_accuracy_mean", "SS Accuracy", "ss_accuracy_by_case.png", (0, 1.05)),
        ("adj_mean_min_distance_mean", "ADJ Mean Min Distance", "adj_distance_by_case.png", None),
        ("ss_ratio_achieved_mean", "Achieved SS Ratio E/(H+E)", "ss_ratio_by_case.png", (0, 1.05)),
    ]
    for col, title, name, ylim in metrics:
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col]).copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["mode", "case"])
        fig, ax = plt.subplots(figsize=(max(10, len(sub) * 0.42), 5.5))
        colors = pd.factorize(sub["mode"])[0]
        ax.bar(np.arange(len(sub)), sub[col], color=plt.cm.tab20(colors % 20))
        ax.set_title(title)
        ax.set_ylabel(col.replace("_mean", ""))
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub["label"], rotation=70, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
        _save(fig, out_dir / name)

    # Aggregate by mode with error bars.
    agg_cols = [c for c in ["adj_block_recovery_mean", "ss_accuracy_mean"] if c in df.columns]
    if agg_cols:
        rows = []
        for mode, g in df.groupby("mode"):
            row = {"mode": mode}
            for c in agg_cols:
                vals = g[c].dropna().astype(float)
                row[c] = vals.mean() if len(vals) else np.nan
                row[c + "_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
            rows.append(row)
        agg = pd.DataFrame(rows).sort_values("mode")
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(agg))
        width = 0.36
        for i, c in enumerate(agg_cols):
            off = (i - (len(agg_cols)-1)/2) * width
            ax.bar(x + off, agg[c], width=width, yerr=agg[c + "_std"], capsize=3, label=c.replace("_mean", ""))
        ax.set_ylim(0, 1.05)
        ax.set_title("Mean Adherence By Mode")
        ax.set_xticks(x)
        ax.set_xticklabels(agg["mode"], rotation=35, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        _save(fig, out_dir / "adherence_by_mode.png")


def plot_binder_metrics(csv_path: Path, out_dir: Path, prefix: str) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    metric_cols = [c for c in [
        "interface_min_cb_ca_dist",
        "binder_beta_frac",
        "loop_frac_in_loop_span",
        "target_beta_frac",
        "adjacency_recovered",
        "n_contacts_6A",
        "target_res_contacted_6A",
        "loop_res_contacted_6A",
    ] if c in df.columns]
    for col in metric_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df[col].dropna().astype(float), bins=min(12, max(4, len(df)//2)), alpha=0.8)
        ax.set_title(f"{prefix}: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.grid(axis="y", alpha=0.25)
        _save(fig, out_dir / f"{prefix}_{col}.png")

    if {"interface_min_cb_ca_dist", "binder_beta_frac"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(df["interface_min_cb_ca_dist"], df["binder_beta_frac"], alpha=0.8)
        ax.axvline(6.0, color="red", linestyle="--", linewidth=1, label="6A ADJ threshold")
        ax.set_xlabel("interface min distance")
        ax.set_ylabel("binder beta fraction")
        ax.set_title(f"{prefix}: Interface vs Beta Adherence")
        ax.legend()
        ax.grid(alpha=0.25)
        _save(fig, out_dir / f"{prefix}_interface_vs_beta.png")


def _plot_series(df: pd.DataFrame, x: str, cols: list[str], title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for col in cols:
        if col not in df.columns:
            continue
        sub = df[[x, col]].dropna()
        if sub.empty:
            continue
        sub = sub.sort_values(x)
        ax.plot(sub[x], sub[col], label=col)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, out)


_MODE_LABEL = {
    "adj_only": "ADJ only",
    "ss_only": "SS only",
    "full_ss_adj": "Full SS+ADJ",
    "full_ss_adj_ssratio": "Full+β-ratio",
    "interaction_site_partial": "Partial site",
    "interaction_site_partial_ssratio": "Partial+β-ratio",
    "interaction_site_partial_strict_no": "Partial+strict NO",
    "remodeling_partial": "Remodel",
    "select_ss": "Inline select",
    "unconditional": "Baseline",
    "ss_ratio": "ss_ratio",
    "is_non_loopy": "Non-loopy",
    "scaffold_expansion_masked": "Expand masked",
    "scaffold_expansion_terminal": "Expand NC",
    "scaffold_expansion_loop": "Expand loops",
    "fixed_target_beta_binder": "Beta binder",
    "fixed_target_beta_binder_ssratio": "Binder+β-ratio",
    "fixed_target_beta_binder_strict_no": "Binder+strict NO",
    "fixed_target_loop_binder": "Loop binder",
    "fixed_target_loop_binder_strict_no": "Loop binder+strict NO",
    "antibody_cdr_loop": "Antibody CDR",
    "antibody_cdr_loop_strict_no": "Antibody CDR+strict NO",
    "multi_patch_binder": "Multi-patch binder",
    "scaffold_multi_insert": "Multi-insert",
    "large_full_ss_adj": "Large Rossmann",
    "conflict_ss_ratio": "Conflict SS/ratio",
}

_MODE_ORDER = [
    "adj_only",
    "ss_only",
    "full_ss_adj",
    "full_ss_adj_ssratio",
    "interaction_site_partial",
    "interaction_site_partial_ssratio",
    "interaction_site_partial_strict_no",
    "remodeling_partial",
]


def plot_combined_adherence_by_topology(summary_csv: Path, out_dir: Path) -> None:
    """SS accuracy + ADJ recovery side-by-side for helix vs beta topologies (no Rossmann)."""
    if not summary_csv.exists():
        return
    df = pd.read_csv(summary_csv)
    if df.empty:
        return

    helix = df[df["case"].str.contains("helix_bundle", na=False)].copy()
    beta = df[df["case"].str.contains("beta_sandwich", na=False)].copy()

    # Keep only modes present in _MODE_ORDER
    helix = helix[helix["mode"].isin(_MODE_ORDER)].copy()
    beta = beta[beta["mode"].isin(_MODE_ORDER)].copy()
    if helix.empty and beta.empty:
        return

    # Per-panel mode lists: each panel only shows modes it actually has data for
    helix_modes = [m for m in _MODE_ORDER if m in set(helix["mode"])]
    beta_modes  = [m for m in _MODE_ORDER if m in set(beta["mode"])]

    ss_color  = "#4C78A8"
    adj_color = "#F58518"
    bar_w = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    legend_handles: list = []

    for ax_idx, (ax, sub, topo_label, panel_modes) in enumerate([
        (axes[0], helix, "α-Helical (helix bundle)", helix_modes),
        (axes[1], beta,  "β-Sheet (beta sandwich)",  beta_modes),
    ]):
        x = np.arange(len(panel_modes))
        mode_to_row = {row["mode"]: row for _, row in sub.iterrows()}

        def _val(m, col):
            if m not in mode_to_row:
                return np.nan
            v = mode_to_row[m][col]
            return float(v) if v == v else np.nan

        ss_arr    = np.array([_val(m, "ss_accuracy_mean")        for m in panel_modes])
        adj_arr   = np.array([_val(m, "adj_block_recovery_mean") for m in panel_modes])
        ss_std_arr  = np.array([_val(m, "ss_accuracy_std")         for m in panel_modes])
        adj_std_arr = np.array([_val(m, "adj_block_recovery_std")  for m in panel_modes])
        ss_std_arr[~np.isfinite(ss_std_arr)]   = 0.0
        adj_std_arr[~np.isfinite(adj_std_arr)] = 0.0

        ss_mask  = np.isfinite(ss_arr)
        adj_mask = np.isfinite(adj_arr)
        if ss_mask.any():
            bars = ax.bar(x[ss_mask]  - bar_w / 2, ss_arr[ss_mask],
                          width=bar_w, color=ss_color, alpha=0.85,
                          yerr=ss_std_arr[ss_mask], capsize=3, label="SS accuracy", zorder=3)
            if ax_idx == 0:
                legend_handles.append(bars)
        if adj_mask.any():
            bars = ax.bar(x[adj_mask] + bar_w / 2, adj_arr[adj_mask],
                          width=bar_w, color=adj_color, alpha=0.85,
                          yerr=adj_std_arr[adj_mask], capsize=3, label="ADJ recovery", zorder=3)
            if ax_idx == 0:
                legend_handles.append(bars)

        ax.set_title(topo_label, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([_MODE_LABEL.get(m, m) for m in panel_modes], rotation=38, ha="right", fontsize=8.5)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Accuracy / Recovery")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    axes[0].legend(handles=legend_handles, labels=["SS accuracy", "ADJ recovery"],
                   fontsize=9, loc="upper left")

    fig.suptitle("SS Accuracy & ADJ Recovery by Conditioning Mode", fontsize=12)
    _save(fig, out_dir / "combined_ss_adj_by_topology.png")


def plot_binder_cross_topology(summary_csv: Path, out_dir: Path) -> None:
    """SS accuracy + ADJ recovery for all binder cases, grouped by target topology."""
    if not summary_csv.exists():
        return
    df = pd.read_csv(summary_csv)
    if df.empty:
        return

    # Each entry: (case_key, x_label, group)
    BINDER_CASES = [
        # ── Helix target ──────────────────────────────────────────────────────
        ("helix_target_helix_binder",            "Helix\nbinder",   "α target"),
        ("helix_target_helix_binder_strict_no",  "Helix binder\n+strict NO", "α target"),
        ("helix_target_beta_binder",             "β binder",        "α target"),
        ("helix_target_loop_binder",             "Loop\nbinder",    "α target"),
        # ── Beta target ───────────────────────────────────────────────────────
        ("binder_beta_adj",                      "β binder",        "β target"),
        ("binder_beta_adj_strict_no",            "β binder\n+strict NO", "β target"),
        ("helix_binder_beta_target",             "Helix\nbinder",   "β target"),
        ("helix_binder_beta_target_strict_no",   "Helix binder\n+strict NO", "β target"),
        ("binder_loop_adj",                      "Loop\nbinder",    "β target"),
        # ── Other targets ─────────────────────────────────────────────────────
        ("helix_binder_mixed_target",            "Helix binder\nmixed target", "Other"),
        ("antibody_cdr_loop",                    "Antibody\nCDR loop",  "Other"),
        ("antibody_cdr_loop_strict_no",          "CDR loop\n+strict NO", "Other"),
        ("multi_patch_binder",                   "Multi-patch\nbinder", "Other"),
    ]

    case_to_row = {r["case"]: r for _, r in df.iterrows()}
    present = [(key, lbl, grp) for key, lbl, grp in BINDER_CASES if key in case_to_row]
    if not present:
        return

    ss_color  = "#4C78A8"
    adj_color = "#F58518"
    bar_w = 0.3
    gap   = 0.7   # extra gap between groups

    # Assign x positions with gaps between groups
    groups = []
    for _, _, grp in present:
        if not groups or groups[-1] != grp:
            groups.append(grp)

    x_pos = []
    pos = 0.0
    prev_grp = None
    for _, _, grp in present:
        if prev_grp is not None and grp != prev_grp:
            pos += gap
        x_pos.append(pos)
        pos += 1.0
        prev_grp = grp

    x = np.array(x_pos)
    labels = [lbl for _, lbl, _ in present]
    grp_list = [grp for _, _, grp in present]

    ss_vals  = np.array([float(case_to_row[k]["ss_accuracy_mean"])      if case_to_row[k]["ss_accuracy_mean"]      == case_to_row[k]["ss_accuracy_mean"]      else np.nan for k, _, _ in present])
    adj_vals = np.array([float(case_to_row[k]["adj_block_recovery_mean"]) if case_to_row[k]["adj_block_recovery_mean"] == case_to_row[k]["adj_block_recovery_mean"] else np.nan for k, _, _ in present])
    ss_std   = np.array([float(case_to_row[k]["ss_accuracy_std"])        if case_to_row[k]["ss_accuracy_std"]        == case_to_row[k]["ss_accuracy_std"]        else 0.0  for k, _, _ in present])
    adj_std  = np.array([float(case_to_row[k]["adj_block_recovery_std"]) if case_to_row[k]["adj_block_recovery_std"] == case_to_row[k]["adj_block_recovery_std"] else 0.0  for k, _, _ in present])

    fig, ax = plt.subplots(figsize=(max(12, len(present) * 1.1), 5.5))

    # Shaded backgrounds per group
    grp_colors = {"α target": "#EDF3FB", "β target": "#FEF3E6", "Other": "#F0F0F0"}
    prev_grp = None
    grp_start = 0.0
    for i, (_, _, grp) in enumerate(present):
        if prev_grp is not None and grp != prev_grp:
            ax.axvspan(grp_start - 0.5, x[i] - gap / 2, color=grp_colors.get(prev_grp, "#F8F8F8"), alpha=0.4, zorder=0)
            grp_start = x[i] - gap / 2
        if i == len(present) - 1:
            ax.axvspan(grp_start, x[i] + 0.5, color=grp_colors.get(grp, "#F8F8F8"), alpha=0.4, zorder=0)
        prev_grp = grp

    ss_mask  = np.isfinite(ss_vals)
    adj_mask = np.isfinite(adj_vals)
    if ss_mask.any():
        ax.bar(x[ss_mask]  - bar_w / 2, ss_vals[ss_mask],  width=bar_w, color=ss_color,  alpha=0.85,
               yerr=ss_std[ss_mask],   capsize=3, label="SS accuracy",   zorder=3)
    if adj_mask.any():
        ax.bar(x[adj_mask] + bar_w / 2, adj_vals[adj_mask], width=bar_w, color=adj_color, alpha=0.85,
               yerr=adj_std[adj_mask], capsize=3, label="ADJ recovery",  zorder=3)

    # Group labels at top
    prev_grp = None
    grp_xs = {}
    for xi, (_, _, grp) in zip(x, present):
        grp_xs.setdefault(grp, []).append(xi)
    for grp, xs in grp_xs.items():
        mid = (min(xs) + max(xs)) / 2
        ax.text(mid, 1.10, grp, ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#444444", transform=ax.get_xaxis_transform())

    # Separator lines between groups
    prev_grp = None
    for i, (_, _, grp) in enumerate(present):
        if prev_grp is not None and grp != prev_grp:
            ax.axvline(x[i] - gap / 2, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)
        prev_grp = grp

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Accuracy / Recovery")
    ax.set_title("Binder Conditioning: SS Accuracy & ADJ Recovery by Target × Binder Type", fontsize=11)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(fontsize=9, loc="upper left")
    _save(fig, out_dir / "binder_cross_topology.png")


def plot_training(train_root: Path, out_dir: Path) -> None:
    metrics_csv = train_root / "lightning_logs/version_0/metrics.csv"
    val_csv = train_root / "val_metrics/validation_output_all_epochs.csv"
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        _plot_series(
            df,
            "step",
            [
                "train/batch_mean/total_loss",
                "train/per_epoch_total_loss",
                "train/batch_mean/mse_loss_mean",
                "train/per_epoch_mse_loss_mean",
            ],
            "Phase 2 Training Loss",
            out_dir / "train_loss.png",
        )
        _plot_series(
            df,
            "step",
            [
                "train/learning_rate",
            ],
            "Learning Rate",
            out_dir / "learning_rate.png",
        )
        _plot_series(
            df,
            "step",
            [
                "val/loop/backbone_metrics.fraction_chainbreaks",
                "val/loop/backbone_metrics.frac_clashing",
                "val/loop/general_metrics.loop_fraction",
                "val/loop/general_metrics.sheet_fraction",
                "val/loop/general_metrics.helix_fraction",
            ],
            "Logged Loop Validation Metrics",
            out_dir / "val_loop_logged_metrics.png",
        )
    if val_csv.exists():
        val = pd.read_csv(val_csv)
        numeric = [c for c in val.select_dtypes(include=[np.number]).columns.tolist() if c != "epoch"]
        agg = val.groupby("epoch")[numeric].mean(numeric_only=True).reset_index()
        _plot_series(
            agg,
            "epoch",
            [
                "backbone_metrics.fraction_chainbreaks",
                "backbone_metrics.frac_clashing",
                "backbone_metrics.max_ca_deviation",
            ],
            "Validation Backbone Metrics By Epoch",
            out_dir / "val_backbone_by_epoch.png",
        )
        _plot_series(
            agg,
            "epoch",
            [
                "general_metrics.loop_fraction",
                "general_metrics.sheet_fraction",
                "general_metrics.helix_fraction",
                "general_metrics.non_loop_fraction",
            ],
            "Validation Secondary Structure Fractions By Epoch",
            out_dir / "val_ss_fractions_by_epoch.png",
        )
        _plot_series(
            agg,
            "epoch",
            [
                "sidechain_metrics.mean_valid_sidechain_percent",
                "sidechain_metrics.mean_clash_percent",
                "sidechain_metrics.mean_unintended_bonds_percent",
            ],
            "Validation Sidechain Metrics By Epoch",
            out_dir / "val_sidechain_by_epoch.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--beta-binder-root", type=Path)
    parser.add_argument("--loop-binder-root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.benchmark_root:
        summary_csv = args.benchmark_root / "analysis/summary_by_case.csv"
        plot_benchmark(summary_csv, args.out_dir / "benchmark")
        plot_combined_adherence_by_topology(summary_csv, args.out_dir / "benchmark")
        plot_binder_cross_topology(summary_csv, args.out_dir / "benchmark")
    if args.beta_binder_root:
        plot_binder_metrics(args.beta_binder_root / "analysis/beta_binder_metrics.csv", args.out_dir / "beta_binder", "beta_binder")
    if args.loop_binder_root:
        plot_binder_metrics(args.loop_binder_root / "analysis/loop_binder_metrics.csv", args.out_dir / "loop_binder", "loop_binder")
    if args.train_root:
        plot_training(args.train_root, args.out_dir / "training")


if __name__ == "__main__":
    main()
