#!/usr/bin/env python
"""Build the curated wandb dashboard for Protein-Design-Tools/HARMONIA.

Unlike the TRIAD dashboard (one run, train-vs-val lines per panel), HARMONIA has
one run PER TRAINING PHASE, so the interesting comparison is across runs. Panels
therefore show a SINGLE metric each and let wandb colour by run -- five lines,
one per phase, directly comparable because everything is logged against absolute
epoch (see wandb_sync_harmonia.py).

auto_organize_prefix is forced to 0 or the library regroups everything by metric
prefix and our sections disappear.

Run with the tools venv (the training env has no wandb_workspaces):
  /path/to/work/wandb_tools_venv/bin/python \
      scripts/wandb_make_workspace_harmonia.py
"""
import json
import wandb_workspaces.workspaces as ws
from wandb_workspaces.workspaces import internal
import wandb_workspaces.reports.v2.interface as wr

ENTITY, PROJECT = "Protein-Design-Tools", "HARMONIA"
XAXIS = "epoch"          # absolute epoch -> charts read in checkpoint numbering
SM = 0.85
PERSONAL_PREFIX = "your-wandb-username"   # set None to skip the landing-view update

B = "train/batch_mean/"
E = "train/per_epoch_"

# Okabe-Ito colourblind-safe palette, used only where a panel holds >1 metric.
BLUE, ORANGE, GREEN, PINK = "#0072B2", "#E69F00", "#009E73", "#CC79A7"


def LP(title, y, colors=None, titles=None, log_y=False, smooth=True):
    # This wandb_workspaces version expects {metric_name: value} maps, not lists.
    cmap = dict(zip(y, colors)) if colors else None
    tmap = dict(zip(y, titles)) if titles else None
    return wr.LinePlot(
        title=title, x=XAXIS, y=y, log_y=log_y or None,
        smoothing_type="exponential" if smooth else "none",
        smoothing_factor=SM if smooth else 0.0,
        smoothing_show_original=False,
        line_colors=cmap, line_titles=tmap,
    )


def build_sections():
    return [
        # ---- the separated "most important" panel -------------------------
        ws.Section(
            name="★ Key metrics — read these first", is_open=True,
            layout_settings=ws.SectionLayoutSettings(columns=2), panels=[
                LP("Adjacency contact loss — the objective this project tunes",
                   [B + "adjacency_contact_loss"]),
                LP("Backbone accuracy — mean lDDT (protein)",
                   [B + "mean_lddt_protein"]),
                LP("Structure loss at LOW t — where geometry corruption showed up",
                   [B + "mse_loss_low_t"]),
                LP("Total loss", [B + "total_loss"], log_y=True),
            ]),

        # ---- adjacency: the actual subject of the work ---------------------
        ws.Section(
            name="Adjacency conditioning", is_open=True,
            layout_settings=ws.SectionLayoutSettings(columns=3), panels=[
                LP("Contact loss (per-epoch mean)", [E + "adjacency_contact_loss"],
                   smooth=False),
                LP("Contact loss split by noise level",
                   [B + "adjacency_contact_loss_high_t", B + "adjacency_contact_loss_low_t"],
                   [BLUE, ORANGE], ["high t", "low t"]),
                LP("Conditioned block pairs per batch (signal available)",
                   [B + "adjacency_n_conditioned_block_pairs"]),
            ]),

        # ---- structure / diffusion ----------------------------------------
        ws.Section(
            name="Structure & diffusion", is_open=True,
            layout_settings=ws.SectionLayoutSettings(columns=3), panels=[
                LP("MSE loss (mean)", [B + "mse_loss_mean"]),
                LP("MSE loss by noise level",
                   [B + "mse_loss_high_t", B + "mse_loss_low_t"],
                   [BLUE, ORANGE], ["high t", "low t"]),
                LP("mean lDDT (all tokens)", [B + "mean_lddt"]),
                LP("LP norm", [B + "lp_norm"]),
                LP("LP norm — unindexed diffused", [B + "lp_norm_unindexed_diffused"]),
                LP("lDDT protein (per-epoch)", [E + "mean_lddt_protein"], smooth=False),
            ]),

        # ---- sequence ------------------------------------------------------
        ws.Section(
            name="Sequence", is_open=False,
            layout_settings=ws.SectionLayoutSettings(columns=3), panels=[
                LP("Sequence recovery", [B + "seq_recovery"]),
                LP("Sequence recovery at lowest t", [B + "lowest_t_seq_recovery"]),
                LP("Token-level sequence loss", [B + "token_lvl_sequence_loss"], log_y=True),
            ]),

        # ---- health --------------------------------------------------------
        ws.Section(
            name="Training health", is_open=False,
            layout_settings=ws.SectionLayoutSettings(columns=3), panels=[
                LP("Learning rate", ["train/learning_rate"], smooth=False),
                LP("Valid t fraction", [B + "valid_t_fraction"]),
                LP("Number of valid t", [B + "n_valid_t"]),
            ]),
    ]


def settings():
    return ws.WorkspaceSettings(
        x_axis=XAXIS, smoothing_type="exponential", smoothing_weight=int(SM * 100),
        remove_legends_from_panels=False,
    )


def persist(workspace, label):
    """Upsert a workspace with auto_organize_prefix forced OFF; verify; return url."""
    view = workspace._to_model()
    if not view.name:
        view.name = internal._generate_view_name()
    pbc = view.spec.section.panel_bank_config
    pbc.settings.auto_organize_prefix = 0
    pbc.settings.show_empty_sections = True
    resp = internal.upsert_view2(view)
    name = resp["upsertView"]["view"]["name"]
    workspace._internal_name, workspace._internal_id = name, resp["upsertView"]["view"]["id"]
    # Read-back is verification only -- personal workspaces ("nw-<user>-w") do not
    # round-trip through the url-query helper, so don't let that fail the upsert.
    try:
        raw = internal.get_view_dict(ENTITY, PROJECT,
                                     internal._internal_name_to_url_query_str(name))
        spec = raw["spec"]
        spec = json.loads(spec) if isinstance(spec, str) else spec
        pbc = spec["section"]["panelBankConfig"]
        print(f"[{label}] {workspace.url}\n    verified: "
              f"autoOrganizePrefix={pbc['settings'].get('autoOrganizePrefix')} "
              f"sections={len(pbc.get('sections', []))}")
    except Exception as e:
        print(f"[{label}] {workspace.url}\n    upserted (read-back skipped: {type(e).__name__})")
    return workspace.url


SAVED_NAME = "HARMONIA — curated"


def list_views():
    import wandb
    # newer wandb.Api() has no .client -- go through the library's own helper
    from wandb_workspaces._graphql import execute_graphql
    q = """query($entityName:String,$name:String){project(name:$name,entityName:$entityName){
            allViews(viewType:"project-view"){edges{node{id displayName name}}}}}"""
    res = execute_graphql(wandb.Api(), q, {"entityName": ENTITY, "name": PROJECT})
    return [e["node"] for e in res["project"]["allViews"]["edges"]]


def apply_to(node, label):
    """Rewrite an EXISTING view in place rather than minting a new one."""
    wsp = ws.Workspace(entity=ENTITY, project=PROJECT, name=node["displayName"],
                       sections=build_sections(), settings=settings())
    wsp._internal_name, wsp._internal_id = node["name"], node["id"]
    return persist(wsp, label)


if __name__ == "__main__":
    views = list_views()

    # 1) curated saved view -- reuse if it already exists so re-runs don't pile up
    mine = [n for n in views if n["displayName"] == SAVED_NAME]
    if mine:
        apply_to(mine[0], "saved view (updated in place)")
        for dup in mine[1:]:
            print(f"    NOTE: duplicate saved view still present: {dup['name']}")
    else:
        w = ws.Workspace(entity=ENTITY, project=PROJECT, name=SAVED_NAME,
                         sections=build_sections(), settings=settings())
        persist(w, "saved view (created)")

    # 2) same layout on the personal workspace = the default landing view.
    #    Internal names look like `nw-your-wandb-username-w`, so match on substring.
    if PERSONAL_PREFIX:
        personal = next((n for n in views if PERSONAL_PREFIX in n["name"]), None)
        if personal:
            apply_to(personal, "personal landing view")
        else:
            print("personal workspace not found -- open the project once, then re-run")
