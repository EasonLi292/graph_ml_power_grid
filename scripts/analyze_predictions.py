"""Deep-dive analysis of the droop surrogate's predictions.

Produces, for the coordinate-free (2-dim node feature) model and the older
coordinate-using (6-dim) model:

  * per-topology correlation + precision (Pearson, Spearman, R², MAE, rel-MAE),
    separating in-distribution (n_top 3,7) from the held-out OOD topology
    (n_top 4);
  * precision vs droop magnitude (where in the dynamic range error lives);
  * a design-space error map over (wire_width, C_decap) for the OOD topology;
  * per-load-site error (which sites on the grid are intrinsically harder);
  * residual distribution / calibration.

Numbers are dumped to docs/analysis/prediction_metrics.json and figures to
docs/figures/. Run with python3.12 (needs torch).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eason import EncoderConfig, PDNDroopRegressor
from tools.pyg_dataset import LOG_FLOOR, RegularPDNDataset

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
DATA_DIR = ROOT / "docs" / "analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_H5 = ROOT / "datasets" / "regular_v5" / "dataset.h5"
NOCOORD_CKPT = ROOT / "checkpoints" / "droop_v5_nocoord.pt"


def load_model(ckpt: Path) -> PDNDroopRegressor:
    model = PDNDroopRegressor(
        EncoderConfig(hidden_dim=64, n_layers=7, conv_type="admittance",
                      drop_edge_p=0.0),
        target_space="log",
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_split(model, split: str, batch_size: int = 64):
    """Return per-load-site arrays for one split.

    Columns: pred_v, true_v (volts), n_top, wire_width, C_decap, site_idx,
    sample_id.
    """
    ds = RegularPDNDataset(DATA_H5, split=split, target="log")

    preds, trues, ntops, wws, cds, sites, sids = [], [], [], [], [], [], []
    buf, meta = [], []
    sample_counter = 0

    def flush():
        nonlocal buf, meta
        if not buf:
            return
        batch = Batch.from_data_list(buf)
        out = model(batch).cpu().numpy()  # log10 volts, flat over all load edges
        pv = 10.0 ** out
        off = 0
        for (nt, ww, cd, sid, y_true) in meta:
            # load count is per-sample (varies across anchors) = len(y_true).
            n_loads = y_true.shape[0]
            chunk = pv[off:off + n_loads]
            tv = 10.0 ** y_true
            for s in range(n_loads):
                preds.append(chunk[s]); trues.append(tv[s])
                ntops.append(nt); wws.append(ww); cds.append(cd)
                sites.append(s); sids.append(sid)
            off += n_loads
        buf, meta = [], []

    for i in range(len(ds)):
        data = ds[i]
        nt = int(ds._n_top[i])
        y_true = data["y"].numpy()
        ww = float(ds._global[i, ds._ww_col]); cd = float(ds._global[i, ds._cd_col])
        buf.append(data)
        meta.append((nt, ww, cd, sample_counter, y_true))
        sample_counter += 1
        if len(buf) >= batch_size:
            flush()
    flush()

    return {
        "pred_v": np.array(preds), "true_v": np.array(trues),
        "n_top": np.array(ntops), "wire_width": np.array(wws),
        "C_decap": np.array(cds), "site": np.array(sites),
        "sample_id": np.array(sids),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(pred_v, true_v) -> dict:
    err = pred_v - true_v
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true_v - true_v.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    rel = mae / max(float(true_v.mean()), LOG_FLOOR)
    # correlations in log space (spans orders of magnitude)
    lp = np.log10(np.maximum(pred_v, LOG_FLOOR))
    lt = np.log10(np.maximum(true_v, LOG_FLOOR))
    pear = float(pearsonr(lp, lt)[0])
    spear = float(spearmanr(pred_v, true_v)[0])
    return {
        "n": int(pred_v.size), "r2": r2, "mae_mV": mae * 1e3,
        "rmse_mV": rmse * 1e3, "rel_mae": rel,
        "pearson_log": pear, "spearman": spear,
    }


def per_sample_worst(d):
    """Collapse per-site to per-sample worst-load droop (the design metric)."""
    sids = d["sample_id"]
    order = np.argsort(sids, kind="stable")
    sids_s = sids[order]
    pv = d["pred_v"][order]; tv = d["true_v"][order]; nt = d["n_top"][order]
    uniq, idx = np.unique(sids_s, return_index=True)
    pw, tw, nts = [], [], []
    bounds = list(idx) + [len(sids_s)]
    for k in range(len(uniq)):
        a, b = bounds[k], bounds[k + 1]
        pw.append(pv[a:b].max()); tw.append(tv[a:b].max()); nts.append(nt[a])
    return np.array(pw), np.array(tw), np.array(nts)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_pred_vs_true(nocoord_by_split):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        ("n_top = 3  (train-distribution)", "train", 3),
        ("n_top = 7  (train-distribution)", "train", 7),
        ("n_top = 4  (HELD-OUT / OOD)", "test", 4),
    ]
    for ax, (title, split, nt) in zip(axes, panels):
        d = nocoord_by_split[split]
        m = d["n_top"] == nt
        tv = d["true_v"][m] * 1e3; pv = d["pred_v"][m] * 1e3
        ax.scatter(tv, pv, s=4, alpha=0.25,
                   color="#c0392b" if nt == 4 else "#2c6fbb")
        lo = min(tv.min(), pv.min()); hi = max(tv.max(), pv.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal y = x")
        mt = metrics(d["pred_v"][m], d["true_v"][m])
        ax.set_title(title)
        ax.set_xlabel("true droop (mV)"); ax.set_ylabel("predicted droop (mV)")
        ax.text(0.05, 0.95,
                f"R² = {mt['r2']:.3f}\nSpearman = {mt['spearman']:.3f}\n"
                f"MAE = {mt['mae_mV']:.4f} mV",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Predicted vs true per-load droop (coordinate-free model)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_pred_vs_true.png", dpi=130)
    plt.close(fig)


def fig_error_vs_magnitude(test):
    tv = test["true_v"] * 1e3
    pv = test["pred_v"] * 1e3
    abs_err = np.abs(pv - tv)
    rel_err = abs_err / np.maximum(tv, 1e-4)

    # log-spaced bins over the true-droop dynamic range
    edges = np.geomspace(max(tv.min(), 1e-3), tv.max(), 13)
    centers = np.sqrt(edges[:-1] * edges[1:])
    which = np.digitize(tv, edges) - 1
    mae_b, rel_b, cnt = [], [], []
    for b in range(len(centers)):
        sel = which == b
        if sel.sum() == 0:
            mae_b.append(np.nan); rel_b.append(np.nan); cnt.append(0); continue
        mae_b.append(np.mean(abs_err[sel]))
        rel_b.append(np.median(rel_err[sel]))
        cnt.append(int(sel.sum()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(centers, mae_b, "o-", color="#c0392b")
    ax1.set_xscale("log")
    ax1.set_xlabel("true droop (mV, log)"); ax1.set_ylabel("mean |error| (mV)")
    ax1.set_title("Absolute error grows with droop magnitude")
    ax1b = ax1.twinx()
    ax1b.bar(centers, cnt, width=centers * 0.3, alpha=0.15, color="gray")
    ax1b.set_ylabel("# load sites in bin", color="gray")

    ax2.plot(centers, np.array(rel_b) * 100, "s-", color="#2c6fbb")
    ax2.set_xscale("log")
    ax2.set_xlabel("true droop (mV, log)")
    ax2.set_ylabel("median relative error (%)")
    ax2.set_title("Relative error is worst at the small-droop tail")
    ax2.axhline(0, color="k", lw=0.5)
    fig.suptitle("Where precision differs — error vs droop magnitude "
                 "(OOD n_top = 4)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_error_vs_magnitude.png", dpi=130)
    plt.close(fig)


def fig_designspace_error(test):
    ww = test["wire_width"]; cd = test["C_decap"]
    abs_err = np.abs(test["pred_v"] - test["true_v"]) * 1e3
    rel_err = abs_err / np.maximum(test["true_v"] * 1e3, 1e-4)

    nb = 12
    ww_edges = np.linspace(ww.min(), ww.max(), nb + 1)
    cd_edges = np.geomspace(cd.min(), cd.max(), nb + 1)
    wi = np.clip(np.digitize(ww, ww_edges) - 1, 0, nb - 1)
    ci = np.clip(np.digitize(cd, cd_edges) - 1, 0, nb - 1)

    grid_abs = np.full((nb, nb), np.nan)
    grid_rel = np.full((nb, nb), np.nan)
    for a in range(nb):
        for b in range(nb):
            sel = (wi == a) & (ci == b)
            if sel.sum() >= 3:
                grid_abs[b, a] = np.mean(abs_err[sel])
                grid_rel[b, a] = np.median(rel_err[sel]) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ext = [ww_edges[0], ww_edges[-1], 0, nb]
    im1 = ax1.imshow(grid_abs, origin="lower", aspect="auto", extent=ext,
                     cmap="magma")
    ax1.set_title("mean |error| (mV)")
    fig.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(grid_rel, origin="lower", aspect="auto", extent=ext,
                     cmap="viridis")
    ax2.set_title("median relative error (%)")
    fig.colorbar(im2, ax=ax2)
    for ax in (ax1, ax2):
        ax.set_xlabel("wire_width")
        ax.set_yticks(np.linspace(0.5, nb - 0.5, 5))
        ax.set_yticklabels([f"{v:.1e}" for v in
                            np.geomspace(cd_edges[0], cd_edges[-1], 5)])
        ax.set_ylabel("C_decap (F)")
    fig.suptitle("Design-space error map (OOD n_top = 4): error concentrates "
                 "at thin-wire / low-cap (high-droop) corner", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_designspace_error.png", dpi=130)
    plt.close(fig)


def fig_per_site(test):
    sites = np.unique(test["site"])
    mae, rel, truemean = [], [], []
    for s in sites:
        sel = test["site"] == s
        ae = np.abs(test["pred_v"][sel] - test["true_v"][sel]) * 1e3
        mae.append(ae.mean())
        rel.append(np.median(ae / np.maximum(test["true_v"][sel] * 1e3, 1e-4)) * 100)
        truemean.append(test["true_v"][sel].mean() * 1e3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(sites, mae, color="#c0392b", alpha=0.8)
    ax1.set_xlabel("load-site index (0–13)"); ax1.set_ylabel("MAE (mV)")
    ax1.set_title("Per-site absolute error")
    ax2.bar(sites, truemean, color="#2c6fbb", alpha=0.6, label="mean true droop")
    ax2.set_xlabel("load-site index (0–13)")
    ax2.set_ylabel("mean true droop (mV)")
    ax2.set_title("Per-site mean droop (harder sites = deeper droop)")
    fig.suptitle("Per-load-site breakdown (OOD n_top = 4)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_per_site.png", dpi=130)
    plt.close(fig)


def fig_residual(test):
    lp = np.log10(np.maximum(test["pred_v"], LOG_FLOOR))
    lt = np.log10(np.maximum(test["true_v"], LOG_FLOOR))
    resid = lp - lt  # log10 residual: 0.043 ≈ +10%
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(resid, bins=60, color="#7f3fbf", alpha=0.8)
    ax1.axvline(0, color="k", lw=1)
    ax1.set_xlabel("log10(pred) − log10(true)")
    ax1.set_ylabel("count")
    ax1.set_title(f"Residuals (log10): bias={resid.mean():+.4f}, "
                  f"σ={resid.std():.4f}")
    ax2.scatter(lt, resid, s=4, alpha=0.2, color="#7f3fbf")
    ax2.axhline(0, color="k", lw=1)
    ax2.set_xlabel("log10(true droop, V)")
    ax2.set_ylabel("log10 residual")
    ax2.set_title("Residual vs magnitude (heteroscedasticity check)")
    fig.suptitle("Residual structure (OOD n_top = 4, coordinate-free model)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_residual.png", dpi=130)
    plt.close(fig)


def main() -> None:
    print("Loading model...")
    nocoord = load_model(NOCOORD_CKPT)

    splits = ["train", "val", "test"]
    nocoord_by_split = {s: predict_split(nocoord, s) for s in splits}

    summary = {"nocoord": {}}

    def fill(tag, by_split):
        out = {}
        # overall per split
        for s in ["train", "val", "test"]:
            d = by_split[s]
            out[f"{s}_all"] = metrics(d["pred_v"], d["true_v"])
            for nt in sorted(np.unique(d["n_top"])):
                m = d["n_top"] == nt
                out[f"{s}_n{int(nt)}"] = metrics(d["pred_v"][m], d["true_v"][m])
        # worst-load (design-relevant) on test
        pw, tw, nts = per_sample_worst(by_split["test"])
        out["test_worstload"] = metrics(pw, tw)
        summary[tag] = out

    fill("nocoord", nocoord_by_split)

    # ---- figures ----
    print("Rendering figures...")
    fig_pred_vs_true(nocoord_by_split)
    fig_error_vs_magnitude(nocoord_by_split["test"])
    fig_designspace_error(nocoord_by_split["test"])
    fig_per_site(nocoord_by_split["test"])
    fig_residual(nocoord_by_split["test"])

    (DATA_DIR / "prediction_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nFigures -> {FIG_DIR}")
    print(f"Metrics -> {DATA_DIR / 'prediction_metrics.json'}")


if __name__ == "__main__":
    main()
