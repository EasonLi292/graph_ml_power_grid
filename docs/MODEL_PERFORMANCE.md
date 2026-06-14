# Model Performance — Figure Gallery

*The most visually useful plots for judging the droop surrogate at a glance,
collected in one place. Each links to the analysis doc with the full context.
Model: coordinate-free heterogeneous GNN, conductance gate, 7 hops
(`checkpoints/droop_v5_nocoord.pt`). All metrics are on the held-out topology
n_top = 4 (OOD), the only honest test.*

| headline | value |
|---|---|
| per-load-site R² (OOD) | 0.827 |
| **worst-load R² (OOD)** — the design-binding metric | **0.944** |
| worst-load Spearman (ranking) | **0.987** |
| in-distribution R² (n_top 3 & 7) | 0.99999 |

Regenerate everything:

```bash
python3.12 scripts/analyze_predictions.py        # forward-accuracy figures
python3.12 scripts/plot_generation.py            # inverse-design figures
python3.12 scripts/plot_surrogate_vs_sim.py      # surrogate-vs-simulator sweeps
python3.12 scripts/plot_designspace_pred_vs_sim.py  # 2-D + capacitance sweeps
python3.12 scripts/plot_single_chip.py           # per-chip spatial droop map
python3.12 scripts/plot_design_choice.py         # design trajectory / convergence / atlas
```

---

## 1. Predicted vs true droop — the headline

![Predicted vs true droop](figures/fig_pred_vs_true.png)

In-distribution (left, middle) the predictions collapse onto `y = x`. On the
**unseen** topology (right) the cloud fans out but stays tightly correlated and
monotonic — high-droop designs are still predicted high. This single figure is
the "does it work?" answer. → [PREDICTION_ANALYSIS.md](PREDICTION_ANALYSIS.md)

## 2. One chip, laid out — predicted vs actual at every load

![Single-chip droop map](figures/fig_single_chip_droop.png)

The most intuitive view: a single chip (held-out n_top = 4) with all 14 load
sources at their physical mesh locations, colored by droop — predicted vs
actual, same scale — plus the per-load scatter. The spatial droop gradient is
reproduced even though the model sees **no coordinates**, and the worst load
(the spec-setting one) is nearly exact. → §4.1 of the prediction analysis.

## 3. Correlation vs precision — error by droop magnitude

![Error vs magnitude](figures/fig_error_vs_magnitude.png)

Absolute error grows with droop, but **relative** error is worst at the shallow
tail (~15%) and best at deep droop (~6%) — i.e. the model is most trustworthy
exactly where droop is large and decisions are made. → §2 of the prediction
analysis.

## 4. Inferred vs actual droop — surrogate vs simulator

![Inferred vs actual scatter](figures/fig_inferred_vs_actual.png)

A direct surrogate-vs-simulator scatter along a wire-width sweep. In-distribution
points sit on `y = x`; OOD points ride **above** (conservative / safe) and peel
away at the low-droop end, where the surrogate refuses to follow the simulator
down. → §4 of [GENERATION_ANALYSIS.md](GENERATION_ANALYSIS.md).

## 5. Where the surrogate fails — non-monotonicity on the unseen topology

![Surrogate vs sim sweep](figures/fig_surrogate_vs_sim_sweep.png)

The single most diagnostic plot of the model's weakness: on n_top = 4 (left) the
surrogate is **non-monotonic** in wire width — it predicts droop bottoms out and
turns *up* with more copper, which is unphysical and causes one false-infeasible
verdict. On n_top = 3 (right) it tracks the simulator perfectly. → §4 of the
generation analysis.

## 6. Predicted vs actual across the full 2-D design space

![Predicted vs actual across the 2-D design space](figures/fig_designspace_pred_vs_sim.png)

Both knobs swept (wire_width × C_decap), predicted vs actual droop as paired
heatmaps plus signed error. In-distribution (bottom) the maps are identical;
OOD (top) the error concentrates in a fat-wire blob (~0.10 mV, one-signed /
conservative). A companion [capacitance sweep](figures/fig_cap_sweep_pred_vs_sim.png)
shows the distortion is **wire-axis-specific** — along decap the OOD surrogate
is monotonic and only mildly conservative. → §4.1 of the generation analysis.

## 7. The design the optimizer chooses — descent over the droop landscape

![Design-choice trajectory](figures/fig_design_trajectory.png)

For each droop budget, the gradient-descent path through `(wire_width, C_decap)`
space over the predicted-droop contours, with the spec boundary in cyan and the
chosen design as ★. It lands exactly on the cheapest feasible point when the
budget is reachable, and stalls above the boundary when it isn't. Companion
[convergence trace](figures/fig_design_convergence.png) and
[design atlas](figures/fig_design_atlas.png). → §3.6 of the generation analysis.

## 8. Surrogate fidelity at the design optimum

![Surrogate fidelity at optimum](figures/fig_surrogate_fidelity.png)

The most important plot for *design* use: at the points the optimizer actually
lands on, in-distribution predictions sit on `y = x` and OOD predictions sit in
the conservative (safe) zone. Validates the model where it's used, not just on
average. → §3.1 of the generation analysis.

---

*Secondary / supporting figures (design-space error map, per-site breakdown,
residual structure, recovered-knobs-vs-budget, topology difficulty) live in
[`figures/`](figures/) and are referenced inline from the two analysis docs.*
