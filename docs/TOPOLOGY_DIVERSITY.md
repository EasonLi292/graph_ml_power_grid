# Topological diversity, not sample count, was the binding constraint

Training set went from **4 topologies** to **10**. This records what that
bought, what it cost, and the selection defect that motivated it.

Everything below is seed 0, single runs, forward accuracy only. It is not
the sensitivity gate — the repair loop consumes *derivatives*, and none of
these numbers speak to those.

## The defect: val was anti-correlated with what we care about

The v7 split holds out **samples** of the four training topologies for val,
and **topologies** for test. Those turn out to point in opposite directions.
Measured over 50 epochs on three architectures:

| arch | corr(val, OOD test) | best OOD epoch | val-selected OOD |
|---|---|---|---|
| bilinear | **−0.290** | 1 | +0.110 |
| kernel n_rff=128 | −0.034 | 2 | −0.044 |
| kernel n_rff=512 | **−0.411** | 1 | +0.058 |

Bilinear's full curve: val climbs +0.675 → +0.969 while held-out-topology
test **falls** +0.755 → +0.117. Selecting on val costs **0.645 test R²**
against stopping at the OOD optimum.

Epoch 1 is 16 000 gradient steps (batch size 1), so this is not an
undertrained artifact. Topology memorisation sets in almost immediately,
and every subsequent epoch buys in-topology fit at the direct expense of
cross-topology transfer. That all three architectures show it is the tell:
it is a property of the training set, not of any model.

**Consequence for prior work.** Any architecture comparison made at the
val-selected epoch on v7 — including the item-2 comparison — compares two
models chosen by a criterion pointing the wrong way. On v7 at *best-OOD*
epoch the ranking is bilinear +0.755, kernel512 +0.724, kernel128 +0.569:
the kernel score does **not** beat the bilinear one there.

## The fix: 4 → 10 training topologies

The 6-anchor ceiling was structural. Only `n_bot ∈ {7, 13}` had a Vdd/Vss
column pattern, and each die size admits exactly three valid `n_top` under
via alignment plus cluster-tap coverage. The two hand-written patterns are
one periodic family (`n_bot = 6k+1`), so generating them instead of listing
them unlocks 18 anchors up to `n_bot=37`. The generator is asserted to
reproduce both shipped tuples exactly.

- **Train (10):** (3,7) (7,7) · (5,13) (13,13) · (7,19) (10,19) (19,19) ·
  (9,25) (13,25) (25,25)
- **Test (4):** (4,7) (7,13) — unchanged, so prior results stay comparable
  — plus **(11,31) (31,31)**, pure size extrapolation. Nothing in training
  exceeds 1250 electrical nodes; (31,31) has 1922.

Sim cost is not the constraint (0.03 s at (13,13), 0.15 s at the largest).
The factor cache is: 0.24 → 0.71 MB per training sample.

## What it bought

### 1. Selection is no longer actively misleading

| | corr(val, old-2 test) |
|---|---|
| v7, 4 topologies | −0.290 / −0.034 / −0.411 |
| v8, 10 topologies, no holdout | **+0.640** |

### 2. Size extrapolation works — and it comes from the big training grids

| arm | trains up to | (11,31) | (31,31) |
|---|---|---|---|
| kernel512, all 10 anchors | n_bot=25 | **+0.872** | **+0.899** |
| kernel512, holdout n_bot=25 | n_bot=19 | +0.473 | −0.731 |
| bilinear, holdout n_bot=25 | n_bot=19 | −0.020 | −1.114 |

R² +0.90 on a 1922-node grid having never trained above 1250. The arms that
stop at 722 nodes fail on the same grid. **This is the axis that matters for
the IBM target, and v7 could not test it at all.**

### 3. The old anchors improved too, and the peak moved later

Best-OOD on (4,7)/(7,13): +0.748/+0.762 on v7 → +0.827/+0.806 (kernel+ho),
+0.854/+0.778 (bilinear+ho). And the no-holdout arm peaks around **epoch 9**
rather than epoch 1 — a genuinely trained model, not v7's trivial early
solution.

## The mistake in the holdout design, and the correction

`--holdout-anchor` excludes topologies from training and selects on them.
Holding out the **largest** trained die size (n_bot=25) makes the selection
signal match the test axis beautifully — corr(val, extrapolation) = **+0.941**
(kernel) and **+0.966** (bilinear).

But it removes exactly the training data that makes extrapolation possible.
Those arms are honest and crippled: −0.731 and −1.114 on (31,31).

The correction is to hold out a **middle** die size: train on {7, 13, 25},
select on unseen 19, test on unseen 31. Training keeps the largest grids so
extrapolation survives, and selection still measures generalization to an
unseen size. Running as arms D/E.

Given corr(val, old-2) = +0.640 with 10 topologies and no holdout at all,
plain val-based selection may now be good enough — the holdout is belt and
braces, and it is not free.

## Open

- Single seeds. Retraining variance on this track has previously exceeded
  the gate's own CIs, so no arm ranking here should be believed yet.
- Kernel vs bilinear remains unsettled: bilinear wins on v7 at best-OOD,
  the v8 holdout arms are mixed.
- None of this is the sensitivity gate. Forward R² has repeatedly failed to
  predict derivative fidelity on this project.
