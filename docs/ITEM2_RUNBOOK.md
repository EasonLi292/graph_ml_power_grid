# Item 2 runbook — full synthetic training + 12-design sensitivity gate

The decisive experiment for `docs/IMPEDANCE_ATTENTION_DESIGN.md`: can one
global impedance-attention layer replace depth-dependent message passing?
Everything below is tested locally end to end; nothing was launched here.

Prereqs on the box: `datasets/regular_v7_anchors/dataset.h5` (gitignored,
already rebuilt there per `docs/SOBOLEV_RESULTS.md`) and the existing
`checkpoints/droop_v7_edgeconv*.pt` baselines.

## 0. Warm the factor cache FIRST — do not skip

All three attention configs share one factor cache keyed by
`(m, n_power, n_freq, n_channels)`. Launching them concurrently against a
cold cache makes all three compute the same factors and race on the write.
Run one short job first, then fan out:

```bash
python scripts/train_impedance_attention.py --epochs 1 --device cpu
```

Precompute is ~10 ms/sample single-threaded (~3 min for 16 k on one core,
seconds across 96). It is CPU-bound dense linear algebra.

## 1. The five forward baselines (run in parallel after step 0)

```bash
# a) combined  — the hypothesis
python scripts/train_impedance_attention.py --ablation combined --epochs 50 \
    --ckpt checkpoints/imp_attn_combined.pt

# b) content-only  — no impedance term
python scripts/train_impedance_attention.py --ablation content --epochs 50 \
    --ckpt checkpoints/imp_attn_content.pt

# c) impedance-only — no learned content term
python scripts/train_impedance_attention.py --ablation impedance --epochs 50 \
    --ckpt checkpoints/imp_attn_impedance.pt

# d) + e) the existing local baselines, SAME metric, no retraining
python scripts/train_impedance_attention.py \
    --compare-baseline checkpoints/droop_v7_edgeconv_l0.pt --n-layers 7
python scripts/train_impedance_attention.py \
    --compare-baseline checkpoints/droop_v7_edgeconv_deep20_l0.pt --n-layers 20
```

~65 min per attention config on CPU (~80 s/epoch × 50). **The H100 buys
little here** — each graph is ≤390 nodes at hidden dim 64, so the loop is
kernel-launch-latency bound, not FLOP bound. Parallelism across configs and
cores is the win; run the three concurrently.

## 2. The gate at 12 designs

```bash
for A in combined content impedance; do
  python scripts/sensitivity_gate.py --arch impedance \
      --ckpt checkpoints/imp_attn_$A.pt --n-designs 12 \
      --out docs/analysis/sensgate_imp_$A.json
done
```

The gate rebuilds impedance factors under autograd for every perturbation
(no cache), so the gradients it measures are the real ones. Slower than the
local-GNN gate — budget a few minutes per checkpoint.

## 3. Metric warning — read before comparing

`train_impedance_attention.py` reports **per-anchor worst-load R²**. That is
*not* the number in `docs/SOBOLEV_RESULTS.md` (0.793 / 0.905), which is
pooled across both test anchors and computed per-load. The two are not
comparable. Measured locally, the same 7-layer checkpoint scores:

| metric | (4,7) | (7,13) |
|--------|-------|--------|
| per-anchor worst-load R² (this harness) | −1.64 | −0.23 |
| pooled per-load R² (older docs) | 0.79 (both anchors together) | |

Always compare via `--compare-baseline`, which puts both architectures
through the identical metric. Never quote a number from one harness against
a number from the other.

## 4. Decision criteria — stop and look before items 3/5

The hypothesis is confirmed only if the combined model, **per anchor**:

1. **(7,13)** — non-zero distant sensitivity: gate magnitude ratio ≫ 0
   (7-layer baseline: exactly 0.00) and sign well above the ~0.72 trivial
   "widening always helps" rate;
2. **(4,7)** — no over-smoothing: sign and ρ at or above the 7-layer level
   (0.86 / 0.65 at 12 designs), *not* the 20-layer collapse (0.41 / −0.18);
3. both simultaneously, since that combination is exactly what no fixed
   depth achieves.

Ablations tell you *why*: if `content` ≈ `combined`, the impedance factors
are not paying rent and the physics term is decorative; if `impedance` ≈
`combined`, the learned content term is redundant.

Gate CIs are now reported and PASS requires the CI **lower** bound to clear
the bar — do not read a point estimate as a result.

## 5. Preliminary signal (smoke only — not a result)

800 training samples (5 % of the split), 8 epochs, corrected complex
factors: per-anchor worst-load R² **+0.53 at (4,7)** and **+0.55 at (7,13)**
simultaneously, against the fully-trained 7-layer baseline's −1.64 / −0.23
on the identical metric. Encouraging and on-thesis, but it is 5 % of the
data and a single seed — it is a reason to run item 2, not a substitute for
it.
