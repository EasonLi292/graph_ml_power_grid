# Design note — one-shot impedance-factorized global attention

Replaces depth-dependent local message passing with a single global
attention layer. Research question:

> Can one shared, directional, impedance-factorized attention layer provide
> global circuit interaction and transferable component sensitivities
> without deep local message passing, explicit all-pairs edges, or manually
> encoded node-type rules?

Written before implementation. Every physics claim below was verified
numerically first (`scripts/probes/impedance_attention_checks.py`
reproduces them); the numbers quoted are measured, not derived on paper.

---

## 0. Why the current architecture fails, in one line

Depth *is* the receptive field: 7 layers cannot reach across a 13×13 die
(measured: distant-edge sensitivity magnitude ratio **0.00**), and 20
layers that do reach over-smooth the small die (measured: sign accuracy
0.41 at (4,7) vs 0.86 at 7 layers). The required depth scales with circuit
diameter, so no fixed depth serves both. See `docs/SOBOLEV_RESULTS_REVIEW.md`.

## 1. Tensor shapes

Per graph: `n` electrical nodes (**all** original nodes retained — no Kron
reduction, no elimination), `L` load nodes appended to the same flat list,
so the node axis has `N = n + L` entries. Batched graphs concatenate along
the node axis with a `graph_id` vector (no dense padding).

| symbol | shape | meaning |
|--------|-------|---------|
| `x` | `[N, d_x]` | node features (§3) |
| `h` | `[N, d_h]` | hidden state after input MLP, `d_h = 64` |
| `q`, `k` | `[N, H, d_qk]` | content query / key, `H = 4`, `d_qk = 4` |
| `v` | `[N, H, d_v]` | content value, `d_v = 32` |
| `p` | `[N, C, m]` | **observer** impedance factor, `C = 1 + 4(F-1)` channels, `m = 16` |
| `s` | `[N, C, m]` | **source** impedance factor |
| `phi`, `psi` | `[N, H, C]` | learned per-head channel gains (§5) |
| `q̃` | `[N, H, d_qk·C·m]` | `q ⊗ (phi·p)` |
| `k̃` | `[N, H, d_qk·C·m]` | `k ⊗ (psi·s)` |
| `cache` | `[H, d_qk·C·m, d_v]` | the KV cache — **independent of N** |

`F = 3` frequencies → `C = 9` channels: one at DC, **four** per non-zero
frequency. Four is required, not padding — see §5.
`d_qk·C·m = 4·9·16 = 576`; cache is `4·576·32 ≈ 74 k` floats per graph.

## 2. The factorized attention equation

Conceptually
```
a_(i←j) = (q_iᵀ k_j) · (p_iᵀ s_j)
out_i   = Σ_j a_(i←j) v_j                        (no softmax)
```
Because the score is a product of two inner products it is a single inner
product in the Kronecker space:
```
q̃_i = q_i ⊗ p_i ,  k̃_j = k_j ⊗ s_j
a_(i←j) = q̃_iᵀ k̃_j
```
so the sum over `j` is evaluated as a cache, never as pairwise scores:
```python
cache = einsum('nhd,nhe->hde', k_tilde, v)     # [H, d_qk*C*m, d_v]
out   = einsum('nhd,hde->nhe', q_tilde, cache) # [N, H, d_v]
h'    = LayerNorm(h + W_o · out)
```
Both contractions are linear in `N`. **No `[N, N]` tensor is ever
materialised.** The naive `[N, N]` path exists only inside the test suite,
guarded by an explicit `naive=True` flag, for equivalence checking on tiny
graphs.

Directionality is structural: `q̃_iᵀ k̃_j ≠ q̃_jᵀ k̃_i` because `Q ≠ K` and
`phi ≠ psi` are independent projections. Nothing enforces or forbids
symmetry — it is learnable in both directions.

## 3. Loads as nodes, with oriented terminals

Every load is a node in the same list as electrical nodes, carrying
```
x_load = [I_peak, freq, duty, sin 2πφ, cos 2πφ, type_onehot…]
```
Orientation is stored explicitly as `load_terminals [L, 2] = (a_vdd, b_vss)`
and enters the factors as an oriented difference:
```
p_load = p_a − p_b        s_load = s_a − s_b
```
(sign convention: VDD terminal minus VSS terminal, applied identically to
both factors). Electrical nodes get `type_onehot` distinguishing
mesh-top / mesh-bot / pad; loads get their own type bit. **One shared
encoder and one shared `Q`/`K`/`V`/`phi`/`psi` set for all node types** —
no per-type parameter blocks.

The load node *is* the prediction site: the decoder reads its post-attention
hidden state and emits that load's droop. No endpoint concatenation head.

**Why this is exactly right, not merely convenient.** For a load injecting
`I_ℓ` out of terminal `a` into terminal `b`, superposition gives
```
droop_ℓ = Σ_ℓ' (u_ℓ · u_ℓ') I_ℓ' ,    u_ℓ = F_a − F_b
```
i.e. the target *is* an unnormalised attention sum over load nodes with the
oriented-difference factor as the score and current as the value. Verified
against the DC solver: **max rel err 2.0e-11** (`claim2` in the checks).
So the architecture contains the exact physics as a special case
(`phi = psi = 1`, one head, `v = I`) — that is the initialisation.

## 4. How R, C, L reach the factors

Build the nodal admittance at frequency `ω` from the branch list
(incidence `B [E, N_free]`, branch admittance `w`), clamped pads removed:
```
Y(ω) = Bᵀ diag(w(ω)) B ,   w = 1/R  (straps, vias)
                               jωC   (decaps)
                               1/(jωL) (package inductors, IBM)
Z(ω) = Y(ω)⁻¹
```
Factors come from **randomized subspace iteration** — one uniform
mechanism at every frequency, no hand-designed time constants:
```
X  = solve(Y, Ω)                 Ω [N_free, m] fixed random per graph
repeat q times:  X ← solve(Y, X)          (q = 2 power iterations)
Qr = qr(X).Q                     [N_free, m]
T  = Qrᵀ solve(Y, Qr)            [m, m]  small and dense
p_i = Qr_i                       s_j = (Qr Tᵀ)_j        →  p_iᵀ s_j ≈ Z_ij
```
QR of a complex matrix returns a **unitary** `Qr` (`Qr^H Qr = I`), *not* a
complex-orthogonal one — measured `‖QrᵀQr − I‖ = 0.94` at `ω>0` — so the
Galerkin projection must use the conjugate transpose (`T = Qr^H Z Qr`,
`s = conj(Qr) Tᵀ`). Plain transposes stay exact at DC, where `Y` is real,
and give ~100 % error at every `ω>0`.

Clamped pads get `p = s = 0` (they are voltage sources, zero transfer
impedance) — physically correct, not a mask hack.

Measured accuracy (droop reconstructed from rank-`m` factors vs the DC
solver, `q = 2`):

| anchor | n_free | m=16 | m=32 | optimal rank-m |
|--------|--------|------|------|----------------|
| (3,7)  | 55  | 1.4 % | 1.0 % | 1.6 % / 0.8 % |
| (13,13)| 325 | 2.3 % | 1.7 % | 1.9 % / 1.6 % |

Two properties that matter:
1. It reaches **optimal rank-`m` accuracy** (randomized ≈ eigendecomposition),
   so `m` is not wasted.
2. The error is **invariant to grid size** — 2.3 % at `n_free = 325` vs
   1.4 % at 55 with the same `m = 16`. This is the O(1)-reach property that
   depth cannot provide: factor dimension is set by the spectrum, not the
   diameter.

Frequencies `ω_f` are **learnable** (stored as `log ω`, `F = 3`, initialised
at `{0, ω_load, 5ω_load}`). At `ω = 0` capacitors contribute nothing, so a
DC-only factorisation has **exactly zero** gradient to `C`; `ω > 0` channels
are what make decap sensitivities exist at all. Rejected alternatives, both
measured: random JL sketching (66 % droop error at m=16 — the
`tools/impedance_sketch.py` stage-0 approach) and Galerkin projection onto
the DC eigenbasis (30–60 % error, because decap shifts `Z` by >100 % at the
operating frequency).

## 5. Impedance term and basis invariance — the binding constraint

The factor coordinates are defined only up to an orthogonal mixing (`Ω` is
random per graph; degenerate spectra rotate freely). Inner products
`p_i·s_j` are invariant; **any learned map on raw factor coordinates is
not**, and violating this silently destroys cross-grid transfer (learned on
stage 0). Therefore the factors enter only through invariant scalars, and
directionality is carried by feature-dependent *scalar* gains:
```
p_iᵀ s_j  ≡  Σ_c phi_c(x_i) · psi_c(x_j) · (p_i^c · s_j^c)
```
Each `(p_i^c · s_j^c)` is invariant; `phi ≠ psi` makes the operator
directional; both are learned from `x`, so source/observer/active/passive
behaviour is inferred from features rather than hard-coded.

**Why four channels per frequency, not two.** `Z(ω)` is complex, and both
parts carry signal (`|Im Z| / |Re Z| ≈ 0.30` at the load frequency — this is
the phase information the transient target depends on). Expanding the
complex product:
```
Re(Z) = <p_re, s_re> − <p_im, s_im>          (channels rr, ii)
Im(Z) = <p_re, s_im> + <p_im, s_re>          (channels ri, ir)
```
Emitting only the diagonal pairings `rr` and `ii` — the obvious encoding —
leaves `Im(Z)` **unrepresentable**: fitting it from `{rr, ii}` by least
squares leaves 44 % residual. So each non-zero frequency emits all four
pairings, and the learned gains recover either part. DC emits one channel
(`Y` is real there). Scores are divided by a per-graph
invariant scale (mean of `|p_i·s_i|` over loads) for conditioning.

## 6. Gradient paths

Every path stays inside autograd — `torch.linalg.solve` / `qr` on tensors
built from the knobs, **no NumPy/SciPy step and no cached detached sketch**:

| target | path |
|--------|------|
| resistor `R_e` (near *and* far) | `R → w → Y → solve → p,s → score → droop` |
| capacitor `C_k` | same, but only via `ω>0` channels (zero at DC by construction) |
| load waveform (`I_peak`, duty, …) | `x_ℓ → v_ℓ` (values), and `→ q,k,phi,psi`. Loads are ideal current sources so they correctly do **not** enter `Y` |
| frequencies `ω_f` | learnable, `→ Y(ω) → factors` |

Distance is irrelevant to the gradient path: a far resistor and a near one
both reach the output through exactly one linear solve, so gradient
magnitude does not decay with circuit diameter. That is the core claim the
sensitivity gate will test.

`tools/impedance_sketch.py` (SciPy `splu`) **breaks** gradients and is kept
only as a forward-only debug reference; it is not used by this model.

## 7. Complexity

| stage | time | memory |
|-------|------|--------|
| factor construction (prototype, dense) | `O(F · N_free³)` | `O(N_free²)` |
| factor construction (scalable path) | `O(F · q · m · solve)` | `O(N_free · m)` |
| attention | `O(N · H · d_qk · C · m · d_v)` | `O(N · H · d_qk · C · m)` |
| decoder | `O(L · d_h²)` | `O(L · d_h)` |

The attention is strictly linear in `N` and never allocates `[N, N]`.

## 8. Unresolved

1. **The dense solve is the only non-linear-in-`N` step.** The prototype
   uses `torch.linalg.solve` on a dense `Y` (`N_free ≤ 325` on this track →
   milliseconds), which does allocate `O(N²)`. `Y` is sparse (≈4 nnz/row),
   and the algorithm above only ever needs `solve(Y, ·)` against `m`
   right-hand sides — so the scalable replacement is a sparse iterative
   solve with an implicit-adjoint `autograd.Function` (the adjoint of a
   linear solve is another solve with `Yᵀ`). This is understood but **not
   implemented**; it is what gates running on IBM-scale grids (25 k–1.7 M
   nodes). The prototype's *attention* already satisfies the constraint;
   its *factor construction* does not.
2. **Rank vs. transient target.** Factors are exact for the linear
   frequency-domain response; the label is a *peak over time* of a pulsed
   transient. `F = 3` learnable frequencies is a compact stand-in whose
   sufficiency is an empirical question. If it under-fits, the principled
   next step is more channels, not deeper local stacks.
3. **Reciprocity.** For RLC networks `Z` is symmetric (a theorem, not an
   assumption), so the *physical* factor satisfies `p = s` up to the small
   `T`. Asymmetry in the operator comes entirely from learned `phi/psi` and
   `Q/K`. If genuinely non-reciprocal elements ever appear, `p` and `s` are
   already separate tensors and nothing needs restructuring.
4. **Batching.** Factors are per-graph (each graph has its own `Y`).
   Batched execution loops over graphs for the solve, then does one fused
   attention pass over the concatenated node axis.

## 9. Synthetic vs. IBM as the training target

Recommendation: **do the falsifiable experiment on the synthetic track,
keep IBM as a later forward-only reach check.**

- The objective (`docs/OBJECTIVE.md`) is *repair*, which needs design knobs
  and sensitivity labels. IBM grids have no knobs — they can validate a
  forward model but cannot test the thing being built.
- The failing acceptance test (sensitivity gate) and the exact adjoint
  Jacobians both live on the synthetic track.
- Stage 1 showed IBM forward droop is already ~solved by cheap physics
  (timing-QS floor 0.846/0.793) and no learned model beat it — so IBM is a
  poor discriminator of architecture quality.
- IBM's 25 k–1.7 M nodes are blocked on unresolved item 1 anyway.

IBM becomes valuable *after* the sparse solver lands, as the extreme-diameter
test of the O(1)-reach claim — validation, not primary training.

## 10. What gets built

New, additive — existing 7-layer and 20-layer baselines untouched:

- `tools/impedance_factors.py` — differentiable branch assembly + factors
- `eason/impedance_attention_model.py` — encoder, one attention layer, decoder
- `scripts/probes/impedance_attention_checks.py` — the 6 required tests
- `scripts/train_impedance_attention.py` — training + baseline comparison

Falsification criteria, per anchor, at ≥12 gate designs:
- **(7,13)**: non-zero distant sensitivity — magnitude ratio ≫ 0 (7-layer: 0.00)
- **(4,7)**: no over-smoothing — sign ≥ 0.86, ρ ≥ 0.65 (20-layer: 0.41 / −0.18)
- gradient magnitude to a fixed-rank component must not decay with diameter

---

## 11. Prototype status — measured

All seven checks pass (`python scripts/probes/impedance_attention_checks.py`):

| check | result |
|-------|--------|
| 1 physics anchor | `Z = p·s` to **8e-16**; DC droop identity to **2.6e-12** |
| 1 rank-16 factors | droop err **1.88 %** at (3,7) `n_free=55`, **1.91 %** at (13,13) `n_free=325` — **diameter-invariant** |
| 2 naive equivalence | output **1.5e-16**, gradients **1.2e-15** vs explicit `[N,N]` |
| 3 permutation equivariance | **3.0e-16** |
| 4 directionality | operator asymmetry **1.66**, while the DC physical channel is symmetric to **5.0e-16** — asymmetry comes only from learned `Q/K`, `phi/psi` |
| 5 load orientation | terminal swap negates the factor **exactly** (0.0), other loads untouched |
| 6 no-n² / scaling | attention peak memory **0.002 MB, constant** from N=72→390; time/node growth **1.16×** (linear = 1.0, quadratic = 5.4) |
| 7 gradients | near R, **far R**, top R, **C**, load `I_peak` all match finite differences to **≤2e-6**; none zero |

Distant-component gradient at (7,13), where a 7-layer stack is structurally
blind: far/near ratio **0.028** untrained, against a true simulator ratio of
**0.0016** — small but *live*, versus exactly **0.0** for the local baseline.

Smoke training (800 samples = 5 % of the train split, 20 epochs, CPU,
3 s/epoch) reaches held-out **R² +0.41 at (4,7) and +0.49 at (7,13)
simultaneously** — the combination neither depth setting achieves (7-layer
is blind at (7,13); 20-layer collapses to −0.18 at (4,7)). Full-data runs
are the handoff; extrapolated cost ≈ 60 s/epoch on 16 k samples.

**Not yet done** (needs the full training run, launched by the user):
the forward-baseline table across all five configurations and the
sensitivity gate at ≥12 designs. The gate is wired
(`scripts/sensitivity_gate.py --arch impedance`) and rebuilds factors under
autograd on every perturbation, so the gradient path it measures is the
real one, not a cached surrogate.
