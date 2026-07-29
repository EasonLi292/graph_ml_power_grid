# Unified dynamic-impedance kernel

Replaces the hybrid score — a bilinear multi-frequency term plus a
separately-named Gaussian kernel of DC effective resistance — with one
learned kernel over the complete multi-frequency impedance.

Selected with `--score dynamic_kernel`. The bilinear control (`bilinear`)
and the bilinear+DC-kernel model (`kernel`) are untouched.

## The score

For head `h`, receiver `i`, sender `j`:

```
A_h(i <- j) = Phi_h(i) . Psi_h(j)
            = sum_{c,k} alpha_h[c,k] . phi_h[c,k](i) . psi_h[c,k](j) . (z^c_ij)^k

message_i   = sum_h sum_j A_h(i <- j) V_h(j)
```

- `c` runs over the **basis-invariant impedance channels**: `Z` at DC, and
  `Re Z(w)`, `Im Z(w)` at each non-zero frequency.
- `k` runs over polynomial degrees `0..max_degree` in that channel.
- `alpha_h[c,k]` is a free **signed** per-head mixture weight.
- `phi_h`, `psi_h` are separate linear maps of the node embedding, which
  carries node features, node type, load waveform features and per-channel
  self-impedance.

Nothing is named after physics. Degree 1 on the DC channel *is* the old
bilinear score; degree 0 is a pure content score; the rest is new. Which
combination a head uses is learned. Heads are initialised at different
scales (`logspace(-1, 0, H)`) purely to break symmetry — no diversity loss,
no assigned roles.

### How each requirement is met

| requirement | mechanism |
|---|---|
| node identity | `phi`/`psi` consume the shared node embedding |
| ordered sender/receiver | `phi != psi`, so `A_h(i<-j) != A_h(j<-i)` — measured 2.1e-1 relative asymmetry while every impedance channel stays reciprocal at 1.0e-15 |
| complete multi-frequency connectivity | every invariant channel enters at every degree; mixing is `alpha` |
| no manual RC encoding | no time constant is ever formed; the model mixes frequency channels itself |
| no per-head physics | `alpha` is free and symmetric across heads at init up to a scale |

## Basis invariance — the finding that forced the design

`impedance_factors` emits four real channels per non-zero frequency:
`(re,re)`, `(im,im)`, `(re,im)`, `(im,re)`. **Those four are not
individually basis-invariant.** Measured at exact rank (`m = n_free`, so
the factorization is exact and only the random probe basis differs):

| channel | rel. change under a different probe seed |
|---|---|
| DC | 1.4e-15 |
| AC channels 1-8 | **0.19 – 0.76** |

Only the physical combinations are stable:

```
Re Z = <p_re, s_re> - <p_im, s_im>
Im Z = <p_re, s_im> + <p_im, s_re>
```

**This means the existing bilinear and kernel scores apply independent
learned gains to quantities that are artifacts of the probe draw at
`w > 0`.** Nothing forces `phi_c psi_c` to combine the four raw channels
into the two invariant ones, so generically they do not, and the AC part of
the score cannot transfer between grids. Every arm trained so far — v7 and
v8, bilinear and kernel — has this defect.

`invariant_channels()` regroups first. Each combination is still one inner
product, so O(N) is preserved, at width `2m` per AC channel instead of `m`:

```
Re Z_ij = <[p_re_i, p_im_i], [ s_re_j, -s_im_j]>
Im Z_ij = <[p_re_i, p_im_i], [ s_im_j,  s_re_j]>
```

Verified: all 5 invariant channels agree to ~1e-15 across probe seeds.

A second leak was found and fixed: `normalize_factors` averages `(p*s)`
over **all raw channels**, so the normalising scalar was itself
basis-dependent. It leaked 1.4e-3 of drift into the model output at exact
rank. The dynamic kernel normalises on the DC channel only, which restores
3.2e-11.

## A known limitation of the factorization, not of this score

Physical `Z(w)` is reciprocal, and the DC factors are symmetric at any
rank. **The reduced-rank AC factors are not:**

| rank m | max AC reciprocity error |
|---|---|
| 8 | 1.86 |
| 16 | 1.52 |
| 32 | 0.62 |
| 55 (exact) | 1.8e-15 |

`Z(w)` is complex *symmetric* while `torch.linalg.qr` returns a *unitary*
basis, so the Galerkin projection `Qr T Qr^H` does not preserve complex
symmetry. This predates the dynamic kernel and applies to every arm that
uses AC channels. It is a candidate explanation for why the AC part of the
impedance score has never demonstrably helped, and it is worth fixing
independently — a complex-symmetric (`Qr^T`-based) projection or an
explicit symmetrization `(Z + Z^T)/2` on the projected core.

## Capacitance gradient

The DC-only Gaussian kernel has **exactly zero** derivative to `C_decap` —
a capacitor is an open circuit at DC, and autograd returns `None`. The
dynamic kernel's AC channels carry it: measured autograd `-7.664111e+09`
against finite differences `-7.664111e+09`, relative `8.4e-9`.

## Cost

Feature width per (channel, degree) at `m=8`, `n_freq=3`:
`[1, 8, 36] + 4 x [1, 16, 136]` = 657. At `m=16` it is 2397, comparable to
the `kernel` score's 2112. Parameters: **33,973** vs bilinear 33,129 and
kernel 33,673 — matched budget.

Measured: time growth / node growth = **0.83** over N = 72 -> 390 (1.0 is
linear), peak attention memory flat at 0.006 MB versus 1.2 MB for an
`[N,N]` tensor at N=390.

## Checks

`scripts/probes/dynamic_kernel_checks.py` — 9/9 passing:

1. factorized vs explicit `[N,N]`, forward: 4.6e-15
2. factorized vs explicit gradients — params 9.8e-16, `d/d ww` 6.0e-14,
   `d/d C` 3.5e-15
3. capacitance gradient non-zero and matches FD to 8.4e-9
4. permutation equivariance: 1.6e-15
5. basis invariance across probe seeds: 3.2e-11
6. directionality 2.1e-1 with reciprocity 1.0e-15
7. no `[N,N]` allocation
8. scaling sub-linear in N
9. per-head mixture observable (no specialization required to pass)

## Not yet done

Training. The comparison (bilinear / bilinear+DC-kernel / dynamic kernel)
on v8 with the middle-die holdout, seed 0 screening then seeds 1-2, and the
full metric set including the corrected sensitivity gate, has not been run.
