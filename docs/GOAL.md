# Goal

## Long-term: generative model for analog / transistor-level circuits

The end target is a generative model over transistor-level analog circuits
(filters, references, amplifiers, on-chip mixed-signal). That sits behind
two hard problems at once:

- the design space is large and continuous-discrete mixed (topology +
  device sizing + bias);
- the cost function (SPICE / signoff) is expensive, so a learned surrogate
  must be useful inside an outer optimization loop.

We are not solving that directly yet. We are using power-delivery networks
(PDNs) as an intermediate, smaller-scope problem that exercises the same
generation flow.

## This repo: PDN regression as the intermediate milestone

A regular PDN is a useful intermediate because:

- It has a **small, mostly-regular design space** — a handful of physical
  knobs (sheet resistance, wire width, decap, supply pattern, switching
  load) instead of a full transistor netlist.
- The **ground truth is cheap and clean** — a sparse linear solve per
  timestep, no convergence drama, no model files. We can generate
  large, statistically-controlled datasets.
- It is **practically useful on its own** — fast IR-drop / droop
  surrogates are something signoff teams care about, so positive results
  here have a separate story.
- It exercises the **same scaffolding** we want for transistor circuits:
  heterogeneous graph encoder, VAE latent split, a regression head, an
  autoregressive head for generation.

The plan is explicitly to **keep the design parameter count small** —
# metal tracks, total caps, wire width — and assume a regular grid; if
the generation flow works here, that is the green light to scale up to
transistor-level. If it doesn't, the iteration loop is much shorter than
retooling for SPICE.

## Concretely, what counts as "this works"

1. Forward task — peak droop regression on a held-out test split: < 1 mV
   MAE on the worst-node droop, R² > 0.95 on per-(sample, node) droop.
2. Encoder is **topology-aware**: trained jointly across pad patterns
   (`corner`, `checker`), the model recovers a different droop map per
   pattern from the same continuous parameters — i.e. it has actually
   learned the supply geometry, not just memorized a fixed grid.
3. Latent space has structure: PCA / nearest-neighbor in latent recovers
   neighbors in design space (similar Rsheet, wire width, etc.). Required
   before we touch the generative head.
4. Generation head, decoder-side: given a target peak-droop spec, the
   decoder produces a parameter vector inside the design ranges whose
   resimulated droop matches the target to within forward-task error.

Hitting 1 + 2 means the regression / encoder side is healthy. 3 + 4 means
the generative scaffolding works on this small problem and is worth
porting to the transistor-level setting.

## What is intentionally out of scope right now

- **Variable grid size.** `n_top`, `n_bot` are fixed at (4, 7). Varying
  them changes the y-vector size and forces a refactor of the regressor
  output; not worth doing before the fixed-size case is validated.
- **VSS mesh / package & bump RL.** The current model is an RC Vdd
  network with ideal ground. That is fine for static IR drop and the
  slow part of droop, but misses high-frequency package resonance. We
  add this only if the fixed-size results are good enough that high-freq
  fidelity becomes the next limiter.
- **Irregular grids** (random strap drop-outs, non-uniform pitches).
  Same reason — regular topology first, irregular only after the regular
  case is validated.
- **Real-design parsing** (DEF / LEF / industry benchmarks). Synthetic
  regular grids are enough to iterate on the model and the generation
  pipeline.

## Iteration discipline

When something here surprises us — model fits perfectly, model fails to
fit, a parameter knob has no effect — the question we answer is "what
does this tell us about the transistor-level case?" rather than
"how do we squeeze another mV of accuracy on this benchmark."
The PDN problem is a vehicle, not the destination.
