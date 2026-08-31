# Differentiable Silicon

**Backpropagating a class-balanced ECG-classification loss through a spiking
network, through a subthreshold neuron circuit, and into a closed-source
commercial TCAD solver — to obtain ∂L/∂(ferroelectric process parameters).**

Tesseract Hackathon 2026 · Track 3 — Hybrid ML + Mechanistic Models
Repository: <https://github.com/hozaifa1/differentiable-silicon> (Apache-2.0)

---

## 1. The claim, in one paragraph

A spiking neural network's accuracy depends on the physics of the device its
synapses are made from. That dependence is normally broken by hand: a device
engineer maximises a figure of merit, hands the numbers over, and an ML engineer
trains on whatever arrives. This project closes the loop. It backpropagates a
class-balanced cross-entropy over MIT-BIH ECG beats, through a PyTorch spiking
network, through a JAX subthreshold-DPI transducer, and into **Synopsys Sentaurus
— a closed-source binary with no adjoint, driven over SSH from a CentOS 7 host
with Python 2.7 and nothing else on it** — and returns a usable gradient with
respect to four *fabrication* parameters of a GAA ferroelectric FET. The
optimiser then descends it and finds a better device.

**Every forward value in this project is ground truth from the real solver. Only
the derivative is manufactured**, from directed finite-difference probes of that
same solver. `results/runs/provenance_audit_d6.json` is the evidence, not the
assertion: 5,910 forward evaluations, each stamped with the backend that produced
it and the sha256 of the inputs it was produced at.

## 2. What crosses what

The gradient crosses three mutually unaware AD regimes, one SSH hop, and one
closed binary:

```
        theta (4 fabrication knobs, normalised to [0,1]^4)
          |  G : R^4 -> R^7      Sentaurus / DEVSIM. NO adjoint, and never will have one.
          v                      (T1 sentaurus-fefet | T2 devsim-fefet, via T3 adjoint-shim)
        y (7 figures of merit)
          |  H : R^7 -> R^5      DPI transducer. Pure JAX, exact, free.
          v
        phi (beta, g_min, g_max, th_th, sig_w)
          |  F : R^5 -> R        LSNN classifier. PyTorch.
          v
        L (class-balanced cross-entropy)
```

Four Tesseracts, and the boundary between them is a real one — not two JAX
functions in one script:

| | Tesseract | wraps | AD | endpoints |
|---|---|---|---|---|
| **T1** | `sentaurus-fefet` | Synopsys Sentaurus 2023.12, Preisach ferroelectric | **none** (closed binary) | `apply`, `abstract_eval` |
| **T2** | `devsim-fefet` | DEVSIM 2.10 (Apache-2.0) + clean-room Miller FE gate | none | `apply`, `abstract_eval` |
| **T3** | `adjoint-shim` | trust-region FD + Broyden black-box adjoint | NumPy/JAX | + `jacobian`, `jvp`, **`vjp`** |
| **T4** | `snn-lif-ecg` | the thesis LSNN (100 LIF + 60 adaptive, delayed synapses) on 2000 MIT-BIH beats | PyTorch | + **`vjp`**, `jvp` |

**T1 and T2 publish a byte-identical frozen schema.** Swapping the closed-source
commercial solver for the Apache-2.0 one is one environment variable —
`ORACLE_BACKEND` — and a test asserts the schemas still match. Nothing downstream
can tell them apart except by reading the `backend` string that comes back. That
is a fact about the code, not a claim in a README, and it is the reason this is
built on Tesseract at all.

## 3. The reverse sweep, hop by hop

**Hop 1, PyTorch → wire → JAX.** T4's `vjp` returns dL/dphi as five float64s.
PyTorch's tape ends there; JAX picks the cotangent up with no shared autodiff
state whatsoever.

dL/dphi is obtained by **central differences of the network's own loss**, not by
autograd, and that was a measurement rather than a preference. Backpropagating
through 111 recurrent timesteps returns |g| = 1.21e6 (hard spikes) or 3.11e16
(smooth) against a finite difference of 3.86 and 1.76e5, with cosine −0.98 and
−0.00. A four-class balanced cross-entropy is bounded by ~5, so a pointwise
derivative of 3e16 is amplified floating-point noise, not a slope. Truncated BPTT
was tried first and fixes the magnitude but not the direction (best cosine +0.54
at k=2). So the same discipline the solver gets is applied here: **the forward
pass is never a surrogate; only derivatives are estimated, from evaluations that
really happened.**

**Hop 2, JAX, exact and free.** (dH/dy)ᵀ applied by `jax.vjp` in R⁷.

**Hop 3, the manufactured adjoint.** T3's `vjp` is Jᵀ ḡ_y with J ∈ R^(7×4),
maintained as: a central-difference anchor (2D+1 = 9 calls), forward-difference
refreshes (D+1), rank-one Broyden updates between them from the secant pair every
accepted step supplies free, and a trust region that forces a ground-truth
refresh when ρ < 0.25. **A repeat VJP at a fresh point costs zero solver calls**
— asserted in the container smoke test, not in prose.

We make no claim that a rank-one-updated local linear model is categorically
different from a surrogate *of the derivative*. It is one. The claim is narrower
and checkable: every loss ever reported came out of the solver at the design
point it is attributed to.

**The surrogate spike gradient is not ad hoc.** The forward Heaviside has a
written-down smooth relaxation, H_k(u) = ½(1 + ku/(1+k|u|)), and the backward
rule is *exactly* H_k′. That consistency is why the organisers' own
`check-gradients` can verify T4 at all — finite differences of a hard Heaviside
are either zero or one spike-flip large, so a hard forward pass cannot be
gradient-checked by anybody. **0 failures / 56 checks** on the shim, in CI, on
every push.

Two conditions on that number, because neither is the default and a reader who
runs the checker without them will not reproduce it. The shim reuses `J` between
calls and refreshes it with forward differences, both to save solver calls; a
point-wise checker assumes neither, and its own reference is a central
difference. `SHIM_ALWAYS_CENTRAL=1` with `SHIM_REFRESH_EVERY=1` puts the shim in
the regime the checker assumes. Under the defaults the same 56 checks report 5
failures, all on `ss` and `dg_dvth`, whose Jacobian entries move by a factor of
12 across one `alpha` — the gap is the stencil, not the gradient.

The network is a separate case. T4 must be gradient-checked under
`SNN_TRAIN_MODE=frozen`: the default `adapt` mode trains the readout inside
`apply`, so the forward is not a function of its inputs and an `eps` of 1e-6
divides run-to-run variation by a million. Measured in `adapt`, all 10 checks
fail with finite differences three orders off the returned gradient. Frozen is
what every banked result already uses, and it is what makes the VJP exact rather
than approximate.

## 4. Prior work, head-on

Pasteur Labs' own arXiv **2511.10761** (Rehmann, Haefner, Lavin) covers
differentiable-simulation-driven device co-design, and a judge will reasonably
ask whether this is a re-run of it. The distinction is one sentence and we do not
hair-split it: **our forward pass is never a surrogate.** Where that work learns a
differentiable stand-in for the simulator and optimises through the stand-in, this
optimises through the simulator itself and manufactures only the adjoint, from
probes of the same binary, with a per-iteration backend-and-hash log as the
receipt. That is the entire claimed difference, and it is checkable in
`results/runs/provenance_audit_d6.json`.

## 5. What is optimised — and what is deliberately locked

The design vector is four **fabrication** knobs: ferroelectric HZO thickness
`t_fe`, gate length `L_g`, channel doping `N_ch`, interfacial-layer thickness
`t_IL`.

Remanent polarization and coercive field are **locked** to the measured film
(P_r = 32 µC/cm², P_s = 40 µC/cm², E_c = 1.4 MV/cm, node `cal_n16`, reproducing
Liao et al. 2022 Fig. 7), and `run_flagship` **raises an exception** if handed a
design vector that exposes them.

That restriction is the point, not a limitation. You cannot order a remanent
polarization from a fab — you deposit a different film and re-calibrate. An
optimiser given P_r will widen the memory window by changing the material and
present it as a design result. **Every number here is obtained on one fixed
film.**

**A check we did not put in, and did not have to.** Pooling the 1116-step beat by
10 gives a 5.556 ms timestep, matching the frozen `dt_bio` to 1.2%. At that
timestep the reference LSNN's membrane decay is **0.6065**; the decay this
project's *device* produces — from a FeFET's subthreshold leak, through the DPI
relation τ = C_mem·SS/(ln10·I_tau) — is **0.6033**. One from an RC fit to a VO₂
circuit, one from device physics, agreeing to half a percent.

## 6. Results

### 6.1 The flagship

From a deliberately poor corner (a thin, weakly polarised film whose memory
window cannot separate the two conductance states), on DEVSIM, budget-capped by
solver calls rather than by a convergence criterion:

| | start | final |
|---|---:|---:|
| balanced cross-entropy | 1.3996 | **1.0177** |
| accuracy | 0.250 | **0.688** |
| memory window | 0.415 V | 0.576 V |
| subthreshold slope | 71.1 mV/dec | 97.4 mV/dec |
| solver calls | — | **64** |

`t_fe` 5.50 → 7.65 nm, `L_g` 52.0 → 35.4 nm, log₁₀N_ch 17.80 → 17.17, `t_IL`
1.550 → 1.374 nm. **Figure 3** draws the Id–Vg hysteresis loop at every accepted
step.

Note that SS *degrades*. To anyone who designs transistors for switching that
reads as a defect; here it is the correct trade. A thicker HZO film and a thinner
interlayer buy memory window at the cost of electrostatic control, and this
network needs window — window is what separates the two conductance states the
synapse stores. **The optimiser found that trade without being told it existed.**

**Figure 4** shows what it does to the classifier, and it is not what we expected.
The layer does *not* fire more: population rate 0.4344 → 0.4572, per-neuron rate
correlation 0.9999. **One spike in eleven moves (8.9%)** — and on the start device
the network answers class F for *all sixteen beats*. Accuracy 0.250 is exactly
the score of a constant answer. It is not a weak classifier, it is a collapsed
one, and moving one spike in eleven is what unsticks it.

### 6.2 The strongest objection, and its own proposal

*"Why not optimise phi freely and then build the device nearest to it?"* Measured
over 192 devices sampled across the design box:

| strategy | solver calls | loss | accuracy |
|---|---:|---:|---:|
| free phi\* — **impossible to build** | 0 | **1.0033** | 0.750 |
| nearest real device to phi\* (the objection's proposal) | 192 | **1.1128** | 0.688 |
| best of 192 devices spread across the box | 192 | 1.0221 | 0.688 |
| **descend through the solver (this project)** | **64** | **1.0177** | 0.688 |

Two claims, in this order.

**First: the objection's own proposal is the worst strategy on the board.**
1.1128, for 192 solver calls. Every other row beats it. Chasing an unreachable
target drags you to a corner of the design box.

**Second: sample efficiency.** 64 calls found a device better than 192
evenly-spread ones.

**And we say plainly that free (1.0033) beats joint (1.0177).** It has five
unconstrained numbers against our four physical knobs; it is a **lower bound that
nothing physical can reach**, and it is *unbuildable*. Across 192 devices two of
its five coordinates lie outside the reachable range entirely, and it wants
g_max/g_min = **1.03** where real devices span 2.2 to 6.5e7. A ferroelectric
memory whose two states conduct alike is not a memory. **Figure 1** measures the
reachable set: two principal directions carry **90.5%** of the variation across
192 devices — four knobs, about two effective dimensions — and phi\* sits **13.5
typical device-spacings** off that sheet.

**The caveat that travels with it:** "nearest in a standardised Euclidean metric"
is a naive projection. A smarter engineer would not project — they would search.
That strategy is the "best of 192" row at 1.0221, and joint descent still beats
it with a third of the calls.

### 6.3 Sample efficiency, phrased honestly

Five budgets × five arms × three seeds, 75 runs (**Figure 2**). **We do not win at
four of the five budgets.** There are three crossovers, not one: gradient descent
overtakes Latin hypercube between 20 and 32 calls, random search between 32 and
48, and a warm-started GP between 48 and 64.

The defensible claim is different and stronger: **gradient descent is the only arm
that converts extra budget into performance.**

| | 12 calls | 48 calls | change |
|---|---:|---:|---:|
| gradient descent | 1.073836 | 1.023138 | **−5.07e-02** |
| random search | 1.030783 | 1.030783 | **0** |

Random search is flat *to six decimal places* from 12 calls to 48. The
derivative-free arms saturate; this one does not. Everything about the trade
improves with dimension — a Jacobian costs 2D+1 calls while covering a box costs
exponentially many — and D = 4 is the least favourable end of that.

### 6.4 Does the middle link mean anything? (V7)

`check-gradients` proves the chain rule is *implemented*. It does not prove the
transducer means anything — three consistent but physically arbitrary maps would
pass identically. So we replaced J_H = dH/dy with a **norm-matched random matrix
in the backward pass only** (mean cosine against the truth +0.056, i.e.
orthogonal) and re-ran the same descent from the same corner on the same budget.

The loss still falls — four knobs in a box with a trust region find *something* —
but a random J_H recovers only **32% of the memory window** and **half the
accuracy**, **stalls at 40 of 64 calls** because the trust region collapses, and
**moves the interlayer in the wrong direction** (control thins it 1.550 → 1.374
nm; every ablated run thickens it to 1.645–1.666). It never finds the
channel-doping knob at all.

*Correction to our own plan:* the spec predicted the ablation signature would be
"SS driven up". Measured, **the true gradient raises SS the most** (+26.3 vs
+16.1 mV/dec) — SS is a correlate of distance travelled, not a failure signature.
The discriminators that work are accuracy, memory window, whether the run can
spend its budget, and the sign of the `t_IL` step.

### 6.5 No single figure of merit predicts performance

Re-measured on the current objective over **192 devices**:

| quantity | Pearson r | R² alone |
|---|---:|---:|
| forward threshold | −0.382 | **0.146** ← best single predictor |
| subthreshold slope | +0.362 | 0.131 |
| conductance ratio (log) | −0.351 | 0.123 |
| **memory window** | **−0.108** | **0.012** ← one percent |

All seven figures of merit in one linear model, fitted *and scored* on the same
192 points — generous to the shortcut, not to us — give **R² = 0.354**. Roughly
two thirds of what makes one device better than another for this task is not in
any linear reading of the device's own summary numbers. **That is the argument for
doing this with gradients**: if one scalar predicted performance you would
maximise that scalar and skip the pipeline. (At r = −0.108 there is no memory-window
effect to report in *either* direction; we do not claim one.)

### 6.6 The manufactured Jacobian, and how long it lasts

**Figure 5** shows the 7×4 matrix and its decay. The Broyden patch is free and the
refresh is not, so the only question is how many free steps you get. Measured
along the flagship's own path: the composed-gradient cosine is **0.70 at four
steps** — what a line search needs — and **0.43 at five**. `refresh_every = 4`
sits exactly at the last usable step.

The curve then *recovers* to 0.94, and the reason is the useful part: the model
goes stale with **distance travelled**, not with steps taken (distance predicts
the row cosine at r = −0.79; step count predicts the composed gradient at
r = −0.36). The shim already refreshes on *either* `steps_since_refresh ≥ K` **or**
`‖θ − θ_anchor‖ > radius` — and this says the distance test is the one carrying
the load.

## 7. Reproduction

**Tier A — no Docker, no licence.** `uv sync --extra snn --group dev && uv run
pytest`. 152 tests against an analytic mock (6 more skip without DEVSIM); every
wire, schema and gradient hop. Timed on a fresh clone into a fresh venv: 74 s to
sync, then 352 s cold / 314 s warm.

**Tier B — Docker, no licence.** `docker pull ghcr.io/hozaifa1/devsim-fefet`,
`tesseract serve` it, point `ORACLE_URL` at it. This is the swap-one-variable
claim, executed rather than asserted: `tests/test_tier_b_served.py` checks the
container's served schema against the frozen one, checks its figures of merit
against the values DEVSIM gave on the development machine to 1% (a different
operating system, a different BLAS underneath), refuses to pass if anything
falls back to the analytic mock, and takes `jax.grad` to `dL/dθ` with nine
container solves in the middle. The environment it runs in has no DEVSIM, no
BLAS and no licence; the solver is in the container. It is a CI job, so it runs
on every push, and the runner does `docker logout ghcr.io` before it pulls.

Writing that test found a bug on exactly the path a judge would have walked.
Every other backend writes its cache record through `encode_output`. The `url`
backend built its from the raw wire response, so the first real container call
died on `TypeError: Object of type ndarray is not JSON serializable`, *after*
the solve, which is the part you pay for. The README had also been sending
readers at `tests/test_tier_a_pipeline.py` with `ORACLE_BACKEND=url` in front of
it; that file pins the backend to `mock` in an autouse fixture, so the command
ran green and touched no container at all.

**Tier C — every Sentaurus number, no licence and no network.** Verified, not
asserted: `scripts/tier_c_replay_d6.py` blocks `socket` and `subprocess` and
strips every `SENTAURUS_*` variable *before* touching project code, then
regenerates **164 of 164 float64 values bit-identically** from
`results/cache/sentaurus/` in **1.4 s** — standing in for **0.93 h** of
commercial-solver time. If anything came off the solver it dies with a traceback
rather than returning a number.

The cache also produced an argument for its own design. Sixteen of its 32 entries
are unexercised by current artefacts, **zero are orphans**, and all sixteen hold
an Id–Vg curve identical *to the byte* while reading **different** figures of merit
off it — because `cache_key` folds in a hash of `extract.py`, and the D3
extraction rewrite really did change what a stored record means.

**Tier D — bring your own licence.**

## 8. Limitations, stated before you find them

- **The ECG split is intra-patient.** The curated beat files have record identity
  stripped during preprocessing, so an inter-patient AAMI DS1/DS2 split cannot be
  constructed from them at all. Reported accuracies also come from a deliberately
  cheap inner training loop and are **not** comparable to the thesis' fully
  trained 0.793.
- **The commercial-solver design space is narrower than the open one.** On the
  fixed-mesh T1 path only `t_fe` reaches the deck. Measured against the cache:
  `log10_N_ch` and `t_IL` are **structurally zero** (byte-identical curve,
  identical figures of merit), and `L_g` is **spuriously non-zero** — identical
  curve, but the threshold moves because the constant-current criterion is
  I_crit = 100 nA·W/L_g. So the T1↔T2 comparison is stated on the `t_fe` column
  only. `T1_REBUILD_MESH=1` closes this and is off by default and unvalidated.
- **One seed** for the correlation study, and gradient descent has zero spread
  across seeds because it starts from a fixed corner and follows a deterministic
  rule — a property, not an advantage.
- **The gradient arm slightly overran two budgets** (22 calls at a cap of 20, 33
  at 32), at precisely the budgets where it *loses*, making our reported score
  there slightly too generous.

**Two bugs we found by measuring our own claims**, both worth more than the checks
that found them. (1) The README's own flagship command did not reproduce the
flagship: every banked result used `SNN_TRAIN_MODE=frozen`, the module defaulted
to `adapt`, and nothing pinned it — 1.3152 against a reported 1.3996, with
nothing on disk saying which was which. Now pinned and serialised into
`result.json`. (2) A crashing solver could kill a whole run: the Jacobian salvaged
a probe the *extraction* refused but not one whose DEVSIM process *died*, despite
a docstring claiming both were handled alike. Both were caught by checks that
exist — a figure that refuses to draw unless it reproduces its own flagship, and a
measurement that walked a path the flagship had not.

## 9. Upstream

Two bugs found by using the toolkit rather than reading it, both drafted in
`docs/UPSTREAM.md` with the failing snippet and the fix.

1. **`tesseract_jax.apply_tesseract` crashes on any non-array output leaf** —
   `TypeError: string indices must be integers`, from a comprehension that
   unconditionally subscripts leaves the line above it explicitly skipped. One-line
   fix. The motivating case is in-repo: it is why `backend` and `content_hash`
   live in `provenance.jsonl` instead of on `OracleOutput`.
2. **Vector-valued per-parameter `eps` in `finite_differences.py`** — the scalar
   `eps` is inherited by `check-gradients`, and T4's inputs span beta ≈ 0.6 down
   to g_min ≈ 2.6e-5, where an `eps` of 1e-4 is four times g_min itself. The
   `check-gradients` docs already warn that the step has to be chosen against the
   inputs, though only in the other direction: a derivative reported as 0.0 means
   `eps` was too small. Too large is the failure this repository hits, and at
   g_min it stops being a perturbation at all.

---

### Figures

| | |
|---|---|
| `docs/figures/fig1_pca_manifold` | the reachable set, and why phi\* cannot be built |
| `docs/figures/fig2_budget_crossover` | sample efficiency against solver calls |
| `docs/figures/fig3_hysteresis_descent` | the device moving, step by step |
| `docs/figures/fig4_spike_raster` | what that does to the classifier |
| `docs/figures/fig5_jacobian_and_decay` | the manufactured adjoint, and how long it lasts |

Daily measurement logs, including every correction and superseded number, are in
`docs/D1_FINDINGS.md` … `docs/D6_FINDINGS.md`.
