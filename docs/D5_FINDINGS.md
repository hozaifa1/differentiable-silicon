# D5 findings — 2026-08-28

Plain English. Short sentences. Terms explained the first time.

Words used below, beyond D4's list:

- **phi (φ)** = the five numbers the device hands the network: membrane decay,
  two conductances, firing threshold, weight noise.
- **The reachable set** = every phi a real device in the design box can produce.
  Four fabrication knobs, so it is a thin sheet inside five-dimensional space.
- **phi\*** = the best phi if you ignore whether any device can make it.
- **Principal components** = the directions a cloud of points actually spreads
  along, in order of how much it spreads.
- **Budget** = how many times the optimiser is allowed to call the solver. It is
  the only cost that matters here; everything else is seconds.
- **J_H** = the transducer's Jacobian, dH/dy. How each of the five numbers
  responds to each of the seven the solver measured.

---

## The short version

Four things were asked for. All four are done, and one of them turned up a
mistake in yesterday's headline that is worth more than the task itself.

1. **Figure 1 is drawn** — `docs/figures/fig1_pca_manifold.png` and `.pdf`.
2. **The budget sweep located the crossover.** 75 runs, 3.04 h. There are three
   crossovers, not one: gradient descent overtakes Latin hypercube between 20 and
   32 calls, random search between 32 and 48, and a warm-started Bayesian
   optimiser between 48 and 64. The better finding is that **it is the only arm
   that converts extra budget into performance at all** — random search is flat
   to six decimal places from 12 calls to 48. Figure 2 draws it. See §3.
3. **The H-ablation says the transducer is carrying physics.** Scramble which
   figures of merit each of the five numbers responds to, keeping every response
   magnitude exact, and the optimiser recovers 32% of the memory window, half the
   accuracy, stalls at 40 of its 64 calls — and moves one of the four fabrication
   knobs in the wrong direction. See §4.
4. **All four Tesseracts smoke-test clean and all 155 tests pass** — see §5.

**And the correction, which is the important part.** Yesterday's answer to the
single strongest objection to this project was built on a free optimiser that
never actually optimised. Re-run properly, every number in that table moves, and
**the D5 gate you set — beat the projected baseline by 5% — now passes at 8.6%,
having "failed" yesterday at 0.79%.** Details in §2. Nothing about the pipeline
changed; only the baseline was being measured wrongly, and it was being measured
wrongly in a direction that *flattered the objection*, not us.

---

## 1. Figure 1 — what a device can hand the network

`docs/figures/fig1_pca_manifold.png` (300 dpi) and `.pdf` (vector).
Drawn by `scripts/fig1_pca_manifold.py` from two banked files and no solver call.

Three panels, one argument:

**(a) The sheet is thin.** Across 192 devices sampled evenly over the design box,
two principal directions carry **90.5%** of the variation (73.9% and 16.6%). Four
fabrication knobs, about two effective dimensions of freedom. This is the claim
"four knobs cannot fill five dimensions" measured rather than asserted.

**(b) phi\* is not on it.** The freely optimised phi\* sits **13.5 typical
device-spacings** from the nearest buildable device, measured in the full
five dimensions. An inset repeats the projection against the third principal
direction instead of the second, because any single 2-D view of a 5-D set can be
accused of being the one projection that made the point. It is outside in both.

**(c) And here is why it cannot be built.** Two of the five coordinates are
outside the range 192 devices spanning the whole box can reach — not near the
edge, outside:

| | phi\* | what devices can reach |
|---|---:|---|
| max conductance | 9.17e-05 S | 1.09e-04 … 8.07e-04 S |
| weight noise | 0.265 | 0.064 … 0.236 |

The mechanism is worth stating in words, because it is not subtle. **phi\* wants
g_max/g_min = 1.03.** Real devices in this box span 2.2 to 6.5e7, median 290. The
free optimum is asking for a ferroelectric memory whose two stored states conduct
almost identically. That is not a hard device to fabricate — it is not a memory
at all. The objection says "then find the device that makes phi\*". There is no
such device, and this is the reason.

---

## 2. The correction — yesterday's free arm never optimised

### What was wrong

`scripts/v6_manifold_control.py` optimises phi freely in units of the reachable
cloud's own spread, and its first restart starts at the cloud's centroid:

```python
z0 = Z[rng.integers(len(Z))] if r else Z.mean(0)
```

`Z` is standardised by that same cloud, so `Z.mean(0)` is **exactly** the zero
vector. SciPy's Nelder-Mead builds its initial simplex by perturbing each
coordinate of the start by 5% — except a coordinate that is exactly zero, which
it perturbs by `zdelt = 0.00025` instead. Every coordinate here is exactly zero.
So the initial simplex is 2.5e-4 across against a convergence tolerance `xatol`
of 1e-4. It was converged before it started, and it returned its own starting
point. The five genuine restarts that followed never beat it.

**Measured:** the phi\* in `v6_manifold_control_d4.json` equals the cloud mean to
**1.7e-14** in standardised units.

So that file does not record a free optimum. It records **the loss at the average
device**. And "phi\* sits 1.9 typical device-spacings off the sheet" is really
"the centroid of a curved cloud is 1.9 spacings from the cloud" — true, and
nearly content-free, because the centroid of any curved sheet lies off it.

This mattered because that section is the answer to the strongest objection
anybody raises, and Figure 1 was to be built on it. An objection answered with a
number that does not mean what it says is worse than an objection not answered.
Drawn with the old file, Figure 1 puts phi\* in the dead centre of the cloud —
it *disproves* the claim it was meant to support.

### What was done instead

`scripts/v6_free_refit_d5.py`. The free arm is made as strong as it honestly can
be, because a weak free arm flatters this project and we need to know before a
judge does:

- **differential evolution** over the standardised box, a global search that does
  not care where it starts (3,776 evaluations);
- then a **Nelder-Mead polish** from the winner with an **explicit
  non-degenerate initial simplex**, which is the line whose absence caused the
  failure;
- plus polishes from the cloud centroid and from the best real device, so the old
  answer is still in the running and can only be beaten, never lost.

4,519 network evaluations, no solver calls — the manifold cloud already paid for
those. The search box is the cloud's own range padded by one standardised unit
per coordinate, recorded in the output file. Free means free of the device, not
free of arithmetic; letting it run to infinity would find the corner where the
network stops being a classifier and reporting that as "the free optimum" would
be a strawman in our own favour.

### What changed

| strategy | solver calls | loss (D4) | **loss (D5, corrected)** | accuracy |
|---|---:|---:|---:|---:|
| free phi\* — impossible to build | 0 | 1.0186 | **1.0033** | 0.750 |
| build the nearest real device to phi\* | 192 | 1.0258 | **1.1128** | 0.688 |
| best of 192 devices sampled across the box | 192 | 1.0221 | 1.0221 | 0.688 |
| **descend through the solver (this project)** | **64** | **1.0177** | **1.0177** | 0.688 |
| distance of phi\* off the sheet | — | 1.9 spacings | **13.5 spacings** | |

Read it carefully, because two things move in opposite directions:

**The free arm got better and now beats us** — 1.0033 against our 1.0177. That is
expected and it is not a loss. Free optimisation has five unconstrained numbers
against our four physical knobs; its score is a **lower bound that nothing
physical can beat**, and D4 already said so. It is also the first phi in this
project to reach **0.750 accuracy** where everything buildable sits at 0.688. It
is a genuine optimum, not a degenerate corner.

**The projected arm got much worse** — 1.1128 against yesterday's 1.0258. This is
the number that carries the argument, and the mechanism is clean: yesterday it
was projecting the cloud's own centroid, which is by construction a middling
device and therefore scores respectably. A *real* free optimum sits far outside
the reachable set, and the nearest device to it is dragged to a corner of the
design box (film thickness 5.03 nm, the minimum allowed). Chasing an unreachable
target takes you somewhere bad.

### Your D5 gate, re-answered

Your gate: *joint descent must beat the projected baseline by at least 5%
relative, or say so and move the headline to sample-efficiency.*

**It passes.** Joint 1.0177 against projected 1.1128 is **8.6% relative**
(9.3% if you divide by the joint score instead). Yesterday's "0.79%, so it fails"
was a verdict on a baseline that had not been computed.

**Both claims are now available and they should both be made**, in this order:

1. **The objection's own proposal is the worst strategy on the board.** Optimise
   phi freely and build the nearest device: 1.1128, using 192 solver calls.
   Every other row beats it.
2. **Sample-efficiency.** Descending through the solver found a better device
   than 192 evenly-spread ones using **a third of the calls**, and the device it
   found beats every one of those 192.

**And the caveat that must travel with it.** The gap between joint and projected
is now large *because the free optimum is genuinely far away*, and "nearest in a
standardised Euclidean metric" is a naive way to project. A smarter engineer
would not project at all — they would notice phi\* is unreachable and go
searching. That strategy is also on the table, it is the "best of 192" row at
1.0221, and joint descent still beats it with a third of the calls. Say both.

---

## 3. The budget sweep — where the crossover actually sits

D4 ran the race at two budgets, 20 solver calls and 64, and the ordering reversed
between them. Two points prove a crossover exists; they do not place it, and
"somewhere between 20 and 64" is a weak sentence for a writeup.

Five budgets, five arms, three seeds, one harness, one objective. **75 runs,
3.04 hours**, driven by `scripts/race_sweep_d5.py`, banked to
`results/runs/race_crossover_sweep.json`. Figure 2 is
`docs/figures/fig2_budget_crossover.png` / `.pdf`.

### The table

Median best balanced cross-entropy over three seeds:

| arm | 12 | 20 | 32 | 48 | 64 |
|---|---:|---:|---:|---:|---:|
| **gradient descent (this project)** | 1.073836 | 1.038718 | 1.033119 | 1.023138 | **1.017666** |
| Bayesian optimisation, warm-started | **1.038707** | 1.032951 | 1.032311 | **1.022400** | 1.020981 |
| Latin hypercube | **1.027902** | 1.031666 | 1.034359 | 1.023662 | 1.027694 |
| random search | 1.030783 | **1.030783** | **1.030783** | 1.030783 | 1.030066 |
| Nelder-Mead | 1.297846 | 1.059021 | 1.050448 | 1.048446 | 1.048191 |
| *best arm at this budget* | *lhs* | *random* | *random* | *bayes* | ***gradient*** |

### The harness reproduces D4 exactly

Before anything new: budgets 20 and 64 were re-run through the new driver and
returned **1.038718** and **1.017666** for the gradient arm, **1.048191** for
Nelder-Mead, **1.036441** and **1.018232** for random and LHS at their D4 seeds —
identical to the banked race. The sweep and the banked race are measuring the
same thing.

### There is not one crossover, there are three

Quoting a single number would be wrong. Gradient descent overtakes each baseline
at a different budget, and the honest statement names them:

| it overtakes | between |
|---|---|
| Nelder-Mead | already ahead at 12 calls |
| Latin hypercube | **20 and 32** calls |
| random search | **32 and 48** calls |
| warm-started Bayesian optimisation | **48 and 64** calls |

So D4's "somewhere between 20 and 64" is now placed: against the *sampling*
baselines the crossover is at 20–48, and against a well-tuned GP it is at 48–64.
Below 20 calls this project is fourth of five and that must be stated.

### The finding that is better than the crossover

**Gradient descent is the only arm that converts extra budget into performance.**

| | 12 calls | 48 calls | change |
|---|---:|---:|---:|
| gradient descent | 1.073836 | 1.023138 | **−5.07e-02** |
| random search | 1.030783 | 1.030783 | **0** |

Random search is flat **to six decimal places** from 12 calls through 48. The
best point in its first twelve draws is still the best after forty-eight. LHS is
not even monotone — it is *worse* at 32 than at 12. Bayesian optimisation
improves, but by 0.016 across the whole range against the gradient arm's 0.056.

The honest qualifier: each arm's draws are a nested sequence at a fixed seed, so
"random at 48" contains "random at 12" as a prefix rather than being an
independent run. That is the right comparison for a budget sweep — it is what
"give the same method more money" means — but it is why the flatness is a
statement about the objective's shape as much as about the method. The objective
has a broad shallow basin that a dozen uniform draws already find, and a floor
inside it that only a directed method reaches.

**That is the sample-efficiency claim in its strongest and most defensible form.**
Not "we win" — at four of the five budgets we do not. It is: *the derivative-free
arms saturate, and this one does not.* Everything about the trade improves with
dimension, because a Jacobian costs 2D+1 calls while covering a box costs
exponentially many, and D = 4 is the least favourable end of that.

### Two things a judge would catch, said first

- **Zero spread.** Gradient descent and Nelder-Mead give the same answer on all
  three seeds to six decimal places, because both start from a fixed corner and
  follow a deterministic rule. The bands in Figure 2 are the other arms being
  averaged over their own luck. Report it as a property, not as an advantage.
- **The gradient arm slightly overran two budgets.** It used 22 calls at budget
  20 and 33 at budget 32 (exactly 12, 48 and 64 at the others), because the cap
  is tested before a step and a forced Jacobian refresh can carry it past.
  That is a ~10% overrun, and it happens at precisely the budgets where this
  project *loses* — so it makes our reported score at those budgets slightly
  too generous, not too harsh. Worth disclosing for that reason.

---

## 4. V7, the H-ablation — the transducer is carrying physics

### What the question is, and why V2 does not answer it

V2 checks that the composed gradient agrees with a finite difference of the
composed loss. That proves the chain rule is **implemented**. It does not prove
the middle link means anything — a chain rule assembled from three consistent but
physically arbitrary maps would pass V2 exactly as well.

So replace the middle link with noise. `J_H = dH/dy` is the 5x7 Jacobian of the
DPI transducer: how each of the five numbers the network runs on responds to each
of the seven the solver measured. This replaces it, **in the backward pass only**,
with a norm-matched random matrix, and reruns the same trust-region descent from
the same poor corner on the same 64-call budget.

The forward pass is never ablated. Every loss, every figure of merit and every
phi below still came from the solver at the design point it is attributed to.
Only the direction the optimiser reads is scrambled, so the two runs are scored
by the identical yardstick and differ only in where they walked.

`scripts/h_ablation_d5.py`, 7 runs, **74.7 minutes**, banked to
`results/runs/h_ablation_d4.json`.

### Two ways to match the norm, and they are not equally hard to survive

- **`frobenius`** — `R = G * ||J_H||_F / ||G||_F`. The literal reading of the
  BUILD_SPEC, and the **weak** ablation. The rows of `J_H` live on wildly
  different scales, so one global rescaling dumps almost all of a random matrix's
  mass into whichever row was largest and starves the rest.
- **`rowwise`** — each row of `R` is a random direction in R^7 with the **length
  of the corresponding row of `J_H`**. Every phi component responds exactly as
  strongly as it really does; only *which* figures of merit it responds to is
  scrambled. This is the one that is hard to survive, so it is the one to lead
  with.

Both are genuinely scrambled: mean cosine against the true `J_H` is **+0.056**
(rowwise) and **+0.053** (frobenius), i.e. essentially orthogonal.

### The result

| mode | seed | loss | accuracy | SS (mV/dec) | memory window (V) | calls used | steps accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| **control** (true J_H) | 0 | 1.399577 → **1.017666** | 0.250 → **0.688** | 71.1 → 97.5 | 0.415 → **0.576** | **64** | **8 of 15** |
| rowwise | 0 | → 1.220928 | → 0.438 | → 81.3 | → 0.450 | 40 | 3 of 8 |
| rowwise | 1 | → 1.147751 | → 0.500 | → 87.2 | → 0.473 | 40 | 4 of 9 |
| rowwise | 2 | → 1.149260 | → 0.500 | → 87.2 | → 0.467 | 42 | 4 of 9 |
| frobenius | 0 | → **1.399577** | → 0.250 | → 71.1 | → 0.415 | 25 | **0 of 5** |
| frobenius | 1 | → 1.147149 | → 0.500 | → 87.2 | → 0.472 | 45 | 4 of 9 |
| frobenius | 2 | → **1.399577** | → 0.250 | → 71.1 | → 0.415 | 25 | **0 of 5** |

**The control reproduces the flagship exactly** — 1.017666, accuracy 0.688, 64
calls. The ablation harness is not changing the answer.

How much of the control's improvement a random `J_H` recovers, at the median:

| | loss reduction | accuracy gain | memory-window gain |
|---|---:|---:|---:|
| control | 0.3819 | +0.4375 | +0.1610 |
| **rowwise random** | **66%** | **57%** | **32%** |
| **frobenius random** | **0%** | **0%** | **0%** |

**The loss does still fall**, which is what the BUILD_SPEC predicted and what
makes this a real test rather than a formality — four knobs in a box with a trust
region will find *something*. But it recovers a third of the memory window, half
the accuracy, and it **cannot spend its budget**: every ablated run stalls, using
40–45 of 64 calls against the control's 64, because the random directions keep
failing and the trust region collapses onto itself. Two of three `frobenius`
seeds accept **zero steps out of five** and end exactly where they started.

### The recovered devices, which is what the claim is actually about

| | t_fe (nm) | L_g (nm) | log10 N_ch | t_IL (nm) |
|---|---:|---:|---:|---:|
| start | 5.500 | 52.00 | 17.800 | 1.550 |
| **control** | **7.653** | **35.43** | **17.167** | **1.374** |
| rowwise (3 seeds) | 5.96 – 6.26 | 40.0 – 43.0 | 17.745 – 17.804 | 1.645 – 1.666 |

Three things, and the third is the sharpest:

1. **The film barely thickens.** The control adds 2.15 nm of ferroelectric; the
   ablated runs add 0.5–0.8 nm, a third of the way.
2. **The channel doping does not move at all.** 17.800 → 17.167 under the true
   Jacobian; 17.800 → 17.75 under a random one. That knob is simply not found.
3. **The interlayer moves the WRONG WAY.** The control thins it, 1.550 → 1.374,
   which improves gate coupling. Every ablated run *thickens* it, to 1.645–1.666.
   A random `J_H` does not merely fail to find the design — on one of the four
   fabrication knobs it walks in the opposite direction.

### A correction to what the BUILD_SPEC expected

The spec predicted the ablation signature would be **"SS driven up, MW driven
down"**. Measured, that is not what separates them, and the first half is
backwards:

| | ΔSS (mV/dec) | ΔMW (V) |
|---|---:|---:|
| control | **+26.3** | **+0.161** |
| rowwise | +16.1 | +0.052 |
| frobenius | 0 | 0 |

**The true gradient raises SS the most.** That is not a failure — it is a
coherent physical trade, and it is the interesting part of the result. A thicker
HZO film and a thinner interlayer buy a larger memory window at the cost of
electrostatic control, so the subthreshold slope degrades. The network needs the
memory window, because that is what separates the two conductance states and
therefore sets the weight dynamic range it has to work with. **Giving up
subthreshold slope to buy memory window is the correct answer here**, and the
optimiser found it. The ablated runs move SS *less* simply because they travel
less far.

So SS is not a failure signature; it is a correlate of distance travelled. The
discriminators that do work, and that the writeup should use, are: **accuracy**
(0.688 / 0.500 / 0.250), **memory window** (+0.161 / +0.052 / 0), **whether the
run can spend its budget** (64 / 40 / 25 calls), and **the sign of the interlayer
step**.

**V2 proves the chain rule is implemented. This proves it is meaningful.**

---

## 5. The four Tesseracts, and the tests

### Containers

`scripts/smoke_tesseracts.py` — new, and it is the check that was missing. The
four Tesseracts are what a judge actually runs, and between them they carry three
schema surfaces and four import paths that nothing else exercises. A field
renamed in one schema and not the other shows up when somebody else tries to
serve the container, not in our optimiser logs.

**22 of 22 checks pass, in 32 seconds.** Per Tesseract: the module imports on its
own; both schemas build; `apply` returns finite numbers; and **`abstract_eval`
agrees with what `apply` actually returns** — same keys, same shapes, same
dtypes. That last one is the check that earns its keep, because `abstract_eval`
is hand-written and a mismatch breaks tesseract-jax's tracing with an error that
points at the caller rather than at the file.

| Tesseract | checks | backend used | why that backend |
|---|---:|---|---|
| `sentaurus-fefet` | 4 | replay | the commercial solver is on a shared licensed host; replay serves a curve that host really produced |
| `devsim-fefet` | 4 | devsim | the open solver is the one anybody can run, so this one is real |
| `adjoint-shim` | 7 | mock | the derivative endpoints cost 2D+1 probes at a fresh point; the mock exercises identical endpoint code in milliseconds |
| `snn-lif-ecg` | 5 | frozen | the real network |

Two of the shim's checks are not smoke tests but assertions with content:

- **`vjp` equals `J^T` applied to the cotangent.** With an all-ones cotangent
  that is exactly the column sums of the Jacobian, so the two are checkable
  against each other. If they disagreed, the endpoint is not wired to the shim.
- **A repeat VJP costs zero solver calls.** Measured: 0 extra probes. That is the
  entire economic argument for the apparatus, asserted in code rather than in a
  README.

One thing found and fixed while writing it: the smoke test first ran with
`batch=4`, which misses the cached reference weights — their disk key includes
the batch size — and silently re-ran an 800-step fit. Twenty minutes to smoke
test one endpoint. It now uses `batch=16`, which is what every banked run used.

### Tests

**All 155 pass.** `pytest tests/ -q`, exit 0, no failures and no errors. One
warning, pre-existing and benign (`float()` on a tensor with `requires_grad` in
`test_circuit.py`).

### One thing worth recording from the smoke test

At the frozen reference device, `dL/dphi` comes back as

```
beta -3.3994   g_min 0.0   g_max 0.0   th_th +0.3737   sig_w +3.6544
```

The two conductances are **exactly** zero. That is not a bug and it is the
known behaviour of `SNN_VJP=fd` on a hard-spike network: the loss is piecewise
constant in phi, and a finite-difference step that does not flip a single spike
anywhere in 111 timesteps returns exactly zero change. It is the same fact that
`FlagshipConfig.min_radius` exists to handle. Worth knowing that it can happen at
a reported point, not just in the middle of a run.

---

## 6. Bonus, and it was free — no single figure of merit predicts performance

The D6 checklist says the FoM-vs-loss correlations must be re-measured on the
post-trim objective before they go in any figure, because the trims and the
frozen network changed every loss number. The old measurement was **8 design
points and one seed**.

`scripts/fom_correlations_d5.py` re-measures it on **192 devices**, on the
current objective, with zero solver calls — the manifold cloud has the figures of
merit and the V6 refit has their losses.

| quantity | Pearson r | R² alone | Spearman |
|---|---:|---:|---:|
| forward threshold | −0.382 | 0.146 | −0.410 |
| subthreshold slope | +0.362 | 0.131 | +0.277 |
| low conductance (log) | +0.347 | 0.120 | +0.379 |
| leakage current (log) | +0.343 | 0.118 | +0.374 |
| conductance ratio (log) | −0.351 | 0.123 | −0.381 |
| membrane decay | −0.337 | 0.113 | −0.374 |
| conductance slope (log) | +0.176 | 0.031 | +0.205 |
| high conductance (log) | +0.164 | 0.027 | +0.164 |
| weight noise | +0.134 | 0.018 | +0.226 |
| firing threshold | −0.061 | 0.004 | −0.164 |
| **memory window** | **−0.108** | **0.012** | **−0.085** |

**The best single predictor explains 15% of the variation.** The memory window —
the first number any device engineer reaches for, and the thing a FeFET paper
would headline — explains **1%**.

And the fair upper bound on any shortcut: **all seven figures of merit in one
linear model give R² = 0.354**, fitted and scored on the same 192 points, so it
is generous to the shortcut rather than to us. Roughly two thirds of what makes
one device better than another for this task is not in any linear reading of the
device's own summary numbers.

**That is the argument for doing this with gradients**, and it is now on 192
points instead of 8. If one scalar predicted performance you would maximise that
scalar and skip the whole pipeline.

The D6 note warning against writing up "bigger memory window is worse" is
confirmed and should be obeyed: at −0.108 over 192 devices there is no effect to
report in either direction.

---

## 7. What I did not do

- **I did not touch the device physics**, the circuit, the trims, or the frozen
  constants.
- **I did not rewrite D4's section 6.** It carries a superseded banner pointing
  here. The wrong numbers stay visible with the reason next to them.
- **I did not overwrite `v6_manifold_control_d4.json`.** The corrected run is a
  new file, `v6_manifold_control_d5.json`.
- **Nothing committed, nothing pushed.**

---

## 8. Files this day produced

| path | what |
|---|---|
| `docs/figures/fig1_pca_manifold.png` / `.pdf` | Figure 1 — the reachable manifold |
| `docs/figures/fig2_budget_crossover.png` / `.pdf` | Figure 2 — sample efficiency |
| `scripts/fig2_budget_crossover.py` | draws Figure 2 |
| `results/runs/v6_manifold_control_d5.json` | the corrected V6 control |
| `results/runs/race_crossover_sweep.json` | the budget sweep |
| `results/runs/h_ablation_d4.json` | V7 |
| `results/runs/smoke_tesseracts.json` | the container smoke test |
| `results/runs/fom_correlations_d5.json` | FoM-vs-loss over 192 devices |
| `scripts/fig1_pca_manifold.py` | draws Figure 1 |
| `scripts/v6_free_refit_d5.py` | the corrected free arm |
| `scripts/race_sweep_d5.py` | the budget sweep driver |
| `scripts/h_ablation_d5.py` | V7 |
| `scripts/smoke_tesseracts.py` | the container smoke test |
| `scripts/fom_correlations_d5.py` | the correlations |
