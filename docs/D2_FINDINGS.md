# D2 findings — the open oracle, and four things that were wrong underneath it

Dated 2026-08-24. Companion to `docs/D1_FINDINGS.md`. Everything here was measured
on this machine on D2; nothing is quoted from the literature without saying so.

---

## G5 — PASSED

**Gate:** hysteretic Id–Vg from DEVSIM with a memory window `MW = vth_fwd − vth_rev > 0.1 V`.

**Result at the nominal d=3 device** (`t_fe` 10 nm, `Pr` 15 µC/cm², `Ec` 1.2 MV/cm),
extracted by the frozen `shared.extract` on the frozen 96-point grid:

| FoM | T2 (DEVSIM) | mock | ratio |
|---|---|---|---|
| SS, mV/dec | 91.21 | 85.21 | 1.07 |
| vth_fwd, V | 0.5191 | 0.5163 | 1.005 |
| vth_rev, V | 0.1254 | 0.1142 | 1.10 |
| **MW, V** | **0.3936** | 0.4021 | 0.98 |
| I_leak, A | 2.560e-10 | 1.700e-10 | 1.51 |
| g_lo, S | 1.273e-05 | 2.600e-05 | 0.49 |
| g_hi, S | 1.910e-04 | 2.000e-04 | 0.95 |
| dg/dV_th, S/V | 3.662e-04 | 5.610e-04 | 0.65 |

MW = 0.394 V, four times the gate. The fallback branch (b) was not needed: the
ferroelectric is meshed and solved, not bolted on. **Wall clock 36–40 s per design
point**, against 306 s for one Sentaurus run — which is what makes DEVSIM the
solver the D3 validation suite runs on.

The five transduced hyperparameters all land inside the healthy ranges
`tests/test_tier_a_pipeline.py` asserts: β ≈ 0.49, th_th ≈ 5.2, σ_w ≈ 0.14.
That is not luck, it is the point of the three constants below.

## What T2 actually solves

2-D drift-diffusion, Poisson + electron continuity, Scharfetter–Gummel, SRH, over a
three-region gate stack (Si film / SiO₂ interfacial layer / HZO). The ferroelectric
is a **meshed region whose Poisson flux carries the Miller polarization**

```
D = eps_bg * E + eta * Ps * tanh[ (E - s*Ec) / (2*delta) ],   delta = Ec / ln[(1+Pr/Ps)/(1-Pr/Ps)]
```

with the analytic `dP/dE` handed to Newton, so the film responds to the channel at
every bias point. `s = ±1` selects the saturated branch and is the **only**
difference between the two rows of `id_vg`. Clean-room from Miller & McWhorter,
JAP 72, 5999 (1992); QS-Devsim is not used and not vendored (non-commercial licence
+ patent CN 113297818 B, both incompatible with Apache-2.0).

The branch sign falls out rather than being imposed: the up-sweep branch passes
through `P = −ηPr` at `E = 0`, the bound charge repels electrons, V_th is high —
the erased state. `MW = vth_fwd − vth_rev > 0`, exactly the D1 convention.

### Three constants are chosen rather than solved

`MU_N_EFF = 150 cm²/Vs`, `PHI_M_GATE = 4.5943 eV`, `FE_ACTIVE_FRACTION = 0.0377`.
Each is one scalar, each has a physical name, and **none of them can change a sign
in the Jacobian**, which is what V4 compares across the two solvers.

* **Mobility.** DEVSIM's default `mu_n = 400` is bulk phonon-limited silicon, wrong
  for a surface channel under a high-k stack. At 400 the nominal device reads 3×
  too conductive and th_th lands at 1.7 spikes-to-fire.
* **Work function.** This is work-function engineering, which is how V_th is set in
  a metal-gate process. 4.5943 eV puts vth_fwd at 0.519 V, so the frozen
  `V_read = 0.60` and `V_leak = 0.246391250` mean what they were computed to mean.
* **Active fraction η.** An ideal, fully-active, perfectly-screened film gives
  `MW = 2·Ec·t_fe = 2.4 V` at nominal, which no measured 10 nm HZO FeFET shows.
  HZO is a phase mixture, only the orthorhombic fraction switches, and part of what
  switches is compensated by interface traps and by the depolarising field of the
  interfacial layer. η is fixed once, at the nominal point, and never refitted.

---

## Four things that were wrong, and how they showed up

### 1. A bulk-planar body punches through at L_g = 40 nm

The first T2 device was a conventional bulk MOSFET. Measured: **I_d fell by only
14× over 1.1 V of gate swing** and there was no subthreshold region anywhere in the
frozen sweep window. With the frozen channel-doping box (`N_ch ≤ 1e18`) each
junction depletion region is ~36 nm wide and they meet under a 40 nm gate.

Fixed by going to an **8 nm ultra-thin body** on an ideal insulating substrate
(the film's bottom face carries no contact and no interface, which is DEVSIM's
natural zero-flux condition — a thick buried oxide with no back gate). Electrostatic
integrity then comes from T_SI rather than from doping, and the device is
well-behaved across the whole box including L_g = 20 nm. It is also the right
device: FeFETs of this class are demonstrated on fully-depleted films, and a 2-D
cross-section of a nanosheet is exactly this.

### 2. The hole continuity equation is a stiff trap, and the device does not need it

The film is p-type and its only contacts are n+, where the boundary condition pins
`p = n_i²/N_d ≈ 1 cm⁻³`. The body's hole population is therefore coupled to the
rest of the system only through SRH. Driving the gate to the bottom of the frozen
window puts ~1e20 holes into an accumulation layer that nothing but generation can
fill, and Newton's hole residual then falls **linearly, ~5–6% per iteration**:
measured 106 iterations to move it from 1e-2 to 1.3e-4 while Potential was already
at 1e-8 and the electron residual at 7e-7.

Every ampere of I_d is electrons. Holes are now an equilibrium node model
`p = n_i exp(−ψ/V_t)` — the textbook unipolar MOSFET approximation. The stiff mode
is gone, the system is two equations instead of three, and the DC solution stops
being path-dependent, which matters because this oracle is about to be
finite-differenced.

### 3. Three silent numerical traps in DEVSIM's expression language

* `exp(-(u)^2)` **binds the unary minus tighter than the power**, so it evaluates
  `exp(+u²)`. The halo Gaussian reached 1e92 a hundred nanometres from the junction
  and the equilibrium solve converged anyway, to nonsense. Write `exp(0-(u^2))`.
* The textbook initial potential `V_t*log(0.5*(N + sqrt(N²+4n_i²))/n_i)` **cancels
  catastrophically on the p-side**: at N = −1e17, N² = 1e34 swamps 4n_i² = 4e20, the
  square root returns exactly |N|, the sum is exactly zero, and DEVSIM aborts on
  log(0). Factor the sign out first.
* `add_2d_contact` needs a region interface to attach to. Without a margin of inert
  "air" around the structure the body contact **silently fails to exist** and the
  gate contact picks up two corner nodes instead of twenty-three.

### 4. The classifier was never trained — the flagship had nothing to descend

The most expensive one. `snn-lif-ecg` built a `LIFNet` with random weights and
never updated them, so the class-balanced cross-entropy sat at ln(4) = 1.3863 plus
noise for **every** device in the box. Measured, stepping along the manufactured
descent direction from the nominal point:

| step along −g | 0.005 | 0.01 | 0.02 | 0.05 | 0.10 | 0.20 | 0.40 |
|---|---|---|---|---|---|---|---|
| loss | 1.363 | 1.431 | 1.398 | 1.427 | 1.398 | 1.389 | 1.387 |

A random walk converging back to chance. The optimiser would have spent its entire
solver budget descending a surface with no slope in it, and the run would have
looked like a gradient-quality problem rather than a missing training loop.

T4 now trains W to approximate stationarity under the φ it is handed, and reports
the loss there — which is the question the project is actually asking: *given a
device, how well can a network built on it do*. The VJP still holds W fixed, and
that is legitimate rather than lazy: with `L(φ) = L(φ; W*(φ))`,

```
dL/dφ = ∂L/∂φ + (∂L/∂W)·dW*/dφ
```

and at a stationary W* the second term vanishes. The envelope theorem does the
work; nothing differentiates through the optimiser. The approximation is the word
"approximately", and `SNN_TRAIN_STEPS` is the knob that controls it.

After the fix the landscape is real: CE ranges from **0.019 to 1.25** across the
d=3 box, against 1.386 ± noise before.

---

## The mini-flagship

Launched by `scripts/overnight_d2.sh`. Two runs: the d=3 mini-flagship at exactly the
budget priced for the commercial solver (45 calls), then the deeper d=5 run the open
solver can afford at 36 s a point. Live output in
`results/runs/mini-flagship-devsim-d3/` and `results/runs/flagship-devsim-d5/`;
`steps.jsonl` is written per step, not at the end, so a run that dies overnight still
leaves everything it learned.

**It runs on DEVSIM, not Sentaurus.** No credential was reachable in this session, and
that is a blocker rather than an inconvenience — see the open items above.

Both start from `theta0 = (0.20, 0.40, 0.30, …)`, a deliberately poor corner, and that
is an experimental-design decision worth stating: the nominal FeFET already solves this
task, so a run started there accepts zero steps and proves nothing in either direction.
The recovery is the result.

Two bugs in the loop itself, both worth remembering because both look like physics:

* **An unfloored trust radius dies against a staircase.** Every rejected step halves it,
  and a step shorter than the distance to the next spike-flip produces *exactly* zero
  change, hence rho = 0, hence another rejection. Measured: the run ended after five
  steps having spent six of its forty-five solver calls. There is now a floor, and
  below it the loop refreshes J from the solver instead of shrinking further.
* **On the floor, it cycles.** Same theta, same J, same direction, same rejected point,
  forever — ten identical steps in one smoke run. Two strikes at the floor now ends the
  descent, which is what a second rejection there actually means.

And one that was quietly inflating the budget: `content_hash` hashes the *inputs*, so the
mock and DEVSIM produce the same key at the same design point. Counting distinct bare
hashes therefore counted a real DEVSIM evaluation as already-seen because a mock smoke
run had visited that theta earlier in the project's history. The counter now keys on
`(backend, hash)` and is scoped to the run by byte offset into `provenance.jsonl`.

### Result — d=3, DEVSIM, 45 calls, 28.6 minutes

`results/runs/mini-flagship-devsim-d3/`. **G6 is satisfied with room to spare.**

| | start | end |
|---|---|---|
| balanced CE | 1.2289 | **0.0223** |
| accuracy | 0.9375 | 1.0000 |
| t_fe, nm | 7.00 | 9.30 |
| Pr, µC/cm² | 13.0 | **23.4** |
| Ec, MV/cm | 1.160 | 1.185 |
| memory window, V | 0.241 | **0.541** |
| I_leak, A | 8.16e-10 | 2.63e-11 |
| g_hi / g_lo | 5.0 | **79** |
| β | 0.089 | 0.928 |
| σ_w | 0.250 | 0.100 |

14 steps, 9 accepted, 5 rejected, 9 Jacobian refreshes and 19 Broyden updates; ρ was
positive on 64% of steps. Every one of those loss values came out of a 2-D
drift-diffusion solve at the design point it is attributed to, and
`results/runs/provenance.jsonl` says so per call.

The physics is coherent and it was not put there by hand: the optimiser widened the
memory window, dropped the leakage by 31× (which is what took β from 0.089 — a membrane
that forgets everything in one timestep — to 0.928), opened the conductance ratio from
5 to 79, and halved the weight noise. It got there almost entirely through `Pr` and
`t_fe` and **left `Ec` essentially alone** — which is the same conclusion the
cross-check below reaches by a completely independent route.

### Result — d=5, DEVSIM: the same win bought a different way, and then a stall

`results/runs/flagship-devsim-d5/`. 57 of 120 calls, 32.7 min, 12 steps, 5 accepted,
7 rejected. Balanced CE **1.2289 → 0.2328**, accuracy 0.9375 → 1.0000. It stopped on
the stall break, not on the budget.

The interesting part is *how* it got there, because it is not how d=3 got there:

| | d=3 (45 calls) | d=5 (57 calls) |
|---|---|---|
| balanced CE | 0.0223 | 0.2328 |
| route | the ferroelectric | the gate length |
| L_g, nm | fixed 40 | 40 → **49.1** |
| Pr, µC/cm² | 13 → **23.4** | 13 → 13.5 |
| memory window, V | 0.241 → **0.541** | 0.241 → 0.250 |
| SS, mV/dec | 85.5 → 88.9 | 85.5 → **74.0** |
| I_leak, A | 8.2e-10 → 2.6e-11 | 8.2e-10 → 7.4e-11 |
| β | 0.089 → 0.928 | 0.089 → 0.777 |

Given a gate length, the optimiser spent almost nothing on the ferroelectric: it
lengthened the channel, which sharpened the subthreshold slope by 11 mV/dec, which cut
the leakage by 11×, which is what the membrane time constant actually cares about.
Denied that knob at d=3, it bought the same β by widening the memory window instead.
Two routes to one objective, selected by which knobs exist — which is the thing d=5 was
put in the design vector to expose.

**But d=5 did worse with more freedom, and that is the finding.** ρ over its eleven
steps was +2.70, +0.63, −0.44, +0.02, +0.02, −0.79, +0.81, −6.75, −4.22, −1.71, −4.56 —
positive on 45% of steps, and the last four are wild. A local model that mispredicts by
a factor of five in both directions is not describing the solver, and the run spent its
last four steps and 24 solver calls confirming that before the stall break fired.

There is a specific and checkable suspect. **`L_g` is the one d=5 parameter that moves
the mesh** — T2 regenerates the grid per design point, so a finite difference in `L_g`
differences two different discretisations, which is exactly the deterministic
non-smoothness V1 exists to measure and the reason V1 is not a repeatability test. The
optimiser leaned hardest on `L_g` and then stalled. **Look at the `L_g` column first on
D3**, and if it is the noisy one, G7's fallback (drop to d=3 material-only) has a
measured justification rather than a precautionary one.

### V4 early warning: T2 against the mock

`scripts/cross_check_oracles.py`, 2D+1 calls per oracle, at nominal d=3. Compared in
relative units (`d log FoM / d theta_i`) so that a solver whose currents are uniformly
larger is not reported as disagreeing.

**Sign agreement 17/21. Rank-order agreement 1/7.** Three separate things are going on
and they need separating, because only one of them is a problem.

**1. Two of the four sign disagreements are the mock being blind, not the two solvers
disagreeing.** `d(SS)/d(Pr)` and `d(SS)/d(Ec)` in the mock are −2.7e-14 and −2.1e-14:
exactly zero, to float noise. The mock's ideality factor has no `Pr` or `Ec` term at
all. DEVSIM returns −2.2e-2 and +1.4e-2, because in a real stack the ferroelectric's
differential capacitance sits in series with the body and moves the subthreshold slope.
Comparing a sign against numerical noise is not a comparison.

**2. `dg/dV_th` disagrees on all three columns, and this one is real.** Mock
(+0.147, +0.130, +0.174) against DEVSIM (−0.907, −0.662, −0.076) — neither side is near
zero, so it is a genuine systematic flip. The cause is the mock's above-threshold
branch: it is a log-linear soft-min with `GAMMA_ON = 1.6 dec/V`, i.e. **exponential**
above threshold, so its transconductance keeps growing as the reverse branch is driven
further on. A real MOSFET's does not — it rolls off. At `V_read = 0.60 V` the
programmed branch is well above threshold, which is exactly where the two models part
company.

`dg/dV_th` feeds `sigma_w` and nothing else, so the consequence is bounded but sharp:
**the weight-noise term would be pushed in opposite directions on the two oracles.**
This is precisely the failure V4 exists to catch, and catching it on D2 rather than on
D6 is most of the value of running the check early. Note what it does NOT threaten:
V4 as specified is Preisach against Miller, i.e. **T1 against T2**, and both of those
are real solvers. The mock is the Tier A oracle, not a party to V4.

**3. The rank-order disagreement is one mechanism difference, stated three times.** In
DEVSIM the `Ec` column is roughly 10× weaker than `t_fe` and `Pr`; in the mock it is
first-order. The mock sets `MW ∝ Ec · t_fe · tanh(Pr/15)`. T2's window is set by
`2·η·Pr·t_fe / eps_eff`, and `Ec` enters only through `eps_eff` — weakly, and with the
opposite curvature. Both are defensible readings of a ferroelectric gate; they are not
the same reading. The mini-flagship independently agreed with T2's version by barely
moving `Ec` over 45 solver calls.

---

## Open, and carried to D3

### The synthetic task is too easy, and it is the next bottleneck

With W trained, accuracy is **1.000 at almost every device in the box** — at batch
32 and at batch 256 alike. Only a film thin enough to collapse the memory window
(`t_fe ≲ 5.3 nm`) degrades it. The landscape is close to binary: the device either
works or is dead, with little slope in between, which is a hard surface for any
first-order method. This is a *data* problem and MIT-BIH inter-patient DS1/DS2 is
already scheduled for D4; it should be treated as a prerequisite for the D6
flagship, not as an enhancement.

### V4 cannot be stated against the mock, and dg/dV_th is why

See the cross-check above. Before D6, either fix the mock's above-threshold branch so
its transconductance rolls off like a MOSFET's (it is `GAMMA_ON` in
`shared/mock_device.py`, and changing it moves the D1-frozen `K_syn` calibration, so it
is not a free edit), or state V4 strictly between T1 and T2 and say plainly in the
writeup that the analytic mock is a wiring harness rather than a third opinion. The
second is cheaper and more honest; the first is only worth doing if the mock is going
to be quoted as evidence anywhere.

### The manufactured gradient's magnitude is not yet trustworthy

At the nominal point, `g·u` from the shim is ~0.21 against a central-difference
directional derivative of ~6.3 in the same direction. The **sign** agreed; the
magnitude did not. Over four random directions at α = 0.02 the model/truth ratios
were 0.016, −0.006, −0.104, 0.037 — essentially uncorrelated at that step size.

This is exactly what V1 (the α selection curve) and V2 (cosine vs steps-since-
refresh) exist to settle, and both are D3 items. Two things make it survivable in
the meantime: the step direction is used **normalised**, so only the direction
matters; and every step is accepted only after `actual < 0` is checked against the
solver, so the reported loss can never be made worse by a bad gradient.

### The T1 deck has not been run against a licence

`t1/sdevice_fefet_idvg.cmd` and `.par` are written, and `deck_values` renders both
for d = 3, 5 and 12. **No Sentaurus credential was available in this session**
(`SENTAURUS_PASSWORD` unset, no `.env`, no cached PuTTY host key), so the deck has
never reached sdevice. The `Ferroelectric` keyword set must be checked against the
installed version's Device manual before the flagship. The host itself is reachable
(TCP 22 open).

Three driver changes went in alongside it:

* **`-hostkey` is now passed to plink and pscp.** `plink -batch` refuses to connect
  to a host whose key it has not cached, and this machine has cached nothing — so
  the fingerprint measured on D1 is carried in `T1Config` and is overridable.
* **The frozen defaults are filled for every design vector.** A d=3 point still has
  an L_g and an N_ch, and they are the same numbers T2 uses. Without this a deck
  mentioning `@L_g@` renders at d=5 and fails at d=3.
* **`content_tag` replaces `hash()`** for remote directory names. Python salts
  string hashing per process, so the same design point was landing in a different
  remote directory on every run.

### Mesh invariance of d=3, made exact

`t_fe` is a design variable and the Sentaurus grid is not regenerated per point, so
the meshed ferroelectric slab has a fixed thickness `t_slab` and the material
parameters are remapped. With `k = t_fe / t_slab`:

```
eps_eff = eps_bg / k          Ec_eff = Ec * k          Ps, Pr unchanged
```

The two scales are **reciprocals**; the first draft used one for both, which
preserves nothing. The check that catches it is `Ec_eff · t_slab == Ec · t_fe` —
the coercive *voltage* across the layer is what sets the memory window, and it is
now preserved exactly.
