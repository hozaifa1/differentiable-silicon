# D3 — Recalibration against the thesis baseline

Written 2026-08-24. This records what changed, why, and what it invalidates.

The short version: the project had drifted into optimising a device that cannot
be built, on a task that was not the thesis' task, with a network that was not
the thesis' network, reading figures of merit off a curve with a broken fit. All
four are now realigned. Nothing about the gradient path changed.

---

## 1. Pr and Ec are locked material constants now

**Was:** `Pr` and `Ec` were entries in the design vector. The optimiser moved
them, and the D2 mini-flagship's headline result was largely *bought* by moving
`Pr` from 13 to 23.4 µC/cm².

**Is:** `src/diffsilicon/shared/material.py`

| | value | |
|---|---|---|
| P_r | 32.0 µC/cm² | |
| P_s | 40.0 µC/cm² | |
| E_c | 1.4 MV/cm | |

Locked node **`cal_n16`** of the thesis' hysteresis calibration
(`Simulations/Calibration/HYSTERESIS_CALIBRATION_RESULT.md`, "Calibrated
parameters & the real-world physics extracted"). That fit reproduces Liao 2022
Fig. 7 branch by branch: MW = 1.296 V against Liao's 1.30 V, I_on/I_off ~ 1.8e7,
SS = 45–75 mV/dec.

**Why.** Remanent polarization and coercive field are properties *of a material*.
You change them by depositing a different film and re-running the calibration,
not by asking a fab for a different number. An optimiser handed them will improve
the memory window by changing the material and call it a design result — and it
is the first knob a judge would reach for.

**The lock is enforced, not documented.** `diffsilicon.optimise.run_flagship`
raises on any design vector that exposes either constant.
`DIFFSILICON_ALLOW_LOCKED_MATERIAL=1` is the escape hatch, for reproducing
pre-recalibration runs only.

## 2. The design vector is d=4, the fabrication knobs

    t_fe   ferroelectric thickness      5 – 15 nm      nominal 7.0
    L_g    gate length                 20 – 60 nm      nominal 40
    N_ch   channel doping           1e16 – 1e18 cm^-3  nominal 1e17
    t_IL   interfacial layer          0.5 – 2.0 nm     nominal 1.0

Bounds are carried over unchanged from the old d=5 / d=12 entries, so a d=4 point
and a d=5 point at the same `t_fe` mean the same physical thickness.

`t_fe` nominal is **7.0 nm**, not 10.0. Seven is the calibrated device:
`t1/calibration.local.json` builds its mesh at `t_fe_slab_nm = 7.0`, and the
thesis' own device study settled there. It also makes the Sentaurus fixed-slab
remap exact at the nominal point — `deck_values` reports
`t_fe_snap_error_nm = 0`.

**d=3, d=5 and d=12 are kept, not deleted.** They are what every cached result,
every banked run and the D2 cross-check were computed against. Deleting them
would silently invalidate that evidence instead of superseding it. They can be
evaluated and replayed; they cannot be optimised.

### What locking Pr and Ec exposed on the commercial solver

Locking the material made a latent gap load-bearing, and it is worth stating
plainly because the failure mode is a silent zero.

The Sentaurus deck's only design-dependent tokens were `@PR@`, `@PS@` and
`@FC@`. Those are now constants. So without a change, **every d=4 design point
would render an identical deck** and T1's Jacobian would be exactly zero.

**Fixed: the fixed-slab thickness remap is now implemented.** It was described in
`deck_values`' own docstring and specified in a D2 test, but never written. The
deck runs on one mesh built at `t_fe_slab_nm`, so the slab is made to behave like
a film of thickness `t_fe`:

    k          = t_fe / t_slab
    Ec_eff     = Ec * k        coercive VOLTAGE preserved:  Ec_eff*t_slab = Ec*t_fe
    eps_fe_eff = eps_bg / k    capacitance/area preserved:  eps_eff/t_slab = eps_bg/t_fe

Reciprocal factors. Using the same one for both renders fine, runs fine, and
silently multiplies the memory window by (t_slab/t_fe)². `P_r` and `P_s` are not
remapped — polarization is a charge per unit *area*.

The film's background permittivity moved out of the `.par` literal into
`calibration.local.json` as `eps_fe_bg`, because the remap has to scale it. **At
t_fe = 7 nm the remap is exactly the identity** — k = 1.000, eps_eff = 33.000,
F_c = the calibrated value — so the fitted device is untouched at the thickness
it was fitted at. That is asserted in the tests.

### Three zero columns, and the path that closes them

`L_g`, `log10_N_ch` and `t_IL` — three of the four d=4 variables — do not reach
the sdevice deck at all. They are geometry and doping, baked into the mesh when
it was built, and the driver ships one `.cmd` and one `.par` against a fixed
`.tdr`. **So on the fixed-mesh path a d=4 Jacobian on the commercial solver has
one live column, `t_fe`, and three identically zero.**

This was first written up here as an accepted limit. That was too quick. Looking
properly at what already exists on disk: the mesh builder `sde_ambi16.cmd` — the
deck that builds the ambipolar structure the calibration was fitted on — opens
with

    (define L_gate 0.100)  (define T_ox 0.002)  (define T_fe 0.010)  (define N_sub 1e16)

All four design variables, already named parameters of a validated mesh builder.
The same lesson as the sdevice deck: the working version existed.

**So `T1_REBUILD_MESH=1` now exists**, and it is OFF by default.

* `t1/sde_fefet_mesh.cmd` is that deck with five literals turned into
  placeholders. Geometry, doping windows, contacts and refinement are unchanged.
* `t1_driver.build_mesh` renders it per design point, pushes it, runs `sde` on
  the host under the same single-licence discipline as `sdevice`, and hands the
  resulting `.tdr` to the solve.
* The mesh geometry is folded into the content tag, so two points differing only
  in gate length no longer collide.

**The remap must be OFF when the mesh is exact, and that is a silent trap.** If
the film is built at its true thickness AND the fixed-slab remap is applied on
top, the thickness is counted twice — once in the geometry, once in the material
— and the memory window is wrong by (t_fe/t_slab)² on a deck that renders and
runs perfectly. `deck_values(..., mesh_is_exact=True)` sets k = 1, and there is a
test.

**What this costs, and why it is opt-in:** an `sde` run per design point on top
of sdevice's ~306 s, more time on the one shared licence, and a mesh that changes
with theta — so a finite-difference column differences two discretisations. That
last one is exactly the noise V1/G7 exist to measure, and it is the reason an
unattended overnight run should not be the first thing to try it.

**It is a code path, not a result, until the control case runs.** Build a mesh at
the parameters that reproduce `cfg.grid`, run the nominal design point, and check
it lands on what the fixed-slab path gives. `baseline_mesh_values()` is that
control — and it currently **raises**, because the geometry `fe07_msh.tdr` was
built at is not known: `sde_ambi16.cmd` is a 10 nm film on a 100 nm gate and the
calibrated grid is a 7 nm film. Those three numbers are left at 0 in
`calibration.local.json` rather than guessed. A plausible wrong baseline would
make a rebuilt mesh look validated when it is not.

One assumption to state either way: the calibration constants (workfunction,
fixed charge, Dit, GIDL) were fitted on a 100 nm gate. Rebuilding at 20–60 nm
carries them to a different geometry. That is ordinary TCAD practice — they are
material and interface parameters — but it is an assumption.

Until the control case has run, the T1-vs-T2 cross-check (V4) must compare the
`t_fe` column only, and must not read the other three "agreeing" at zero as
agreement.

## 3. Real data: 2000 MIT-BIH beats, four AAMI classes

`src/diffsilicon/snn/ecg.py`, adopting the thesis' own `dataset.py`
(`DeltaTransformedECG`) unchanged in structure — class order, up/down interleave,
cue padding, the 116-step tail.

    x: (2000, 1116, 3)   up / down / cue      y: N 1000, F 250, SVEB 250, VEB 500

The CSVs are **not committed** — a preprocessing of a public database, and this
repository is public. Set `DIFFSILICON_ECG_DIR`; the first load writes an .npz
cache under `results/cache/ecg/` (gitignored) in about eight seconds.

### The split is intra-patient, and that is forced

The plan called for inter-patient DS1/DS2 (de Chazal), correctly: an
intra-patient split shares beats from one patient across train and test and
inflates every number.

**It cannot be produced from this data.** The curated CSVs are
`(n_beats, n_timesteps)` matrices with record identity dropped during
preprocessing. There is no column, no index and no side file saying which patient
a row came from. The choice is between the thesis' own protocol and a *different
dataset* — and a different dataset means the thesis baseline no longer applies,
which is the one thing this recalibration exists to prevent.

So: stratified random 1664/336, the thesis protocol. **Every ECG number this
project reports is intra-patient and must say so.** Recovering DS1/DS2 means
going back to the raw PhysioNet records and re-running the delta encoder with
record IDs kept. Real work, worth doing, not a quiet substitution.

Thesis reference points on this exact arrangement: software full-precision
accuracy 0.857 / macro-F1 0.793; on 15 measured FeFET levels 0.830 / 0.754.

## 4. The network is the thesis LSNN

`src/diffsilicon/snn/lsnn.py`

    input (3) --DelayedLinear--> [ LIF x100 | ALIF x60 ] --LP--> Linear(4)
                                    ^                |
                                    |_DelayedLinear__|
                                      (diag disconnected)

Every number from the thesis' deployed config (`post_quantize.py`, and the
training command in its README/METHODOLOGY): `num_in=3, num_lif=100,
num_alif=60, num_out=4, max_delay=10, refractory=5, dt=5.556e-4,
tau_mem=tau_lp=11.11e-3, Ca=5556e-9, Ra=100e3`. Reference: Yuan et al.,
*Nat. Commun.* 14:3695 (2023); LSNN after Bellec et al., NeurIPS 2018.

The device still supplies `beta`, `th_th`, `g_min`, `g_max`, `sig_w`, through
`LIFNet._weights` **unchanged**. The readout stays full-precision CMOS, as the
thesis labels it. The VJP wiring was not touched.

### The timestep agreement — the part worth putting in the writeup

A beat is 1116 steps of 0.5556 ms. Pooling 10 gives a **5.556 ms** timestep,
which is the frozen `dt_bio = 5.625 ms` to 1.2%, so `V_leak`, `K_syn`, `dt_hw`
and `th_th` all keep meaning exactly what they meant. Nothing frozen had to move
to accommodate real data.

And at that timestep:

| | |
|---|---|
| thesis membrane decay, exp(−dt/τ), τ = 11.11 ms | **0.6065** |
| this project's device-derived β at the nominal device | **0.6033** |

Those come from completely different places — one from an RC circuit fitted to a
VO₂ neuron, the other from a FeFET's subthreshold leak through the DPI relation
τ = C_mem·SS/(ln10·I_τ). They agree to half a percent. That is a **check** on the
alignment, not an input to it.

The one thing pooling moves is `A_accel`, which becomes 505 rather than the
assumed 512. `A_accel` is labelled ASSUMED in `circuit.yaml` and enters no frozen
derivation.

### Two departures, both forced and both stated

- **Refractory period.** The thesis holds a neuron silent 5 steps of 0.5556 ms =
  2.8 ms. At the pooled timestep that is under one step, so the counter is
  dropped rather than rounded up to one step — which would impose 5.6 ms, twice
  the baseline's.
- **ALIF adaptation.** Ported in mechanism and time constant (spike-driven,
  subtractive, τ = Ca·Ra = 0.5556 s), not in MOSFET constants. The thesis'
  `ALIFVO2` works in volts against Vdd = 5 V and v_threshold = 3.6 V; this
  network is in normalised spikes-to-fire units because that is what the device
  hands it, so copying those numbers would be worse than useless. The form used
  is the canonical LSNN adaptive threshold of Bellec et al. — the paper the
  thesis' own `model.py` cites for this layer. `ADAPT_BETA = 1.8` is the one
  coefficient of the port that is a choice rather than a transcription, and it is
  named as such in the source.

## 5. The extraction is fixed for the widened window

This is the one that was actively producing wrong numbers.

**Symptom**, on the commercial solver's own curves: SS = **9618 mV/dec**, memory
window **30 V**, on curves that were themselves fine.

**Two causes, one mistake** — naming a current level instead of a place on a
curve. The D1 window centred on a fixed I_ref = 1e-10 A, which worked while the
sweep was [−1.2, 1.4] V and 1e-10 A occurred once per branch, on the turn-on.

1. **The floor sits within one decade of I_ref.** The erased branch spends −3.5 V
   to −0.5 V between 1.1e-11 and 2.3e-11 A — 0.6 to 1.0 decades away — so at
   σ_dec = 0.6 fifty-odd near-flat points got as much weight as the handful on
   the real turn-on.
2. **Part of that floor has NEGATIVE slope.** It is the ambipolar/GIDL tail,
   falling from −10.64 to −10.95 decades as V_g rises. The regression averaged a
   falling branch against a rising one, the covariance collapsed towards zero,
   and SS = s_vv/s_vl exploded. That is where 9618 came from, and why the number
   was large rather than merely wrong.

**Fix:** name the property, not the current. Subthreshold is the steep, rising,
log-linear stretch, so the window is now built from the local log-slope —
`exp(s / t_slope)` — and re-centres on the current and the voltage that steep
region actually occupies. No argmax, no selection; a softmax over a smooth slope
is smooth in theta, which is the property the whole module exists for.

**The voltage term is not redundant, and leaving it out is a bug that survives
inspection of the weights.** Weighted least squares gives a point leverage
proportional to (v − v̄)². The far end of a 5 V sweep sits four volts out, so its
leverage is ~1600× a neighbour's and a weight of 1e-5 does not begin to pay for
that. Measured: with only the current-domain window the weights were textbook —
0.41 / 0.27 / 0.21 on the three steepest points, under 1e-3 everywhere else — and
SS still came out at 237 mV/dec instead of 69.

Result on the same four real curves:

| | before | after | hand-read from the curve |
|---|---|---|---|
| SS | 9618 mV/dec | **68.1** | ~67–69 |
| memory window | 30.06 V | **2.132 V** | ~2.13 |
| I_leak at V_leak | 3.6e-11 A | **4.36e-11** | 4.32e-11 |

SS of 68 mV/dec sits inside the calibration's own 45–75 mV/dec.

### Two things fixing SS exposed

- **I_leak was an extrapolation, and it is now read off the curve.** The fitted
  subthreshold line reaches 1.4e-15 A at V_leak; the device actually draws
  4.4e-11 A there, because it bottomed out on its floor two volts earlier. Four
  decades. It was hidden because the old SS of 9618 mV/dec is almost a flat line,
  so extrapolating it barely moved and I_leak came out plausible for entirely the
  wrong reason. I_leak sets the DPI leak current and hence β, so this is not
  cosmetic. `analytic_foms` in the mock was updated to match — the curve is the
  truth and the line was an approximation to it, not the other way round.
- **The local-polynomial window was half a grid cell.** `sigma_v = 25 mV` was
  "slightly under one grid spacing" when 96 points spanned 2.6 V. The same 96
  points now span 5.0 V, so a spacing is 52.6 mV: ten coefficients over three
  weighted points, held up only by the Tikhonov floor. The width is now set from
  the grid (`sigma_v_grid_frac`), and the degree went 5 → 9 to cover it. Measured
  noise sensitivity on real curves: 7.6e-7 at the width used, 4.9e-2 at the old
  one.

### THE CACHE INVALIDATION — read this before replaying anything

A cache record stores the seven **extracted** figures of merit, not just the raw
curve, and the extraction is Python in this repository. The key did not cover it.
Without a fix, SS = 9618 would have been replayed for the same theta
indefinitely, with the provenance log cheerfully recording that a real solver
produced it.

`shared.cache.cache_key` now folds a hash of `extract.py` into the key on **every**
backend. Consequence: **every cached DEVSIM and mock result is invalidated and
must be recomputed.** That is the intended cost. `results/cache/sentaurus/` was
already empty.

---

### One scale change that would otherwise be silent

`LSNNNet` returns `spikes` divided by the neuron count; `LIFNet` does not, so its
value is a whole-network spike rate rather than the per-neuron figure the output
schema advertises. The new number matches its own description, but it is ~160x
smaller than the synthetic task's.

**So `lambda_e` cannot be carried over.** It needed picking anyway — at 1e6 the
energy term was 9e-9 x spikes against a loss of order 1, i.e. "energy-aware" with
no term behind it — but it must now be set against this scale. That is still an
open item on the D3 list.

## What is NOT changed

- The gradient path, the shim, the frozen contract, the schema.
- `V_leak = 0.246391250`, `K_syn = 5.450675e-05`, `dt_hw = 11.0 µs`, `th_th = 5`.
- The memory-window sign convention (forward = erased = high V_th).
- `LIFNet._weights`, including `dL/dg_min ≈ 1e-13` being correct by construction.
- The surrogate gradient: still the exact derivative of `soft_heaviside`.
- The sweep window itself, [−3.50, 1.50] V, 96 points. It was right; the fit was
  wrong.

## The measured settings, so nobody re-guesses them

Nominal d=4 device, batch 16, 300 Adam steps on the real task
(`results/runs/snn_calibration_d3.json`):

| step | train CE | test acc | test macro-F1 |
|---:|---:|---:|---:|
| 0 | 1.4992 | 0.297 | 0.178 |
| 100 | 1.2888 | 0.359 | 0.269 |
| **150** | 1.2446 | 0.484 | **0.461** |
| 200 | 1.2089 | 0.453 | 0.412 |
| 299 | 1.1552 | 0.484 | 0.436 |

Test macro-F1 plateaus around step 150 while train CE keeps falling — past that,
the extra steps buy overfitting to a 16-beat batch, not generalisation. So
`SNN_TRAIN_STEPS = 150`, not D2's 400.

**Cost, which sets the call budget.** ~3.0 s per Adam step at batch 16, so 150
steps is ~7.5 min per design point and the flagship pays it once per oracle call.
65 calls is ~8.1 h, past the overnight window; **45 calls is ~5.6 h** plus ~27 min
of solver. If it must be shortened further, cut `SNN_TRAIN_STEPS` to 100 (~5 min a
point) before cutting calls.

**Batch scales sublinearly** — the 111-step Python loop dominates the forward
pass. Batch 16 is 0.65 s forward / 2.02 s backward; batch 64 is 0.84 s / 5.89 s.
Four times the batch is 2.5 times the time, if a better gradient is ever worth
buying.

### The design box discriminates, and no single figure of merit orders it

Eight points of the d=4 box, each trained from scratch under its own device, 150
Adam steps, one seed (`results/runs/snn_calibration_d3.json`), sorted by loss:

| point | t_fe | L_g | log N_ch | t_IL | MW | beta | th_th | CE | acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand0 | 11.37 | 30.8 | 16.08 | 0.52 | 0.681 | 0.056 | 1.11 | **1.0554** | 0.438 |
| rand1 | 13.13 | 56.5 | 17.21 | 1.59 | 0.787 | 1.000 | 4.98 | 1.0743 | 0.500 |
| rand4 | 13.63 | 41.7 | 16.60 | 1.13 | 0.816 | 0.993 | 2.01 | 1.1159 | 0.500 |
| nominal | 7.00 | 40.0 | 17.00 | 1.00 | 0.419 | 0.713 | 4.84 | 1.2446 | **0.688** |
| rand2 | 10.44 | 57.4 | 17.63 | 0.50 | 0.625 | 1.000 | 8.65 | 1.2543 | 0.438 |
| rand3 | 13.57 | 21.3 | 17.46 | 0.76 | 0.813 | 0.839 | 0.93 | 1.3129 | 0.500 |
| corner-poor | 5.50 | 56.0 | 17.80 | 1.85 | 0.329 | 0.992 | 15.86 | 1.3239 | 0.438 |
| corner-rich | 14.50 | 24.0 | 16.20 | 0.65 | **0.868** | 0.177 | 0.52 | 1.3425 | 0.250 |

**The loss separates devices: CE spans 1.0554 to 1.3425, a spread of 0.287.**
That is the D2 worry answered. The synthetic task had saturated — accuracy 1.000
almost everywhere, the landscape near-binary — and this one has structure.

**No single figure of merit orders the table.** Correlation of CE against each:

| | MW | beta | th_th | sig_w | SS |
|---|---:|---:|---:|---:|---:|
| corr with CE | −0.24 | +0.07 | +0.30 | +0.20 | +0.45 |

Every one is weak. The device with the largest memory window (corner-rich, 0.868
V) is the worst in the table, and the device with the second largest (rand4,
0.816 V) is third best — so the window does not decide it either way. SS is the
strongest single predictor at +0.45, and that still leaves most of the variance
unexplained.

**A CORRECTION, recorded because it nearly became the headline.** On the first
three points this read as "the biggest memory window is the worst device", and it
was written up that way. Across all eight that claim does not survive: the
MW-vs-CE correlation is −0.24, i.e. weakly in the *opposite* direction. Three
points were not enough and the story was too tidy.

What is defensible is the weaker, more useful statement: **the objective depends
on the combination of the device's figures of merit, not on any one of them, so
there is no scalar proxy to maximise instead.** That is the argument for
gradient-based co-design over "just make the memory window bigger", and it is the
one to put in the writeup.

Two caveats that belong next to these numbers. **Eight points at one seed is thin
for a correlation** — treat the table as evidence that the landscape has
structure, not as a measurement of which knob matters. And **accuracy is coarse
here**: 16 beats means it moves in steps of 0.0625, which is why CE and accuracy
disagree on rand0.

**These absolute numbers are not comparable to the thesis' 0.793 macro-F1.** That
is a fully trained classifier: 1664 beats, 60+ epochs, LR schedule. This is a
proxy trained on one fixed 16-beat batch, because the flagship pays for it 45
times. What matters for the optimiser is that the loss *separates devices*, not
that it is a good classifier — and that is what the design-box sweep measures.

## What still needs doing

- `python scripts/rebaseline_d3.py --backend devsim` — the cache is cold by
  construction now, so this is not optional. The mock run is already banked at
  `results/runs/rebaseline_d3_mock.json`.
- `bash scripts/overnight_d3.sh` — the flagship on d=4. `overnight_d2.sh` is
  superseded and will be refused by the material lock.
- V1 α conditioning study and V2 refresh-K, per the D3 task list. Neither was
  touched today.
- Bank Sentaurus points into `results/cache/sentaurus/`; it is still empty.
- The writeup must state, in one sentence each: the material is locked and why;
  the split is intra-patient and why that is forced; and that the reported ECG
  numbers come from a deliberately cheap inner training loop.
