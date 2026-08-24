# Handoff — D3, written 2026-08-24 evening, REVISED after the recalibration

Read this before touching anything. It is written for an agent starting cold.

---

## Rule zero: how to talk to Hozaifa

**Plain, simple English. Short sentences. No jargon.** Explain any term the first
time it appears. Do not use internal labels (T1, T2, G5, V4, D3) without saying
what they mean. Lead with the answer, then the reason. Keep it short.

He said this directly, after several replies he could not read. It is not a style
preference — if he cannot read it, he cannot check the work or steer the project.

## Rule one: read his existing work before writing any

His own working code almost always exists. On D2 five Sentaurus runs were wasted
reverse-engineering a simulation deck that was already sitting, finished and
calibrated, in `F:\RESEARCH\FeFET x ML\TCAD Files\GAAFet\Simulations\Calibration\autocal\`.

Search `F:\RESEARCH\` and `F:\Projects\` first. `Project_Documentation/` folders
hold his own runbooks, passwords and gotchas.

## Rule two: his thesis is confidential

The repository is **public**. His GAA-FeFET calibration is unpublished thesis
work heading for IEEE TED.

Already gitignored, and must stay that way:

- `t1/sdevice_fefet_idvg.cmd`, `t1/sdevice_fefet_idvg.par` — his calibrated deck
- `t1/calibration.local.json` — his fitted constants
- `.env` — the solver-host password

Never put his fitted numbers, his protocol, or his device's identity into tracked
source, docs, or commit messages. `t1/calibration.example.json` shows the shape
with no values.

---

## Where the project stands

**Works and is banked:**

- The open solver (DEVSIM, Apache-2.0) simulates the ferroelectric transistor.
  ~36 s per design point. This carries the whole pipeline.
- The full loop runs unattended: a spiking-network loss, differentiated back
  through a circuit model and into the solver, driving an optimiser.
- D2 overnight, on the OLD voltage window AND the old design vector: 3 design
  variables, 45 solver calls, 28.6 min, loss 1.2289 -> 0.0223. 5 variables, 57
  calls, loss -> 0.2328. **Superseded** — both tuned Pr and Ec, which are locked
  now, and much of that win was bought by moving Pr from 13 to 23.4.
- The commercial solver (Sentaurus) is reachable and produces real hysteresis
  through the same interface. 200 s per point.

**Changed on D2 evening, re-validated on D3:**

- **The sweep window widened from [-1.20, 1.40] V to [-3.50, 1.50] V.** The real
  device's memory window does not fit in 2.6 V — measured, the programmed state
  never switches off anywhere in the old window. This invalidates every cached
  result automatically (the voltage list is part of the cache key).
- **The Pr search range widened from [5, 25] to [5, 40] uC/cm2.** The real film
  sits above the old ceiling, so the optimiser could never reach it. **This is
  now moot** — Pr is not searched at all; it is fixed at the measured 32.
- Both mean **last night's results must be recomputed.** They are not wrong, they
  describe a narrower experiment. The D2 numbers just below are kept for the
  record, but they were obtained while the material was still adjustable, so they
  are not the result this project now reports.

**FIXED on D3, 2026-08-24 — the recalibration. Read `docs/D3_RECALIBRATION.md`.**

Five things were realigned with his thesis. In plain terms:

1. **The remanent polarization and coercive field are no longer things we tune.**
   They are the measured film: 32 uC/cm2 and 1.4 MV/cm, from his own calibration
   (locked node `cal_n16`). You cannot order a different remanent polarization
   from a fab; you can only deposit a different film. The optimiser now refuses
   to touch them, in code, not by convention.
2. **The things we DO tune are the four a process engineer can actually ask
   for:** ferroelectric thickness, gate length, channel doping, interfacial-layer
   thickness. That is the new `d=4` design vector. The nominal film is 7 nm, his
   calibrated thickness.
3. **The task is his real ECG data** — 2000 heartbeats, four classes, loaded from
   his own preprocessing. Note: the split is intra-patient, because the curated
   files have the patient identity stripped out, so an inter-patient split cannot
   be built from them at all. Say so in the writeup.
4. **The network is his LSNN**, not one invented here — 100 LIF plus 60 adaptive
   neurons, 10 synaptic delay taps, low-pass readout. One nice result fell out:
   the membrane decay his circuit uses (0.6065) and the one our DEVICE produces
   (0.6033) agree to half a percent, and they were derived completely separately.
5. **The curve-fitting bug is fixed.** It reported a subthreshold slope of 9618
   mV/dec and a memory window of 30 V; it now reports 68.1 mV/dec and 2.132 V on
   the same real curves, which is what those curves say by hand. Cause: the fit
   aimed at a fixed current level, and on the wider sweep that level sits in the
   flat leakage floor — half of which slopes the wrong way.

**Consequence you must not skip: every cached device result is now stale and is
refused rather than served.** The fingerprint of the fitting code is part of the
cache key on every backend. So the DEVSIM numbers have to be recomputed before
anything is compared to anything.

**One more thing, and it is a limit rather than a bug.** On the COMMERCIAL solver
only the film thickness actually reaches the simulation. Gate length, channel
doping and interfacial-layer thickness are baked into the mesh, and this driver
does not rebuild the mesh for each design point. So on Sentaurus, three of the
four design variables do nothing, and their sensitivity comes back as exactly
zero. The open solver rebuilds its mesh every time, so all four work there, and
the main run uses the open solver. When comparing the two solvers, compare the
thickness column only. Do not read the three zeros as the two solvers agreeing.

---

## What the 3am run should do, in order

Everything below is unattended-safe. Stop and report rather than guess.

### 1. Check the open solver still converges on the wider window

It now sweeps to -3.5 V, deeper into accumulation than before, and that region
was already the hard part. One design point on the new vector:

```
python scripts/rebaseline_d3.py --backend devsim --points 0
```

That evaluates the nominal d=4 device and the two thickness corners, prints the
figures of merit, and PASS/FAIL for the memory-window gate. If it fails to
converge, **stop and report**. Everything else depends on it.

### 2. Fix the figure-of-merit extraction — ALREADY DONE, 2026-08-24

**Do not redo this.** `shared/extract.py` was rewritten and the fix is verified
against the commercial solver's own curves: subthreshold slope went from 9618
mV/dec to 68.1, and the memory window from 30 V to 2.132 V, against ~67-69 and
~2.13 read off the curve by hand. All the analytic tests still pass at 0.5%, and
`tests/test_extract_wide_window.py` is a regression test on a curve shaped like a
real one (floor, ambipolar tail, steep turn-on) that the old code fails hard.

Full reasoning: **`docs/D3_RECALIBRATION.md`**. Read it before touching anything
device-side. The three things to know here:

1. The window is now built from the local log-slope, not from a fixed current.
2. `I_leak` is read off the curve, not extrapolated down the fitted line — the
   extrapolation was four decades wrong on a device with a leakage floor.
3. **THE CACHE IS COLD.** `cache_key` now folds a hash of `extract.py` in, on
   every backend, so nothing stale can be served. Every DEVSIM and mock result
   banked before today has to be recomputed. That is intended, not a fault.

### 3. Re-run the baselines — THE CACHE IS COLD, SO THIS IS NOT OPTIONAL

```
python scripts/rebaseline_d3.py --backend devsim
```

Nine design points on the recalibrated d=4 vector, one JSON in `results/runs/`,
and it prints PASS/FAIL for the memory-window gate and the sign convention. About
six minutes at ~36 s a point. The mock run is already banked at
`results/runs/rebaseline_d3_mock.json`.

Then the optimisation runs:

```
bash scripts/overnight_d3.sh
```

**Note the `d3`.** `overnight_d2.sh` is superseded and will refuse to run: it
passes `--d 3` and `--d 5`, and both expose `Pr` and `Ec`, which are locked
material constants of the calibrated film. The design vector is **d=4** —
`t_fe`, `L_g`, `N_ch`, `t_IL`. If a pre-recalibration run genuinely needs
reproducing, `DIFFSILICON_ALLOW_LOCKED_MATERIAL=1` is the explicit escape hatch.

**The d=4 path was smoke-tested end to end** before this handoff was written:
13 oracle calls on the mock, one Jacobian refresh, four Broyden updates, one
trust-region rejection, no errors. That says the PLUMBING works on the new design
vector -- it does not say the optimiser converges, which is what the real run is
for.

`overnight_d3.sh` does the re-baseline, then the d=4 flagship, then the
cross-check. It is set to **45 solver calls, not 65**, and
`SNN_TRAIN_STEPS = 150`. Both are measured, not guessed: the inner training loop
costs ~3.0 s an Adam step, so ~7.5 min per design point, and 65 calls would run
~8.1 hours — past the window. If it has to be shortened further, cut
`SNN_TRAIN_STEPS` to 100 before cutting calls; the loss still separates devices
there, it is only a weaker classifier.

### 4. Bank commercial-solver results for replay

Hozaifa chose to publish the current-voltage curves (not the deck, not the
constants). So run the commercial solver at a handful of design points and let
the cache fill. 5 to 8 points is enough for the figures.

**Licence safety — there is exactly ONE licence and it is shared:**

- Never run two at once.
- A run that times out locally keeps going remotely and holds the licence. Free
  it with `pkill -9 sdevice` over ssh.
- Check before starting: `ps -ef | grep -v grep | grep -c sdevice` should be 0.
- Stop by 06:00 so he has the machine back.

### 5. Report in plain English

What ran, what the numbers were, what broke. Short.

---

## Decisions already made — do not reopen

- **The commercial solver stays in the submission, and its curves get published.**
  He chose this explicitly over dropping it, because it is the strongest claim.
  The deck and the fitted constants stay private; only numeric outputs go public.
- **A generic ferroelectric device should eventually replace his calibrated one**
  for the published tier. His idea, and a good one. It must still be a
  ferroelectric transistor, not a plain MOSFET — the whole claim is that both
  solvers answer the SAME question, and the design knobs are ferroelectric ones
  (film thickness, Pr, Ec) that a plain MOSFET does not have. Build the geometry
  with `sde` on the host from a self-authored deck, with textbook parameters.
  This is a nice-to-have, not a blocker.
- **Do not tune his device physics.** That is his thesis. **Updated on D3:** this
  project no longer varies Pr, Ps or Ec at all — they are locked to his measured
  film. What the commercial deck now varies is the ferroelectric THICKNESS, and
  only through the fixed-slab remap in `t1_driver.deck_values`, which is exactly
  the identity at his calibrated 7 nm. His fitted numbers are untouched.
- **The ECG split is intra-patient, and that is forced, not chosen.** The curated
  beat files have record identity stripped during preprocessing, so an
  inter-patient DS1/DS2 split cannot be built from them. Do not "fix" this by
  swapping in a different dataset — that would break the comparison to his
  thesis, which is the whole point of the recalibration. It must be stated as a
  limitation in the writeup.

## Open questions for him — ask, do not decide

- The open solver shrinks Pr by 26x (`FE_ACTIVE_FRACTION` in `oracle_devsim.py`)
  to keep its idealised model realistic. The commercial solver gets the raw
  value. That is now the one place the two describe different devices. Removing
  it would make them match, but it re-calibrates the open solver and invalidates
  results again. He knows about it and has not decided.
- The energy term in the objective is inert — nine orders of magnitude too small
  to matter. "Energy-aware co-design" is currently a claim with no term behind
  it. **One extra wrinkle from D3:** the new network divides its spike count by
  the neuron count and the old one did not, so the number is ~160x smaller now.
  Pick `lambda_e` against the new scale; do not carry a value over.
- **DONE, no longer an open question:** the synthetic task saturated (nearly
  every device scored 100%) and real ECG data has now replaced it. The landscape
  is not near-binary any more — see below.

## The one result worth showing him

**The new design box separates good devices from bad ones, and no single number
tells you which is which.**

Eight devices, each trained from scratch on the real heartbeat data. Sorted by
loss, best first (lower loss is better):

| device | film | gate | memory window | leak factor β | threshold | loss | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| rand0 | 11.4 nm | 31 nm | 0.681 V | 0.056 | 1.11 | **1.055** | 0.438 |
| rand1 | 13.1 nm | 57 nm | 0.787 V | 1.000 | 4.98 | 1.074 | 0.500 |
| rand4 | 13.6 nm | 42 nm | 0.816 V | 0.993 | 2.01 | 1.116 | 0.500 |
| nominal | 7.0 nm | 40 nm | 0.419 V | 0.713 | 4.84 | 1.245 | **0.688** |
| rand2 | 10.4 nm | 57 nm | 0.625 V | 1.000 | 8.65 | 1.254 | 0.438 |
| rand3 | 13.6 nm | 21 nm | 0.813 V | 0.839 | 0.93 | 1.313 | 0.500 |
| poor corner | 5.5 nm | 56 nm | 0.329 V | 0.992 | 15.86 | 1.324 | 0.438 |
| rich corner | 14.5 nm | 24 nm | **0.868 V** | 0.177 | 0.52 | 1.343 | 0.250 |

Two things to take from it.

**One: the task is no longer saturated.** Loss runs from 1.055 to 1.343. The old
synthetic task scored 100% almost everywhere, so the optimiser had nothing to
climb. This one has a real surface.

**Two: you cannot pick the best device by looking at one number.** The device
with the biggest memory window is last. The device with the second biggest is
third. Checked against every figure of merit the device hands up, the strongest
single link to the loss is the subthreshold slope, and even that only explains a
small part of it:

| | memory window | β | threshold | weight noise | SS |
|---|---:|---:|---:|---:|---:|
| link to loss | −0.24 | +0.07 | +0.30 | +0.20 | +0.45 |

That is the argument for doing this with gradients at all. If one number
predicted performance you would just maximise that number and skip the whole
pipeline.

**A correction, so it does not get repeated.** After the first three devices this
looked like "the biggest memory window is the worst device", and it was written
up that way. Across all eight it is not true — the link runs weakly the other
way. Three points were too few and the story was too neat. The honest version is
the one above.

**Two caveats to keep attached to this table.** Eight devices at one random seed
is thin, so read it as "the landscape has structure", not as a measurement of
which knob matters. And accuracy is coarse — 16 heartbeats means it can only move
in steps of 0.0625, which is why loss and accuracy disagree on rand0.
