# D6 findings — 2026-08-29

Plain English. Short sentences. Terms explained the first time.

Words used below, beyond D5's list:

- **Tier C** = the promise that anyone can regenerate every Sentaurus number in
  this project without a Sentaurus licence, from files in the repository.
- **Bit-identical** = the two float64 numbers are the same number, not the same
  number to some number of decimals. Checked with `==`, and the distance in
  representable floats is reported alongside so a near-miss cannot pass.
- **An orphan cache entry** = a solve this project paid for that nothing in the
  repository reads any more.
- **A structural zero** = a Jacobian column that is zero because the design knob
  never reaches the solver, not because the device is insensitive to it.
- **`train_mode`** = whether the classifier's weights are held fixed at a shared
  starting point (`frozen`), briefly tuned per device (`adapt`), or retrained
  from scratch.

---

## The short version

Three things were open for D6. All three are closed. Two of them turned up
something the day was not looking for, and both of those are worth more than the
task that found them.

1. **Tier C is verified rather than asserted.** Every banked Sentaurus number —
   164 float64 values — regenerates **bit-identically** from
   `results/cache/sentaurus/`, in 1.4 seconds, in a process that cannot open a
   socket or spawn a subprocess. See §1.
2. **Figures 3 and 4 are drawn**, both with zero solver calls. Figure 3 is the
   device moving; Figure 4 is what that does to the classifier. See §3 and §4.
3. **The provenance log is an audit now, not a 1.1 MB file.** All 15 flagship
   steps have a provenance line, and a real solver wrote every one. See §5.

**And the two things the day was not looking for.**

**A reproduction bug, found by checking that a figure matched its own flagship.**
Every banked result in this project was produced under `SNN_TRAIN_MODE=frozen`.
The module defaults to `adapt`, and `run_flagship` set neither — so the command
the README gives returned **1.3152** where the flagship reports **1.3996**, and
nothing on disk said which was which. Fixed at the source and verified. See §2.

**The T1 structural-zero count in the notes is wrong, and wrong in the direction
that misleads.** It says three of four Jacobian columns are identically zero.
Measured against the cache: **two are, and the third is spuriously non-zero** —
it moves the threshold without the solver returning a different device. See §6.

**A crashing solver could kill a whole flagship run, and now cannot.**
`fd_jacobian` salvaged a probe the extraction refused but not a probe whose
solver process died — the second escaped as a bare `RuntimeError`. The V2 run hit
exactly that and lost 25 minutes of solver time. Fixed; the re-run then salvaged
two crashed probes and completed. See §7.

**Everything is committed**, working tree clean. Not pushed — see §8.

---

## 1. Tier C, verified end to end

`scripts/tier_c_replay_d6.py` → `results/runs/tier_c_replay_d6.json`.

Until today Tier C was proven on DEVSIM (D2) and **asserted** for Sentaurus. The
commercial solver is the one nobody reproducing this can run, so it is the one
where the claim actually has to hold.

### How the network is unplugged

Not by asserting it. By breaking it:

- `socket.socket` is replaced by a subclass whose `connect`, `connect_ex` and
  `sendto` raise; `socket.create_connection` and `socket.getaddrinfo` raise
  outright.
- `subprocess.Popen`, `subprocess.run`, `subprocess.check_output`,
  `subprocess.call` and `os.system` raise. `t1_driver` reaches the licensed host
  by shelling out to plink/pscp, so this closes the door the socket guard does
  not.
- `SENTAURUS_HOST`, `SENTAURUS_PASSWORD`, `SENTAURUS_REMOTE_ROOT`, `PLINK`,
  `PSCP` and `SENTAURUS_GRID` are deleted from the environment before anything
  reads them.

One ordering detail, recorded because it cost a debugging cycle: the guards go in
at the top of `main`, not at import. `ssl` subclasses `socket.socket` and
`asyncio` subclasses `subprocess.Popen` at import time, so a guard installed
first takes `import jax` down with it with a `TypeError` that names neither.

If any number below had come from the solver, the script does not produce a wrong
answer — it dies with a traceback.

### The result

| | |
|---|---|
| float64 values compared | **164** |
| bit-identical | **164** |
| replay wall clock | **1.4 s** |
| solver time it stands in for | **0.93 h** |
| speed-up | ~2,500x |

Both banked Sentaurus artefacts, regenerated and diffed field by field:

- `rebaseline_d3_sentaurus.json` — eight design points, sixteen recorded
  quantities each. `solver_seconds` is included, because it lives in the cache
  record and therefore replays exactly: a reproduction that reports how long the
  original solve took is a better artefact than one reporting how long a lookup
  took. `wall_seconds` is the only excluded field, and it is excluded because it
  is the one number that must differ.
- `cross_check_sentaurus_devsim_d4.json` — the V4 cross-solver Jacobian. This is
  the harder half: it runs the shim's own 2D+1 central-difference machinery, so
  nine probes have to find cache entries and the arithmetic on top has to land on
  the same float64s.

**Eight points, not nine.** `rebaseline_d3.py` builds `rand0..rand5`; `rand5` has
no Sentaurus entry and was never solved on the commercial host. The replay
reports it as `not-in-banked-file` rather than skipping it quietly.

### The bonus finding, which is better than the task

The cache holds 32 entries and the two artefacts exercise 16. The obvious thing
to say about the other 16 is "leftovers". The script classifies them instead, and
by the **raw Id-Vg curve** rather than by the extracted figures of merit:

| | |
|---|---|
| exercised | 16 |
| superseded — byte-identical curve to an exercised entry | **16** |
| **orphans — a solve nothing reads** | **0** |
| of the superseded, how many read *different* figures of merit off that identical curve | **16 of 16** |

Two statements come out of that, and the second is the useful one.

**Zero orphans.** Every Sentaurus solve this project ever paid for is read by
something current.

**All sixteen superseded entries hold a curve that is identical to the byte and
seven numbers that are not.** `vth_fwd`, `vth_rev`, `i_leak`, `g_lo` and
`dg_dvth` all move between extraction generations while the solver output does
not. That is `cache_key`'s `_extraction_source_hash` justified by evidence
instead of by its own docstring: an edit to `extract.py` really does change what a
stored record *means*, on every backend — including the one whose curves are
expensive enough that nobody would think to recompute them. Without that hash the
old seven numbers would be served off the right curve indefinitely, with the
provenance log cheerfully recording that a real solver had produced them.

---

## 2. The reproduction bug, and how it was found

### How it surfaced

Figure 4 draws the network at the flagship's `phi_initial` and `phi_final`. Before
drawing anything it checks that it reproduces the flagship's own `ce_initial` and
`ce_final`. It did not:

| | figure | banked | |
|---|---:|---:|---|
| start loss | 1.315219 | 1.399577 | off by 8.4e-2 |
| final loss | 1.017146 | 1.017666 | off by 5.2e-4 |
| final accuracy | 0.6250 | 0.6875 | one beat in sixteen |

A figure that disagrees with the run it illustrates is not a drawing problem.

### The cause

`tesseracts/snn-lif-ecg/tesseract_api.py` reads `SNN_TRAIN_MODE` at import and
defaults to **`adapt`**. Every driver that produced a reported number sets
**`frozen`** for itself — `race_d4.py`, `race_sweep_d5.py`, `v6_free_refit_d5.py`,
`h_ablation_d5.py`, `smoke_tesseracts.py`, `v6_manifold_control.py`. Two things
did not: `src/diffsilicon/optimise.py` and `scripts/run_flagship.py`.

So the flagship's own numbers were produced under `frozen` — from a shell that
happened to have the variable set — and the command in the README ran under
`adapt`. `result.json` did not record the mode either, so nothing on disk
distinguished the two runs. Under `frozen` every number matches to nine decimals.

### The fix

`FlagshipConfig` gains `train_mode`, defaulting to `frozen`. `run_flagship` sets
`os.environ["SNN_TRAIN_MODE"]` **before** the Tesseract that reads it is
constructed — the ordering is load-bearing and is commented as such — and the
value is serialised into `result.json` with the rest of the config.
`scripts/run_flagship.py` gains `--train-mode`.

**Verified, not assumed.** A 10-call run under the new default:

```
=== d6-trainmode-check (devsim, d=4, train_mode=frozen) ===
objective   1.399577 -> 1.293473
theta (phys) t_fe 5.838876  L_g 49.754159  log10_N_ch 17.743642  t_IL 1.495808
```

`steps.jsonl` step 0 of the banked flagship: loss 1.3995774659, `loss_try`
1.2934729294, `theta_next` 0.08388757332989662 → `t_fe` 5.838875733. Identical.
Banked at `results/runs/d6-trainmode-check/`.

### And `frozen` is right on the merits, not merely conventional

It is worth stating positively rather than as a bug fix. `frozen` is what the
thesis does — train in software, deploy onto measured FeFET levels — and it makes
the VJP **exact rather than approximate**: the envelope-theorem argument for
holding `W` fixed while differentiating `L(phi; W*(phi))` needs a stationary
`W*`, and a `W` that never moves is trivially stationary.

---

## 3. Figure 3 — the device the optimiser is actually moving

`docs/figures/fig3_hysteresis_descent.png` / `.pdf`, drawn by
`scripts/fig3_hysteresis_descent.py` from `results/cache/devsim/`. **Zero solver
calls**; if a design point were missing from the cache the script fails rather
than drawing a plausible curve.

Figures 1 and 2 are arguments about method. Neither shows a device. This one
does, and it is what makes the project legible to somebody who reads transistors
rather than optimisers.

**(a) The loop opens.** The Id-Vg double sweep at each of the nine distinct
points the descent stood on — forward branch (erased, high V_th) and reverse
(programmed, low V_th) — coloured by solver calls spent. The gap between them at
the constant-current criterion is the memory window, and it widens **0.415 V →
0.576 V**. The threshold marker is drawn per curve rather than as one line,
because `I_crit = 100 nA · W/L_g` moves when the optimiser moves the gate.

**(b) Slope traded for window, on purpose.** Memory window against subthreshold
slope, step by step. They rise together: SS degrades **71.1 → 97.4 mV/dec**. To
anyone who designs transistors for switching that reads as a defect. It is the
correct trade here — a thicker HZO film and a thinner interlayer buy window at the
cost of electrostatic control, and the network needs window, because window is
what separates the two conductance states the synapse stores. The optimiser found
that without being told the trade existed. It is the same trade the V7 ablation
could not find.

**(c) Where the four knobs went.**

| knob | start | final |
|---|---:|---:|
| t_fe | 5.50 nm | **7.65 nm** |
| L_g | 52.00 nm | 35.43 nm |
| log10 N_ch | 17.80 | 17.17 |
| t_IL | 1.55 nm | 1.37 nm |

One presentation note recorded because it changes what the figure shows: the
y-axis is clipped at 1e-13 A. Below that the DEVSIM curves sit on the extraction's
1e-16 floor and what is plotted is the solver's numerical noise, which takes two
thirds of the axis and hides the switching region.

---

## 4. Figure 4 — and it did not show what it was drawn to show

`docs/figures/fig4_spike_raster.png` / `.pdf`, drawn by
`scripts/fig4_spike_raster.py`. Zero solver calls: `phi` comes out of the banked
flagship JSON, which is where the solver already put it.

The figure was drawn expecting a quiet layer to become an active one. **That is
not what happens**, and the honest version of the figure is better than the one
that was planned.

| | start | final |
|---|---:|---:|
| population firing rate | 0.4344 | 0.4572 |
| per-neuron rate profile | — | correlation **0.9999** |
| individual spikes that move | — | **8.93%** |

The layer does not fire more. It fires *differently*, by about one spike in
eleven, and all 160 neurons shift a little. Two dense rasters side by side would
have shown nothing, and a figure that shows nothing under a caption claiming a
transformation is worse than no figure. Panel (a) draws the **difference**:
spikes common to both devices in grey, start-only in blue, final-only in green.

**And the point of the 8.9% is panel (b).**

```
truth:              S N V S F N N V F V F N V F S S
start predictions:  F F F F F F F F F F F F F F F F     <- one class, every beat
final predictions:  S N N S S N N V F V S N N V S S     <- 11 of 16
```

On the start device the network answers **class F for all sixteen beats**. Its
accuracy of 0.250 is exactly the score of a constant answer, and the four
"correct" beats are the four that happen to be F. **It is not a weak classifier,
it is a collapsed one.**

So the claim is not "the network improved". It is: *moving one spike in eleven is
what separates a readout stuck on a single output from a working one.* That is
sharper, and it is what the numbers say.

Panel (c) is per-class accuracy — 0/0/0/1.00 for start against 1.00/1.00/0.50/0.25
for final. The three zero bars are labelled, because a bar of height zero is
invisible and here the zero *is* the finding.

The figure refuses to draw itself if either column stops reproducing
`ce_initial` / `ce_final` to nine decimals. That check is what caught §2.

`LSNNNet.forward` gained an optional `spike_log` out-parameter (default `None`)
so the raster can record what the network emitted. An out-parameter rather than a
second return value: the return signature is what `tesseract_api` and every test
call, and a figure has no business changing it.

---

## 5. G10 — the provenance log, turned into an audit

`scripts/provenance_audit_d6.py` → `results/runs/provenance_audit_d6.json`.

1.1 MB of JSONL is not evidence, it is a file. Four statements, each checkable:

**What produced the numbers.** The log grows every time the repository is
exercised, so any count here is a snapshot; these are the figures from the final
D6 audit, and `results/runs/provenance_audit_d6.json` is always authoritative.
5,910 forward evaluations over 1,805 distinct design points:

| backend | calls | distinct points | solver time |
|---|---:|---:|---:|
| devsim | 5,648 | 1,747 | 32.26 h |
| mock | 223 | 119 | 0.02 h |
| sentaurus | 39 | 16 | 2.26 h |

The unconverged evaluations are logged rather than dropped — refused by the
extraction, not silently used.

**Every step of the flagship is in it.** All 15 steps of `flagship-d4-fixed` have
a provenance line at their recorded `content_hash`, a real solver wrote every one,
and the backend each step recorded for itself agrees with the log. Step 0 shows
`devsim,mock` because that same design point was also evaluated on the mock at
some other time by a gradient check; the step's own recorded backend is `devsim`,
which is checked explicitly.

**Nothing reported came off the mock.** Zero flagship steps missing, zero served
only by the mock. 57 design points anywhere in the log have the mock as their only
source; none of them is a flagship step.

**The span, stated honestly. This is a correction.** The log runs
**2026-08-26 14:05 → 2026-08-29 08:09**, *not* "since D1" as the D6 task note and
earlier docs say. The D3 rewrite of `shared/extract.py` re-keyed every cache entry,
and the log was restarted with the cache it describes rather than left to mix two
incompatible generations of the same field names. **The writeup must not say
"appending since D1"** — it is wrong and it is checkable in thirty seconds.

---

## 6. A correction: the T1 structural zeros are two, not three

`scripts/t1_structural_zeros_d6.py` → `results/runs/t1_structural_zeros_d6.json`.
Zero solver calls; every point is in the Sentaurus cache.

Since D4 the notes have said: on the commercial solver's fixed-mesh path only
`t_fe` reaches the deck, so *three of four* T1 Jacobian columns are identically
zero. Measured at the cross-check design point, that is not what is on disk:

| column | Id-Vg curve vs centre | figures of merit that move | verdict |
|---|---|---|---|
| `t_fe` | **differs** | all seven | real physics |
| `L_g` | **byte-identical** | `vth_fwd`, `vth_rev` | **spuriously non-zero** |
| `log10_N_ch` | byte-identical | none | structurally zero |
| `t_IL` | byte-identical | none | structurally zero |

**Two columns are identically zero. The third is worse than zero.** The solver
returns exactly the same current at every gate voltage whether `L_g` is 38.4 nm or
41.6 nm — and the threshold still moves, because the constant-current criterion is
`I_crit = 100 nA · W/L_g`, so shrinking the gate raises the bar the same curve has
to cross and reads out a different V_th.

Why this matters more than the arithmetic: anyone reading the T1 Jacobian and
finding the `L_g` column non-zero would conclude the fixed-mesh path carries some
gate-length physics. It does not. A zero column announces itself; a column that is
non-zero for the wrong reason does not. The writeup should say **"one real column
(`t_fe`), two structural zeros, and one column that is non-zero through the
extraction's `W/L_g` criterion rather than through the device"**.

This also makes the fixed-mesh limitation something visible in the repository
rather than something inferred from a `.cmd` file nobody can ship.

---

## 7. V2 re-measured — how fast the manufactured Jacobian goes stale

`scripts/v2_broyden_decay_d6.py` → `results/runs/v2_broyden_decay_d6.json`.

`ShimConfig.refresh_every` is 4, and the comment beside it says "K; the V2 cosine
curve revises it on D3, not asserted". That is the right thing to say, and until
today nothing on disk backed it — the curve was measured on D2/D3 and never
banked. A repository that asserts a constant and cites a measurement nobody can
see is one question away from an awkward answer.

The measurement walks the flagship's own accepted path: anchor a central-difference
Jacobian at the first point, patch it by Broyden at each subsequent point from the
secant pair that step supplies free, and rebuild the *true* Jacobian there as well.
Two cosines are reported, because they are different questions:

- **`cos_J`** — mean over the seven rows of the 7×D Jacobian. The shim's own
  accuracy, independent of what the network wants.
- **`cos_g`** — the cosine of the composed gradient, which is what the optimiser
  actually steps along. It can stay high while individual rows rot, if the rows
  that rot are ones the loss is insensitive to.

Cosine and not norm, because the optimiser uses the direction: a model whose
magnitude is wrong takes a badly sized step and the trust region catches it, while
a model whose direction is wrong walks uphill and the trust region only finds out
after it has spent the calls.

The script reports its cache hit rate before it runs and **refuses to call the
solver without `--allow-solver`**, so a run on a clean clone either reproduces
from cache or says plainly which points it cannot serve. 81 probes required, 37
already cached, 44 solved today on DEVSIM.

### The curve

81 probes over nine design points. 70 came from the cache and 11 were solved
today on DEVSIM; 3.8 minutes.

| steps since the anchor | 0 | 1 | 2 | 3 | **4** | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cosine, Jacobian rows | 1.000 | 0.997 | 0.976 | 0.927 | **0.785** | 0.773 | 0.809 | 0.937 | 0.935 |
| cosine, composed gradient | 1.000 | 0.996 | 0.962 | 0.866 | **0.701** | **0.431** | 0.474 | 0.940 | 0.957 |

**`refresh_every = 4` survives the measurement, and only just.** At the fourth
free step the composed gradient still points 0.70 of the way toward the truth,
which is above the ~0.7 a line search needs. At the fifth it is 0.43. The
constant sits exactly at the last usable step, which is what a well-chosen
budget should look like and is not what an arbitrary one would look like.

### And the curve is not monotone, which is the more useful finding

It decays through step 5 and then **recovers to 0.94**. That is not noise and it
is not a defect in the measurement; the explanation is in the trust region:

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| \|Δθ\| | 0.080 | 0.128 | 0.205 | 0.328 | 0.328 | 0.082 | 0.082 | 0.020 |

The flagship's trust region grows to 0.33 through steps 4–5 — exactly where the
cosine is worst — and then collapses to 0.08 and 0.02, and a rank-one patch over
a step that short barely has to be right to still point the right way.

Over these nine points, **distance from the anchor predicts the row cosine
(r = −0.79) about twice as well as step count predicts the composed gradient
(r = −0.36)**. The model goes stale with *how far you have walked*, not with
*how many times you have stepped*.

**That is a justification for the design rather than an embarrassment for it.**
`AdjointShim.jacobian` already refreshes on either of two conditions —
`steps_since_refresh >= refresh_every` **or**
`||theta − theta_anchor|| > radius` — and this measurement says the second is
the one carrying the load. K = 4 is a proxy for a distance budget, and it is a
proxy that holds only while the step size holds. Both tests are needed and the
writeup should say which one is doing the work.

Figure 5 draws all of it: `docs/figures/fig5_jacobian_and_decay.png` / `.pdf`.

### The bug this measurement found, which is worth more than the curve

**The first run of this died**, at the seventh of nine design points, after 25
minutes of solver time, none of it recoverable:

```
devsim_py3.error: Convergence failure!
There was a floating point exception of type "Invalid, Divide-by-zero"
during LU Factorization
```

`fd_jacobian` has salvaged a failed probe since D4 — one bad side falls back to a
one-sided difference against the centre rather than losing the column. It
salvaged exactly **one of the two ways a probe can fail**:

- `OracleNotConverged` — the solver returns numbers the extraction refuses (a
  threshold read past the end of the sweep, a non-finite current). Caught,
  recorded, salvaged.
- **The solver process dying** — DEVSIM runs out of process, and
  `oracle_devsim._solve_out_of_process` turns a non-zero exit into a plain
  `RuntimeError`. That sailed straight past `_try` and out of `fd_jacobian`.

`OracleNotConverged`'s own docstring says a refusal is *"handled exactly like a
solver crash"*. A solver crash was not handled at all. On the flagship, one
unlucky probe of nine costs the whole run and every call already paid for.

Fixed: both failures now cost their own side, and they are recorded separately in
`ctr.refused` — `extraction-refused` versus `solver-crashed` — because a device
the extraction cannot read is a statement about the design box and a solver that
died is a statement about the solver. `OracleBudgetExhausted` is re-raised
explicitly, since it is also a `RuntimeError` and swallowing the budget cap as a
bad probe would turn a hard limit into a silently degraded Jacobian. A systematic
failure still stops the run, because the caller raises when *both* neighbours of a
coordinate are gone.

**The re-run then completed all nine points**, and the fix earned its keep on the
way: **two probes crashed and were salvaged one-sided**, both on the `L_g`
coordinate, at θ = [0.217, 0.322, 0.581, 0.584] and [0.265, 0.366, 0.583, 0.583].
Under the old code either one would have ended the run.

`v2_broyden_decay_d6.py` also now banks what it measured instead of losing
everything to a late failure, and reports salvaged probes and one-sided columns.

### Figure 5, panel (a) — the Jacobian itself

The 7×4 matrix at the flagship's starting corner, in relative units
`d log(FoM)/dθ`, from nine cached DEVSIM probes:

| | t_fe | L_g | log10 N_ch | t_IL |
|---|---:|---:|---:|---:|
| subthreshold slope | +0.10 | −0.38 | −0.04 | +0.11 |
| forward threshold | +0.59 | +0.14 | +0.50 | +0.02 |
| reverse threshold | −1.64 | +0.39 | +1.40 | +0.06 |
| **leakage current** | **−11.14** | **−8.74** | **−10.89** | +1.00 |
| **low conductance** | **−10.91** | −3.67 | **−9.08** | −0.32 |
| high conductance | +0.90 | −1.01 | −0.91 | −0.61 |
| conductance slope | −0.83 | −0.93 | −0.33 | −0.58 |

Drawn on a **symmetric-log** colour scale, not a linear one. Two rows — leakage
current and low conductance — respond about eleven times more strongly in
relative terms than anything else, because both are exponential in the surface
potential. On a linear scale those two rows saturate the colormap and every other
entry reads as white, so the figure would say "only two things happen", which is
the opposite of what it is for.

---

## 8. Everything is committed. Nothing is pushed.

Every change made today is committed and the working tree is clean.
`git log --oneline 1e4c55b..HEAD` is the authoritative list; at the time of
writing it reads:

| commit | what |
|---|---|
| `41cda3d` | source, scripts, the `train_mode` fix, the `spike_log` hook |
| `23c88c0` | the replay caches — 1,804 files |
| `8bbcf98` | D3–D6 docs, the figures, run artefacts, `provenance.jsonl` |
| `951364b` | both broken README commands, and the eleven cited-but-untracked artefacts |
| `7e77fd8` | README: Figure 4 needs beats this repo does not ship |
| `db36309` | the shim's solver-crash salvage, and Figure 5 |
| `f92d834` | V2's numbers, Figure 5 drawn, and these findings |
| `5feeba4` | the audits re-run after the salvage fix, and their artefacts |

A count is deliberately not quoted in the line above: any commit that corrects
the count invalidates it, which is the sort of number that is wrong in every
document that carries it.

The repository had been running out of an uncommitted working tree since D3 —
5,886 lines of source changes and every figure and finding — on a repository that
is public and that judges are expected to clone.

**Not pushed.** Publishing is outward-facing and D9 is the task that owns it. The
commits are local and reviewable; `git push` is one command when D7 or D9 wants it.

`.env` holds the licensed host's credentials and is gitignored; verified with
`git check-ignore` before the first `git add`.

---

## 9. What I did not do

- **I did not touch the device physics**, the circuit, the trims, or the frozen
  constants.
- **I did not draw the accuracy–energy Pareto.** `lambda_e` is 1e6 against an
  energy term of order 1e-9 × spikes, so it is inert at this scale and there is no
  Pareto front to draw — a two-point line with both points at the same energy
  would be a figure that implies a trade-off nobody measured.
- **I did not delete `scripts/v6_manifold_control.py`.** It keeps its SUPERSEDED
  banner explaining the `zdelt`/`xatol` trap, and it is now committed with it. The
  banner is worth more than the file's absence, and D4's numbers must stay
  replayable.
- **I did not chase the beta-gradient anomaly.** The D6 note says not to unless
  the writeup is finished, and it is not.
- **I did not re-run anything that the shim's crash fix could have changed.**
  The fix only takes effect on a path that previously raised, so no banked number
  can move; the flagship, the race, V6 and V7 all completed without a crashed
  probe. Said here rather than left to be wondered about.
- **I did not push, and I did not touch the Tally submission.**

---

## 10. Files this day produced

| path | what |
|---|---|
| `docs/figures/fig3_hysteresis_descent.png` / `.pdf` | Figure 3 — the device moving |
| `docs/figures/fig4_spike_raster.png` / `.pdf` | Figure 4 — what that does to the classifier |
| `docs/figures/fig5_jacobian_and_decay.png` / `.pdf` | Figure 5 — the manufactured adjoint, and how long it lasts |
| `scripts/tier_c_replay_d6.py` | Tier C, verified with the network unplugged |
| `scripts/provenance_audit_d6.py` | G10, as an audit |
| `scripts/t1_structural_zeros_d6.py` | the structural-zero correction |
| `scripts/v2_broyden_decay_d6.py` | V2, re-measured and banked |
| `scripts/fig3_hysteresis_descent.py` | draws Figure 3 |
| `scripts/fig4_spike_raster.py` | draws Figure 4 |
| `scripts/fig5_jacobian_and_decay.py` | draws Figure 5 |
| `results/runs/tier_c_replay_d6.json` | 164 of 164 bit-identical |
| `results/runs/provenance_audit_d6.json` | the G10 evidence |
| `results/runs/t1_structural_zeros_d6.json` | two zeros, one spurious column |
| `results/runs/v2_broyden_decay_d6.json` | the Broyden decay curve |
| `results/runs/v2_broyden_decay_d6.log` | its console output, including the salvaged probes |
| `results/runs/fig3_hysteresis_descent.json` | Figure 3's numbers |
| `results/runs/fig4_spike_raster.json` | Figure 4's numbers |
| `results/runs/fig5_jacobian_and_decay.json` | Figure 5's numbers |
| `results/runs/d6-trainmode-check/` | the `train_mode` fix, verified |
