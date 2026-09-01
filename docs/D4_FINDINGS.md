# D4 findings — 2026-08-27

Plain English. Short sentences. Terms explained the first time.

Words used below:

- **Open solver** = DEVSIM, the free simulator. It does the main work.
- **Commercial solver** = Sentaurus, the expensive one on the shared machine.
- **Design point** = one imaginary device, described by four numbers.
- **The four numbers** = film thickness, gate length, channel doping, and the
  thin oxide layer under the film. Things a factory can actually change.
- **Gradient** = the slope. Which way to move the four numbers to make the
  network do better, and how steeply.
- **The five numbers** = what the device hands the network: membrane decay, two
  conductances, firing threshold, weight noise.
- **Membrane decay** = how much charge a neuron keeps from one timestep to the
  next. 1.0 means it never forgets, 0 means it forgets instantly.
- **Firing threshold** = how much a neuron must build up before it fires,
  counted in input spikes.

---

## The short version

**The optimiser works now.** It takes a bad device and makes it good:

| | start | end |
|---|---:|---:|
| loss | 1.3996 | **1.0177** |
| accuracy | 0.250 | **0.6875** |
| film thickness | 5.50 nm | 7.65 nm |
| gate length | 52.0 nm | 35.4 nm |
| doping (log10) | 17.80 | 17.17 |
| thin oxide | 1.550 nm | 1.374 nm |

Sixteen steps, eight accepted, 64 solver calls, **16.6 minutes**. Every previous
run moved the loss by less than 0.002 and left accuracy unchanged.

**I had to fix four separate faults, and the last one was the real blocker:**

1. One bad neighbouring device was poisoning the gradient. Worth a factor of
   400,000. But not the cause.
2. The membrane decay was pinned against its limit over most of the design
   range, with a derivative of exactly zero. Real, but not the cause.
3. The firing threshold was collapsing to "fires on its first input spike" at
   one corner. Real, but not the cause.
4. **The network's gradient pointed backwards.** It had been telling the
   optimiser to walk uphill all along. *That* was the cause.

**I was wrong twice on the way and both corrections are below.** Last night's
document blamed the circuit. Corrected. Then I blamed the mesh and one
particular measured number, also wrong, and section 2 says so. Every claim here
is a measurement.

**Two things need your sign-off**: section 6.

---

## 1. The gradient pointed backwards — the real blocker

At the design point the flagship starts from, I compared the gradient the
optimiser uses against a plain finite difference of the loss: actually moving
each of the four numbers a little and watching what happens.

| | gradient the optimiser used | what the loss actually does |
|---|---:|---:|
| film thickness | +4.78e5 | −0.986 |
| gate length | −5.48e5 | +0.755 |
| doping | −4.59e5 | +0.546 |
| thin oxide | −3.40e5 | +0.706 |

Every sign is wrong and every size is about 700,000 times too big. As an angle,
the two point almost exactly opposite: **cosine −0.975**.

So the optimiser was handed a direction that goes uphill and walked it
faithfully. Everything that looked broken this week (steps rejected at every
size, the trust region collapsing, runs stalling after two accepted steps)
follows from that one fact.

### Which link, measured

- **The solver's part is fine.** Its Jacobian agrees with a fresh finite
  difference to within 9%, which is ordinary finite-difference error.
- **The circuit's part is closed-form maths.** Nothing to be wrong.
- **The network's part is the culprit.** Against a finite difference of the
  network's own loss: cosine **−0.983**, magnitude 300,000 times too large.

### Why — arithmetic, not a coding mistake

The network is recurrent over 111 timesteps. Backpropagation multiplies 111
step-by-step sensitivities together. If each is a little above 1 (say 1.4), then
1.4 to the power 111 is 3e16. That is what comes out.

And 3e16 cannot be a real slope. The loss is a cross-entropy over four classes,
bounded by about 5. A function that never exceeds 5 cannot have an average slope
over an interval of 0.0006 larger than about 8,000. A pointwise slope of 3e16
means the backward pass is amplifying floating-point noise.

**I checked the obvious suspect and it was not it.** The forward pass uses a hard
on/off spike while the backward pass uses a smooth stand-in, and that mismatch is
the usual thing to blame. Running the smooth version in the forward pass too
(making the network genuinely differentiable) made it **worse**: 3.11e16 against
1.21e6.

**I tried the standard remedy and it was not enough.** Truncating the backward
pass to a short window is the textbook fix for exploding gradients. It fixes the
size but not the direction:

| window | 1 | 2 | 5 | 10 | 20 | 40 | none |
|---|---:|---:|---:|---:|---:|---:|---:|
| cosine against the truth | +0.43 | +0.54 | +0.45 | −0.46 | −0.47 | −0.27 | −0.79 |

Nothing reaches the +0.7 a descent direction needs. The smooth stand-in for a
spike is an excellent trick for **training weights**, which is what it was
invented for. It is not a way to measure how sensitive the loss is to five
settings that act on every neuron at every timestep.

### The fix

**Measure it.** The network's sensitivity now comes from moving each of the five
numbers a little and seeing what the loss does: ten forward passes, all real.

That is not a retreat from what this project claims; it is the claim applied
consistently. The forward pass is never a stand-in, and derivatives are estimated
from evaluations that really happened. It is exactly what the project already
does for the solver, for the same reason: the thing in the middle has no usable
derivative of its own.

| at the same design point | before | after |
|---|---:|---:|
| size of the gradient | 9.25e5 | **1.11** |
| what the loss actually does | 1.53 | 1.53 |
| angle between them | **−0.975** | **+0.954** |

The old path stays behind a switch, because it is what the organisers' gradient
checker exercises and because the measurement above should stay reproducible.

---

## 2. The objective was not reproducible — also fixed

Before finding the direction problem I chased a different, real one: the loss was
not a function. Nudging the four numbers by one part in a million million
(about ten femtometres of film thickness, a hundredth of an atom) moved the loss
by 0.0056.

**My first explanation was wrong.** I said it came from the steepness of the
current curve at the read voltage, caused by the open solver rebuilding its mesh
for every device. Measured:

- The solver is **deterministic**. Run twice at the same device with the cache
  emptied, it returns bit-identical curves. The mesh is not jumping.
- Under that nudge the steepness moves by 1e-16 or less: one of the steadiest
  numbers in the set.
- What moves is the **threshold voltage on the reverse sweep**: 23 millivolts.
  Nothing else moves at all.

### The real cause

> Every curve point that differs by more than 0.1% has a current **at or below
> 4.5e-19 amps**. Nothing above that moves, anywhere.

That is the deep-off end of the sweep at −3.5 V, where the electron density is
about 1e-2 per cubic centimetre against 1e20 in the contacts. That is sixty
decimal places of range inside one matrix, against the sixteen ordinary
arithmetic carries. What comes back is drift, not a current.

The curve reader's floor sat at **1e-20 amps**, *below* the drift. So the drift
kept its shape, kept a slope, and the window that picks out the subthreshold
region was free to weight it.

### The fix, and how the number was chosen

The floor is now **1e-16 amps**. Both bounds measured, over seven devices:

| floor | worst memory-window jump under a ten-femtometre nudge | drift it puts into the leak current |
|---|---:|---:|
| 1e-20 (before) | **0.239 V** | — |
| 1e-18 | 3.1e-06 V | 0.04% |
| **1e-16** | **1.0e-06 V** | **0.5%** |
| 1e-15 | 6.3e-08 V | 3.6% |

- **Above the noise**: 200 times the 4.5e-19 ceiling where the drift lives.
- **Below the signal**: the smallest genuine leak current in the design range is
  3.1e-15 A, thirty-one times the floor.
- Still a hundred times below what any real instrument can measure.

### It cured the bug your first instruction was catching

**rand0's threshold went from +4.16 V (outside a sweep that stops at +1.50 V)
to +0.62 V**, and its memory window from a fictitious 4.44 V to 0.853 V. **Every
device now converges; nothing is refused.**

So the threshold landing outside the sweep was never a separate bug. It was the
same fit being dragged by drift. Your guard stays, and now guards against
something that no longer happens.

---

## 3. Retraining at every device was destroying the signal

With the floor raised, the five numbers became reproducible to seven decimal
places, and the loss still swung by 0.038. All of that was the network's
training.

The network was retrained from scratch at every design point. That is chaotic:
two nearly identical devices send the training down different paths. And it gets
**worse** the better the network gets:

| training steps | loss | accuracy | noise |
|---|---:|---:|---:|
| 150 | 1.298 | 0.56 | 5.5e-4 |
| 400 | 1.163 | 0.69 | 8.2e-3 |
| 800 | 0.982 | 0.75 | 2.4e-2 |

Decaying the learning rate does not fix it either: measured, it helps slightly
at 150 steps and is worse at 300.

### The measurement that decided it

**Signal** is how far the loss ranges across genuinely different devices;
**noise** is how far it moves when the device does not really change.

| | signal | noise | signal ÷ noise |
|---|---:|---:|---:|
| **A** one network, trained once, frozen | **0.296** | **0** | exact |
| **B** shared start, then a short tune per device | 0.260 | 5.3e-5 | 4,882 |
| **C** retrain from scratch (what it was) | 0.069 | 0.026 | **2.7** |

Two things there, and the second matters more:

1. Retraining from scratch is noisy: signal only 2.7 times the noise.
2. **Retraining from scratch destroys four fifths of the signal.** Training
   partly compensates for a bad device, so every device ends up scoring about
   the same. An objective for co-design must do the opposite.

The network is now fitted **once**, at a fixed reference device. Three modes:
**frozen** (no per-device training, what your thesis does, and it makes the
gradient's "hold the weights fixed" step exactly valid), **adapt** (a short tune
per device), **scratch** (the old behaviour, kept so old runs replay). The
flagship ran frozen: a design point costs about a second of network time instead
of 330.

---

## 4. The two trims (your second instruction)

Both were needed, both are real, and neither was the cause of the backwards
gradient. I said so at the time and it held up.

**The leak-bias trim.** The leak voltage was frozen at one value and applied to
every device while the design range moves the threshold by whole volts. The leak
current spanned **nine and a half factors of ten**, and the membrane decay came
back as exactly 0 or exactly 1 over most of the range, derivative exactly zero.
The trim works in decades (your own convention) and squashes the result
smoothly into the band where the network behaves. Smooth, never a hard clip: a
clip is flat outside its range and would hand back exactly zero slope on the
devices that most need one.

Result: membrane decay went from pinned at one end or the other to **0.583 to
0.626**, with a live slope everywhere.

**And an agreement comes back.** Your LSNN's membrane decay is 0.6065. The
trimmed nominal device gives **0.6057**: one part in a thousand. Not arranged:
the band came from a gradient measurement that knew nothing about your LSNN.
Before the trim the four-knob nominal device gave 0.9592 and that agreement had
quietly been lost.

**The synapse-mirror trim.** The firing threshold goes as one over the device's
on-conductance, four times larger at the leaky corner, so it fell to 1.22: a
neuron firing on roughly its first input spike. Your own configuration file
predicted this in the note frozen onto `K_syn`. It now spans **3.86 to 5.14**,
and the frozen "five spikes to fire" is preserved exactly.

**The measured band**, at two seeds, because one seed cannot tell a property of
the decay from a property of one training run:

| membrane decay | 0.30 | 0.50 | 0.58 | 0.6065 | 0.62 | 0.66 | 0.70 | 0.90 | 0.999 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed 0 | 2.9e9 | 2.0e11 | 39 | **8.1** | 31 | 8.6e3 | 9.9e3 | 6.4e6 | 1.1e10 |
| seed 1 | — | — | 368 | — | 139 | 2.1e4 | — | — | — |

**The low side is not explained.** 0.50 is worse than 0.30, out of order, and it
survived both a short and a long training run. I did not try to explain it.

---

## 5. The flagship

Frozen network, four design knobs, open solver, starting from the poor corner.

| | |
|---|---|
| steps | 16 |
| **accepted** | **8** |
| rejected | 8 |
| solver calls | 64 of 120 |
| **loss** | **1.399577 → 1.017666** |
| **accuracy** | **0.250 → 0.6875** |
| wall clock | 16.6 min |

The first five steps' quality scores: **+1.196, +0.981, +0.512, +0.075, +0.334**.
A score near 1 means the gradient predicted the drop almost exactly. The trust
region *grew* from 0.080 to 0.328 because the model kept being right. That is
what a working optimiser looks like, and it is the first time this project has
produced one.

The device it found sits close to your calibrated film: 7.65 nm against the
nominal 7.0 nm.

---

## 6. The strongest objection, answered with numbers

> **⚠ SUPERSEDED 2026-08-28 (D5). The table in this section is wrong, and the
> "0.79%" gate verdict below it is wrong.**
>
> The free arm never optimised. Its first Nelder-Mead restart started at the
> standardised origin, where SciPy builds an initial simplex 2.5e-4 wide against
> a tolerance of 1e-4, so it returned its own starting point, and that starting
> point was the cloud's own centroid. Measured: the `phi*` in
> `v6_manifold_control_d4.json` equals the cloud mean to 1.7e-14. So this
> section does not report a free optimum. It reports the loss at the AVERAGE
> device, and "1.9 device-spacings off the sheet" is really "the centroid of a
> curved cloud is 1.9 spacings from the cloud", which is nearly content-free.
>
> Re-run properly (differential evolution, then a polish with an explicit
> non-degenerate simplex), the numbers move a long way and **the conclusion
> reverses in this project's favour**: the gate PASSES at 8.6%, not 0.79%.
> See `docs/D5_FINDINGS.md` §2 and `results/runs/v6_manifold_control_d5.json`.
> The paragraphs below are kept so the correction has something to point at.

The objection is: *"the device only hands the network five numbers. So skip the
solver — optimise those five freely, find the best set, then find the device that
makes them."*

That is fatal if the best free set is buildable. It is not. Four fabrication
knobs cannot fill five dimensions, so what a device can produce is a thin sheet
in five-dimensional space. Measured, on **192 devices** sampled evenly across the
design range: **two directions carry 90% of the variation.** Four knobs, and only
about two effective dimensions of freedom.

Optimising the five numbers freely lands **1.9 typical device-spacings off that
sheet**: it describes a device nobody can build.

| strategy | solver calls | loss |
|---|---:|---:|
| free five numbers (impossible to build) | 0 | 1.0186 |
| build the nearest real device to it | 192 | 1.0258 |
| best of 192 devices sampled across the range | 192 | 1.0221 |
| **descend through the solver (this project)** | **64** | **1.0177** |

**Gradient descent through the solver wins on both counts**: a better device,
with a third of the solver calls. Projecting the free optimum onto reality is the
*worst* of the four.

### Your D5 gate, answered plainly

Your gate said: *joint descent must beat the projected baseline by at least 5%
relative, or say so and move the headline to sample-efficiency.*

**It does not.** Joint 1.0177 against projected 1.0258 is **0.79% relative**, not
5%. So, following your own instruction: **the headline is sample-efficiency, not
the margin.** Descending through the solver found a better device than 192
evenly-spread ones using **a third of the solver calls**, and the device it
found beats every one of those 192. That is the claim to make.

The margins between all four strategies are small: 0.008 from best to worst.
That also means "which device you pick matters less than how many calls it
costs you to pick it" on this task, and overselling the margin would be the
easiest thing for a judge to knock down.

A caveat that belongs with the table: the free and projected arms were scored
with the same frozen network as everything else, and the free arm's optimiser
was a derivative-free search that may not have found the true free optimum. Its
1.0186 is an upper bound on how good "free" can be, so the gap to joint could be
smaller than it looks, not larger.

---

## 6b. The race: five ways to spend the same solver budget

Every method got the **same 64 solver calls** and the **same poor starting
corner**. The solver call is the unit of cost because it is the only expensive
thing here. Three random seeds.

| method | median | best seed | worst seed |
|---|---:|---:|---:|
| **gradient descent (this project)** | **1.017666** | 1.017666 | **1.017666** |
| Bayesian optimisation, warm-started | 1.020981 | **1.017639** | 1.023622 |
| Latin hypercube | 1.027694 | 1.018232 | 1.032968 |
| random search | 1.030066 | 1.027038 | 1.036441 |
| Nelder-Mead from the same corner | 1.048191 | 1.048191 | 1.048191 |

**Read it honestly.**

**Bayesian optimisation's best run beats us**, by 2.7e-5, which is nothing. On
its *typical* run it is 0.0033 behind. So the fair statement is: **gradient
descent reaches, every single time, the best score that a well-tuned Bayesian
optimiser reaches on its luckiest run.** Its median is better and its spread is
zero.

Two arms have no randomness: gradient descent and Nelder-Mead both start from a
fixed corner and follow a deterministic rule, so all three seeds give the same
answer. That is a property, not a defect: the other methods are being *averaged
over their own luck* and these two have none to average.

**Nelder-Mead is the arm that matters most**, and it is last by a wide margin.
It is what a sensible engineer does when the simulator has no derivative
(derivative-free local search), and from a bad starting corner it gets stuck and
stays stuck. That gap, 1.048 against 1.018, is the clearest thing in the table.

**Bayesian optimisation was deliberately given the advantage** of a warm start
from an even initial design. A cold Gaussian process on a few dozen calls would
be a strawman, and beating a strawman proves nothing. It is also by far the most
expensive arm in wall-clock time (37 minutes against 16), though that is not the
cost that matters here.

### There is a crossover, and below it we lose

The race was run again at a **20-call** budget. The ordering reverses:

| method | median at 20 calls | median at 64 calls |
|---|---:|---:|
| random search | **1.030783** | 1.030066 |
| Latin hypercube | 1.031666 | 1.027694 |
| Bayesian optimisation | 1.032951 | 1.020981 |
| **gradient descent** | **1.038718** | **1.017666** |
| Nelder-Mead | 1.059021 | 1.048191 |

**At 20 calls gradient descent is fourth of five.** The reason is plain: building
the solver's Jacobian by finite differences costs **nine calls before a single
step is taken**. Out of twenty that is half the budget
spent before the method does anything, and it manages two steps. Random search
gets twenty independent looks at the box.

So the honest claim has a condition attached: **gradient descent through the
solver wins once the budget is large enough to pay for the derivative, and the
crossover on this problem sits somewhere between 20 and 64 calls.** Below it,
just sampling the box is better.

Better to say that plainly than only show the budget where we win.
It is also the expected shape for any derivative-based method on a
four-dimensional problem, and the effect gets *better* for us as dimensions grow,
because a Jacobian costs 2D+1 calls while covering a box costs exponentially many.
This project stands at D=4, which is the least favourable end of that trade.

### A harness bug I introduced, found, and fixed

The first run of this race gave the gradient arm 1.025376 in 41 calls on seeds 1
and 2, instead of 1.017666 in 64. That was **my race harness, not the project**:
the shim that holds the solver's Jacobian is cached per input template at module
scope, so the second run in the same process inherited the first one's Jacobian
*and* its call counter, and stopped early. Every other arm starts clean, so that
one had to as well. Fixed, re-run, and all three seeds now agree exactly.

I am recording it because the wrong numbers were on screen for an hour and
someone reading the raw log would otherwise trust them.

---

## 7. What was re-banked and re-checked

- **Open solver, nine devices.** Eight solve; the ninth is the exact thickest
  film allowed, 15.0 nm, which has failed since D3 and is not a regression.
- **Commercial solver, eight devices.** All pass. Numbers unchanged from
  yesterday, as expected: the new floor sits below anything those curves reach.
- **All 155 tests pass**, including three new ones pinning reproducibility
  directly. The fixture needed care: *additive* noise does not reproduce the
  failure, so the test would have passed for the wrong reason. With realistic
  multiplicative drift the old floor moves the memory window by **tens of volts**
  on a 5 V sweep and the new one by 1e-16 V.
- **The shared licence** was free every time I checked, one job at a time, and
  nothing of ours is left running.

---

## 7b. The two solvers agree about the physics

The free solver and the commercial one were differenced at the same design point
and their sensitivities compared. Only **film thickness** reaches the commercial
deck. The other three knobs are baked into its mesh and come back as exactly
zero, so that is the only column that carries information.

| figure of merit | commercial | open | same direction? |
|---|---:|---:|:---:|
| subthreshold slope | +0.124 | +0.183 | yes |
| forward threshold | +1.099 | +0.546 | yes |
| reverse threshold | −1.375 | −6.015 | yes |
| leakage current | −0.973 | −7.428 | yes |
| low conductance | −29.52 | −7.598 | yes |
| high conductance | +0.044 | +0.489 | yes |
| conductance slope | −1.701 | −1.159 | yes |

**Seven out of seven.** The magnitudes differ (they are different physical
models, one a 2-D drift-diffusion solve with a meshed ferroelectric and one the
commercial reference), but every single figure of merit moves the same way with
film thickness on both. That is the claim that matters: the optimisation is
being steered by physics, not by one solver's numerics.

**Do not quote the script's own headline number.** It prints "25% sign
agreement" over all 28 entries, and 21 of those are the structurally-zero
columns. Two identical zeros are not agreement. The honest figure is 7 of 7 on
the one column that exists.

---

## 8. What needs your sign-off

**One: the two trims.** They are what makes the gradient exist across the whole
design range. The cost is that the membrane decay now moves over a range of 0.04
instead of 0 to 1, so less of the device reaches the network. That is what a real
chip's bias trims do, but it changes what the project optimises.

**Two: fitting the network once instead of at every device.** This changes the
question from "how well can a network trained on this device do" to "how well
does a deployed network do on this device". Your thesis asks the second question,
and the measurement says it carries four times more signal.

Both are single switches with the measurements written next to them.

**And this follows from both:** every loss number from before today answers a
different question. Not wrong, different. They cannot share a table.

---

## 9. What I did not do

- **I did not touch the device physics**, yours or the open solver's.
- **I did not change the 26x polarization scaling.** Your instruction, unchanged.
- **Nothing committed, nothing pushed.**
