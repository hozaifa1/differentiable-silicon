# D3 findings — the overnight run, executed 2026-08-26

Plain English. Short sentences. Terms explained the first time.

Some words used below:

- **Open solver** = DEVSIM, the free simulator. It does the main work.
- **Commercial solver** = Sentaurus, the expensive one on the remote machine.
- **Design point** = one imaginary device, described by four numbers.
- **The four numbers** = film thickness, gate length, channel doping, and the
  thin oxide layer under the film. These are things a factory can actually
  change.
- **Memory window** = how far apart the device's two states are, in volts.
  Bigger is generally better.
- **SS** = subthreshold slope. How many millivolts of gate voltage it takes to
  change the current tenfold. Smaller is better. About 60 is the physical floor
  for an ordinary transistor.

---

## Short version

All four jobs on the list ran. Here is the state of things.

**Good:**

1. **The open solver still works on the wider voltage sweep.** The normal device
   is fine. Only one device in the whole design range fails, and it is the exact
   thickest film allowed, 15.0 nm. 14.5 nm is fine.
2. **Eight commercial-solver devices are banked** with their full curves, ready
   to publish. All checks passed.
3. **The file that lets us talk to the commercial solver had gone missing.** I
   got it back, and proved it was right.
4. **Two scripts used to die when one device failed. Now they don't.**

**Bad, and this is the thing to read:**

5. **The optimisation run barely moved.** One accepted step out of six, loss
   1.2603 → 1.1662, then it stalled. D2's run on the old setup went to 0.0223.
6. **The gradient comes back as 6e13 when the loss is about 1**, so the
   optimiser's judge is meaningless and it spends its budget rebuilding the same
   gradient. Section 4b said why. **The "why" in section 4b was WRONG and has
   been corrected on 2026-08-27.** It blamed the circuit, and the circuit is
   fine. It is the spiking network, plus one poisoned neighbouring device.
   Read the corrected section 4b, and `docs/D4_FINDINGS.md`.
7. **I found a second bug in the curve reading.** I did NOT fix it, on purpose.

**Two things need your decision** (section 6).

---

## 1. The open solver on the wider sweep

The sweep now runs from −3.50 V to +1.50 V. The question was whether the solver
still copes.

**The normal device is fine.** It gives:

| | value |
|---|---|
| SS | 79.2 mV/dec |
| memory window | 0.523 V |
| leakage current | 1.3e-11 A |
| on/off ratio | 96.7 |

The memory-window check passes. The sign check passes.

**One device out of the whole range fails.** It is the thickest film the design
range allows: 15.0 nm. I measured where the edge is by trying thicknesses one by
one:

| film thickness | reaches −3.50 V? |
|---|---|
| 5.0 nm | yes |
| 7.0 nm | yes |
| 9.0 nm | yes |
| 11.0 nm | yes |
| 13.0 nm | yes |
| 13.5 nm | yes |
| 14.0 nm | yes |
| 14.5 nm | yes |
| **15.0 nm** | **no — stops at −3.25 V** |

So the unusable part is a razor-thin sliver at the very top of the range.

**Why it fails.** At −3.25 V the device is fully off. The current is 2e-19 amps,
which is not a real number: it is smaller than anything you could ever measure.
The electron count in the channel drops to about 0.015 per cubic centimetre,
while the source and drain sit at 1e20. That is sixty-odd decimal places between
the largest and smallest numbers in the same matrix. A computer using ordinary
double-precision arithmetic has about sixteen. The matrix becomes singular and
the solve dies with a divide-by-zero.

I checked three things so I am not guessing:

- **It is not the step size.** Going down in steps of 1.5 microvolts fails just
  the same as going down in steps of 50 millivolts. A step-size problem gets
  better when you make the step smaller. This does not.
- **It is not the order.** Ramping the gate first and the drain second gets to
  −3.5 V, and then fails on the drain instead. Both routes hit the same wall.
- **It is not rounding.** DEVSIM has an extended-precision mode. Turning it on
  changes nothing. It also turns out this build cannot use it at all. See the
  note on SuperLU below.

**What I did about it.** Nothing to the physics. Changing the model would change
every number we have already banked. I made the two scripts survive it instead.

---

## 2. One bad device no longer kills the run

Two separate problems, both fixed.

**The baseline script used to stop dead.** It ran three of nine devices, hit the
15.0 nm one, and threw away everything, including the eight devices that had
worked. Now it records the failure as a row and carries on. A failed device
counts as a failed check, so nothing is quietly passed.

**The optimiser used to blame the wrong thing.** When a device failed to solve,
the optimiser caught it and wrote "we ran out of calls" into the run log. That
was wrong, and it was wrong confidently: the run had plenty of calls left. Both
problems raised the same generic error, so the optimiser could not tell them
apart. They now raise two different errors and cannot be confused.

A device we cannot evaluate is a device we cannot accept. That is a rejection,
and the optimiser already knows what to do with a rejection: take a smaller step
and try nearer home. Stopping the whole run instead threw away every remaining
call because one corner of the range is unsolvable.

The starting device is a different matter. If that one fails there is no smaller
step to fall back on, so the run now stops with a clear message naming the
device in real units.

Files touched:

- `src/diffsilicon/shim/adjoint.py`: new error type for "out of calls"
- `src/diffsilicon/optimise.py`: handle a failed device as a rejected step
- `scripts/rebaseline_d3.py`: record a failed device, keep going

All 129 tests still pass.

---

## 3. The commercial solver's settings file had gone missing

`t1/calibration.local.json` was not on this machine. That file holds your fitted
device constants. Without it nothing can be sent to the commercial solver at all,
so the whole commercial-solver job was blocked.

It is deliberately kept out of git, so git could not bring it back.

**I got it back without guessing any of it.** Every one of those constants had
already been substituted into the simulation files that ran on the machine on
24 August, and those files are still on disk in `t1_scratch/`. Turning the
template into a pattern and matching it against a file that actually ran reads
each value straight back out. That is reading, not reconstructing.

Eleven constants came back, and every file on disk agreed on all of them. Two
more, the film thickness the mesh was built at (7.0 nm) and the film's
background permittivity (33.0), are written down in `docs/D3_RECALIBRATION.md`
in plain text, so those came from your own notes.

**Then I proved it.** I re-rendered the simulation file at the same device the
old one used and compared the two, line by line. The only differences were the
two changes you deliberately made on 24 August in the evening:

- the polarization numbers, which you locked to the measured film that night
- the permittivity, which now gets scaled with thickness and did not before

Everything else was character-for-character identical. So the recovery is exact.

**And then I ran it.** The normal device on the commercial solver:

| | value |
|---|---|
| SS | 74.2 mV/dec |
| memory window | 2.061 V |
| leakage current | 4.1e-11 A |
| time | 221 seconds |

For comparison, on 24 August the same solver gave 68.1 mV/dec and 2.132 V. Those
came from a slightly different device (a 6 nm film with the old, unlocked
polarization), so they should be close but not equal, and they are. The file is
good.

The recovered file has a comment at the top saying it was recovered and how.
**Please check it against your own copy if you still have one anywhere.**

---

## 3b. Eight commercial-solver devices are now banked

All eight ran. All four checks passed. About 190 to 206 seconds each.

| device | film | memory window | SS |
|---|---:|---:|---:|
| t_fe_min | 5.00 nm | 1.443 V | 73.8 |
| nominal | 7.00 nm | 2.061 V | 74.2 |
| rand2 | 10.44 nm | 2.809 V | 76.4 |
| rand0 | 11.37 nm | 2.956 V | 77.3 |
| rand1 | 13.13 nm | 3.241 V | 77.7 |
| rand3 | 13.57 nm | 3.326 V | 78.0 |
| rand4 | 13.63 nm | 3.325 V | 78.0 |
| t_fe_max | 15.00 nm | 3.517 V | 78.2 |

Two things fall straight out of that table.

**The memory window grows steadily with film thickness, and nothing else moves
it.** rand3 and rand4 have almost the same film (13.57 and 13.63 nm) but very
different gate lengths (21 and 42 nm) and different doping. Their windows are
3.326 and 3.325 V, the same to three decimal places. That is the known limit,
now visible in the data: only thickness reaches the commercial solver. Gate
length, doping and the thin oxide layer are baked into the mesh and this driver
does not rebuild it per device.

**The 15 nm device the open solver cannot solve runs perfectly here**, and gives
the biggest window of the eight. So that device is not physically impossible. It
is just out of reach for the free solver's arithmetic.

The curves are saved in `results/cache/sentaurus/`, eight records, each holding
the actual current values at all 96 voltages for both sweep directions. I checked
what is inside them: the curves, the seven extracted numbers, and a hash. **No
deck, no fitted constants, no file names.** That is exactly what you said could
be published. They are not gitignored, so they are ready to commit when you want.
I have not committed anything.

---

## 3c. How far apart the two solvers are, measured

This is the number for the open question you had not decided.

The free solver shrinks the film's polarization by 26x to keep its simpler model
realistic. The commercial one gets the full value. So the two describe slightly
different films. Nobody had measured what that costs. Now it is measured:

| device | film | free solver window | commercial window | commercial ÷ free |
|---|---:|---:|---:|---:|
| t_fe_min | 5.00 nm | 0.362 V | 1.443 V | **3.98** |
| nominal | 7.00 nm | 0.523 V | 2.061 V | **3.94** |
| rand1 | 13.13 nm | 0.986 V | 3.241 V | **3.29** |
| rand3 | 13.57 nm | 0.976 V | 3.326 V | **3.41** |
| rand4 | 13.63 nm | 1.011 V | 3.325 V | **3.29** |
| rand0 | 11.37 nm | 4.439 V | 2.956 V | 0.67 ← |
| rand2 | 10.44 nm | 0.473 V | 2.809 V | 5.94 ← |
| t_fe_max | 15.00 nm | failed | 3.517 V | — |

**On the five well-behaved devices the ratio sits between 3.3 and 4.0.** That is
tight. The commercial solver's memory window is about three and a half to four
times the free solver's, consistently, across a film thickness range of nearly
3x. Both solvers agree the window grows with thickness.

**The two odd ones out are exactly the two devices whose curve reading is
suspect**: rand0, the one with the threshold outside the sweep, and rand2, the
one reading below the physical floor. That is a useful cross-check on its own:
the points that break the pattern are the points already flagged as unreliable,
not random noise.

So when you decide what to do about that 26x shrink, the cost of keeping it is
about a 3.5x gap in memory window. It is a clean, consistent offset, not a
disagreement about which direction things move.

---

## 4. The warning: a new bug in the curve reading, NOT fixed

**The threshold voltage can come back from outside the swept range.**

On one of the eight devices the reader returned a threshold of **+4.16 V**. The
sweep stops at **+1.5 V**. There is no data up there. The fit was extended past
the end of the measurements and the answer was taken from the extension.

That device's memory window then comes out as **4.44 V**, which is larger than
any of the others by a factor of four, and larger than the honest devices by
about eight. It is not real.

The device in question has SS = 431 mV/dec. That means it has no proper off
region at all: it never really switches off, so there is no threshold there to
find. Low doping on a short gate. The reader had nothing to work with and
extended a line instead of saying so.

This is the same family as the bug fixed on 24 August, but it is not the same
bug. That one was cured. Nothing yet stops the answer landing outside the data.

**One device in eight. Worth fixing, but not by me and not tonight.**

**Why I left it alone.** The reader's fingerprint is part of the key on every
stored result. Editing it throws away every banked number instantly. The long
optimisation run was already hours deep when I found this. Editing the file
mid-run would have wasted the whole night for a one-line guard.

**What the fix probably is:** refuse a threshold outside the swept range, and say
so, rather than returning a number from the extension. Then re-run the baselines.

---

## 4a. The optimisation run: it barely moved

It finished. 43 minutes, not the 5.6 hours planned.

| | |
|---|---|
| steps taken | 6 |
| **steps accepted** | **1** |
| steps rejected | 5 |
| solver calls used | 38 of 45 |
| gradient rebuilds | 6 |
| loss | 1.2603 → **1.1662** |
| accuracy | 0.375 → 0.500 |
| stopped because | trust region hit its floor twice in a row |

The device hardly changed:

| | start | end |
|---|---:|---:|
| film | 5.500 nm | 5.641 nm |
| gate | 52.00 nm | 50.42 nm |
| doping (log10) | 17.80 | 17.73 |
| thin oxide | 1.550 nm | 1.513 nm |

**Compare this with the D2 run**, which used 45 calls on the old setup and drove
the loss from 1.2289 to 0.0223 with nine accepted steps. That one worked. This
one took a single step and stalled.

**This is not a broken pipeline.** Everything ran: the gradient reached the
solver, the solver answered, the optimiser stepped, and the one step it accepted
genuinely reduced the loss and raised accuracy from 0.375 to 0.500. What it did
not do is descend. The reason is the next section, and it is measured, not
guessed.

Two honest differences from D2, taken together: D2 was allowed to change
the *material* (polarization and coercive field), which are now locked, and D2
did not have this gradient problem in the same form. So the D2 number cannot be
quoted as this project's result, and this run is not yet a replacement for it.

---

## 4b. The big one: the gradient is far too large, and WHY was wrong here

> **CORRECTED 2026-08-27.** Everything in this section about the SIZE of the
> gradient was right. Everything about its CAUSE was wrong.
>
> The old version blamed the leakage-to-membrane-decay line in the circuit
> model. That was worked out from one line of algebra and never measured. It was
> then measured, and it is not the cause. The wrong text has been replaced
> rather than left standing, because it was already being acted on.
>
> The correct account is below. The measurements are in `docs/D4_FINDINGS.md`.

**The gradient is about 60,000,000,000,000 when the loss is about 1.**

That part stands. Here is the run:

| step | loss before | loss after | gradient size | result |
|---|---:|---:|---:|---|
| 0 | 1.2603 | 1.1662 | 2.3e+06 | accepted |
| 1 | 1.1662 | 1.2411 | 6.1e+13 | rejected |
| 2 | 1.1662 | 1.2372 | 6.1e+13 | rejected |

**Why this ruins the run.** The optimiser judges each step by comparing how much
the loss actually fell against how much the gradient said it should fall. With a
gradient of 6e13 it predicts a drop of about 3,700,000,000,000. The real drop is
about 0.07. So the ratio is zero, every step looks like a total failure, and the
rule "if the step went badly, rebuild the gradient from scratch" fires every
single time. Rebuilding costs five solver calls. The run has forty-five. So most
of the budget is spent re-deriving the same gradient rather than descending.

That part also stands.

### What was WRONG here, and what is actually true

The loss reaches the design numbers through three links in a chain:

    the four numbers -> the device -> the circuit -> the network -> the loss

The old text said the middle link, the circuit, was the problem, because the
circuit turns the device's leakage current into the neuron's membrane decay
through an exponential, and the sensitivity of that step is about three billion
per amp. Three billion is a big number, so it looked like the answer.

**It is not, and the reason is units.** Three billion per AMP is only alarming if
something moves the current by an amp. Nothing does. The leakage current is
around a hundredth of a billionth of an amp, and changing one of the four design
numbers moves it by about a millionth of a millionth of an amp. Three billion
times a millionth of a millionth is three thousandths. Measured at the exact
design point where the gradient reached 6.1e13:

| link | what it is | largest entry |
|---|---|---:|
| device | how the seven measured numbers move with the four design numbers | 4.1e+03 |
| circuit | how the five network settings move with the seven measured numbers | 3.4e+08 |
| **device and circuit together** | **how the five network settings move with the four design numbers** | **7.8** |
| network | how the loss moves with the five network settings | 7.2e+10 |

Seven point eight. The circuit and the solver are fine. **The whole of the
explosion is in the network**, and the leakage path contributes 0.003 of that 7.8.

### The two things that were actually wrong

**One: one bad probe was poisoning the gradient.** To work out a gradient the
code evaluates eight neighbouring devices around the one it is at. At this design
point the device itself is perfectly healthy. One of its eight neighbours is not:
the curve reader hands back a subthreshold slope of MINUS 2890 mV/dec and a
threshold voltage of minus 36 volts, on a sweep that only goes down to minus 3.5
volts. Comparing that against a healthy neighbour puts about nine hundred volts
per unit into the gradient, and everything downstream inherits it. That is the
same fault as section 4 below, and section 4 said one device in eight was
affected. It was worse than that, because the eight neighbours are extra devices
nobody was looking at.

Fixed on D4. The curve reader now says when a threshold came from outside the
data, the design point is refused, and the affected neighbour is dropped rather
than the whole gradient. **That alone took the gradient from 6.1e13 to 1.5e8.**

**Two: the membrane decay sits against its upper limit, and the network explodes
there.** The network is recurrent and runs 111 timesteps. Its gradient grows by
one to two factors of ten for every step the membrane decay takes towards 1.0.
Measured through the network on its own:

| membrane decay | 0.6065 | 0.70 | 0.80 | 0.90 | 0.999 |
|---|---:|---:|---:|---:|---:|
| gradient | 12 | 211 | 1.7e4 | 2.3e6 | 1.1e11 |

That is textbook exploding gradients in a recurrent network, and it is a property
of the network, not of the device. It is why 1.5e8 remained after the first fix.

### What survives from the old text, and it is still important

The observation that **the membrane decay is a switch and not a dial** was right
and is unchanged. Over the design range the leakage current spans ten factors of
ten, and the membrane decay comes back as exactly 0 or exactly 1 on most of it,
with a derivative of exactly zero. That is a real defect: a design knob with no
slope is a knob the optimiser cannot turn.

But it is a DIFFERENT defect from the huge gradient, and fixing it was never
going to fix the huge gradient. Both are addressed on D4, separately. See
`docs/D4_FINDINGS.md`.


## 4c. The two-solver cross-check that ran, and the one that did not

The pipeline's last step compared the **mock** against the open solver at the new
four-knob design vector. Result:

- sign agreement: **78.6%** (22 of 28 entries)
- rank-order agreement: **0%** (0 of 7 rows)

Do not read much into that. Your own note from D2 already says the mock is a
wiring harness, not a third opinion, and that its disagreements on some rows are
expected because it has no term for those effects at all. The comparison that
matters is the **commercial solver against the open solver**, and that is a D6
task, not tonight's.

**Good news for when you do it:** it is now cheap. The settings file is restored,
the licence was free all evening, and each point takes about 200 seconds. The
comparison needs nine commercial-solver points, so roughly 30 minutes.

**I did not run it.** It was not on tonight's list, and it is the middle of your
working day. Holding your single shared licence for another half hour without
being asked is not my call. The licence is free right now and I left no stray
processes on the machine.

---

## 5. Also worth knowing

**A change sitting uncommitted may not do what it says.** There is uncommitted
work in `src/diffsilicon/shared/devsim_env.py` that falls back to a solver called
SuperLU when the fast one is missing, aimed at fixing the Linux build server. Its
comment says SuperLU "is always a valid fallback".

On this machine it is not. Setting it succeeds silently, and then the first solve
dies with `Solver "superlu" not supported in this build`. So the fallback moves
the crash rather than preventing it. It may well be present in the Linux build.
Check it before trusting that change.

**One device reads below the physical floor.** One point came back at 36.9
mV/dec. For an ordinary transistor about 60 is the floor at room temperature.
Ferroelectric devices are sometimes claimed to beat it, so this may be real, but
this model should not produce it. Worth a look, not urgent.

---

## 6. What needs your decision

**One: the gradient cliff (section 4b). This is blocking the headline result.**

> **SUPERSEDED 2026-08-27.** The suggestion that used to be here (put the
> leakage current on a log scale before it reaches the membrane decay) was
> aimed at the wrong link, because the diagnosis above it was wrong. Measured,
> the leakage path contributes 0.003 of a total of 7.8, so rescaling it could
> never have fixed a gradient of 6e13. The two things that were actually wrong,
> and what was done about them, are in `docs/D4_FINDINGS.md`. The log scale was
> still worth doing, for a different reason, and it was done, but as a fix for
> the membrane decay being pinned flat, not as a fix for the size of the
> gradient.

**Two: the 26x polarization shrink on the open solver.** You had left this
open. It is now measured: keeping it costs a memory window about 3.3 to 4.0 times
smaller than the commercial solver's, consistently, across the thickness range.
Both solvers agree on direction. See section 3c.

**Three, smaller: the curve reader returning a threshold from outside the
sweep** (section 4). One device in eight. Worth a guard, and re-running the
baselines after.

**Four, smaller: you can afford many more solver calls than you thought.** The
cost model behind "45 calls" was wrong by about seven times. The whole run took
43 minutes.

---

## What I did not change

- No device physics. Not yours, not the open solver's.
- Not the voltage sweep range. That was your decision on 24 August.
- Not the curve reader.
- Nothing committed or pushed.
