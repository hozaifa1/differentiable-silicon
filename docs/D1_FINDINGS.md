# D1 — Foundations: what was measured, and what changed because of it

23 August 2026. Gates G2, G3 and G4 closed; the contract, `V_leak` and `K_syn`
frozen; 94 tests green.

Everything below is a change to the build spec that came out of a measurement
made today, not out of a preference. Each one is recorded because a later day
will otherwise re-litigate it.

---

## Gate results

| Gate | Criterion | Result |
|---|---|---|
| **G0** timing | ≤ 6 min per sdevice run | **PASS** — 306 s, measured 23 Aug |
| **G1** remote net | HTTP 200 from the solver host | **PASS** |
| **G2** DEVSIM | `import devsim`; diode example exits 0 | **PASS** — rectifying I(V), 5.9 decades over 0.2→0.6 V |
| **G3** CI/GHCR | first Actions run builds and pushes | see the badge; workflow builds all five Tesseracts |
| **G4** extraction smoothness | max 2nd diff / mean 1st diff < 0.15 | **PASS** — ~5e-7 for SS and both thresholds |

## Extraction accuracy, measured over 150 random points of the d=5 box

| FoM | p95 | worst |
|---|---|---|
| `ss` | 0.015 % | 0.016 % |
| `vth_fwd` | 0.010 % | 0.013 % |
| `vth_rev` | 0.05 mV | 0.05 mV |
| `i_leak` | 0.076 % | 0.15 % |
| `g_lo` | 0.023 % | 0.043 % |
| `g_hi` | 0.0004 % | 0.0007 % |
| `dg_dvth` | 0.022 % | 0.033 % |

V3 (`tesseract-runtime check-gradients`): **0 failures / 56 checks** on
`adjoint-shim`, **0 failures / 10 checks** on `snn-lif-ecg`.

---

## Changes to the spec, with the measurement that forced each

### 1. The conductance fit is a degree-5 polynomial in log current, not a local quadratic

The spec called for a local quadratic on `Id`. Two separate problems, both measured:

- A quadratic in **linear** `Id` is fitted across a window spanning two e-folds,
  because `V_read` often sits where the erased branch is still exponential. In log
  space the same curve is a straight line.
- A quadratic's **slope** estimate carries an O(σ²) bias from the third derivative,
  and the subthreshold-to-on knee sits exactly where `V_read` does. Worst-case
  error on `dg/dV_th` across the design box: **14 % at degree 2, 0.26 % at degree 5**.
  Degrees 3 and 4 each remove one more bias term.

Also: the abscissa is scaled to `u = (V − V_read)/σ_v`. In raw volts the degree-5
normal matrix has condition number ~1e12; in `u` it is ~5e3.

**Why this is safe rather than just accurate:** σ_v = 25 mV is slightly under one
grid spacing, so the obvious worry is a grid-locked fit that re-introduces the
staircase. Measured: injecting 1e-6 relative noise moves `dg/dV_th` by 0.01 %, and
the G4 metric halves exactly under grid refinement (40 → 80 → 160 points gives
0.476 → 0.246 → 0.125), which is the O(h) signature of ordinary curvature.

### 2. The subthreshold window is σ = 0.6 decades, not 1.0

A real Id–Vg curve leaves the log-linear regime about 4.4 decades above 1e-10 A,
and at σ = 1.0 the Gaussian tail reaches into that knee and biases the slope.
Narrowing to 0.6 took worst-case SS error from **0.28 % → 0.016 %**, V_th from
**0.68 mV → 0.05 mV**, and `I_leak` — which amplifies the SS error over however
many decades separate `V_leak` from the window — from **1.72 % → 0.15 %**.

### 3. The sweep grid is [−1.2, 1.4] V

Still 96 points, as frozen. The range is set by the worst corner of the design
box: minimum V_th centre (L_g = 20 nm, N_ch = 1e16) combined with the maximum
memory window puts the reverse branch's 1e-10 A point near −0.98 V. The original
[−0.4, 1.0] would have clipped the soft window off the end of the grid there, and
the failure would have looked like a physics result.

### 4. Memory-window sign convention fixed

The spec has both `MW = V_th,rev − V_th,fwd > 0` and `g_lo = Id_fwd(V_read)`,
`g_hi = Id_rev(V_read)`. Those cannot both hold. For a counterclockwise FeFET the
forward (up) sweep sees the erased, **high**-V_th state and the reverse sweep sees
the programmed, low-V_th state, so `g_hi` does come from the reverse branch and
**MW = V_th,fwd − V_th,rev**. Convention fixed here and asserted in
`test_memory_window_sign_convention`.

### 5. `K_syn` added and frozen — the transducer was producing an unbuildable neuron

Following the spec's transducer arithmetic gives, at the nominal device,
`th_th = C_mem·V_spk/(g_max·V_ds·dt_hw) = 2.7e-4`. Spikes-to-fire cannot be 2.7e-4.

Working back: the DPI integrator needs `C_mem·V_spk/(th_th·dt_hw)` of synaptic
current, which at a sane `th_th` is a few nanoamps — a synapse conductance around
1e-8 S. A real 40 nm FeFET read at V_read = 0.60 V, V_ds = 50 mV delivers ~2e-4 S.
The gap is ~1.8e4 and it is not a modelling error: it is why analog neuromorphic
synapse arrays **attenuate** the read current into the integrator rather than
driving it directly.

`K_syn` is that fixed attenuation, one named design constant in
`config/circuit.yaml`, frozen today at **5.450675e-05**. It multiplies `g_min`,
`g_max` and `dg/dV_th` identically, so `sigma_w` is invariant to it and `beta`
does not involve it at all — it sets the firing threshold and nothing else. Both
invariances are asserted in `test_k_syn_does_not_touch_sigma_w`.

### 6. `th_th` is 5, not 20 — the first choice produced a silent, dead network

`th_th = 20` looked reasonable and was not. With `beta = 0.60` a neuron integrates
over `1/(1−beta) = 2.5` steps, so it can reach at most about
`2.5 × fan-in × input-rate × mean-weight`. At `th_th = 20` **no neuron in the
network ever fired**: every logit was identically zero, the loss sat at exactly
ln 4 for every θ in the box — and `jax.grad` still returned smooth, plausible,
entirely meaningless numbers, because a surrogate gradient does not care whether
a spike happened.

This is the most dangerous class of bug this project can have, and it would have
survived every numerical check in the suite. Two tests now catch it directly:
`test_the_network_actually_fires_at_the_nominal_operating_point` and
`test_the_loss_actually_moves_with_the_device`.

### 7. The mock's on-branch was 180× too conductive

Calibrated against a real 40 nm nMOS in the linear region
(`Id = μ_eff C_ox (W/L) V_ds (V_ov − V_ds/2)`, μ_eff = 200 cm²/Vs, EOT = 1 nm),
which gives ~10 µA at W = 100 nm, V_ov = 0.49 V, V_ds = 50 mV, i.e. g ≈ 2e-4 S.
The first draft produced 3.7e-2 S — 37 mS for a 100 nm device — and it silently
poisoned every downstream circuit number including the one above.

The mock's soft-min temperature is also now `2/ln 10` decades, which makes the
knee exactly `2 n U_T` wide in gate voltage — the same width the standard EKV
interpolation produces. The first draft's sharper knee was a mock artefact that
punished the conductance fit for something no real device does.

### 8. T4 runs in float64

The five hyperparameters span `beta ≈ 0.6` down to `g_min ≈ 2.6e-5`. In float32
the loss differences a gradient check probes land at machine epsilon and come back
quantised to powers of two; V3 failed for that reason alone. Doubling the cost of
a 2-layer LIF on 32 samples is not a consideration.

### 9. The surrogate gradient is now derived, not asserted

The spike is a Heaviside forward with the derivative of an explicitly written
smooth relaxation backward:

    H_k(u) = ½(1 + k u / (1 + k|u|)),   H_k'(u) = (k/2)/(1 + k|u|)²

so the training gradient is the true gradient of a network that exists on paper.
It also gives `check-gradients` something it can legitimately verify: finite
differences of a Heaviside are either exactly zero or one spike-flip large, so a
hard spiking forward pass cannot be gradient-checked by anyone. `smooth_spikes` on
T4's input schema selects the relaxation; it is never used for a reported result.

### 10. Provenance moved out of the schema and into an append-only log

`backend` and `content_hash` cannot be `OutputSchema` fields: every output leaf
crossing into JAX must be an array, and `tesseract_jax.apply_tesseract` raises
`TypeError: string indices must be integers` on anything else (see
[UPSTREAM.md](UPSTREAM.md)). Both are now written per call to the cache record and
to `results/runs/provenance.jsonl`. An append-only on-disk log is better evidence
than a return field anyway — it is what answers *"was the forward pass ever a
surrogate"*, and a return value could not.

### 11. `dL/dg_min` and `dL/dg_max` are ~1e-13, and that is correct

The normalised weight is `sigmoid(W) + sig_w·ε`: `g_min` and `g_max` cancel
exactly. A synapse programmable anywhere in `[g_min, g_max]` has the same usable
weight range whatever the window is. The window earns its place through `sig_w`,
which is already defined relative to it, and through `th_th ∝ 1/g_max`. Adding a
direct dependence would double-count the memory window and inflate every gradient
attribution in the writeup. Documented at the point of cancellation so it is not
later "fixed".

---

## Environment findings that change how later days must be built

### DEVSIM and PyTorch cannot share a Windows process

Both link Intel OpenMP. Whichever initialises second aborts the interpreter:

```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
Fatal Python error: Aborted
```

No traceback, no exception to catch. The documented escape hatch
`KMP_DUPLICATE_LIB_OK=TRUE` is explicitly "unsafe, unsupported, undocumented" and
Intel warns it can silently produce incorrect results — the last thing a solver in
a gradient path should do.

**Consequence for D2:** the DEVSIM oracle must run out of process from the network.
It already does — T2 is a served Tesseract in its own container and T4 is another —
so this is the process boundary earning its keep rather than being decorative, and
it is worth a sentence in the writeup. But it means the D2 DEVSIM work cannot be
debugged in a session that has imported torch.

### The Windows DEVSIM wheel ships no BLAS

It looks for `libopenblas`/`liblapack`/`libblas` or an Intel MKL whose highest
*tested* name is `mkl_rt.2.dll`; current MKL wheels install `mkl_rt.3.dll`.
`diffsilicon.shared.devsim_env` finds it, adds the directory to the DLL search
path and sets `DEVSIM_MATH_LIBS`. `mkl` is now a Windows-only dependency of the
`devsim` extra.

### Schemas must be named `InputSchema` / `OutputSchema` literally

Aliasing (`InputSchema = OracleInput`) generates an OpenAPI component named
`OracleInput`, and `tesseract_jax` looks up `Apply_InputSchema` by name and fails
with `KeyError`. The frozen contract is therefore *subclassed*, not aliased —
same fields, right component name.

---

## Frozen today, not to be changed

| Constant | Value | Basis |
|---|---|---|
| `V_leak` | **0.246391250 V** | the bias at which the nominal d=5 device draws 170 pA on its forward branch |
| `K_syn` | **5.450675e-05** | gives `th_th` = 5 spikes-to-fire at max weight on that same device |
| contract fields | `OracleInput` / `OracleOutput` | `tests/test_contract.py` fails loudly on any drift |
| Jacobian row order | ss, vth_fwd, vth_rev, i_leak, g_lo, g_hi, dg_dvth | fixes the cotangent layout and every cached record |
| sweep grid | 96 points, [−1.2, 1.4] V | worst-corner subthreshold window must stay on the grid |

Sanity numbers at the nominal device, all reproduced by `tests/test_circuit.py`:
`beta = 0.6034`, `tau = 21.8 µs ≈ 2·dt_hw`, `sigma_Vth = 74.6 mV`
(63 mV Pelgrom ⊕ 40 mV domain), `MW = 0.402 V`, `th_th = 5.00`, `sig_w = 0.227`.
