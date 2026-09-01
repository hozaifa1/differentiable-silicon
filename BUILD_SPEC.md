# DIFFERENTIABLE SILICON — Build Spec

### Backpropagating a class-balanced ECG classification loss through a spiking network, through a subthreshold neuron circuit, and into a closed-source commercial TCAD solver — to obtain ∂L/∂(ferroelectric process parameters)

**Tesseract Hackathon 2026** (Pasteur Labs & ISI) · **Track 3 — Hybrid ML + Mechanistic Models**
**D1 = Sat 23 Aug 2026 · Submit Sun 31 Aug** (AoE deadline = 1 Sep 17:59 Dhaka: reserve, never plan)

Selected by gauntlet loop over competing candidates. Bar = the 2025 first-place winner
(Multi-Agent-DPC, SOLARIS-JHU). Blind critic verdict: **this beats the bar**; the competing
composition-benchmark candidate lost to the bar.

---

## 0. WHY THIS WINS

The 2025 first-place entry crossed **no real boundary**: JAX↔JAX, Python↔Python, containerised PDE
solvers. The 2026 criteria were rewritten in response: criterion #1 is now "composition across a real
boundary," with the explicit warning that *"artificial boundaries inside a single script will score low."*

This project puts a **closed-source, license-metered, csh-driven commercial semiconductor solver,
running on a CentOS 7 machine that cannot host the training loop, inside the gradient path**, and
produces ∂(balanced cross-entropy)/∂(HZO ferroelectric thickness). The gradient crosses three mutually
unaware AD regimes (none / PyTorch / JAX) and one SSH hop. Every forward value is ground truth from the
real solver; only the derivative is manufactured, by directed probes of that solver.

No other entry will have a commercial closed-source solver inside the gradient path.

---

## 1. VERIFIED GROUND TRUTH (measured 23 Aug, not assumed)

| Fact | Status | Evidence |
|---|---|---|
| **sdevice wall-clock = 306 s (5.1 min), exit status 0** | **MEASURED** | `_timing/timing_t1_stamps.txt` on the remote box: START 1787422850 → END 1787423156. Workload was the *transient* LIF pulse protocol (9 pulses x 100 ns, Preisach FE) — heavier than the DC double sweep this design needs. **G0 PASSES (<6 min) → d=5 stays alive.** |
| Sentaurus host = **CentOS 7, kernel 3.10.0-514.el7, glibc 2.17** | MEASURED | `uname -a`, `ldd --version` |
| Host has **Python 2.7.5 ONLY** — no python3, no conda, no Docker, no podman | MEASURED | `ls /usr/bin/python*`, `which docker podman conda` |
| Host: 16 cores, 31 GB RAM, outbound internet OK | MEASURED | `nproc`, `free -g`, curl to pypi.org/simple → **HTTP 200** |
| `sdevice` + `sde` resolve after sourcing `~/.cshrc` under **csh** | MEASURED | csh -c 'source ~/.cshrc && which sdevice sde' |
| SSH host key cached; plink/pscp working from Windows | MEASURED | `SHA256:v9sOpO64cVy+vAh1VDJqpldWBX9KaGZJGxT89iwlvSE` |
| License free at time of test (no queued sdevice) | MEASURED | ps -ef grep sdevice |
| `tesseract-runtime serve` runs on **Windows with no Docker**; two Tesseracts in separate OS processes compose via `Tesseract.from_url()` + `tesseract_jax.apply_tesseract` with `jax.grad`/`jax.jit` flowing end-to-end | MEASURED | gradient matched analytic reference to **2e-16** (float64). Forward ~0.35 s, grad ~0.33 s, jitted grad ~0.05 s |
| tesseract-core 1.11.0, tesseract-jax 0.4.1 install clean on Windows/Py3.12 | MEASURED | `uv pip install` |
| `TesseractReference(type="url")` constructs `Tesseract.from_url`; SDK timeout default `None` (disabled) | VERIFIED in source | long solver calls safe with no config |
| `jax.grad` through a VJP-only Tesseract works via the `apply_tesseract` primitive | VERIFIED | the bayesian-inference demo does exactly this |
| `eps` in `runtime/experimental/finite_differences.py` is a **scalar float** on all 3 FD functions | VERIFIED in source | the upstream PR is real and correct |
| DEVSIM: Apache-2.0, `pip install devsim`, `cp39-abi3-win_amd64`, FeFET published (Sanchez & Chen, IEEE TED) | VERIFIED | |
| **QS-Devsim CANNOT be used** — non-commercial licence + patent CN 113297818 B, incompatible with required Apache-2.0 | VERIFIED | Miller model must be **clean-room** from Miller & McWhorter, *J. Appl. Phys.* **72**, 5999 (1992) |

### Architecture decision (gauntlet-adjudicated)

**T1 runs LOCALLY on Windows and drives the remote solver over plink/pscp.** It does not run on the Sentaurus host.

Rationale: the boundary criterion #1 actually judges (PyTorch ↔ JAX ↔ closed binary) is identical
either way; installing Python 3.11 on the host would *delete the strongest sentence in the writeup*:
**"the solver host has Python 2.7.5 and nothing else. I installed nothing on it. Tesseract's boundary
landed exactly where the technology ran out."** No judge can run T1 under any option; T2 is the
reproducible artifact. Day-1 hours go to T3 instead, which is the fragile piece.

Consequences for the spec: add SSH host config, an in-process license lock, retry-with-backoff, and a
parameter-keyed result cache. Delete the tunnel / remote-serve section. The writeup states the boundary
is PyTorch↔JAX↔closed binary, not laptop↔server.

---

## 2. ARCHITECTURE

| T | Name | Wraps | Lang / AD | Runs | Endpoints |
|---|---|---|---|---|---|
| **T1** | `sentaurus-fefet` | Sentaurus 2023.12 `sde`+`sdevice`, Preisach FE | Fortran/C++ closed-source, csh-driven | **local Windows**; `apply` shells out via pscp/plink to CentOS 7 | `apply`, `abstract_eval` — **no gradients, by design** |
| **T2** | `devsim-fefet` | DEVSIM 2.10 (Apache-2.0 C++ FVM) + clean-room Miller FE gate | C++ core / Python API | Docker via GitHub Actions → GHCR | `apply`, `abstract_eval` — **no gradients** |
| **T3** | `adjoint-shim` | trust-region FD + Broyden black-box adjoint | Python / NumPy+JAX | local | `apply` (**proxies to oracle — forward always exact**), `abstract_eval`, `jacobian`, `jvp`, **`vjp`** |
| **T4** | `snn-lif-ecg` | surrogate-gradient LIF, MIT-BIH, inter-patient DS1/DS2 | **PyTorch autograd** | local CPU / Kaggle | `apply`, `abstract_eval`, **`vjp`** (torch.autograd.grad), `jvp` |

Orchestrator: JAX + Optax on Windows. No Docker, no GPU locally.

**Boundaries:** three mutually unaware AD regimes (none / PyTorch / JAX) · Fortran+C++ under csh ⟷
C++ FVM ⟷ Python · device physics ⟷ subthreshold circuit dynamics ⟷ classification · a license-metered
remote host the training loop can never share a process with.

### Frozen contract (`shared/contract.py` — D1 hour 2, everything imports this)

```python
class OracleInput(BaseModel):
    theta:   Differentiable[Array[(D,), Float64]]   # D=3|5|12, normalised to [0,1]
    vg_grid: Array[(96,), Float64]                  # non-diff, fixed
    vds_lin: Float64 = 0.05
    vds_sat: Float64 = 0.80

class OracleOutput(BaseModel):
    # --- differentiable: 7 SMOOTH extracted FoMs. J is 7xD. ---
    ss:      Differentiable[Float64]   # mV/dec, weighted-LS fit
    vth_fwd: Differentiable[Float64]   # V, constant-current on the fit
    vth_rev: Differentiable[Float64]
    i_leak:  Differentiable[Float64]   # A, from the fitted line
    g_lo:    Differentiable[Float64]   # S, Id_fwd(V_read)/V_ds
    g_hi:    Differentiable[Float64]   # S, Id_rev(V_read)/V_ds
    dg_dvth: Differentiable[Float64]   # S/V, smooth local slope at V_read
    # --- non-differentiable ---
    id_vg:   Array[(2, 96), Float64]
    converged: bool
    solver_seconds: Float64
```

**The FD Jacobian is 7xD, not 132xD.** No efficiency claim rides on output dimensionality: a
scalar loss with D design variables needs D+1 forward-difference calls and always did. V7 justifies
the apparatus empirically.

### Design vectors

- **d=3** (mesh-invariant, safe branch): `t_fe` [5–15 nm], `Pr` [5–25 uC/cm2], `Ec` [0.8–2.0 MV/cm]
- **d=5** (+ geometry/doping): `L_g` [20–60 nm], `log10 N_ch` [16–18]
- **d=12** (**THE headline experiment**, DEVSIM only): + `t_IL`, `log10 N_sd`, `x_ov`, `log10 N_halo`, `x_halo`, `Gamma` (Miller minor-loop), `W_dev`

At d=5 the answer (t_fe down, Pr up, Ec down) is what any device engineer writes before running
anything. At d=12 it is not.

### T1 containerisation (criterion 5)

Ship `t1/Dockerfile` that COPYs only the driver, expects the Sentaurus tree bind-mounted at
`/opt/synopsys`, and takes `SNPSLMD_LICENSE_FILE` for license-server passthrough. Ship
`docs/T1_CONTAINER.md` explaining why the flagship runs uncontainerised.
**Uncontainerised T1 with no Dockerfile reads as "didn't containerise."**

### Smooth FoM extraction (`shared/extract.py`) — HIGHEST-VALUE HOUR, DO IT D1

Threshold-crossing V_th and max-slope SS on a 96-point grid are argmax/search ops: piecewise-constant
derivative in theta, kinking every time the crossing migrates a grid cell. **This, not Newton tolerance,
dominates FD error.** Pure JAX fixes it for free, and it lives inside the oracle so smoothing happens
*before* differencing.

Soft subthreshold window, centred on a decade, never on an index:

    w_k  proportional to  exp[ -(log10 I_k - log10 I_ref)^2 / (2 s^2) ],   I_ref = 1e-10 A,  s = 1.0 dec

Weighted least squares on (V_g_k, log10 I_k):

    SS     = [ sum_k w_k (V_k - Vbar)(L_k - Lbar) ]^-1  *  sum_k w_k (V_k - Vbar)^2
    V_th   = Vbar + SS * (log10 I_crit - Lbar),          I_crit = 100 nA * W / L_g
    I_leak = 10^( Lbar + (V_leak - Vbar)/SS )

`g_lo` / `g_hi` / `dg_dvth` from a second softmax-weighted **local quadratic** fit near V_read.

Every operation is a smooth reduction over the whole curve: no argmax, no branch, no interpolation
search, no index arithmetic. Unit tests run against synthetic curves with analytically known SS and
V_th, and against curves sampled on a deliberately coarse grid, asserting extraction error < 0.5% and
**derivative continuity under a fine theta sweep**.

---

## 3. THE TRANSDUCER H — real circuit, explicit capacitor

The naive H was numerology: with C_g ~ 1e-16 F, tau ~ 3 ns against dt = 8 ms gives
beta = exp(-2.7e6) = **0** in float64, so d(beta)/d(SS) = 0 and the whole device-to-algorithm channel
is dead. **The transistor is not the membrane.**

### Circuit

A **DPI (differential-pair integrator) neuron** (Bartolozzi & Indiveri, *Neural Computation*
**19**(10):2581–2603, 2007) with an explicit MIM integration capacitor C_mem and a leak transistor in
subthreshold. The FeFET plays two roles:

- **Role A: synapse.** The programmed FeFET conductance *is* the weight; the two hysteresis branches at V_read set [g_min, g_max]. This is where the memory window does its work.
- **Role B: leak device.** Biased at fixed V_leak, its subthreshold current sets the DPI time constant.

Standard DPI result tau = C_mem * U_T / (kappa * I_tau) with kappa = 1/n, combined with
SS = ln(10) * n * U_T, makes U_T cancel exactly:

    tau = C_mem * SS / ( ln(10) * I_tau ),      I_tau = I_leak = Id(V_leak)

### Named constants (`config/circuit.yaml` — all in one place, all in the writeup)

| Symbol | Value | Provenance |
|---|---|---|
| C_mem | **100 fF** MIM | design choice (~50–100 um2) |
| V_spk | 0.30 V | comparator threshold |
| V_read | 0.60 V | read bias, on-region |
| V_ds | 0.05 V | linear region, so g is a conductance |
| V_leak | **frozen** at the value giving Id = 170 pA on the *nominal* device | stated operating point, computed once D1, never changed |
| A (accel) | **512x** | standard practice (BrainScaleS runs 1e3–1e4x) |
| dt_hw | **11.0 us** | = dt_bio / A; dt_bio = 0.72 s / 128 = 5.6 ms |
| I_crit | 100 nA * W / L_g | constant-current V_th criterion |
| A_Vth | 4.0 mV*um | Pelgrom (JSSC 1989) — **assumed, stated as such** |
| A_dom | 100 nm2 | FE domain area — **assumed, stated as such** |
| W | 100 nm (d<=5), free at d=12 | device width |

### phi = H(y)

    beta   = exp( - dt_hw * ln(10) * I_tau / (C_mem * SS) )
    g_min  = g_lo,   g_max = g_hi
    th_th  = C_mem * V_spk / ( g_max * V_ds * dt_hw )          # spikes-to-fire at max weight
    sig_w  = (dg/dV_th) * sig_Vth / (g_max - g_min)
    sig_Vth^2 = A_Vth^2/(W*L_g)  +  (MW/2)^2 * A_dom/(W*L_g),   MW = V_th_rev - V_th_fwd

**Sanity check at nominal** (SS = 85 mV/dec, I_tau = 170 pA, C_mem = 100 fF, dt_hw = 11 us):

    x = (11e-6 * 2.303 * 170e-12) / (100e-15 * 0.085) = 0.507
      -> beta = 0.602,  tau = 21.8 us  ~=  2 * dt_hw                    HEALTHY
    d(beta)/d(SS)    = beta*x/SS  = 3.59 per V/dec  -> dSS = 10 mV/dec gives d(beta) = 0.036
    d(beta)/d(I_tau) = -beta*x/I_tau               -> 10% change in I_tau gives d(beta) = -0.031

I_tau is a *subthreshold* current at fixed V_leak, so a 60 mV V_th shift moves it ~5x.
**The device-to-beta channel is strong, well-conditioned, and both of its inputs come from the solver.**

At W = 100 nm, L_g = 40 nm, MW = 0.5 V: Pelgrom term 63 mV, domain term 40 mV, sig_Vth = 74 mV,
realistic for a scaled FeFET. At L_g = 60 nm it falls to 61 mV. **This creates the tension that makes
d=5 non-trivial:** shrinking L_g improves density and energy but wrecks variability through *both* terms.

### Honest provenance table (goes in the writeup, not left for a judge to find)

| phi component | From solver | Analytic constant |
|---|---|---|
| beta | SS, I_tau — **both** | C_mem, dt_hw |
| g_min, g_max | **both branches at V_read** | V_ds |
| th_th | g_max | C_mem, V_spk, dt_hw |
| sig_w | dg/dV_th, MW, g-window (3 of 4 factors) | A_Vth, A_dom — **assumed, named in config** |

sig_w is the one place an assumed coefficient enters. Say so in the paper rather than letting it be
discovered.

---

## 4. OBJECTIVE AND GRADIENT PATH

    min over (W, theta in Theta):
        J = F(W, H(G(theta)))  +  lambda_E * E(H(G(theta)))  +  lambda_R * R(theta)
            ^balanced CE           ^energy                       ^regulariser

G: R^D -> R^7 commercial solver, no adjoint. H: R^7 -> R^5 pure JAX. F: PyTorch.
E proportional to C_mem * V_spk^2 * E[spikes]. Box constraint on Theta by projection every step.

Weights: `W_tilde_ij = g_min + (g_max - g_min) * sigmoid(W_ij) + sig_w * eps_ij`, eps ~ N(0,1),
reparameterised so the gradient reaches sig_w. LIF, soft reset, fast-sigmoid surrogate
dS/dU ~= (1 + k|U - th_th|)^-2.

### Reverse sweep — three hops, each across a boundary

1. **PyTorch -> wire -> JAX.** T4 `vjp`: gbar_phi = torch.autograd.grad(L, phi, grad_outputs=cbar) in R^5. dL/dW consumed locally by Adam.
2. **JAX, exact, free.** gbar_y = (dH/dy)^T gbar_phi in R^7 via `jax.vjp`.
3. **Manufactured adjoint.** T3 `vjp`: gbar_theta ~= J^T gbar_y, J in R^(7xD). **Zero solver calls at gradient time.**

### Building and maintaining J

- **Anchor:** central differences, 2D+1 calls, h_i = alpha * (theta_i_max - theta_i_min), alpha from V1.
- **Refresh:** **forward** differences, D+1 calls. Central only for the anchor and V2 checkpoints, which halves refresh cost.
- **Broyden rank-1** between refreshes, from the free secant pair each accepted step supplies: `J <- J + (dy - J s) s^T / (s^T s)`.
- **Trust region** on rho_k; rho < 0.25 => Delta/2 + forced refresh; hard refresh every K steps (**K chosen from the V2 curve, not asserted**).
- **Hard budget cap.** `max_oracle_calls` in config. The run stops at the cap and reports calls used. Budget-capped optimisation is schedulable and honest; convergence-criterion optimisation is neither.

### Revised budgets (at the measured 306 s/run)

| D | calls | cap | wall-clock at 5.1 min |
|---|---|---|---|
| d=3 | 7 + 3x4 + ~15 = **34** | 45 | 2.9 h (cap 3.8 h) |
| d=5 | 11 + 3x6 + ~20 = **49** | 65 | 4.2 h (cap 5.5 h) |

---

## 5. VALIDATION SUITE V1–V7

**V1 — Deterministic non-smoothness sweep** *(replaces repeatability)*. A deterministic batch solver
rerun 5x at identical theta gives eps_rep = 0 exactly; the V-curve then has no lower branch and you
pick alpha far too small. What actually breaks FD here is deterministic non-smoothness in theta: mesh
regeneration, extraction kinks, Newton path changes.
→ **~40-point fine sweep across a single parameter (t_fe), inspected for staircase structure.**
Metric: max second difference / mean first difference. Then the alpha-selection V-curve on top of it.
**Run on DEVSIM, not on the license.** A 3-point spot check on Sentaurus only.

**V2 — Directional derivative vs the *composed* pipeline.** For 8 random unit directions u: model
g^T u from `jax.grad` vs truth [J(theta + h u) - J(theta - h u)] / 2h, where truth re-runs the entire
stack with no gradients anywhere.
→ **Reported as a CURVE of cosine similarity vs steps-since-refresh s = 0..5, mean +/- range over
several refresh cycles, not a pass/fail threshold.** Broyden corrects J only along s_k; the other
D-1 directions stay at last refresh, and descent steps are strongly correlated so secant directions
span a degenerate subspace. Quasi-Newton converges superlinearly *without* J_k -> J* (Dennis–Moré), so
a fixed "cos > 0.95" would pass at s=0 and fail at s=3. The curve justifies K empirically and converts
a liability into a result. Many samples on DEVSIM; 2-point spot check (s=0, s=4) on Sentaurus.

**V3 — The organizers' own checker.** `tesseract-runtime check-gradients @payload.json --endpoints
vector_jacobian_product --eps <alpha> --rtol 0.15` on T3-over-DEVSIM and on T4. **In CI.** Loosened
rtol justified in text.

**V4 — Cross-solver sign agreement.** V2 protocol on both oracles. Preisach vs Miller differ in
magnitude; the **sign matrix and rank-ordering of d(FoM)/d(theta_i) must agree.** Physics, not
numerics, is steering the optimisation.

**V5 — Descent audit.** rho_k histogram; fraction rho_k > 0; accept/reject log with per-step call cost.

**V6 — REACHABLE-MANIFOLD CONTROL. FIGURE 1. Zero Sentaurus calls.**

Closes the refutation: *"phi is five scalars, so H∘G is R^D -> R^5. Make phi five free JAX parameters,
train once (zero solver calls) -> phi*; then invert G to hit phi* in ~25 Nelder–Mead calls. No shim, no
Broyden, no Tesseract."* The defence is that phi* found unconstrained is generically **off** the reachable
set M = {H(G(theta)) : theta in Theta}, a curved bounded 5-manifold, and that projecting onto M after
the fact is not the constrained optimum. That defence is worth nothing unless it is measured.

| Arm | Feasible? | Solver calls | Balanced CE |
|---|---|---|---|
| (a) Free-phi — phi unconstrained learnable | NO | **0** | lower bound |
| (b) Projected-phi — the napkin baseline, implemented honestly: theta_proj = argmin ‖H(G(theta)) - phi*‖^2 by Nelder–Mead, then retrain W | YES | ~25 | worse |
| (c) Joint descent — ours | YES | ~49 | **must be better than (b)** |

Plus the **manifold visualisation**: 200-point LHS on DEVSIM -> phi cloud -> PCA pairplot with phi*
plotted visibly outside it. Beautiful and dispositive.

**Numeric criterion: (c) must beat (b) by >= 5% relative balanced CE on DEVSIM over 5 seeds.**
If it does not, say so plainly and move the headline to d=12, where manifold curvature is far more
consequential. Run entirely on DEVSIM + replay cache.

**V7 — H-ablation and apparatus ablation.**

- *H-ablation:* replace J_H = dH/dy with a norm-matched **random matrix**; rerun joint descent. Expect the loss to still fall somewhat (descent on a scrambled-but-consistent map can) while the **recovered design is physically wrong** (SS driven up, MW driven down). V2 proves the chain rule is *implemented*; this proves it is *meaningful*.
- *Apparatus ablation:* oracle-calls-to-target for `FD-every-step` vs `FD+Broyden+TR` at D = 3, 5, 12. Honest expectation ~1.8x saving at d=5, ~2.4x at d=12. **Report it; do not claim more.**

---

## 6. HEADLINE CLAIM, BASELINES, REALISTIC NUMBERS

> **On a 12-dimensional FeFET process-design space, joint gradient descent on (network weights, process
> parameters) — with every forward value produced by an unmodified TCAD solver — reaches a lower
> class-balanced cross-entropy than warm-started Bayesian optimisation at equal oracle budget, and than
> the free-phi-then-project baseline at any budget. The same pipeline runs unchanged against a
> closed-source commercial solver on a remote license-locked host and against an Apache-2.0 solver in a
> container, differing by one environment variable.**

d=12 on DEVSIM (5 seeds, error bars) is the *scientific* claim. d=5 on Sentaurus (1 run) is the
*demonstration nobody else can make*. Both stated as such.

### Baseline set — all on DEVSIM, 5 seeds, d=5 and d=12

| Arm | Note |
|---|---|
| **Joint gradient (ours)** | |
| **Warm-started BO** (GP-EI, W warm-started, 3-epoch fine-tune per candidate) | **the honest headline baseline** |
| Cold-start BO (30-epoch retrain) | reported only to show *why* warm-starting matters — NOT the comparison |
| Warm-started CMA-ES | |
| Warm-started random search | |
| 20-point LHS + quadratic surrogate | what a device engineer actually does |
| Free-phi + projection (V6b) | the napkin refutation |
| FD-every-step (no Broyden/TR) | apparatus ablation |

Cost reported in **both** oracle calls **and** SNN epochs. The naive "1,800 SNN-epoch" BO figure is
inflated ~6x: nobody retrains from scratch per candidate.

### MIT-BIH numbers — corrected to published reality

Inter-patient DS1/DS2 (de Chazal). **Primary reporting: 4-class N/S/V/F**, as most inter-patient papers
do; Q reported separately (DS2 has ~7 Q beats, so a 5-class macro-F1 is dominated by a class with
single-digit support).

| Metric | Nominal device | Target after co-design |
|---|---|---|
| **Balanced CE** (the optimised objective) | — | **>= 20% relative reduction** |
| Macro-F1, 4-class | **0.55–0.65** | +0.04–0.08 |
| Macro-F1, 5-class | **0.50–0.58** | +0.03–0.06 |
| Per-class F1 | N ~0.95 · S ~0.40–0.55 · V ~0.85 · F ~0.10–0.25 | |
| Energy / inference | 1.0x | 0.6–0.8x |
| Memory window | ~0.4 V | ~0.8 V |
| sig_Vth | ~74 mV | trade-off, may rise if L_g shrinks |

**State explicitly:** a 2-layer LIF SNN with device non-idealities lands below published CNNs, and that
is expected. The claim is about *improvement from co-design*, not SOTA. (The earlier 0.62–0.70 target
had an untuned SNN beating published deep CNNs before optimisation, checkable in thirty seconds and fatal.)

### Relation to prior work — one paragraph, no semantics

> Rehmann et al. (arXiv 2511.10761) replace non-differentiable CAE components with trained surrogates
> and differentiate the surrogate. **We differ in one respect, and it is the only one we claim: our
> forward pass is never a surrogate.** Every loss value reported here is produced by an unmodified TCAD
> solver at the current design point; only the derivative is estimated, from directed finite-difference
> probes of that same solver, with a trust region forcing ground-truth refresh. Per-iteration logs
> record the solver backend and the content hash of every forward evaluation. We make no claim that a
> rank-one-updated local linear model is categorically different from a surrogate of the derivative —
> it is one, and least-change secant fitting is exactly how it is constructed.

**Delete every instance of "quasi-Newton adjoint, NOT a learned surrogate."**

---

## 7. REPRODUCTION WITHOUT A LICENSE — four tiers

- **Tier A — 2 min, no Docker, no license.** `uv sync && uv run pytest`. Complete pipeline against an analytic mock oracle via `Tesseract.from_tesseract_api()`. Proves every wire, every schema, every gradient hop. *This is what a time-pressed judge will actually run.*
- **Tier B — ~45 min, Docker, no license.** `docker pull ghcr.io/<user>/devsim-fefet:v1`; `tesseract serve devsim-fefet --port 8101`; `ORACLE_URL=... uv run python -m diffsilicon.race --seeds 3`. **Swapping the commercial solver for the Apache-2.0 one is ONE environment variable.** The README says so at that exact spot; that line *is* the why-Tesseract demonstration.
- **Tier C — 3 min, regenerates every Sentaurus figure.** `oracle_backend=replay` serves a content-addressed cache of every Sentaurus call (~250 files x ~7 floats, a few hundred KB, committed to the repo). Bit-for-bit, no license, no network. **Wire the cache on D1: it must populate as a side effect of every run, not be reconstructed at the end.**
- **Tier D — bring your own license.** `.cmd` templates, csh driver, `@placeholder@` list, SSH config.

Determinism: `uv.lock`, GHCR digests pinned by sha256, a single seed threaded to NumPy/JAX/PyTorch,
`results/manifest.json` with a sha256 per figure.

**License caution:** publish only self-authored input decks and numeric outputs. No Synopsys binaries,
no Synopsys-shipped parameter files, no `.cmd` fragments copied from Synopsys documentation.
**Confirm with the license admin before committing the `.plt` tarball.**

---

## 8. DECISION TABLE — every gate, measurement, threshold, branch

| Gate | When | Measurement | Threshold | Branch if failed |
|---|---|---|---|---|
| **G0 Timing** | D1 h1 | median wall-clock, 3 sdevice runs, distinct theta | <= 6 min | **PASSED — 306 s measured 23 Aug.** 6–15 min -> d=3, no negotiation. >15 min -> Sentaurus becomes a 3-point validation, DEVSIM becomes the flagship and the paper says so |
| **G1 Remote net** | D1 h1 | curl to pypi.org on remote | HTTP 200 | **PASSED — 200.** (Moot under the local-T1 decision) |
| **G2 DEVSIM import** | D1 h1 | `import devsim`; bundled diode example converges | exit 0 | conda-forge build; else fallback (c) at G5 |
| **G3 CI/GHCR** | **D1 h2** | first Actions run builds + pushes an image | green | debug immediately — Tier B is gated on it and there is no local Docker |
| **G4 Extraction smoothness** | D1 EOD | 40-pt t_fe sweep on **mock**; max 2nd diff / mean 1st diff of SS and V_th | **< 0.15** | tighten the soft window; widen s; smooth the curve before the fit |
| **G5 Open oracle** | **D2 20:00** | DEVSIM hysteretic Id–Vg memory window | **MW > 0.1 V** | (b) DEVSIM MOSFET + analytic series FE cap (Miller), solved self-consistently — *the fallback I expect to ship*; (c) pure-Python 1-D Poisson/DD + Miller, DEVSIM demoted to cross-check |
| **G6 Mini-flagship** | **D3 09:00** | d=3 run complete, balanced CE decreased | >= 2 accepted steps, dCE < 0 | if CE flat: check J sign against V4 before touching anything else |
| **G7 FD conditioning** | D3 EOD | V1 staircase metric on DEVSIM; >= (D-1) of D columns stable to < 10% when alpha halved | >= D-1 | drop to d=3 material-only; or smoothed local-quadratic Jacobian over a small LHS |
| **G8 Manifold claim** | **D5 EOD** | V6: joint vs projected, balanced CE, 5 seeds, DEVSIM | joint >= 5% better | report honestly; move the headline entirely to d=12 sample-efficiency; V6 stays in the paper as a negative result at d=5 |
| **G9 Flagship** | D6 12:00 | d=5 Sentaurus run complete within cap | complete | restart that night at d=3 with the D2 config (already validated) |
| **G10 Framing** | D7 | can you state the counterfactual in 2 sentences **and** show the per-iteration backend+hash log? | yes | rerun flagship with forced ground-truth forwards using reserved overnight headroom |

---

## 9. RISKS, RERANKED

**R1 — The pipeline factorises and the free-phi-then-project baseline matches us.** Now the top risk;
it was invisible before. Kills criteria 2 and 3 simultaneously and costs a judge thirty seconds to
construct. → **V6 is Figure 1**, run on free compute, with a pre-stated numeric criterion (G8) and a
pre-stated honest fallback. Plus d=12 promoted to headline, where the reachable manifold is far more
curved and the projection argument far stronger. **Gate G8, D5.**

**R2 — Sentaurus runs slower than assumed and the schedule is fiction.** → **MITIGATED: 306 s measured
on the heavier transient workload.** Hard oracle-call caps, forward-difference refreshes, d=3
mini-flagship banked on D2 night.

**R3 — Non-smoothness in the *extraction layer* poisons the FD Jacobian.** argmax/threshold-crossing
derivatives are piecewise constant in theta and kink whenever a crossing migrates a grid cell; this
dominates FD error long before Newton tolerance does. → smooth extraction (section 2), built D1,
unit-tested for derivative continuity, living inside the oracle so smoothing precedes differencing.
**Gate G4, D1 EOD.**

**R4 — No open oracle, so nothing is reproducible.** QS-Devsim is unusable (non-commercial + patent
CN 113297818 B, incompatible with Apache-2.0), so Miller must be clean-room. → three-tier fallback
decided at a fixed hour (G5); plus the replay cache is an *independent* reproducibility path that
survives total DEVSIM failure. **Gate G5, D2 20:00.**

**R5 — Broyden does not track and V2 says so.** Now *designed for* rather than risked: reported as a
curve, used to choose K empirically. Residual risk only if cosine collapses at s=1, in which case
refresh every step (D+1 calls) and report the honest cost. **Gate G7 / V2 curve, D3.**

**R6 — Framing collapses into "you built a derivative surrogate."** → exact-forward is the whole
defence; per-iteration backend+hash log; semantic hair-splitting deleted. **Gate G10, D7.**

**R7 — Solo entrant, finals + GRE, one-day slip near-certain.** → D2 mini-flagship banks a real result
on day 2; every gate has a same-day fallback; the stretch goal is pre-cut.

---

## 10. CUT LIST

**CUT: Mosaic deal.II implementation.** `mosaic/benchmarks/problems/thermal_mesh/exclusions.py`
excludes deal.II from every gradient experiment with a categorical *"the C++ solver ships no AD path"*
label, and `adjoint-shim` is exactly the thing that fills that cell. It is a genuine gift to the
maintainers and a strong Best-Engineering story, **and** building a deal.II Tesseract plus learning
Mosaic's harness is 1–2 days that would land on D7–D8 and endanger the flagship.

**SHIP INSTEAD (zero cost, most of the value):** one paragraph in the writeup citing the
`exclusions.py` line by path, stating that `adjoint-shim` retrofits a validated VJP onto exactly that
class of solver, and offering to fill the cell in collaboration. If D7 finishes early, do a single
minimal cell as a bonus commit. **Not before.**

**KEEP: the upstream PR.** `eps` is a scalar `float` on all three functions in
`tesseract_core/runtime/experimental/finite_differences.py`. The `check-gradients` docs warn that
`eps` has to be chosen against the inputs, in the direction of it being too small; this project hits
the other end (t_fe in nm, N_ch in cm^-3, eighteen orders apart). Vector-valued per-parameter `eps`, with this project as the motivating
case. **Cheapest $1,000 on the board. Land it D7.**

---

## 11. PRIZE POSITIONING

- **Primary: Track 3 — Hybrid ML + Mechanistic Models.** The track is literally "embedding a neural network inside a physics loop and training it by differentiating through the physics solver." The differentiator *within* the track: every other entry will differentiate through a solver that was already differentiable, or one they wrote themselves.
- **Grand ($8,000) / Second ($5,000)** — the actual target.
- **Best Engineering / Tesseract Hack ($1,000)** — reusable `adjoint-shim`; uncontainerised license-locked deployment; hot-swappable oracle contract; the upstream `eps`-vector PR.
- **Best Visual ($1,000)** — manifold cloud with phi* outside it; animated Id–Vg hysteresis morphing along the descent path; before/after LIF spike rasters on the same ECG beat; accuracy–energy Pareto front; cosine-vs-steps-since-refresh curve; the 7xD Jacobian heatmap.
- **Credential:** the writeup and LinkedIn post should note the **IEEE EDS TCAD Hackathon 2026 win**. It is externally verifiable, and it makes the Sentaurus access and TCAD competence read as credible rather than surprising.

---

## 12. STILL UNVERIFIED — all gated

1. `import devsim` on Py3.12 / Windows — **G2**
2. Clean-room Miller convergence in one day — **G5**, three-tier fallback
3. Synopsys license terms on publishing simulation *outputs* — confirm with the license admin before committing the `.plt` tarball
4. A_Vth = 4.0 mV*um and A_dom = 100 nm2 — **assumed values, named in config and in the writeup**, not measured
5. Tally form fields / whether track selection is single-select
