# Differentiable Silicon

**Backpropagating a class-balanced ECG classification loss through a spiking network, through a
subthreshold neuron circuit, and into a closed-source commercial TCAD solver — to obtain
∂L/∂(ferroelectric process parameters).**

[![CI](https://github.com/hozaifa1/differentiable-silicon/actions/workflows/ci.yml/badge.svg)](https://github.com/hozaifa1/differentiable-silicon/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Tesseract Hackathon 2026 (Pasteur Labs & ISI) · Track 3 — Hybrid ML + Mechanistic Models

---

The gradient in this repository crosses three mutually unaware AD regimes (none / PyTorch / JAX),
one SSH hop, and one closed-source Fortran+C++ binary that is driven from `csh` on a CentOS 7 host
with Python 2.7.5 and nothing else installed on it.

**Every forward value is ground truth from the real solver. Only the derivative is manufactured,
by directed finite-difference probes of that same solver.**

## The four Tesseracts

| | Name | Wraps | AD | Endpoints |
|---|---|---|---|---|
| **T1** | `sentaurus-fefet` | Synopsys Sentaurus 2023.12, Preisach ferroelectric | none (closed binary) | `apply`, `abstract_eval` |
| **T2** | `devsim-fefet` | DEVSIM 2.10 (Apache-2.0) + clean-room Miller FE gate | none | `apply`, `abstract_eval` |
| **T3** | `adjoint-shim` | trust-region FD + Broyden black-box adjoint | NumPy/JAX | `apply`, `abstract_eval`, `jacobian`, `jvp`, **`vjp`** |
| **T4** | `snn-lif-ecg` | thesis LSNN (100 LIF + 60 ALIF, delayed synapses) on 2000 MIT-BIH beats | PyTorch autograd | `apply`, `abstract_eval`, **`vjp`**, `jvp` |

T3 proxies its forward pass straight to whichever oracle `ORACLE_URL` points at, so the forward
value is never a surrogate — only `vjp` is.

## What is optimised, and what is not

The design vector is the four **fabrication** knobs — ferroelectric thickness, gate length,
channel doping, interfacial-layer thickness. Remanent polarization and coercive field are
**locked** to the measured HZO film (P_r = 32 µC/cm², P_s = 40 µC/cm², E_c = 1.4 MV/cm) and the
optimiser refuses to move them.

That restriction is the point, not a limitation. P_r and E_c are properties of a deposited film:
you change them by depositing a different film and re-calibrating, not by asking a fab for a
different number. An optimiser given them will widen the memory window by changing the material
and present it as a design result. Every result here is obtained on one fixed film.

The ECG split is **intra-patient**, matching the reference protocol. The curated beat files have
record identity stripped during preprocessing, so an inter-patient AAMI DS1/DS2 split cannot be
constructed from them — see `src/diffsilicon/snn/ecg.py`. Numbers here are not comparable to
inter-patient results.

Details and the full before/after: [`docs/D3_RECALIBRATION.md`](docs/D3_RECALIBRATION.md).

## Reproduction — four tiers

**Tier A — no Docker, no license.** The complete pipeline against an analytic mock oracle,
in-process via `Tesseract.from_tesseract_api()`. Proves every wire, every schema, every gradient hop.

```bash
uv sync --extra snn --group dev && uv run pytest
```

`--extra snn` is not optional: T4 is the PyTorch end of the gradient path, so `torch` is a
requirement of Tier A rather than a nicety, and `uv sync --group dev` alone leaves the test
collection failing on `ModuleNotFoundError: No module named 'torch'`.

**Timed on a fresh clone into a fresh venv, not estimated:** `uv sync` 74 s, then **352 s cold /
314 s warm** for the 152 tests that run (6 skip without DEVSIM). The cold/warm gap is small because
`results/cache/mock/` is only 124 KiB — nearly all of that time is the spiking network itself, 111
timesteps per beat in float64. Budget six minutes, not two.

**Tier B — Docker, no license.** Swap the commercial solver for the Apache-2.0 one.

```bash
docker pull ghcr.io/hozaifa1/devsim-fefet:latest
```

Then serve it and point the orchestrator at it:

```bash
tesseract serve devsim-fefet --port 8101
```

```bash
ORACLE_BACKEND=url ORACLE_URL=http://localhost:8101 uv run pytest tests/test_tier_a_pipeline.py
```

**Swapping the closed-source commercial solver for the open one is one environment variable.**
Nothing else in the pipeline changes — not the shim, not the transducer, not the network, not the
optimiser — because `sentaurus-fefet` and `devsim-fefet` publish a byte-identical frozen schema, and
[a test asserts that they still do](tests/test_tier_a_pipeline.py). That line is the whole reason
this is built on Tesseract.

All five images are on GHCR and public. Pin by digest, not by tag — `latest` is a moving target:

| Tesseract | Image |
|---|---|
| T1 | `ghcr.io/hozaifa1/sentaurus-fefet` |
| T2 | `ghcr.io/hozaifa1/devsim-fefet` |
| T3 | `ghcr.io/hozaifa1/adjoint-shim` |
| T4 | `ghcr.io/hozaifa1/snn-lif-ecg` |
| mock | `ghcr.io/hozaifa1/mock-oracle` |

### Running an optimisation

```bash
python scripts/run_flagship.py --backend devsim --d 4 --max-oracle-calls 120 --trust-radius 0.08 --theta0 0.05,0.80,0.90,0.70 --tag flagship
```

That is the exact command behind every headline number here: balanced
cross-entropy **1.3996 -> 1.0177** and accuracy **0.250 -> 0.688** in **64
solver calls**, banked at `results/runs/flagship-d4-fixed/`. `--d 3` is refused
on purpose -- it exposes P_r and E_c, which are locked material constants.

The run starts from a deliberately poor corner of the design box — a thin, weakly polarised film
whose memory window is too small to separate the two conductance states — and is capped by solver
calls rather than by a convergence criterion, because a finite-difference Jacobian over a commercial
solver is bought by the call and a budgeted run is one you can start before bed. It writes
`steps.jsonl` and `result.json` as it goes, and the per-step trust-region ratios in that file are
validation item V5.

**Tier C — regenerates every Sentaurus number with no license and no network.**
`results/cache/sentaurus/` is a content-addressed replay of every Sentaurus call ever made,
populated as a side effect of every run rather than reconstructed at the end.

```bash
uv run python scripts/tier_c_replay_d6.py
```

This is checked, not claimed. The script blocks `socket` and `subprocess` and strips every
`SENTAURUS_*` variable from the environment before it touches project code, then regenerates
**164 of 164 float64 values bit-identically** — every figure of merit in
`rebaseline_d3_sentaurus.json` and every entry of the V4 cross-solver Jacobian — in **1.4 s**,
standing in for **0.93 h** of commercial-solver time. If anything came off the solver rather than
the cache it fails with a traceback instead of returning a number.

Figure 3 replays the same way, from `results/cache/devsim/`, with no solver call:

```bash
uv run python scripts/fig3_hysteresis_descent.py
```

Figure 4 does not, and the reason is worth stating rather than hiding: it needs the 2000 curated
MIT-BIH beats, and **this repository deliberately does not ship them** — they are the thesis' own
preprocessing of a public database, and this repo is public. Set `DIFFSILICON_ECG_DIR` to the
folder holding `up/` and `down/` and it runs; without it, the numbers behind the figure are banked
in `results/runs/fig4_spike_raster.json`. The network weights it needs are committed
(`results/cache/w0/`), so nothing is retrained either way.

**Tier D — bring your own license.** See [`docs/T1_CONTAINER.md`](docs/T1_CONTAINER.md) and
`.env.example`. `t1/Dockerfile` expects the Sentaurus tree bind-mounted at `/opt/synopsys` and takes
`SNPSLMD_LICENSE_FILE` for license-server passthrough; the flagship itself runs uncontainerised, and
that document explains why.

## Documents

- [`docs/D1_FINDINGS.md`](docs/D1_FINDINGS.md) — every gate result, every measured
  number, and the eleven places where a measurement overruled the plan.
- [`docs/D2_FINDINGS.md`](docs/D2_FINDINGS.md) — the open oracle, and the four things
  that were wrong underneath it: a body that punched through, a stiff equation the device
  did not need, three silent traps in DEVSIM's expression language, and a classifier
  that was never trained.
- [`docs/D3_RECALIBRATION.md`](docs/D3_RECALIBRATION.md) — locking the material, and
  wiring the device under the thesis' own validated LSNN instead of a network invented
  for this project.
- [`docs/D3_FINDINGS.md`](docs/D3_FINDINGS.md) · [`docs/D4_FINDINGS.md`](docs/D4_FINDINGS.md)
  · [`docs/D5_FINDINGS.md`](docs/D5_FINDINGS.md) · [`docs/D6_FINDINGS.md`](docs/D6_FINDINGS.md)
  — the daily measurement logs, including the corrections. D4 section 6 is superseded by
  D5 and carries a banner saying so.
- [`docs/T1_CONTAINER.md`](docs/T1_CONTAINER.md) — why the flagship Tesseract runs
  uncontainerised, and what the driver has to survive on a csh-only CentOS 7 host.
- [`docs/UPSTREAM.md`](docs/UPSTREAM.md) — two bugs found by using the toolkit rather
  than reading it, with the motivating case in this repository.

## Figures

| | |
|---|---|
| [`fig1_pca_manifold`](docs/figures/fig1_pca_manifold.png) | what a device can hand the network. Two principal directions carry **90.5%** of the variation across 192 devices; the freely optimised phi\* sits **13.5** typical device-spacings off that sheet, and wants a ferroelectric memory whose two states conduct alike |
| [`fig2_budget_crossover`](docs/figures/fig2_budget_crossover.png) | sample efficiency against solver calls. Gradient descent is the **only** arm that converts extra budget into performance — random search is flat to six decimal places from 12 calls to 48 |
| [`fig3_hysteresis_descent`](docs/figures/fig3_hysteresis_descent.png) | the device moving. The Id–Vg loop at every accepted step: memory window **0.415 → 0.576 V**, bought by giving up subthreshold slope, **71 → 97 mV/dec** |
| [`fig4_spike_raster`](docs/figures/fig4_spike_raster.png) | what that does to the classifier. The layer does **not** fire more (rate 0.4344 → 0.4572, per-neuron correlation 0.9999) — **one spike in eleven moves**, and that is enough to unstick a readout that answered one class for all sixteen beats |

## Status

| | |
|---|---|
| Contract | **frozen** — `OracleInput` / `OracleOutput`, seven smooth FoMs |
| Extraction | all seven within **0.5 %** of a closed-form reference across the design box |
| Smoothness (G4) | ~5e-7 against a 0.15 threshold; the metric halves exactly under grid refinement |
| V3 `check-gradients` | **0 failures / 56 checks** on the shim, **0 / 10** on the network |
| Open oracle (G2) | DEVSIM converges a pn diode, 5.9 decades of rectification |
| Open oracle (G5) | **passed** — hysteretic Id–Vg, memory window **0.394 V** against a 0.1 V gate, ~36 s per design point |
| Containers (G3) | all five build and push to GHCR from CI |
| Tier C (Sentaurus replay) | **verified** — 164 of 164 float64 values bit-identical, with sockets and subprocesses blocked; zero orphan cache entries |
| Provenance (G10) | 5,749 forward evaluations logged with backend and input hash; all 15 flagship steps present, a real solver wrote every one |
| Tests | **155**, lint clean |

## License

Apache-2.0. Only self-authored input decks and numeric outputs are published here — no Synopsys
binaries, no Synopsys-shipped parameter files, no `.cmd` fragments from Synopsys documentation.
