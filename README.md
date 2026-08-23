# Differentiable Silicon

**Backpropagating a class-balanced ECG classification loss through a spiking network, through a
subthreshold neuron circuit, and into a closed-source commercial TCAD solver — to obtain
∂L/∂(ferroelectric process parameters).**

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
| **T4** | `snn-lif-ecg` | surrogate-gradient LIF on MIT-BIH, inter-patient DS1/DS2 | PyTorch autograd | `apply`, `abstract_eval`, **`vjp`**, `jvp` |

T3 proxies its forward pass straight to whichever oracle `ORACLE_URL` points at, so the forward
value is never a surrogate — only `vjp` is.

## Reproduction — four tiers

**Tier A — 2 min, no Docker, no license.** The complete pipeline against an analytic mock oracle,
in-process via `Tesseract.from_tesseract_api()`. Proves every wire, every schema, every gradient hop.

```bash
uv sync --group dev && uv run pytest
```

**Tier B — Docker, no license.** Swap the commercial solver for the Apache-2.0 one.

```bash
docker pull ghcr.io/hozaifa1/devsim-fefet:latest
```

Then `tesseract serve devsim-fefet --port 8101` and point the orchestrator at it. **Swapping the
closed-source commercial solver for the open one is one environment variable — `ORACLE_URL`.**
Nothing else in the pipeline changes. That line is the whole reason this is built on Tesseract.

**Tier C — regenerates every Sentaurus figure with no license and no network.** `results/cache/`
is a content-addressed replay of every Sentaurus call ever made, populated as a side effect of every
run rather than reconstructed at the end.

```bash
ORACLE_BACKEND=replay uv run python -m diffsilicon.race
```

**Tier D — bring your own license.** See [`docs/T1_CONTAINER.md`](docs/T1_CONTAINER.md) and
`.env.example`. `t1/Dockerfile` expects the Sentaurus tree bind-mounted at `/opt/synopsys` and takes
`SNPSLMD_LICENSE_FILE` for license-server passthrough; the flagship itself runs uncontainerised, and
that document explains why.

## License

Apache-2.0. Only self-authored input decks and numeric outputs are published here — no Synopsys
binaries, no Synopsys-shipped parameter files, no `.cmd` fragments from Synopsys documentation.
