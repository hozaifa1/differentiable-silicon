# SPDX-License-Identifier: Apache-2.0
"""snn-lif-ecg (T4) -- the PyTorch end of the gradient path.

Takes the five numbers the device produced (beta, g_min, g_max, th_th, sig_w),
runs a surrogate-gradient LIF classifier under them, and returns the
class-balanced cross-entropy plus the mean spike count that the energy term uses.

`vector_jacobian_product` is `torch.autograd.grad(L, phi, grad_outputs=cbar)`.
That is hop 1 of the reverse sweep, and it is a real boundary crossing: PyTorch's
tape ends here, the cotangent goes over the wire as five float64s, and JAX picks
it up on the other side with no shared autodiff state whatsoever.

Data, as of the D3 recalibration: 2000 curated MIT-BIH beats in four AAMI
classes, loaded from the thesis' own preprocessing, on the thesis' own LSNN --
delayed synapses, a mixed LIF/adaptive-LIF recurrent layer, a low-pass readout.
See `diffsilicon.snn.ecg` (including why the split is intra-patient and why that
is forced, not chosen) and `diffsilicon.snn.lsnn`.

The synthetic path is still here behind SNN_TASK=synth, because CI and the
gradient checks must run with no dataset on disk. The objective, the class
weighting and the gradient path are the same either way -- only the tensors and
the network change.
"""

import os
import sys
from pathlib import Path

# Locally the package lives at <repo>/src/diffsilicon; inside a built Tesseract,
# build_config.package_data drops the same tree beside this file at
# /tesseract/diffsilicon. Note the `parents` slice has to be bounded: in the
# container this file sits two levels from the filesystem root, and indexing past
# it raises IndexError -- which surfaces only as "Could not load module from
# /tesseract/tesseract_api.py" during `tesseract build`.
_HERE = Path(__file__).resolve()
_CANDIDATES = [_HERE.parent, *(p / "src" for p in _HERE.parents)]
for _cand in _CANDIDATES:
    if (_cand / "diffsilicon").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
        break

import torch  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from tesseract_core.runtime import Differentiable, Float64, ShapeDType  # noqa: E402

from diffsilicon.snn.lif import (  # noqa: E402
    DTYPE,
    PHI_KEYS,
    LIFNet,
    balanced_ce,
    synthetic_batch,
)
from diffsilicon.snn.lsnn import THESIS_LSNN, LSNNNet  # noqa: E402

N_CLASSES = 4

# --- the D3 recalibration: real data, and the thesis' own network -------------
#
# SNN_TASK = ecg   -> 2000 curated MIT-BIH beats and the thesis LSNN (default)
#          = synth -> the D1 synthetic spike set and LIFNet
#
# The synthetic path is KEPT, and not out of sentiment. It needs no dataset on
# disk, so it is what CI, the gradient checks and the Tier A "runs in two minutes
# with no license and no network" claim actually execute. It is no longer what
# any reported result is measured on: as of D2 it SATURATES, with accuracy 1.000
# almost everywhere in the design box, which is why real data became the
# bottleneck rather than a nicety.
TASK = os.environ.get("SNN_TASK", "ecg")

# Timesteps are POOLED, and the pool is not arbitrary.
#
# A beat is 1116 steps of 0.5556 ms. Pooling 10 of them gives a 5.556 ms
# timestep, which matches the frozen dt_bio = 5.625 ms of config/circuit.yaml to
# 1.2% -- so V_leak, K_syn, dt_hw and th_th all keep meaning exactly what they
# meant, and nothing frozen has to move to accommodate real data.
#
# It also lands on a genuine agreement rather than a convenience. At that
# timestep the thesis' own membrane decay exp(-dt/tau_mem) is 0.6065, and the
# beta this project's DEVICE produces at the nominal design point is 0.6033.
# Those were derived independently -- one from an RC circuit fitted to VO2, one
# from a FeFET's subthreshold leak through the DPI relation -- and they agree to
# half a percent. That is the alignment this recalibration was asking for.
#
# The one thing pooling does move is A_accel, the assumed hardware acceleration
# factor, which becomes dt_bio/dt_hw = 505 rather than the assumed 512. A_accel
# is labelled ASSUMED in circuit.yaml and enters no frozen derivation.
ECG_POOL = int(os.environ.get("SNN_ECG_POOL", "10"))
BATCH_DEFAULT = int(os.environ.get("SNN_BATCH", "16"))

if TASK == "ecg":
    from diffsilicon.snn.ecg import N_IN, ecg_batch  # noqa: E402

    N_HIDDEN = THESIS_LSNN["n_lif"] + THESIS_LSNN["n_alif"]
else:
    N_IN = 16
    N_HIDDEN = 24
T_STEPS = 24  # synthetic path only; the ECG path gets its length from the data


class InputSchema(BaseModel):
    beta: Differentiable[Float64] = Field(description="Membrane decay per hardware timestep.")
    g_min: Differentiable[Float64] = Field(description="Low synaptic conductance, S.")
    g_max: Differentiable[Float64] = Field(description="High synaptic conductance, S.")
    th_th: Differentiable[Float64] = Field(description="Spikes-to-fire at max weight.")
    sig_w: Differentiable[Float64] = Field(description="Relative weight-noise sigma.")
    seed: int = Field(default=0, description="Threaded to torch for a deterministic batch.")
    batch: int = Field(default=BATCH_DEFAULT, description="Batch size.")
    smooth_spikes: bool = Field(
        default=False,
        description="Replace the Heaviside with the soft-Heaviside it relaxes to. "
        "Makes the network differentiable in the ordinary sense so that "
        "check-gradients has something it can legitimately verify. Never used for "
        "a reported result.",
    )


class OutputSchema(BaseModel):
    loss: Differentiable[Float64] = Field(description="Class-balanced cross-entropy.")
    spikes: Differentiable[Float64] = Field(
        description="Mean spikes per neuron per timestep; the energy term is proportional to this."
    )
    accuracy: Float64 = Field(description="Plain accuracy, reported but never optimised.")


# --- the inner training loop -------------------------------------------------
# W IS TRAINED. Skipping this was the single most expensive bug of D2: with
# random fixed weights the classifier sits at chance for every device, the
# class-balanced cross-entropy is ln(4) = 1.3863 plus noise everywhere in the
# design box, and the flagship optimiser spends its whole solver budget
# descending a surface that has no slope in it. Measured: stepping 0.005 to 0.4
# along the manufactured descent direction moved the loss to 1.363, 1.431, 1.398,
# 1.427, 1.398, 1.389, 1.387 -- a random walk converging back to chance.
#
# So each evaluation trains W to (approximate) stationarity under the phi it was
# handed, and reports the loss THERE. That is the question the project is
# actually asking: given a device, how well can a network built on it do.
#
# Why the VJP may then hold W fixed. L(phi) = L(phi; W*(phi)), so
#
#     dL/dphi = partial L/partial phi + (partial L/partial W) . dW*/dphi
#
# and at a stationary W* the second term vanishes because partial L/partial W is
# zero. The envelope theorem does the work; no differentiation through the
# optimiser is needed, and none is done. The approximation is the "stationary"
# in "approximately stationary", and TRAIN_STEPS is the knob that controls it.
TRAIN_STEPS = int(os.environ.get("SNN_TRAIN_STEPS", "200"))
TRAIN_LR = float(os.environ.get("SNN_TRAIN_LR", "5e-3"))

# Keyed on the full (phi, seed, batch, smoothing) tuple, so `apply` and
# `vector_jacobian_product` at the same design point share one training run
# instead of doing it twice and disagreeing in the last digit.
_TRAINED: dict[tuple, dict] = {}


def _key(inputs: InputSchema) -> tuple:
    return (
        int(inputs.seed),
        int(inputs.batch),
        bool(inputs.smooth_spikes),
        tuple(float(getattr(inputs, k)) for k in PHI_KEYS),
    )


def _build_net(seed: int):
    """The network this task runs on. See TASK above for why both still exist."""
    if TASK == "ecg":
        return LSNNNet(dt_s=THESIS_LSNN["dt_s"] * ECG_POOL, seed=seed)
    return LIFNet(N_IN, N_HIDDEN, N_CLASSES, seed=seed)


def _batch(inputs: InputSchema):
    """The FIXED batch this design point is scored on.

    Fixed, not resampled: the optimiser compares losses between design points, so
    a batch that moved between calls would turn every trust-region rho into
    noise. Same reason `FlagshipConfig.seed` is held constant across steps.
    """
    if TASK == "ecg":
        return ecg_batch(inputs.batch, seed=inputs.seed, pool=ECG_POOL, dtype=DTYPE)
    return synthetic_batch(inputs.batch, T_STEPS, N_IN, N_CLASSES, seed=inputs.seed)


def _net(inputs: InputSchema):
    """A network trained under this phi. Deterministic in the whole key."""
    net = _build_net(inputs.seed)
    key = _key(inputs)
    if key in _TRAINED:
        net.load_state_dict(_TRAINED[key])
        return net

    phi = {k: torch.tensor(float(getattr(inputs, k)), dtype=DTYPE) for k in PHI_KEYS}
    x, y = _batch(inputs)
    opt = torch.optim.Adam(net.parameters(), lr=TRAIN_LR)
    # enable_grad, explicitly: `apply` evaluates under torch.no_grad(), and the
    # inner loop still needs a tape of its own.
    with torch.enable_grad():
        for _ in range(TRAIN_STEPS):
            opt.zero_grad()
            logits, _ = net(x, phi, seed=inputs.seed, smooth=inputs.smooth_spikes)
            balanced_ce(logits, y, N_CLASSES).backward()
            opt.step()
    _TRAINED[key] = {k: v.detach().clone() for k, v in net.state_dict().items()}
    return net


def _forward(inputs: InputSchema, requires_grad: bool):
    phi = {
        k: torch.tensor(float(getattr(inputs, k)), dtype=DTYPE, requires_grad=requires_grad)
        for k in PHI_KEYS
    }
    net = _net(inputs)
    for prm in net.parameters():
        prm.requires_grad_(False)  # the envelope theorem argument above
    x, y = _batch(inputs)
    logits, spikes = net(x, phi, seed=inputs.seed, smooth=inputs.smooth_spikes)
    loss = balanced_ce(logits, y, N_CLASSES)
    acc = (logits.argmax(1) == y).to(DTYPE).mean()
    return phi, loss, spikes, acc


def apply(inputs: InputSchema) -> OutputSchema:
    with torch.no_grad():
        _, loss, spikes, acc = _forward(inputs, requires_grad=False)
    return OutputSchema(loss=float(loss), spikes=float(spikes), accuracy=float(acc))


def abstract_eval(abstract_inputs):
    scalar = ShapeDType(shape=(), dtype="float64")
    return {"loss": scalar, "spikes": scalar, "accuracy": scalar}


def vector_jacobian_product(
    inputs: InputSchema, vjp_inputs: set[str], vjp_outputs: set[str], cotangent_vector
):
    """Hop 1 of the reverse sweep: PyTorch's tape ends here and JAX picks it up."""
    phi, loss, spikes, _ = _forward(inputs, requires_grad=True)
    outs, cts = [], []
    if "loss" in vjp_outputs:
        outs.append(loss)
        cts.append(torch.tensor(float(cotangent_vector["loss"]), dtype=DTYPE))
    if "spikes" in vjp_outputs:
        outs.append(spikes)
        cts.append(torch.tensor(float(cotangent_vector["spikes"]), dtype=DTYPE))
    if not outs:
        return {k: 0.0 for k in vjp_inputs}

    wanted = [phi[k] for k in PHI_KEYS if k in vjp_inputs]
    grads = torch.autograd.grad(outs, wanted, grad_outputs=cts, allow_unused=True)
    keys = [k for k in PHI_KEYS if k in vjp_inputs]
    return {k: float(g) if g is not None else 0.0 for k, g in zip(keys, grads, strict=True)}


def jacobian_vector_product(
    inputs: InputSchema, jvp_inputs: set[str], jvp_outputs: set[str], tangent_vector
):
    """Built from the VJP one output at a time: there are only two of them."""
    out = {}
    for name in jvp_outputs:
        row = vector_jacobian_product(inputs, jvp_inputs, {name}, {name: 1.0})
        out[name] = sum(row[k] * float(tangent_vector[k]) for k in jvp_inputs)
    return out
