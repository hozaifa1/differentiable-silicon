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
#
# W IS TRAINED, and that part of D2's lesson still stands: with RANDOM fixed
# weights the classifier sits at chance for every device, the balanced
# cross-entropy is ln(4) = 1.3863 everywhere in the design box, and the
# optimiser descends a surface with no slope in it. What changed on D4 is WHERE
# the training happens -- once, at a fixed reference device, instead of again at
# every design point. See the block below for the measurement that forced it.
#
# Why the VJP may hold W fixed. L(phi) = L(phi; W*(phi)), so
#
#     dL/dphi = partial L/partial phi + (partial L/partial W) . dW*/dphi
#
# and the second term vanishes at a stationary W*. In `frozen` mode W does not
# depend on phi at all, so dW*/dphi is exactly zero and the VJP is EXACT -- which
# it never was before. In `adapt` mode it is an approximation again, but a far
# better one than it used to be: twenty steps at 1e-4 from a shared start cannot
# move W far, and the measured noise of 5.3e-05 against a signal of 0.26 is what
# that buys.
# --- HOW THE NETWORK IS FITTED TO A DESIGN POINT ----------------------------
#
# CHANGED 2026-08-27 (D4). Until today every design point trained a network from
# scratch. That is what made the objective unusable, and the measurement is
# blunt enough to quote in full.
#
# Scoring the eight banked devices three ways. SIGNAL is how far the loss ranges
# across genuinely different devices; NOISE is how far it moves when the device
# does NOT really change (phi nudged in its seventh decimal):
#
#     scheme                       signal     noise    signal/noise
#     frozen                       0.2960    0.0000       exact
#     adapt (shared start + 20)    0.2595    5.3e-05        4882
#     scratch (what it was)        0.0691    0.0260         2.7
#
# Two things there, and the second is the more interesting one.
#
# 1. Retraining from scratch is CHAOTIC in phi. Two nearly identical devices
#    send Adam down different paths. It gets worse the better the network gets,
#    which is the opposite of trainable-away: at 150 steps the noise is 5.5e-4,
#    at 400 steps 8.2e-3, at 800 steps 2.4e-2. Decaying the learning rate does
#    not fix it either (150 steps: 1.9e-3 -> 1.0e-3; 300 steps: 6.2e-3 -> 9.2e-3,
#    i.e. worse).
#
# 2. Retraining from scratch also DESTROYS FOUR FIFTHS OF THE SIGNAL. Given a
#    fresh network and enough steps, a bad device is partly compensated for by
#    training, so every device ends up scoring about the same. An objective for
#    CO-DESIGN must do the opposite: it has to let the device matter.
#
# So the network is fitted once, at a fixed reference device, and every design
# point starts from that. Three modes:
#
#   SNN_TRAIN_MODE=adapt   (default) shared start, then ADAPT_STEPS at ADAPT_LR.
#       "Chip-in-the-loop": the deployed network is briefly tuned to the part it
#       lands on. Keeps the question "how well can a network built on THIS
#       device do" while giving the trajectory no room to diverge.
#   SNN_TRAIN_MODE=frozen  shared start, no per-device training at all. This is
#       exactly what his thesis does -- train in software, deploy onto measured
#       FeFET levels (`post_quantize.py`, accuracy 0.857 -> 0.830). It also makes
#       the VJP EXACT rather than approximate: the envelope-theorem argument for
#       holding W fixed needs a stationary W*, and a W that never moves is
#       trivially stationary.
#   SNN_TRAIN_MODE=scratch the pre-D4 behaviour. Kept so old runs replay and so
#       the comparison above can be reproduced. Not usable for optimisation.
TRAIN_MODE = os.environ.get("SNN_TRAIN_MODE", "adapt")

TRAIN_STEPS = int(os.environ.get("SNN_TRAIN_STEPS", "200"))
TRAIN_LR = float(os.environ.get("SNN_TRAIN_LR", "5e-3"))

# The shared network. Trained once, at the reference device in
# config/circuit.yaml, and cached on disk so a 120-call flagship pays for it
# once rather than 120 times.
W0_STEPS = int(os.environ.get("SNN_W0_STEPS", "800"))
W0_LR = float(os.environ.get("SNN_W0_LR", "5e-3"))

# The per-device tune. SHORT and SLOW on purpose: the whole failure above was a
# long trajectory at a high rate. 20 steps at 1e-4 moves the weights enough to
# adapt and not enough to wander -- measured noise 5.3e-05 against a signal of
# 0.26.
# --- HOW dL/dphi IS OBTAINED, AND WHY IT IS NOT AUTOGRAD ---------------------
#
# CHANGED 2026-08-27 (D4), after measuring the thing nobody had measured.
#
# `torch.autograd.grad(L, phi)` through this network DISAGREES WITH THE NETWORK'S
# OWN LOSS. Measured at the flagship's starting corner, against a central finite
# difference of the same network:
#
#     spikes      |autograd|     |finite diff|    cosine
#     hard          1.21e+06         3.86          -0.983
#     smooth        3.11e+16         1.76e+05      -0.000
#
# It is not a wiring error and it is not the hard/smooth mismatch -- the smooth
# relaxation, which is a genuinely differentiable function, is WORSE. It is
# arithmetic: the network is recurrent over 111 timesteps, and if the per-step
# Jacobian has spectral radius a little over one, backpropagation multiplies 111
# of them together. 1.4^111 is 3e16.
#
# And 3e16 cannot be a derivative of this loss. A class-balanced cross-entropy
# over four classes is bounded by about 5, so across an interval of 6e-4 its
# average slope cannot exceed ~8000. A pointwise derivative of 3e16 on a
# function that bounded means the backward pass is amplifying floating-point
# noise rather than measuring a slope.
#
# TRUNCATED BACKPROPAGATION WAS TRIED FIRST and is not enough. Detaching the
# recurrence every k steps fixes the magnitude but not the direction; measured
# cosine against finite differences: k=1 +0.43, k=2 +0.54, k=5 +0.45, k=10 -0.46,
# k=20 -0.47, k=40 -0.27, none -0.79. Nothing reaches the +0.7 a line search
# needs. The surrogate gradient is an excellent heuristic for TRAINING WEIGHTS,
# which is what it was invented for; it is not an estimator of the sensitivity
# to five hyperparameters that act on every neuron at every timestep.
#
# SO dL/dphi IS MEASURED, from real forward evaluations of this same network, by
# central differences. That is not a retreat from the project's claim -- it is
# the project's claim, applied consistently. The forward pass is never a
# surrogate; only derivatives are estimated, and they are estimated from
# evaluations that really happened. It is exactly what the adjoint shim already
# does for the solver, for exactly the same reason: the thing in the middle has
# no usable adjoint.
#
# Five inputs, central differences, ten evaluations. In `frozen` mode an
# evaluation is a forward pass and this costs about a second.
#
# The autograd path is KEPT behind SNN_VJP=autograd, because it is what
# `check-gradients` can verify against the smooth relaxation, and because the
# measurement above should stay reproducible.
VJP_MODE = os.environ.get("SNN_VJP", "fd")

# Relative step. Not arbitrary: with hard spikes the loss is piecewise constant
# in phi and only moves when a spike flips, so the step has to be large enough
# to cross flips and small enough to stay local. 1e-3 relative was checked
# against the composed loss at the starting corner.
VJP_REL_H = float(os.environ.get("SNN_VJP_REL_H", "1e-3"))

ADAPT_STEPS = int(os.environ.get("SNN_ADAPT_STEPS", "20"))
ADAPT_LR = float(os.environ.get("SNN_ADAPT_LR", "1e-4"))

# Keyed on the full (phi, seed, batch, smoothing) tuple, so `apply` and
# `vector_jacobian_product` at the same design point share one training run
# instead of doing it twice and disagreeing in the last digit.
_TRAINED: dict[tuple, dict] = {}


def _key(inputs: InputSchema) -> tuple:
    return (
        int(inputs.seed),
        int(inputs.batch),
        bool(inputs.smooth_spikes),
        TRAIN_MODE,
        int(TRAIN_STEPS), int(W0_STEPS), int(ADAPT_STEPS),
        float(TRAIN_LR), float(W0_LR), float(ADAPT_LR),
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


def _reference_phi():
    """The device the shared network is trained at. Frozen in circuit.yaml.

    It must NOT depend on the design point being scored. That is the whole
    mechanism: a shared starting point only removes the chaos if it is shared.
    """
    from diffsilicon.shared.circuit import load_circuit

    c = load_circuit()
    return {"beta": c.w0_beta, "g_min": c.w0_g_min, "g_max": c.w0_g_max,
            "th_th": c.w0_th_th, "sig_w": c.w0_sig_w}


def _fit(net, phi_d, steps, lr, seed, smooth, x, y):
    """`steps` Adam steps on `net` under `phi_d`. Returns `net`."""
    phi = {k: torch.tensor(float(phi_d[k]), dtype=DTYPE) for k in PHI_KEYS}
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # enable_grad, explicitly: `apply` evaluates under torch.no_grad(), and the
    # inner loop still needs a tape of its own.
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            logits, _ = net(x, phi, seed=seed, smooth=smooth)
            balanced_ce(logits, y, N_CLASSES).backward()
            opt.step()
    return net


_W0: dict[tuple, dict] = {}


def _w0_state(inputs: InputSchema):
    """The shared network, trained once at the reference device.

    Cached in memory for the process AND on disk, because a 120-call flagship
    would otherwise pay 800 training steps on every call, and a fresh process
    would pay them again. The disk key covers everything that changes the
    answer, so a stale one cannot be served.
    """
    key = (int(inputs.seed), int(inputs.batch), bool(inputs.smooth_spikes),
           int(W0_STEPS), float(W0_LR), TASK, int(ECG_POOL),
           tuple(round(v, 12) for v in _reference_phi().values()))
    if key in _W0:
        return _W0[key]

    import hashlib
    import json as _json

    tag = hashlib.sha256(_json.dumps(key, default=str).encode()).hexdigest()[:32]
    root = Path(os.environ.get("DIFFSILICON_CACHE_ROOT",
                               str(_HERE.parents[2] / "results" / "cache"))) / "w0"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{tag}.pt"
    if path.is_file():
        _W0[key] = torch.load(path, map_location="cpu", weights_only=True)
        return _W0[key]

    net = _build_net(inputs.seed)
    x, y = _batch(inputs)
    _fit(net, _reference_phi(), W0_STEPS, W0_LR, inputs.seed,
         inputs.smooth_spikes, x, y)
    state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    torch.save(state, path)
    _W0[key] = state
    return state


def _net(inputs: InputSchema):
    """The network this design point is scored on. Deterministic in the key."""
    net = _build_net(inputs.seed)
    key = _key(inputs)
    if key in _TRAINED:
        net.load_state_dict(_TRAINED[key])
        return net

    phi_d = {k: float(getattr(inputs, k)) for k in PHI_KEYS}
    x, y = _batch(inputs)

    if TRAIN_MODE == "scratch":
        _fit(net, phi_d, TRAIN_STEPS, TRAIN_LR, inputs.seed,
             inputs.smooth_spikes, x, y)
    else:
        net.load_state_dict(_w0_state(inputs))
        if TRAIN_MODE == "adapt":
            _fit(net, phi_d, ADAPT_STEPS, ADAPT_LR, inputs.seed,
                 inputs.smooth_spikes, x, y)
        elif TRAIN_MODE != "frozen":
            raise ValueError(
                f"SNN_TRAIN_MODE={TRAIN_MODE!r}; expected one of "
                "'adapt', 'frozen', 'scratch'."
            )

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


def _vjp_by_finite_difference(inputs: InputSchema, vjp_inputs, vjp_outputs,
                              cotangent_vector):
    """dL/dphi from central differences of this network's own loss.

    See the block on VJP_MODE for why autograd is not used here. Ten forward
    evaluations, all of them real; nothing is modelled.
    """
    keys = [k for k in PHI_KEYS if k in vjp_inputs]
    want_loss = "loss" in vjp_outputs
    want_spikes = "spikes" in vjp_outputs
    if not (want_loss or want_spikes):
        return dict.fromkeys(vjp_inputs, 0.0)

    c_loss = float(cotangent_vector.get("loss", 0.0)) if want_loss else 0.0
    c_spk = float(cotangent_vector.get("spikes", 0.0)) if want_spikes else 0.0

    def scalar_at(**over):
        upd = inputs.model_copy(update=over)
        with torch.no_grad():
            _, loss, spikes, _ = _forward(upd, requires_grad=False)
        return c_loss * float(loss) + c_spk * float(spikes)

    out = {}
    for k in keys:
        v = float(getattr(inputs, k))
        # An absolute floor so a phi component that legitimately sits near zero
        # still gets a real step rather than one of size zero.
        h = max(abs(v) * VJP_REL_H, 1e-12)
        out[k] = (scalar_at(**{k: v + h}) - scalar_at(**{k: v - h})) / (2.0 * h)
    for k in vjp_inputs:
        out.setdefault(k, 0.0)
    return out


def vector_jacobian_product(
    inputs: InputSchema, vjp_inputs: set[str], vjp_outputs: set[str], cotangent_vector
):
    """Hop 1 of the reverse sweep: PyTorch's tape ends here and JAX picks it up."""
    if VJP_MODE == "fd":
        return _vjp_by_finite_difference(inputs, vjp_inputs, vjp_outputs,
                                         cotangent_vector)
    if VJP_MODE != "autograd":
        raise ValueError(f"SNN_VJP={VJP_MODE!r}; expected 'fd' or 'autograd'.")
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
