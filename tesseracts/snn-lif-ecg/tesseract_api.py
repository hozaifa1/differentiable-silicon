# SPDX-License-Identifier: Apache-2.0
"""snn-lif-ecg (T4) -- the PyTorch end of the gradient path.

Takes the five numbers the device produced (beta, g_min, g_max, th_th, sig_w),
runs a surrogate-gradient LIF classifier under them, and returns the
class-balanced cross-entropy plus the mean spike count that the energy term uses.

`vector_jacobian_product` is `torch.autograd.grad(L, phi, grad_outputs=cbar)`.
That is hop 1 of the reverse sweep, and it is a real boundary crossing: PyTorch's
tape ends here, the cotangent goes over the wire as five float64s, and JAX picks
it up on the other side with no shared autodiff state whatsoever.

Data: a deliberately imbalanced synthetic spike set today; MIT-BIH inter-patient
DS1/DS2 (de Chazal) lands on D4. The objective, the class weighting and the
gradient path are the same either way -- only the tensors change.
"""

import sys
from pathlib import Path

for _cand in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[2] / "src"):
    if (_cand / "diffsilicon").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

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

N_CLASSES = 4
N_IN = 16
N_HIDDEN = 24
T_STEPS = 24


class InputSchema(BaseModel):
    beta: Differentiable[Float64] = Field(description="Membrane decay per hardware timestep.")
    g_min: Differentiable[Float64] = Field(description="Low synaptic conductance, S.")
    g_max: Differentiable[Float64] = Field(description="High synaptic conductance, S.")
    th_th: Differentiable[Float64] = Field(description="Spikes-to-fire at max weight.")
    sig_w: Differentiable[Float64] = Field(description="Relative weight-noise sigma.")
    seed: int = Field(default=0, description="Threaded to torch for a deterministic batch.")
    batch: int = Field(default=32, description="Batch size.")
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


_NETS: dict[int, LIFNet] = {}


def _net(seed: int) -> LIFNet:
    if seed not in _NETS:
        _NETS[seed] = LIFNet(N_IN, N_HIDDEN, N_CLASSES, seed=seed)
    return _NETS[seed]


def _forward(inputs: InputSchema, requires_grad: bool):
    phi = {
        k: torch.tensor(float(getattr(inputs, k)), dtype=DTYPE, requires_grad=requires_grad)
        for k in PHI_KEYS
    }
    x, y = synthetic_batch(inputs.batch, T_STEPS, N_IN, N_CLASSES, seed=inputs.seed)
    logits, spikes = _net(inputs.seed)(x, phi, seed=inputs.seed, smooth=inputs.smooth_spikes)
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
