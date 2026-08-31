# SPDX-License-Identifier: Apache-2.0
"""TIER B: the same pipeline, with the solver in a container, over HTTP.

Tier A proves every wire in one process. What it cannot prove is the claim this
project is actually built on (that swapping the closed-source commercial solver
for the Apache-2.0 one is one environment variable) because in one process
there is no container to swap in and no wire to lose anything across.

This file is that claim, tested. It needs a served oracle Tesseract and skips
itself without one:

    docker pull ghcr.io/hozaifa1/devsim-fefet:latest
    tesseract serve ghcr.io/hozaifa1/devsim-fefet:latest --port 8101
    ORACLE_URL=http://localhost:8101 uv run pytest tests/test_tier_b_served.py

The three tests are the three ways that claim could quietly be false: the schema
drifts once it goes over the wire, the pipeline falls back to the mock and nobody
notices, or the gradient does not survive the hop.

WHY THIS FILE EXISTS AT ALL. The README used to send a judge at
`tests/test_tier_a_pipeline.py` with ORACLE_BACKEND=url in front of it. That file
has an autouse fixture pinning ORACLE_BACKEND to `mock`, so the command ran green
against the analytic mock and touched no container. Green, and worth nothing.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tesseract_core import Tesseract

jax.config.update("jax_enable_x64", True)

from diffsilicon.pipeline import composed_loss, oracle_call  # noqa: E402
from diffsilicon.shared.circuit import load_circuit  # noqa: E402
from diffsilicon.shared.contract import (  # noqa: E402
    DIFFERENTIABLE_OUTPUTS,
    make_oracle_input,
)
from diffsilicon.shared.oracle import run_oracle  # noqa: E402

ORACLE_URL = os.environ.get("ORACLE_URL")

pytestmark = pytest.mark.skipif(
    not ORACLE_URL,
    reason="Tier B needs a served oracle Tesseract. Set ORACLE_URL; see this file's docstring.",
)

API = "tesseracts/{}/tesseract_api.py"
CC = load_circuit()

# The d=4 nominal device, and the numbers DEVSIM produced for it on the
# development machine on 2026-08-27, banked in
# results/runs/rebaseline_d3_devsim.json. The container has to reproduce these
# on a different operating system with a different BLAS underneath it.
NOMINAL_THETA = np.array([0.2, 0.5, 0.5, 1.0 / 3.0], dtype=np.float64)
BANKED_DEVSIM = {
    "ss": 82.13841011089325,
    "vth_fwd": 0.5983457135190848,
    "vth_rev": 0.07038718941748451,
    "i_leak": 1.3022296241015844e-11,
    "g_lo": 2.751196907937893e-06,
    "g_hi": 0.0002660189920715208,
    "dg_dvth": 0.00035114333796607477,
}


@pytest.fixture(scope="module")
def served():
    with Tesseract.from_url(ORACLE_URL) as t:
        yield t


@pytest.fixture
def _url_backend(monkeypatch):
    monkeypatch.setenv("ORACLE_BACKEND", "url")
    monkeypatch.setenv("ORACLE_URL", ORACLE_URL)
    # A cache hit would answer from disk and never touch the container, which is
    # the one thing this file is here to exercise.
    monkeypatch.setenv("DIFFSILICON_CACHE_DISABLE", "1")


def test_the_served_container_publishes_the_frozen_schema(served):
    """The 'one environment variable' claim, over the wire this time.

    Tier A asserts T1 and T2 publish identical schemas in-process. That is the
    easy half: both are the same Python objects imported from the same module. A
    container serialises its schema through FastAPI and back, and this is where a
    field would quietly change type or lose a default.
    """
    local = Tesseract.from_tesseract_api(API.format("sentaurus-fefet"))
    with local:
        ref = local.openapi_schema["components"]["schemas"]
    got = served.openapi_schema["components"]["schemas"]
    for name in ("Apply_InputSchema", "Apply_OutputSchema"):
        assert got[name] == ref[name], (
            f"{name} served by the container has drifted from the frozen contract"
        )


def test_the_forward_value_comes_from_the_container_and_not_the_mock(_url_backend):
    """Two failures at once: a silent fall back to the mock, and a container that
    solves something other than what the development machine solved.

    `i_leak` and `g_lo` are the discriminators. The mock and DEVSIM agree on
    subthreshold swing at this device to about 1%, so `ss` would not catch a
    substitution; on those two they are eight and a half times apart.
    """
    out = run_oracle(make_oracle_input(NOMINAL_THETA))

    assert float(out.converged) == 1.0
    # An analytic mock returns in microseconds; a drift-diffusion solve does not,
    # and the number is carried on the record even when the container replays one
    # of its own. So this check catches a mock standing in for a cold solve.
    assert float(out.solver_seconds) > 1.0, (
        "this value was produced in under a second; that is an analytic model "
        "answering, not a drift-diffusion solve"
    )

    mock = run_oracle(make_oracle_input(NOMINAL_THETA), "mock")
    for key in ("i_leak", "g_lo"):
        ratio = float(getattr(out, key)) / float(getattr(mock, key))
        assert not 0.2 < ratio < 5.0, (
            f"{key} from the served container is within a factor of five of the "
            f"analytic mock's value; the url backend may have fallen back"
        )

    for key in DIFFERENTIABLE_OUTPUTS:
        got, want = float(getattr(out, key)), BANKED_DEVSIM[key]
        assert got == pytest.approx(want, rel=1e-2), (
            f"{key}: container {got:.6e} vs the banked development-machine value "
            f"{want:.6e}. Both are DEVSIM; a disagreement above 1% is a real "
            f"difference in the build, not floating-point weather."
        )


def test_the_gradient_survives_the_container_hop(_url_backend):
    """The headline claim with the container in the loop.

    PyTorch autograd -> wire -> JAX -> manufactured adjoint -> HTTP -> DEVSIM in a
    container -> back. Every component of dL/dtheta has to come back finite and
    non-zero; a zero column is a hop that silently detached.

    This costs 2D+1 = 9 container solves and is the slow test in the suite. It is
    also the only one that runs the thing the README tells a judge to run.
    """
    shim = Tesseract.from_tesseract_api(API.format("adjoint-shim"))
    snn = Tesseract.from_tesseract_api(API.format("snn-lif-ecg"))
    with shim, snn:
        theta = jnp.asarray(NOMINAL_THETA)

        y = oracle_call(shim, theta)
        assert float(y["ss"]) == pytest.approx(BANKED_DEVSIM["ss"], rel=1e-2), (
            "the shim must PROXY the container's forward value, not model it"
        )

        g = np.asarray(jax.grad(lambda t: composed_loss(shim, snn, t, CC, lambda_e=1e6))(theta))

    assert g.shape == (4,)
    assert np.all(np.isfinite(g))
    assert np.all(np.abs(g) > 0.0), f"a design parameter received no gradient: {g}"
