"""Make the repo importable and keep tests off the committed replay cache."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# THE TEST SESSION RUNS THE SYNTHETIC TASK. Set before any test module imports a
# tesseract_api, because T4 reads SNN_TASK once at module scope.
#
# Two reasons, and neither is that the real task is optional.
#
# 1. CI has no MIT-BIH beats and never will -- they are a preprocessing of a
#    public database, not this project's to redistribute, and they are
#    gitignored. A test suite that needs them is a test suite that only runs on
#    one laptop.
# 2. Tier A promises the whole pipeline in two minutes with no Docker, no
#    licence and no network. The real task is a 160-neuron recurrent LSNN over
#    111 timesteps with an inner Adam loop; it does not fit in two minutes and it
#    is not supposed to.
#
# `setdefault`, so `SNN_TASK=ecg uv run pytest` still exercises the real path,
# and `tests/test_recalibration.py` sets it explicitly for the tests that must.
# There is deliberately NO automatic fallback in the tesseract itself: a run that
# quietly swapped the real task for a synthetic one would be exactly the kind of
# silent surrogate this project exists to rule out.
os.environ.setdefault("SNN_TASK", "synth")


# THE SHARED NETWORK IS TRAINED CHEAPLY IN TESTS. Same code path, fewer steps.
#
# From D4 the network is fitted once at a reference device (800 Adam steps) and
# every design point starts from that. 800 steps is right for a real run and
# absurd inside a unit test -- the suite timed out on the first attempt, paying
# for it before any assertion ran.
#
# So the STEP COUNTS are shrunk and nothing else is. The mode, the reference
# device, the caching and the code path are exactly what a real run uses, which
# is the part the tests are there to protect. A test that asserts on a loss
# VALUE must set these itself; none currently does, because the suite asserts on
# structure, gradients and liveness rather than on classifier quality.
#
# `setdefault` again, so a deliberate `SNN_W0_STEPS=800 pytest` still works.
os.environ.setdefault("SNN_W0_STEPS", "5")
os.environ.setdefault("SNN_ADAPT_STEPS", "2")
os.environ.setdefault("SNN_TRAIN_STEPS", "5")


# --------------------------------------------------------------------------
# devsim: importing it proves it found a math library, not that the library
# carries a working direct solver. That answer only arrives from inside a real
# solve(), and on a machine where it is "no" the failure is indistinguishable
# from a broken device model unless it is named.
#
# A machine that cannot factor a matrix cannot run the oracle AT ALL, which is
# a fact about the machine. Every other assertion in the DEVSIM tests still
# fails normally: this matches only the signatures devsim emits when it has no
# usable solver, never a wrong number.
# --------------------------------------------------------------------------

_NO_DIRECT_SOLVER = (
    "Matrix factorization failed",
    "during LU Factorization",
    "not supported in this build",
    '"direct_solver" parameter value "unknown"',
)


def skip_if_devsim_cannot_factor(output: str) -> None:
    """Skip, rather than fail, when devsim here has no usable direct solver."""
    import pytest

    for signature in _NO_DIRECT_SOLVER:
        if signature in output:
            pytest.skip(
                f"devsim on this machine has no usable direct solver ({signature}). "
                f"Install MKL to run the DEVSIM oracle tests."
            )
