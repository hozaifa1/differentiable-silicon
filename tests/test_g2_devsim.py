"""G2 -- DEVSIM imports, meshes, and converges. The gate for the open oracle.

DEVSIM is what makes any of this reproducible without a Synopsys license, so if
it cannot solve a pn diode on this machine the whole Tier B story is gone.

The solve runs in a SUBPROCESS, and that is not tidiness. Measured on D1: PyTorch
and DEVSIM cannot share a Windows process. Both link Intel OpenMP, and the second
one to initialise aborts the interpreter outright:

    OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll
    already initialized.

`Fatal Python error: Aborted`, no traceback, no exception to catch. The documented
escape hatch, KMP_DUPLICATE_LIB_OK=TRUE, is explicitly "unsafe, unsupported,
undocumented" and Intel warns it can silently produce incorrect results -- which
is the last thing a solver in a gradient path should do.

The architectural consequence lands on D2: the DEVSIM oracle must run out of
process from the PyTorch network. It already does -- T2 is a served Tesseract in
its own container and T4 is another. This is the process boundary earning its keep
rather than being decorative, and it is worth saying so in the writeup.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("devsim", reason="devsim not installed")
pytestmark = pytest.mark.needs_devsim

SCRIPT = Path(__file__).resolve().parent / "g2_devsim_diode.py"


def test_devsim_math_libraries_are_wired_up():
    """The Windows wheel ships no BLAS and looks for an MKL whose highest TESTED
    name is mkl_rt.2.dll, while current MKL wheels install mkl_rt.3.dll. Without
    devsim_env the import fails with 'Error loading math libraries'."""
    from diffsilicon.shared.devsim_env import ensure_math_libs

    if sys.platform == "win32":
        assert ensure_math_libs(), "no MKL found for devsim on Windows"


@pytest.mark.slow
def test_pn_diode_converges_and_rectifies():
    """G2 as specified: the diode example runs to exit status 0."""
    r = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, f"DEVSIM diode failed (exit {r.returncode}):\n{r.stdout[-3000:]}"
    assert "G2 DEVSIM: OK" in r.stdout

    iv = {}
    for line in r.stdout.splitlines():
        if line.startswith("V ="):
            parts = line.replace("V =", "").replace("I =", "").split()
            iv[float(parts[0])] = float(parts[2])

    assert set(iv) == {0.0, 0.2, 0.4, 0.6}
    assert iv[0.6] > 1e-3, "no forward conduction at 0.6 V"
    assert abs(iv[0.0]) < 1e-6, "leaky at zero bias"
    assert iv[0.2] < iv[0.4] < iv[0.6], "I(V) is not monotonic"
    import math

    decades = math.log10(iv[0.6] / iv[0.2])
    assert decades > 4.0, f"only {decades:.1f} decades of forward current over 400 mV"


def test_torch_and_devsim_are_kept_in_separate_processes():
    """Guard the constraint above. If anything ever imports both into one process
    the failure is an abort with no traceback, which is expensive to diagnose twice."""
    import ast

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported
