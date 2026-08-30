"""Make `import devsim` work without the caller knowing anything about BLAS.

The devsim wheel ships NO math library on either platform. It looks for
libopenblas / liblapack / libblas by bare name, or for an Intel MKL whose highest
*tested* name is `mkl_rt.2.dll`.

* **Windows:** current MKL wheels install `mkl_rt.3.dll` inside
  `<prefix>/Library/bin`, which devsim loads happily once it is told the name and
  the directory is on the DLL search path.
* **Linux:** a bare `ubuntu-latest` has none of the three, and `import devsim`
  fails with `RuntimeError: Issues initializing DEVSIM`. Debian ships the
  unversioned `libopenblas.so` symlink only in `libopenblas-dev`, so either that
  package is installed (what CI and the T2 image do) or this module finds a
  versioned `libopenblas.so.N` and passes its full path.

Note the failure mode: a RuntimeError, not an ImportError, so
`pytest.importorskip` does NOT skip on it -- it errors out during collection.

Import this module (or call `ensure_math_libs()`) before `import devsim`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["ensure_math_libs", "ensure_direct_solver", "import_devsim"]

_done = False


def _candidate_dll_dirs() -> list[Path]:
    dirs = [Path(sys.prefix) / "Library" / "bin"]
    # uv puts the venv beside the project; also try the base interpreter.
    base = getattr(sys, "base_prefix", sys.prefix)
    dirs.append(Path(base) / "Library" / "bin")
    return [d for d in dirs if d.is_dir()]


def _mkl_candidates() -> list[str]:
    """Every libmkl_rt this interpreter might be able to load, newest name first."""
    from glob import glob

    roots = [Path(sys.prefix), Path(getattr(sys, "base_prefix", sys.prefix))]
    patterns = [str(r / "lib" / "libmkl_rt.so*") for r in roots]
    patterns += ["/usr/lib/x86_64-linux-gnu/libmkl_rt.so*", "/usr/lib64/libmkl_rt.so*"]
    seen: list[str] = []
    for pat in patterns:
        for m in sorted(glob(pat), reverse=True):
            if m not in seen:
                seen.append(m)
    return seen


def _ensure_linux_blas() -> str | None:
    """Find a math library devsim can load, preferring one that carries a solver.

    OpenBLAS is enough to make `import devsim` succeed and is NOT enough to run
    anything: devsim picks its direct solver from the math library it loaded, and
    with OpenBLAS it picks none, leaving `direct_solver` at "unknown". SuperLU is
    not compiled into every wheel, so the fallback can fail too. The run then gets
    all the way through meshing and parameter setup and dies on the first solve().

    MKL is the one option that always brings a direct solver with it, so it is
    searched for first, and BLAS/LAPACK stay as the fallback for a machine that
    has no MKL at all.
    """
    import ctypes.util
    from glob import glob

    for cand in _mkl_candidates():
        if _loadable(cand):
            os.environ["DEVSIM_MATH_LIBS"] = cand
            return cand

    for bare in ("libopenblas.so", "liblapack.so", "libblas.so"):
        if ctypes.util.find_library(bare[3:-3]) and _loadable(bare):
            return None  # the default search string already works

    patterns = [
        "/usr/lib/x86_64-linux-gnu/libopenblas.so.*",
        "/usr/lib64/libopenblas.so.*",
        "/usr/lib/x86_64-linux-gnu/liblapack.so.*",
        "/usr/lib64/liblapack.so.*",
        str(Path(sys.prefix) / "lib" / "libmkl_rt.so*"),
    ]
    found = [m for pat in patterns for m in sorted(glob(pat))]
    if not found:
        return None
    os.environ["DEVSIM_MATH_LIBS"] = ":".join(found[:3])
    return os.environ["DEVSIM_MATH_LIBS"]


def _loadable(name: str) -> bool:
    import ctypes

    try:
        ctypes.CDLL(name)
        return True
    except OSError:
        return False


def ensure_math_libs() -> str | None:
    """Set DEVSIM_MATH_LIBS if unset. Returns the value in effect, or None."""
    global _done
    if _done:
        return os.environ.get("DEVSIM_MATH_LIBS")
    _done = True

    if os.environ.get("DEVSIM_MATH_LIBS"):
        return os.environ["DEVSIM_MATH_LIBS"]

    if sys.platform != "win32":
        return _ensure_linux_blas()

    for d in _candidate_dll_dirs():
        matches = sorted(d.glob("mkl_rt*.dll"), reverse=True)
        if not matches:
            continue
        os.add_dll_directory(str(d))
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        os.environ["DEVSIM_MATH_LIBS"] = matches[0].name
        return matches[0].name
    return None


_VALID_SOLVERS = ("mkl_pardiso", "superlu", "custom")


def ensure_direct_solver(devsim) -> str:
    """Make sure `direct_solver` names a solver that actually exists here.

    devsim picks its direct solver from the math library it managed to load. Load
    MKL and it selects `mkl_pardiso`; load a plain OpenBLAS -- which is all a
    Linux CI runner has -- and it selects nothing, leaving the parameter at
    `"unknown"`. Nothing complains at import. The failure arrives later, from
    inside the first `solve()`:

        DEVSIM FATAL: Unrecognized "direct_solver" parameter value "unknown".
        Valid options are "mkl_pardiso", "superlu" or "custom".

    So the whole oracle imports cleanly, builds its mesh, sets every parameter,
    and dies on the first Newton step -- on Linux only, which is why this
    survived every local run and surfaced the moment CI ran it.

    SuperLU is built into the devsim wheel and needs no external library, so it
    is always a valid fallback. MKL is left alone where it is available: it is
    substantially faster, and switching solvers changes the last digits of a
    converged solve, which would move every cached result.
    """
    try:
        current = devsim.get_parameter(name="direct_solver")
    except Exception:  # parameter not defined at all on some builds
        current = None
    if current in _VALID_SOLVERS:
        return str(current)
    devsim.set_parameter(name="direct_solver", value="superlu")
    return "superlu"


def import_devsim():
    """Import and return the devsim module, wiring up math libraries first."""
    ensure_math_libs()
    import devsim

    ensure_direct_solver(devsim)
    return devsim
