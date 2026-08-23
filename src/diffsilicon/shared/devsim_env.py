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

__all__ = ["ensure_math_libs", "import_devsim"]

_done = False


def _candidate_dll_dirs() -> list[Path]:
    dirs = [Path(sys.prefix) / "Library" / "bin"]
    # uv puts the venv beside the project; also try the base interpreter.
    base = getattr(sys, "base_prefix", sys.prefix)
    dirs.append(Path(base) / "Library" / "bin")
    return [d for d in dirs if d.is_dir()]


def _ensure_linux_blas() -> str | None:
    """Find a BLAS/LAPACK devsim can load, preferring the bare names it expects."""
    import ctypes.util
    from glob import glob

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


def import_devsim():
    """Import and return the devsim module, wiring up math libraries first."""
    ensure_math_libs()
    import devsim

    return devsim
