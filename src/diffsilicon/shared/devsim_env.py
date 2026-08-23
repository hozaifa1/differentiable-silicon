"""Make `import devsim` work without the caller knowing anything about BLAS.

The Windows devsim wheel ships no math library and looks for
libopenblas/liblapack/libblas or an Intel MKL whose highest *tested* name is
`mkl_rt.2.dll`. Current MKL wheels install `mkl_rt.3.dll` inside
`<prefix>/Library/bin`, which devsim will happily load once it is told the name
and the directory is on the DLL search path. Linux wheels are self-contained.

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


def ensure_math_libs() -> str | None:
    """Set DEVSIM_MATH_LIBS if unset. Returns the value in effect, or None."""
    global _done
    if _done:
        return os.environ.get("DEVSIM_MATH_LIBS")
    _done = True

    if os.environ.get("DEVSIM_MATH_LIBS"):
        return os.environ["DEVSIM_MATH_LIBS"]

    if sys.platform != "win32":
        return None  # manylinux wheels bundle their own OpenBLAS

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
