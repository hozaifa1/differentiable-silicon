"""Content-addressed replay cache.

Tier C of the reproduction story: every Sentaurus call this project ever makes is
written here as a small JSON file keyed by the sha256 of its inputs, so anyone
without a license can regenerate every Sentaurus figure bit-for-bit with no
network. ~250 files x ~7 floats plus two 96-point curves is a few hundred KB, and
it is committed to the repository.

The cache populates as a SIDE EFFECT of every run. It is never reconstructed at
the end -- a cache assembled after the fact proves nothing about what the solver
actually returned during the optimisation, and the per-iteration backend+hash log
is the whole defence against "you built a derivative surrogate".
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["CONTRACT_VERSION", "content_hash", "cache_key", "CacheStore"]

# Bump ONLY if OracleInput changes meaning. Part of every key, so a bump
# invalidates the cache rather than silently mixing incompatible entries.
CONTRACT_VERSION = "1"

_DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "results" / "cache"


def _canonical(inputs) -> str:
    """Canonical JSON of the physically meaningful inputs, backend-independent.

    Floats are rounded to 12 significant digits before hashing. Without this, a
    theta that round-trips through JSON at 1 ULP difference would miss its own
    cache entry, and Tier C would silently fall back to recomputation.
    """

    def r(x):
        return float(f"{float(x):.12e}")

    get = (lambda k: getattr(inputs, k)) if hasattr(inputs, "theta") else (lambda k: inputs[k])
    payload = {
        "contract": CONTRACT_VERSION,
        "theta": [r(v) for v in np.asarray(get("theta"), dtype=np.float64).ravel()],
        "vg_grid": [r(v) for v in np.asarray(get("vg_grid"), dtype=np.float64).ravel()],
        "vds_lin": r(get("vds_lin")),
        "vds_sat": r(get("vds_sat")),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def content_hash(inputs) -> str:
    """sha256 of the canonical inputs. This is the cache key AND the provenance id."""
    return hashlib.sha256(_canonical(inputs).encode("utf-8")).hexdigest()


def _mock_source_hash() -> str:
    """sha256 over the analytic mock's own source.

    The content hash keys on INPUTS. For a real solver that is enough: the
    implementation is a binary and a deck, and it does not change under you. The
    mock is Python in this repository, and a cache keyed only on inputs would keep
    serving yesterday's physics after an edit -- silently, and most damagingly
    inside a Jacobian where half the columns are stale and half are not.
    """
    from . import mock_device

    src = Path(mock_device.__file__).read_bytes()
    return hashlib.sha256(src).hexdigest()[:16]


def cache_key(inputs, backend: str) -> str:
    """The cache filename. Equals content_hash() except for the mock, which is
    additionally keyed on its own source so that editing it invalidates entries."""
    h = content_hash(inputs)
    if backend == "mock":
        return hashlib.sha256(f"{h}:{_mock_source_hash()}".encode()).hexdigest()
    return h


class CacheStore:
    """One directory per backend, one JSON file per evaluation."""

    def __init__(self, backend: str, root: str | Path | None = None):
        self.backend = backend
        env_root = os.environ.get("DIFFSILICON_CACHE_ROOT")
        self.root = Path(root or env_root or _DEFAULT_ROOT) / backend
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def path_for(self, key: str) -> Path:
        # Two-level fan-out keeps any one directory under a few hundred entries.
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self.path_for(key)
        if not p.is_file():
            self.misses += 1
            return None
        self.hits += 1
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    def put(self, key: str, record: dict[str, Any]) -> Path:
        p = self.path_for(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so an interrupted run never leaves a half-written
        # entry that a later replay would read as truth.
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record, fh, sort_keys=True, separators=(",", ":"))
        tmp.replace(p)
        self.writes += 1
        return p

    def __len__(self) -> int:
        return sum(1 for _ in self.root.glob("*/*.json")) if self.root.is_dir() else 0


def encode_output(out, backend: str, content_hash: str) -> dict[str, Any]:
    """OracleOutput -> a JSON-safe record, stamped with its provenance.

    backend and content_hash are passed in rather than read off the output because
    they are not schema fields: a Tesseract output leaf must be an array to cross
    into JAX. The cache record is where they live, and it is what Tier C replays.
    """
    get = (lambda k: getattr(out, k)) if hasattr(out, "ss") else (lambda k: out[k])
    rec = {k: float(get(k)) for k in
           ("ss", "vth_fwd", "vth_rev", "i_leak", "g_lo", "g_hi", "dg_dvth",
            "solver_seconds", "converged")}
    rec["id_vg"] = np.asarray(get("id_vg"), dtype=np.float64).tolist()
    rec["backend"] = str(backend)
    rec["content_hash"] = str(content_hash)
    return rec
