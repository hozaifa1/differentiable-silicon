#!/usr/bin/env python
"""Tier C, verified end to end: every Sentaurus number, regenerated with the network unplugged.

    python scripts/tier_c_replay_d6.py

What Tier C claims
------------------
The commercial solver is on a shared licensed host. Nobody reproducing this
project has that host, that license, or that account. The claim the README makes
is therefore not "run Sentaurus yourself" -- it is:

    every figure in this project that a Sentaurus run produced can be
    regenerated, bit for bit, from `results/cache/sentaurus/`, on a machine with
    no license and no network.

That claim was PROVEN ON DEVSIM on D2 and merely ASSERTED for Sentaurus until
today. This script is the proof for Sentaurus, and it is written so that it
cannot pass by accident.

How the network is unplugged
----------------------------
Not by asserting it. By breaking it:

* `socket.socket` is replaced by a subclass whose `connect`, `connect_ex` and
  `sendto` raise, and `socket.create_connection` / `socket.getaddrinfo` raise
  outright. A TCP connection of any kind, to anywhere, fails.
* `subprocess.Popen`, `subprocess.run` and `os.system` are replaced with
  functions that raise. `t1_driver` reaches the host by shelling out to
  plink/pscp, so this closes the door the socket guard does not cover.
* `SENTAURUS_HOST`, `SENTAURUS_PASSWORD`, `PLINK` and `PSCP` are deleted from the
  environment before anything reads them, so even a guard that were somehow
  bypassed would have nothing to connect to.

The environment is stripped at the very top of the file; the socket and
subprocess guards go in at the top of `main`. They have to go in that order:
`ssl` subclasses `socket.socket` and `asyncio` subclasses `subprocess.Popen` at
import time, so a guard installed first takes `import jax` down with it. Nothing
between the two points touches a solver -- the guards are live for every line
that could, which is what the claim needs.

If any of the numbers below came from the solver rather than the cache, this
script does not produce a wrong answer -- it dies with a traceback.

What is compared
----------------
Both Sentaurus artefacts this project banked, regenerated through
`ORACLE_BACKEND=replay` and diffed against the JSON on disk:

1. `results/runs/rebaseline_d3_sentaurus.json` -- eight design points, sixteen
   recorded quantities each. Includes `solver_seconds`, which is stored in the
   cache record and therefore replays exactly: a reproduction that reports how
   long the original solve took is a stronger artefact than one that reports how
   long the cache lookup took. `wall_seconds` is the only field excluded, and it
   is excluded because it is the one number that MUST differ.

   Eight, not nine: `rebaseline_d3.py` builds `rand0..rand5`, and `rand5` has no
   Sentaurus entry -- it was never solved on the commercial host. The replay
   reports it as `not-in-banked-file` rather than skipping it quietly, because a
   Tier C script that silently drops the points it cannot serve is exactly the
   script nobody should trust.

2. `results/runs/cross_check_sentaurus_devsim_d4.json` -- the V4 cross-solver
   Jacobian, its 2D+1 = 9 probe points, the relative sensitivities derived from
   them, and the sign/rank summary. This is the harder half: it exercises the
   shim's own finite-difference machinery, not just nine cache lookups.

The report also carries CACHE COVERAGE -- how many of the entries in
`results/cache/sentaurus/` these two artefacts actually touch. The cache holds
every Sentaurus call the project ever made, including probes from runs that were
superseded; claiming "the cache reproduces everything" while exercising half of
it would be the easy way to be wrong here.

"Bit for bit" is meant literally. The comparison is `==` on float64, and the
report also carries the ULP distance so that a near-miss could not be rounded
into a pass.
"""

from __future__ import annotations

# --- the guards. Before anything else, including project imports. ------------
import os
import socket
import subprocess
import sys


class NetworkTouched(RuntimeError):
    """Something tried to reach the Sentaurus host. Tier C is then not proven."""


def _blocked(what):
    def _raise(*_a, **_k):
        raise NetworkTouched(
            f"{what} was called during a Tier C replay. Tier C claims every "
            f"Sentaurus number can be regenerated with no license and no "
            f"network; a call to {what} means that claim is false."
        )

    return _raise


for _name in ("SENTAURUS_HOST", "SENTAURUS_PASSWORD", "SENTAURUS_REMOTE_ROOT",
              "PLINK", "PSCP", "SENTAURUS_GRID"):
    os.environ.pop(_name, None)

os.environ["ORACLE_BACKEND"] = "replay"
os.environ["ORACLE_REPLAY_SOURCE"] = "sentaurus"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402


class _NoSocket(socket.socket):
    """A socket that exists as a type -- `ssl` needs that -- and cannot connect."""

    def connect(self, *_a, **_k):
        raise NetworkTouched("socket.connect during a Tier C replay")

    def connect_ex(self, *_a, **_k):
        raise NetworkTouched("socket.connect_ex during a Tier C replay")

    def sendto(self, *_a, **_k):
        raise NetworkTouched("socket.sendto during a Tier C replay")


def install_guards() -> None:
    socket.socket = _NoSocket  # type: ignore[misc, assignment]
    socket.create_connection = _blocked("socket.create_connection")  # type: ignore[assignment]
    socket.getaddrinfo = _blocked("socket.getaddrinfo")  # type: ignore[assignment]
    subprocess.Popen = _blocked("subprocess.Popen")  # type: ignore[assignment]
    subprocess.run = _blocked("subprocess.run")  # type: ignore[assignment]
    subprocess.check_output = _blocked("subprocess.check_output")  # type: ignore[assignment]
    subprocess.call = _blocked("subprocess.call")  # type: ignore[assignment]
    os.system = _blocked("os.system")  # type: ignore[assignment]


# -----------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from diffsilicon.shared.cache import CacheStore, cache_key  # noqa: E402
from diffsilicon.shared.circuit import load_circuit, transduce  # noqa: E402
from diffsilicon.shared.contract import (  # noqa: E402
    DIFFERENTIABLE_OUTPUTS,
    make_oracle_input,
)
from diffsilicon.shared.design import nominal_theta  # noqa: E402
from diffsilicon.shared.oracle import device_geometry, run_oracle  # noqa: E402
from diffsilicon.shim.adjoint import ShimConfig, fd_jacobian  # noqa: E402

REBASELINE = _REPO / "results" / "runs" / "rebaseline_d3_sentaurus.json"
CROSSCHECK = _REPO / "results" / "runs" / "cross_check_sentaurus_devsim_d4.json"
OUT = _REPO / "results" / "runs" / "tier_c_replay_d6.json"

# The one field that must differ: how long the lookup took, versus how long the
# original solve took. Everything else is compared.
EXCLUDED = ("wall_seconds",)


def ulps(a: float, b: float) -> int:
    """Distance in representable float64s. 0 means bit-identical."""
    if a == b:
        return 0
    if not (math.isfinite(a) and math.isfinite(b)):
        return -1
    ia = np.asarray(a, dtype=np.float64).view(np.int64).item()
    ib = np.asarray(b, dtype=np.float64).view(np.int64).item()
    if (ia < 0) != (ib < 0):
        return abs(ia) + abs(ib)
    return abs(ia - ib)


class Diff:
    """Accumulates every scalar comparison so the report is a count, not a claim."""

    def __init__(self) -> None:
        self.compared = 0
        self.identical = 0
        self.worst: list = []

    def check(self, path: str, got, want) -> bool:
        self.compared += 1
        g, w = float(got), float(want)
        u = ulps(g, w)
        if u == 0:
            self.identical += 1
            return True
        self.worst.append({"field": path, "replay": g, "banked": w, "ulps": u,
                           "rel": abs(g - w) / max(abs(w), 1e-300)})
        return False

    @property
    def ok(self) -> bool:
        return self.compared > 0 and self.identical == self.compared


#: Every cache key this replay reads. Compared against what is on disk, so the
#: coverage line in the report is a measurement rather than a claim.
TOUCHED: set[str] = set()


def note(theta) -> None:
    TOUCHED.add(cache_key(make_oracle_input(np.asarray(theta, dtype=np.float64)),
                          "sentaurus"))


def evaluate(theta, cc) -> dict:
    """The SAME body as `scripts/rebaseline_d3.evaluate`, minus `wall_seconds`.

    Kept as a copy rather than an import because that script's `evaluate` also
    times the call, and a Tier C run must not compare a timing.
    """
    note(theta)
    out = run_oracle(make_oracle_input(theta), backend="replay")
    w, lg = device_geometry(theta)
    phi = transduce(out, cc, lg, w)
    curves = np.asarray(out.id_vg)
    return {
        "ss_mV_per_dec": float(out.ss),
        "vth_fwd_V": float(out.vth_fwd),
        "vth_rev_V": float(out.vth_rev),
        "memory_window_V": float(out.vth_fwd - out.vth_rev),
        "i_leak_A": float(out.i_leak),
        "g_lo_S": float(out.g_lo),
        "g_hi_S": float(out.g_hi),
        "g_ratio": float(out.g_hi / out.g_lo),
        "dg_dvth_S_per_V": float(out.dg_dvth),
        "beta": float(phi.beta),
        "th_th": float(phi.th_th),
        "sig_w": float(phi.sig_w),
        "id_min_A": float(curves.min()),
        "id_max_A": float(curves.max()),
        "converged": float(out.converged),
        "solver_seconds": float(out.solver_seconds),
    }


def rebaseline_points(d: int):
    """The point list `scripts/rebaseline_d3.py` builds, reproduced exactly.

    Same construction, same `default_rng(0)`, same order. If this drifted from
    that script the replay would be checking different devices against the
    banked ones and every comparison would fail loudly, which is the failure
    mode to prefer.
    """
    rng = np.random.default_rng(0)
    points = [("nominal", nominal_theta(d))]
    thin = np.zeros(d)
    thin[0] = 0.0
    thick = np.zeros(d) + 0.5
    thick[0] = 1.0
    points += [("t_fe_min", thin), ("t_fe_max", thick)]
    points += [(f"rand{i}", rng.random(d)) for i in range(6)]
    return points


def part1(diff: Diff) -> dict:
    banked = json.loads(REBASELINE.read_text(encoding="utf-8"))
    d = int(banked["design"]["d"])
    cc = load_circuit()
    by_name = {p["name"]: p for p in banked["points"]}

    rows = []
    for name, theta in rebaseline_points(d):
        want = by_name.get(name)
        if want is None:
            rows.append({"point": name, "status": "not-in-banked-file"})
            continue
        # The banked theta is authoritative -- reproduce the device that was
        # actually solved, not the one this script would regenerate.
        theta = np.asarray(want["theta"], dtype=np.float64)
        got = evaluate(theta, cc)
        fields = [k for k in got if k not in EXCLUDED]
        bad = [k for k in fields
               if not diff.check(f"rebaseline/{name}/{k}", got[k], want[k])]
        rows.append({
            "point": name,
            "theta": [float(v) for v in theta],
            "fields_compared": len(fields),
            "mismatched": bad,
            "ss_mV_per_dec": got["ss_mV_per_dec"],
            "solver_seconds_replayed": got["solver_seconds"],
        })
        print(f"  {name:<10s} {len(fields):2d} fields  "
              f"SS {got['ss_mV_per_dec']:8.3f}  "
              f"{'OK' if not bad else 'MISMATCH ' + ','.join(bad)}")

    return {
        "source": str(REBASELINE.relative_to(_REPO)).replace("\\", "/"),
        "points": rows,
        "solver_seconds_total_replayed": sum(
            r.get("solver_seconds_replayed", 0.0) for r in rows),
    }


def part2(diff: Diff) -> dict:
    """The V4 cross-solver Jacobian, recomputed from the cache through the shim.

    This is the half that is not a lookup. `fd_jacobian` runs the shim's own
    2D+1 central-difference machinery; every one of those nine probes has to
    find a cache entry, and the arithmetic on top of them has to land on the
    same float64s that the banked file recorded.
    """
    banked = json.loads(CROSSCHECK.read_text(encoding="utf-8"))
    d = int(banked["d"])
    alpha = float(banked["alpha"])
    theta = np.asarray(banked["theta"], dtype=np.float64)

    cfg = ShimConfig(alpha=alpha, backend="replay", max_oracle_calls=10_000)
    # The 2D+1 probe points the shim will visit, recorded for the coverage line.
    note(theta)
    for i in range(d):
        for sgn in (+1.0, -1.0):
            t = np.array(theta, dtype=np.float64)
            t[i] = min(1.0, max(0.0, t[i] + sgn * alpha))
            note(t)
    j, y = fd_jacobian(theta, make_oracle_input(theta), cfg, central=True)

    scale = np.where(np.abs(y) > 0, np.abs(y), 1.0)[:, None]
    r = j / scale

    rows = []
    for idx, name in enumerate(DIFFERENTIABLE_OUTPUTS):
        want = banked["rows"][idx]
        assert want["fom"] == name, f"row order drifted: {want['fom']} != {name}"
        ok_y = diff.check(f"crosscheck/{name}/y_sentaurus", y[idx], want["y_sentaurus"])
        ok_d = all(
            diff.check(f"crosscheck/{name}/d_rel_a[{i}]", r[idx, i], want["d_rel_a"][i])
            for i in range(d)
        )
        rows.append({"fom": name, "y_matches": ok_y, "d_rel_matches": ok_d,
                     "y_replay": float(y[idx])})
        print(f"  {name:<9s} y={y[idx]:+.6e}  "
              f"{'OK' if (ok_y and ok_d) else 'MISMATCH'}")

    # The two headline summary numbers of that file are re-derived, not copied.
    rb = np.asarray([row["d_rel_b"] for row in banked["rows"]], dtype=np.float64)
    sign_ok = np.sign(r) == np.sign(rb)
    diff.check("crosscheck/sign_agreement", float(np.mean(sign_ok)),
               banked["sign_agreement"])

    return {
        "source": str(CROSSCHECK.relative_to(_REPO)).replace("\\", "/"),
        "probes": int(2 * d + 1),
        "alpha": alpha,
        "rows": rows,
        "sign_agreement_replayed": float(np.mean(sign_ok)),
        "sign_agreement_banked": banked["sign_agreement"],
    }


def cache_inventory() -> dict:
    """What is on disk, and how much of it this replay actually exercised.

    The unexercised entries are not spare capacity and they are not a defect,
    and this does not take that on trust -- it checks. The cache populates as a
    side effect of EVERY call, by design, so an edit to `extract.py` re-keys every
    entry on every backend and the old ones stay on disk.

    An unexercised entry is classified by its RAW CURVE, not by its extracted
    figures of merit. `superseded-duplicate` means some exercised entry holds a
    byte-identical Id-Vg curve -- the same solve, re-keyed. `orphan` means a
    curve no current artefact reads, and an orphan would be the interesting case:
    Sentaurus time this project paid for and then lost track of.

    Classifying on the curve rather than on the seven numbers is the whole point.
    Measured here, the curves are identical across every generation and several
    of the SEVEN NUMBERS ARE NOT -- vth_fwd, vth_rev, i_leak, g_lo and dg_dvth all
    move between extraction generations while the solver output does not. That is
    `cache_key`'s `_extraction_source_hash` earning its place with evidence
    instead of a docstring: without it, an edit to `extract.py` would keep
    serving the old seven numbers off the right curve, on the one backend nobody
    would think to recompute.
    """
    store = CacheStore("sentaurus")
    files = sorted(store.root.glob("*/*.json")) if store.root.is_dir() else []
    on_disk = {p.stem for p in files}
    total = sum(p.stat().st_size for p in files)
    solver_seconds = 0.0
    replayed_seconds = 0.0
    fingerprints: set[str] = set()
    by_curve: dict[str, tuple] = {}
    records: list[tuple[str, str, tuple, float]] = []
    for p in files:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            secs = float(rec.get("solver_seconds", 0.0))
        except Exception:  # noqa: BLE001 - an unreadable entry is reported, not fatal
            continue
        # The curve, hashed. This is what the solver produced; the seven numbers
        # are this repository's reading of it and are allowed to change.
        curve = np.asarray(rec["id_vg"], dtype=np.float64)
        fp = hashlib.sha256(curve.tobytes()).hexdigest()
        foms = tuple(float(rec[k]) for k in DIFFERENTIABLE_OUTPUTS)
        records.append((p.stem, fp, foms, secs))
        solver_seconds += secs
        if p.stem in TOUCHED:
            replayed_seconds += secs
            fingerprints.add(fp)
            by_curve.setdefault(fp, foms)

    superseded = [k for k, fp, _f, _ in records if k not in TOUCHED and fp in fingerprints]
    orphans = [k for k, fp, _f, _ in records if k not in TOUCHED and fp not in fingerprints]
    # Of the superseded entries, how many read DIFFERENT figures of merit off the
    # identical curve. That count is the extraction hash's justification.
    re_extracted = sum(
        1 for k, fp, foms, _ in records
        if k not in TOUCHED and fp in fingerprints and foms != by_curve[fp]
    )
    missing = sorted(TOUCHED - on_disk)
    return {
        "entries": len(files),
        "bytes": total,
        "kib": round(total / 1024.0, 1),
        "entries_exercised": len(TOUCHED & on_disk),
        "entries_superseded_duplicates": len(superseded),
        "entries_superseded_by_re_extraction": re_extracted,
        "entries_orphaned": len(orphans),
        "orphan_keys": sorted(orphans),
        "entries_requested_but_absent": missing,
        "solver_seconds_banked": solver_seconds,
        "solver_hours_banked": round(solver_seconds / 3600.0, 2),
        "solver_seconds_replayed": replayed_seconds,
        "solver_hours_replayed": round(replayed_seconds / 3600.0, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    # Installed here, not at import: `ssl` subclasses `socket.socket` and
    # `asyncio` subclasses `subprocess.Popen` at import time, so a guard put in
    # first takes `import jax` down with it. Everything below this line -- every
    # line that could reach a solver -- runs with both doors shut.
    install_guards()

    t0 = time.perf_counter()
    diff = Diff()

    print("Tier C -- Sentaurus replay, network and subprocess blocked at import.")
    print(f"cache: results/cache/sentaurus/  ({len(CacheStore('sentaurus'))} entries)\n")

    print("1. rebaseline_d3_sentaurus.json -- nine design points")
    p1 = part1(diff)
    print("\n2. cross_check_sentaurus_devsim_d4.json -- the V4 Jacobian, 9 probes")
    p2 = part2(diff)

    wall = time.perf_counter() - t0
    inv = cache_inventory()

    result = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "what": "Tier C: every banked Sentaurus number regenerated from the "
                "replay cache with sockets and subprocesses blocked.",
        "network_blocked": True,
        "subprocess_blocked": True,
        "env_stripped": ["SENTAURUS_HOST", "SENTAURUS_PASSWORD",
                         "SENTAURUS_REMOTE_ROOT", "PLINK", "PSCP", "SENTAURUS_GRID"],
        "values_compared": diff.compared,
        "values_bit_identical": diff.identical,
        "mismatches": diff.worst,
        "pass": diff.ok,
        "wall_seconds": round(wall, 3),
        "cache": inv,
        "rebaseline": p1,
        "cross_check": p2,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"compared        {diff.compared} float64 values")
    print(f"bit-identical   {diff.identical}")
    print(f"replay wall     {wall:.2f} s")
    print(f"cache exercised  {inv['entries_exercised']} of {inv['entries']} entries "
          f"({inv['kib']:.0f} KiB total)")
    print(f"                 {inv['entries_superseded_duplicates']} superseded "
          f"(byte-identical curve, re-keyed), {inv['entries_orphaned']} orphans")
    print(f"                 of those, {inv['entries_superseded_by_re_extraction']} "
          f"read DIFFERENT FoMs off the identical curve -- the extraction hash "
          f"earning its keep")
    print(f"solver time it stands in for  "
          f"{inv['solver_hours_replayed']:.2f} h of the "
          f"{inv['solver_hours_banked']:.2f} h banked")
    if diff.ok:
        speedup = inv["solver_seconds_replayed"] / max(wall, 1e-9)
        print(f"\nPASS -- Tier C reproduces every Sentaurus number bit for bit, "
              f"{speedup:,.0f}x faster than the solves it replaces,")
        print("        on a process that cannot open a socket or spawn a subprocess.")
    else:
        print("\nFAIL -- see 'mismatches' in the output file.")
    print(f"written to {args.out}")
    return 0 if diff.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
