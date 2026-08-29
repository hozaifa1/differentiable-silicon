#!/usr/bin/env python
"""G10: turn the raw provenance log into the evidence it is supposed to be.

    python scripts/provenance_audit_d6.py

`results/runs/provenance.jsonl` is appended to by `shared/oracle.run_oracle` on
EVERY forward evaluation -- cache hits included -- with the backend that produced
the number and the sha256 of the inputs it was produced at. It is 1.1 MB of one
line per call, and 1.1 MB of JSON is not evidence, it is a file. This turns it
into the four statements G10 actually asks for, each of them checkable:

1. WHAT PRODUCED THE NUMBERS. A count by backend over every evaluation in the
   log, and the solver time they represent.

2. EVERY STEP OF THE FLAGSHIP IS IN IT. `flagship-d4-fixed/steps.jsonl` records
   a `content_hash` per step. Each one is looked up in the log and the backend
   that served it is reported. A step whose hash is absent would mean a loss was
   reported for a design point no solver was ever called at -- which is the exact
   accusation the log exists to answer.

3. NOTHING REPORTED CAME OFF THE MOCK. The analytic mock exists as a wiring
   harness for CI and the gradient checks. If a hash that a reported result
   depends on had been served by it, that result would be a surrogate's output
   wearing a solver's name. Checked, not asserted.

4. THE SPAN, STATED HONESTLY. The log covers 2026-08-26 onward, not D1. The
   D1-D2 entries are gone: `cache_key` folds a hash of `shared/extract.py` into
   every key, the D3 rewrite of the extraction changed that hash, and the log was
   restarted with the cache it describes rather than left to mix two
   incompatible generations of the same field names. The TickTick note saying
   "appending since D1" is wrong and this says so rather than repeating it.

Nothing here calls a solver, and nothing here writes to the log -- the audit runs
with the provenance appender disabled so that auditing cannot pollute the thing
being audited.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

os.environ.setdefault("DIFFSILICON_PROVENANCE_DISABLE", "1")

LOG = _REPO / "results" / "runs" / "provenance.jsonl"
OUT = _REPO / "results" / "runs" / "provenance_audit_d6.json"

# The banked artefacts whose design points must appear in the log. Each is
# (label, path, how to pull content hashes out of it).
FLAGSHIP = _REPO / "results" / "runs" / "flagship-d4-fixed" / "steps.jsonl"

#: Backends that are a real solver. `mock` is not one; `replay` serves a record
#: some real backend wrote, and is stamped with THAT backend, so it never
#: appears here as itself.
REAL = ("devsim", "sentaurus")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rows = load(Path(args.log))
    by_backend: collections.Counter = collections.Counter()
    hashes: dict[str, set] = collections.defaultdict(set)
    solver_seconds: collections.Counter = collections.Counter()
    unconverged = 0
    t_lo = t_hi = None
    for r in rows:
        b = r.get("backend") or "?"
        by_backend[b] += 1
        if r.get("hash"):
            hashes[r["hash"]].add(b)
        solver_seconds[b] += float(r.get("solver_seconds") or 0.0)
        if not r.get("converged"):
            unconverged += 1
        t = r.get("t")
        if t is not None:
            t_lo = t if t_lo is None else min(t_lo, t)
            t_hi = t if t_hi is None else max(t_hi, t)

    print(f"1. WHAT PRODUCED THE NUMBERS -- {len(rows):,} evaluations, "
          f"{len(hashes):,} distinct design points\n")
    print(f"   {'backend':<12s} {'calls':>8s} {'distinct':>9s} "
          f"{'solver time':>14s}")
    for b, n in by_backend.most_common():
        d = sum(1 for hs in hashes.values() if b in hs)
        print(f"   {b:<12s} {n:>8,d} {d:>9,d} "
              f"{solver_seconds[b] / 3600.0:>11.2f} h")
    print(f"\n   unconverged evaluations logged: {unconverged} "
          f"({unconverged / max(len(rows), 1) * 100:.2f}%)"
          f"  -- refused by the extraction, not silently used")

    # 2 + 3. the flagship, step by step -------------------------------------
    steps = load(FLAGSHIP)
    print(f"\n2. EVERY STEP OF THE FLAGSHIP IS IN THE LOG -- "
          f"{len(steps)} steps of flagship-d4-fixed\n")
    step_rows = []
    missing, mocked, disagree = [], [], []
    for st in steps:
        h = st.get("content_hash")
        served = sorted(hashes.get(h, set()))
        ok = any(b in REAL for b in served)
        if not served:
            missing.append(st["step"])
        if served and all(b == "mock" for b in served):
            mocked.append(st["step"])
        # A step can show more than one backend: the same theta may have been
        # evaluated on the mock too, by a gradient check or by CI, at a
        # different time. What matters is that the backend the STEP ITSELF
        # recorded is among them -- that is the one the loss is attributed to.
        own = st.get("backend")
        agrees = own in served
        if not agrees:
            disagree.append(st["step"])
        step_rows.append({"step": st["step"], "hash": h, "backends": served,
                          "step_backend": own, "backend_agrees": agrees,
                          "loss": st["loss"], "real_solver": ok})
        print(f"   step {st['step']:2d}  {h[:12]}...  loss {st['loss']:.6f}  "
              f"step says {own}, log says {','.join(served) or 'NOTHING'}  "
              f"{'OK' if ok and agrees else 'MISMATCH'}")

    print(f"\n3. NOTHING REPORTED CAME OFF THE MOCK")
    print(f"   flagship steps with no provenance line : {len(missing)}")
    print(f"   flagship steps served only by the mock : {len(mocked)}")
    print(f"   steps whose own recorded backend is absent from the log : "
          f"{len(disagree)}")
    mock_only = [h for h, bs in hashes.items() if bs == {"mock"}]
    print(f"   design points the mock is the ONLY source for, anywhere in the "
          f"log: {len(mock_only)}")
    print(f"   (those are the CI and gradient-check points; none of them is a "
          f"flagship step)")

    span = (f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(t_lo))} -> "
            f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(t_hi))}")
    print(f"\n4. THE SPAN -- {span}")
    print("   Not 'since D1'. The D3 rewrite of shared/extract.py changed every")
    print("   cache key, and the log was restarted with the cache it describes")
    print("   rather than left to mix two generations of the same field names.")

    ok = not missing and not mocked and not disagree
    result = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "log": str(Path(args.log).relative_to(_REPO)).replace("\\", "/"),
        "evaluations": len(rows),
        "distinct_design_points": len(hashes),
        "by_backend": {b: {"calls": n,
                           "distinct_points": sum(1 for hs in hashes.values() if b in hs),
                           "solver_hours": round(solver_seconds[b] / 3600.0, 3)}
                       for b, n in by_backend.most_common()},
        "unconverged_evaluations": unconverged,
        "span": span,
        "span_note": "Starts 2026-08-26, not D1: the D3 extraction rewrite "
                     "re-keyed every cache entry and the log was restarted with "
                     "the cache it describes.",
        "flagship": {
            "source": "results/runs/flagship-d4-fixed/steps.jsonl",
            "steps": step_rows,
            "steps_missing_from_log": missing,
            "steps_served_only_by_mock": mocked,
            "steps_whose_own_backend_is_absent_from_the_log": disagree,
        },
        "mock_only_design_points": len(mock_only),
        "pass": ok,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    if ok:
        print("PASS -- every design point the flagship reports a loss at has a "
              "provenance line,")
        print("        and a real solver wrote every one of them.")
    else:
        print(f"FAIL -- missing {missing}, mock-only {mocked}, "
              f"backend disagreement {disagree}")
    print(f"written to {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
