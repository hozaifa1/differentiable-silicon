#!/usr/bin/env python
"""The budget sweep: WHERE does descending through the solver start to win?

WHY THIS EXISTS. D4 ran the race at two budgets, 20 solver calls and 64, and the
ordering reversed between them: gradient descent is last-but-one at 20 and first
at 64. Two points prove a crossover exists. They do not say where it is, and
"somewhere between 20 and 64" is a weak sentence to put in a writeup when the
measurement that sharpens it is a night of compute.

So: five budgets, five arms, three seeds, one harness, one objective.

    12  20  32  48  64      solver calls

The claim this is evidence for is a CONDITIONAL one, and the condition is the
interesting half: a manufactured derivative has to be paid for before it can be
used -- 2D+1 = 9 calls for the anchor Jacobian on this four-knob problem -- so
below some budget you are better off just sampling the box. Showing only the
budget where the method wins would be the easiest thing in this project for a
judge to knock down.

HOW IT RUNS, and why it is one process per budget

Each budget is a FRESH `python scripts/race_d4.py` subprocess. That is not
tidiness. Three pieces of state in this stack are module-scoped and survive
between runs inside one interpreter:

  * `diffsilicon.shim.adjoint._REGISTRY` caches one shim per input template, so a
    second `run_flagship` inherits the first one's Jacobian AND its call counter.
    This cost D4 an hour and a wrong number on screen -- see D4_FINDINGS 6b.
    `race_d4.py` clears it, and a fresh process makes that belt-and-braces.
  * `run_flagship` counts its solver calls by reading results/runs/provenance.jsonl
    from a byte offset taken at its own start. Correct in one process; wrong the
    moment two runs append concurrently.
  * the network's training mode and the fitted reference weights are read at
    import.

which is also why this driver is SERIAL. Running budgets in parallel was
measured and is not worth it: DEVSIM already uses every core, so four concurrent
solves take 200 s each against 60 s alone -- a 1.2x throughput gain for a
correctness risk in the call accounting. Pinning threads instead (OMP/MKL = 2)
makes the LU factorisation fail outright with a divide-by-zero, so that door is
closed too. The honest cost of this measurement is its wall clock.

ORDER. Budgets 20 and 64 run FIRST even though they are already banked, because
their design points are already in the content-addressed cache and they therefore
cost minutes rather than hours. If this harness has a bug, it surfaces in the
first twenty minutes instead of the fourth hour -- and reproducing D4's two
banked numbers through the new code path is itself the check that the sweep and
the banked race are measuring the same thing.

    python scripts/race_sweep_d5.py
    python scripts/race_sweep_d5.py --budgets 32,48 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# Not sorted. See ORDER in the module docstring: the two budgets whose design
# points are already cached go first so the harness is validated cheaply.
DEFAULT_ORDER = (20, 64, 32, 48, 12)
ARMS = ("gradient", "bayes", "lhs", "random", "nelder_mead")


def _median(v):
    v = sorted(v)
    return v[len(v) // 2] if v else float("nan")


def _table(rows, budgets, arms) -> str:
    """Median best-loss per arm per budget, with the winner of each budget marked."""
    out = [f"{'arm':14s}" + "".join(f"{b:>12d}" for b in budgets)]
    out.append("-" * (14 + 12 * len(budgets)))
    med = {}
    for arm in arms:
        cells = []
        for b in budgets:
            v = [r["best"] for r in rows
                 if r["arm"] == arm and r["budget"] == b and r["best"] < 1e30]
            med[(arm, b)] = _median(v) if v else None
            cells.append(f"{med[(arm, b)]:12.6f}" if v else f"{'-':>12s}")
        out.append(f"{arm:14s}" + "".join(cells))
    out.append("")
    winners = []
    for b in budgets:
        cand = [(med[(a, b)], a) for a in arms if med.get((a, b)) is not None]
        winners.append(f"  {b:3d} calls -> {min(cand)[1]}" if cand else f"  {b:3d} calls -> -")
    out.append("best arm at each budget:")
    out.extend(winners)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default=",".join(str(b) for b in DEFAULT_ORDER))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--backend", default="devsim")
    ap.add_argument("--theta0", default="0.05,0.80,0.90,0.70")
    ap.add_argument("--out", default=str(_REPO / "results" / "runs"
                                         / "race_crossover_sweep.json"))
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    arms = args.arms.split(",")
    out_path = Path(args.out)
    part_dir = out_path.parent / "race_sweep_parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    print("BUDGET CROSSOVER SWEEP")
    print(f"  budgets  {budgets}   (run in this order; see the docstring)")
    print(f"  arms     {arms}")
    print(f"  seeds    {args.seeds}    backend {args.backend}")
    print(f"  start    {args.theta0}")
    print(f"  out      {out_path}\n", flush=True)

    rows: list[dict] = []
    t_all = time.time()

    for b in budgets:
        part = part_dir / f"budget{b}.json"
        print(f"\n{'=' * 70}\nBUDGET {b}  ({time.strftime('%H:%M:%S')})\n{'=' * 70}",
              flush=True)
        t0 = time.time()
        cmd = [
            sys.executable, str(_REPO / "scripts" / "race_d4.py"),
            "--budget", str(b), "--seeds", str(args.seeds),
            "--arms", ",".join(arms), "--backend", args.backend,
            "--theta0", args.theta0, "--tag", "sweep-",
            "--out", str(part),
        ]
        # Streamed, not captured: a run this long has to be watchable while it
        # runs, and the child already flushes every line it prints.
        rc = subprocess.call(cmd, cwd=str(_REPO))
        dt = time.time() - t0
        print(f"\nbudget {b} finished in {dt / 60:.1f} min (exit {rc})", flush=True)

        if part.is_file():
            for r in json.loads(part.read_text(encoding="utf-8")):
                rows.append({**r, "budget": b})

        # Rewritten after EVERY budget, so an interrupted sweep still leaves a
        # usable, self-describing artefact rather than nothing.
        out_path.write_text(json.dumps({
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "backend": args.backend, "theta0": args.theta0,
            "seeds": args.seeds, "arms": arms,
            "budgets_requested": budgets,
            "budgets_done": sorted({r["budget"] for r in rows}),
            "snn_train_mode": "frozen",
            "wall_seconds": round(time.time() - t_all, 1),
            "runs": rows,
        }, indent=1), encoding="utf-8")

        done = sorted({r["budget"] for r in rows})
        print("\n" + _table(rows, done, arms), flush=True)
        print(f"\nwrote {out_path}  ({len(rows)} runs so far)", flush=True)

    print(f"\n{'=' * 70}")
    print(f"SWEEP COMPLETE in {(time.time() - t_all) / 3600:.2f} h")
    print(_table(rows, sorted({r['budget'] for r in rows}), arms))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
