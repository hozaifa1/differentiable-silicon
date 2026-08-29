#!/usr/bin/env python
"""results/manifest.json: a sha256 for every figure and every measurement artefact.

    python scripts/make_manifest.py          # write it
    python scripts/make_manifest.py --check  # verify nothing drifted

WHY. The writeup and the README quote numbers that live in `results/runs/*.json`
and figures that live in `docs/figures/`. A reader who wants to know whether the
figure they are looking at is the one the text describes has, without this, no
way to tell but to re-run everything. This is the cheap answer: one hash per
file, and a `--check` mode that says which ones moved.

It also records, for each figure, WHICH script draws it and WHICH banked run it
was drawn from, so the chain from a number in the writeup back to the solver call
that produced it is one file lookup rather than an archaeology exercise.

`--check` exits non-zero on any mismatch, so it can be a CI step or the last
thing run before a submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
OUT = _REPO / "results" / "manifest.json"

#: figure stem -> (script that draws it, the banked run it is drawn from)
FIGURE_SOURCE = {
    "fig1_pca_manifold": ("scripts/fig1_pca_manifold.py",
                          "results/runs/manifold_cloud_d4.json"),
    "fig2_budget_crossover": ("scripts/fig2_budget_crossover.py",
                              "results/runs/race_crossover_sweep.json"),
    "fig3_hysteresis_descent": ("scripts/fig3_hysteresis_descent.py",
                                "results/runs/fig3_hysteresis_descent.json"),
    "fig4_spike_raster": ("scripts/fig4_spike_raster.py",
                          "results/runs/fig4_spike_raster.json"),
    "fig5_jacobian_and_decay": ("scripts/fig5_jacobian_and_decay.py",
                                "results/runs/fig5_jacobian_and_decay.json"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(_REPO)).replace("\\", "/")


def collect() -> dict:
    figures, runs = {}, {}

    for p in sorted((_REPO / "docs" / "figures").glob("*")):
        if p.suffix.lower() not in (".png", ".pdf"):
            continue
        entry = {"sha256": sha256(p), "bytes": p.stat().st_size}
        script, source = FIGURE_SOURCE.get(p.stem, (None, None))
        if script:
            entry["drawn_by"] = script
            entry["drawn_from"] = source
        figures[rel(p)] = entry

    for p in sorted((_REPO / "results" / "runs").glob("*.json")):
        runs[rel(p)] = {"sha256": sha256(p), "bytes": p.stat().st_size}

    return {"figures": figures, "runs": runs}


def head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - a manifest without a commit is still useful
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify against the committed manifest; exit 1 on drift")
    args = ap.parse_args()

    now = collect()

    if args.check:
        if not OUT.is_file():
            print(f"{rel(OUT)} does not exist; run without --check first.")
            return 1
        was = json.loads(OUT.read_text(encoding="utf-8"))
        bad = []
        for section in ("figures", "runs"):
            old, new = was.get(section, {}), now[section]
            for k in sorted(set(old) | set(new)):
                if k not in old:
                    bad.append(("added", k))
                elif k not in new:
                    bad.append(("MISSING", k))
                elif old[k]["sha256"] != new[k]["sha256"]:
                    bad.append(("CHANGED", k))
        n = len(now["figures"]) + len(now["runs"])
        for kind, k in bad:
            print(f"  {kind:>8s}  {k}")
        print(f"\n{n - len([b for b in bad if b[0] != 'added'])} of {n} files match "
              f"the manifest recorded at {was.get('commit', '?')[:8]}")
        if bad:
            print("Drift is not automatically wrong -- a regenerated figure is "
                  "expected to change.\nRe-run without --check to re-record.")
        return 1 if bad else 0

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit": head(),
        "what": "sha256 of every published figure and every banked measurement "
                "artefact, so a reader can tell whether the figure they are "
                "looking at is the one the writeup describes.",
        "verify": "python scripts/make_manifest.py --check",
        **now,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{len(now['figures'])} figures, {len(now['runs'])} run artefacts")
    print(f"written to {rel(OUT)} at commit {payload['commit'][:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
