#!/usr/bin/env python
"""Launch a flagship optimisation run.

    # the flagship, on the Apache-2.0 solver
    python scripts/run_flagship.py --backend devsim --d 4 --max-oracle-calls 65 \
        --tag mini-flagship-devsim

    # the same run on the commercial solver, once a licence is reachable
    python scripts/run_flagship.py --backend sentaurus --d 4 --max-oracle-calls 65 \
        --tag mini-flagship-sentaurus

Nothing but --backend changes between those two lines, and that is the claim the
whole repository is built to make true.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diffsilicon.optimise import FlagshipConfig, run_flagship  # noqa: E402
from diffsilicon.shared.design import DEFAULT_D, DESIGN_VECTORS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="devsim", choices=["mock", "devsim", "sentaurus", "replay", "url"])
    # Choices come from the registry, not a literal, so adding a design vector
    # does not need this line edited. DEFAULT_D is 4 -- the fabrication knobs --
    # and d=3/5/12 will be REFUSED by run_flagship because they expose the locked
    # material constants. See docs/D3_RECALIBRATION.md.
    ap.add_argument("--d", type=int, default=DEFAULT_D, choices=sorted(DESIGN_VECTORS))
    ap.add_argument("--max-oracle-calls", type=int, default=45)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--refresh-every", type=int, default=4)
    ap.add_argument("--trust-radius", type=float, default=0.08)
    ap.add_argument("--lambda-e", type=float, default=1.0e6)
    ap.add_argument("--lambda-r", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    # 16, not D2's 32: a beat is 111 pooled timesteps against the synthetic
    # task's 24, and the flagship pays one inner training run per design point.
    # Batches are class-balanced by construction (see snn.ecg.ecg_batch), so 16
    # is four beats of every class rather than a lottery over a 50% N prior.
    ap.add_argument("--batch", type=int, default=16)
    # Defaults to frozen, which is what every banked result was produced under
    # and what makes the VJP exact. See FlagshipConfig.train_mode.
    ap.add_argument("--train-mode", default="frozen",
                    choices=["frozen", "adapt", "scratch"])
    ap.add_argument("--tag", default="mini-flagship")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--theta0", default="", help="comma-separated normalised start point")
    a = ap.parse_args()

    cfg = FlagshipConfig(
        d=a.d, backend=a.backend, max_oracle_calls=a.max_oracle_calls,
        max_steps=a.max_steps, alpha=a.alpha, refresh_every=a.refresh_every,
        trust_radius=a.trust_radius, lambda_e=a.lambda_e, lambda_r=a.lambda_r,
        seed=a.seed, batch=a.batch, train_mode=a.train_mode, tag=a.tag,
        out_dir=a.out_dir, theta0=a.theta0,
    )
    res = run_flagship(cfg)

    print(f"\n=== {cfg.tag} ({cfg.backend}, d={cfg.d}, "
          f"train_mode={cfg.train_mode}) ===")
    print(f"oracle calls      {res['oracle_calls']} of {cfg.max_oracle_calls}")
    print(f"steps             {res['steps']}  accepted {res['accepted']}  rejected {res['rejected']}")
    print(f"objective         {res['objective_initial']:.6f} -> {res['objective_final']:.6f} "
          f"(delta {res['objective_delta']:+.6f})")
    print(f"balanced CE       {res['ce_initial']:.6f} -> {res['ce_final']:.6f} "
          f"(delta {res['ce_delta']:+.6f})   [G6]")
    print(f"accuracy          {res['accuracy_initial']:.4f} -> {res['accuracy_final']:.4f}")
    print(f"J refreshes       {res['jacobian_refreshes']}  broyden {res['broyden_updates']}")
    print(f"theta0 (phys)     {res['theta_initial_phys']}")
    print(f"theta (phys)      {res['theta_final_phys']}")
    print(f"wall clock        {res['wall_seconds'] / 60.0:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
