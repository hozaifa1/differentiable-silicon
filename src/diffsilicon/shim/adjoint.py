"""Manufacture a VJP for a solver that has no adjoint and never will.

The forward pass is NEVER a surrogate. `apply` proxies straight through to
whichever oracle is configured and returns exactly what the solver returned.
Only the DERIVATIVE is estimated, from directed finite-difference probes of that
same solver, with a trust region that forces a ground-truth refresh whenever the
local linear model stops predicting.

We make no claim that a rank-one-updated local linear model is categorically
different from a surrogate of the derivative -- it is one, and least-change
secant fitting is exactly how it is constructed. The claim is narrower and
checkable: every loss value ever reported came out of the solver at the design
point it is attributed to, and `results/cache/` plus the per-iteration
backend+hash log is the evidence.

Jacobian maintenance
--------------------
* **Anchor**: central differences, 2D+1 calls.
* **Refresh**: forward differences, D+1 calls. Central only for the anchor and
  for V2 checkpoints -- that halves refresh cost.
* **Broyden rank-1** between refreshes, from the secant pair every accepted step
  supplies for free: J <- J + (dy - J s) s^T / (s^T s).
* **Trust region** on rho_k: rho < 0.25 halves Delta and forces a refresh; a hard
  refresh happens every K steps, with K chosen from the V2 curve rather than
  asserted.
* **Hard budget cap**: the run stops at `max_oracle_calls` and reports the calls
  it used. Budget-capped optimisation is schedulable and honest;
  convergence-criterion optimisation is neither.

theta is normalised to [0,1]^D by the frozen contract, so a single scalar step
alpha is a sensible step in every coordinate at once -- which is the whole reason
the contract normalises.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ..shared.cache import content_hash
from ..shared.contract import DIFFERENTIABLE_OUTPUTS, OracleInput, make_oracle_input
from ..shared.oracle import run_oracle

__all__ = ["ShimConfig", "AdjointShim", "fd_jacobian", "y_vector"]


def y_vector(out) -> np.ndarray:
    """OracleOutput -> the 7-vector in the frozen row order."""
    get = (lambda k: getattr(out, k)) if hasattr(out, "ss") else (lambda k: out[k])
    return np.array([float(get(k)) for k in DIFFERENTIABLE_OUTPUTS], dtype=np.float64)


@dataclass
class ShimConfig:
    alpha: float = 0.02  # FD step, in normalised theta units. Set from V1 on D3.
    refresh_every: int = 4  # K. Set from the V2 cosine curve on D3, not asserted.
    trust_radius: float = 0.15
    rho_shrink: float = 0.25
    max_oracle_calls: int = 65
    backend: str | None = None
    clip_to_box: bool = True


@dataclass
class _Counter:
    calls: int = 0
    refreshes: int = 0
    broyden_updates: int = 0
    log: list = field(default_factory=list)


def _probe(theta: np.ndarray, template: OracleInput, cfg: ShimConfig, ctr: _Counter) -> np.ndarray:
    if ctr.calls >= cfg.max_oracle_calls:
        raise RuntimeError(
            f"oracle budget exhausted: {ctr.calls} calls of {cfg.max_oracle_calls}. "
            f"This is a hard cap by design -- raise max_oracle_calls deliberately."
        )
    inp = make_oracle_input(theta, template.vg_grid, template.vds_lin, template.vds_sat)
    key = content_hash(inp)
    out = run_oracle(inp, cfg.backend)
    ctr.calls += 1
    # Per-call provenance, mirroring results/runs/provenance.jsonl. Every probe the
    # shim makes is a REAL solver call at a REAL design point, and this says so.
    ctr.log.append(
        {
            "call": ctr.calls,
            "backend": cfg.backend or os.environ.get("ORACLE_BACKEND", "mock"),
            "hash": key,
            "solver_seconds": float(out.solver_seconds),
            "converged": float(out.converged),
        }
    )
    return y_vector(out)


def fd_jacobian(
    theta: np.ndarray,
    template: OracleInput,
    cfg: ShimConfig,
    ctr: _Counter | None = None,
    central: bool = True,
    y0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference Jacobian, shape (7, D). Returns (J, y at theta)."""
    ctr = ctr or _Counter()
    theta = np.asarray(theta, dtype=np.float64).ravel()
    D = theta.size
    h = cfg.alpha

    if y0 is None:
        y0 = _probe(theta, template, cfg, ctr)
    J = np.zeros((len(DIFFERENTIABLE_OUTPUTS), D), dtype=np.float64)

    for i in range(D):
        e = np.zeros(D)
        e[i] = h
        if central:
            tp, tm = theta + e, theta - e
            if cfg.clip_to_box:
                tp, tm = np.clip(tp, 0.0, 1.0), np.clip(tm, 0.0, 1.0)
            denom = float(tp[i] - tm[i])
            J[:, i] = (_probe(tp, template, cfg, ctr) - _probe(tm, template, cfg, ctr)) / denom
        else:
            tp = np.clip(theta + e, 0.0, 1.0) if cfg.clip_to_box else theta + e
            denom = float(tp[i] - theta[i])
            if denom == 0.0:  # sitting exactly on the upper face: step inward instead
                tp = theta - e
                denom = float(tp[i] - theta[i])
            J[:, i] = (_probe(tp, template, cfg, ctr) - y0) / denom
    return J, y0


class AdjointShim:
    """Holds the local linear model of the oracle and keeps it honest."""

    def __init__(self, template: OracleInput, cfg: ShimConfig | None = None):
        self.template = template
        self.cfg = cfg or ShimConfig()
        self.ctr = _Counter()
        self.J: np.ndarray | None = None
        self.theta: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.steps_since_refresh = 0
        self.radius = self.cfg.trust_radius
        self.rho_history: list[float] = []

    # --- ground truth -----------------------------------------------------
    def refresh(self, theta: np.ndarray, central: bool | None = None) -> np.ndarray:
        """Rebuild J from the solver. Central for the anchor, forward thereafter."""
        if central is None:
            central = self.J is None  # anchor once, then forward differences
        self.J, self.y = fd_jacobian(theta, self.template, self.cfg, self.ctr, central=central)
        self.theta = np.asarray(theta, dtype=np.float64).ravel().copy()
        self.steps_since_refresh = 0
        self.ctr.refreshes += 1
        return self.J

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64).ravel()
        stale = (
            self.J is None
            or self.theta is None
            or self.steps_since_refresh >= self.cfg.refresh_every
            or float(np.linalg.norm(theta - self.theta)) > self.radius
        )
        if stale:
            self.refresh(theta)
        return self.J

    # --- cheap maintenance ------------------------------------------------
    def broyden_update(self, theta_new: np.ndarray, y_new: np.ndarray) -> None:
        """J <- J + (dy - J s) s^T / (s^T s), from a secant pair that cost nothing."""
        if self.J is None or self.theta is None:
            return
        s = np.asarray(theta_new, dtype=np.float64).ravel() - self.theta
        ss = float(s @ s)
        if ss <= 0.0:
            return
        dy = np.asarray(y_new, dtype=np.float64).ravel() - self.y
        self.J = self.J + np.outer(dy - self.J @ s, s) / ss
        self.theta = np.asarray(theta_new, dtype=np.float64).ravel().copy()
        self.y = np.asarray(y_new, dtype=np.float64).ravel().copy()
        self.steps_since_refresh += 1
        self.ctr.broyden_updates += 1

    def record_step(self, rho: float) -> None:
        """Trust-region bookkeeping on the actual-vs-predicted reduction ratio."""
        self.rho_history.append(float(rho))
        if rho < self.cfg.rho_shrink:
            self.radius *= 0.5
            self.steps_since_refresh = self.cfg.refresh_every  # force a refresh

    # --- the endpoint the whole project exists for ------------------------
    def vjp(self, theta: np.ndarray, cotangent_y: np.ndarray) -> np.ndarray:
        """gbar_theta = J^T gbar_y, in R^D. ZERO solver calls when J is fresh."""
        J = self.jacobian(theta)
        return J.T @ np.asarray(cotangent_y, dtype=np.float64).ravel()

    def jvp(self, theta: np.ndarray, tangent_theta: np.ndarray) -> np.ndarray:
        J = self.jacobian(theta)
        return J @ np.asarray(tangent_theta, dtype=np.float64).ravel()

    @property
    def calls(self) -> int:
        return self.ctr.calls
