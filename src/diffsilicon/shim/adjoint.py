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

__all__ = ["ShimConfig", "AdjointShim", "OracleBudgetExhausted", "OracleNotConverged",
           "fd_jacobian", "y_vector", "shim_for", "shim_key"]


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
    #: Probes the oracle ran but whose answer was not a measurement. See
    #: `OracleNotConverged`. Kept so a run can report how many devices in its
    #: box the extraction could not read.
    refused: list = field(default_factory=list)
    #: Jacobian columns that fell back from a central to a one-sided difference
    #: because one neighbour was refused. See `fd_jacobian`.
    salvaged_columns: list = field(default_factory=list)


class OracleBudgetExhausted(RuntimeError):
    """The call cap was reached. NOT the same thing as a solver that failed.

    Both used to be a bare RuntimeError, and the optimiser caught the pair of
    them under one `except RuntimeError` labelled "budget". So a design point
    the solver could not converge was written into the run log as "we ran out
    of calls" -- a wrong reason, recorded confidently, for a run that had
    plenty of budget left. Give the two failures two types and that cannot
    happen.
    """


class OracleNotConverged(RuntimeError):
    """The solver RAN and returned numbers, but those numbers are not a
    measurement of this device.

    ADDED 2026-08-27 (D4). This is the third distinct failure, and it is the
    dangerous one, because unlike a crash it produces a full set of finite,
    plausible-looking figures of merit that nothing downstream can tell from a
    good one.

    `run_oracle` sets `converged = 0` when the currents are not all finite, or
    when the extraction had to read a threshold voltage from OUTSIDE the
    voltages actually swept. Measured on 2026-08-26: one device in eight
    returned vth_fwd = +4.16 V from a sweep that stops at +1.50 V, which made a
    memory window of 4.44 V out of thin air, which then made sig_w = 1.007 --
    i.e. 100% weight noise -- which the network answered with a gradient of
    7e10. A number invented past the end of the data travelled the whole chain
    and came back as a direction to walk in.

    A design point we cannot measure is a design point we cannot accept. That is
    a REJECTION, handled exactly like a solver crash: shrink the trust region
    and try nearer home. It is a distinct type only so that the run log says
    which of the two happened.
    """


def _probe(theta: np.ndarray, template: OracleInput, cfg: ShimConfig, ctr: _Counter) -> np.ndarray:
    if ctr.calls >= cfg.max_oracle_calls:
        raise OracleBudgetExhausted(
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
    # The probe is logged FIRST and rejected SECOND, deliberately: it was a real
    # solver call, it cost real time, and the provenance record has to say so
    # whether or not the answer was usable.
    if float(out.converged) <= 0.5:
        raise OracleNotConverged(
            f"the oracle did not converge at this design point (hash {key}): "
            f"either the currents were not all finite, or a threshold voltage "
            f"was read from outside the swept window. "
            f"SS={float(out.ss):.4g} mV/dec, vth_fwd={float(out.vth_fwd):.4g} V, "
            f"vth_rev={float(out.vth_rev):.4g} V, i_leak={float(out.i_leak):.4g} A."
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
    """Finite-difference Jacobian, shape (7, D). Returns (J, y at theta).

    ONE BAD PROBE MUST NOT COST THE WHOLE GRADIENT.
    ------------------------------------------------
    Added 2026-08-27 (D4), and it is what makes Directive 1 usable rather than
    merely correct.

    The measurement that forced it. At the design point where the D3 flagship's
    gradient reached 6.15e13, the CENTRE is perfectly healthy -- SS 69.2 mV/dec,
    memory window 0.570 V, threshold well inside the sweep. One of its eight
    finite-difference neighbours is not: at t_fe 5.641 -> 6.041 nm the extraction
    returns SS = -2890 mV/dec, i.e. a NEGATIVE subthreshold slope, and a
    threshold of -36.4 V on a sweep that stops at -3.50 V. Differencing that
    against a healthy neighbour over an alpha of 0.04 puts about -900 volts per
    unit theta into the threshold row, and everything downstream inherits it.
    So the famous 6e13 was never a property of the design point. It was one
    poisoned column of the Jacobian.

    Now that `_probe` refuses such an answer, the naive thing happens: the
    exception escapes, the whole gradient is lost, and the optimiser rejects a
    step it had no reason to reject. At the STARTING point it would kill the run
    outright. Nine probes, any one of which can land on a device the extraction
    cannot read, is not a good bet to take eight times a run.

    So a failed probe costs its own SIDE, not the column and not the run:

    * central, one side gone -> fall back to a one-sided difference against the
      centre, which is the same estimator the forward refresh already uses and
      is accurate to O(h) instead of O(h^2);
    * central, both sides gone, or one-sided and its only probe gone -> try the
      opposite side; if that is gone too, give up and raise, because a design
      point surrounded on both sides by devices we cannot measure is a design
      point we cannot differentiate.

    `salvaged_columns` on the counter records every column that fell back, so a
    run log says how much of its Jacobian is one-sided rather than leaving it to
    be inferred.
    """
    ctr = ctr or _Counter()
    theta = np.asarray(theta, dtype=np.float64).ravel()
    D = theta.size
    h = cfg.alpha

    if y0 is None:
        y0 = _probe(theta, template, cfg, ctr)
    J = np.zeros((len(DIFFERENTIABLE_OUTPUTS), D), dtype=np.float64)

    def _try(t):
        """The probe, or None if the oracle could not measure that device."""
        try:
            return _probe(t, template, cfg, ctr)
        except OracleNotConverged as exc:
            ctr.refused.append({"theta": [float(v) for v in t], "detail": str(exc)[:300]})
            return None

    for i in range(D):
        e = np.zeros(D)
        e[i] = h
        tp = np.clip(theta + e, 0.0, 1.0) if cfg.clip_to_box else theta + e
        tm = np.clip(theta - e, 0.0, 1.0) if cfg.clip_to_box else theta - e

        if central:
            yp, ym = _try(tp), _try(tm)
            if yp is not None and ym is not None:
                J[:, i] = (yp - ym) / float(tp[i] - tm[i])
                continue
            # One side is gone. Fall back to one-sided against the centre.
            side = ("plus", tp, yp) if yp is not None else ("minus", tm, ym)
            if side[2] is None:
                raise OracleNotConverged(
                    f"neither finite-difference neighbour of coordinate {i} could "
                    f"be measured, so this design point has no gradient in that "
                    f"direction. theta = {[float(v) for v in theta]}"
                )
            ctr.salvaged_columns.append({"coord": i, "kept": side[0]})
            J[:, i] = (side[2] - y0) / float(side[1][i] - theta[i])
        else:
            # Forward refresh. Sitting exactly on the upper face makes the
            # forward step a no-op, so step inward instead -- that case predates
            # D4 and is unrelated to convergence.
            first, second = (tp, tm) if float(tp[i] - theta[i]) != 0.0 else (tm, tp)
            y_first = _try(first)
            if y_first is not None:
                J[:, i] = (y_first - y0) / float(first[i] - theta[i])
                continue
            y_second = _try(second)
            if y_second is None or float(second[i] - theta[i]) == 0.0:
                raise OracleNotConverged(
                    f"neither finite-difference neighbour of coordinate {i} could "
                    f"be measured, so this design point has no gradient in that "
                    f"direction. theta = {[float(v) for v in theta]}"
                )
            ctr.salvaged_columns.append({"coord": i, "kept": "flipped"})
            J[:, i] = (y_second - y0) / float(second[i] - theta[i])
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

    # --- self-driving bookkeeping ----------------------------------------
    def observe(self, theta: np.ndarray, y_new: np.ndarray) -> None:
        """Take the free secant pair every forward evaluation hands over.

        `apply` proxies to the oracle, so the shim sees a true (theta, y) pair on
        every forward pass at no extra cost. Feeding it back here is what makes
        the local model self-maintaining: without it `steps_since_refresh` never
        moves, K never fires, and J silently stays pinned to wherever it was
        anchored while the optimiser walks away from it.
        """
        theta = np.asarray(theta, dtype=np.float64).ravel()
        if self.J is None or self.theta is None:
            return
        if float(np.linalg.norm(theta - self.theta)) <= 1e-12:
            return
        self.broyden_update(theta, np.asarray(y_new, dtype=np.float64).ravel())

    @property
    def calls(self) -> int:
        return self.ctr.calls


# --- process-wide registry ------------------------------------------------
# The shim is stateful on purpose: J, the trust radius and the
# steps-since-refresh counter have to survive between endpoint calls, or every
# VJP would cost a full refresh and the apparatus would be pointless.
#
# It lives HERE rather than inside tesseract_api.py so that an in-process
# orchestrator and the Tesseract endpoints share one object. When T3 is served in
# a container the orchestrator has no handle and the shim runs purely on its own
# staleness rule (K, and the trust radius); locally the orchestrator can also feed
# it the measured rho, which is strictly more information.
_REGISTRY: dict[tuple, AdjointShim] = {}


def shim_key(inputs: OracleInput) -> tuple:
    """A different sweep grid is a different function, so it gets its own shim."""
    return (
        int(np.asarray(inputs.theta).shape[-1]),
        float(inputs.vds_lin),
        float(inputs.vds_sat),
        np.asarray(inputs.vg_grid, dtype=np.float64).tobytes(),
    )


def shim_for(inputs: OracleInput, cfg: ShimConfig | None = None) -> AdjointShim:
    key = shim_key(inputs)
    if key not in _REGISTRY:
        _REGISTRY[key] = AdjointShim(inputs, cfg or ShimConfig(
            alpha=float(os.environ.get("SHIM_ALPHA", 0.02)),
            refresh_every=int(os.environ.get("SHIM_REFRESH_EVERY", 4)),
            trust_radius=float(os.environ.get("SHIM_TRUST_RADIUS", 0.15)),
            max_oracle_calls=int(os.environ.get("SHIM_MAX_ORACLE_CALLS", 65)),
        ))
    return _REGISTRY[key]
