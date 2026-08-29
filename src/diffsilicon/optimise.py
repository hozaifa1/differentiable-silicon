"""The flagship loop: trust-region descent on J(theta) through the whole stack.

Every loss value in the log came out of the configured solver at the design point
it is attributed to. Every step direction came out of `jax.grad` over the
composition PyTorch -> JAX -> manufactured adjoint. Nothing here models the
device; it only decides where to evaluate it next.

Budget, not convergence
-----------------------
The run stops at `max_oracle_calls` and reports the calls it used. A budget-capped
optimisation is schedulable -- 34 calls at the measured 306 s/run is 2.9 hours and
you can start it before bed -- and it is honest about the fact that a
finite-difference Jacobian over a commercial solver is bought by the call. A
convergence criterion would be neither.

Trust region
------------
Steepest descent under the manufactured gradient, with the step length as the
trust radius:

    s_k    = -Delta_k * g_k / ||g_k||
    rho_k  = (J(theta_k) - J(theta_k + s_k)) / (Delta_k * ||g_k||)

rho_k < 0.25 halves Delta and forces the shim to refresh its Jacobian from the
solver; rho_k > 0.75 on an accepted step grows it. Every accepted step also hands
the shim a free secant pair for its Broyden update, which is the only reason the
per-step cost is not D+1 solver calls.

The rho histogram and the accept/reject log this writes ARE validation item V5.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from tesseract_jax import apply_tesseract  # noqa: E402

from .pipeline import box_project, composed_loss, oracle_call, transduce_jax  # noqa: E402
from .shared.cache import content_hash  # noqa: E402
from .shared.circuit import load_circuit  # noqa: E402
from .shared.contract import make_oracle_input  # noqa: E402
from .shared.design import (  # noqa: E402
    denormalise,
    get_design,
    nominal_theta,
    tunes_locked_material,
)
from .shared.material import EC_MV_CM, PR_UC_CM2  # noqa: E402
from .shim.adjoint import (  # noqa: E402
    OracleBudgetExhausted,
    OracleNotConverged,
    shim_for,
)
from .snn.lif import PHI_KEYS  # noqa: E402

__all__ = ["FlagshipConfig", "run_flagship"]

_REPO = Path(__file__).resolve().parents[2]


@dataclass
class FlagshipConfig:
    """Everything that decides what the run is. Serialised verbatim into the log."""

    d: int = 4  # D3 recalibration: the four fabrication knobs. See shared.design.
    backend: str = "devsim"
    max_oracle_calls: int = 45
    max_steps: int = 40
    alpha: float = 0.02  # FD step in normalised theta; V1 revises it on D3
    refresh_every: int = 4  # K; the V2 cosine curve revises it on D3
    trust_radius: float = 0.08
    # FLOOR, not a stopping criterion. The objective is a spiking network's loss:
    # it is smooth in theta only until a layer-1 spike flips, and then it steps.
    # A step shorter than the distance to the next flip produces EXACTLY zero
    # change, rho = 0, and a rejection -- so an unfloored trust region halves
    # itself to nothing in five steps and the run ends having spent six of its
    # forty-five solver calls. Below this radius the run stops shrinking and
    # forces the shim to refresh J from the solver instead.
    min_radius: float = 0.012
    max_radius: float = 0.40
    # A SAFETY NET ON THE GRADIENT'S MAGNITUDE. It should not bind.
    #
    # History, because the value has moved twice and the reason matters more
    # than the number. This started as a workaround: the gradient came back at
    # 6.1e13 against a loss of order 1, so `predicted = radius * |g|` was
    # meaningless, every rho was zero, and the "rho < 0.25 forces a refresh"
    # rule spent the whole budget rebuilding Jacobians. Bounding |g| made rho
    # mean something again without touching the step DIRECTION, which is
    # normalised anyway.
    #
    # It was a workaround for a real bug, and the bug is now fixed. The
    # network's dL/dphi was being taken from autograd through 111 recurrent
    # timesteps, and it did not agree with the network's own loss -- it pointed
    # BACKWARDS. Measured at the flagship's starting corner, analytic against a
    # central finite difference of the composed loss:
    #
    #     before   |g| = 9.25e5   cosine -0.975   (uphill)
    #     after    |g| = 1.11     cosine +0.954   (downhill, right size)
    #
    # See `tesseracts/snn-lif-ecg/tesseract_api.py` under VJP_MODE.
    #
    # So the gradient's magnitude is now the truth -- the finite difference at
    # that corner is 1.53 against the analytic 1.11 -- and the cap is set well
    # above it, at 10. It exists only to stop a pathological point from taking
    # a run with it, and on a healthy run it never binds. If a run log shows
    # `grad_norm_used` pinned at the cap, that is a signal to investigate, not
    # a knob to raise.
    grad_norm_cap: float = 10.0
    lambda_e: float = 1.0e6
    lambda_r: float = 0.0
    seed: int = 0  # FIXED across steps: a resampled batch would make the
    batch: int = 16  # loss noisy and every rho meaningless.
    # 16, not D2's 32: an ECG beat is 111 pooled timesteps against the synthetic
    # task's 24, and the flagship pays one inner training run per design point.
    # Batches are class-balanced by construction (snn.ecg.ecg_batch), so 16 is
    # four beats of every class, not a lottery over a 50% N prior.
    # THE INNER TRAINING MODE, PINNED HERE RATHER THAN LEFT TO THE SHELL.
    #
    # FOUND 2026-08-29 (D6), while checking that the spike-raster figure
    # reproduces the flagship. It did not, and the reason was this: every banked
    # result in this project -- the flagship, the race, the budget sweep, V6, V7,
    # the correlations -- was produced under SNN_TRAIN_MODE=frozen, which each of
    # those drivers sets for itself. `tesseract_api` defaults to `adapt`, and
    # this loop set neither. So `python scripts/run_flagship.py --backend devsim`,
    # the command the README gives, ran in `adapt` and returned 1.3152 where the
    # reported run returned 1.3996. Nothing was wrong with either number; they
    # were answers to different questions, and nothing on disk said which was
    # which -- `result.json` did not record the mode either.
    #
    # `frozen` is also the right default on the merits, not merely the one that
    # matches the bank: it is what the thesis does (train in software, deploy
    # onto measured FeFET levels), and it makes the VJP EXACT rather than
    # approximate, because the envelope-theorem argument for holding W fixed
    # needs a stationary W* and a W that never moves is trivially stationary.
    # See `tesseracts/snn-lif-ecg/tesseract_api.py` under TRAIN_MODE.
    #
    # It is serialised into result.json with everything else, so a run now says
    # what it was.
    train_mode: str = "frozen"
    tag: str = "mini-flagship"
    out_dir: str = ""
    # Where the run STARTS, normalised. Empty means the nominal device.
    #
    # Starting at nominal is a bad experiment: the nominal FeFET is already a
    # good device for this task, the optimiser has nowhere to go, and a run that
    # accepts zero steps proves nothing either way. Starting from a deliberately
    # poor corner -- a thin, weakly polarised film whose memory window is too
    # small to separate the two conductance states -- gives the loop something to
    # recover, and the recovery is the result.
    theta0: str = ""


@dataclass
class _State:
    step: int = 0
    accepted: int = 0
    rejected: int = 0
    radius: float = 0.0
    stalled: int = 0
    solved_hashes: set = field(default_factory=set)


def _provenance_since(path: Path, offset: int) -> set[tuple[str, str]]:
    """Distinct (backend, input-hash) pairs appended to the log after `offset`.

    Two details, both learned the hard way on D2:

    * **Key on (backend, hash), not on hash.** `content_hash` hashes the INPUTS,
      so the mock and DEVSIM produce the same key at the same design point. A
      set of bare hashes therefore counts a devsim evaluation as already-seen
      because a mock smoke run visited that theta last week.
    * **Scope to this run by byte offset.** The log is append-only and shared by
      every run in the project's history; `set(now) - set(before)` is not the
      same as "what this run did" once any earlier run touched the same points.

    The result is the honest cost of the run: one entry per design point the
    solver was actually asked about, cache hits within the run collapsing into
    the point they hit.
    """
    if not path.is_file():
        return set()
    out: set[tuple[str, str]] = set()
    with open(path, encoding="utf-8") as fh:
        fh.seek(offset)
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001 -- a truncated last line is not an error
                continue
            out.add((str(rec.get("backend")), str(rec.get("hash"))))
    return out


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def _theta0(cfg: FlagshipConfig) -> np.ndarray:
    if not cfg.theta0:
        return nominal_theta(cfg.d)
    v = np.array([float(x) for x in cfg.theta0.split(",")], dtype=np.float64)
    if v.size != cfg.d:
        raise ValueError(f"theta0 has {v.size} entries, expected d={cfg.d}")
    return np.clip(v, 0.0, 1.0)


def _phys_row(theta, d: int) -> dict[str, float]:
    spec = get_design(d)
    return dict(
        zip(spec.names, (float(v) for v in denormalise(np.asarray(theta), spec)), strict=True)
    )


def run_flagship(cfg: FlagshipConfig, out_dir: Path | None = None) -> dict:
    """Run the loop. Writes steps.jsonl and result.json as it goes, not at the end."""
    from tesseract_core import Tesseract

    # THE MATERIAL LOCK, enforced here rather than trusted.
    #
    # Pr and Ec are properties of the deposited HZO film, measured once against
    # Liao 2022 Fig. 7 (see shared.material). An optimiser that is handed them
    # will improve the memory window by changing the material, which is not a
    # design result -- it is asking for a different film and calling it a win.
    # The legacy d=3/5/12 vectors still expose them, deliberately, so that every
    # result banked before the D3 recalibration can still be REPLAYED. They just
    # cannot be descended on.
    locked = tunes_locked_material(cfg.d)
    if locked and os.environ.get("DIFFSILICON_ALLOW_LOCKED_MATERIAL") != "1":
        raise ValueError(
            f"d={cfg.d} ({get_design(cfg.d).label}) exposes {', '.join(locked)}, which "
            f"are LOCKED material constants of the calibrated HZO film "
            f"(Pr = {PR_UC_CM2} uC/cm2, Ec = {EC_MV_CM} MV/cm; see "
            f"src/diffsilicon/shared/material.py). The design vector to optimise is "
            f"d=4: t_fe, L_g, N_ch, t_IL. Set DIFFSILICON_ALLOW_LOCKED_MATERIAL=1 "
            f"only to reproduce a pre-recalibration run."
        )

    out = Path(out_dir or cfg.out_dir or (_REPO / "results" / "runs" / cfg.tag))
    out.mkdir(parents=True, exist_ok=True)
    steps_path = out / "steps.jsonl"
    result_path = out / "result.json"

    os.environ["ORACLE_BACKEND"] = cfg.backend
    os.environ["SHIM_ALPHA"] = str(cfg.alpha)
    os.environ["SHIM_REFRESH_EVERY"] = str(cfg.refresh_every)
    os.environ["SHIM_MAX_ORACLE_CALLS"] = str(cfg.max_oracle_calls)
    # ORDER MATTERS AND IT IS LOAD-BEARING. T4 reads SNN_TRAIN_MODE once, at
    # import, and the Tesseract that imports it is constructed a few lines below.
    # Setting it after that construction would be silently ignored. The one case
    # this cannot cover is a process that already imported T4 for some other
    # reason; `result.json` records `train_mode` so such a run can be caught.
    if cfg.train_mode not in ("frozen", "adapt", "scratch"):
        raise ValueError(
            f"train_mode={cfg.train_mode!r}; expected 'frozen', 'adapt' or 'scratch'."
        )
    os.environ["SNN_TRAIN_MODE"] = cfg.train_mode

    prov = Path(
        os.environ.get(
            "DIFFSILICON_PROVENANCE_LOG", _REPO / "results" / "runs" / "provenance.jsonl"
        )
    )
    prov_offset = _file_size(prov)

    cc = load_circuit()
    api = _REPO / "tesseracts" / "{}" / "tesseract_api.py"
    shim_t = Tesseract.from_tesseract_api(str(api).format("adjoint-shim"))
    snn_t = Tesseract.from_tesseract_api(str(api).format("snn-lif-ecg"))

    st = _State(radius=cfg.trust_radius)
    t_start = time.time()

    with shim_t, snn_t:

        def loss_of(theta):
            return composed_loss(
                shim_t, snn_t, theta, cc,
                lambda_e=cfg.lambda_e, lambda_r=cfg.lambda_r,
                seed=cfg.seed, batch=cfg.batch, smooth_spikes=False,
            )

        value_and_grad = jax.value_and_grad(loss_of)

        def calls_used() -> int:
            return len(_provenance_since(prov, prov_offset))

        def observables(theta):
            """FoMs, phi and the network metrics at theta.

            Free in solver calls: the oracle result at this theta is already in
            the content-addressed cache because the loss was just evaluated there.
            """
            y = oracle_call(shim_t, theta)
            phi = transduce_jax(y, theta, cc)
            out = apply_tesseract(
                snn_t,
                {
                    **{k: phi[k] for k in PHI_KEYS},
                    "seed": cfg.seed, "batch": cfg.batch, "smooth_spikes": False,
                },
            )
            return (
                {k: float(y[k]) for k in ("ss", "vth_fwd", "vth_rev", "i_leak",
                                          "g_lo", "g_hi", "dg_dvth")},
                {k: float(phi[k]) for k in PHI_KEYS},
                {k: float(out[k]) for k in ("loss", "spikes", "accuracy")},
            )

        theta = jnp.asarray(_theta0(cfg))
        template = make_oracle_input(np.asarray(theta))
        shim = shim_for(template)  # the same object the T3 endpoints hold

        # A trial point the solver cannot handle is survivable -- see the
        # `solver_failed` branch below. The STARTING point is not: there is no
        # smaller step to fall back to, and a run that begins nowhere has nothing
        # to report. Say which point it was, in physical units, and stop.
        try:
            loss, grad = value_and_grad(theta)
        except OracleBudgetExhausted:
            raise
        except OracleNotConverged as exc:
            # Same reasoning as the solver-crash case just below, for the other
            # failure: a starting point whose figures of merit are not a
            # measurement gives the run nothing true to descend from, and there
            # is no smaller step to retreat to. Stop, and name the device.
            raise RuntimeError(
                f"the oracle did not converge at the STARTING design point "
                f"{_phys_row(theta, cfg.d)}: {exc}"
            ) from exc
        except RuntimeError as exc:
            raise RuntimeError(
                f"the solver failed at the STARTING design point "
                f"{_phys_row(theta, cfg.d)}: {exc}"
            ) from exc
        loss = float(loss)
        loss0 = loss
        fom0, phi0, net0 = observables(theta)

        rows = []
        while st.step < cfg.max_steps and calls_used() < cfg.max_oracle_calls:
            gnorm = float(jnp.linalg.norm(grad))
            if gnorm == 0.0:
                break

            s = -st.radius * grad / gnorm
            theta_try = box_project(theta + s)
            # The DIRECTION uses the whole gradient. Only the predicted decrease
            # is bounded -- see `grad_norm_cap`.
            gnorm_pred = (min(gnorm, cfg.grad_norm_cap) if cfg.grad_norm_cap > 0
                          else gnorm)
            predicted = st.radius * gnorm_pred  # first-order predicted decrease

            try:
                loss_try, grad_try = value_and_grad(theta_try)
            except OracleBudgetExhausted as exc:
                rows.append({"step": st.step, "event": "budget", "detail": str(exc)})
                break
            except (OracleNotConverged, RuntimeError) as exc:
                # TWO DIFFERENT THINGS CAN GO WRONG AT A TRIAL POINT, and both
                # are handled the same way but must be RECORDED apart.
                #
                # `solver_failed` -- the solver crashed and returned nothing.
                # Measured on the open solver: every point in the d=4 box sweeps
                # to -3.50 V except the exact ceiling of the film thickness
                # range, 15.0 nm, where the Jacobian goes singular in deep
                # accumulation. `box_project` clamps to the bounds, so the
                # optimiser can land on that face exactly.
                #
                # `not_converged` -- the solver RAN and the answer is not a
                # measurement: the currents were not all finite, or a threshold
                # voltage came from outside the swept window. Added D4. This one
                # is invisible without the flag, because it returns a complete
                # set of finite, plausible figures of merit.
                #
                # Neither is the same thing as running out of budget, and until
                # D3 the first was recorded as if it were.
                #
                # A trial point we cannot evaluate is a trial point we cannot
                # accept. That is a rejection, and the trust-region already knows
                # what to do with one: shrink and try nearer home. Stopping the
                # whole run instead throws away every remaining call because one
                # corner of the box is unusable.
                rows.append({
                    "step": st.step,
                    "event": ("not_converged" if isinstance(exc, OracleNotConverged)
                              else "solver_failed"),
                    "theta_try": [float(v) for v in np.asarray(theta_try)],
                    "theta_try_phys": _phys_row(theta_try, cfg.d),
                    "radius": st.radius,
                    "detail": str(exc)[:400],
                })
                st.rejected += 1
                if st.radius <= cfg.min_radius + 1e-12:
                    st.stalled += 1
                    if st.stalled >= 2:
                        rows.append({"step": st.step, "event": "stalled_unsolvable"})
                        st.step += 1
                        break
                st.radius = max(st.radius * 0.5, cfg.min_radius)
                shim.radius = max(shim.radius, cfg.min_radius)
                st.step += 1
                continue
            loss_try = float(loss_try)

            actual = loss - loss_try
            rho = actual / predicted if predicted > 0 else 0.0
            accept = actual > 0.0

            row = {
                "step": st.step,
                "t": round(time.time() - t_start, 2),
                "backend": cfg.backend,
                "theta": [float(v) for v in np.asarray(theta)],
                "theta_phys": _phys_row(theta, cfg.d),
                "loss": loss,
                "loss_try": loss_try,
                "grad_norm": gnorm,
                # Both are recorded. The raw norm is the measurement; the capped
                # one is what rho was judged against. Reporting only the second
                # would hide how far apart they are, which is the finding.
                "grad_norm_used": gnorm_pred,
                "radius": st.radius,
                "rho": rho,
                "accepted": bool(accept),
                "oracle_calls": calls_used(),
                "refreshes": shim.ctr.refreshes,
                "broyden_updates": shim.ctr.broyden_updates,
                "content_hash": content_hash(make_oracle_input(np.asarray(theta))),
            }

            # Hand the measured rho to the shim: rho < 0.25 forces it to rebuild J
            # from the solver instead of trusting another Broyden update. This is
            # the only feedback the local model gets about whether it is still
            # describing the solver, and it is what the budget is FOR.
            shim.record_step(rho)

            if accept:
                theta, loss, grad = theta_try, loss_try, grad_try
                st.accepted += 1
                st.stalled = 0
                if rho > 0.5:
                    st.radius = min(st.radius * 1.6, cfg.max_radius)
            else:
                st.rejected += 1
                # STOP, do not cycle. Once the radius is on the floor a rejected
                # step proposes the SAME point again next iteration -- same theta,
                # same J, same direction -- and the loop spins until max_steps
                # burning wall clock on a cached evaluation it has already seen.
                # A forced refresh at the same theta rebuilds the same Jacobian,
                # so a second rejection at the floor really is the end of the
                # descent this local model can find.
                if st.radius <= cfg.min_radius + 1e-12:
                    st.stalled += 1
                    if st.stalled >= 2:
                        rows.append({"step": st.step, "event": "stalled_at_min_radius"})
                        st.step += 1
                        break
                st.radius = max(st.radius * 0.5, cfg.min_radius)
            shim.radius = max(shim.radius, cfg.min_radius)

            row["theta_next"] = [float(v) for v in np.asarray(theta)]
            row["loss_next"] = loss
            rows.append(row)
            with open(steps_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")

            st.step += 1
            _write_result(result_path, cfg, st, theta, loss, loss0, fom0, phi0, net0,
                          observables(theta), calls_used(), rows, t_start, shim)

        return _write_result(result_path, cfg, st, theta, loss, loss0, fom0, phi0, net0,
                             observables(theta), calls_used(), rows, t_start, shim)


def _write_result(path, cfg, st, theta, loss, loss0, fom0, phi0, net0,
                  obs_now, calls, rows, t_start, shim) -> dict:
    fom, phi, net = obs_now
    rhos = [r["rho"] for r in rows if "rho" in r]
    res = {
        "tag": cfg.tag,
        "config": asdict(cfg),
        "wall_seconds": round(time.time() - t_start, 1),
        "steps": st.step,
        "accepted": st.accepted,
        "rejected": st.rejected,
        "oracle_calls": calls,
        "shim_probes": shim.ctr.calls,
        "jacobian_refreshes": shim.ctr.refreshes,
        "broyden_updates": shim.ctr.broyden_updates,
        "trust_radius": st.radius,
        "objective_initial": loss0,
        "objective_final": loss,
        "objective_delta": loss - loss0,
        # G6 is stated on the CLASS-BALANCED CROSS-ENTROPY, not on the composite
        # objective, and the two are not the same thing: the objective carries the
        # energy penalty. Both are reported so neither can be quoted selectively.
        "ce_initial": net0["loss"],
        "ce_final": net["loss"],
        "ce_delta": net["loss"] - net0["loss"],
        "accuracy_initial": net0["accuracy"],
        "accuracy_final": net["accuracy"],
        "spikes_initial": net0["spikes"],
        "spikes_final": net["spikes"],
        "theta_initial": [float(v) for v in np.asarray(_theta0(cfg))],
        "theta_final": [float(v) for v in np.asarray(theta)],
        "theta_initial_phys": _phys_row(_theta0(cfg), cfg.d),
        "theta_final_phys": _phys_row(theta, cfg.d),
        "fom_initial": fom0,
        "fom_final": fom,
        "phi_initial": phi0,
        "phi_final": phi,
        "rho_positive_fraction": (
            float(np.mean([r > 0 for r in rhos])) if rhos else None
        ),
        "rho_history": rhos,
    }
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res
