"""T1: drive Synopsys Sentaurus on a remote CentOS 7 host from Windows.

This runs LOCALLY. It does not serve on the Sentaurus host and nothing is
installed there. That is a deliberate architectural decision, not a limitation
worked around: the boundary criterion #1 actually judges -- PyTorch <-> JAX <->
closed binary -- is identical either way, and putting Python 3.11 on the solver
host would delete the strongest sentence available:

    the solver host has Python 2.7.5 and nothing else. I installed nothing on it.
    Tesseract's boundary landed exactly where the technology ran out.

Everything the host is asked to do is expressible as `csh -c 'source ~/.cshrc && ...'`.

What this module has to survive
-------------------------------
* **csh, not bash.** `~/.cshrc` uses `setenv`/`set path`; sourcing it from bash
  dies with "Illegal variable name". Every remote command is wrapped in csh.
* **csh has no `2>/dev/null`.** It gives "Ambiguous output redirect". Use `>&`.
* **A `#` inside an echo argument starts a csh comment** and truncates the line.
* **Exactly ONE sdevice license.** Parallel runs queue; a run that times out on
  this side keeps running remotely and HOLDS the license. Hence an in-process
  lock, a bounded wait, and an explicit `pkill -9 sdevice` recovery path.
* **~5 min per run**, measured: a full transient sdevice run took 306 s, exit 0.

Placeholder convention (kept intact so the deck can be re-imported into SWB):
`@tdr@`, `@tdrdat@`, `@plot@`, `@log@`, and one `@NAME@` per swept parameter.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .shared.contract import OracleInput

__all__ = ["T1Config", "RemoteRunner", "render_template", "deck_values", "id_vg_curves"]

# One sdevice license exists. This lock serialises every call made by this
# process; it cannot serialise other people on the host, which is why
# `wait_for_license` exists as well.
_LICENSE_LOCK = threading.Lock()

_PLACEHOLDER = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)@")


@dataclass
class T1Config:
    host: str = os.environ.get("SENTAURUS_HOST", "du@103.28.121.70")
    password: str = os.environ.get("SENTAURUS_PASSWORD", "")
    # plink -batch REFUSES to connect to a host whose key it has not cached, and
    # a fresh machine has cached nothing -- so the fingerprint measured on D1 is
    # carried here rather than left to an interactive prompt that -batch will
    # never show. Overridable, because pinning a key you did not verify yourself
    # is worse than not pinning one.
    host_key: str = os.environ.get(
        "SENTAURUS_HOST_KEY",
        "SHA256:v9sOpO64cVy+vAh1VDJqpldWBX9KaGZJGxT89iwlvSE",
    )
    remote_root: str = os.environ.get(
        "SENTAURUS_REMOTE_ROOT", "Sentaurus-files/Sami_Hozaifa/GAAFet"
    )
    plink: str = os.environ.get("PLINK", r"C:/Program Files/PuTTY/plink.exe")
    pscp: str = os.environ.get("PSCP", r"C:/Program Files/PuTTY/pscp.exe")
    workdir: str = "diffsilicon_runs"

    # REBUILD THE MESH PER DESIGN POINT. Default OFF.
    #
    # Off, the driver ships one .cmd and one .par against the fixed .tdr in
    # `grid`, and only t_fe reaches the solver -- through the fixed-slab remap in
    # `deck_values`. L_g, N_ch and t_IL are geometry and doping, they live in the
    # mesh, and three of the four d=4 Jacobian columns come back identically zero.
    #
    # On, `build_mesh` renders t1/sde_fefet_mesh.cmd at this design point, runs
    # sde on the host, and sdevice then solves on THAT mesh. All four variables
    # are live. Two costs, both real: an sde run per design point on top of
    # sdevice's ~306 s, and a mesh that changes with theta, so a finite-difference
    # column differences two discretisations. That second one is exactly the
    # noise V1/G7 exist to measure, and it is why this is opt-in rather than the
    # default -- an unattended overnight run should not be the first thing to try
    # it.
    rebuild_mesh: bool = os.environ.get("T1_REBUILD_MESH", "0") == "1"
    timeout_s: int = 1800  # 6x the measured 306 s single-run wall clock
    retries: int = 3
    backoff_s: float = 20.0
    license_wait_s: int = 2400

    # --- deck wiring -------------------------------------------------------
    drain_electrode: str = "drain_contact"
    gate_electrode: str = "gate_contact"
    # --- device calibration -------------------------------------------------
    # DELIBERATELY EMPTY IN TRACKED SOURCE. These are the output of a
    # calibration campaign against measured hysteresis -- somebody's unpublished
    # research, not a library constant. They load from t1/calibration.local.json,
    # which is gitignored. See t1/calibration.example.json for the shape.
    grid: str = ""
    t_fe_slab_nm: float = 0.0
    # Baseline geometry of `grid`, i.e. the parameters the CALIBRATED mesh was
    # built at. Only used when rebuilding the mesh per design point: it is what
    # `build_mesh` reproduces to check itself against the fixed-slab path.
    l_gate_slab_nm: float = 0.0
    t_il_slab_nm: float = 0.0
    n_ch_slab: float = 0.0
    # Background relative permittivity of the ferroelectric. It used to be a
    # literal in the .par; the fixed-slab remap has to scale it with t_fe, so it
    # lives here now. Unchanged at t_fe = t_fe_slab_nm, i.e. the calibration is
    # exactly preserved at the thickness it was fitted at.
    eps_fe_bg: float = 0.0
    workfunction: float = 0.0
    area_factor: float = 0.0
    fixq: str = ""
    dit: str = ""
    dite: str = ""
    agen: str = ""
    bgen: str = ""
    taun_srh: str = ""
    taup_srh: str = ""
    tau_e: float = 0.0
    tau_p: float = 0.0

    def __post_init__(self) -> None:
        cal = _load_calibration()
        for k, v in cal.items():
            if k.startswith("_"):
                continue
            if not getattr(self, k, None):
                setattr(self, k, v)
        missing = [
            k for k in ("grid", "workfunction", "area_factor", "tau_e", "tau_p")
            if not getattr(self, k)
        ]
        if missing:
            raise RuntimeError(
                f"T1 device calibration missing: {missing}. Copy "
                f"t1/calibration.example.json to t1/calibration.local.json and fill "
                f"it in. The repository ships no calibration of its own."
            )


def render_template(template: str, values: dict[str, str | float]) -> str:
    """Substitute every @placeholder@ and refuse to ship a deck with one left.

    The guardrail is not decoration: an unresolved `@V_read@` reaches sdevice as
    a literal, the parse fails several minutes in, and the license is held for
    the whole of it.
    """
    out = template
    for k, v in values.items():
        out = out.replace(f"@{k}@", f"{v}")
    leftover = sorted(set(_PLACEHOLDER.findall(out)))
    if leftover:
        raise ValueError(f"unresolved placeholders in rendered deck: {leftover}")
    return out


class RemoteRunner:
    """plink/pscp round trip with retry-with-backoff and license serialisation."""

    def __init__(self, cfg: T1Config | None = None):
        self.cfg = cfg or T1Config()
        if not self.cfg.password:
            raise RuntimeError(
                "SENTAURUS_PASSWORD is unset. T1 is the bring-your-own-license tier; "
                "see .env.example and docs/T1_CONTAINER.md. Tier A/B/C need none of this."
            )

    # --- primitives -------------------------------------------------------
    def _run(self, argv: list[str], timeout: int) -> subprocess.CompletedProcess:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )

    def sh(self, command: str, timeout: int | None = None) -> subprocess.CompletedProcess:
        """Run one command on the host inside a csh that has sourced ~/.cshrc."""
        wrapped = f"csh -c 'source $HOME/.cshrc && {command}'"
        argv = [self.cfg.plink, "-batch", "-ssh", "-pw", self.cfg.password,
                *self._hostkey_args(), self.cfg.host, wrapped]
        return self._run(argv, timeout or self.cfg.timeout_s)

    def _hostkey_args(self) -> list[str]:
        return ["-hostkey", self.cfg.host_key] if self.cfg.host_key else []

    def push(self, local: Path, remote_rel: str) -> None:
        argv = [self.cfg.pscp, "-batch", "-pw", self.cfg.password,
                *self._hostkey_args(), str(local), f"{self.cfg.host}:{remote_rel}"]
        r = self._run(argv, 300)
        if r.returncode != 0:
            raise RuntimeError(f"pscp upload failed: {r.stderr.strip()}")

    def pull(self, remote_rel: str, local: Path) -> None:
        argv = [self.cfg.pscp, "-batch", "-pw", self.cfg.password,
                *self._hostkey_args(), f"{self.cfg.host}:{remote_rel}", str(local)]
        r = self._run(argv, 300)
        if r.returncode != 0:
            raise RuntimeError(f"pscp download failed: {r.stderr.strip()}")

    # --- license ----------------------------------------------------------
    def license_busy(self) -> bool:
        # `grep -c` exits 1 on zero matches, which breaks a csh && chain, so this
        # deliberately uses `;` and reads the count from stdout instead.
        r = self.sh("ps -ef | grep -v grep | grep -c sdevice ; echo DONE", timeout=120)
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                return int(line) > 0
        return False

    def wait_for_license(self) -> None:
        deadline = time.time() + self.cfg.license_wait_s
        while time.time() < deadline:
            if not self.license_busy():
                return
            time.sleep(30)
        raise TimeoutError(
            f"sdevice license still held after {self.cfg.license_wait_s} s. "
            f"If it is a zombie of ours, free it with: pkill -9 sdevice"
        )

    def free_license(self) -> None:
        """Kill any sdevice we may have orphaned. Only ever call this on OUR runs."""
        self.sh("pkill -9 sdevice ; echo DONE", timeout=120)

    # --- the actual job ---------------------------------------------------
    def run_deck(self, deck_local: Path, tag: str, extra: list | None = None) -> str:
        """Upload a fully-resolved deck, run sdevice, return the remote work dir.

        NOT a single .plt path. Every `NewCurrentPrefix` in the deck makes sdevice
        emit its own `<prefix>_<basename>.plt` beside the `Current=` file, so a
        deck with a forward and a reverse sweep produces two curve files plus one
        per set-up step. The first working run of this deck was parsed by taking
        the un-prefixed file and cutting it in half; the forward branch looked
        plausible and the reverse came back flat at 1e-21, which is what reading
        the wrong file looks like.
        """
        rroot = f"{self.cfg.remote_root}/{self.cfg.workdir}"
        self.sh(f"mkdir -p $HOME/{rroot} ; echo DONE", timeout=120)
        self.push(deck_local, f"{rroot}/{tag}_des.cmd")
        for f in extra or []:
            self.push(f, f"{rroot}/{f.name}")

        last = None
        for attempt in range(1, self.cfg.retries + 1):
            with _LICENSE_LOCK:
                self.wait_for_license()
                r = self.sh(
                    f"cd $HOME/{self.cfg.remote_root} && "
                    f"sdevice {self.cfg.workdir}/{tag}_des.cmd >& "
                    f"{self.cfg.workdir}/{tag}_runlog.txt ; echo EXIT=$status"
                )
            if "EXIT=0" in r.stdout:
                return rroot
            last = r.stdout + r.stderr
            # A timed-out run keeps going remotely and keeps the license. Clear it
            # before retrying, or the retry queues behind our own orphan.
            self.free_license()
            time.sleep(self.cfg.backoff_s * attempt)
        raise RuntimeError(f"sdevice failed after {self.cfg.retries} attempts: {last}")


def parse_plt(text: str) -> dict[str, np.ndarray]:
    """Parse a DF-ISE .plt curve file.

    The dataset names appear in `datasets = [ ... ]` and the numbers follow in a
    flat `Data { ... }` block, column-major in HEADER order -- which is not the
    order they are listed elsewhere in the file. Index by NAME, never by position.
    """
    names_m = re.search(r"datasets\s*=\s*\[(.*?)\]", text, re.S)
    data_m = re.search(r"Data\s*\{(.*?)\}", text, re.S)
    if not names_m or not data_m:
        raise ValueError("not a DF-ISE .plt file: missing datasets or Data block")
    names = re.findall(r'"([^"]+)"', names_m.group(1))
    vals = np.fromstring(data_m.group(1).replace("\n", " "), sep=" ")
    ncol = len(names)
    if vals.size % ncol:
        raise ValueError(f"{vals.size} values do not divide into {ncol} columns")
    arr = vals.reshape(-1, ncol)
    return {n: arr[:, i] for i, n in enumerate(names)}


_T1_DIR = Path(__file__).resolve().parents[2] / "t1"


def _load_calibration() -> dict:
    """Device calibration, from a file this repository does not ship.

    The numbers that make a TCAD deck reproduce a measured device are somebody's
    research output. Hard-coding them here would publish them, and this repo is
    public, so they live in a gitignored local file and the code refuses to run
    without one rather than silently using a plausible-looking default.
    """
    import json

    path = Path(os.environ.get("T1_CALIBRATION", _T1_DIR / "calibration.local.json"))
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def content_tag(values: dict) -> str:
    """Deterministic 8-hex tag for a rendered deck.

    NOT `hash()`: Python salts string hashing per process, so the same design
    point would land in a different remote directory on every run and the remote
    scratch would be neither reusable nor auditable.
    """
    import hashlib

    blob = "\n".join(f"{k}={values[k]}" for k in sorted(values)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def deck_values(
    inputs: OracleInput, cfg: T1Config | None = None, mesh_is_exact: bool = False
) -> dict:
    """Every @token@ the deck and the parameter file need, for ANY design vector.

    Two things happen here rather than in the deck, because sdevice cannot do
    arithmetic and because getting either wrong is silent:

    * **The frozen defaults are filled in.** A d=3 point still has an L_g and an
      N_ch; they are the same numbers `oracle_devsim.fefet_params` and
      `shared.mock_device` use. Without this, a deck mentioning @L_g@ would fail
      to render at d=3 and succeed at d=5 -- exactly the kind of asymmetry that
      makes two oracles quietly stop being comparable.
    * **The fixed-slab thickness remap**, unless the mesh is being rebuilt.
      t_fe is a design variable and the grid is normally not regenerated per
      point, so a slab of `cfg.t_fe_slab_nm` is made to behave as one of
      thickness t_fe by scaling eps and Ec by t_slab/t_fe. Exact for a uniform
      field.

      `mesh_is_exact=True` turns it OFF, and that is not an optimisation. When
      `build_mesh` has just built the film at its true thickness, remapping on
      top would count the thickness twice -- once in the geometry and once in
      the material -- and the memory window would come out wrong by (t_fe/t_slab)
      squared, silently, on a deck that renders and runs perfectly.
    """
    from .oracle_devsim import (
        fefet_params,
    )

    cfg = cfg or T1Config()
    # `fefet_params` is what fills in the frozen defaults AND the locked material
    # constants, so the deck and DEVSIM see the same device by construction
    # rather than by two copies of the same table agreeing.
    p = fefet_params(np.asarray(inputs.theta, dtype=np.float64))
    t_slab_cm = cfg.t_fe_slab_nm * 1e-7
    # t_fe is real geometry on this host: `cfg.grid` is a mesh built at a
    # specific ferroelectric thickness. Record the mismatch rather than hide it,
    # in `t_fe_snap_error_nm` at the bottom of this function.

    # --- THE FIXED-SLAB THICKNESS REMAP -----------------------------------
    #
    # The deck runs on ONE mesh, built at cfg.t_fe_slab_nm. t_fe is a design
    # variable, so the slab has to be made to BEHAVE like a film of thickness
    # t_fe. Two quantities carry the thickness, and they scale by RECIPROCAL
    # factors:
    #
    #     k          = t_fe / t_slab
    #     Ec_eff     = Ec * k          coercive VOLTAGE preserved: Ec_eff*t_slab = Ec*t_fe
    #     eps_fe_eff = eps_bg / k      capacitance/area preserved: eps_eff/t_slab = eps_bg/t_fe
    #
    # Using the same factor for both renders fine, runs fine, and silently
    # multiplies the memory window by (t_slab/t_fe)^2. The invariant that catches
    # it is the one in the comments above, and it is asserted in
    # tests/test_g5_devsim_fefet.py.
    #
    # Pr and Ps are NOT remapped: polarization is a charge per unit AREA, so it
    # does not depend on how thick the film is.
    #
    # THIS IS WHY THE REMAP IS NOW LOAD-BEARING RATHER THAN TIDY. Before the D3
    # recalibration the deck varied with Pr and Ec, which were design variables.
    # They are locked material constants now, so `t_fe` -- through this remap --
    # is the ONLY thing that makes one d=4 design point's deck differ from
    # another's. Without it every point would render the identical deck and T1's
    # Jacobian would be exactly zero.
    k = 1.0 if mesh_is_exact else p.t_fe / t_slab_cm
    ec_eff = p.ec * k
    eps_fe_eff = cfg.eps_fe_bg / k

    values = {
        # --- the design vector, which is the ONLY thing this project varies ---
        # Raw Pr and Ps, NOT de-rated by FE_ACTIVE_FRACTION. That constant is a
        # DEVSIM-side correction for T2's idealised model. The commercial
        # solver's polarization model is calibrated against measured hysteresis
        # by whoever supplied calibration.local.json, and pushing a correction
        # for a different model into it would corrupt that calibration.
        "PR": f"{p.pr:.9g}",
        "PS": f"{p.ps:.9g}",
        # .12g, not the .9g the untouched calibration constants use. These two
        # are the REMAPPED values, and the remap has an exact invariant across
        # them -- Ec_eff*t_slab == Ec*t_fe. At 9 significant digits the round
        # trip through the string loses ~2.5e-9 relative, which is larger than
        # the tolerance the invariant is worth asserting at. Cheap to keep exact.
        "FC": f"{ec_eff:.12g}",
        "EPS_FE_EFF": f"{eps_fe_eff:.12g}",
        # --- everything below is the user's calibrated setup, left alone ------
        "WF": f"{cfg.workfunction:.9g}",
        "AREA": f"{cfg.area_factor:.9g}",
        "FIXQ": cfg.fixq,
        "DIT": cfg.dit,
        "DITE": cfg.dite,
        "TAUE": f"{cfg.tau_e:.9g}",
        "TAUP": f"{cfg.tau_p:.9g}",
        "TAUN": cfg.taun_srh,
        "TAUP_SRH": cfg.taup_srh,
        "AGEN": cfg.agen,
        "BGEN": cfg.bgen,
        "tdr": cfg.grid,
    }
    # Recorded, not hidden: the remap makes the slab behave like a film of
    # thickness t_fe electrically, but the MESH is still the slab's, so anything
    # that depends on the geometry rather than on the field sees t_slab.
    values["t_fe_snap_error_nm"] = f"{p.t_fe / 1e-7 - cfg.t_fe_slab_nm:.9g}"
    values["t_fe_remap_k"] = f"{k:.9g}"

    # WHAT T1 CANNOT SEE, stated here because a silent zero column is worse than
    # a missing feature.
    #
    # `L_g`, `log10_N_ch` and `t_IL` are three of the four d=4 design variables
    # and NONE of them reaches this deck. They are geometry and doping, baked
    # into `cfg.grid` when the mesh was built, and this driver does not rebuild
    # the mesh per design point -- it ships one .cmd and one .par against a fixed
    # .tdr. So on the commercial solver a d=4 Jacobian has ONE live column,
    # t_fe, and three that are identically zero.
    #
    # DEVSIM regenerates its grid per design point, so all four are live on T2,
    # and the flagship runs on T2. This matters for the T1-vs-T2 cross-check
    # (V4): compare the t_fe column, and do not read anything into the other
    # three agreeing at zero.
    #
    # Closing it means running `sde` per design point to rebuild the mesh. That
    # is real work and it is not done.
    return values


def mesh_values(inputs: OracleInput, cfg: T1Config | None = None) -> dict:
    """Every @token@ the sde mesh template needs, in MICRONS.

    sde works in microns; the design vector is in nm and cm^-3. Getting the unit
    conversion wrong here builds a device a thousand times too big and it still
    meshes, still solves, and still returns a curve.
    """
    from .oracle_devsim import fefet_params

    cfg = cfg or T1Config()
    p = fefet_params(np.asarray(inputs.theta, dtype=np.float64))
    return {
        "L_GATE": f"{p.l_g / 1e-4:.9g}",  # cm -> um
        "T_OX": f"{p.t_il / 1e-4:.9g}",
        "T_FE": f"{p.t_fe / 1e-4:.9g}",
        "N_SUB": f"{p.n_ch:.9g}",  # cm^-3, as sde wants
    }


def baseline_mesh_values(cfg: T1Config | None = None) -> dict:
    """The mesh parameters that reproduce `cfg.grid`, i.e. the calibrated device.

    This is the control. Build a mesh at THESE numbers, run the nominal design
    point on it, and it must land on the figures of merit the fixed-slab path
    already gives for `cfg.grid`. Until that check has been run, a rebuilt-mesh
    result is a code path and not a measurement -- see t1/sde_fefet_mesh.cmd.
    """
    cfg = cfg or T1Config()
    missing = [
        n for n, v in (
            ("l_gate_slab_nm", cfg.l_gate_slab_nm),
            ("t_il_slab_nm", cfg.t_il_slab_nm),
            ("n_ch_slab", cfg.n_ch_slab),
            ("t_fe_slab_nm", cfg.t_fe_slab_nm),
        ) if not v
    ]
    if missing:
        raise ValueError(
            f"calibration.local.json is missing {', '.join(missing)}. Those are the "
            f"geometry {cfg.grid!r} was actually built at, and without them there is "
            f"nothing to validate a rebuilt mesh against."
        )
    return {
        "L_GATE": f"{cfg.l_gate_slab_nm * 1e-3:.9g}",
        "T_OX": f"{cfg.t_il_slab_nm * 1e-3:.9g}",
        "T_FE": f"{cfg.t_fe_slab_nm * 1e-3:.9g}",
        "N_SUB": f"{cfg.n_ch_slab:.9g}",
    }


def build_mesh(runner: RemoteRunner, values: dict, tag: str) -> str:
    """Render the sde template at `values`, run sde on the host, return the mesh name.

    Returns the BASENAME sde was told to write, e.g. "ds_1a2b3c_msh"; sde appends
    the .tdr itself. The caller hands that to sdevice as @tdr@.
    """
    cfg = runner.cfg
    template_path = _T1_DIR / "sde_fefet_mesh.cmd"
    if not template_path.is_file():
        raise NotImplementedError(
            f"mesh template not found at {template_path}. T1_REBUILD_MESH=1 needs it; "
            f"it is adopted from the user's own sde deck and is gitignored."
        )
    mesh = f"{tag}_msh"
    deck = render_template(
        template_path.read_text(encoding="utf-8"), {**values, "MESH": f"{cfg.workdir}/{mesh}"}
    )
    scratch = Path("t1_scratch")
    scratch.mkdir(exist_ok=True)
    local = scratch / f"{tag}_sde.cmd"
    local.write_text(deck, encoding="utf-8", newline='\n')
    runner.push(local, f"{cfg.workdir}/{tag}_sde.cmd")
    # sde is licensed too. Same single-licence discipline as sdevice.
    runner.wait_for_license()
    runner.sh(f"cd {cfg.remote_root} && sde -e -l {cfg.workdir}/{tag}_sde.cmd")
    return f"{cfg.workdir}/{mesh}.tdr"


def id_vg_curves(inputs: OracleInput) -> np.ndarray:
    """Evaluate one design point on the real solver. Returns (2, 96) Id-Vg.

    The deck template lives at `t1/sdevice_fefet_idvg.cmd` and is authored here,
    not copied from Synopsys documentation -- only self-authored decks and
    numeric outputs may be published under this repository's Apache-2.0 licence.
    """
    cfg = T1Config()
    template_path = _T1_DIR / "sdevice_fefet_idvg.cmd"
    par_path = _T1_DIR / "sdevice_fefet_idvg.par"
    if not template_path.is_file():
        raise NotImplementedError(f"T1 deck template not found at {template_path}.")

    vg = np.asarray(inputs.vg_grid, dtype=np.float64)
    # `mesh_is_exact` and `rebuild_mesh` are the same fact stated twice, and they
    # must not drift apart: if the mesh is built at this point's real geometry,
    # the fixed-slab remap has to be off, or the film's thickness is counted
    # twice. See deck_values.
    values = deck_values(inputs, cfg, mesh_is_exact=cfg.rebuild_mesh)
    if cfg.rebuild_mesh:
        # Fold the mesh geometry into the content tag as well, so two design
        # points that differ ONLY in gate length get different tags and different
        # cached decks. Without this the tag is blind to exactly the variables
        # rebuilding the mesh exists to expose.
        values = {**values, **{f"mesh_{k}": v for k, v in mesh_values(inputs, cfg).items()}}
    tag = f"ds_{content_tag(values)}"
    # EVERY path in the File block is resolved relative to sdevice's cwd, and cwd
    # has to stay at GAAFet/ so the mesh and the host's own material .par
    # includes resolve. So the deck's own files need the workdir prefix -- the
    # first run died on "Cannot open parameter file", with the file sitting right
    # there one directory down.
    # NOTE THE NAMES. In this deck `@plot@` is the CURRENT file (.plt) and
    # `@tdrdat@` is the field plot (.tdr) -- the opposite of what the names
    # suggest. Every path is relative to sdevice's cwd, which stays at GAAFet/,
    # so each needs the workdir prefix.
    w = cfg.workdir
    values.update(
        tdrdat=f"{w}/{tag}_des.tdr",
        plot=f"{w}/{tag}_des.plt",
        log=f"{w}/{tag}_des.log",
        par=f"{w}/{tag}_des.par",
    )

    deck = render_template(template_path.read_text(encoding="utf-8"), values)
    par = render_template(par_path.read_text(encoding="utf-8"), values)

    runner = RemoteRunner(cfg)

    # THE MESH. Either rebuild it at this design point's real geometry, or use
    # the one calibrated mesh and accept that only t_fe reaches the solver.
    if cfg.rebuild_mesh:
        values["tdr"] = build_mesh(runner, mesh_values(inputs, cfg), tag)

    scratch = Path("t1_scratch")
    scratch.mkdir(exist_ok=True)
    local_deck = scratch / f"{tag}_des.cmd"
    local_par = scratch / f"{tag}_des.par"
    local_deck.write_text(deck, encoding="utf-8", newline="\n")
    local_par.write_text(par, encoding="utf-8", newline="\n")

    rroot = runner.run_deck(local_deck, tag, extra=[local_par])

    branches = {}
    for prefix in ("fwd", "rev"):
        local = scratch / f"{prefix}_{tag}_des.plt"
        runner.pull(f"{rroot}/{prefix}_{tag}_des.plt", local)
        cols = parse_plt(local.read_text(encoding="utf-8"))
        i_d = np.abs(cols[f"{cfg.drain_electrode} TotalCurrent"])
        v_g = cols[f"{cfg.gate_electrode} OuterVoltage"]
        # The reverse branch is swept high -> low, so its abscissa descends and
        # np.interp would silently return nonsense. Sort rather than assume.
        order = np.argsort(v_g)
        branches[prefix] = np.interp(vg, v_g[order], i_d[order])

    return np.stack([branches["fwd"], branches["rev"]], axis=0)
