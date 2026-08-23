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
from .shared.design import get_design

__all__ = ["T1Config", "RemoteRunner", "render_template", "id_vg_curves"]

# One sdevice license exists. This lock serialises every call made by this
# process; it cannot serialise other people on the host, which is why
# `wait_for_license` exists as well.
_LICENSE_LOCK = threading.Lock()

_PLACEHOLDER = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)@")


@dataclass
class T1Config:
    host: str = os.environ.get("SENTAURUS_HOST", "du@103.28.121.70")
    password: str = os.environ.get("SENTAURUS_PASSWORD", "")
    remote_root: str = os.environ.get(
        "SENTAURUS_REMOTE_ROOT", "Sentaurus-files/Sami_Hozaifa/GAAFet"
    )
    plink: str = os.environ.get("PLINK", r"C:/Program Files/PuTTY/plink.exe")
    pscp: str = os.environ.get("PSCP", r"C:/Program Files/PuTTY/pscp.exe")
    workdir: str = "diffsilicon_runs"
    timeout_s: int = 1800  # 6x the measured 306 s single-run wall clock
    retries: int = 3
    backoff_s: float = 20.0
    license_wait_s: int = 2400


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
                self.cfg.host, wrapped]
        return self._run(argv, timeout or self.cfg.timeout_s)

    def push(self, local: Path, remote_rel: str) -> None:
        argv = [self.cfg.pscp, "-batch", "-pw", self.cfg.password, str(local),
                f"{self.cfg.host}:{remote_rel}"]
        r = self._run(argv, 300)
        if r.returncode != 0:
            raise RuntimeError(f"pscp upload failed: {r.stderr.strip()}")

    def pull(self, remote_rel: str, local: Path) -> None:
        argv = [self.cfg.pscp, "-batch", "-pw", self.cfg.password,
                f"{self.cfg.host}:{remote_rel}", str(local)]
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
    def run_deck(self, deck_local: Path, tag: str) -> str:
        """Upload a fully-resolved deck, run sdevice, return the remote .plt path."""
        rroot = f"{self.cfg.remote_root}/{self.cfg.workdir}"
        self.sh(f"mkdir -p $HOME/{rroot} ; echo DONE", timeout=120)
        self.push(deck_local, f"{rroot}/{tag}_des.cmd")

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
                return f"{rroot}/{tag}_des.plt"
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


def id_vg_curves(inputs: OracleInput) -> np.ndarray:
    """Evaluate one design point on the real solver. Returns (2, 96) Id-Vg.

    The deck template lives at `t1/sdevice_fefet_idvg.cmd` and is authored here,
    not copied from Synopsys documentation -- only self-authored decks and
    numeric outputs may be published under this repository's Apache-2.0 licence.
    """
    spec = get_design(int(np.asarray(inputs.theta).shape[-1]))
    phys = spec.lo + np.asarray(inputs.theta, dtype=np.float64) * (spec.hi - spec.lo)
    values: dict[str, str | float] = {n: f"{v:.9g}" for n, v in zip(spec.names, phys, strict=True)}

    template_path = Path(__file__).resolve().parents[2] / "t1" / "sdevice_fefet_idvg.cmd"
    if not template_path.is_file():
        raise NotImplementedError(
            f"T1 deck template not found at {template_path}. The Sentaurus FeFET deck "
            f"lands on D2; the driver, templating, license lock and .plt parser are "
            f"D1 work and are exercised by tests/test_t1_driver.py without a license."
        )

    vg = np.asarray(inputs.vg_grid, dtype=np.float64)
    values.update(
        vg_min=f"{vg.min():.9g}", vg_max=f"{vg.max():.9g}", vg_n=f"{vg.size:d}",
        vds=f"{float(inputs.vds_lin):.9g}",
    )
    deck = render_template(template_path.read_text(encoding="utf-8"), values)

    runner = RemoteRunner()
    tag = f"ds_{abs(hash(deck)) & 0xFFFFFFFF:08x}"
    scratch = Path("t1_scratch")
    scratch.mkdir(exist_ok=True)
    local_deck = scratch / f"{tag}_des.cmd"
    local_deck.write_text(deck, encoding="utf-8", newline="\n")

    remote_plt = runner.run_deck(local_deck, tag)
    local_plt = scratch / f"{tag}_des.plt"
    runner.pull(remote_plt, local_plt)

    cols = parse_plt(local_plt.read_text(encoding="utf-8"))
    i_d = np.abs(cols["drain_contact TotalCurrent"])
    v_g = cols["gate_contact OuterVoltage"]
    half = v_g.size // 2
    fwd = np.interp(vg, v_g[:half], i_d[:half])
    rev = np.interp(vg, v_g[half:][::-1], i_d[half:][::-1])
    return np.stack([fwd, rev], axis=0)
