"""T1 driver pieces that can be tested without a license, a host, or a network.

Everything here guards a failure mode that was observed on this specific machine
during earlier Sentaurus work, not a hypothetical one. Each of them costs about
five minutes of a single-license solver to rediscover.
"""

import numpy as np
import pytest

from diffsilicon.t1_driver import T1Config, parse_plt, render_template

PLT = """DF-ISE text

Info {
  version = 1.0
  type    = xyplot
  datasets = [ "time" "gate_contact OuterVoltage" "drain_contact TotalCurrent" ]
  functions = [ time V I ]
}

Data {
 0.0 -0.4 1.0e-12
 1.0  0.0 5.0e-11
 2.0  0.4 3.0e-08
 3.0  0.8 2.0e-06
}
"""


def test_render_substitutes_every_placeholder():
    out = render_template("t_fe=@t_fe@ Pr=@Pr@ grid=@tdr@", {"t_fe": 10.0, "Pr": 15, "tdr": "n1_msh.tdr"})
    assert out == "t_fe=10.0 Pr=15 grid=n1_msh.tdr"


def test_unresolved_placeholder_is_fatal_before_upload():
    """The guardrail that matters most. An unresolved @V_read@ reaches sdevice as a
    literal, the parse fails minutes in, and the single sdevice license is held for
    the whole of it."""
    with pytest.raises(ValueError, match=r"unresolved placeholders.*V_read"):
        render_template("a=@t_fe@ b=@V_read@", {"t_fe": 1.0})


def test_error_names_every_missing_placeholder_not_just_the_first():
    with pytest.raises(ValueError) as e:
        render_template("@a@ @b@ @c@", {"a": 1})
    assert "'b'" in str(e.value) and "'c'" in str(e.value)


def test_placeholder_regex_ignores_bare_at_signs():
    """sdevice decks use `Potential@n0` in edge models. Those are not placeholders,
    and treating them as such would make every real deck unrenderable."""
    deck = "ElectricField = (Potential@n0 - Potential@n1) * EdgeInverseLength"
    assert render_template(deck, {}) == deck


def test_plt_columns_are_indexed_by_name():
    """A .plt Data block is column-major in HEADER order, which is not the order the
    names appear anywhere else in the file. Indexing by position silently returns
    the wrong physical quantity."""
    cols = parse_plt(PLT)
    assert set(cols) == {"time", "gate_contact OuterVoltage", "drain_contact TotalCurrent"}
    assert np.allclose(cols["gate_contact OuterVoltage"], [-0.4, 0.0, 0.4, 0.8])
    assert np.allclose(cols["drain_contact TotalCurrent"], [1e-12, 5e-11, 3e-08, 2e-06])
    assert np.allclose(cols["time"], [0.0, 1.0, 2.0, 3.0])


def test_plt_parser_rejects_a_truncated_file():
    """A run killed for holding the license leaves a partial .plt behind. Reading it
    as if it were complete would quietly corrupt a Jacobian column."""
    with pytest.raises(ValueError, match="do not divide"):
        parse_plt(PLT.replace(" 3.0  0.8 2.0e-06\n", " 3.0  0.8\n"))


def test_plt_parser_rejects_a_non_plt_file():
    with pytest.raises(ValueError, match="not a DF-ISE"):
        parse_plt("this is a log file, not a curve file")


def test_remote_commands_are_wrapped_in_csh():
    """~/.cshrc uses setenv and `set path`; sourcing it from bash dies with
    'Illegal variable name'. Every remote invocation has to go through csh."""
    import inspect

    from diffsilicon import t1_driver

    src = inspect.getsource(t1_driver.RemoteRunner.sh)
    assert "csh -c" in src and "source $HOME/.cshrc" in src


def test_no_bash_style_redirects_reach_csh():
    """csh answers `2>/dev/null` with 'Ambiguous output redirect'. Merging must use
    `>&`, and `#` inside an echo argument starts a csh comment that eats the line."""
    import inspect

    from diffsilicon import t1_driver

    src = inspect.getsource(t1_driver)
    remote = [
        line for line in src.splitlines()
        if ("self.sh(" in line or "sdevice " in line) and not line.strip().startswith("#")
    ]
    joined = "\n".join(remote)
    assert "2>/dev/null" not in joined
    assert "2>&1" not in joined


def test_driver_refuses_to_start_without_credentials(monkeypatch):
    """Tier A, B and C must never touch this path, and the error has to say so."""
    from diffsilicon.t1_driver import RemoteRunner

    monkeypatch.setenv("SENTAURUS_PASSWORD", "")
    with pytest.raises(RuntimeError, match="bring-your-own-license"):
        RemoteRunner(T1Config(password=""))


def test_timeout_has_headroom_over_the_measured_run_time():
    """A full transient sdevice run was measured at 306 s. A timeout anywhere near
    that would abandon runs that are still going -- and an abandoned run keeps the
    single license."""
    assert T1Config().timeout_s >= 5 * 306


def test_retry_path_frees_the_license_before_trying_again():
    """A timed-out run keeps executing remotely and holds the license, so a naive
    retry queues behind our own orphan and times out too, forever."""
    import inspect

    from diffsilicon.t1_driver import RemoteRunner

    src = inspect.getsource(RemoteRunner.run_deck)
    assert "free_license" in src
    assert src.index("free_license") < src.index("backoff_s")
