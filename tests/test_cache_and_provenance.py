"""Tier C: the replay cache, and the provenance log that answers the hard question.

The cache has to populate as a SIDE EFFECT of every run. A cache assembled at the
end of a project proves nothing about what the solver returned during the
optimisation, and "was the forward pass ever a surrogate" is the one question
this project cannot afford to answer with a shrug.
"""

import json
import os

import numpy as np
import pytest

from diffsilicon.shared.cache import CONTRACT_VERSION, CacheStore, content_hash
from diffsilicon.shared.contract import make_oracle_input
from diffsilicon.shared.design import nominal_theta


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("DIFFSILICON_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("DIFFSILICON_PROVENANCE_LOG", str(tmp_path / "provenance.jsonl"))
    import importlib

    import diffsilicon.shared.oracle as om

    importlib.reload(om)
    return om, tmp_path


def test_hash_is_stable_across_processes():
    a = content_hash(make_oracle_input(nominal_theta(5)))
    b = content_hash(make_oracle_input(nominal_theta(5)))
    assert a == b and len(a) == 64


def test_hash_survives_a_json_round_trip():
    """Floats are rounded to 12 significant digits before hashing. Without that, a
    theta that came back from JSON one ULP off would miss its own cache entry and
    Tier C would silently recompute instead of replaying."""
    theta = nominal_theta(5)
    direct = content_hash(make_oracle_input(theta))
    roundtripped = content_hash(make_oracle_input(np.array(json.loads(json.dumps(theta.tolist())))))
    assert direct == roundtripped


def test_hash_changes_with_theta_and_with_the_grid():
    base = make_oracle_input(nominal_theta(5))
    moved = make_oracle_input(nominal_theta(5) + np.array([1e-6, 0, 0, 0, 0]))
    assert content_hash(base) != content_hash(moved)
    other_grid = make_oracle_input(nominal_theta(5), np.linspace(-1.0, 1.0, 96))
    assert content_hash(base) != content_hash(other_grid)


def test_contract_version_participates_in_the_key():
    """Bumping the contract must invalidate the cache, not silently mix generations."""
    import diffsilicon.shared.cache as cm

    inp = make_oracle_input(nominal_theta(5))
    before = content_hash(inp)
    cm.CONTRACT_VERSION = str(int(CONTRACT_VERSION) + 1)
    try:
        assert content_hash(inp) != before
    finally:
        cm.CONTRACT_VERSION = CONTRACT_VERSION


def test_cache_populates_on_every_run(isolated):
    om, tmp = isolated
    store = CacheStore("mock", root=tmp / "cache")
    assert len(store) == 0
    om.run_oracle(make_oracle_input(nominal_theta(5)), "mock")
    assert len(store) == 1
    om.run_oracle(make_oracle_input(nominal_theta(3)), "mock")
    assert len(store) == 2


def test_cached_result_is_returned_bit_for_bit(isolated):
    om, _ = isolated
    inp = make_oracle_input(nominal_theta(5))
    first = om.run_oracle(inp, "mock")
    second = om.run_oracle(inp, "mock")
    for k in ("ss", "vth_fwd", "vth_rev", "i_leak", "g_lo", "g_hi", "dg_dvth"):
        assert float(getattr(first, k)) == float(getattr(second, k))
    assert np.array_equal(np.asarray(first.id_vg), np.asarray(second.id_vg))


def test_replay_backend_serves_the_cache_with_no_solver(isolated):
    om, tmp = isolated
    inp = make_oracle_input(nominal_theta(5))
    truth = om.run_oracle(inp, "mock")
    os.environ["ORACLE_REPLAY_SOURCE"] = "mock"
    try:
        replayed = om.run_oracle(inp, "replay")
    finally:
        del os.environ["ORACLE_REPLAY_SOURCE"]
    assert float(replayed.ss) == float(truth.ss)
    assert np.array_equal(np.asarray(replayed.id_vg), np.asarray(truth.id_vg))


def test_replay_refuses_to_invent_a_point_it_never_saw(isolated):
    """The replay cache is a reproduction path, not a surrogate. Asking it for a
    design point nobody ever evaluated must be an error, never an interpolation."""
    om, _ = isolated
    os.environ["ORACLE_REPLAY_SOURCE"] = "mock"
    try:
        with pytest.raises(KeyError, match="no cached"):
            om.run_oracle(make_oracle_input(np.full(5, 0.123456)), "replay")
    finally:
        del os.environ["ORACLE_REPLAY_SOURCE"]


def test_provenance_log_records_backend_and_hash_for_every_call(isolated):
    om, tmp = isolated
    log = tmp / "provenance.jsonl"
    inp = make_oracle_input(nominal_theta(5))
    om.run_oracle(inp, "mock")
    om.run_oracle(inp, "mock")  # a cache hit still has to be logged
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    for entry in lines:
        assert entry["backend"] == "mock"
        assert entry["hash"] == content_hash(inp)
        assert entry["converged"] == 1.0


def test_unknown_backend_is_rejected(isolated):
    om, _ = isolated
    with pytest.raises(ValueError, match="expected one of"):
        om.run_oracle(make_oracle_input(nominal_theta(5)), "definitely-not-a-solver")
