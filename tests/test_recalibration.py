"""The D3 recalibration, held in place.

Five things were realigned with the thesis on 2026-08-24, and each of them is
the kind of change that decays quietly: a locked constant drifts back into a
design vector, a dataset gets swapped for a convenient one, an architecture
number gets "tidied". These tests are the ratchet.
"""

import os

import numpy as np
import pytest
import torch

from diffsilicon.shared.design import (
    LOCKED_MATERIAL,
    get_design,
    nominal_theta,
    tunes_locked_material,
)
from diffsilicon.shared.material import EC_MV_CM, HZO_CALIBRATION, PR_UC_CM2, PS_UC_CM2

# ---------------------------------------------------------------- 1. the lock


def test_the_locked_constants_are_the_calibrated_ones():
    """cal_n16, the node the thesis' hysteresis calibration locked.

    HYSTERESIS_CALIBRATION_RESULT.md, "Calibrated parameters & the real-world
    physics extracted": FE: P_r, P_s, F_c = 32, 40 uC/cm^2, 1.4 MV/cm.
    """
    assert PR_UC_CM2 == 32.0
    assert PS_UC_CM2 == 40.0
    assert EC_MV_CM == 1.4
    assert HZO_CALIBRATION["node"] == "cal_n16"


def test_the_design_vector_is_the_fabrication_knobs():
    spec = get_design(4)
    assert spec.names == ("t_fe", "L_g", "log10_N_ch", "t_IL")
    assert tunes_locked_material(4) == ()


@pytest.mark.parametrize("d", [3, 5, 12])
def test_legacy_vectors_are_marked_not_silently_kept(d):
    """They are kept so pre-recalibration results can still be replayed.

    Kept, but marked -- the whole failure mode this guards against is a legacy
    vector quietly remaining the default and Pr drifting back into the search.
    """
    assert set(tunes_locked_material(d)) == set(LOCKED_MATERIAL)


@pytest.mark.parametrize("d", [3, 5, 12])
def test_the_optimiser_refuses_to_tune_the_material(d):
    from diffsilicon.optimise import FlagshipConfig, run_flagship

    with pytest.raises(ValueError, match="LOCKED material"):
        run_flagship(FlagshipConfig(d=d, backend="mock"))


def test_both_oracles_see_the_locked_material_at_d4():
    """The lock is worth nothing if only one solver honours it.

    T1 and T2 must describe the same film, or every cross-solver comparison is
    measuring the disagreement between two different materials.
    """
    import jax.numpy as jnp

    from diffsilicon.oracle_devsim import fefet_params
    from diffsilicon.shared.mock_device import device_params

    p = fefet_params(nominal_theta(4))
    assert p.pr * 1e6 == pytest.approx(PR_UC_CM2)
    assert p.ps * 1e6 == pytest.approx(PS_UC_CM2)
    assert p.ec / 1e6 == pytest.approx(EC_MV_CM)

    d = device_params(jnp.asarray(nominal_theta(4)))
    assert float(d["Pr"]) == pytest.approx(PR_UC_CM2)
    assert float(d["Ec"]) == pytest.approx(EC_MV_CM)


def test_the_nominal_film_is_the_calibrated_thickness():
    """7 nm: the mesh calibration.local.json builds, and the thesis' own choice.

    It also makes the Sentaurus fixed-slab remap exact rather than approximate at
    the nominal point.
    """
    from diffsilicon.shared.design import denormalise

    phys = denormalise(nominal_theta(4), get_design(4))
    assert phys[0] == pytest.approx(7.0)


def test_legacy_results_can_still_be_reproduced():
    """The escape hatch exists and is explicit."""
    from diffsilicon.shared.design import DESIGN_VECTORS

    assert 3 in DESIGN_VECTORS and 5 in DESIGN_VECTORS and 12 in DESIGN_VECTORS


# ------------------------------------------------------------ 2. the ECG task

pytest.importorskip("pandas")


def _have_ecg():
    from diffsilicon.snn.ecg import _CACHE, _source_dir

    return _CACHE.is_file() or (_source_dir() / "up").is_dir()


needs_ecg = pytest.mark.skipif(
    not _have_ecg(),
    reason="MIT-BIH CSVs not on this machine; set DIFFSILICON_ECG_DIR",
)


@needs_ecg
def test_the_dataset_is_the_curated_2000_beats():
    from diffsilicon.snn.ecg import ECG_CLASSES, N_IN, T_FULL, ecg_arrays

    x, y = ecg_arrays()
    assert x.shape == (2000, T_FULL, N_IN)
    assert ECG_CLASSES == ("N", "F", "SVEB", "VEB")
    # The thesis' own class counts: N/VEB/SVEB/F = 1000/500/250/250.
    assert [int((y == c).sum()) for c in range(4)] == [1000, 250, 250, 500]
    assert set(np.unique(x[:, :1000, :2])) <= {0.0, 1.0}, "the delta code is binary"


@needs_ecg
def test_the_cue_channel_is_the_thesis_arrangement():
    """Zero over the beat, one over the 116-step tail. The tail is silent."""
    from diffsilicon.snn.ecg import OUTPUT_CUE_LENGTH, T_RAW, ecg_arrays

    x, _ = ecg_arrays()
    assert x[:, :T_RAW, 2].max() == 0.0
    assert x[:, T_RAW:, 2].min() == 1.0
    assert x[:, T_RAW:, :2].max() == 0.0
    assert x.shape[1] - T_RAW == OUTPUT_CUE_LENGTH


@needs_ecg
def test_the_split_is_1664_336_and_stratified():
    from diffsilicon.snn.ecg import ecg_split

    s = ecg_split(seed=0)
    assert s["train"].size == 1664
    assert s["test"].size == 336
    assert not set(s["train"]) & set(s["test"]), "train and test overlap"
    # Stratified: every class present in test, in roughly its overall proportion.
    for c, n_all in enumerate([1000, 250, 250, 500]):
        n_test = int((s["y"][s["test"]] == c).sum())
        assert n_test == pytest.approx(336 * n_all / 2000, abs=1)


@needs_ecg
def test_pooling_conserves_spikes():
    """Pooling is a sum, so it must move no spikes -- only their resolution."""
    from diffsilicon.snn.ecg import ecg_batch

    x1, y1 = ecg_batch(8, seed=0, pool=1)
    x10, y10 = ecg_batch(8, seed=0, pool=10)
    assert torch.equal(y1, y10), "pooling changed which beats were drawn"
    # 1116 is not divisible by 10, so the last 6 raw steps are dropped.
    assert x10.shape[1] == 111
    assert float(x10.sum()) == pytest.approx(float(x1[:, :1110].sum()))


@needs_ecg
def test_batches_are_class_balanced_and_reproducible():
    from diffsilicon.snn.ecg import ecg_batch

    x_a, y_a = ecg_batch(16, seed=0, pool=10)
    x_b, y_b = ecg_batch(16, seed=0, pool=10)
    assert torch.equal(x_a, x_b) and torch.equal(y_a, y_b), "batch is not deterministic"
    assert y_a.bincount(minlength=4).tolist() == [4, 4, 4, 4]


# ----------------------------------------------------------- 3. the thesis LSNN


def test_the_architecture_is_the_thesis_baseline():
    from diffsilicon.snn.lsnn import THESIS_LSNN, LSNNNet

    assert THESIS_LSNN["n_lif"] == 100
    assert THESIS_LSNN["n_alif"] == 60
    assert THESIS_LSNN["max_delay"] == 10
    assert THESIS_LSNN["n_in"] == 3
    assert THESIS_LSNN["n_out"] == 4

    net = LSNNNet()
    assert net.n_hidden == 160
    assert net.fc1.max_delay == 10 and net.rc.max_delay == 10
    assert net.rc.n_in == net.rc.n_out == 160


def test_the_recurrent_layer_has_no_self_connections():
    from diffsilicon.snn.lsnn import LSNNNet

    net = LSNNNet()
    diag = torch.diagonal(net.rc.delay_mask.sum(dim=2))
    assert float(diag.abs().max()) == 0.0


def test_every_connection_has_exactly_one_delay():
    """Ten taps per connection, one of them programmed. Not a dense tensor."""
    from diffsilicon.snn.lsnn import LSNNNet

    net = LSNNNet()
    per_connection = net.fc1.delay_mask.sum(dim=2)
    assert torch.all(per_connection == 1.0)
    assert net.fc1.delay_mask.sum(dim=(0, 1)).min() > 0, "some delay tap is never used"


def _phi(**over):
    p = {"beta": 0.6033, "g_min": 2.6e-5, "g_max": 2.0e-4, "th_th": 5.0, "sig_w": 0.05}
    p.update(over)
    return {k: torch.tensor(float(v), dtype=torch.float64, requires_grad=True)
            for k, v in p.items()}


@needs_ecg
def test_the_network_actually_fires_on_real_beats():
    """THE dead-network trap, on the new architecture and the new data.

    At th_th = 20 on the old 16-input layer, not one neuron ever fired: the loss
    sat at exactly ln(4) for every theta and jax.grad still returned smooth,
    plausible, meaningless numbers, because a surrogate gradient does not care
    whether a spike happened. Widening the fan-in to 160 and swapping the data
    is exactly the change that could resurrect it.
    """
    from diffsilicon.snn.ecg import ecg_batch
    from diffsilicon.snn.lsnn import THESIS_LSNN, LSNNNet

    net = LSNNNet(dt_s=THESIS_LSNN["dt_s"] * 10)
    x, _ = ecg_batch(8, seed=0, pool=10)
    with torch.no_grad():
        _, spikes = net(x, _phi())
    assert float(spikes) > 0.01, (
        "no neuron fired: the device-to-classifier channel is dead and every "
        "gradient below is meaningless"
    )
    assert float(spikes) < 1.0, "every neuron fires every step; the threshold is inert"


@needs_ecg
def test_the_loss_moves_with_the_device():
    """A device the network cannot use must cost more than one it can.

    The other half of the dead-network trap: a loss that is constant in theta
    still yields a smooth-looking gradient.
    """
    from diffsilicon.snn.ecg import ecg_batch
    from diffsilicon.snn.lif import balanced_ce
    from diffsilicon.snn.lsnn import THESIS_LSNN, LSNNNet

    net = LSNNNet(dt_s=THESIS_LSNN["dt_s"] * 10)
    x, y = ecg_batch(16, seed=0, pool=10)
    with torch.no_grad():
        good, _ = net(x, _phi())
        dead, _ = net(x, _phi(th_th=400.0))
    assert float(balanced_ce(dead, y, 4)) != pytest.approx(float(balanced_ce(good, y, 4)))


@needs_ecg
def test_the_gradient_reaches_the_device_parameters():
    """The VJP endpoint is real: dL/dphi is a tensor gradient, not a stand-in.

    g_min and g_max are excluded on purpose -- they cancel exactly in
    `LIFNet._weights` and are ~1e-13 BY CONSTRUCTION, which is correct and
    documented, not a detached gradient.
    """
    from diffsilicon.snn.ecg import ecg_batch
    from diffsilicon.snn.lif import balanced_ce
    from diffsilicon.snn.lsnn import THESIS_LSNN, LSNNNet

    net = LSNNNet(dt_s=THESIS_LSNN["dt_s"] * 10)
    x, y = ecg_batch(8, seed=0, pool=10)
    phi = _phi()
    logits, _ = net(x, phi)
    balanced_ce(logits, y, 4).backward()
    for k in ("beta", "th_th", "sig_w"):
        g = phi[k].grad
        assert g is not None and np.isfinite(float(g)), f"dL/d{k} is {g}"
        assert abs(float(g)) > 1e-9, f"dL/d{k} vanished ({float(g):.3e})"


def test_the_pooled_timestep_matches_the_frozen_circuit_timebase():
    """Why the pool is 10 and not a round number picked for speed.

    A beat is 1116 steps of 0.5556 ms. Ten of them is 5.556 ms, which is the
    frozen dt_bio of config/circuit.yaml to 1.2% -- so V_leak, K_syn and th_th
    keep meaning what they meant. And at that timestep the thesis' membrane
    decay and this project's device-derived beta agree to half a percent, which
    is the alignment, independently arrived at from an RC fit and from a
    subthreshold leak.
    """
    from diffsilicon.shared.circuit import load_circuit
    from diffsilicon.snn.lsnn import THESIS_LSNN, decay

    cc = load_circuit()
    dt_pooled = THESIS_LSNN["dt_s"] * 10
    assert dt_pooled == pytest.approx(5.556e-3, rel=1e-3)
    assert dt_pooled == pytest.approx(cc.dt_hw * cc.accel, rel=0.02)

    thesis_beta = decay(dt_pooled, THESIS_LSNN["tau_mem_s"])
    assert thesis_beta == pytest.approx(0.6033, rel=0.01), (
        f"the thesis membrane decay is {thesis_beta:.4f}; the device's nominal "
        "beta is 0.6033. If these have drifted apart the timebases no longer agree."
    )


@needs_ecg
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="slow; runs a training loop")
def test_the_t4_tesseract_runs_the_real_task():
    """End to end through the served API, on real data, with a real gradient."""
    import importlib.util
    from pathlib import Path

    api = Path(__file__).resolve().parents[1] / "tesseracts" / "snn-lif-ecg" / "tesseract_api.py"
    # conftest pins the session to the synthetic task so CI can run without the
    # dataset. This test is the one that must not be pinned, and a fresh module
    # object is loaded below, so the override takes effect.
    os.environ["SNN_TASK"] = "ecg"
    os.environ["SNN_TRAIN_STEPS"] = "2"
    spec = importlib.util.spec_from_file_location("t4_recal", api)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
        assert m.TASK == "ecg" and m.N_IN == 3 and m.N_HIDDEN == 160
        inp = m.InputSchema(
            beta=0.6033, g_min=2.6e-5, g_max=2.0e-4, th_th=5.0, sig_w=0.05, batch=8
        )
        out = m.apply(inp)
        assert np.isfinite(out.loss) and out.spikes > 0.0
        g = m.vector_jacobian_product(inp, set(m.PHI_KEYS), {"loss"}, {"loss": 1.0})
        assert abs(g["beta"]) > 0.0 and abs(g["th_th"]) > 0.0
    finally:
        os.environ["SNN_TASK"] = "synth"
