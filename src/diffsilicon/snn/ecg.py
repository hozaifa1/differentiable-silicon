"""The MIT-BIH ECG task, imported from the thesis rather than rebuilt.

WHAT THIS IS
------------
2000 delta-encoded heartbeats from the MIT-BIH Arrhythmia Database, in the four
AAMI classes the thesis reports: N (1000), F (250), SVEB (250), VEB (500). Each
beat is 1000 timesteps of a two-channel level-crossing (delta) spike train --
one channel for upward threshold crossings, one for downward -- plus a CUE
channel that is zero during the beat and one during a 116-step tail, which is
what prompts the network to commit to an answer.

    x: (B, 1116, 3)   spikes, {0, 1}      y: (B,)  in {0=N, 1=F, 2=SVEB, 3=VEB}

WHERE IT COMES FROM, AND WHAT IS NOT REDONE HERE
------------------------------------------------
`F:/RESEARCH/FeFET x ML/.../ecg detection/data_ecg`, eight CSVs written by the
thesis' own preprocessing. The loading logic below is that project's
`dataset.py` (`DeltaTransformedECG`) -- the class order, the up/down interleave,
the cue padding and the 116-step tail are all unchanged, because the reported
thesis baseline is measured on exactly this arrangement and re-deriving it would
mean comparing against a different task while quoting the same number.

The CSVs are NOT committed. They are a preprocessing of a public database, this
repository is public, and nothing here needs them at import time. Point
DIFFSILICON_ECG_DIR at them; the first load writes a compact .npz cache under
`results/cache/ecg/` and every later load reads that in milliseconds.

THE SPLIT IS INTRA-PATIENT, AND THAT IS FORCED
-----------------------------------------------
The project's earlier plan called for the inter-patient AAMI DS1/DS2 split (de
Chazal), on the correct general grounds that an intra-patient split shares beats
from one patient across train and test and inflates every number.

That split CANNOT be produced from this data, and not for want of trying: the
curated CSVs are (n_beats, n_timesteps) matrices with the record identity
dropped during preprocessing. There is no column, no index and no side file that
says which patient a row came from, so there is nothing to partition on. The
choice is therefore between the thesis' own protocol and a different dataset --
and a different dataset would mean the thesis baseline no longer applies, which
is the one thing this recalibration exists to prevent.

So: random stratified 1664/336, matching the thesis, seeded and reproducible.
Every accuracy this project reports on ECG is intra-patient and must say so.
Recovering DS1/DS2 means going back to the raw PhysioNet records and re-running
the delta encoder with record IDs kept -- a real piece of work, worth doing, and
not a substitution that can be made quietly.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

__all__ = [
    "ECG_CLASSES",
    "N_IN",
    "N_CLASSES",
    "T_FULL",
    "OUTPUT_CUE_LENGTH",
    "ecg_arrays",
    "ecg_split",
    "ecg_batch",
    "pool_time",
]

# The thesis' class order, index = label. Do not sort it: its confusion matrices
# and per-class F1 are reported in this order.
ECG_CLASSES = ("N", "F", "SVEB", "VEB")
N_CLASSES = 4
N_IN = 3  # up, down, cue
OUTPUT_CUE_LENGTH = 116  # the value in its `make_ecg_figures.py` / `post_quantize.py`
T_RAW = 1000
T_FULL = T_RAW + OUTPUT_CUE_LENGTH  # 1116

_DEFAULT_DIR = (
    r"F:/RESEARCH/FeFET x ML/TCAD Files/GAAFet/Simulations/"
    r"Python_LK_and_SNN/ecg detection/data_ecg"
)
_CACHE = Path(__file__).resolve().parents[3] / "results" / "cache" / "ecg" / "mitbih_2000.npz"


def _source_dir() -> Path:
    return Path(os.environ.get("DIFFSILICON_ECG_DIR", _DEFAULT_DIR))


def _build_from_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """`DeltaTransformedECG.__init__` + `.preprocess`, in numpy.

    Kept structurally identical to the thesis' `dataset.py` so the arrays are the
    same ones it trained on: per class, read `up/up_<cat>_guiyi.csv` and
    `down/down_<cat>_guiyi.csv`, stack them as the last axis in (up, down) order,
    concatenate the classes in ECG_CLASSES order, then append the cue.
    """
    import pandas as pd  # only needed on the cold path

    blocks, labels = [], []
    for index, cat in enumerate(ECG_CLASSES):
        up = pd.read_csv(path / "up" / f"up_{cat}_guiyi.csv", header=None).to_numpy()
        down = pd.read_csv(path / "down" / f"down_{cat}_guiyi.csv", header=None).to_numpy()
        if up.shape != down.shape:
            raise ValueError(f"{cat}: up {up.shape} != down {down.shape}")
        # (n, times) x2 -> (n, times, 2), up first. Its axes=[1, 2, 0].
        blocks.append(np.transpose(np.asarray([up, down]), axes=[1, 2, 0]))
        labels.append(np.full(up.shape[0], index, dtype=np.int64))

    data = np.concatenate(blocks, axis=0)
    y = np.concatenate(labels, axis=0)

    # The cue: pad `OUTPUT_CUE_LENGTH` silent steps onto the signal channels,
    # then add a third channel that is 0 over the beat and 1 over the pad.
    n, times, n_sig = data.shape
    data = np.concatenate([data, np.zeros((n, OUTPUT_CUE_LENGTH, n_sig))], axis=1)
    cue = np.concatenate([np.zeros((n, times, 1)), np.ones((n, OUTPUT_CUE_LENGTH, 1))], axis=1)
    data = np.concatenate([data, cue], axis=-1)
    return data.astype(np.float32), y


def ecg_arrays() -> tuple[np.ndarray, np.ndarray]:
    """(x, y) for all 2000 beats. Cached to .npz after the first build."""
    if _CACHE.is_file():
        z = np.load(_CACHE)
        return z["x"], z["y"]

    src = _source_dir()
    if not (src / "up").is_dir():
        raise FileNotFoundError(
            f"MIT-BIH delta-encoded CSVs not found under {src}. This project does not "
            f"ship them -- they are the thesis' own preprocessing and this repository is "
            f"public. Set DIFFSILICON_ECG_DIR to the folder containing up/ and down/."
        )
    x, y = _build_from_csv(src)
    if x.shape != (2000, T_FULL, N_IN):
        raise ValueError(f"expected (2000, {T_FULL}, {N_IN}) beats, got {x.shape}")
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_CACHE, x=x, y=y)
    return x, y


def ecg_split(seed: int = 0, n_test: int = 336) -> dict[str, np.ndarray]:
    """Stratified random 1664/336 split -- the thesis protocol. INTRA-PATIENT.

    Stratified, not plain random: at 336 test beats a plain shuffle can leave F
    or SVEB (250 beats each, 12.5% of the set) with a handful of examples, and
    macro-F1 over four classes is then dominated by the sampling of the two
    smallest. Stratifying holds the class proportions of the full set in both
    halves, which is what makes two runs at different seeds comparable.
    """
    x, y = ecg_arrays()
    rng = np.random.default_rng(seed)
    frac = n_test / len(y)
    test_idx = []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        test_idx.append(idx[: int(round(frac * idx.size))])
    test = np.concatenate(test_idx)
    train = np.setdiff1d(np.arange(len(y)), test)
    rng.shuffle(train)
    rng.shuffle(test)
    return {"x": x, "y": y, "train": train, "test": test}


def pool_time(x, pool: int):
    """Sum spikes into windows of `pool` steps. `pool=1` is a no-op.

    The beat is 1116 steps and the flagship pays one inner training run PER
    DESIGN POINT, so the full-length sequence is the single largest cost in the
    loop. Summing is the right reduction rather than sampling: the delta code is
    sparse (about 0.04 spikes per step per channel), so a window of 8 carries
    ~0.3 spikes and pooling preserves the rate the LIF integrates, where
    striding would throw most of the signal away.

    Every channel including the cue is scaled identically, so the network sees
    the same relative drive it had before.
    """
    if pool <= 1:
        return x
    t = x.shape[1] - (x.shape[1] % pool)
    if isinstance(x, torch.Tensor):
        return x[:, :t].reshape(x.shape[0], t // pool, pool, x.shape[2]).sum(dim=2)
    return x[:, :t].reshape(x.shape[0], t // pool, pool, x.shape[2]).sum(axis=2)


_SPLIT_CACHE: dict[int, dict] = {}


def ecg_batch(
    batch: int = 64,
    seed: int = 0,
    split: str = "train",
    pool: int = 1,
    dtype=torch.float64,
):
    """A fixed, reproducible batch. Drop-in replacement for `synthetic_batch`.

    Class-balanced by construction, and FIXED for a given seed. Both matter to
    the flagship for the same reason the batch seed is frozen across steps: the
    optimiser compares losses between design points, so anything that resamples
    between calls turns rho into noise. Balancing is on top of -- not instead of
    -- `balanced_ce`: the weighting fixes the objective, and drawing every class
    into a 64-beat batch is what stops F and SVEB from being absent from it.
    """
    if seed not in _SPLIT_CACHE:
        _SPLIT_CACHE[seed] = ecg_split(seed)
    s = _SPLIT_CACHE[seed]
    idx_pool = s[split]
    y_all = s["y"]

    rng = np.random.default_rng(seed + 1)
    per = max(1, batch // N_CLASSES)
    take = []
    for c in range(N_CLASSES):
        idx = idx_pool[y_all[idx_pool] == c]
        take.append(rng.choice(idx, size=per, replace=idx.size < per))
    take = np.concatenate(take)[:batch]
    rng.shuffle(take)

    x = torch.as_tensor(s["x"][take], dtype=dtype)
    y = torch.as_tensor(y_all[take], dtype=torch.long)
    return pool_time(x, pool), y
