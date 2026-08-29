#!/usr/bin/env python
"""Figure 4: what the classifier does differently on the device the optimiser found.

    python scripts/fig4_spike_raster.py

Figure 3 shows the device moving. This shows the consequence -- the thing the
whole apparatus exists to change. Same network, same weights, same 16 MIT-BIH
beats, same seed; the ONLY difference between the two columns is the five numbers
the FeFET hands up, taken from the flagship run's own `phi_initial` and
`phi_final`:

    start   beta 0.6137  g_min 6.85e-07  g_max 1.21e-04  th_th 5.622  sig_w 0.1158
    final   beta 0.6005  g_min 4.31e-06  g_max 2.81e-04  th_th 4.611  sig_w 0.1016
                                                          ^^^^^^^^^^
th_th is spikes-to-fire at maximum weight and goes as 1/g_max, so widening the
memory window -- which is what Figure 3 shows the optimiser buying -- lowers the
bar the membrane has to clear.

WHAT THIS FIGURE MEASURED, AND IT IS NOT WHAT THE OBVIOUS VERSION WOULD HAVE
SHOWN. The first draft drew two rasters side by side and expected a quiet layer
to become an active one. Measured, that is not what happens, and the two rasters
are indistinguishable by eye:

    population rate           0.4344  ->  0.4572   (+5%)
    per-neuron rate profile   correlation 0.9999
    individual spikes that move                      8.9%

The layer does not fire more. It fires DIFFERENTLY, by about one spike in eleven,
and all 160 neurons shift a little. Two dense blobs would have shown nothing, and
a figure that shows nothing under a caption claiming a transformation is worse
than no figure. Panel (a) draws the DIFFERENCE instead.

AND THE POINT OF THE 8.9% IS PANEL (b). On the start device the network predicts
class F for ALL SIXTEEN BEATS. Its accuracy of 0.250 is exactly the score of a
constant answer, and the four "correct" beats are the four that happen to be F.
It is not a weak classifier, it is a collapsed one. On the final device it
predicts four different classes and gets 11 of 16. Moving one spike in eleven is
what separates a readout stuck on one output from a working one -- which is a
sharper claim than "the layer became active", and it has the advantage of being
what the numbers say.

Panels:

(a) the difference raster, one beat of each of the four AAMI classes: spikes
    common to both devices in grey, start-only in blue, final-only in green.
(b) the prediction strip -- what each device answered for each of the 16 beats,
    against the truth. This is where the collapse is visible.
(c) per-class accuracy, which is where the class-balanced objective shows up:
    the gain is spread, and it is not the majority class.

NOTHING HERE IS TRAINED. The network is the shared W0, built through
`tesseract_api._net` under SNN_TRAIN_MODE=frozen -- which is what every banked
result in this project was produced under. Both columns are checked against
`ce_initial` and `ce_final` of the flagship JSON to nine decimals before a single
dot is drawn, and the script refuses to draw the figure if they drift.

DEVSIM AND PYTORCH CANNOT SHARE A PROCESS (OMP #15). Nothing here imports the
oracle: phi comes out of the banked flagship JSON, which is where the solver
already put it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

# Must be set before snn-lif-ecg is imported -- it reads these at import time.
os.environ.setdefault("SNN_TASK", "ecg")
# frozen, NOT the module's own default of adapt. Every banked flagship number was
# produced under frozen, and the check in main() fails loudly if this figure ever
# stops reproducing them. The same discrepancy was found here on D6 and fixed at
# the source -- see FlagshipConfig.train_mode.
os.environ.setdefault("SNN_TRAIN_MODE", "frozen")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from diffsilicon.snn.lif import DTYPE, PHI_KEYS, balanced_ce  # noqa: E402

RESULT = _REPO / "results" / "runs" / "flagship-d4-fixed" / "result.json"
API = _REPO / "tesseracts" / "snn-lif-ecg" / "tesseract_api.py"
OUT = _REPO / "docs" / "figures" / "fig4_spike_raster.png"

CLASS_SHORT = ("N", "S", "V", "F")
CLASS_NAME = ("N  normal", "S  supraventricular", "V  ventricular", "F  fusion")
C_START, C_FINAL, C_BOTH = "#3b3b6d", "#1b7837", "#c8c8c8"


def load_api():
    """Import the T4 Tesseract module by path, as `scripts/smoke_tesseracts.py` does."""
    spec = importlib.util.spec_from_file_location("_fig4_snn", str(API))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(mod, phi_d: dict, batch: int, seed: int) -> dict:
    """One forward pass of the network this design point is scored on.

    Built through `mod._net`, which is the same call `apply` makes. Not a
    re-implementation of the network -- if it were, it could agree with the
    flagship by luck rather than by construction.
    """
    inputs = mod.InputSchema(**{k: float(phi_d[k]) for k in PHI_KEYS},
                             seed=seed, batch=batch, smooth_spikes=False)
    net = mod._net(inputs)
    for prm in net.parameters():
        prm.requires_grad_(False)
    x, y = mod._batch(inputs)
    phi = {k: torch.tensor(float(phi_d[k]), dtype=DTYPE) for k in PHI_KEYS}
    log: list = []
    with torch.no_grad():
        logits, spikes = net(x, phi, seed=seed, smooth=False, spike_log=log)
        loss = balanced_ce(logits, y, mod.N_CLASSES)
    s = torch.stack(log, dim=1).numpy().astype(bool)  # (B, T, H)
    pred = logits.argmax(1).numpy()
    return {"spikes": s, "y": y.numpy(), "pred": pred, "logits": logits.numpy(),
            "loss": float(loss), "accuracy": float((pred == y.numpy()).mean()),
            "rate": float(spikes)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    flag = json.loads(RESULT.read_text(encoding="utf-8"))
    mod = load_api()

    print(f"task={mod.TASK}  train_mode={mod.TRAIN_MODE}  "
          f"batch={args.batch}  seed={args.seed}")
    runs = {}
    for tag, key in (("start", "phi_initial"), ("final", "phi_final")):
        print(f"\n{tag}: " + "  ".join(f"{k}={flag[key][k]:.6g}" for k in PHI_KEYS))
        runs[tag] = run(mod, flag[key], args.batch, args.seed)
        r = runs[tag]
        print(f"  loss {r['loss']:.9f}  accuracy {r['accuracy']:.4f}  "
              f"mean rate {r['rate']:.6f}  "
              f"raster {'x'.join(map(str, r['spikes'].shape))}")

    # The banked flagship numbers this must land on, checked rather than assumed.
    for tag, suffix in (("start", "initial"), ("final", "final")):
        got = runs[tag]
        want_ce = flag[f"ce_{suffix}"]          # objective_* carries the energy term
        want_a = flag[f"accuracy_{suffix}"]
        dl, da = abs(got["loss"] - want_ce), abs(got["accuracy"] - want_a)
        ok = dl < 1e-9 and da < 1e-12
        print(f"reproduces banked {tag}: loss {got['loss']:.9f} vs {want_ce:.9f} "
              f"(d={dl:.1e}), accuracy {got['accuracy']:.4f} vs {want_a:.4f} "
              f"(d={da:.1e})  {'OK' if ok else 'DRIFT'}")
        if not ok:
            raise SystemExit(
                f"{tag}: this figure no longer reproduces the flagship it claims "
                f"to illustrate. Refusing to draw it.")

    A, B = runs["start"]["spikes"], runs["final"]["spikes"]
    y = runs["start"]["y"]
    n_cls = mod.N_CLASSES
    n_beats, t_steps, n_hidden = A.shape
    moved = float((A != B).mean())
    rate_corr = float(np.corrcoef(A.mean((0, 1)), B.mean((0, 1)))[0, 1])
    print(f"\nspikes that move: {moved * 100:.2f}%   "
          f"per-neuron rate correlation {rate_corr:.4f}   "
          f"population rate {A.mean():.4f} -> {B.mean():.4f}")
    for tag in ("start", "final"):
        p = runs[tag]["pred"]
        print(f"{tag} predictions: {''.join(CLASS_SHORT[c] for c in p)}"
              f"{'   <- one class for every beat' if len(set(p.tolist())) == 1 else ''}")
    print(f"truth:             {''.join(CLASS_SHORT[c] for c in y)}")

    picks = [int(np.where(y == c)[0][0]) if np.any(y == c) else None
             for c in range(n_cls)]

    # --- figure ---------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.linewidth": 0.8, "axes.labelsize": 9.5, "axes.titlesize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "figure.dpi": 130, "savefig.dpi": 300, "axes.axisbelow": True,
    })
    fig = plt.figure(figsize=(12.8, 4.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.15, 0.80],
                          wspace=0.27, left=0.062, right=0.985,
                          bottom=0.265, top=0.855)
    axA, axB, axC = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
                     fig.add_subplot(gs[2]))

    # (a) the difference raster ------------------------------------------------
    gap = 14
    for k, bi in enumerate(picks):
        if bi is None:
            continue
        base = k * (n_hidden + gap)
        axA.axhspan(base - 1, base + n_hidden + 1, color="0.975", zorder=0)
        a, b = A[bi], B[bi]
        for mask, colour, ms, z in ((a & b, C_BOTH, 1.1, 2),
                                    (a & ~b, C_START, 2.0, 4),
                                    (~a & b, C_FINAL, 2.0, 3)):
            tt, nn = np.nonzero(mask)
            axA.plot(tt, base + nn, ls="none", marker=".", ms=ms, color=colour,
                     alpha=0.9, zorder=z, rasterized=True)
        # Short labels, horizontal. The full class names are long enough that
        # rotated they run into each other between bands.
        axA.text(-4.0, base + n_hidden * 0.5, CLASS_SHORT[k], va="center",
                 ha="center", fontsize=9.0, color="0.35", weight="bold")
    axA.set_xlim(-15, t_steps + 1)
    axA.set_ylim(-6, n_cls * (n_hidden + gap) - gap + 6)
    axA.set_yticks([])
    axA.set_xlabel("timestep  (5.556 ms each)")
    axA.set_ylabel(f"recurrent layer, {n_hidden} neurons "
                   f"({mod.THESIS_LSNN['n_lif']} LIF + {mod.THESIS_LSNN['n_alif']} "
                   f"adaptive)", fontsize=8.4)
    axA.set_title(f"(a)  one spike in eleven moves -- {moved * 100:.1f}% of the "
                  f"raster; one beat per class",
                  loc="left", fontsize=9.6)
    axA.grid(axis="x", lw=0.4, color="0.9", zorder=1)
    axA.legend(handles=[
        Line2D([], [], ls="none", marker=".", ms=6, color=C_BOTH, label="both"),
        Line2D([], [], ls="none", marker=".", ms=8, color=C_START, label="start only"),
        Line2D([], [], ls="none", marker=".", ms=8, color=C_FINAL, label="final only"),
    ], loc="upper left", bbox_to_anchor=(0.12, 1.0), frameon=True,
        framealpha=0.94, fontsize=7.6, handletextpad=0.15, borderpad=0.4,
        ncol=3, columnspacing=0.8)

    # (b) the prediction strip -------------------------------------------------
    # Three sub-rows inside each class band: truth, then the two answers. Drawn
    # on top of one another they hide each other, and the row that gets hidden is
    # the one the reader needs to compare against.
    h = 0.17
    for ri, (label, vals, colour) in enumerate((("truth", y, "0.30"),
                                                ("start", runs["start"]["pred"], C_START),
                                                ("final", runs["final"]["pred"], C_FINAL))):
        dy = (ri - 1) * 0.21
        for bi, c in enumerate(vals):
            hit = label == "truth" or c == y[bi]
            axB.add_patch(plt.Rectangle((bi - 0.40, c + dy - h / 2), 0.80, h,
                                        facecolor=colour if hit else "white",
                                        edgecolor=colour, linewidth=0.8, zorder=5))
    for ri, label in enumerate(("truth", "start", "final")):
        axB.text(-0.62, (ri - 1) * 0.21, label, fontsize=6.4, va="center",
                 ha="right", color="0.45")
    axB.set_xlim(-1.9, n_beats - 0.3)
    axB.set_ylim(n_cls - 0.55, -0.55)
    axB.set_yticks(range(n_cls))
    axB.set_yticklabels(CLASS_NAME, fontsize=8)
    axB.set_xticks(range(0, n_beats, 2))
    axB.set_xlabel("beat in the fixed batch of 16")
    axB.set_title("(b)  the start device answers F for every beat", loc="left",
                  fontsize=9.6)
    axB.grid(axis="y", lw=0.4, color="0.92", zorder=0)
    # No legend: the three row labels inside each band already name the rows,
    # and a legend here lands on the caption. Only the hollow convention needs
    # saying, and it says itself in four words.
    axB.text(0.99, 0.985, "filled = right,  hollow = wrong",
             transform=axB.transAxes, ha="right", va="top", fontsize=7.2,
             color="0.45")

    # (c) per-class accuracy ---------------------------------------------------
    width, idx = 0.38, np.arange(n_cls)
    for off, (tag, colour) in zip((-width / 2, width / 2),
                                  (("start", C_START), ("final", C_FINAL)),
                                  strict=True):
        r = runs[tag]
        acc = np.array([float((r["pred"][y == c] == c).mean()) if np.any(y == c)
                        else np.nan for c in range(n_cls)])
        axC.barh(idx + off, acc, height=width, color=colour, alpha=0.9,
                 edgecolor="white", linewidth=0.6, label=tag, zorder=3)
        # A bar of height zero is invisible, and three of the start bars ARE
        # zero -- which is the finding. Label them so they cannot be read as
        # missing data.
        for c, v in zip(idx, acc, strict=True):
            if v == 0.0:
                axC.text(0.012, c + off, "0", va="center", ha="left",
                         fontsize=7.2, color=colour, zorder=5)
    counts = [int((y == c).sum()) for c in range(n_cls)]
    axC.set_yticks(idx)
    axC.set_yticklabels([f"{CLASS_SHORT[c]}  (n={counts[c]})" for c in range(n_cls)],
                        fontsize=8)
    axC.invert_yaxis()
    axC.set_xlim(0, 1.06)
    axC.set_xlabel("per-class accuracy")
    axC.set_title("(c)  balanced, not the\n      majority class", loc="left",
                  fontsize=9.6)
    axC.grid(axis="x", lw=0.4, color="0.9", zorder=0)
    axC.legend(loc="lower right", frameon=True, framealpha=0.94, fontsize=7.6)

    caption = (
        "Same weights, same 16 MIT-BIH beats, same seed. The only difference is the "
        "five numbers the FeFET hands up -- th_th "
        f"{flag['phi_initial']['th_th']:.3f} $\\to$ {flag['phi_final']['th_th']:.3f} "
        "as the memory window widens.\n"
        f"The population rate barely moves ({A.mean():.4f} $\\to$ {B.mean():.4f}) and "
        f"the per-neuron rate profile is unchanged (r = {rate_corr:.4f}); what moves "
        "is WHICH spikes, and that is enough to unstick the readout.\n"
        f"Loss {flag['ce_initial']:.4f} $\\to$ {flag['ce_final']:.4f}, accuracy "
        f"{flag['accuracy_initial']:.3f} $\\to$ {flag['accuracy_final']:.3f}, nothing "
        "trained in between. The split is INTRA-PATIENT -- record identity is stripped "
        "by the thesis' preprocessing, so an\n"
        "inter-patient AAMI DS1/DS2 split cannot be built from these files at all -- "
        "and the inner training loop is deliberately cheap, so these accuracies are "
        "NOT comparable to the thesis' fully trained 0.793."
    )
    fig.text(0.062, 0.035, caption, fontsize=7.4, color="0.3", va="bottom",
             linespacing=1.55)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {out}\nwrote {pdf}")

    bank = _REPO / "results" / "runs" / "fig4_spike_raster.json"
    bank.write_text(json.dumps({
        "source": "results/runs/flagship-d4-fixed/result.json",
        "task": mod.TASK, "train_mode": mod.TRAIN_MODE,
        "batch": args.batch, "seed": args.seed, "solver_calls_made": 0,
        "spikes_that_move_fraction": moved,
        "per_neuron_rate_correlation": rate_corr,
        "truth": [int(v) for v in y],
        "class_counts": counts,
        "runs": {tag: {
            "phi": {k: float(flag["phi_initial" if tag == "start" else "phi_final"][k])
                    for k in PHI_KEYS},
            "loss": r["loss"], "accuracy": r["accuracy"], "mean_rate": r["rate"],
            "predictions": [int(v) for v in r["pred"]],
            "distinct_predictions": len(set(r["pred"].tolist())),
            "rate_by_step": [float(v) for v in r["spikes"].mean(axis=(0, 2))],
            "per_class_accuracy": [
                float((r["pred"][y == c] == c).mean()) if np.any(y == c) else None
                for c in range(n_cls)],
        } for tag, r in runs.items()},
    }, indent=2), encoding="utf-8")
    print(f"wrote {bank}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
