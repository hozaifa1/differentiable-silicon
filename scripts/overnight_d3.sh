#!/bin/sh
# D3 night, after the recalibration. Supersedes overnight_d2.sh.
#
# WHAT CHANGED, and why the D2 script will not run any more
# ---------------------------------------------------------
# overnight_d2.sh asks for --d 3 and --d 5. Both expose Pr and Ec, which are now
# LOCKED material constants of the calibrated HZO film, and run_flagship refuses
# them by design. The design vector is d=4: t_fe, L_g, N_ch, t_IL -- the four
# knobs a process engineer can actually be asked for. See docs/D3_RECALIBRATION.md.
#
# THE CACHE IS COLD. cache_key now folds a hash of extract.py in, on every
# backend, because a cache record stores the seven EXTRACTED figures of merit and
# the extraction was rewritten today. Nothing stale can be served; everything has
# to be recomputed. Step 0 below is that recomputation.
#
# theta0 is a deliberately poor corner, as on D2: a film too thin to hold a
# memory window, on a long gate, heavily doped. Starting at nominal is a bad
# experiment -- the nominal device already solves this task, so the optimiser has
# nowhere to go and a run that accepts zero steps proves nothing. The recovery IS
# the result.
#
#   theta0 = 0.05, 0.80, 0.90, 0.70
#          =  t_fe 5.5 nm | L_g 52 nm | N_ch 6.3e17 | t_IL 1.55 nm
#
# SNN_TRAIN_STEPS = 150, AND THE CALL BUDGET IS 45, NOT 65. Both from measurement.
#
# Measured on the nominal d=4 device, batch 16, 300 Adam steps:
#
#   step    0   train CE 1.4992   test acc 0.297   macro-F1 0.178
#   step  100   train CE 1.2888   test acc 0.359   macro-F1 0.269
#   step  150   train CE 1.2446   test acc 0.484   macro-F1 0.461   <- knee
#   step  200   train CE 1.2089   test acc 0.453   macro-F1 0.412
#   step  299   train CE 1.1552   test acc 0.484   macro-F1 0.436
#
# Test macro-F1 plateaus at ~150 while train CE keeps falling: past the knee the
# extra steps buy overfitting to a 16-beat batch, not generalisation. So 150,
# not D2's 400.
#
# COST. ~3.0 s per Adam step at batch 16, so 150 steps is ~7.5 min PER DESIGN
# POINT and the flagship pays it once per oracle call. 65 calls would be ~8.1 h,
# past the overnight window; 45 calls is ~5.6 h plus ~27 min of solver. If it has
# to be shortened further, cut SNN_TRAIN_STEPS to 100 (~5.0 min a point) before
# cutting calls -- the loss still separates devices there, it is only a weaker
# classifier.
#
# Batch scales SUBLINEARLY here, because the 111-step Python loop dominates the
# forward pass: batch 16 costs 0.65 s forward / 2.02 s backward, batch 64 costs
# 0.84 s / 5.89 s. So 4x the batch is 2.5x the time, if a better gradient is ever
# worth buying.
#
# Full curve: results/runs/snn_calibration_d3.json.
set -x
PY=./.venv/Scripts/python.exe
export SNN_TRAIN_STEPS=${SNN_TRAIN_STEPS:-150}
export SNN_BATCH=${SNN_BATCH:-16}

mkdir -p results/runs

# --- 0. Re-baseline. The cache is cold; do this before comparing anything. ----
$PY scripts/rebaseline_d3.py --backend devsim --d 4 \
  > results/runs/rebaseline_d3_devsim.log 2>&1

# --- 1. The flagship, on the recalibrated design vector ----------------------
$PY scripts/run_flagship.py --backend devsim --d 4 \
    --max-oracle-calls 45 --max-steps 40 \
    --alpha 0.04 --trust-radius 0.06 --theta0 0.05,0.80,0.90,0.70 \
    --tag flagship-devsim-d4 \
  > results/runs/flagship-devsim-d4.log 2>&1

# --- 2. Cross-check the two oracles on the same locked film ------------------
# Now a fairer comparison than D2's: both oracles read Pr and Ec from
# shared/material.py, so any disagreement is about geometry and transport rather
# than about two different materials.
$PY scripts/cross_check_oracles.py --a mock --b devsim --d 4 --alpha 0.04 \
    --out results/runs/cross_check_mock_devsim_d4.json \
  > results/runs/cross_check_mock_devsim_d4.log 2>&1

echo OVERNIGHT_D3_DONE
