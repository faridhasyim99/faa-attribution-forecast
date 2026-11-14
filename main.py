# polymarket-faa-attribution-02 entry: adaptive steppy-price forecaster.
#
# Forecasts the Yes price of every trading market of Polymarket's FAA
# outage attribution family, one minute at a time, 4 hours ahead, as five
# quantiles (0.05, 0.25, 0.5, 0.75, 0.95) per market per minute.
#
# Model (Python 3.13 standard library only, no network, deterministic).
# Adapted from the NVIDIA v3 forecaster (practice loss 0.256382 on the
# same six practice cases, median error 0.746646, coverage_90 0.936621),
# which this competition's practice reuses verbatim (the FAA markets were
# created 2026-11-13 and hold no history to cut a practice from).
#
#   1. MEDIAN = last price (persistence). A LAD median regression of the
#      240-minute move on every price-based feature (trend windows, MA
#      deviations, RMS, jump counts, price level) finds the conditional
#      median is ~0 for all of them -- the median is already optimal
#      single-market. The only median signal that lives BETWEEN markets
#      is the sum residual (see 3).
#
#   2. WIDTHS from per-market realized volatility. sigma = RMS of
#      1-minute moves over the last 120 minutes, floored at
#      max(0.0005, 0.005*sqrt(x*(1-x))) (a quiet market's band must still
#      cover the jump that eventually comes, and the jump a market can
#      make scales with its price level) and capped at 0.005.
#      spread(h) = WIDTH_SCALE * sigma * sqrt(h / (1 + h/120)), a
#      saturating random-walk spread (SAT_T = 120). Inner quantiles
#      narrowed by b(h) = 0.4 + 0.5*h/240, outer x1.0 (calibrated),
#      with a +/-20% trend-conditioned asymmetry (60-min trend,
#      reversion side wider). WIDTH_SCALE = 0.95 is the global width
#      lever, calibrated on practice (see TESTS.md).
#
#   3. SUM-TO-ONE MEDIAN CORRECTION, ASYMMETRIC + GATED. The practice
#      markets are mutually exclusive outcomes of one event, so their
#      Yes prices should sum to ~1; at the checkpoints they do not
#      (S = 0.81-1.09). The median is shrunk toward the renormalised
#      price x/S, with strength K_DOWN = 1.5 when S > 1 and K_UP = 0.5
#      when S < 1, accruing as 1 - exp(-h/120) (the family reprices
#      fast after a checkpoint). TRANSFER GATE: the correction applies
#      only when 0.7 <= S <= 1.3. The final's four FAA markets are NOT
#      mutually exclusive (an attribution to Iran is also an
#      attribution to a foreign actor) and sum to only ~0.11 (checked
#      live 2026-11-14), so the gate keeps the correction OFF there --
#      an ungated k = 1 would drift every FAA median up by ~8x its
#      price. All six practice cases pass the gate.
#
#   4. REJECTED LEVERS (tested on practice, all lose -- see TESTS.md).
#      Jump give-back after large 60-min moves (0.257-0.262); leader
#      fade for high-price markets (0.255-0.270, wins p00 only --
#      overfits the regime break); EWMA volatility at every halflife
#      (0.254-0.260); MAD robust sigma (0.282); stronger k_down
#      (2.0-3.0) and wider tails (all lose on the quiet half).
#
# Practice score (LOCAL scorer, see TESTS.md): loss 0.251601
# (v3 baseline 0.256382, house baseline 0.363589), median error
# 0.737078, coverage_90 0.912109. Per case: p00 0.396179, p01 0.296418,
# p02 0.312752, p03 0.250120, p04 0.190891, p05 0.115382. Split-half:
# volatile (p00-p02) 0.335116 vs v3's 0.348684; quiet (p03-p05)
# 0.184554 vs v3's 0.182281. HONEST LIMIT: the 0.234 rank-50 target
# is NOT reached -- every median lever beyond the sum correction loses
# (LAD: conditional median ~0 for all price features), and the width
# plateau is flat (0.2555-0.2564 across the whole sweep). The gap is
# p00's regime break, which no causal price-only model can foresee.
#
# Usage (mirrors the kit starter):
#   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# The grader runs one case at a time with only that case's file in /input;
# the program reads every *.in file it finds, in name order, and writes one
# /output/<id>.out per case. Runs in ~0.1 s per case locally, far under the
# 8 s cap; two runs on the same case write byte-identical output.

import json
import math
import os
from os import path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The quantile levels the contest scores, in the order each row lists them.
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at those levels.
Z_OUTER = 1.644854
Z_INNER = 0.674490

# Sigma estimation window: last 120 minutes of 1-minute prices.
SIGMA_WINDOW = 120
SIGMA_FLOOR = 0.0005
SIGMA_CAP = 0.005
# Price-scaled sigma floor: max(SIGMA_FLOOR, PFLOOR_C * sqrt(x*(1-x))).
PFLOOR_C = 0.005
# Horizon saturation: spread(h) = sigma * sqrt(h / (1 + h / SAT_T)).
SAT_T = 120.0
# Global width multiplier (calibrated on practice: 0.95 beats 1.0).
WIDTH_SCALE = 0.95
# Outer (5%/95%) widening factor (calibrated: 1.0 beats 1.1).
OUTER_WIDEN = 1.0
# Inner (25%/75%) narrowing: b(h) = INNER_B0 + INNER_BGROW * h / horizon.
INNER_B0 = 0.4
INNER_BGROW = 0.5
# Trend-conditioned asymmetry: 60-minute trend window and side factor.
TREND_WINDOW = 60
TREND_A = 0.2
# Sum-to-one median correction: asymmetric strength -- K_DOWN when the
# family is overpriced (S > 1), K_UP when underpriced (S < 1).
# Calibrated on practice: K_DOWN = 1.5, K_UP = 0.5 (symmetric k = 1
# scores 0.256382; asymmetric scores 0.252464 linear, 0.251601 with
# the exponential accrual below). The correction accrues as
# 1 - exp(-h/ACC_T), not linearly: the family reprices fast after a
# checkpoint, and the fast accrual wins the volatile half.
K_DOWN = 1.5
K_UP = 0.5
ACC_T = 120.0
# Sum gate: the correction only applies when the family's prices sum
# near 1 (0.7 <= S <= 1.3), i.e. when mutual exclusivity is a sane
# premise. The final's FAA family sums to ~0.11 (checked live
# 2026-11-14: Iran 0.014 + foreign 0.031 + hack 0.0505 + AI 0.0125),
# so the gate keeps the correction OFF there -- an ungated k would
# drift every FAA median up by ~8x its price. All six practice cases
# (S = 0.81-1.09) pass the gate, so the practice score is unaffected.
SUM_GATE_LO = 0.7
SUM_GATE_HI = 1.3


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def estimate_sigma(prices, x):
    """Per-market 1-minute RMS over the last SIGMA_WINDOW minutes,
    floored at max(SIGMA_FLOOR, PFLOOR_C * sqrt(x*(1-x))) and capped."""
    seg = prices[-SIGMA_WINDOW - 1:] if len(prices) > SIGMA_WINDOW else prices
    if len(seg) < 2:
        rms = SIGMA_FLOOR
    else:
        ss = 0.0
        for i in range(len(seg) - 1):
            d = seg[i + 1] - seg[i]
            ss += d * d
        rms = math.sqrt(ss / (len(seg) - 1))
    floor = max(SIGMA_FLOOR, PFLOOR_C * math.sqrt(max(x * (1.0 - x), 0.0)))
    return min(SIGMA_CAP, max(floor, rms))


def forecast(market, horizon, sum_s=1.0):
    """240 rows (horizon rows) of five quantiles for one market."""
    prices = market["prices"]
    if not prices:
        raise ValueError("market %s has no prices" % market["id"])
    x = prices[-1]
    sigma = estimate_sigma(prices, x)
    # 60-minute trend: positive when the market rose into the checkpoint.
    # Mean reversion: widen the side opposite the trend, narrow the other.
    ref = prices[-1 - TREND_WINDOW] if len(prices) > TREND_WINDOW else prices[0]
    t = x - ref
    sgn = 1.0 if t > 0 else (-1.0 if t < 0 else 0.0)
    lo_m = 1.0 + TREND_A * sgn
    hi_m = 1.0 - TREND_A * sgn
    # Sum-to-one median correction, gated: only when 0.7 <= S <= 1.3.
    # Strength is asymmetric (K_DOWN when S > 1, K_UP when S < 1) and
    # accrues exponentially (1 - exp(-h/ACC_T)): the family reprices
    # fast after a checkpoint. Zero when S == 1 or outside the gate.
    if SUM_GATE_LO <= sum_s <= SUM_GATE_HI and sum_s > 0:
        k = K_DOWN if sum_s > 1.0 else K_UP
        med_shift = -k * x * (sum_s - 1.0) / sum_s
    else:
        med_shift = 0.0
    rows = []
    for h in range(1, horizon + 1):
        frac = h / float(horizon)
        acc = 1.0 - math.exp(-h / ACC_T)
        spread = WIDTH_SCALE * sigma * math.sqrt(h / (1.0 + h / SAT_T))
        b = INNER_B0 + INNER_BGROW * frac
        z = (-Z_OUTER * OUTER_WIDEN * lo_m, -Z_INNER * b * lo_m, 0.0,
             Z_INNER * b * hi_m, Z_OUTER * OUTER_WIDEN * hi_m)
        med = x + med_shift * acc
        rows.append([price6(med + zi * spread) for zi in z])
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        cid = fname[:-3]
        with open(path.join(INPUT_DIR, fname), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        if (len(case["quantiles"]) != len(QUANTILES)
                or any(t != q for t, q in zip(case["quantiles"], QUANTILES))):
            raise ValueError("%s asks for quantiles %r, not %r"
                             % (fname, case["quantiles"], QUANTILES))
        horizon = case["horizon_minutes"]
        sum_s = sum(m["prices"][-1] for m in case["markets"])
        forecasts = [{"id": m["id"], "q": forecast(m, horizon, sum_s)}
                     for m in case["markets"]]
        with open(path.join(OUTPUT_DIR, cid + ".out"), "w",
                  encoding="utf-8") as fh:
            fh.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
