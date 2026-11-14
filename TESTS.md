# Tests

## Entry

`main.py` — Python 3.13 standard library only (`math`, `json`, `os`,
`sys`). No network at runtime, no file reads outside the input
directory. Reads `$MAKEFUN_INPUT_DIR` (default `/input`), writes
`$MAKEFUN_OUTPUT_DIR` (default `/output`), one `.out` per `.in` found,
in name order. Runtime ~0.06 s per case (limit 8 s).

## Local scorer

The kit ships no local scorer, so one was built at
`~/workspace/makefun-faa/dev/score.py` (dev tool, not part of the
entry). It fetches the TRUE prices for each practice case's markets
for minutes `at+1..at+240` from Polymarket's public CLOB API
(`GET https://clob.polymarket.com/prices-history?market=<yes_token>&startTs=<s>&endTs=<s>&fidelity=1`,
no key), builds the 1-minute grid exactly as the house does (price at
minute m = last point at or before m, carried forward), then computes
mean pinball loss exactly as RULES.txt Scoring:
`100 * max(tau*(y-q), (tau-1)*(y-q))`, mean over every case, market,
minute and quantile. It also reports median error
(mean `100*|y-median|`) and 90% coverage.

Validation: the scorer reproduces the submitted NVIDIA v3's official
practice loss EXACTLY — 0.256382, per-case p00 0.425255, p01 0.296172,
p02 0.324625, p03 0.247400, p04 0.183670, p05 0.118486, median error
0.746646, coverage 0.936621, valid=True. The true-price history is
cached at `~/workspace/makefun-nvidia/dev/history.json` (24 tokens,
complete).

## Calibration (all numbers from the local scorer, mean over p00-p05)

Tuning scripts in `~/workspace/makefun-faa/dev/` (diag2.py, tune.py,
diag3.py, diag4.py, diag5.py). Every lever was tested with split-half
reporting (volatile p00-p02 vs quiet p03-p05) to avoid overfitting a
single case.

| variant | loss |
|---|---|
| house baseline (starter) | 0.363589 |
| NVIDIA v3 as-is (k=1/1 lin, W=1.0, SAT=120, OUT=1.1) | 0.256382 |
| WIDTH_SCALE 0.90 | 0.255859 |
| OUTER_WIDEN 1.00 | 0.255565 |
| combo W=0.95 SAT=120 OUT=1.00 | 0.255555 |
| asymmetric sum-k k_down=1.5 / k_up=0.5 (linear) | 0.253191 |
| + exponential accrual T=120 | **0.251601** |
| k_down=2.0 / 2.5 / 3.0 (exp T=120) | 0.252244 / 0.254795 / 0.259099 |
| SAT_T 60 / 90 / 180 | 0.253560 / 0.251843 / 0.251861 |
| inner b0 0.5 / 0.6 | 0.251458 / 0.251455 |
| TREND_A 0.3 | 0.251375 |
| EWMA sigma halflife 30 / 60 / 120 | 0.254202 / 0.257259 / 0.259673 |
| MAD robust sigma | 0.281919 |
| jump give-back G=0.1-0.3 | 0.257-0.262 |
| leader fade f=0.1 / 0.2 | 0.255020 / 0.262239 |

Final locked config: WIDTH_SCALE 0.95, SAT_T 120, OUTER_WIDEN 1.0,
inner b(h) = 0.4 + 0.5*h/240, TREND_A 0.2 (60-min), SIGMA_WINDOW 120,
SIGMA_CAP 0.005, PFLOOR_C 0.005, K_DOWN 1.5 / K_UP 0.5, exponential
accrual T=120, sum gate 0.7-1.3. The b0/TREND_A variants within
+-0.0002 of the locked value are noise, not real gains, so the v3
values were kept.

## Final practice result (finished entry, dev/score.py)

| case | loss | median error | coverage_90 |
|---|---|---|---|
| p00 | 0.396179 | 1.180535 | 0.8428 |
| p01 | 0.296418 | 0.776414 | 0.9053 |
| p02 | 0.312752 | 1.009936 | 0.8906 |
| p03 | 0.250120 | 0.733739 | 0.9755 |
| p04 | 0.190891 | 0.578499 | 0.8502 |
| p05 | 0.115382 | 0.300636 | 0.9906 |
| **overall** | **0.251601** | **0.737078** | **0.912109** |

Split-half: volatile (p00-p02) 0.335116 vs v3's 0.348684 (wins);
quiet (p03-p05) 0.184554 vs v3's 0.182281 (loses +0.0023). The
exponential accrual wins the volatile half by more than it loses
on the quiet half.

Per-tau decomposition of v3: tau 0.5 = 29.1% of loss, 0.75 = 27.6%,
0.25 = 22.8%, 0.95 = 13.4%, 0.05 = 7.1% — the median and inner
quantiles dominate, so the width/median levers are the right ones.

## Determinism

Two runs over the same 6 practice inputs are byte-identical
(`diff -r` clean; see command below). Runtime for all 6 cases
0.355 s total (~0.06 s per case).

```sh
MAKEFUN_INPUT_DIR=kit/polymarket-faa-attribution-02-kit/input \
  MAKEFUN_OUTPUT_DIR=/tmp/faa-run1 python3 entry/main.py
MAKEFUN_INPUT_DIR=kit/polymarket-faa-attribution-02-kit/input \
  MAKEFUN_OUTPUT_DIR=/tmp/faa-run2 python3 entry/main.py
diff -r /tmp/faa-run1 /tmp/faa-run2   # no output
```

## Output shape

Every `.out` is `{"forecasts": [{"id", "q": [[5 numbers] x 240]}]}`,
240 rows per market, each row 5 non-decreasing quantiles in [0,1],
rounded to 6 decimals. Sizes 225-294 KB per case (limit 1,048,576).
All 6 practice outputs valid (`valid=True` from the scorer).

## Transfer caveat (practice -> final) and the sum gate

Practice is the NVIDIA CEO event (19-24 markets, family sum S =
0.81-1.09). The final is 4 FAA attribution markets. Checked live
2026-11-14 via the Polymarket gamma API, the FAA Yes prices are
Iran 0.014, foreign actor 0.031, hack 0.0505, AI sandbox escape
0.0125 — sum S = 0.108, NOT ~1. The four attributions are not
mutually exclusive (Iran is a subset of foreign actor) and the
crowd thinks any attribution is unlikely.

The sum-to-one correction therefore only applies when
0.7 <= S <= 1.3. All six practice cases pass the gate (so the
practice score is unaffected); on the final the correction is OFF,
which is the correct behaviour — an ungated k would compute a
median shift of +8.26*x, drifting a 0.014 market toward ~0.13.

## Honest limit: the 0.234 rank-50 target is NOT reached

Best practice loss 0.251601 vs target <= 0.234 (rank 50 on the
practice board). The gap is p00's regime break: the top-3 leader
markets fell ~4-5 points after the checkpoint (median errors
4.1-4.7), and no causal price-only model can foresee that — the
LAD median regression on 6,200 settled-regime samples (NVIDIA v2
CALIBRATION-NOTE.md) gives in-sample median |residual| 0.00146 vs
persistence 0.00150 (2.7% gain), single-feature coefficients ~0.
Every median lever beyond the asymmetric sum correction loses on
the split-half (leader fade wins p00 but loses p03-p05 badly;
stronger k_down wins p00 but loses the quiet half). The width
plateau is flat (0.2555-0.2564 across the whole sweep). The
0.111 entries at the top of the practice board are consistent with
fitting the public practice answers, which cannot transfer to the
final's 12 unseen cases.
