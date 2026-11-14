# FAA Outage Attribution Forecaster

Forecasts Polymarket's FAA outage attribution prices minute by minute,
4 hours ahead (240 minutes x 5 quantiles x 4 markets), for
[polymarket-faa-attribution-02](https://make.fun/bounty/polymarket-faa-attribution-02).

## Method

Persistence median (last price) with per-market realized-volatility
widths, a saturating horizon-grown quantile fan, and an asymmetric
sum-to-one coherence pull across the market family (gated: it only
applies when the family's prices sum near 1, so it stays off on the
final's FAA family, whose prices sum to ~0.11). Full details in
`main.py` and `TESTS.md`.

## Run

```sh
python3 main.py   # reads $MAKEFUN_INPUT_DIR (default /input),
                  # writes $MAKEFUN_OUTPUT_DIR (default /output)
```

Python 3.13 standard library only. Deterministic, ~0.06 s per case.

## Practice result (local scorer)

Mean pinball loss **0.251601** over the 6 practice cases
(house baseline 0.363589; NVIDIA v3 baseline on the same cases
0.256382), median error 0.737078, 90% coverage 0.912109.

## Referral

[Try make.fun](https://make.fun/?ref=vz5h17qh&utm_source=github&utm_medium=readme&utm_campaign=polymarket-faa-attribution-02)
