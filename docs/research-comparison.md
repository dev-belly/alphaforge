# Real-market comparison with audited trading-session splits

[Open all 22 refitted cases](experiments/real-sp500-session-v2/index.html) · [中文研究简报](research-comparison.zh-CN.md)

The **2026-11-14 refit** trains every model under the corrected session-based
purge/embargo rules and current finite-pair evaluation. Its four raw source
hashes and supervised dataset digest match the original experiment:
**840,736 rows, 24 usable features and 551,698 shared forecasts**. No test
outcome selects parameters, a model or a seed for deployment.

| At base assumed costs | Net CAGR | Net Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: |
| Equal-weight baseline | 12.75% | 1.141 | -18.70% |
| Fixed 12–1 momentum | 11.38% | 0.871 | -23.40% |
| Ridge | 12.77% | 1.009 | -21.65% |
| LightGBM, three-seed mean | 17.07% | 1.252 | -22.10% |
| Ridge without the explicit size predictor | 12.43% | 0.970 | -21.57% |

Ridge does not improve Sharpe over the simpler equal-weight baseline in this
sample. LightGBM's doubled-cost mean CAGR is **16.36%**, with Sharpe **1.206**.
Its base-cost Sharpe seed standard deviation is **0.034**; seed variation is not
uncertainty across market histories. All three predefined seeds (11, 29, 42),
both cost scenarios, baselines and the size-feature ablation remain published.

## Audit the actual training boundaries

The supervised dataset and the original market panel can have different date
coverage. The new [split audit](experiments/real-sp500-session-v2/split_audit.json)
retains the original market sessions, supervised sample counts and every fold's
actual boundaries. It computes each last training label's endpoint on the
**original 21-session label calendar**, and requires it to finish strictly
before the first test session.

For the first fold the last training observation is **2013-11-29**; its label
finishes on **2013-12-31** and testing starts on **2014-01-02**. All six folds
retain 21 market sessions between the final training observation and first
test observation. Before fitting comparison models, the builder rejects
overlapping labels, overlapping test folds and folds too small for the model.
Every model must emit the audited sample count in every fold.

The verifier reconstructs the saved date masks, label endpoints and sample
denominators, then independently recomputes portfolio headlines and aggregates
from saved daily returns. It checks internal consistency, not source truth or
an external signature.

## Source and comparison boundaries

The source is 1,173,649 mirrored price rows, with current index members and entry
dates but incomplete removals/delistings. Fundamental availability uses an
assumed publication lag; actual market capitalization is missing and size uses
a dollar-volume proxy. Historical source vintages and industry classification
changes are not independently verified. The usable model features in this run
are the 24 price/volume-derived features named in the saved JSON.

Forecasts cover 2014-01-02–2019-10-16. Portfolio returns cover
2015-01-02–2019-10-31, with **1,217 daily returns per case**. The source ends
2019-11-14 where SPY benchmark coverage ends. It is a historical experiment;
SPY is a price-return proxy, not a dividend-inclusive index benchmark.

Score strategies equally weight their top 50 eligible names, with symbol tie
breaking and no optimizer. The equal-weight baseline holds all forecastable
eligible names, so diversification differs. Rebalance is monthly with a
one-session execution lag. Base costs are commission 2bp, slippage 5bp and
impact coefficient 10bp, with a 10% participation cap. These are assumptions,
not measured fills. Removing the explicit size predictor does not remove
earlier size neutralization.

The data retain the same identity, but new/old metric differences are not a
controlled estimate of one fix: evaluation code and dependency versions also
changed. These results are exploratory rather than estimates of tradable alpha.
`--require-complete-sources` continues to reject the incomplete mirror contract.

## Reproduction and verification

Use CPython 3.12 and the recorded
[research dependency constraints](https://github.com/dev-belly/alphaforge/blob/main/research/requirements-session-refit.txt).
Run from the repository root:

```bash
python -m pip install -e ".[viz]" -c research/requirements-session-refit.txt
python scripts/verify_comparison.py docs/experiments/real-sp500-session-v2
```

Verification uses saved evidence, without downloading prices or refitting.
To refit the declared grid:

```bash
python scripts/fetch_real_data.py --start 2010-01-01 --end 2019-11-14
python scripts/run_comparison.py --out research/comparison
python scripts/verify_comparison.py research/comparison
```

Compare the new assurance record's `raw_sha256` map with
`summary.json:source_assurance.raw_sha256` before calling a fresh download the
same source snapshot. Mirror URLs can change. Parquet byte hashes may differ
with writer versions; this refit's raw files and supervised table match.
The saved summary records parameters, actual split settings, dependency
versions and research-module file hashes.

## Preserved archive

The [pre-correction bundle](experiments/real-sp500/index.html) remains
byte-identical at its original path. It used calendar-day buffers for
trading-session labels and should not be treated as valid evidence of
training/test isolation. The [synthetic sample](sample-run.md) is a separate
archived run with its own source-version boundary.

Both frozen comparison directories disable Git line-ending conversion.
Current CI verifies the old return evidence and the new audited bundle on
Linux and Windows; source refitting is separate from saved-output replay.
