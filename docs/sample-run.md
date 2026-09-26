# Reproduce the sample

This is a computed run on **synthetic data**, with seed 42,
160 symbols and data from 2015-01-01 to 2024-12-31.
The backtest covers 2019-01-01 to 2024-12-31.
All README figures and this report come from the same pipeline state.

[Open the full report](sample/research_report.html) · [Run manifest](sample/manifest.json) ·
[Daily results CSV](sample/daily_results.csv) · [Configuration](sample/config.json) · [Metrics JSON](sample/metrics.json)

## Recorded performance

Values below are decimal ratios unless named otherwise; for example, 0.01 CAGR is 1%.
Sharpe, Sortino, Calmar and information ratio are ratios, not percentages.

| Metric | Value |
| --- | ---: |
| start | 2019-01-01 |
| end | 2024-12-31 |
| n_periods | 1566 |
| years | 6.21429 |
| periods_per_year | 252 |
| total_return | 0.0309223 |
| cagr | 0.00491264 |
| ann_vol | 0.133442 |
| ann_downside_dev | 0.0946157 |
| sharpe | 0.103438 |
| sortino | 0.145884 |
| calmar | 0.0210709 |
| max_drawdown | -0.233148 |
| max_drawdown_date | 2020-12-31 |
| max_drawdown_duration_days | 1082 |
| var_95 | -0.0141114 |
| cvar_95 | -0.0201959 |
| skew | -0.0703616 |
| kurtosis | 2.54766 |
| best_period | 0.0376829 |
| worst_period | -0.0398015 |
| positive_period_share | 0.427203 |
| avg_period_return | 5.47736e-05 |
| hit_rate | 0.427203 |
| avg_turnover | 0.236627 |
| cost_drag_ann | 0.00181532 |
| benchmark_return | -0.267025 |
| benchmark_cagr | -0.0487597 |
| benchmark_vol | 0.241829 |
| active_return | 0.297947 |
| beta | 0.46539 |
| alpha_ann | 0.0234565 |
| correlation | 0.843399 |
| r_squared | 0.711321 |
| tracking_error | 0.147834 |
| information_ratio | 0.233681 |
| up_capture | 0.490524 |
| down_capture | 0.473705 |
| treynor | 0.0296589 |
| gross_total_return | 0.0426918 |
| gross_cagr | 0.00675002 |
| gross_ann_vol | 0.133456 |
| gross_sharpe | 0.117128 |
| gross_sortino | 0.165316 |
| gross_calmar | 0.0290758 |
| gross_max_drawdown | -0.232152 |
| cost_drag_cagr | 0.00183738 |
| gross_benchmark_cagr | -0.0487597 |

## Run it again

```bash
pip install -e ".[viz]"
python scripts/publish_sample.py
```

This command reads `configs/default.yaml` directly, without environment overrides,
and refuses a non-synthetic provider. Review the manifest's Python and dependency
versions to reproduce this environment. Exact source file hashes accompany the
[source revision](https://github.com/dev-belly/alphaforge/commit/f4b396e5facafca52fd0e9b11feb529fe7cf65b4).

Output files go to `docs/sample/`; README figures go to `assets/`.
For normal research runs with other providers or settings, use the [Quickstart](quickstart.md).
Read the [validation limits](validation.md) before interpreting performance.
