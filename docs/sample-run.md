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
| total_return | 0.0465339 |
| cagr | 0.00734605 |
| ann_vol | 0.133732 |
| ann_downside_dev | 0.0947119 |
| sharpe | 0.12159 |
| sortino | 0.171684 |
| calmar | 0.0321392 |
| max_drawdown | -0.22857 |
| max_drawdown_date | 2020-12-08 |
| max_drawdown_duration_days | 1081 |
| var_95 | -0.0139561 |
| cvar_95 | -0.0202322 |
| skew | -0.0734462 |
| kurtosis | 2.56614 |
| best_period | 0.0379743 |
| worst_period | -0.0398118 |
| positive_period_share | 0.427842 |
| avg_period_return | 6.45257e-05 |
| hit_rate | 0.427842 |
| avg_turnover | 0.242191 |
| cost_drag_ann | 0.00186365 |
| benchmark_return | -0.267025 |
| benchmark_cagr | -0.0487597 |
| benchmark_vol | 0.241829 |
| active_return | 0.313559 |
| beta | 0.466775 |
| alpha_ann | 0.0259428 |
| correlation | 0.844072 |
| r_squared | 0.712457 |
| tracking_error | 0.147548 |
| information_ratio | 0.250789 |
| up_capture | 0.492358 |
| down_capture | 0.473789 |
| treynor | 0.0348358 |
| gross_total_return | 0.0587827 |
| gross_cagr | 0.00923407 |
| gross_ann_vol | 0.133746 |
| gross_sharpe | 0.135591 |
| gross_sortino | 0.1916 |
| gross_calmar | 0.0405787 |
| gross_max_drawdown | -0.227559 |
| cost_drag_cagr | 0.00188801 |
| gross_benchmark_cagr | -0.0487597 |

## Run it again

```bash
pip install -e ".[viz]"
python scripts/publish_sample.py
```

This command reads `configs/default.yaml` directly, without environment overrides,
and refuses a non-synthetic provider. Review the manifest's Python and dependency
versions to reproduce this environment. Exact source file hashes accompany the
[source revision](https://github.com/dev-belly/alphaforge/commit/678cf75b7ab1d67149ab039b055c6e44792d58bf).

Output files go to `docs/sample/`; README figures go to `assets/`.
For normal research runs with other providers or settings, use the [Quickstart](quickstart.md).
Read the [validation limits](validation.md) before interpreting performance.
