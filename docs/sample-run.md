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
| total_return | 0.0520076 |
| cagr | 0.00819204 |
| ann_vol | 0.10748 |
| ann_downside_dev | 0.0758553 |
| sharpe | 0.12962 |
| sortino | 0.183659 |
| calmar | 0.0429131 |
| max_drawdown | -0.190898 |
| max_drawdown_date | 2020-12-31 |
| max_drawdown_duration_days | 1065 |
| var_95 | -0.0109244 |
| cvar_95 | -0.01643 |
| skew | -0.00136024 |
| kurtosis | 3.43624 |
| best_period | 0.0375985 |
| worst_period | -0.0349239 |
| positive_period_share | 0.427842 |
| avg_period_return | 5.52839e-05 |
| hit_rate | 0.427842 |
| avg_turnover | 0.191267 |
| cost_drag_ann | 0.00144403 |
| benchmark_return | -0.267025 |
| benchmark_cagr | -0.0487597 |
| benchmark_vol | 0.241829 |
| active_return | 0.319032 |
| beta | 0.371263 |
| alpha_ann | 0.0216326 |
| correlation | 0.835341 |
| r_squared | 0.697795 |
| tracking_error | 0.163123 |
| information_ratio | 0.212566 |
| up_capture | 0.388454 |
| down_capture | 0.37303 |
| treynor | 0.0375247 |
| gross_total_return | 0.0615388 |
| gross_cagr | 0.00965637 |
| gross_ann_vol | 0.107485 |
| gross_sharpe | 0.143122 |
| gross_sortino | 0.202939 |
| gross_calmar | 0.0508307 |
| gross_max_drawdown | -0.189971 |
| cost_drag_cagr | 0.00146433 |
| gross_benchmark_cagr | -0.0487597 |

## Run it again

```bash
pip install -e ".[viz]"
python scripts/publish_sample.py
```

This command reads `configs/default.yaml` directly, without environment overrides,
and refuses a non-synthetic provider. Review the manifest's Python and dependency
versions to reproduce this environment. Exact source file hashes accompany the
[source revision](https://github.com/dev-belly/alphaforge/commit/06904490bca4465292fd286d0cac518571441cde).

Output files go to `docs/sample/`; README figures go to `assets/`.
For normal research runs with other providers or settings, use the [Quickstart](quickstart.md).
Read the [validation limits](validation.md) before interpreting performance.
