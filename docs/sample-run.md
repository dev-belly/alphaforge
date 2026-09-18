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
| total_return | 0.047828 |
| cagr | 0.00754641 |
| ann_vol | 0.133822 |
| ann_downside_dev | 0.0947531 |
| sharpe | 0.123084 |
| sortino | 0.173834 |
| calmar | 0.0331433 |
| max_drawdown | -0.227691 |
| max_drawdown_date | 2020-12-31 |
| max_drawdown_duration_days | 1081 |
| var_95 | -0.0138557 |
| cvar_95 | -0.0202274 |
| skew | -0.0725561 |
| kurtosis | 2.55934 |
| best_period | 0.0379004 |
| worst_period | -0.0401379 |
| positive_period_share | 0.426564 |
| avg_period_return | 6.53622e-05 |
| hit_rate | 0.426564 |
| avg_turnover | 0.242712 |
| cost_drag_ann | 0.00186847 |
| benchmark_return | -0.267025 |
| benchmark_cagr | -0.0487597 |
| benchmark_vol | 0.241829 |
| active_return | 0.314853 |
| beta | 0.466728 |
| alpha_ann | 0.0261526 |
| correlation | 0.843422 |
| r_squared | 0.711361 |
| tracking_error | 0.147648 |
| information_ratio | 0.252048 |
| up_capture | 0.492651 |
| down_capture | 0.473929 |
| treynor | 0.0352909 |
| gross_total_return | 0.0601197 |
| gross_cagr | 0.00943904 |
| gross_ann_vol | 0.133836 |
| gross_sharpe | 0.137107 |
| gross_sortino | 0.193786 |
| gross_calmar | 0.04164 |
| gross_max_drawdown | -0.226682 |
| cost_drag_cagr | 0.00189264 |
| gross_benchmark_cagr | -0.0487597 |

## Run it again

```bash
pip install -e ".[viz]"
python scripts/publish_sample.py
```

This command reads `configs/default.yaml` directly, without environment overrides,
and refuses a non-synthetic provider. Review the manifest's Python and dependency
versions to reproduce this environment. Exact source file hashes accompany the
[source revision](https://github.com/dev-belly/alphaforge/commit/5a9802738b6757159b06f60ef404d95529fa6813).

Output files go to `docs/sample/`; README figures go to `assets/`.
For normal research runs with other providers or settings, use the [Quickstart](quickstart.md).
Read the [validation limits](validation.md) before interpreting performance.
