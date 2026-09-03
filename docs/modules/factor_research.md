# Factor Research — `alphaforge.factors`

The factor layer turns raw prices (and fundamentals) into a panel of
cross-sectional signals, preprocesses them, and evaluates every one with the
same information-coefficient discipline a real research desk would demand.
Nothing is accepted on authority: a factor must show a stable, economically
sensible Rank-IC **and** survive multiple-testing correction before it is
allowed anywhere near a portfolio.

```bash
alphaforge research run --config configs/research.yaml
```

```python
from alphaforge.factors.library import FactorLibrary
lib = FactorLibrary.from_config(ctx, cfg)
lib.run()                    # compute -> preprocess -> evaluate
table = lib.summary_table()  # ranked by |rank_icir|, with FDR flags
composite = lib.composite()  # one (date x symbol) alpha panel
```

## The factor contract

Every factor is a `(date x symbol)` panel plus a `FactorSpec` that records its
*a-priori economic direction* (`+1` = "higher raw value should earn a higher
return", `-1` the opposite). The preprocessor multiplies by `direction`, so
every factor stored downstream is "higher is better" and that assumption is
documented rather than implicit.

```python
@dataclass(frozen=True)
class FactorSpec:
    name: str
    category: str             # momentum | reversal | value | quality | risk | liquidity | size
    direction: int = 1        # +1 or -1, the a-priori sign
    requires_fundamentals: bool = False
    data_requirement: str = "price"
```

Factors are registered in a global `REGISTRY`. A factor that cannot be computed
from the configured vendor (e.g. a fundamentals factor run on price-only data)
returns `None` and is dropped rather than silently zero-filled, so availability
is always honest.

## The 42-factor catalog

`FactorLibrary.available()` returns the specs that can actually be computed for
the active dataset. The bundled library ships **42 factors** across seven
categories:

| Category | Count | Examples |
|----------|-------|----------|
| momentum | 7 | `mom_12_1`, `mom_6_1`, `mom_60d`, `mom_120d`, `residual_momentum`, `industry_momentum` |
| reversal | 2 | `rev_5d`, `rev_21d` |
| value | 9 | `book_to_price`, `earnings_yield`, `ebit_to_ev`, `fcf_yield`, `pe_ratio`, `value_composite` |
| quality | 9 | `roa`, `roe`, `gross_margin`, `gross_profitability`, `accruals`, `earnings_quality`, `quality_composite` |
| risk | 7 | `volatility_252d`, `beta_252d`, `downside_volatility`, `idiosyncratic_volatility`, `max_drawdown_252d` |
| liquidity | 6 | `adv_21d`, `amihud_illiquidity`, `turnover_21d`, `zero_trading_days` |
| size | 2 | `log_market_cap`, `log_price` |

## Preprocessing — `FactorPreprocessor`

Raw factor values are never used directly. `ProcessingConfig` drives a
deterministic pipeline:

1. **Winsorise** at the configured percentile per date (tames fat tails without
   deleting observations).
2. **Standardise** cross-sectionally (z-score or rank).
3. **Neutralise** against orthogonalisation controls (market cap, industry, the
   existing book) so a "new" factor is not just beta or size in disguise.

Because all three steps are per-date and cross-sectional, they use only data
available on that date — no look-ahead is possible.

## Evaluation — `evaluate_factor`

`evaluate_factor(name, category, direction, factor, close, horizon=21,
n_quantiles=5)` returns a `FactorResult` with the full tear sheet:

* **Information coefficients** — per-date cross-sectional Pearson `ic` and rank
  (Spearman) `rank_ic`, computed only on names present in *both* the factor and
  the realised forward return; dates with fewer than five names return `NaN`
  rather than a spurious correlation.
* **Quantile portfolios** — equal-weighted long-short spread `qN - q1` and its
  annualised return and hit ratio.
* **IC decay** — mean Rank-IC against forward returns at horizons
  `[1, 5, 10, 21, 42, 63, 126, 252]` days.
* **Year-by-year stability** — mean / ICIR per calendar year.
* **Turnover** of the top-quantile membership.

The summary reports *distributional* statistics, not just point estimates:
`ic_mean`, `ic_std`, `icir = ic.mean() / ic.std()`, `rank_ic_mean`,
`rank_icir`, `t_stat` (with an overlapping-window correction — daily sampling of
an `h`-day forward return overstates the naive t-stat by ~`sqrt(h)`),
`positive_ic_ratio`, `p_value`, and `quantile_spread`.

> A factor with a great mean IC and a terrible ICIR is not a factor. The tear
> sheet is designed to make that obvious.

## Multiple-testing control — Benjamini-Hochberg

Screening 42 factors at the 5% level produces ~2 false positives by
construction. `rank_summary_table(..., fdr_alpha=0.05)` attaches a
`significant_fdr` column (Benjamini-Hochberg) alongside the naive
`significant_naive` (`p < 0.05`). Reporting which factors survive FDR control is
the difference between factor research and data mining.

## Composition

`FactorLibrary.composite(names, weights)` builds an equal- (or custom-) weighted
z-score composite of processed factors — the signal handed to the model and the
portfolio constructor. `factor_correlation()` reports the average
cross-sectional correlation between processed panels so redundant factors are
visible before they inflate the book.
