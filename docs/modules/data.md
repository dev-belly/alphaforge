# Data — Providers & Quality Gates

The single most expensive failure mode in quant research is silent bad data, so
every ETL pass runs a quality report and the findings are persisted next to the
artefacts.

## Providers — `alphaforge.data.providers`

| Provider | Status |
|----------|--------|
| `sample` | ships; fully synthetic but **point-in-time** (carries a survivorship disclaimer) |
| `local` | reads a local long price table |
| `yahoo` | reserved adapter (historical OHLCV) |
| `akshare` | reserved adapter (A-share market data) |
| `tushare` | **reserved** — wiring point exists, no live adapter ships (supply token + fetch) |

`DataPipeline` persists the canonical long table; `build_panel` pivots it into
the aligned wide `MarketPanel` (close / returns / volume / market_cap / industry
/ universe) that every downstream layer consumes.

## Quality — `alphaforge.data.quality`

```python
from alphaforge.data.quality import build_quality_report, clean_prices

report = build_quality_report(prices, provenance="sample")
assert report.is_acceptable()        # worst missingness + no duplicates
clean  = clean_prices(prices, drop_duplicates=True, drop_non_positive=True)
```

`DataQualityReport` records coverage, missingness, outliers (robust MAD on the
log scale), OHLC violations, stale-price runs, extreme returns and
delisting candidates — and emits a survivorship note for names that stop trading
before the panel end (they are *retained*, never silently dropped). Cleaning
primitives (`clean_prices`, `align_calendar`, `detect_missing_blocks`) are
deterministic and flag, never overwrite, implausible moves.

See `tests/unit/test_data_quality.py` for the contract.
