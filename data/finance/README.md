# Financial price data for Module 3

The Session 18 laboratory (`modules/randomness/returns-data.qmd` and the
pages after it) reads daily closing prices through
`amlab.finance.load_prices()`.

## Expected file

`data/finance/prices.csv`, comma separated, UTF-8, with a header row:

```
date,close
2006-01-03,1268.80
2006-01-04,1273.46
...
```

| column  | type                     | meaning                                        |
|---------|--------------------------|------------------------------------------------|
| `date`  | ISO date `YYYY-MM-DD`    | trading day, one row per day, ascending order  |
| `close` | positive float           | closing price or index level on that day       |

Other columns are ignored. Rows with a missing or non-positive `close` are
dropped by the loader. The values above are placeholders that show the
format; they are not real quotes.

## Status

The file is **not yet in the repository**. Until it is added,
`load_prices()` prints a warning and returns a synthetic series: a GARCH(1,1)
process (Bollerslev 1986) with Student-t innovations (4 degrees of freedom),
omega = 2e-6, alpha = 0.08, beta = 0.90, 5000 business days starting on
2006-01-02, initial price 100, seed 2026. See
`amlab/finance/data.py::synthetic_garch_prices`. The synthetic series has
fat tails and volatility clustering by construction, so the lab still shows
the intended effects, but any number computed from it says nothing about a
real market.

## Suggested dataset and how to freeze it

<!-- TODO(Daniel): choose the series and check the licence / terms of use of the source. -->

- **Series:** one broad equity index or a liquid stock, daily close. A
  long series (at least 15 years, about 4000 trading days) gives stable
  tail estimates. Include at least one crisis period (for example 2008 or
  2020) so that volatility clustering is visible.
- **Source:** any provider whose terms allow redistribution in a course
  repository (for example a central bank or an exchange that publishes
  historical index levels). Record the source URL and the download date
  below.
- **Freezing:** download once, convert to the two-column schema above,
  commit the CSV, and never overwrite it during the course. Students must
  get the same numbers when they rerun the pages. Do not download data at
  render time.

A minimal conversion script, run once by the instructor:

```python
import pandas as pd

raw = pd.read_csv("downloaded_file.csv")           # provider format
df = raw.rename(columns={"Date": "date", "Close": "close"})[["date", "close"]]
df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
df.dropna().sort_values("date").to_csv("data/finance/prices.csv", index=False)
```

## Provenance (fill in when the file is added)

- Series:
- Source and URL:
- Date range:
- Download date:
- Licence or terms of use:
