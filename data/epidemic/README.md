# Epidemic incidence data

The Module 1 pages `fitting-epidemic-data.qmd` and `peak-prediction.qmd` read daily incidence through `amlab.epidemics.load_cases()`.

## Expected file

`data/epidemic/cases.csv`, comma separated, with a header row:

```
date,new_cases
2020-03-01,12
2020-03-02,15
```

- `date`: ISO date `YYYY-MM-DD`, one row per day, no gaps.
- `new_cases`: number of new cases reported on that day (non-negative integer).

Store the file together with a note on its source, licence, download date and the region it covers.

## Fallback

If the file is missing, `load_cases()` prints a warning and returns synthetic data: a deterministic SIR epidemic with R0 = 3 (beta = 0.3, gamma = 0.1 per day), population 1,000,000, initial infected fraction 1e-5, 120 days, with Poisson counts around a lognormal reporting factor (sigma = 0.2) and seed 2020. See `amlab/epidemics/data.py`.
