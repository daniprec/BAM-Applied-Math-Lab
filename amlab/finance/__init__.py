"""Financial time series tools for the Module 3 laboratory.

Main functions
--------------
load_prices              Frozen price data, or a synthetic fallback series.
synthetic_garch_prices   GARCH(1,1) prices with Student-t innovations.
log_returns              Log returns from a price series.
aggregate_returns        Non-overlapping sums of log returns over a horizon.
autocorrelation          Sample autocorrelation function.
ks_normal_bootstrap      KS normality test calibrated by parametric bootstrap.
hill_estimator           Hill estimator of the tail index.

Only numpy, scipy, pandas and matplotlib are used.
"""

from amlab.finance.data import load_prices, synthetic_garch_prices
from amlab.finance.stats import (
    aggregate_returns,
    autocorrelation,
    hill_estimator,
    ks_normal_bootstrap,
    log_returns,
)

__all__ = [
    "aggregate_returns",
    "autocorrelation",
    "hill_estimator",
    "ks_normal_bootstrap",
    "load_prices",
    "log_returns",
    "synthetic_garch_prices",
]
