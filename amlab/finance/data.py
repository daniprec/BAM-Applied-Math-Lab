"""Loading of price data for the financial time series lab.

The lab reads a frozen file ``data/finance/prices.csv`` with columns
``date,close`` (see ``data/finance/README.md``). If the file is missing,
``load_prices`` returns a synthetic series so that every page still runs.
The synthetic series is NOT market data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRICES = REPO_ROOT / "data" / "finance" / "prices.csv"


def synthetic_garch_prices(
    n: int = 5000,
    omega: float = 2e-6,
    alpha: float = 0.08,
    beta: float = 0.90,
    nu: float = 4.0,
    p0: float = 100.0,
    start: str = "2006-01-02",
    seed: int = 2026,
) -> pd.DataFrame:
    """Simulate daily closing prices whose log returns follow a GARCH(1,1).

    The model is (Bollerslev 1986)

        r_t = sigma_t * z_t,
        sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2,

    where z_t are independent Student-t variables with ``nu`` degrees of
    freedom, rescaled to unit variance. With the default values the
    unconditional variance is omega / (1 - alpha - beta) = 1e-4, which is
    a daily volatility of 1 %.

    Parameters
    ----------
    n : int
        Number of returns (the price series has n + 1 values).
    omega, alpha, beta : float
        GARCH(1,1) parameters. Requires alpha + beta < 1.
    nu : float
        Degrees of freedom of the Student-t innovations (nu > 2).
    p0 : float
        Initial price.
    start : str
        First date. Dates are business days.
    seed : int
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    pandas.DataFrame
        Columns ``date`` and ``close``.
    """
    if alpha + beta >= 1:
        raise ValueError("alpha + beta must be smaller than 1")
    if nu <= 2:
        raise ValueError("nu must be larger than 2 for a finite variance")
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, size=n) * np.sqrt((nu - 2.0) / nu)
    r = np.empty(n)
    var = omega / (1.0 - alpha - beta)
    r_prev = 0.0
    for t in range(n):
        var = omega + alpha * r_prev**2 + beta * var
        r[t] = np.sqrt(var) * z[t]
        r_prev = r[t]
    close = p0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    dates = pd.bdate_range(start=start, periods=n + 1)
    return pd.DataFrame({"date": dates, "close": close})


def load_prices(path: str | Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """Load daily closing prices.

    Reads ``data/finance/prices.csv`` (columns ``date,close``) when it exists.
    Otherwise it returns ``synthetic_garch_prices()`` with its default,
    fixed seed and prints a warning.

    Parameters
    ----------
    path : str or Path, optional
        CSV file to read. Defaults to ``data/finance/prices.csv`` in the repo.
    verbose : bool
        Print which source was used.

    Returns
    -------
    pandas.DataFrame
        Columns ``date`` (datetime64) and ``close`` (float), sorted by date,
        with missing and non-positive prices removed. The attribute
        ``df.attrs["source"]`` is either the file path or ``"synthetic"``.
    """
    path = DEFAULT_PRICES if path is None else Path(path)
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        missing = {"date", "close"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} lacks the columns {sorted(missing)}")
        df = df[["date", "close"]].dropna()
        df = df[df["close"] > 0].sort_values("date").reset_index(drop=True)
        df.attrs["source"] = str(path)
        if verbose:
            print(f"Loaded {len(df)} prices from {path.name}")
        return df
    if verbose:
        print(
            "WARNING: data/finance/prices.csv not found. Using a SYNTHETIC "
            "GARCH(1,1) series with Student-t innovations (seed 2026). "
            "These are not market data. See data/finance/README.md."
        )
    df = synthetic_garch_prices()
    df.attrs["source"] = "synthetic"
    return df
