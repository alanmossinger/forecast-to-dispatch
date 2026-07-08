"""Exogenous market drivers: hourly ERCOT load, wind, and solar actuals.

Why this matters: price scarcity in ERCOT is a *net load* story — demand minus
wind minus solar. A price-only model can chase its own tail; giving the
forecaster the physical drivers is what makes its SHAP explanations read in
market terms (net-load stress, solar fade) instead of autoregressive noise.

Honesty note (leakage): ERCOT's *forecast* reports (load forecast, STPPF/WGRPP
solar/wind forecasts) expire from the MIS after a retention window, so for a
historical study year they cannot be reconstructed. Rather than pretend actuals
for day D were forecastable, the feature layer only ever uses these series at
**lags of 48h or more** — persistence proxies that were genuinely observable
before the forecast issue time. This is documented in the model card as a known
limitation: a production system would subscribe to the live forecast feeds and
likely improve on the numbers reported here.

Sources (public ERCOT archives, no login):
- Hourly load:  "Hourly Load Data Archives" ``Native_Load_<year>.zip``
- Wind/solar:   "Fuel Mix Report" 15-minute generation by fuel
  (2024 lives inside ``FuelMixReport_PreviousYears.zip``)
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

NATIVE_LOAD_URLS = {
    2024: "https://www.ercot.com/files/docs/2024/02/06/Native_Load_2024.zip",
}
FUEL_MIX_PREVIOUS_YEARS_URL = (
    "https://www.ercot.com/files/docs/2021/03/10/FuelMixReport_PreviousYears.zip"
)

EXOG_COLUMNS = ["load_mw", "wind_mw", "solar_mw", "net_load_mw"]


def _download(url: str, timeout: int = 600) -> bytes:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "forecast-to-dispatch"})
    resp.raise_for_status()
    return resp.content


def _parse_hour_ending_index(df: pd.DataFrame, date_col: str, tz: str) -> pd.DatetimeIndex:
    """ERCOT publishes 'HourEnding' stamps like '01/01/2024 01:00' plus a DST
    convention ('24:00' rows; duplicated hours in November). Convert to a
    tz-aware hour-*beginning* index to match the price panel."""
    raw = df[date_col].astype(str).str.strip()
    # '24:00' means the hour ending at midnight of the NEXT day; hour-beginning 23:00.
    date_part = raw.str.slice(0, 10)
    hour_part = raw.str.slice(11, 16).replace("", "00:00")
    hour_ending = hour_part.str.split(":").str[0].astype(int)
    # DST flag column present in native load files marks repeated fall-back hour.
    naive = pd.to_datetime(date_part, format="%m/%d/%Y") + pd.to_timedelta(
        hour_ending - 1, unit="h"
    )
    if "DSTFlag" in df.columns:
        ambiguous = df["DSTFlag"].astype(str).str.strip().str.upper().ne("Y").to_numpy()
    else:
        ambiguous = "infer"
    return pd.DatetimeIndex(
        naive.dt.tz_localize(tz, ambiguous=ambiguous, nonexistent="shift_forward"),
        name="interval_start",
    )


def fetch_native_load(year: int, cache_dir: Path, tz: str) -> pd.Series:
    """Hourly ERCOT system load (MW) for a year, tz-aware hour-beginning index."""
    cache = cache_dir / f"ercot_native_load_{year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["load_mw"]
    if year not in NATIVE_LOAD_URLS:
        raise KeyError(f"No Native_Load URL configured for {year}; add it to NATIVE_LOAD_URLS")
    z = zipfile.ZipFile(io.BytesIO(_download(NATIVE_LOAD_URLS[year])))
    (name,) = [n for n in z.namelist() if n.endswith((".xlsx", ".xls", ".csv"))]
    df = pd.read_excel(z.open(name)) if not name.endswith(".csv") else pd.read_csv(z.open(name))
    df.columns = [str(c).strip() for c in df.columns]
    time_col = next(c for c in df.columns if "hour" in c.lower())
    total_col = next(c for c in df.columns if c.upper() in ("ERCOT", "TOTAL"))
    idx = _parse_hour_ending_index(df, time_col, tz)
    load = pd.Series(
        pd.to_numeric(df[total_col], errors="raise").to_numpy(), index=idx, name="load_mw"
    ).sort_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    load.to_frame().to_parquet(cache)
    return load


def fetch_wind_solar(year: int, cache_dir: Path, tz: str) -> pd.DataFrame:
    """Hourly ERCOT wind and solar generation (MW) from the Fuel Mix Report.

    The report stores 15-minute MWh per fuel in wide format (one column per
    settlement interval); we sum to hourly MWh, which numerically equals the
    hourly average MW.
    """
    cache = cache_dir / f"ercot_wind_solar_{year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    outer = zipfile.ZipFile(io.BytesIO(_download(FUEL_MIX_PREVIOUS_YEARS_URL)))
    (inner_name,) = [n for n in outer.namelist() if str(year) in n]
    df = pd.read_excel(io.BytesIO(outer.read(inner_name)), sheet_name=None)

    frames = []
    for sheet, data in df.items():
        data.columns = [str(c).strip() for c in data.columns]
        if "Fuel" not in data.columns:
            continue
        fuel = data["Fuel"].astype(str).str.strip().str.lower()
        keep = data[fuel.isin(["wind", "solar"])].copy()
        if keep.empty:
            continue
        date_col = next(c for c in keep.columns if "date" in c.lower())
        interval_cols = [c for c in keep.columns if ":" in str(c)]
        long = keep.melt(
            id_vars=[date_col, "Fuel"],
            value_vars=interval_cols,
            var_name="interval",
            value_name="mwh",
        )
        long["mwh"] = pd.to_numeric(long["mwh"], errors="coerce")
        long = long.dropna(subset=["mwh"])
        # Columns like '1:15 (DST)' are the SECOND occurrence of the repeated
        # fall-back hour in November; flag them, then strip the suffix.
        interval = long["interval"].astype(str)
        long["second_occurrence"] = interval.str.contains("DST")
        interval = interval.str.replace(r"\s*\(DST\)", "", regex=True).str.strip()
        # Column '0:15' is the interval ENDING 00:15 → interval start 00:00.
        hhmm = interval.str.split(":", expand=True).astype(int)
        end_minutes = hhmm[0] * 60 + hhmm[1]
        long["start_naive"] = pd.to_datetime(long[date_col]) + pd.to_timedelta(
            end_minutes - 15, unit="m"
        )
        frames.append(long[["start_naive", "second_occurrence", "Fuel", "mwh"]])

    if not frames:
        raise ValueError(f"No wind/solar rows parsed from fuel mix report for {year}")
    long = pd.concat(frames, ignore_index=True)
    # Localize at the 15-min level so the repeated November hour lands on two
    # distinct tz-aware timestamps instead of double-counting one naive hour.
    long["start"] = long["start_naive"].dt.tz_localize(
        tz,
        ambiguous=~long["second_occurrence"].to_numpy(),
        nonexistent="shift_forward",
    )
    wide = long.pivot_table(index="start", columns="Fuel", values="mwh", aggfunc="sum")
    wide.columns = [str(c).lower() + "_mw" for c in wide.columns]
    hourly = wide.resample("1h").sum()  # 4 x 15-min MWh per hour = hourly avg MW
    hourly.index.name = "interval_start"
    cache.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(cache)
    return hourly


def fetch_exogenous(year: int, cache_dir: Path, tz: str) -> pd.DataFrame:
    """Assemble the hourly exogenous frame: load, wind, solar, net load (MW).

    Net load = load - wind - solar is THE scarcity driver in ERCOT: prices spike
    when demand is high exactly while renewables underdeliver.
    """
    load = fetch_native_load(year, cache_dir, tz)
    ws = fetch_wind_solar(year, cache_dir, tz)
    exog = ws.join(load, how="inner")
    exog["net_load_mw"] = exog["load_mw"] - exog["wind_mw"] - exog["solar_mw"]
    exog = exog[EXOG_COLUMNS]
    if exog.isna().any().any():
        raise ValueError("Exogenous frame has missing values after alignment")
    return exog
