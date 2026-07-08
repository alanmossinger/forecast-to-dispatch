"""Ingest wholesale power prices (ERCOT primary, CAISO comparison) into one hourly panel.

Why this matters: every downstream number — forecast skill, dispatch schedules,
the headline revenue-capture figure — settles against the prices assembled here.
A misaligned timestamp or a silently missing scarcity day would corrupt the whole
proof. This module pulls day-ahead (DAM) settlement point prices, real-time (RTM)
settlement point prices, and ancillary-service clearing prices (MCPC) from public
`gridstatus`/ERCOT archives, aligns them on one hourly, timezone-aware index, and
fails loudly on any gap.

Data-source notes (verified against gridstatus 0.36.0 on 2026-07-07):
- ERCOT's daily MIS reports (``get_spp``, ``get_as_prices``) expire after a
  retention window, so a 2024 study period must use ERCOT's *historical annual
  archives* instead:
    * DAM SPP:  ``Ercot.get_dam_spp(year)``   (report NP4-180-ER)
    * RTM SPP:  ``Ercot.get_rtm_spp(year)``   (report NP6-785-ER, 15-minute)
    * AS MCPC:  report type 13091 "Historical DAM Clearing Prices for Capacity"
      (annual ``DAMASMCPC_<year>.csv``) — gridstatus has no public wrapper for
      this archive, so we fetch it through the same internal document API that
      ``get_dam_spp`` uses.
- RTM prices are published per 15-minute settlement interval; we average the
  four intervals to an hourly price. This is the standard simplification for
  an hourly dispatch model and is stated on every figure that uses RTM data.

Sign convention: ``dart_spread = dam_price - rtm_price`` (positive = day-ahead
premium). The DART spread is the basic arbitrage product between the two markets.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from forecast_to_dispatch.config import load_config, resolve_path

# ERCOT MIS report type: "Historical DAM Clearing Prices for Capacity" (annual CSVs).
HISTORICAL_DAM_MCPC_RTID = 13091

AS_PRODUCTS = ["REGUP", "REGDN", "RRS", "NSPIN", "ECRS"]

# Canonical schema of the processed panel. Everything downstream depends on it.
PANEL_COLUMNS = ["dam_price", "rtm_price", "dart_spread"] + [p.lower() for p in AS_PRODUCTS]


# ---------------------------------------------------------------------------
# ERCOT raw pulls (cached per year, per source)
# ---------------------------------------------------------------------------


def _cached_pull(cache_file: Path, pull) -> pd.DataFrame:
    """Return a cached parquet if present, else pull live and cache.

    Why this matters: the annual archives are 10-100 MB downloads. Caching per
    (source, year) means an interrupted 6-month ingest resumes instead of
    restarting, and re-runs are offline and deterministic.
    """
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    df = pull()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file)
    return df


def _fetch_ercot_dam_year(iso: Any, year: int, cache_dir: Path) -> pd.DataFrame:
    return _cached_pull(cache_dir / f"ercot_dam_spp_{year}.parquet", lambda: iso.get_dam_spp(year))


def _fetch_ercot_rtm_year(iso: Any, year: int, cache_dir: Path) -> pd.DataFrame:
    return _cached_pull(cache_dir / f"ercot_rtm_spp_{year}.parquet", lambda: iso.get_rtm_spp(year))


def _fetch_ercot_mcpc_year(iso: Any, year: int, cache_dir: Path) -> pd.DataFrame:
    """Pull the annual 'Historical DAM Clearing Prices for Capacity' CSV.

    Why this matters: ancillary-service clearing prices are what make revenue
    *stacking* possible — without them the battery is an energy-only arbitrage
    asset and the RTC+B story (and ~a third of the revenue) disappears.
    """

    def pull() -> pd.DataFrame:
        doc = iso._get_document(  # no public wrapper exists; see module docstring
            report_type_id=HISTORICAL_DAM_MCPC_RTID,
            constructed_name_contains=f"DAMASMCPC_{year}",
        )
        resp = requests.get(doc.url, timeout=300)
        resp.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        (inner,) = z.namelist()
        df = pd.read_csv(z.open(inner))
        df.columns = [c.strip() for c in df.columns]  # source has 'REGUP ' with a space
        return df

    return _cached_pull(cache_dir / f"ercot_dam_mcpc_{year}.parquet", pull)


def _mcpc_to_hourly(raw: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Convert the MCPC CSV (Delivery Date + Hour Ending + DST flag) to a tz-aware
    hour-beginning index, matching the SPP convention."""
    hour_ending = raw["Hour Ending"].str.split(":").str[0].astype(int)
    naive = pd.to_datetime(raw["Delivery Date"], format="%m/%d/%Y") + pd.to_timedelta(
        hour_ending - 1, unit="h"
    )
    # DST: 'Repeated Hour Flag' == Y marks the second occurrence of the repeated
    # hour in November; spring-forward gaps are nonexistent local times.
    ambiguous = raw["Repeated Hour Flag"].astype(str).str.strip().ne("Y").to_numpy()
    idx = naive.dt.tz_localize(tz, ambiguous=ambiguous, nonexistent="shift_forward")
    out = raw[AS_PRODUCTS].copy()
    out.columns = [c.lower() for c in AS_PRODUCTS]
    out.index = pd.DatetimeIndex(idx, name="interval_start")
    return out.sort_index()


def fetch_ercot_prices(
    start: pd.Timestamp, end: pd.Timestamp, cache_dir: Path, hub: str, tz: str
) -> pd.DataFrame:
    """Assemble the hourly ERCOT price panel for [start, end) at one trading hub.

    Returns a DataFrame indexed by tz-aware hour-beginning ``interval_start``
    with columns: dam_price, rtm_price, dart_spread, regup, regdn, rrs, nspin, ecrs.
    """
    import gridstatus

    iso = gridstatus.Ercot()
    years = range(start.year, end.year + 1)

    dam_parts, rtm_parts, as_parts = [], [], []
    for year in years:
        print(f"[ingest] ERCOT {year}: DAM SPP ...")
        dam_parts.append(_fetch_ercot_dam_year(iso, year, cache_dir))
        print(f"[ingest] ERCOT {year}: RTM SPP (15-min) ...")
        rtm_parts.append(_fetch_ercot_rtm_year(iso, year, cache_dir))
        print(f"[ingest] ERCOT {year}: AS MCPC ...")
        as_parts.append(_fetch_ercot_mcpc_year(iso, year, cache_dir))

    dam = pd.concat(dam_parts, ignore_index=True)
    rtm = pd.concat(rtm_parts, ignore_index=True)
    mcpc = pd.concat(as_parts, ignore_index=True)

    dam_hub = dam[dam["Location"] == hub].set_index("Interval Start")["SPP"].sort_index()
    if dam_hub.empty:
        raise ValueError(
            f"No DAM prices for hub {hub}; available: {sorted(dam['Location'].unique())}"
        )
    rtm_hub = rtm[rtm["Location"] == hub].set_index("Interval Start")["SPP"].sort_index()
    if rtm_hub.empty:
        raise ValueError(f"No RTM prices for hub {hub}")

    # 15-min settlement intervals -> hourly average price (see module docstring).
    rtm_hourly = rtm_hub.resample("1h").mean()
    as_hourly = _mcpc_to_hourly(mcpc, tz)

    panel = pd.DataFrame({"dam_price": dam_hub, "rtm_price": rtm_hourly}).join(as_hourly)
    panel["dart_spread"] = panel["dam_price"] - panel["rtm_price"]
    panel = panel[PANEL_COLUMNS]
    panel.index.name = "interval_start"
    panel = panel.tz_convert(tz)

    start_l, end_l = pd.Timestamp(start, tz=tz), pd.Timestamp(end, tz=tz)
    panel = panel.loc[(panel.index >= start_l) & (panel.index < end_l)]
    _validate_panel(panel, start_l, end_l)
    return panel


def fetch_caiso_prices(
    start: pd.Timestamp, end: pd.Timestamp, cache_dir: Path, hub: str, tz: str
) -> pd.DataFrame:
    """CAISO equivalent of :func:`fetch_ercot_prices` (energy prices only).

    Why this matters: the CAISO panel is the generalization check — the same
    forecaster and dispatcher run on a second ISO to show the pattern is not
    ERCOT-specific. Ancillary products differ across ISOs, so the comparison
    panel is energy-only (AS columns are NaN) and is never used for the
    headline revenue numbers.
    """
    import gridstatus

    iso = gridstatus.CAISO()

    def pull_dam() -> pd.DataFrame:
        return iso.get_lmp(date=start, end=end, market="DAY_AHEAD_HOURLY", locations=[hub])

    def pull_rtm() -> pd.DataFrame:
        return iso.get_lmp(date=start, end=end, market="REAL_TIME_15_MIN", locations=[hub])

    stamp = f"{start:%Y%m%d}_{end:%Y%m%d}"
    dam = _cached_pull(cache_dir / f"caiso_dam_lmp_{stamp}.parquet", pull_dam)
    rtm = _cached_pull(cache_dir / f"caiso_rtm_lmp_{stamp}.parquet", pull_rtm)

    dam_s = dam.set_index("Interval Start")["LMP"].sort_index().tz_convert(tz)
    rtm_s = rtm.set_index("Interval Start")["LMP"].sort_index().tz_convert(tz).resample("1h").mean()

    panel = pd.DataFrame({"dam_price": dam_s, "rtm_price": rtm_s})
    panel["dart_spread"] = panel["dam_price"] - panel["rtm_price"]
    for p in AS_PRODUCTS:
        panel[p.lower()] = float("nan")
    panel = panel[PANEL_COLUMNS]
    panel.index.name = "interval_start"
    start_l, end_l = pd.Timestamp(start, tz=tz), pd.Timestamp(end, tz=tz)
    return panel.loc[(panel.index >= start_l) & (panel.index < end_l)]


def _validate_panel(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Fail loudly on gaps or missing prices — silent holes become fake revenue."""
    expected = pd.date_range(start, end, freq="1h", inclusive="left", tz=start.tz)
    missing_hours = expected.difference(panel.index)
    if len(missing_hours) > 0:
        raise ValueError(
            f"Price panel is missing {len(missing_hours)} hours, e.g. {missing_hours[:5].tolist()}"
        )
    na_counts = panel[PANEL_COLUMNS].isna().sum()
    if na_counts.any():
        raise ValueError(f"Price panel has missing values:\n{na_counts[na_counts > 0]}")


# ---------------------------------------------------------------------------
# Sample handling (offline mode) and CLI
# ---------------------------------------------------------------------------


def make_sample(panel: pd.DataFrame, sample_start: str, sample_end: str, out: Path) -> Path:
    """Slice the committed offline sample from a full panel and write it.

    Why this matters: the sample is what lets CI and any reviewer run the whole
    pipeline with zero network access — reproducibility is a governance gate,
    not a convenience.
    """
    tz = panel.index.tz
    lo = pd.Timestamp(sample_start, tz=tz)
    hi = pd.Timestamp(sample_end, tz=tz) + pd.Timedelta(days=1)  # inclusive end date
    sample = panel.loc[(panel.index >= lo) & (panel.index < hi)]
    if sample.empty:
        raise ValueError(f"Sample window {sample_start}..{sample_end} not covered by panel")
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out)
    return out


def load_sample(config: dict[str, Any]) -> pd.DataFrame:
    sample_file = resolve_path(config, "sample") / "ercot_prices_sample.parquet"
    if not sample_file.exists():
        raise FileNotFoundError(
            f"Committed sample not found: {sample_file}. Run a live ingest with --make-sample first."
        )
    return pd.read_parquet(sample_file)


def run(config: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    """Pipeline stage entry point: produce data/processed/ercot_prices.parquet."""
    processed = resolve_path(config, "processed") / "ercot_prices.parquet"
    processed.parent.mkdir(parents=True, exist_ok=True)

    if getattr(args, "sample", False):
        panel = load_sample(config)
        print(
            f"[ingest] offline sample: {len(panel)} hours "
            f"({panel.index.min()} .. {panel.index.max()})"
        )
    else:
        start = pd.Timestamp(getattr(args, "start", None) or config["dates"]["start"])
        end = pd.Timestamp(getattr(args, "end", None) or config["dates"]["end"])
        panel = fetch_ercot_prices(
            start,
            end,
            cache_dir=resolve_path(config, "raw"),
            hub=config["market"]["hub"],
            tz=config["market"]["timezone"],
        )
        print(
            f"[ingest] live panel: {len(panel)} hours ({panel.index.min()} .. {panel.index.max()})"
        )
        if getattr(args, "make_sample", False):
            out = make_sample(
                panel,
                config["dates"]["sample_start"],
                config["dates"]["sample_end"],
                resolve_path(config, "sample") / "ercot_prices_sample.parquet",
            )
            print(f"[ingest] committed sample written: {out}")

    panel.to_parquet(processed)
    print(f"[ingest] wrote {processed}")
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="YYYY-MM-DD (default from config)")
    parser.add_argument("--end", help="YYYY-MM-DD (default from config)")
    parser.add_argument("--sample", action="store_true", help="offline: use the committed sample")
    parser.add_argument(
        "--make-sample", action="store_true", help="after a live pull, write the committed sample"
    )
    args = parser.parse_args()
    run(load_config(), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
