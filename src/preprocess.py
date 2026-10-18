"""Download and prepare NOAA GHCN-Daily data for Melbourne Airport."""
from __future__ import annotations

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
STATION_ID = "ASN00086282"  # Melbourne Airport, Victoria
NOAA_URL = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{STATION_ID}.dly"
RAW_PATH = DATA_DIR / f"{STATION_ID}.dly"
CSV_PATH = DATA_DIR / "melbourne_weather.csv"
FEATURES = ["tmin", "tmax", "prcp", "day_sin", "day_cos", "year_sin", "year_cos"]


def download_noaa(force: bool = False) -> Path:
    """Fetch the official NOAA fixed-width GHCN-Daily station file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if force or not RAW_PATH.exists():
        request = urllib.request.Request(NOAA_URL, headers={"User-Agent": "MelbourneWeatherML/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, RAW_PATH.open("wb") as output:
            output.write(response.read())
    return RAW_PATH


def parse_ghcn_daily(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Parse TMAX, TMIN and PRCP observations from NOAA's .dly format."""
    rows: list[tuple[str, str, float]] = []
    with Path(path).open(encoding="ascii") as source:
        for line in source:
            station, year, month, element = line[:11], int(line[11:15]), int(line[15:17]), line[17:21]
            if station != STATION_ID or element not in {"TMAX", "TMIN", "PRCP"}:
                continue
            for day in range(1, 32):
                offset = 21 + (day - 1) * 8
                value, measurement_flag = int(line[offset:offset + 5]), line[offset + 6]
                if value == -9999 or measurement_flag == "M":
                    continue
                try:
                    date = pd.Timestamp(year=year, month=month, day=day)
                except ValueError:
                    continue
                # GHCN stores temperature and precipitation in tenths of °C and mm.
                rows.append((date.strftime("%Y-%m-%d"), element, value / 10.0))
    if not rows:
        raise ValueError(f"No usable NOAA observations found for {STATION_ID} in {path}.")
    frame = pd.DataFrame(rows, columns=["date", "element", "value"])
    frame = frame.pivot_table(index="date", columns="element", values="value", aggfunc="mean")
    frame = frame.rename(columns={"TMIN": "tmin", "TMAX": "tmax", "PRCP": "prcp"})
    frame.index = pd.to_datetime(frame.index)
    return frame.reindex(columns=["tmin", "tmax", "prcp"]).sort_index()


def prepare_weather_data(force_download: bool = False) -> pd.DataFrame:
    """Load/refresh NOAA data, fill gaps conservatively and add calendar cycles."""
    if force_download or not CSV_PATH.exists():
        raw = download_noaa(force=force_download)
        weather = parse_ghcn_daily(raw)
        weather.to_csv(CSV_PATH, index_label="date")
    else:
        weather = pd.read_csv(CSV_PATH, parse_dates=["date"], index_col="date")
    weather = weather.apply(pd.to_numeric, errors="coerce").asfreq("D")
    weather["prcp"] = weather["prcp"].clip(lower=0).fillna(0)
    weather[["tmin", "tmax"]] = weather[["tmin", "tmax"]].interpolate(limit=7).ffill().bfill()
    if weather[["tmin", "tmax", "prcp"]].isna().any().any():
        weather = weather.dropna(subset=["tmin", "tmax", "prcp"])
    day = weather.index.dayofyear
    year_length = np.where(weather.index.is_leap_year, 366, 365)
    weather["day_sin"] = np.sin(2 * np.pi * day / year_length)
    weather["day_cos"] = np.cos(2 * np.pi * day / year_length)
    year_day = (weather.index.year - weather.index.year.min()) / max(1, weather.index.year.max() - weather.index.year.min())
    weather["year_sin"] = np.sin(2 * np.pi * year_day)
    weather["year_cos"] = np.cos(2 * np.pi * year_day)
    weather.index.name = "date"
    return weather


def make_sequences(frame: pd.DataFrame, lookback: int = 30):
    """Return windows of observed features and next-day precipitation targets."""
    values = frame[FEATURES].to_numpy(dtype="float32")
    targets = frame["prcp"].to_numpy(dtype="float32")
    x, y = [], []
    for end in range(lookback, len(frame)):
        x.append(values[end - lookback:end])
        y.append(targets[end])
    return np.asarray(x), np.asarray(y)
