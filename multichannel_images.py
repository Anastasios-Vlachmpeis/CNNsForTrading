"""
Build multi-channel 224x224 financial images from OHLCV + indicators.

Channel layout (index -> what it encodes, how pixels are drawn):
  0  close        : min-max normalized close over 60-day window, bars from bottom
  1  volume       : min-max normalized volume, bars from bottom
  2  atr          : min-max normalized ATR (volatility), bars from bottom
  3  sma          : min-max normalized SMA level, bars from bottom
  4  ema          : min-max normalized EMA level, bars from bottom
  5  rsi          : fixed 0-100 scale -> bar height (not window min-max)
  6  mfi          : fixed 0-100 scale -> bar height
  7  close_vs_sma : (close - sma) / sma per day, symmetric clip + gray bars
  8  daily_return : day-over-day % change on close, symmetric clip + gray bars
  9  macd         : min-max normalized MACD line, bars from bottom

Image axes (each channel):
  - width  (x): time, 60 trading days left -> right
  - height (y): feature magnitude; 255=white background, 0=black fill from bottom
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd

WINDOW_DAYS = 60
IMG_SIZE = 224
FWD_HORIZON = 15
LABEL_DAY = WINDOW_DAYS - 1  # last visible day in the window (index 59)

CHANNEL_NAMES = (
    "close",
    "volume",
    "atr",
    "sma",
    "ema",
    "rsi",
    "mfi",
    "close_vs_sma",
    "daily_return",
    "macd",
)


def _window_minmax(values: np.ndarray) -> np.ndarray:
    """Map a 60-day series to [0, 1] using only that window (no future leak)."""
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    out = (values - vmin) / (vmax - vmin)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def _fixed_scale(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map values from [low, high] to [0, 1] (e.g. RSI/MFI on 0-100)."""
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    out = (values - low) / (high - low)
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0).astype(np.float32)


def _symmetric_clip(values: np.ndarray, clip: float = 0.05) -> np.ndarray:
    """Map signed series (e.g. returns) to [0, 1] with 0.5 = neutral."""
    clipped = np.clip(values, -clip, clip)
    return ((clipped / clip) * 0.5 + 0.5).astype(np.float32)


def _draw_bars(norm_60: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """
    Rasterize 60 normalized values into a size x size grayscale channel.
    Each day gets an equal horizontal slice; bar height = norm * (size - 1).
    """
    canvas = np.full((size, size), 255, dtype=np.uint8)
    n = len(norm_60)
    for d in range(n):
        x0 = int(d * size / n)
        x1 = int((d + 1) * size / n)
        if x1 <= x0:
            x1 = x0 + 1
        bar_h = int(float(norm_60[d]) * (size - 1))
        for x in range(x0, min(x1, size)):
            for k in range(bar_h):
                y = size - 1 - k
                if y >= 0:
                    canvas[y, x] = 0
    return canvas


def _series_from_window(frame: pd.DataFrame, col: str, start: int) -> np.ndarray:
    return frame[col].iloc[start : start + WINDOW_DAYS].to_numpy(dtype=np.float64)


def build_sample_channels(frame: pd.DataFrame, start: int) -> np.ndarray:
    """
    One sample -> array shape (n_channels, IMG_SIZE, IMG_SIZE), dtype uint8.
    `start` is the row index in `frame` where the 60-day window begins.
    """
    close = _series_from_window(frame, "Adj Close", start)
    volume = _series_from_window(frame, "Volume", start)
    atr = _series_from_window(frame, "atr", start)
    sma = _series_from_window(frame, "sma", start)
    ema = _series_from_window(frame, "ema", start)
    rsi = _series_from_window(frame, "rsi", start)
    mfi = _series_from_window(frame, "mfi", start)
    macd = _series_from_window(frame, "macd", start)

    with np.errstate(divide="ignore", invalid="ignore"):
        close_vs_sma = np.where(sma != 0, (close - sma) / sma, 0.0)
    daily_return = np.zeros(WINDOW_DAYS, dtype=np.float64)
    if WINDOW_DAYS > 1:
        prev = close[:-1]
        cur = close[1:]
        daily_return[1:] = np.where(prev != 0, (cur - prev) / prev, 0.0)

    norms = [
        _window_minmax(close),
        _window_minmax(volume),
        _window_minmax(atr),
        _window_minmax(sma),
        _window_minmax(ema),
        _fixed_scale(rsi, 0.0, 100.0),
        _fixed_scale(mfi, 0.0, 100.0),
        _symmetric_clip(close_vs_sma, clip=0.05),
        _symmetric_clip(daily_return, clip=0.05),
        _window_minmax(macd),
    ]

    channels = np.stack([_draw_bars(n) for n in norms], axis=0)
    return channels


def compute_label(
    frame: pd.DataFrame,
    start: int,
    f_sliding: float,
    s_sliding: float,
) -> tuple[int, float]:
    """
    Label from 15-day forward return after the last day in the 60-day window.
    Same 3-class rule as the original notebook (Buy/Hold/Sell).
    """
    idx_now = start + LABEL_DAY
    idx_fwd = idx_now + FWD_HORIZON
    price_now = float(frame["Adj Close"].iloc[idx_now])
    price_fwd = float(frame["Adj Close"].iloc[idx_fwd])
    dif_ratio = ((price_fwd - price_now) / price_now) * 100.0

    if dif_ratio >= s_sliding:
        label = 1
    elif dif_ratio > f_sliding:
        label = 0
    else:
        label = 2
    return label, round(price_now, 2)


def create_multichannel_dataset(
    frame: pd.DataFrame,
    output_path: str,
    f_sliding: float,
    s_sliding: float,
) -> tuple[int, Sequence[str]]:
    """
    Build all samples and save:
      X        : (n, 10, 224, 224) uint8
      y        : (n,) int64
      prices   : (n,) float64
      channels : channel name list
    """
    required = [
        "Adj Close",
        "Volume",
        "atr",
        "sma",
        "ema",
        "rsi",
        "mfi",
        "macd",
    ]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in frame: {missing}")

    data = frame.dropna().copy()
    max_start = len(data) - WINDOW_DAYS - FWD_HORIZON
    if max_start < 0:
        raise ValueError("Not enough rows after dropna() for 60-day window + 15-day label horizon.")

    images = []
    labels = []
    prices = []

    for start in range(max_start + 1):
        images.append(build_sample_channels(data, start))
        label, price = compute_label(data, start, f_sliding, s_sliding)
        labels.append(label)
        prices.append(price)

    X = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int64)
    p = np.array(prices, dtype=np.float64)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        prices=p,
        channels=np.array(CHANNEL_NAMES),
        window_days=np.array([WINDOW_DAYS]),
        img_size=np.array([IMG_SIZE]),
    )
    return len(y), CHANNEL_NAMES


def load_multichannel_dataset(path: str) -> dict[str, np.ndarray]:
    """Load .npz produced by create_multichannel_dataset."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
