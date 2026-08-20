from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import talib
import yfinance as yf

METHODS = {
    "MomentumLeader": "Momentum Leader",
    "TrendConfirmation": "Trend Confirmation",
    "RelativeStrength": "Relative Strength",
}

DEFAULT_START_DATE = "2021-01-01"
LOOKBACK_DAYS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252, "2y": 504}
LIQUIDITY_LOOKBACK_DAYS = 20
SLOPE_50_LOOKBACK_DAYS = 20
SLOPE_200_LOOKBACK_DAYS = 40
TREND_RISING_DAYS_REQUIRED = 15
TREND_RISING_WINDOW_DAYS = 20
TREND_CONFIRMATION_DAYS = 5
TREND_RECLAIM_MAX_DAYS = 10
PULLBACK_LOOKBACK_DAYS = 50
PULLBACK_MIN_ATR = 3.0
PULLBACK_MAX_CLOSES_BELOW_SMA_200 = 3
MAX_DAILY_GAIN_PCT = 400.0
MAX_DAILY_LOSS_PCT = -85.0
MAX_SHORT_WINDOW_RETURN_PCT = 500.0
SPLIT_MATCH_FACTOR_TOLERANCE = 1.75
NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&download=true"
NASDAQ_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BiotechUniverseBuilder/1.0)",
    "Accept": "application/json, text/plain, */*",
}
CORE_BIOTECH_INDUSTRIES = {
    "biotechnology: pharmaceutical preparations",
    "biotechnology: biological products (no diagnostic substances)",
    "biotechnology: in vitro & in vivo diagnostic substances",
    "biotechnology: commercial physical & biological resarch",
    "medicinal chemicals and botanical products",
    "misc health and biotechnology services",
    "other pharmaceuticals",
    "pharmaceuticals and biotechnology",
}
NON_COMMON_SECURITY_PATTERN = re.compile(
    r"\b(?:warrants?|units?|rights?|senior notes?|debentures?|bonds?|"
    r"preferred stock)\b",
    re.IGNORECASE,
)
BIOTECH_COMPANY_NAME_PATTERN = re.compile(
    r"\b(?:biotech(?:nology)?|biopharma(?:ceuticals?)?|therapeutics?|"
    r"pharma(?:ceuticals?)?|biosciences?|biologics?|genomics?|genetics?|"
    r"diagnostics?|life sciences?)\b",
    re.IGNORECASE,
)
BIOTECH_KEYWORDS = (
    "BIOTECH",
    "BIO",
    "GENE",
    "THERAPEUT",
    "PHARMA",
    "PHARMACEUT",
    "ONCO",
    "MED",
    "CANCER",
    "VACCINE",
    "CELL",
    "RNA",
    "DNA",
    "LIFE SCIENCE",
    "LIFESCIENCE",
    "HEALTH",
    "IMMUNO",
    "CLINICAL",
    "DIAGNOSTIC",
    "LAB",
    "RESEARCH",
)


def compute_period_performance(prices: pd.Series, lookback_days: int) -> float:
    if prices.empty or len(prices) < 2:
        return float("nan")
    if lookback_days <= 0:
        return float("nan")

    if len(prices) <= lookback_days:
        start_index = 0
    else:
        start_index = len(prices) - lookback_days

    start_price = prices.iloc[start_index]
    end_price = prices.iloc[-1]
    if start_price == 0:
        return float("nan")
    return ((end_price / start_price) - 1) * 100


def rank_tickers(results: pd.DataFrame, period_column: str) -> pd.DataFrame:
    return results.sort_values(by=period_column, ascending=False).reset_index(drop=True)


def is_biotech_candidate(name: str) -> bool:
    if not name:
        return False
    normalized = " ".join(name.upper().split())
    if any(keyword in normalized for keyword in BIOTECH_KEYWORDS):
        return True
    return "ATAI" in normalized or "ATAI" in normalized.replace("-", " ")


def normalize_industry(industry: str) -> str:
    return " ".join(str(industry or "").strip().lower().split())


def is_common_equity(name: str) -> bool:
    """Reject non-equity securities while allowing ordinary ADR/ADS shares."""
    normalized = " ".join(str(name or "").split())
    if not normalized:
        return False
    if NON_COMMON_SECURITY_PATTERN.search(normalized):
        return False
    return not re.search(r"\b(?:ETF|ETN|Fund|Portfolio|Index)\b", normalized, re.IGNORECASE)


def load_universe_overrides(path: str | Path | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    override_path = Path(path)
    if not override_path.exists():
        return {}

    frame = pd.read_csv(override_path).fillna("")
    required = {"ticker", "action", "reason"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Universe override file must contain columns: {', '.join(sorted(required))}")

    overrides: dict[str, dict[str, str]] = {}
    for row in frame.to_dict(orient="records"):
        ticker = str(row["ticker"]).strip().upper()
        action = str(row["action"]).strip().lower()
        if not ticker:
            continue
        if action not in {"include", "exclude"}:
            raise ValueError(f"Invalid universe override action for {ticker}: {action}")
        overrides[ticker] = {"action": action, "reason": str(row["reason"]).strip()}
    return overrides


def classify_biotech_listing(
    ticker: str,
    name: str,
    industry: str,
    overrides: dict[str, dict[str, str]] | None = None,
) -> tuple[bool, str]:
    symbol = str(ticker or "").strip().upper()
    override = (overrides or {}).get(symbol)
    if override and override["action"] == "exclude":
        return False, f"override exclude: {override['reason']}"
    if not symbol or not is_common_equity(name):
        return False, "not common equity"
    if override and override["action"] == "include":
        return True, f"override include: {override['reason']}"
    if normalize_industry(industry) in CORE_BIOTECH_INDUSTRIES:
        return True, "Nasdaq biotech/pharma industry"
    if BIOTECH_COMPANY_NAME_PATTERN.search(name):
        return True, "explicit biotech/life-science company name"
    return False, "outside selected biotech/pharma industries"


def load_tickers(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    if "ticker" not in df.columns:
        raise ValueError(f"Expected a 'ticker' column in {csv_path}")

    tickers = df["ticker"].dropna().astype(str).str.strip().str.upper()
    return [ticker for ticker in tickers if ticker]


def normalize_split_history(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Put raw OHLC and volume on the latest share scale using verified split events."""
    normalized = history.copy()
    price_columns = [column for column in ["Open", "High", "Low", "Close"] if column in normalized.columns]
    for column in price_columns + (["Volume"] if "Volume" in normalized.columns else []):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype(float)
    split_events = (
        pd.to_numeric(normalized.get("Stock Splits", pd.Series(dtype=float)), errors="coerce")
        .fillna(0.0)
    )
    event_rows = [(index, float(ratio)) for index, ratio in split_events.items() if ratio > 0]
    applied: list[str] = []
    ignored: list[str] = []
    unresolved: list[str] = []

    if "Close" not in normalized.columns:
        return normalized, {
            "split_events_detected": len(event_rows),
            "split_adjustment_applied": False,
            "split_adjustments": "",
            "unresolved_split_events": "missing Close column",
        }

    # Later splits must be normalized first so earlier split continuity is
    # evaluated on a consistent share scale.
    for event_date, ratio in reversed(event_rows):
        close = pd.to_numeric(normalized["Close"], errors="coerce")
        before = close.loc[close.index < event_date].dropna()
        after = close.loc[close.index >= event_date].dropna()
        label = f"{pd.Timestamp(event_date).date().isoformat()} ({ratio:g})"
        if before.empty or after.empty:
            unresolved.append(f"{label}: missing surrounding prices")
            continue

        observed_ratio = float(after.iloc[0] / before.iloc[-1])
        expected_ratio = 1.0 / ratio
        if observed_ratio <= 0 or expected_ratio <= 0:
            unresolved.append(f"{label}: invalid price ratio")
            continue

        expected_distance = abs(float(np.log(observed_ratio / expected_ratio)))
        inverse_distance = abs(float(np.log(observed_ratio / ratio)))
        continuity_distance = abs(float(np.log(observed_ratio)))
        tolerance = float(np.log(SPLIT_MATCH_FACTOR_TOLERANCE))

        if expected_distance <= tolerance and expected_distance < continuity_distance:
            price_multiplier = expected_ratio
            for column in price_columns:
                normalized.loc[normalized.index < event_date, column] = (
                    pd.to_numeric(normalized.loc[normalized.index < event_date, column], errors="coerce")
                    * price_multiplier
                )
            if "Volume" in normalized.columns:
                normalized.loc[normalized.index < event_date, "Volume"] = (
                    pd.to_numeric(normalized.loc[normalized.index < event_date, "Volume"], errors="coerce")
                    * ratio
                )
            applied.append(label)
        elif inverse_distance <= tolerance and inverse_distance < continuity_distance:
            # Some feeds encode a postponed or duplicate split in the historical
            # series as an inverse discontinuity. Re-scale the earlier history
            # to continuity while retaining the event in the audit trail.
            price_multiplier = ratio
            for column in price_columns:
                normalized.loc[normalized.index < event_date, column] = (
                    normalized.loc[normalized.index < event_date, column] * price_multiplier
                )
            if "Volume" in normalized.columns:
                normalized.loc[normalized.index < event_date, "Volume"] = (
                    normalized.loc[normalized.index < event_date, "Volume"] / ratio
                )
            applied.append(f"{label} inverse correction")
        elif continuity_distance <= tolerance:
            ignored.append(f"{label}: prices already continuous")
        else:
            unresolved.append(
                f"{label}: observed {observed_ratio:.2f}x, expected {expected_ratio:.2f}x"
            )

    return normalized, {
        "split_events_detected": len(event_rows),
        "split_adjustment_applied": bool(applied),
        "split_adjustments": "; ".join(reversed(applied)),
        "ignored_split_events": "; ".join(reversed(ignored)),
        "unresolved_split_events": "; ".join(reversed(unresolved)),
    }


def evaluate_price_quality(history: pd.DataFrame, split_info: dict[str, object]) -> dict[str, object]:
    close = pd.to_numeric(history.get("Close", pd.Series(dtype=float)), errors="coerce").dropna()
    daily_returns = close.pct_change().dropna() * 100
    max_daily_gain = float(daily_returns.max()) if not daily_returns.empty else float("nan")
    max_daily_loss = float(daily_returns.min()) if not daily_returns.empty else float("nan")
    reasons: list[str] = []

    unresolved = str(split_info.get("unresolved_split_events") or "").strip()
    if unresolved:
        reasons.append(f"unresolved split: {unresolved}")
    if np.isfinite(max_daily_gain) and max_daily_gain > MAX_DAILY_GAIN_PCT:
        reasons.append(f"one-day gain {max_daily_gain:.1f}% exceeds {MAX_DAILY_GAIN_PCT:.0f}%")
    if np.isfinite(max_daily_loss) and max_daily_loss < MAX_DAILY_LOSS_PCT:
        reasons.append(f"one-day loss {max_daily_loss:.1f}% exceeds {abs(MAX_DAILY_LOSS_PCT):.0f}%")

    for days in [10, 15, 30]:
        if len(close) <= days:
            continue
        start = float(close.iloc[-days - 1])
        if start == 0:
            continue
        window_return = ((float(close.iloc[-1]) / start) - 1) * 100
        if abs(window_return) > MAX_SHORT_WINDOW_RETURN_PCT:
            reasons.append(
                f"{days}-day return {window_return:.1f}% exceeds ±{MAX_SHORT_WINDOW_RETURN_PCT:.0f}%"
            )

    return {
        "data_quality_flag": bool(reasons),
        "data_quality_reason": "; ".join(reasons),
        "max_daily_gain": max_daily_gain,
        "max_daily_loss": max_daily_loss,
        **split_info,
    }


def fetch_price_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    history = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False, actions=True)
    if isinstance(history, pd.DataFrame) and not history.empty:
        if isinstance(history.columns, pd.MultiIndex):
            close_keys = [col for col in history.columns if col[0] == "Close"]
            if close_keys:
                history = history.copy()
                history.columns = [col[0] if isinstance(col, tuple) else col for col in history.columns]
        if "Close" in history.columns:
            normalized, split_info = normalize_split_history(history)
            normalized.attrs["price_quality"] = evaluate_price_quality(normalized, split_info)
            return normalized.dropna(subset=["Close"])
    return pd.DataFrame(columns=["Close", "High", "Low", "Open", "Volume", "Stock Splits"])


def compute_volume_metrics(
    volume_series: pd.Series,
    lookback_days: int,
    close_series: pd.Series | None = None,
) -> dict[str, float]:
    if volume_series.empty:
        return {
            "latest_volume": float("nan"),
            "avg_volume_15d": float("nan"),
            "mean_volume_20d": float("nan"),
            "median_volume_20d": float("nan"),
            "median_dollar_volume_20d": float("nan"),
        }

    latest_volume = float(volume_series.iloc[-1])
    if len(volume_series) < lookback_days:
        avg_volume_15d = float(volume_series.mean())
    else:
        avg_volume_15d = float(volume_series.iloc[-lookback_days:].mean())

    volume_20d = volume_series.iloc[-LIQUIDITY_LOOKBACK_DAYS:]
    mean_volume_20d = float(volume_20d.mean())
    median_volume_20d = float(volume_20d.median())
    median_dollar_volume_20d = float("nan")
    if close_series is not None:
        aligned = pd.concat(
            [
                pd.to_numeric(close_series, errors="coerce").rename("close"),
                pd.to_numeric(volume_series, errors="coerce").rename("volume"),
            ],
            axis=1,
        ).dropna().iloc[-LIQUIDITY_LOOKBACK_DAYS:]
        if not aligned.empty:
            median_dollar_volume_20d = float((aligned["close"] * aligned["volume"]).median())

    return {
        "latest_volume": latest_volume,
        "avg_volume_15d": avg_volume_15d,
        "mean_volume_20d": mean_volume_20d,
        "median_volume_20d": median_volume_20d,
        "median_dollar_volume_20d": median_dollar_volume_20d,
    }


def compute_normalized_slope(series: pd.Series, lookback_days: int) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().iloc[-lookback_days:]
    if len(values) < 2:
        return float("nan")
    average = float(values.mean())
    if average == 0:
        return float("nan")
    slope = float(np.polyfit(np.arange(len(values)), values.to_numpy(), 1)[0])
    return slope / average * 100


def compute_raw_slope(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return float("nan")
    return float(np.polyfit(np.arange(len(values)), values.to_numpy(), 1)[0])


def liquidity_score_adjustment(median_dollar_volume: float) -> float:
    if not np.isfinite(median_dollar_volume):
        return -8.0
    if median_dollar_volume < 100_000:
        return -8.0
    if median_dollar_volume < 500_000:
        return -3.0
    if median_dollar_volume <= 2_000_000:
        return 0.0
    return 2.0


def price_score_adjustment(price: float) -> float:
    if not np.isfinite(price) or price < 1:
        return -8.0
    if price < 3:
        return -4.0
    if price < 5:
        return -2.0
    return 0.0


def rsi_score_penalty(rsi: float, rsi_change_5d: float) -> float:
    if not np.isfinite(rsi) or rsi < 75:
        return 0.0
    if rsi < 80:
        penalty = -1.0
    elif rsi < 85:
        penalty = -3.0
    else:
        penalty = -5.0
    # Cooling momentum remains extended, but receives half the penalty.
    return penalty if not np.isfinite(rsi_change_5d) or rsi_change_5d >= 0 else penalty / 2


def tiered_extension_penalty(value: float) -> float:
    if not np.isfinite(value) or value <= 4:
        return 0.0
    if value <= 6:
        return -2.0
    if value <= 8:
        return -4.0
    return -6.0


def compute_short_term_momentum(close_series: pd.Series) -> dict[str, float]:
    close = pd.to_numeric(close_series, errors="coerce").dropna()
    if close.empty:
        return {
            "short_return_10d": float("nan"),
            "short_return_15d": float("nan"),
            "short_return_30d": float("nan"),
            "rsi_14": float("nan"),
            "rsi_change_5d": float("nan"),
            "rsi_change_10d": float("nan"),
            "short_momentum_score": float("nan"),
        }

    def pct_change_from(days: int) -> float:
        available_days = min(days, len(close) - 1)
        if available_days <= 0:
            return float("nan")
        start_price = float(close.iloc[-available_days - 1])
        end_price = float(close.iloc[-1])
        if start_price == 0:
            return float("nan")
        return ((end_price / start_price) - 1) * 100

    short_return_10d = pct_change_from(10)
    short_return_15d = pct_change_from(15)
    short_return_30d = pct_change_from(30)

    rsi_values = talib.RSI(close.to_numpy(), timeperiod=14)
    if len(rsi_values) and np.isfinite(rsi_values[-1]):
        rsi_14 = float(rsi_values[-1])
    else:
        rsi_14 = 50.0

    def rsi_change(days: int) -> float:
        if len(rsi_values) <= days or not np.isfinite(rsi_values[-1]) or not np.isfinite(rsi_values[-days - 1]):
            return float("nan")
        return float(rsi_values[-1] - rsi_values[-days - 1])

    rsi_change_5d = rsi_change(5)
    rsi_change_10d = rsi_change(10)

    recent_returns = [value for value in [short_return_10d, short_return_15d, short_return_30d] if np.isfinite(value)]
    if recent_returns:
        avg_recent_return = float(np.mean(recent_returns))
    else:
        returns = close.pct_change().dropna().to_numpy()
        avg_recent_return = float(np.nanpercentile(returns, 50)) if len(returns) > 0 else 0.0

    if np.isnan(avg_recent_return):
        short_momentum_score = 0.0
    else:
        if np.isfinite(rsi_14):
            rsi_adjustment = (rsi_14 - 50) / 10
            overbought_penalty = max(0.0, (rsi_14 - 70) / 10)
        else:
            rsi_adjustment = 0.0
            overbought_penalty = 0.0
        short_momentum_score = avg_recent_return + rsi_adjustment - overbought_penalty

    return {
        "short_return_10d": short_return_10d,
        "short_return_15d": short_return_15d,
        "short_return_30d": short_return_30d,
        "rsi_14": rsi_14,
        "rsi_change_5d": rsi_change_5d,
        "rsi_change_10d": rsi_change_10d,
        "short_momentum_score": short_momentum_score,
    }


def compute_trend_confirmation(close_series: pd.Series, sma_50_series: pd.Series, sma_200_series: pd.Series) -> dict[str, object]:
    close = pd.to_numeric(close_series, errors="coerce").dropna()
    sma_50 = pd.to_numeric(sma_50_series, errors="coerce").dropna()
    sma_200 = pd.to_numeric(sma_200_series, errors="coerce").dropna()

    aligned = pd.concat([close, sma_50, sma_200], axis=1)
    aligned.columns = ["close", "sma_50", "sma_200"]
    aligned = aligned.dropna()

    if aligned.empty:
        return {"trend_confirmed": False, "trend_confirmed_on": None}

    confirmed_mask = (aligned["close"] > aligned["sma_50"]) & (aligned["sma_50"] > aligned["sma_200"])
    if not confirmed_mask.any():
        return {"trend_confirmed": False, "trend_confirmed_on": None}

    confirmed_runs = []
    run_start = None
    for idx, value in confirmed_mask.items():
        if value and run_start is None:
            run_start = idx
        elif not value and run_start is not None:
            confirmed_runs.append((run_start, idx))
            run_start = None
    if run_start is not None:
        confirmed_runs.append((run_start, aligned.index[-1]))

    if not confirmed_runs:
        return {"trend_confirmed": False, "trend_confirmed_on": None}

    min_run_length = 10
    valid_runs = [run for run in confirmed_runs if len(aligned.loc[run[0] : run[1]]) >= min_run_length]
    if not valid_runs:
        valid_runs = confirmed_runs

    latest_run = valid_runs[-1]
    confirmation_date = latest_run[0]
    return {"trend_confirmed": bool(confirmed_mask.iloc[-1]), "trend_confirmed_on": confirmation_date}


def compute_experimental_trend(
    close_series: pd.Series,
    sma_50_series: pd.Series,
    sma_200_series: pd.Series,
) -> dict[str, object]:
    aligned = pd.concat(
        [
            pd.to_numeric(close_series, errors="coerce").rename("close"),
            pd.to_numeric(sma_50_series, errors="coerce").rename("sma_50"),
            pd.to_numeric(sma_200_series, errors="coerce").rename("sma_200"),
        ],
        axis=1,
    ).dropna()
    empty = {
        "experimental_trend_confirmed": False,
        "experimental_trend_confirmed_on": None,
        "days_since_experimental_confirmation": float("nan"),
        "sma_200_rising_days_20d": 0,
        "trend_reclaim": False,
        "trend_event_type": "",
    }
    if len(aligned) < TREND_RISING_WINDOW_DAYS + TREND_CONFIRMATION_DAYS:
        return empty

    rising = aligned["sma_200"].diff().gt(0)
    rising_count = rising.rolling(TREND_RISING_WINDOW_DAYS, min_periods=TREND_RISING_WINDOW_DAYS).sum()
    conditions = (
        aligned["close"].gt(aligned["sma_50"])
        & aligned["sma_50"].gt(aligned["sma_200"])
        & rising_count.ge(TREND_RISING_DAYS_REQUIRED)
    )
    current_run_length = 0
    for value in reversed(conditions.tolist()):
        if not value:
            break
        current_run_length += 1
    confirmed = current_run_length >= TREND_CONFIRMATION_DAYS
    confirmation_date = None
    days_since = float("nan")
    if confirmed:
        run_start_position = len(aligned) - current_run_length
        confirmation_date = aligned.index[run_start_position]
        days_since = float(len(aligned) - run_start_position - 1)

    above = aligned["sma_50"].gt(aligned["sma_200"])
    crossover_positions = np.flatnonzero((above & ~above.shift(1, fill_value=False)).to_numpy())
    reclaim = False
    event_type = ""
    if len(crossover_positions):
        latest_cross = int(crossover_positions[-1])
        below_run = 0
        cursor = latest_cross - 1
        while cursor >= 0 and not bool(above.iloc[cursor]):
            below_run += 1
            cursor -= 1
        reclaim = cursor >= 0 and below_run <= TREND_RECLAIM_MAX_DAYS
        event_type = "reclaim" if reclaim else "crossover"

    return {
        "experimental_trend_confirmed": bool(confirmed),
        "experimental_trend_confirmed_on": confirmation_date,
        "days_since_experimental_confirmation": days_since,
        "sma_200_rising_days_20d": int(rising.tail(TREND_RISING_WINDOW_DAYS).sum()),
        "trend_reclaim": bool(reclaim),
        "trend_event_type": event_type,
    }


def compute_entry_diagnostics(
    close: pd.Series,
    volume: pd.Series,
    atr_series: pd.Series,
    ema_20_series: pd.Series,
    sma_50_series: pd.Series,
    sma_200_series: pd.Series,
    rsi_series: pd.Series,
    normalized_slope_200: float,
) -> dict[str, object]:
    frame = pd.concat(
        [
            close.rename("close"), volume.rename("volume"), atr_series.rename("atr"),
            ema_20_series.rename("ema_20"), sma_50_series.rename("sma_50"),
            sma_200_series.rename("sma_200"), rsi_series.rename("rsi"),
        ], axis=1,
    ).dropna(subset=["close", "atr", "ema_20", "sma_50", "sma_200"])
    if frame.empty or not np.isfinite(float(frame["atr"].iloc[-1])) or float(frame["atr"].iloc[-1]) <= 0:
        return {}

    latest = frame.iloc[-1]
    current_atr = float(latest["atr"])
    current_close = float(latest["close"])
    distance_ema_20_atr = (current_close - float(latest["ema_20"])) / current_atr
    distance_sma_50_atr = (current_close - float(latest["sma_50"])) / current_atr
    distance_sma_200_atr = (current_close - float(latest["sma_200"])) / current_atr
    normalized_atr_pct = current_atr / current_close * 100 if current_close else float("nan")

    recent = frame.iloc[-PULLBACK_LOOKBACK_DAYS:]
    # A centered three-day median prevents an isolated one-day spike from
    # becoming the sole anchor for a purported constructive pullback.
    reference_highs = recent["close"].rolling(3, center=True, min_periods=2).median()
    peak_position = int(np.argmax(reference_highs.to_numpy()))
    peak_date = recent.index[peak_position]
    peak_close = float(reference_highs.iloc[peak_position])
    peak_atr = float(recent["atr"].iloc[peak_position])
    pullback = recent.iloc[peak_position:]
    pullback_duration = len(pullback) - 1
    pullback_pct = (current_close / peak_close - 1) * 100 if peak_close else float("nan")
    below_high_current_atr = (peak_close - current_close) / current_atr
    below_high_peak_atr = (peak_close - current_close) / peak_atr if peak_atr > 0 else float("nan")
    closes_below_sma_200 = int(pullback["close"].lt(pullback["sma_200"]).sum())

    median_volume = frame["volume"].rolling(LIQUIDITY_LOOKBACK_DAYS, min_periods=5).median()
    prior_close = frame["close"].shift(1)
    damaging = (
        frame["close"].sub(prior_close).le(-1.5 * frame["atr"])
        & frame["volume"].ge(1.5 * median_volume)
    )
    damaging_in_pullback = bool(damaging.loc[damaging.index >= peak_date].any())
    rsi_context = frame.loc[frame.index >= peak_date - pd.Timedelta(days=15), "rsi"].dropna()
    peak_rsi = float(rsi_context.max()) if not rsi_context.empty else float("nan")
    current_rsi = float(frame["rsi"].dropna().iloc[-1]) if frame["rsi"].notna().any() else float("nan")
    constructive_pullback = bool(
        normalized_slope_200 > 0
        and pullback_duration > 0
        and below_high_current_atr >= PULLBACK_MIN_ATR
        and closes_below_sma_200 <= PULLBACK_MAX_CLOSES_BELOW_SMA_200
        and np.isfinite(peak_rsi) and peak_rsi >= 70
        and np.isfinite(current_rsi) and 40 <= current_rsi <= 65
    )

    def advance_in_atr(days: int) -> float:
        if len(frame) <= days:
            return float("nan")
        return (current_close - float(frame["close"].iloc[-days - 1])) / current_atr

    extension_series = (frame["close"] - frame["ema_20"]) / frame["atr"]
    max_ema_20_extension_30d = float(extension_series.tail(30).max())
    current_extension = max(distance_ema_20_atr, distance_sma_50_atr)
    verticality = max(
        value for value in [advance_in_atr(5), advance_in_atr(15), advance_in_atr(30), max_ema_20_extension_30d]
        if np.isfinite(value)
    )
    current_extension_penalty = tiered_extension_penalty(current_extension)
    # Once current extension has cooled below four ATR, an older vertical move
    # no longer creates a stale overextension penalty by itself.
    verticality_penalty = (
        tiered_extension_penalty(verticality) if distance_ema_20_atr > 4 else 0.0
    )
    overextension_penalty = max(-8.0, current_extension_penalty + verticality_penalty)
    prior_trend = compute_experimental_trend(
        close.loc[:peak_date], sma_50_series.loc[:peak_date], sma_200_series.loc[:peak_date]
    )

    return {
        "ema_20": float(latest["ema_20"]),
        "normalized_atr_pct": normalized_atr_pct,
        "distance_ema_20_atr": distance_ema_20_atr,
        "distance_sma_50_atr": distance_sma_50_atr,
        "distance_sma_200_atr": distance_sma_200_atr,
        "atr_below_50d_high_current": below_high_current_atr,
        "atr_below_50d_high_peak": below_high_peak_atr,
        "pullback_from_50d_high_pct": pullback_pct,
        "pullback_peak_date": peak_date,
        "pullback_duration_days": int(pullback_duration),
        "pullback_closes_below_sma_200": closes_below_sma_200,
        "damaging_high_volume_decline": damaging_in_pullback,
        "pullback_peak_rsi": peak_rsi,
        "constructive_pullback": constructive_pullback,
        "trend_confirmed_before_pullback_on": prior_trend["experimental_trend_confirmed_on"],
        "advance_5d_atr": advance_in_atr(5),
        "advance_15d_atr": advance_in_atr(15),
        "advance_30d_atr": advance_in_atr(30),
        "max_ema_20_extension_30d_atr": max_ema_20_extension_30d,
        "current_extension_penalty": current_extension_penalty,
        "verticality_penalty": verticality_penalty,
        "overextension_penalty": overextension_penalty,
    }


def compute_indicators(history: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    high = pd.to_numeric(history["High"], errors="coerce").dropna()
    low = pd.to_numeric(history["Low"], errors="coerce").dropna()
    volume = pd.to_numeric(history["Volume"], errors="coerce").dropna()

    if len(close) < 200:
        raise ValueError("insufficient history")

    atr = talib.ATR(high.to_numpy(), low.to_numpy(), close.to_numpy(), timeperiod=15)
    ema_20 = talib.EMA(close.to_numpy(), timeperiod=20)
    sma_50 = talib.SMA(close.to_numpy(), timeperiod=50)
    sma_200 = talib.SMA(close.to_numpy(), timeperiod=200)
    rsi_14_values = talib.RSI(close.to_numpy(), timeperiod=14)

    atr_series = pd.Series(atr, index=close.index)
    ema_20_series = pd.Series(ema_20, index=close.index)
    sma_50_series = pd.Series(sma_50, index=close.index)
    sma_200_series = pd.Series(sma_200, index=close.index)
    rsi_14_series = pd.Series(rsi_14_values, index=close.index)

    latest_date = close.index[-1].strftime("%Y-%m-%d")
    latest_close = float(close.iloc[-1])
    atr_15d = float(atr_series.iloc[-1])
    volume_metrics = compute_volume_metrics(volume, 15, close)
    sma_50 = float(sma_50_series.iloc[-1])
    sma_200 = float(sma_200_series.iloc[-1])

    slope_50 = compute_raw_slope(sma_50_series)
    slope_200 = compute_raw_slope(sma_200_series)
    normalized_slope_50 = compute_normalized_slope(sma_50_series, SLOPE_50_LOOKBACK_DAYS)
    normalized_slope_200 = compute_normalized_slope(sma_200_series, SLOPE_200_LOOKBACK_DAYS)
    short_term_momentum = compute_short_term_momentum(close)
    trend_confirmation = compute_trend_confirmation(close, sma_50_series, sma_200_series)
    experimental_trend = compute_experimental_trend(close, sma_50_series, sma_200_series)
    entry_diagnostics = compute_entry_diagnostics(
        close, volume, atr_series, ema_20_series, sma_50_series, sma_200_series,
        rsi_14_series, normalized_slope_200,
    )

    return pd.Series(
        {
            "as_of_date": latest_date,
            "closing_price": latest_close,
            "atr_15d": atr_15d,
            "latest_volume": volume_metrics["latest_volume"],
            "avg_volume_15d": volume_metrics["avg_volume_15d"],
            "mean_volume_20d": volume_metrics["mean_volume_20d"],
            "median_volume_20d": volume_metrics["median_volume_20d"],
            "median_dollar_volume_20d": volume_metrics["median_dollar_volume_20d"],
            "volume_1p5x": volume_metrics["latest_volume"] > 1.5 * volume_metrics["avg_volume_15d"],
            "sma_50": sma_50,
            "slope_50": slope_50,
            "normalized_slope_50_20d": normalized_slope_50,
            "sma_200": sma_200,
            "slope_200": slope_200,
            "normalized_slope_200_40d": normalized_slope_200,
            "short_return_10d": short_term_momentum["short_return_10d"],
            "short_return_15d": short_term_momentum["short_return_15d"],
            "short_return_30d": short_term_momentum["short_return_30d"],
            "rsi_14": short_term_momentum["rsi_14"],
            "rsi_change_5d": short_term_momentum["rsi_change_5d"],
            "rsi_change_10d": short_term_momentum["rsi_change_10d"],
            "short_momentum_score": short_term_momentum["short_momentum_score"],
            "trend_confirmed": trend_confirmation["trend_confirmed"],
            "trend_confirmed_on": trend_confirmation["trend_confirmed_on"],
            **experimental_trend,
            **entry_diagnostics,
        }
    )


def build_rankings(tickers: list[str], start_date: str = DEFAULT_START_DATE, end_date: str | None = None) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    results: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []

    for ticker in tickers:
        try:
            history = fetch_price_history(ticker, start=start_date, end=end_date)
            if history.empty or "Close" not in history.columns:
                failures.append((ticker, "no price history"))
                continue

            close_series = pd.to_numeric(history["Close"], errors="coerce").dropna()
            if len(close_series) < 200:
                failures.append((ticker, "insufficient history"))
                continue

            latest = close_series.iloc[-1]
            indicators = compute_indicators(history)
            quality = history.attrs.get("price_quality", evaluate_price_quality(history, {}))
            results.append(
                {
                    "ticker": ticker,
                    "as_of_date": indicators["as_of_date"],
                    "closing_price": round(float(indicators["closing_price"]), 2),
                    "atr_15d": round(float(indicators["atr_15d"]), 2),
                    "latest_volume": round(float(indicators["latest_volume"]), 2),
                    "avg_volume_15d": round(float(indicators["avg_volume_15d"]), 2),
                    "mean_volume_20d": round(float(indicators["mean_volume_20d"]), 2),
                    "median_volume_20d": round(float(indicators["median_volume_20d"]), 2),
                    "median_dollar_volume_20d": round(float(indicators["median_dollar_volume_20d"]), 2),
                    "volume_1p5x": bool(indicators["volume_1p5x"]),
                    "sma_50": round(float(indicators["sma_50"]), 2),
                    "slope_50": round(float(indicators["slope_50"]), 4),
                    "normalized_slope_50_20d": round(float(indicators["normalized_slope_50_20d"]), 4),
                    "sma_200": round(float(indicators["sma_200"]), 2),
                    "slope_200": round(float(indicators["slope_200"]), 4),
                    "normalized_slope_200_40d": round(float(indicators["normalized_slope_200_40d"]), 4),
                    "short_return_10d": round(float(indicators["short_return_10d"]), 2),
                    "short_return_15d": round(float(indicators["short_return_15d"]), 2),
                    "short_return_30d": round(float(indicators["short_return_30d"]), 2),
                    "rsi_14": round(float(indicators["rsi_14"]), 2),
                    "rsi_change_5d": round(float(indicators["rsi_change_5d"]), 2),
                    "rsi_change_10d": round(float(indicators["rsi_change_10d"]), 2),
                    "short_momentum_score": round(float(indicators["short_momentum_score"]), 2),
                    "trend_confirmed": bool(indicators["trend_confirmed"]),
                    "trend_confirmed_on": indicators["trend_confirmed_on"].strftime("%Y-%m-%d") if pd.notna(indicators["trend_confirmed_on"]) else None,
                    "experimental_trend_confirmed": bool(indicators["experimental_trend_confirmed"]),
                    "experimental_trend_confirmed_on": indicators["experimental_trend_confirmed_on"].strftime("%Y-%m-%d") if pd.notna(indicators["experimental_trend_confirmed_on"]) else None,
                    "days_since_experimental_confirmation": indicators["days_since_experimental_confirmation"],
                    "sma_200_rising_days_20d": int(indicators["sma_200_rising_days_20d"]),
                    "trend_reclaim": bool(indicators["trend_reclaim"]),
                    "trend_event_type": str(indicators["trend_event_type"]),
                    "ema_20": round(float(indicators.get("ema_20", float("nan"))), 2),
                    "normalized_atr_pct": round(float(indicators.get("normalized_atr_pct", float("nan"))), 2),
                    "distance_ema_20_atr": round(float(indicators.get("distance_ema_20_atr", float("nan"))), 2),
                    "distance_sma_50_atr": round(float(indicators.get("distance_sma_50_atr", float("nan"))), 2),
                    "distance_sma_200_atr": round(float(indicators.get("distance_sma_200_atr", float("nan"))), 2),
                    "atr_below_50d_high_current": round(float(indicators.get("atr_below_50d_high_current", float("nan"))), 2),
                    "atr_below_50d_high_peak": round(float(indicators.get("atr_below_50d_high_peak", float("nan"))), 2),
                    "pullback_from_50d_high_pct": round(float(indicators.get("pullback_from_50d_high_pct", float("nan"))), 2),
                    "pullback_peak_date": indicators["pullback_peak_date"].strftime("%Y-%m-%d") if pd.notna(indicators.get("pullback_peak_date")) else None,
                    "pullback_duration_days": indicators.get("pullback_duration_days", float("nan")),
                    "pullback_closes_below_sma_200": indicators.get("pullback_closes_below_sma_200", float("nan")),
                    "damaging_high_volume_decline": bool(indicators.get("damaging_high_volume_decline", False)),
                    "pullback_peak_rsi": round(float(indicators.get("pullback_peak_rsi", float("nan"))), 2),
                    "constructive_pullback": bool(indicators.get("constructive_pullback", False)),
                    "trend_confirmed_before_pullback_on": indicators["trend_confirmed_before_pullback_on"].strftime("%Y-%m-%d") if pd.notna(indicators.get("trend_confirmed_before_pullback_on")) else None,
                    "advance_5d_atr": round(float(indicators.get("advance_5d_atr", float("nan"))), 2),
                    "advance_15d_atr": round(float(indicators.get("advance_15d_atr", float("nan"))), 2),
                    "advance_30d_atr": round(float(indicators.get("advance_30d_atr", float("nan"))), 2),
                    "max_ema_20_extension_30d_atr": round(float(indicators.get("max_ema_20_extension_30d_atr", float("nan"))), 2),
                    "current_extension_penalty": float(indicators.get("current_extension_penalty", 0.0)),
                    "verticality_penalty": float(indicators.get("verticality_penalty", 0.0)),
                    "overextension_penalty": float(indicators.get("overextension_penalty", 0.0)),
                    "data_quality_flag": bool(quality["data_quality_flag"]),
                    "data_quality_reason": str(quality["data_quality_reason"]),
                    "split_events_detected": int(quality.get("split_events_detected", 0)),
                    "split_adjustment_applied": bool(quality.get("split_adjustment_applied", False)),
                    "split_adjustments": str(quality.get("split_adjustments", "")),
                    "max_daily_gain": round(float(quality["max_daily_gain"]), 2),
                    "max_daily_loss": round(float(quality["max_daily_loss"]), 2),
                    "latest_price": round(float(latest), 2),
                    "1m": round(compute_period_performance(close_series, LOOKBACK_DAYS["1m"]), 2),
                    "3m": round(compute_period_performance(close_series, LOOKBACK_DAYS["3m"]), 2),
                    "6m": round(compute_period_performance(close_series.iloc[-LOOKBACK_DAYS["6m"] :], LOOKBACK_DAYS["6m"]), 2),
                    "12m": round(compute_period_performance(close_series.iloc[-LOOKBACK_DAYS["12m"] :], LOOKBACK_DAYS["12m"]), 2),
                    "2y": round(compute_period_performance(close_series.iloc[-LOOKBACK_DAYS["2y"] :], LOOKBACK_DAYS["2y"]), 2),
                }
            )
        except Exception as exc:
            failures.append((ticker, str(exc)))

    rankings = pd.DataFrame(results)
    if rankings.empty:
        rankings = pd.DataFrame(
            columns=[
                "ticker",
                "as_of_date",
                "closing_price",
                "atr_15d",
                "sma_50",
                "slope_50",
                "sma_200",
                "slope_200",
                "short_return_10d",
                "short_return_15d",
                "short_return_30d",
                "rsi_14",
                "short_momentum_score",
                "trend_confirmed",
                "trend_confirmed_on",
                "data_quality_flag",
                "data_quality_reason",
                "split_events_detected",
                "split_adjustment_applied",
                "split_adjustments",
                "max_daily_gain",
                "max_daily_loss",
                "latest_price",
                "6m",
                "12m",
                "2y",
            ]
        )

    return rankings, failures


def build_legacy_method_sheets(rankings: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}

    momentum = rankings.copy()
    if not momentum.empty:
        if "data_quality_flag" not in momentum.columns:
            momentum["data_quality_flag"] = False
        else:
            momentum["data_quality_flag"] = momentum["data_quality_flag"].fillna(False).astype(bool)
        momentum["short_momentum_score"] = momentum["short_momentum_score"] if "short_momentum_score" in momentum.columns else pd.Series(0.0, index=momentum.index)
        momentum["short_return_10d"] = momentum["short_return_10d"] if "short_return_10d" in momentum.columns else pd.Series(0.0, index=momentum.index)
        momentum["short_return_15d"] = momentum["short_return_15d"] if "short_return_15d" in momentum.columns else pd.Series(0.0, index=momentum.index)
        momentum["momentum_score"] = 0.5 * momentum["short_momentum_score"] + 0.35 * momentum["short_return_10d"] + 0.15 * momentum["short_return_15d"]
        momentum = momentum.sort_values(
            by=["data_quality_flag", "momentum_score"], ascending=[True, False]
        ).reset_index(drop=True)
        momentum["method"] = METHODS["MomentumLeader"]
        selected_columns = [
            "ticker",
            "method",
            "momentum_score",
            "short_momentum_score",
            "short_return_10d",
            "short_return_15d",
            "short_return_30d",
            "rsi_14",
            "closing_price",
            "atr_15d",
            "sma_50",
            "sma_200",
            "slope_50",
            "slope_200",
            "volume_1p5x",
            "split_adjustment_applied",
            "data_quality_flag",
            "data_quality_reason",
        ]
        selected_columns = [column for column in selected_columns if column in momentum.columns]
        frames["MomentumLeader"] = momentum[selected_columns]

    trend = rankings.copy()
    if not trend.empty:
        if "data_quality_flag" not in trend.columns:
            trend["data_quality_flag"] = False
        else:
            trend["data_quality_flag"] = trend["data_quality_flag"].fillna(False).astype(bool)
        trend["trend_confirmed"] = trend["trend_confirmed"] if "trend_confirmed" in trend.columns else pd.Series(False, index=trend.index)
        trend["trend_confirmed_on"] = trend["trend_confirmed_on"] if "trend_confirmed_on" in trend.columns else pd.Series(pd.NaT, index=trend.index)
        trend["trend_flag"] = ((trend["closing_price"] > trend["sma_50"]) & (trend["sma_50"] > trend["sma_200"]) & (trend["slope_50"] > 0) & (trend["slope_200"] > 0)).astype(int)
        trend = trend.sort_values(
            by=["data_quality_flag", "trend_flag", "trend_confirmed", "6m", "12m", "2y"],
            ascending=[True, False, False, False, False, False],
        ).reset_index(drop=True)
        trend["method"] = METHODS["TrendConfirmation"]
        selected_columns = [
            "ticker",
            "method",
            "trend_flag",
            "trend_confirmed",
            "trend_confirmed_on",
            "6m",
            "12m",
            "2y",
            "closing_price",
            "atr_15d",
            "sma_50",
            "sma_200",
            "slope_50",
            "slope_200",
            "volume_1p5x",
            "split_adjustment_applied",
            "data_quality_flag",
            "data_quality_reason",
        ]
        selected_columns = [column for column in selected_columns if column in trend.columns]
        frames["TrendConfirmation"] = trend[selected_columns]

    relative = rankings.copy()
    if not relative.empty:
        if "data_quality_flag" not in relative.columns:
            relative["data_quality_flag"] = False
        else:
            relative["data_quality_flag"] = relative["data_quality_flag"].fillna(False).astype(bool)
        clean_relative = relative.loc[~relative["data_quality_flag"]]
        median_source = clean_relative if not clean_relative.empty else relative
        universe_median_6m = median_source["6m"].median()
        universe_median_12m = median_source["12m"].median()
        universe_median_2y = median_source["2y"].median()
        relative["relative_strength"] = (
            (relative["6m"] - universe_median_6m) / abs(universe_median_6m + 1e-9)
            + (relative["12m"] - universe_median_12m) / abs(universe_median_12m + 1e-9)
            + (relative["2y"] - universe_median_2y) / abs(universe_median_2y + 1e-9)
        )
        relative = relative.sort_values(
            by=["data_quality_flag", "relative_strength"], ascending=[True, False]
        ).reset_index(drop=True)
        relative["method"] = METHODS["RelativeStrength"]
        selected_columns = [
            "ticker", "method", "relative_strength", "6m", "12m", "2y",
            "closing_price", "sma_50", "sma_200", "slope_50", "slope_200",
            "volume_1p5x", "split_adjustment_applied", "data_quality_flag",
            "data_quality_reason",
        ]
        selected_columns = [column for column in selected_columns if column in relative.columns]
        frames["RelativeStrength"] = relative[selected_columns]

    return frames


def percentile_against_clean(values: pd.Series, clean_values: pd.Series) -> pd.Series:
    reference = pd.to_numeric(clean_values, errors="coerce").dropna().sort_values().to_numpy()
    numeric = pd.to_numeric(values, errors="coerce")
    if len(reference) == 0:
        return pd.Series(float("nan"), index=values.index)
    return numeric.apply(
        lambda value: float(np.searchsorted(reference, value, side="right") / len(reference) * 100)
        if np.isfinite(value) else float("nan")
    )


def build_shadow_sheets(rankings: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if rankings.empty:
        return {
            "ShadowMomentum": pd.DataFrame(),
            "ShadowTrend": pd.DataFrame(),
            "ShadowRelative": pd.DataFrame(),
            "Diagnostics": pd.DataFrame(),
        }

    work = rankings.copy()
    defaults: dict[str, object] = {
        "data_quality_flag": False,
        "median_dollar_volume_20d": float("nan"),
        "short_momentum_score": 0.0,
        "short_return_10d": 0.0,
        "short_return_15d": 0.0,
        "short_return_30d": 0.0,
        "rsi_14": 50.0,
        "rsi_change_5d": float("nan"),
        "rsi_change_10d": float("nan"),
        "overextension_penalty": 0.0,
        "constructive_pullback": False,
        "experimental_trend_confirmed": False,
        "experimental_trend_confirmed_on": None,
        "days_since_experimental_confirmation": float("nan"),
        "sma_200_rising_days_20d": 0,
        "normalized_slope_50_20d": 0.0,
        "normalized_slope_200_40d": 0.0,
        "trend_reclaim": False,
        "trend_event_type": "",
    }
    for column, default in defaults.items():
        if column not in work.columns:
            work[column] = default
    if "1m" not in work.columns:
        work["1m"] = work.get("6m", 0.0)
    if "3m" not in work.columns:
        work["3m"] = work.get("6m", 0.0)
    work["data_quality_flag"] = work["data_quality_flag"].fillna(False).astype(bool)
    clean = work.loc[~work["data_quality_flag"]]
    reference = clean if not clean.empty else work

    work["liquidity_adjustment"] = work["median_dollar_volume_20d"].apply(liquidity_score_adjustment)
    work["price_adjustment"] = work["closing_price"].apply(price_score_adjustment)
    work["rsi_penalty"] = [
        rsi_score_penalty(float(rsi), float(change))
        for rsi, change in zip(work["rsi_14"], work["rsi_change_5d"])
    ]
    work["gross_penalty"] = (
        work["price_adjustment"].clip(upper=0)
        + work["liquidity_adjustment"].clip(upper=0)
        + work["rsi_penalty"].clip(upper=0)
        + work["overextension_penalty"].clip(upper=0)
    )
    work["liquidity_bonus"] = work["liquidity_adjustment"].clip(lower=0)
    work["total_net_adjustment"] = (
        work["liquidity_adjustment"] + work["price_adjustment"]
        + work["rsi_penalty"] + work["overextension_penalty"]
    )

    current_frames = build_legacy_method_sheets(rankings)
    current_rank_maps = {
        name: {ticker: rank for rank, ticker in enumerate(frame["ticker"], start=1)}
        for name, frame in current_frames.items()
    }

    current_momentum = (
        0.5 * work["short_momentum_score"]
        + 0.35 * work["short_return_10d"]
        + 0.15 * work["short_return_15d"]
    )
    work["momentum_current_score"] = current_momentum
    work["momentum_base_score"] = percentile_against_clean(current_momentum, current_momentum.loc[reference.index])
    work["momentum_adjusted_score"] = (
        work["momentum_base_score"] + work["liquidity_adjustment"] + work["price_adjustment"]
        + work["rsi_penalty"] + work["overextension_penalty"]
    ).clip(0, 100)

    structure = (
        work["closing_price"].gt(work["sma_50"])
        & work["sma_50"].gt(work["sma_200"])
    ).astype(float)
    rising_component = (
        pd.to_numeric(work["sma_200_rising_days_20d"], errors="coerce")
        .fillna(0).clip(upper=TREND_RISING_DAYS_REQUIRED) / TREND_RISING_DAYS_REQUIRED * 10
    )
    days_since = pd.to_numeric(work["days_since_experimental_confirmation"], errors="coerce")
    freshness = (5 * (1 - days_since / LOOKBACK_DAYS["6m"])).clip(lower=0, upper=5).fillna(0)
    slope_50_percentile = percentile_against_clean(
        work["normalized_slope_50_20d"], reference["normalized_slope_50_20d"]
    )
    slope_200_percentile = percentile_against_clean(
        work["normalized_slope_200_40d"], reference["normalized_slope_200_40d"]
    )
    work["trend_base_score"] = (
        work["experimental_trend_confirmed"].fillna(False).astype(float) * 40
        + structure * 10 + rising_component
        + slope_50_percentile * 0.15 + slope_200_percentile * 0.20 + freshness
    ).clip(0, 100)
    work["trend_adjusted_score"] = (
        work["trend_base_score"] + work["liquidity_adjustment"] + work["price_adjustment"]
    ).clip(0, 100)

    weights = {"6m": 0.40, "3m": 0.25, "12m": 0.20, "1m": 0.15}
    for period in weights:
        work[f"{period}_percentile"] = percentile_against_clean(work[period], reference[period])
    work["relative_strength_base_score"] = sum(
        weight * work[f"{period}_percentile"] for period, weight in weights.items()
    )
    work["relative_strength_adjusted_score"] = (
        work["relative_strength_base_score"] + work["liquidity_adjustment"]
        + work["price_adjustment"] + work["rsi_penalty"] + work["overextension_penalty"]
    ).clip(0, 100)

    def make_shadow(method: str, score_column: str, columns: list[str]) -> pd.DataFrame:
        frame = work.copy()
        if method == "TrendConfirmation":
            frame["gross_penalty"] = (
                frame["price_adjustment"].clip(upper=0)
                + frame["liquidity_adjustment"].clip(upper=0)
            )
            frame["liquidity_bonus"] = frame["liquidity_adjustment"].clip(lower=0)
            frame["total_net_adjustment"] = (
                frame["liquidity_adjustment"] + frame["price_adjustment"]
            )
        frame["current_rank"] = frame["ticker"].map(current_rank_maps[method])
        frame = frame.sort_values(
            ["data_quality_flag", score_column], ascending=[True, False]
        ).reset_index(drop=True)
        frame["experimental_rank"] = np.arange(1, len(frame) + 1)
        frame["rank_change"] = frame["current_rank"] - frame["experimental_rank"]
        selected = ["ticker", "current_rank", "experimental_rank", "rank_change"] + columns
        return frame[[column for column in selected if column in frame.columns]]

    common_columns = [
        "closing_price", "median_dollar_volume_20d", "liquidity_adjustment",
        "price_adjustment", "gross_penalty", "liquidity_bonus", "total_net_adjustment",
        "constructive_pullback", "data_quality_flag",
    ]
    momentum = make_shadow(
        "MomentumLeader", "momentum_adjusted_score",
        ["momentum_current_score", "momentum_base_score", "momentum_adjusted_score",
         "rsi_14", "rsi_change_5d", "rsi_penalty", "overextension_penalty"] + common_columns,
    )
    trend = make_shadow(
        "TrendConfirmation", "trend_adjusted_score",
        ["trend_base_score", "trend_adjusted_score", "experimental_trend_confirmed",
         "experimental_trend_confirmed_on", "days_since_experimental_confirmation",
         "sma_200_rising_days_20d", "trend_reclaim", "trend_event_type",
         "normalized_slope_50_20d", "normalized_slope_200_40d"] + common_columns,
    )
    relative = make_shadow(
        "RelativeStrength", "relative_strength_adjusted_score",
        ["relative_strength_base_score", "relative_strength_adjusted_score",
         "1m", "1m_percentile", "3m", "3m_percentile", "6m", "6m_percentile",
         "12m", "12m_percentile", "rsi_penalty", "overextension_penalty"] + common_columns,
    )
    diagnostic_columns = [
        "ticker", "as_of_date", "closing_price", "mean_volume_20d", "median_volume_20d",
        "median_dollar_volume_20d", "ema_20", "normalized_atr_pct",
        "normalized_slope_50_20d", "normalized_slope_200_40d",
        "distance_ema_20_atr", "distance_sma_50_atr", "distance_sma_200_atr",
        "atr_below_50d_high_current", "atr_below_50d_high_peak",
        "pullback_from_50d_high_pct", "pullback_peak_date", "pullback_duration_days",
        "pullback_closes_below_sma_200", "damaging_high_volume_decline",
        "pullback_peak_rsi", "constructive_pullback", "trend_confirmed_before_pullback_on",
        "advance_5d_atr",
        "advance_15d_atr", "advance_30d_atr", "max_ema_20_extension_30d_atr",
        "current_extension_penalty", "verticality_penalty", "overextension_penalty",
        "rsi_14", "rsi_change_5d", "rsi_change_10d", "rsi_penalty",
        "data_quality_flag", "data_quality_reason",
    ]
    return {
        "ShadowMomentum": momentum,
        "ShadowTrend": trend,
        "ShadowRelative": relative,
        "Diagnostics": work[[column for column in diagnostic_columns if column in work.columns]],
    }


def build_method_sheets(rankings: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the production sheets using the approved adjusted ranking systems."""
    if rankings.empty:
        return {}
    shadows = build_shadow_sheets(rankings)

    def combine(shadow_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        shadow = shadows[shadow_name].reset_index(drop=True)
        indexed = rankings.drop_duplicates("ticker").set_index("ticker")
        frame = indexed.reindex(shadow["ticker"]).reset_index()
        for column in shadow.columns:
            if column != "ticker":
                frame[column] = shadow[column].to_numpy()
        frame["previous_system_rank"] = frame["current_rank"]
        frame["rank_change_vs_previous"] = frame["rank_change"]
        return frame, shadow

    momentum, _ = combine("ShadowMomentum")
    momentum["method"] = METHODS["MomentumLeader"]
    momentum["momentum_score"] = momentum["momentum_adjusted_score"]
    momentum_columns = [
        "ticker", "method", "momentum_score", "momentum_base_score",
        "total_net_adjustment", "gross_penalty", "liquidity_bonus",
        "previous_system_rank", "rank_change_vs_previous", "short_momentum_score",
        "short_return_10d", "short_return_15d", "short_return_30d", "rsi_14",
        "rsi_change_5d", "rsi_penalty", "overextension_penalty", "closing_price",
        "atr_15d", "median_dollar_volume_20d", "liquidity_adjustment",
        "price_adjustment", "volume_1p5x", "constructive_pullback", "data_quality_flag",
        "data_quality_reason",
    ]

    trend, _ = combine("ShadowTrend")
    trend["method"] = METHODS["TrendConfirmation"]
    trend["trend_score"] = trend["trend_adjusted_score"]
    trend["trend_flag"] = trend["experimental_trend_confirmed"].fillna(False).astype(int)
    trend["trend_confirmed"] = trend["experimental_trend_confirmed"].fillna(False).astype(bool)
    trend["trend_confirmed_on"] = trend["experimental_trend_confirmed_on"]
    trend_columns = [
        "ticker", "method", "trend_score", "trend_base_score",
        "total_net_adjustment", "gross_penalty", "liquidity_bonus",
        "previous_system_rank", "rank_change_vs_previous", "trend_flag",
        "trend_confirmed", "trend_confirmed_on", "days_since_experimental_confirmation",
        "sma_200_rising_days_20d", "trend_reclaim", "trend_event_type",
        "normalized_slope_50_20d", "normalized_slope_200_40d", "closing_price",
        "sma_50", "sma_200", "median_dollar_volume_20d", "liquidity_adjustment",
        "price_adjustment", "volume_1p5x", "constructive_pullback", "data_quality_flag",
        "data_quality_reason",
    ]

    relative, _ = combine("ShadowRelative")
    relative["method"] = METHODS["RelativeStrength"]
    relative["relative_strength"] = relative["relative_strength_adjusted_score"]
    relative_columns = [
        "ticker", "method", "relative_strength", "relative_strength_base_score",
        "total_net_adjustment", "gross_penalty", "liquidity_bonus",
        "previous_system_rank", "rank_change_vs_previous", "1m", "1m_percentile",
        "3m", "3m_percentile", "6m", "6m_percentile", "12m", "12m_percentile",
        "rsi_14", "rsi_penalty", "overextension_penalty", "closing_price",
        "median_dollar_volume_20d", "liquidity_adjustment", "price_adjustment",
        "volume_1p5x", "constructive_pullback", "data_quality_flag", "data_quality_reason",
    ]
    return {
        "MomentumLeader": momentum[[column for column in momentum_columns if column in momentum.columns]],
        "TrendConfirmation": trend[[column for column in trend_columns if column in trend.columns]],
        "RelativeStrength": relative[[column for column in relative_columns if column in relative.columns]],
    }


def write_shadow_workbook(
    rankings: pd.DataFrame,
    failures: list[tuple[str, str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    definitions = pd.DataFrame(
        [
            ("Status", "Experimental only; production list scores and order are unchanged."),
            ("Liquidity", "20-day median dollar volume adjustments: <-$100k -8; $100k-$500k -3; $500k-$2m 0; >$2m +2."),
            ("Price", "Price adjustments: <$1 -8; $1-$3 -4; $3-$5 -2; >=$5 0."),
            ("Trend", "200-day SMA rising at least 15 of 20 days; structure persists 5 days; reclaim interruption <=10 days."),
            ("Slopes", "50-day SMA over 20 days and 200-day SMA over 40 days, expressed as percent per trading day."),
            ("Constructive pullback", "At least 3 current ATR below a three-day-median 50-day high; positive 200-day slope; <=3 closes below SMA200; peak RSI >=70 and current RSI 40-65."),
            ("Relative strength", "Percentile blend: 6m 40%, 3m 25%, 12m 20%, 1m 15%."),
            ("Risk-to-reward", "Reserved for a later phase; no score or trade-quality label is produced."),
        ], columns=["component", "definition"]
    )
    with pd.ExcelWriter(output_path) as writer:
        for name, frame in build_shadow_sheets(rankings).items():
            frame.to_excel(writer, sheet_name=name, index=False)
        definitions.to_excel(writer, sheet_name="Definitions", index=False)
        if failures:
            pd.DataFrame(failures, columns=["ticker", "error"]).to_excel(
                writer, sheet_name="Failed_tickers", index=False
            )


def write_outputs(rankings: pd.DataFrame, failures: list[tuple[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.suffix.lower() == ".xlsx":
        try:
            output_path.unlink()
        except OSError:
            pass

    if output_path.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(output_path) as writer:
            summary = rankings.copy()
            if not summary.empty:
                summary = summary.sort_values(by=["6m", "12m", "2y"], ascending=False)
            summary.to_excel(writer, sheet_name="Summary", index=False)

            for column in ["6m", "12m", "2y"]:
                period_rankings = rank_tickers(rankings[["ticker", column]], column)
                period_rankings.to_excel(writer, sheet_name=f"{column}_rankings", index=False)

            method_frames = build_method_sheets(rankings)
            for method_name, method_frame in method_frames.items():
                method_frame.to_excel(writer, sheet_name=method_name, index=False)

            shadow_frames = build_shadow_sheets(rankings)
            shadow_frames["Diagnostics"].to_excel(writer, sheet_name="Diagnostics", index=False)

            if not rankings.empty and "data_quality_flag" in rankings.columns:
                review_mask = (
                    rankings["data_quality_flag"].fillna(False).astype(bool)
                    | rankings["split_adjustment_applied"].fillna(False).astype(bool)
                )
                rankings.loc[review_mask].to_excel(writer, sheet_name="DataQualityReview", index=False)

            if failures:
                pd.DataFrame({"ticker": [ticker for ticker, _ in failures], "error": [message for _, message in failures]}).to_excel(
                    writer, sheet_name="Failed_tickers", index=False
                )
    else:
        rankings.to_csv(output_path, index=False)
        if failures:
            pd.DataFrame({"ticker": [ticker for ticker, _ in failures], "error": [message for _, message in failures]}).to_csv(
                output_path.with_suffix(".failed.csv"), index=False
            )


def discover_tickers(
    output_path: str | None = None,
    overrides_path: str | Path | None = "biotech_universe_overrides.csv",
    audit_output_path: str | None = None,
) -> list[str]:
    try:
        import requests
    except Exception:
        return []

    try:
        response = requests.get(NASDAQ_SCREENER_URL, headers=NASDAQ_REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
        rows = response.json()["data"]["rows"]
    except Exception:
        return []

    overrides = load_universe_overrides(overrides_path)
    selected: list[str] = []
    audit_rows: list[dict[str, str | bool]] = []
    for row in rows:
        ticker = str(row.get("symbol") or "").strip().upper()
        name = str(row.get("name") or "").strip()
        industry = str(row.get("industry") or "").strip()
        sector = str(row.get("sector") or "").strip()
        include, reason = classify_biotech_listing(ticker, name, industry, overrides)
        audit_rows.append(
            {
                "ticker": ticker,
                "name": name,
                "sector": sector,
                "industry": industry,
                "included": include,
                "reason": reason,
            }
        )
        if include:
            selected.append(ticker)

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ticker": sorted(set(selected))}).to_csv(output_file, index=False)

    if audit_output_path:
        audit_file = Path(audit_output_path)
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(audit_rows).sort_values(by=["included", "ticker"], ascending=[False, True]).to_csv(
            audit_file, index=False
        )

    return sorted(set(selected))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank biotech stocks by recent performance")
    parser.add_argument("--tickers", required=True, help="Path to a CSV file with a ticker column")
    parser.add_argument("--output", default="output/biotech_rankings.xlsx", help="Path to save results (CSV or XLSX)")
    parser.add_argument(
        "--shadow-output",
        default="output/biotech_rankings_shadow.xlsx",
        help="Path to save the experimental side-by-side shadow workbook",
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Start date for downloading historical data")
    parser.add_argument("--discover", action="store_true", help="Attempt to discover an automated biotech ticker universe")
    parser.add_argument("--discovered-output", default="output/discovered_tickers.csv", help="Where to save discovered tickers")
    parser.add_argument(
        "--universe-overrides",
        default="biotech_universe_overrides.csv",
        help="CSV containing explicit ticker include/exclude decisions",
    )
    parser.add_argument(
        "--universe-audit-output",
        default="output/universe_discovery_audit.csv",
        help="Where to save the discovery classification audit",
    )
    args = parser.parse_args()

    if args.discover:
        discovered = discover_tickers(
            args.discovered_output,
            overrides_path=args.universe_overrides,
            audit_output_path=args.universe_audit_output,
        )
        print(f"Discovered {len(discovered)} candidate tickers")
        if discovered:
            tickers = discovered
        else:
            tickers = load_tickers(args.tickers)
    else:
        tickers = load_tickers(args.tickers)

    rankings, failures = build_rankings(tickers, start_date=args.start_date)
    output_path = Path(args.output)
    write_outputs(rankings, failures, output_path)
    write_shadow_workbook(rankings, failures, Path(args.shadow_output))

    print(f"Generated {len(rankings)} ranked tickers and {len(failures)} failed downloads")
    print(f"Generated experimental shadow workbook at {args.shadow_output}")
    print(rankings.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
