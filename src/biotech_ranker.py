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
LOOKBACK_DAYS = {"6m": 126, "12m": 252, "2y": 504}
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


def compute_volume_metrics(volume_series: pd.Series, lookback_days: int) -> dict[str, float]:
    if volume_series.empty:
        return {"latest_volume": float("nan"), "avg_volume_15d": float("nan")}

    latest_volume = float(volume_series.iloc[-1])
    if len(volume_series) < lookback_days:
        avg_volume_15d = float(volume_series.mean())
    else:
        avg_volume_15d = float(volume_series.iloc[-lookback_days:].mean())

    return {"latest_volume": latest_volume, "avg_volume_15d": avg_volume_15d}


def compute_short_term_momentum(close_series: pd.Series) -> dict[str, float]:
    close = pd.to_numeric(close_series, errors="coerce").dropna()
    if close.empty:
        return {
            "short_return_10d": float("nan"),
            "short_return_15d": float("nan"),
            "short_return_30d": float("nan"),
            "rsi_14": float("nan"),
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


def compute_indicators(history: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    high = pd.to_numeric(history["High"], errors="coerce").dropna()
    low = pd.to_numeric(history["Low"], errors="coerce").dropna()
    volume = pd.to_numeric(history["Volume"], errors="coerce").dropna()

    if len(close) < 200:
        raise ValueError("insufficient history")

    atr = talib.ATR(high.to_numpy(), low.to_numpy(), close.to_numpy(), timeperiod=15)
    sma_50 = talib.SMA(close.to_numpy(), timeperiod=50)
    sma_200 = talib.SMA(close.to_numpy(), timeperiod=200)

    atr_series = pd.Series(atr, index=close.index)
    sma_50_series = pd.Series(sma_50, index=close.index)
    sma_200_series = pd.Series(sma_200, index=close.index)

    latest_date = close.index[-1].strftime("%Y-%m-%d")
    latest_close = float(close.iloc[-1])
    atr_15d = float(atr_series.iloc[-1])
    volume_metrics = compute_volume_metrics(volume, 15)
    sma_50 = float(sma_50_series.iloc[-1])
    sma_200 = float(sma_200_series.iloc[-1])

    slope_50 = float(np.polyfit(np.arange(len(sma_50_series.dropna())), sma_50_series.dropna().to_numpy(), 1)[0])
    slope_200 = float(np.polyfit(np.arange(len(sma_200_series.dropna())), sma_200_series.dropna().to_numpy(), 1)[0])
    short_term_momentum = compute_short_term_momentum(close)
    trend_confirmation = compute_trend_confirmation(close, sma_50_series, sma_200_series)

    return pd.Series(
        {
            "as_of_date": latest_date,
            "closing_price": latest_close,
            "atr_15d": atr_15d,
            "latest_volume": volume_metrics["latest_volume"],
            "avg_volume_15d": volume_metrics["avg_volume_15d"],
            "volume_1p5x": volume_metrics["latest_volume"] > 1.5 * volume_metrics["avg_volume_15d"],
            "sma_50": sma_50,
            "slope_50": slope_50,
            "sma_200": sma_200,
            "slope_200": slope_200,
            "short_return_10d": short_term_momentum["short_return_10d"],
            "short_return_15d": short_term_momentum["short_return_15d"],
            "short_return_30d": short_term_momentum["short_return_30d"],
            "rsi_14": short_term_momentum["rsi_14"],
            "short_momentum_score": short_term_momentum["short_momentum_score"],
            "trend_confirmed": trend_confirmation["trend_confirmed"],
            "trend_confirmed_on": trend_confirmation["trend_confirmed_on"],
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
                    "volume_1p5x": bool(indicators["volume_1p5x"]),
                    "sma_50": round(float(indicators["sma_50"]), 2),
                    "slope_50": round(float(indicators["slope_50"]), 4),
                    "sma_200": round(float(indicators["sma_200"]), 2),
                    "slope_200": round(float(indicators["slope_200"]), 4),
                    "short_return_10d": round(float(indicators["short_return_10d"]), 2),
                    "short_return_15d": round(float(indicators["short_return_15d"]), 2),
                    "short_return_30d": round(float(indicators["short_return_30d"]), 2),
                    "rsi_14": round(float(indicators["rsi_14"]), 2),
                    "short_momentum_score": round(float(indicators["short_momentum_score"]), 2),
                    "trend_confirmed": bool(indicators["trend_confirmed"]),
                    "trend_confirmed_on": indicators["trend_confirmed_on"].strftime("%Y-%m-%d") if pd.notna(indicators["trend_confirmed_on"]) else None,
                    "data_quality_flag": bool(quality["data_quality_flag"]),
                    "data_quality_reason": str(quality["data_quality_reason"]),
                    "split_events_detected": int(quality.get("split_events_detected", 0)),
                    "split_adjustment_applied": bool(quality.get("split_adjustment_applied", False)),
                    "split_adjustments": str(quality.get("split_adjustments", "")),
                    "max_daily_gain": round(float(quality["max_daily_gain"]), 2),
                    "max_daily_loss": round(float(quality["max_daily_loss"]), 2),
                    "latest_price": round(float(latest), 2),
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


def build_method_sheets(rankings: pd.DataFrame) -> dict[str, pd.DataFrame]:
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

    print(f"Generated {len(rankings)} ranked tickers and {len(failures)} failed downloads")
    print(rankings.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
