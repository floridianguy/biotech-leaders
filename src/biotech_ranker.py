from __future__ import annotations

import argparse
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


def load_tickers(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    if "ticker" not in df.columns:
        raise ValueError(f"Expected a 'ticker' column in {csv_path}")

    tickers = df["ticker"].dropna().astype(str).str.strip().str.upper()
    return [ticker for ticker in tickers if ticker]


def fetch_price_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    history = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True, actions=False)
    if isinstance(history, pd.DataFrame) and not history.empty:
        if "Close" in history.columns:
            return history.dropna(subset=["Close"])
        if isinstance(history.columns, pd.MultiIndex):
            close_keys = [col for col in history.columns if col[0] == "Close"]
            if close_keys:
                history = history.copy()
                history.columns = [col[0] if isinstance(col, tuple) else col for col in history.columns]
                return history.dropna(subset=["Close"])
    return pd.DataFrame(columns=["Close", "High", "Low", "Open", "Volume"])


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
        momentum["short_momentum_score"] = momentum["short_momentum_score"] if "short_momentum_score" in momentum.columns else pd.Series(0.0, index=momentum.index)
        momentum["short_return_10d"] = momentum["short_return_10d"] if "short_return_10d" in momentum.columns else pd.Series(0.0, index=momentum.index)
        momentum["short_return_15d"] = momentum["short_return_15d"] if "short_return_15d" in momentum.columns else pd.Series(0.0, index=momentum.index)
        momentum["momentum_score"] = 0.5 * momentum["short_momentum_score"] + 0.35 * momentum["short_return_10d"] + 0.15 * momentum["short_return_15d"]
        momentum = momentum.sort_values(by="momentum_score", ascending=False).reset_index(drop=True)
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
        ]
        selected_columns = [column for column in selected_columns if column in momentum.columns]
        frames["MomentumLeader"] = momentum[selected_columns]

    trend = rankings.copy()
    if not trend.empty:
        trend["trend_confirmed"] = trend["trend_confirmed"] if "trend_confirmed" in trend.columns else pd.Series(False, index=trend.index)
        trend["trend_confirmed_on"] = trend["trend_confirmed_on"] if "trend_confirmed_on" in trend.columns else pd.Series(pd.NaT, index=trend.index)
        trend["trend_flag"] = ((trend["closing_price"] > trend["sma_50"]) & (trend["sma_50"] > trend["sma_200"]) & (trend["slope_50"] > 0) & (trend["slope_200"] > 0)).astype(int)
        trend = trend.sort_values(by=["trend_flag", "trend_confirmed", "6m", "12m", "2y"], ascending=[False, False, False, False, False]).reset_index(drop=True)
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
        ]
        selected_columns = [column for column in selected_columns if column in trend.columns]
        frames["TrendConfirmation"] = trend[selected_columns]

    relative = rankings.copy()
    if not relative.empty:
        universe_median_6m = relative["6m"].median()
        universe_median_12m = relative["12m"].median()
        universe_median_2y = relative["2y"].median()
        relative["relative_strength"] = (
            (relative["6m"] - universe_median_6m) / abs(universe_median_6m + 1e-9)
            + (relative["12m"] - universe_median_12m) / abs(universe_median_12m + 1e-9)
            + (relative["2y"] - universe_median_2y) / abs(universe_median_2y + 1e-9)
        )
        relative = relative.sort_values(by="relative_strength", ascending=False).reset_index(drop=True)
        relative["method"] = METHODS["RelativeStrength"]
        frames["RelativeStrength"] = relative[
            ["ticker", "method", "relative_strength", "6m", "12m", "2y", "closing_price", "sma_50", "sma_200", "slope_50", "slope_200", "volume_1p5x"]
        ]

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


def discover_tickers(output_path: str | None = None) -> list[str]:
    try:
        import requests
    except Exception:
        return []

    try:
        response = requests.get("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", timeout=20)
        response.raise_for_status()
    except Exception:
        return []

    rows = []
    for line in response.text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        ticker = parts[0].strip()
        name = parts[1].strip()
        if ticker and is_biotech_candidate(name):
            rows.append(ticker)

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ticker": rows}).to_csv(output_file, index=False)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank biotech stocks by recent performance")
    parser.add_argument("--tickers", required=True, help="Path to a CSV file with a ticker column")
    parser.add_argument("--output", default="output/biotech_rankings.xlsx", help="Path to save results (CSV or XLSX)")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Start date for downloading historical data")
    parser.add_argument("--discover", action="store_true", help="Attempt to discover an automated biotech ticker universe")
    parser.add_argument("--discovered-output", default="output/discovered_tickers.csv", help="Where to save discovered tickers")
    args = parser.parse_args()

    if args.discover:
        discovered = discover_tickers(args.discovered_output)
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
