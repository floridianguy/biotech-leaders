import pandas as pd

from src.biotech_ranker import (
    build_method_sheets,
    build_shadow_sheets,
    classify_biotech_listing,
    compute_entry_diagnostics,
    compute_experimental_trend,
    compute_normalized_slope,
    compute_period_performance,
    compute_short_term_momentum,
    compute_trend_confirmation,
    compute_volume_metrics,
    evaluate_price_quality,
    is_biotech_candidate,
    is_common_equity,
    liquidity_score_adjustment,
    load_tickers,
    load_universe_overrides,
    normalize_split_history,
    price_score_adjustment,
    rank_tickers,
    rsi_score_penalty,
    write_shadow_workbook,
)


def test_compute_period_performance_returns_expected_percentage():
    prices = pd.Series([100.0, 110.0], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))

    result = compute_period_performance(prices, 2)

    assert abs(result - 10.0) < 1e-9


def test_rank_tickers_sorts_descending_by_selected_period():
    results = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "6m": [10.0, 20.0, 5.0],
        }
    )

    ranked = rank_tickers(results, "6m")

    assert ranked["ticker"].tolist() == ["B", "A", "C"]


def test_load_tickers_filters_blank_values(tmp_path):
    csv_path = tmp_path / "tickers.csv"
    pd.DataFrame({"ticker": [" aapl ", "", "msft"]}).to_csv(csv_path, index=False)

    assert load_tickers(str(csv_path)) == ["AAPL", "MSFT"]


def test_fallback_universe_contains_required_validation_tickers():
    tickers = load_tickers("biotech_universe.csv")

    assert {"GLUE", "ATAI", "ENTX"}.issubset(tickers)


def test_is_biotech_candidate_matches_biotech_related_names():
    assert is_biotech_candidate("Gene Therapy Holdings")
    assert is_biotech_candidate("Oncology Therapeutics")
    assert is_biotech_candidate("AtaiBeckley Inc.")
    assert not is_biotech_candidate("Retail Holdings")


def test_common_equity_filter_rejects_funds_and_other_security_types():
    assert is_common_equity("Entera Bio Ltd. Ordinary Shares")
    assert is_common_equity("BioNTech SE American Depositary Share")
    assert not is_common_equity("Entera Bio Ltd. Warrants")
    assert not is_common_equity("Vulcan Infrastructure 8.50% Senior Notes due 2026")
    assert not is_common_equity("Example Biotechnology ETF")


def test_classification_keeps_validation_tickers_and_rejects_false_positives():
    overrides = {
        "CASY": {"action": "exclude", "reason": "retailer"},
        "DJT": {"action": "exclude", "reason": "media company"},
        "GREEL": {"action": "exclude", "reason": "senior notes"},
    }
    industry = "Biotechnology: Biological Products (No Diagnostic Substances)"

    for ticker in ["GLUE", "ATAI", "ENTX"]:
        included, _ = classify_biotech_listing(ticker, f"{ticker} Common Stock", industry, overrides)
        assert included

    for ticker in ["CASY", "DJT", "GREEL"]:
        included, _ = classify_biotech_listing(ticker, f"{ticker} Common Stock", industry, overrides)
        assert not included


def test_unrelated_company_name_fragments_do_not_create_candidates():
    included, reason = classify_biotech_listing(
        "CASY",
        "Casey's General Stores Inc. Common Stock",
        "Retail-Auto Dealers and Gas Stations",
    )

    assert not included
    assert reason == "outside selected biotech/pharma industries"


def test_complete_biotech_terms_recover_misclassified_companies():
    examples = [
        ("BNR", "Burning Rock Biotech Limited American Depositary Shares"),
        ("CUVL", "Clinuvel Pharmaceuticals Limited American Depositary Shares"),
        ("CODX", "Co-Diagnostics Inc. Common Stock"),
        ("ANIK", "Anika Therapeutics Inc. Common Stock"),
    ]

    for ticker, name in examples:
        included, reason = classify_biotech_listing(ticker, name, "Medical Specialities")
        assert included
        assert reason == "explicit biotech/life-science company name"


def test_load_universe_overrides_validates_and_normalizes_rows(tmp_path):
    path = tmp_path / "overrides.csv"
    pd.DataFrame(
        {
            "ticker": [" glue ", "djt"],
            "action": [" INCLUDE ", "exclude"],
            "reason": ["biotech", "media"],
        }
    ).to_csv(path, index=False)

    overrides = load_universe_overrides(path)

    assert overrides["GLUE"]["action"] == "include"
    assert overrides["DJT"]["reason"] == "media"


def test_compute_volume_metrics_returns_recent_and_average_volume():
    volumes = pd.Series([100.0, 200.0, 300.0, 400.0], index=pd.date_range("2024-01-01", periods=4, freq="D"))

    result = compute_volume_metrics(volumes, 15)

    assert result["latest_volume"] == 400.0
    assert abs(result["avg_volume_15d"] - 250.0) < 1e-9


def test_volume_metrics_include_robust_dollar_liquidity():
    index = pd.date_range("2026-01-01", periods=20, freq="B")
    volumes = pd.Series([10_000.0] * 19 + [10_000_000.0], index=index)
    closes = pd.Series([10.0] * 20, index=index)

    result = compute_volume_metrics(volumes, 15, closes)

    assert result["mean_volume_20d"] > 500_000
    assert result["median_volume_20d"] == 10_000
    assert result["median_dollar_volume_20d"] == 100_000


def test_experimental_adjustment_tiers_match_approved_defaults():
    assert liquidity_score_adjustment(99_999) == -8
    assert liquidity_score_adjustment(250_000) == -3
    assert liquidity_score_adjustment(1_000_000) == 0
    assert liquidity_score_adjustment(2_000_001) == 2
    assert price_score_adjustment(0.99) == -8
    assert price_score_adjustment(2.0) == -4
    assert price_score_adjustment(4.0) == -2
    assert price_score_adjustment(5.0) == 0
    assert rsi_score_penalty(82, 2) == -3
    assert rsi_score_penalty(82, -2) == -1.5


def test_normalized_slope_uses_recent_window_and_price_scale():
    series = pd.Series([1000.0] * 50 + list(range(100, 120)), dtype=float)

    result = compute_normalized_slope(series, 20)

    assert result > 0
    assert result < 1


def make_split_history(
    pre_split_prices: list[float],
    post_split_prices: list[float],
    split_ratio: float,
) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(pre_split_prices) + len(post_split_prices) + 1, freq="B")
    prices = pre_split_prices + [float("nan")] + post_split_prices
    split_values = [0.0] * len(prices)
    split_values[len(pre_split_prices)] = split_ratio
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": [1000.0] * len(prices),
            "Stock Splits": split_values,
        },
        index=dates,
    )


def test_reverse_split_normalization_removes_artificial_jump():
    history = make_split_history([0.60, 0.64], [6.30, 8.00], 0.1)

    normalized, info = normalize_split_history(history)

    assert info["split_adjustment_applied"] is True
    assert normalized["Close"].dropna().tolist() == [6.0, 6.4, 6.3, 8.0]
    assert normalized["Volume"].dropna().iloc[0] == 100.0


def test_forward_split_normalization_uses_latest_share_scale():
    history = make_split_history([100.0, 102.0], [51.0, 52.0], 2.0)

    normalized, info = normalize_split_history(history)

    assert info["split_adjustment_applied"] is True
    assert normalized["Close"].dropna().tolist() == [50.0, 51.0, 51.0, 52.0]
    assert normalized["Volume"].dropna().iloc[0] == 2000.0


def test_already_adjusted_or_postponed_split_is_not_applied_twice():
    history = make_split_history([6.0, 6.4], [6.3, 8.0], 0.1)

    normalized, info = normalize_split_history(history)

    assert info["split_adjustment_applied"] is False
    assert "prices already continuous" in info["ignored_split_events"]
    assert normalized["Close"].dropna().tolist() == [6.0, 6.4, 6.3, 8.0]


def test_inverse_split_discontinuity_is_reconciled():
    history = make_split_history([60.0, 64.0], [6.3, 8.0], 0.1)

    normalized, info = normalize_split_history(history)

    assert info["split_adjustment_applied"] is True
    assert "inverse correction" in info["split_adjustments"]
    assert normalized["Close"].dropna().tolist() == [6.0, 6.4, 6.3, 8.0]


def test_multiple_split_records_are_normalized_latest_first():
    dates = pd.date_range("2026-01-01", periods=8, freq="B")
    history = pd.DataFrame(
        {
            "Open": [60.0, 64.0, float("nan"), 6.3, 0.8, 0.64, float("nan"), 8.0],
            "High": [60.0, 64.0, float("nan"), 6.3, 0.8, 0.64, float("nan"), 8.0],
            "Low": [60.0, 64.0, float("nan"), 6.3, 0.8, 0.64, float("nan"), 8.0],
            "Close": [60.0, 64.0, float("nan"), 6.3, 0.8, 0.64, float("nan"), 8.0],
            "Volume": [1000.0] * 8,
            "Stock Splits": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0],
        },
        index=dates,
    )

    normalized, info = normalize_split_history(history)

    assert info["split_events_detected"] == 2
    assert info["split_adjustment_applied"] is True
    assert normalized["Close"].dropna().tolist() == [60.0, 64.0, 63.0, 8.0, 6.4, 8.0]


def test_data_quality_flags_implausible_unexplained_move():
    dates = pd.date_range("2026-01-01", periods=40, freq="B")
    history = pd.DataFrame({"Close": [1.0] * 39 + [10.0]}, index=dates)

    quality = evaluate_price_quality(history, {})

    assert quality["data_quality_flag"] is True
    assert "one-day gain" in quality["data_quality_reason"]
    assert "10-day return" in quality["data_quality_reason"]


def test_data_quality_allows_large_but_plausible_biotech_gap():
    dates = pd.date_range("2026-01-01", periods=40, freq="B")
    history = pd.DataFrame({"Close": [10.0] * 39 + [30.0]}, index=dates)

    quality = evaluate_price_quality(history, {})

    assert quality["data_quality_flag"] is False


def test_build_method_sheets_retains_volume_flag():
    rankings = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "closing_price": [100.0, 110.0],
            "sma_50": [90.0, 105.0],
            "sma_200": [80.0, 100.0],
            "slope_50": [1.0, 0.5],
            "slope_200": [0.8, 0.4],
            "6m": [10.0, 20.0],
            "12m": [5.0, 15.0],
            "2y": [3.0, 12.0],
            "volume_1p5x": [True, False],
        }
    )

    frames = build_method_sheets(rankings)

    assert "volume_1p5x" in frames["MomentumLeader"].columns
    assert "volume_1p5x" in frames["TrendConfirmation"].columns
    assert "volume_1p5x" in frames["RelativeStrength"].columns


def test_flagged_data_sorts_below_clean_data_in_every_method():
    rankings = pd.DataFrame(
        {
            "ticker": ["FLAGGED", "CLEAN"],
            "closing_price": [200.0, 100.0],
            "sma_50": [100.0, 90.0],
            "sma_200": [80.0, 80.0],
            "slope_50": [1.0, 1.0],
            "slope_200": [1.0, 1.0],
            "6m": [1000.0, 10.0],
            "12m": [1000.0, 10.0],
            "2y": [1000.0, 10.0],
            "short_momentum_score": [1000.0, 10.0],
            "short_return_10d": [1000.0, 10.0],
            "short_return_15d": [1000.0, 10.0],
            "short_return_30d": [1000.0, 10.0],
            "trend_confirmed": [True, True],
            "trend_confirmed_on": ["2026-01-01", "2026-01-01"],
            "volume_1p5x": [True, False],
            "data_quality_flag": [True, False],
            "data_quality_reason": ["implausible return", ""],
            "split_adjustment_applied": [False, False],
        }
    )

    frames = build_method_sheets(rankings)

    assert frames["MomentumLeader"].iloc[0]["ticker"] == "CLEAN"
    assert frames["TrendConfirmation"].iloc[0]["ticker"] == "CLEAN"
    assert frames["RelativeStrength"].iloc[0]["ticker"] == "CLEAN"


def test_compute_short_term_momentum_returns_expected_values():
    close = pd.Series([100.0, 105.0, 110.0, 112.0, 113.0, 115.0], index=pd.date_range("2024-01-01", periods=6, freq="D"))

    result = compute_short_term_momentum(close)

    assert pd.notna(result["short_momentum_score"])
    assert pd.notna(result["rsi_14"])
    assert pd.notna(result["short_return_10d"])


def test_compute_trend_confirmation_returns_first_date_of_confirmed_run():
    close = pd.Series([90.0, 92.0, 95.0, 98.0, 100.0], index=pd.date_range("2024-01-01", periods=5, freq="D"))
    sma_50 = pd.Series([85.0, 86.0, 88.0, 90.0, 92.0], index=close.index)
    sma_200 = pd.Series([80.0, 81.0, 82.0, 83.0, 84.0], index=close.index)

    result = compute_trend_confirmation(close, sma_50, sma_200)

    assert result["trend_confirmed"] is True
    assert result["trend_confirmed_on"] == close.index[0]


def test_experimental_trend_requires_persistence_and_recent_rising_sma_200():
    index = pd.date_range("2026-01-01", periods=45, freq="B")
    sma_200 = pd.Series(range(100, 145), index=index, dtype=float)
    sma_50 = sma_200 + 5
    close = sma_50 + 5

    result = compute_experimental_trend(close, sma_50, sma_200)

    assert result["experimental_trend_confirmed"] is True
    assert result["sma_200_rising_days_20d"] == 20
    assert result["days_since_experimental_confirmation"] >= 5


def test_constructive_pullback_diagnostics_follow_approved_definition():
    index = pd.date_range("2026-01-01", periods=60, freq="B")
    close = pd.Series(list(range(51, 101)) + list(range(98, 78, -2)), index=index, dtype=float)
    volume = pd.Series(100_000.0, index=index)
    atr = pd.Series(5.0, index=index)
    ema_20 = close - 2
    sma_50 = pd.Series(75.0, index=index)
    sma_200 = pd.Series(70.0, index=index)
    rsi = pd.Series([60.0] * 49 + [75.0] + [60.0] * 10, index=index)

    result = compute_entry_diagnostics(
        close, volume, atr, ema_20, sma_50, sma_200, rsi, normalized_slope_200=0.05
    )

    assert result["constructive_pullback"] is True
    assert result["atr_below_50d_high_current"] >= 3
    assert result["pullback_closes_below_sma_200"] == 0
    assert result["damaging_high_volume_decline"] is False


def test_isolated_one_day_spike_does_not_define_constructive_pullback():
    index = pd.date_range("2026-01-01", periods=60, freq="B")
    prices = [100.0] * 50 + [150.0] + [100.0] * 8 + [90.0]
    close = pd.Series(prices, index=index)
    volume = pd.Series(100_000.0, index=index)
    atr = pd.Series(5.0, index=index)
    ema_20 = pd.Series(95.0, index=index)
    sma_50 = pd.Series(90.0, index=index)
    sma_200 = pd.Series(80.0, index=index)
    rsi = pd.Series([60.0] * 50 + [80.0] + [55.0] * 9, index=index)

    result = compute_entry_diagnostics(
        close, volume, atr, ema_20, sma_50, sma_200, rsi, normalized_slope_200=0.05
    )

    assert result["atr_below_50d_high_current"] < 3
    assert result["constructive_pullback"] is False


def test_stale_vertical_move_does_not_penalize_after_current_extension_cools():
    index = pd.date_range("2026-01-01", periods=60, freq="B")
    close = pd.Series([100.0] * 40 + list(range(102, 132, 2)) + [130.0] * 5, index=index)
    volume = pd.Series(100_000.0, index=index)
    atr = pd.Series(5.0, index=index)
    ema_20 = close - 10.0
    sma_50 = close - 15.0
    sma_200 = pd.Series(90.0, index=index)
    rsi = pd.Series(60.0, index=index)

    result = compute_entry_diagnostics(
        close, volume, atr, ema_20, sma_50, sma_200, rsi, normalized_slope_200=0.05
    )

    assert max(result["distance_ema_20_atr"], result["distance_sma_50_atr"]) <= 4
    assert result["verticality_penalty"] == 0
    assert result["overextension_penalty"] == 0


def make_shadow_rankings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["LIQUID", "ILLIQUID", "EXTENDED"],
            "as_of_date": ["2026-08-10"] * 3,
            "closing_price": [20.0, 2.0, 30.0],
            "sma_50": [18.0, 1.8, 20.0],
            "sma_200": [15.0, 1.5, 15.0],
            "slope_50": [1.0] * 3,
            "slope_200": [1.0] * 3,
            "normalized_slope_50_20d": [0.1] * 3,
            "normalized_slope_200_40d": [0.05] * 3,
            "1m": [10.0, 10.0, 40.0],
            "3m": [20.0, 20.0, 80.0],
            "6m": [30.0, 30.0, 120.0],
            "12m": [40.0, 40.0, 160.0],
            "2y": [50.0, 50.0, 200.0],
            "short_momentum_score": [20.0, 20.0, 50.0],
            "short_return_10d": [20.0, 20.0, 50.0],
            "short_return_15d": [20.0, 20.0, 50.0],
            "short_return_30d": [20.0, 20.0, 50.0],
            "rsi_14": [60.0, 60.0, 86.0],
            "rsi_change_5d": [1.0, 1.0, 2.0],
            "rsi_change_10d": [2.0, 2.0, 4.0],
            "median_dollar_volume_20d": [3_000_000.0, 50_000.0, 3_000_000.0],
            "overextension_penalty": [0.0, 0.0, -10.0],
            "constructive_pullback": [True, False, False],
            "trend_confirmed": [True] * 3,
            "trend_confirmed_on": ["2026-01-01"] * 3,
            "experimental_trend_confirmed": [True] * 3,
            "experimental_trend_confirmed_on": ["2026-07-01"] * 3,
            "days_since_experimental_confirmation": [20.0] * 3,
            "sma_200_rising_days_20d": [20] * 3,
            "trend_reclaim": [False] * 3,
            "trend_event_type": ["crossover"] * 3,
            "volume_1p5x": [False] * 3,
            "data_quality_flag": [False] * 3,
            "data_quality_reason": [""] * 3,
            "split_adjustment_applied": [False] * 3,
        }
    )


def test_shadow_sheets_preserve_current_rank_and_expose_adjustments():
    sheets = build_shadow_sheets(make_shadow_rankings())

    assert set(sheets) == {"ShadowMomentum", "ShadowTrend", "ShadowRelative", "Diagnostics"}
    momentum = sheets["ShadowMomentum"].set_index("ticker")
    assert momentum.loc["ILLIQUID", "liquidity_adjustment"] == -8
    assert momentum.loc["EXTENDED", "overextension_penalty"] == -10
    assert "current_rank" in momentum.columns
    assert "experimental_rank" in momentum.columns


def test_production_method_sheets_use_adjusted_scores_and_keep_prior_rank():
    frames = build_method_sheets(make_shadow_rankings())
    momentum = frames["MomentumLeader"]
    relative = frames["RelativeStrength"]

    assert momentum.iloc[0]["momentum_score"] == momentum.iloc[0]["momentum_base_score"] + momentum.iloc[0]["total_net_adjustment"]
    assert relative.iloc[0]["relative_strength"] == relative.iloc[0]["relative_strength_base_score"] + relative.iloc[0]["total_net_adjustment"]
    assert "previous_system_rank" in momentum.columns


def test_write_shadow_workbook_contains_comparison_and_definition_sheets(tmp_path):
    path = tmp_path / "shadow.xlsx"

    write_shadow_workbook(make_shadow_rankings(), [], path)

    assert path.is_file()
    assert set(pd.ExcelFile(path).sheet_names) == {
        "ShadowMomentum", "ShadowTrend", "ShadowRelative", "Diagnostics", "Definitions"
    }
    compute_entry_diagnostics,
    compute_experimental_trend,
    compute_normalized_slope,
    liquidity_score_adjustment,
