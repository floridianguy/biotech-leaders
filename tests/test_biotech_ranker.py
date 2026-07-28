import pandas as pd

from src.biotech_ranker import (
    build_method_sheets,
    classify_biotech_listing,
    compute_period_performance,
    compute_short_term_momentum,
    compute_trend_confirmation,
    compute_volume_metrics,
    is_biotech_candidate,
    is_common_equity,
    load_tickers,
    load_universe_overrides,
    rank_tickers,
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
