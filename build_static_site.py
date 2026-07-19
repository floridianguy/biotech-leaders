"""Build the static GitHub Pages dashboard from the rankings workbook."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SHEETS = {
    "MomentumLeader": (
        "This list uses a shorter-term momentum view built from 10-, 15-, and "
        "30-day returns plus RSI. The score gives extra weight to the 10-day "
        "and 15-day moves to favor names that may still be near an entry point."
    ),
    "TrendConfirmation": (
        "This list favors names above both the 50-day and 200-day averages with "
        "positive slope. A trend flag of 1 means the stock is currently in the "
        "confirmed bullish trend setup."
    ),
    "RelativeStrength": (
        "This list compares each stock's performance with the median result for "
        "the biotech universe over the same periods and highlights outperformers."
    ),
}


def json_value(value: Any) -> Any:
    """Convert pandas and NumPy values to strict JSON-compatible values."""
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def workbook_payload(workbook_path: Path) -> dict[str, Any]:
    workbook = pd.ExcelFile(workbook_path)
    missing = [name for name in SHEETS if name not in workbook.sheet_names]
    if missing:
        raise ValueError(f"Workbook is missing required sheets: {', '.join(missing)}")

    sheets: dict[str, Any] = {}
    for name, summary in SHEETS.items():
        frame = pd.read_excel(workbook_path, sheet_name=name)
        rows = [
            {str(column): json_value(value) for column, value in row.items()}
            for row in frame.to_dict(orient="records")
        ]
        sheets[name] = {"summary": summary, "rows": rows}

    return {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sheets": sheets,
    }


def build_site(workbook_path: Path, source_dir: Path, destination: Path) -> None:
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Static site source not found: {source_dir}")

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source_dir, destination)

    data_dir = destination / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = workbook_payload(workbook_path)
    (data_dir / "rankings.json").write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":")), encoding="utf-8"
    )
    shutil.copy2(workbook_path, data_dir / "biotech_rankings.xlsx")
    (destination / ".nojekyll").touch()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the static rankings dashboard")
    parser.add_argument("--workbook", default="output/biotech_rankings.xlsx")
    parser.add_argument("--source", default="site")
    parser.add_argument("--destination", default="_site")
    args = parser.parse_args()
    build_site(Path(args.workbook), Path(args.source), Path(args.destination))
    print(f"Built static dashboard at {args.destination}")


if __name__ == "__main__":
    main()
