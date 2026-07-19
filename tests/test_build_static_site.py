import json
from pathlib import Path

import pandas as pd
import pytest

from build_static_site import SHEETS, build_site, workbook_payload


def make_workbook(path: Path) -> None:
    with pd.ExcelWriter(path) as writer:
        for sheet in SHEETS:
            pd.DataFrame(
                {"ticker": ["TEST"], "score": [1.25], "missing": [float("nan")]}
            ).to_excel(writer, sheet_name=sheet, index=False)


def test_workbook_payload_contains_required_sheets_and_json_safe_null(tmp_path):
    workbook = tmp_path / "rankings.xlsx"
    make_workbook(workbook)

    payload = workbook_payload(workbook)

    assert set(payload["sheets"]) == set(SHEETS)
    assert payload["sheets"]["MomentumLeader"]["rows"][0]["missing"] is None
    json.dumps(payload, allow_nan=False)


def test_build_site_copies_assets_data_and_download(tmp_path):
    workbook = tmp_path / "rankings.xlsx"
    make_workbook(workbook)
    source = tmp_path / "source"
    source.mkdir()
    (source / "index.html").write_text("dashboard", encoding="utf-8")
    destination = tmp_path / "built"

    build_site(workbook, source, destination)

    assert (destination / "index.html").read_text(encoding="utf-8") == "dashboard"
    assert (destination / "data" / "rankings.json").is_file()
    assert (destination / "data" / "biotech_rankings.xlsx").is_file()
    assert (destination / ".nojekyll").is_file()


def test_workbook_payload_rejects_missing_sheet(tmp_path):
    workbook = tmp_path / "rankings.xlsx"
    pd.DataFrame({"ticker": ["TEST"]}).to_excel(workbook, index=False)

    with pytest.raises(ValueError, match="missing required sheets"):
        workbook_payload(workbook)
