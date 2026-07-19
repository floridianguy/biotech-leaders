import os
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path("output/biotech_rankings.xlsx")
DEFAULT_SHEET = "MomentumLeader"
SUMMARY_BY_SHEET = {
    "MomentumLeader": "This list now uses a shorter-term momentum view built from 10-, 15-, and 30-day returns plus RSI, which helps highlight stocks that are rising but not yet too extended. The score gives a bit more weight to the 10-day and 15-day moves so the list leans toward names that are still near an entry point rather than already far ahead.",
    "TrendConfirmation": "This list favors names that are above both the 50-day and 200-day averages with positive slope, and it also shows the date when that trend condition first became established. In technical terms, the signal looks for a stock whose price is above its 50-day moving average, while that 50-day average is itself above the 200-day average, which is a common bullish trend setup. The trend_flag column is a simple yes/no indicator: 1 means the stock is currently in that confirmed trend setup, while 0 means it is not. The trend_confirmed_on date is the first date in the series where this setup became established and persisted long enough to be treated as a meaningful trend confirmation.",
    "RelativeStrength": "This list compares each stock’s performance against the median result for the full universe over the same time windows. It highlights names that are outperforming the broader biotech group on a relative basis.",
}


@st.cache_data
def load_sheet(sheet_name: str) -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Workbook not found at {DATA_PATH}")

    workbook = pd.ExcelFile(DATA_PATH)
    if sheet_name not in workbook.sheet_names:
        raise ValueError(f"Sheet '{sheet_name}' was not found in the workbook")

    df = pd.read_excel(DATA_PATH, sheet_name=sheet_name)
    return df.reset_index(drop=True)


def require_auth() -> None:
    username = os.getenv("STREAMLIT_USERNAME", "").strip()
    password = os.getenv("STREAMLIT_PASSWORD", "").strip()

    if not username or not password:
        st.session_state.authenticated = True
        return

    if st.session_state.get("authenticated"):
        return

    st.set_page_config(page_title="Biotech Leaders", page_icon="📈", layout="wide")
    st.title("Biotech Leaders Dashboard")
    st.caption("Private access required.")

    with st.form("login"):
        entered_username = st.text_input("Username")
        entered_password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted and entered_username == username and entered_password == password:
        st.session_state.authenticated = True
        st.rerun()
    elif submitted:
        st.error("Invalid username or password.")

    st.stop()


def main() -> None:
    st.set_page_config(page_title="Biotech Leaders", page_icon="📈", layout="wide")
    st.title("Biotech Leaders Dashboard")
    st.caption("Local Streamlit view of the top-ranked biotech names from the Excel workbook.")

    require_auth()

    if not DATA_PATH.exists():
        st.error(f"No workbook found at {DATA_PATH}. Run the ranking workflow first.")
        st.stop()

    sheet_names = ["MomentumLeader", "TrendConfirmation", "RelativeStrength"]
    selected_sheet = st.sidebar.selectbox("Trading system", sheet_names, index=0)
    top_n = st.sidebar.slider("Show top N names", min_value=5, max_value=50, value=25, step=1)

    try:
        df = load_sheet(selected_sheet)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if df.empty:
        st.info("No rows were found in the selected sheet.")
        st.stop()

    display_df = df.head(top_n).copy()
    display_df.insert(0, "Rank", range(1, len(display_df) + 1))
    if "atr_15d" in display_df.columns:
        display_df["atr_15d"] = display_df["atr_15d"].round(2)
    if "trend_confirmed_on" in display_df.columns:
        display_df["trend_confirmed_on"] = display_df["trend_confirmed_on"].fillna("-")

    st.subheader(f"{selected_sheet} — Top {top_n}")
    st.info(SUMMARY_BY_SHEET[selected_sheet])
    st.dataframe(display_df, width="stretch", hide_index=True)

    st.caption("This app reads the local Excel workbook only and does not require any remote service to start.")


if __name__ == "__main__":
    main()
