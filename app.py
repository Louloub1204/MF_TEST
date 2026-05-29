"""SVM Outil — Streamlit app.

Replaces the legacy Excel "Outil_SVM.xlsx" workflow with:
  • A live dashboard for any of the 22 FCPs
  • Form-based transaction entry and editing
  • One-click BRVM price refresh (with manual CSV fallback)
  • Historical price archive identical in shape to the Excel "Cours" sheet

Run:
    streamlit run app.py
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

import db
from auth import require_login
from auto_sync import run_daily_auto_sync_if_needed, render_sync_status_sidebar
from fcp_calendar import is_weekly_fcp, weekday_label, WEEKLY_FCPS
from portfolio import (
    build_dashboard,
    compute_attribution,
    compute_exposures,
    compute_recap,
    compute_tracking,
    concentration_metrics,
    previous_business_date,
    resolve_dividends_for_date,
)
from sectors import sector_of, all_sectors, annotate as annotate_sectors

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SVM — Outil de gestion FCP",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Auth gate (no-op if no password configured in secrets)
# ---------------------------------------------------------------------------
require_login()


# ---------------------------------------------------------------------------
# Dune Gold — custom CSS theme injection
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #2C1F0F !important;
}
[data-testid="stSidebar"] > div:first-child {
    color: #D4B896 !important;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stRadio span {
    color: #D4B896 !important;
    font-size: 0.9rem;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #BFA98A !important;
    font-size: 0.8rem;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #EFA823 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #4A3010 !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background-color: #3A2810 !important;
    border-color: #5C3D18 !important;
    color: #FAC775 !important;
}
[data-testid="stSidebar"] label {
    color: #BFA98A !important;
}
[data-testid="stSidebar"] [data-testid="stDateInput"] input {
    background-color: #3A2810 !important;
    border-color: #5C3D18 !important;
    color: #FAC775 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #FAC775 !important;
    background-color: #3A2810 !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] span {
    color: #D4B896 !important;
}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: #8A6E48 !important;
}

/* ── Top bar / header ── */
[data-testid="stHeader"] {
    background-color: #FBF8F3 !important;
    border-bottom: 1px solid #E8D9C0;
}

/* ── Page background ── */
.stApp {
    background-color: #FBF8F3 !important;
}
.main .block-container {
    background-color: #FBF8F3 !important;
}

/* ── Main content text — always dark on light bg ── */
.main p, .main span, .main div,
[data-testid="stMainBlockContainer"] p,
[data-testid="stMainBlockContainer"] span {
    color: #2C1F0F;
}

/* ── Dataframe text — force dark, overrides any inherited color ── */
[data-testid="stDataFrame"] * {
    color: #2C1F0F !important;
}
.dvn-scroller {
    background-color: #FFFFFF !important;
}
[data-testid="stDataFrame"] canvas {
    color: #2C1F0F !important;
}

/* ── st.table (HTML table) ── */
[data-testid="stTable"] table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    color: #2C1F0F !important;
    background-color: #FFFFFF;
    border: 0.5px solid #E8D9C0;
    border-radius: 8px;
    overflow: hidden;
}
[data-testid="stTable"] thead tr {
    background-color: #F2EBE0 !important;
    border-bottom: 1px solid #E8D9C0;
}
[data-testid="stTable"] thead th {
    color: #633806 !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    font-size: 0.8rem !important;
    text-align: left;
}
[data-testid="stTable"] tbody tr:nth-child(even) {
    background-color: #FBF8F3 !important;
}
[data-testid="stTable"] tbody tr:hover {
    background-color: #FAEEDA !important;
}
[data-testid="stTable"] tbody td {
    color: #2C1F0F !important;
    padding: 6px 12px !important;
    font-size: 0.83rem !important;
    border-bottom: 0.5px solid #F0E8DA;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 0.5px solid #E8D9C0;
    border-left: 3px solid #EF9F27;
    border-radius: 8px;
    padding: 0.75rem 1rem !important;
}
[data-testid="stMetricLabel"] {
    color: #8A6E48 !important;
    font-size: 0.78rem !important;
}
[data-testid="stMetricValue"] {
    color: #2C1F0F !important;
    font-size: 1.2rem !important;
    font-weight: 500 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"],
.stButton > button {
    background-color: #EF9F27 !important;
    color: #2C1F0F !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}
.stButton > button:hover {
    background-color: #D98A1A !important;
    color: #2C1F0F !important;
}
.stDownloadButton > button {
    background-color: #FFFFFF !important;
    color: #633806 !important;
    border: 1px solid #E8D9C0 !important;
    border-radius: 6px !important;
}
.stDownloadButton > button:hover {
    background-color: #FAEEDA !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 0.5px solid #E8D9C0 !important;
    border-radius: 8px !important;
    background-color: #FFFFFF !important;
}
[data-testid="stExpander"] summary {
    color: #633806 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}
[data-testid="stExpander"] summary span {
    color: #633806 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 2px solid #E8D9C0 !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #8A6E48 !important;
    border-radius: 6px 6px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #633806 !important;
    border-bottom: 2px solid #EF9F27 !important;
    font-weight: 500 !important;
}

/* ── Info / success / warning boxes ── */
[data-testid="stInfo"] {
    background-color: #FAEEDA !important;
    border-left-color: #EF9F27 !important;
    color: #633806 !important;
}
[data-testid="stSuccess"] {
    background-color: #E8F5EE !important;
    border-left-color: #1D9E75 !important;
}
[data-testid="stWarning"] {
    background-color: #FAEEDA !important;
    border-left-color: #BA7517 !important;
}

/* ── Select boxes & inputs (main content) ── */
.main [data-testid="stSelectbox"] > div,
.main [data-testid="stMultiSelect"] > div {
    border-color: #E8D9C0 !important;
    background-color: #FFFFFF !important;
}
.main [data-testid="stTextInput"] input,
.main [data-testid="stNumberInput"] input {
    border-color: #E8D9C0 !important;
    background-color: #FFFFFF !important;
    color: #2C1F0F !important;
}
.main [data-testid="stTextInput"] input:focus,
.main [data-testid="stNumberInput"] input:focus {
    border-color: #EF9F27 !important;
    box-shadow: 0 0 0 2px rgba(239,159,39,0.2) !important;
}

/* ── Dividers ── */
.main hr {
    border-color: #E8D9C0 !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background-color: #EF9F27 !important;
}

/* ── Headings in main content ── */
.main h1, .main h2, .main h3 {
    color: #2C1F0F !important;
}
.main h1 { border-bottom: 2px solid #EF9F27; padding-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def _bootstrap() -> None:
    """Ensure schema exists. Does NOT seed data — use seed_postgres.py for that."""
    db.init_db()


_bootstrap()


# ---------------------------------------------------------------------------
# Daily auto-sync — runs once per calendar day on first app open
# ---------------------------------------------------------------------------
# This runs BEFORE the cache layer is queried, so any data refreshed here
# is visible immediately in the rest of the app.
_did_sync = run_daily_auto_sync_if_needed()


# ---------------------------------------------------------------------------
# Cached data accessors — minimize Postgres round-trips
# ---------------------------------------------------------------------------
# These wrap the db.* readers with a 5-minute TTL. After any write
# (transactions, prices, dividends), call _clear_data_cache() to refresh.

@st.cache_data(ttl=600, show_spinner=False)
def _cached_transactions() -> pd.DataFrame:
    return db.get_all_transactions_for_compute()


@st.cache_data(ttl=600, show_spinner=False)
def _cached_prices() -> pd.DataFrame:
    return db.get_prices()


@st.cache_data(ttl=300, show_spinner=False)
def _cached_fcps() -> list[str]:
    return db.get_fcps()


@st.cache_data(ttl=300, show_spinner=False)
def _cached_dividends_all(fcps_tuple: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """Single query for all FCPs instead of N separate queries."""
    try:
        from sqlalchemy import text as _text
        eng = db.get_engine()
        import pandas as _pd
        df = _pd.read_sql_query(
            _text("SELECT fcp, ticker, amount FROM dividends"),
            eng,
        )
        result: dict[str, dict[str, float]] = {f: {} for f in fcps_tuple}
        for _, row in df.iterrows():
            if row["fcp"] in result:
                result[row["fcp"]][row["ticker"]] = float(row["amount"])
        return result
    except Exception:
        return {f: db.get_dividends(f) for f in fcps_tuple}


@st.cache_data(ttl=300, show_spinner=False)
def _cached_dividends_dated() -> list[dict]:
    """Return all dividends with their payment dates as a list of dicts."""
    return db.get_dividends_dated()


@st.cache_data(ttl=300, show_spinner=False)
def _cached_known_tickers() -> list[str]:
    return db.get_known_tickers()


@st.cache_data(ttl=600, show_spinner=False)
def _cached_exposures_data(
    as_of_str: str, tx_hash: int, prices_hash: int,
    scope_key: str,
) -> "pd.DataFrame":
    """Cache compute_exposures — expensive cross-FCP calculation."""
    from portfolio import compute_exposures as _ce
    tx = _cached_transactions()
    prices = _cached_prices()
    all_fcps = _cached_fcps()
    if scope_key == "ALL":
        fcps_scope = all_fcps
    else:
        fcps_scope = [scope_key]
    divs = _build_divs_by_fcp(fcps_scope, pd.Timestamp(as_of_str))
    return _ce(tx, prices, fcps_scope, pd.Timestamp(as_of_str), divs)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_recap_data(as_of_str: str, tx_hash: int, prices_hash: int) -> "pd.DataFrame":
    """Cache compute_recap — invalidates when data or date changes."""
    from portfolio import compute_recap as _cr
    tx = _cached_transactions()
    prices = _cached_prices()
    all_fcps = _cached_fcps()
    divs = _build_divs_by_fcp(all_fcps, pd.Timestamp(as_of_str))
    return _cr(tx, prices, all_fcps, pd.Timestamp(as_of_str), divs)


def _data_hash() -> tuple[int, int]:
    """Cheap hash of transactions and prices row counts for cache invalidation."""
    tx = _cached_transactions()
    prices = _cached_prices()
    return (len(tx), len(prices))


def _clear_data_cache() -> None:
    """Invalidate all cached reads. Call after any write."""
    _cached_transactions.clear()
    _cached_prices.clear()
    _cached_fcps.clear()
    _cached_dividends_all.clear()
    _cached_dividends_dated.clear()
    _cached_known_tickers.clear()
    _cached_recap_data.clear()
    _cached_exposures_data.clear()


def _build_divs_by_fcp(
    fcps_list: list[str],
    as_of: "pd.Timestamp",
) -> dict[str, dict[str, float]]:
    """Build {fcp: {ticker: amount}} merging legacy + dated dividends.

    Legacy dividends (no date) apply every day.
    Dated dividends apply ONLY on their payment_date.
    Dated values override legacy values for the same ticker on that date.
    """
    # Legacy: {fcp: {ticker: amount}}
    legacy = _cached_dividends_all(tuple(fcps_list))
    # Dated: {ticker: amount} for as_of date only
    dated = resolve_dividends_for_date(_cached_dividends_dated(), as_of)

    if not dated:
        return legacy

    # Merge: start from legacy, add/override with dated for each FCP
    merged: dict[str, dict[str, float]] = {}
    for fcp_name in fcps_list:
        base = dict(legacy.get(fcp_name, {}))
        base.update(dated)  # dated dividends apply to all FCPs
        merged[fcp_name] = base
    return merged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_xof(v: float, signed: bool = False) -> str:
    if v is None or pd.isna(v):
        return "-"
    if abs(v) < 0.5:
        return "-"
    sign = "+" if signed and v > 0 else ""
    return f"{sign}{v:,.0f} FCFA".replace(",", " ")


def fmt_pct(v: float, signed: bool = False) -> str:
    if v is None or pd.isna(v):
        return "-"
    if signed:
        return f"{v*100:+,.2f}%".replace(",", " ")
    return f"{v*100:,.2f}%".replace(",", " ")


def render_table(
    df: pd.DataFrame,
    height: int | None = None,
    color_cols: list[str] | None = None,
) -> None:
    """Render a DataFrame as a styled HTML table — bypasses Canvas/WebGL.

    All values should be pre-formatted as strings before calling.
    color_cols: columns whose values starting with + or ▲ get green,
                and starting with - or ▼ get red.
    """
    color_cols = set(color_cols or [])
    df = df.copy()

    # Convert any remaining raw numerics to strings
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].map(
                lambda v: "-" if pd.isna(v)
                else f"{v:,.0f}".replace(",", " ")
            )
        else:
            df[col] = df[col].fillna("-").astype(str)

    td_base = "padding:6px 11px;white-space:nowrap;font-size:0.82rem;"

    def _td(col: str, val: str) -> str:
        s = str(val).strip()
        if col not in color_cols:
            return f"<td style='{td_base}'>{s}</td>"
        if s.startswith("+") or s.startswith("▲"):
            return (f"<td style='{td_base}color:#1A6B3E;font-weight:500;'>"
                    f"▲ {s.lstrip('+▲').strip()}</td>")
        elif s.startswith("-") or s.startswith("▼"):
            return (f"<td style='{td_base}color:#B53A2F;font-weight:500;'>"
                    f"▼ {s.lstrip('-▼').strip()}</td>")
        else:
            return f"<td style='{td_base}color:#8A6E48;'>{s}</td>"

    th_style = (
        "style='background:#F2EBE0;color:#633806;font-weight:500;"
        "padding:7px 11px;font-size:0.79rem;text-align:left;"
        "border-bottom:1px solid #E8D9C0;white-space:nowrap;'"
    )
    header = "".join(f"<th {th_style}>{c}</th>" for c in df.columns)

    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        bg = "#FFFFFF" if i % 2 == 0 else "#FBF8F3"
        cells = "".join(_td(col, str(row[col])) for col in df.columns)
        rows_html += (
            f"<tr style='background:{bg};"
            f"border-bottom:0.5px solid #F0E8DA;'>{cells}</tr>"
        )

    scroll = f"max-height:{height}px;overflow-y:auto;" if height else ""
    st.markdown(
        f"<div style='overflow-x:auto;{scroll}border:0.5px solid #E8D9C0;"
        f"border-radius:8px;background:#FFFFFF;margin-bottom:0.5rem;'>"
        f"<table style='width:100%;border-collapse:collapse;color:#2C1F0F;'>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True,
    )



def _to_xlsx(df: pd.DataFrame, sheet_name: str = "Données") -> bytes:
    """Convert a DataFrame to a styled .xlsx file and return bytes.

    Uses openpyxl with CGF GESTION blue/white styling.
    """
    import io as _io
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    BLUE_FILL  = PatternFill("solid", fgColor="004977")
    TOTAL_FILL = PatternFill("solid", fgColor="E6EFF5")
    THIN = Side(style="thin", color="C2D8E8")
    BORDER = Border(bottom=THIN)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name[:31]  # Excel max 31 chars

    # Header row
    for col_i, col_name in enumerate(df.columns, 1):
        cell = ws.cell(row=1, column=col_i, value=str(col_name))
        cell.font      = Font(bold=True, color="FFFFFF", size=10)
        cell.fill      = BLUE_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border    = Border(bottom=Side(style="medium", color="FFFFFF"))

    # Data rows
    for row_i, (_, row) in enumerate(df.iterrows(), 2):
        is_total = str(row.iloc[0]).upper() in ("TOTAL", "TOTAL GÉNÉRAL")
        fill = TOTAL_FILL if is_total else None
        for col_i, val in enumerate(row, 1):
            # Try to store as number for numeric-looking strings
            stored_val = val
            if isinstance(val, str):
                clean = val.replace(" ", "").replace("FCFA","").replace("%","").replace("+","").replace("▲","").replace("▼","").strip()
                try:
                    stored_val = float(clean) if "." in clean else int(clean)
                except (ValueError, AttributeError):
                    stored_val = val
            cell = ws.cell(row=row_i, column=col_i, value=stored_val)
            cell.border = BORDER
            if is_total:
                cell.fill = TOTAL_FILL
                cell.font = Font(bold=True, size=10)
            elif row_i % 2 == 0:
                cell.fill = PatternFill("solid", fgColor="F5F7FA")
            # Right-align numeric cells
            if isinstance(stored_val, (int, float)):
                cell.alignment = Alignment(horizontal="right")
                if isinstance(stored_val, float) and not stored_val.is_integer():
                    cell.number_format = '#,##0.00'
                else:
                    cell.number_format = '#,##0'

    # Auto column widths
    for col in ws.columns:
        max_len = max(
            (len(str(cell.value or "")) for cell in col), default=8
        )
        ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 50)

    ws.row_dimensions[1].height = 20
    ws.freeze_panes = "A2"

    buf = _io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()


def _xlsx_btn(label: str, df: pd.DataFrame, filename: str,
              key: str, sheet_name: str = "Données") -> None:
    """Render a .xlsx download button."""
    try:
        xlsx_bytes = _to_xlsx(df, sheet_name=sheet_name)
        st.download_button(
            label=label,
            data=xlsx_bytes,
            file_name=filename if filename.endswith(".xlsx") else filename.replace(".csv", ".xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=key,
        )
    except Exception as e:
        st.error(f"Export Excel impossible : {e}")


# ---------------------------------------------------------------------------
# Sidebar - FCP selection + global controls
# ---------------------------------------------------------------------------
fcps = _cached_fcps()

with st.sidebar:
    st.markdown(
        "<h2 style='color:#EFA823;font-size:1.2rem;margin-bottom:2px;'>"
        "SVM · BRVM</h2>"
        "<p style='color:#8A6E48;font-size:0.75rem;margin-bottom:1rem;'>"
        "Gestion FCP — CGF GESTION</p>",
        unsafe_allow_html=True,
    )

    if not fcps:
        st.error("Aucun FCP en base. Initialisez les données dans Paramètres.")
        st.stop()

    fcp = st.selectbox("FCP actif", fcps, key="fcp_select")
    as_of = st.date_input("Date de valorisation", value=date.today())
    st.divider()

    page = st.radio(
        "Navigation",
        ["📈 Tableau de bord", "📊 Récap variations", "🎯 Expositions",
         "📋 Suivi des cibles", "💼 Transactions", "🌐 Cours BRVM",
         "📚 Historique cours", "⚙️ Paramètres"],
        label_visibility="collapsed",
    )

# Sync status footer (separate from the sidebar `with` block above so it's
# always rendered, including after the first auto-sync of the day)
render_sync_status_sidebar()

# Pre-fetch shared data once per run (used across all pages)
_tx_all_global   = _cached_transactions()
_prices_global   = _cached_prices()
_all_fcps_global = _cached_fcps()


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
if page == "📈 Tableau de bord":
    st.header(f"{fcp}")
    as_of_ts = pd.Timestamp(as_of)

    tx_all = _tx_all_global
    prices = _prices_global
    # Legacy dividends (sans date) + dividendes datés filtrés sur as_of_ts
    divs = {**_cached_dividends_all(tuple(fcps)).get(fcp, {})}
    dated_divs = resolve_dividends_for_date(_cached_dividends_dated(), as_of_ts)
    divs.update(dated_divs)  # dated overrides/adds on top of legacy

    rows, totals = build_dashboard(tx_all, prices, fcp, as_of_ts, divs)
    prev_date = totals["prev_date"]

    if is_weekly_fcp(fcp):
        wd = weekday_label(WEEKLY_FCPS[fcp])
        st.caption(
            f"Valorisation au **{as_of_ts.strftime('%d/%m/%Y')}** — "
            f"FCP à valorisation hebdomadaire ({wd}) — "
            f"Comparaison avec **{prev_date.strftime('%d/%m/%Y')}**"
        )
    else:
        st.caption(
            f"Valorisation au **{as_of_ts.strftime('%d/%m/%Y')}** — "
            f"Comparaison avec **{prev_date.strftime('%d/%m/%Y')}**"
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coût total", fmt_xof(totals["cout_total"]))
    c2.metric("Valorisation", fmt_xof(totals["valorisation"]))
    c3.metric(
        "+/- value latente",
        fmt_xof(totals["diff_estim"], signed=True),
        delta=fmt_pct(
            totals["diff_estim"] / totals["cout_total"] if totals["cout_total"] else 0
        ),
    )
    c4.metric(
        "Variation journalière",
        fmt_xof(totals["variation_jour"], signed=True),
        delta=fmt_pct(totals["variation_pct"]),
    )

    st.divider()

    if rows.empty:
        st.info("Aucune position pour ce FCP à cette date.")
    else:
        rows_active = rows[rows["quantite"] > 0]
        rows_zero = rows[rows["quantite"] == 0]

        st.subheader("Positions actives")
        render_table(
            pd.DataFrame({
                "Symbole": rows_active["ticker"],
                "Quantité": rows_active["quantite"].map(lambda v: f"{v:,.0f}".replace(",", " ")),
                "CMP": rows_active["cmp"].map(lambda v: f"{v:,.2f}".replace(",", " ")),
                "Coût total": rows_active["cout_total"].map(fmt_xof),
                "Valorisation": rows_active["valorisation"].map(fmt_xof),
                "+/- estim.": rows_active["diff_estim"].map(lambda v: fmt_xof(v, signed=True)),
                "Poids": rows_active["poids"].map(fmt_pct),
                "Cours veille": rows_active["prev_close"].map(lambda v: f"{v:,.0f}".replace(",", " ") if v else "-"),
                "Cours jour": rows_active["close"].map(lambda v: f"{v:,.0f}".replace(",", " ") if v else "-"),
                "Variation": rows_active["variation"].map(lambda v: fmt_xof(v, signed=True)),
                "+/- value jour": rows_active["plus_moins_value"].map(lambda v: fmt_xof(v, signed=True)),
            }),
            color_cols=["+/- estim.", "Variation", "+/- value jour"],
        )

        if not rows_zero.empty:
            with st.expander(f"Lignes soldées ({len(rows_zero)})"):
                render_table(
                    pd.DataFrame({
                        "Symbole": rows_zero["ticker"],
                        "Quantité": rows_zero["quantite"].map(lambda v: f"{v:,.0f}".replace(",", " ")),
                        "CMP": rows_zero["cmp"].map(lambda v: f"{v:,.2f}".replace(",", " ")),
                        "Coût total": rows_zero["cout_total"].map(fmt_xof),
                        "Valorisation": rows_zero["valorisation"].map(fmt_xof),
                        "+/- estim.": rows_zero["diff_estim"].map(lambda v: fmt_xof(v, signed=True)),
                        "Poids": rows_zero["poids"].map(fmt_pct),
                        "Cours veille": rows_zero["prev_close"].map(lambda v: f"{v:,.0f}".replace(",", " ") if v else "-"),
                        "Cours jour": rows_zero["close"].map(lambda v: f"{v:,.0f}".replace(",", " ") if v else "-"),
                        "Variation": rows_zero["variation"].map(lambda v: fmt_xof(v, signed=True)),
                        "+/- value jour": rows_zero["plus_moins_value"].map(lambda v: fmt_xof(v, signed=True)),
                    }),
                    color_cols=["+/- estim.", "Variation", "+/- value jour"],
                )

        st.divider()
        col_a, col_b = st.columns([2, 3])
        with col_a:
            st.subheader("Répartition par titre")
            chart_data = (
                rows_active.set_index("ticker")["valorisation"]
                .sort_values(ascending=False)
            )
            st.bar_chart(chart_data)

        with col_b:
            st.subheader("Top mouvements du jour")
            _top_raw = (
                rows_active
                .assign(_abs=rows_active["plus_moins_value"].abs())
                .nlargest(10, "_abs")
            )
            _top_mov = pd.DataFrame({
                "Symbole": _top_raw["ticker"].values,
                "+/- value jour": _top_raw["plus_moins_value"].map(
                    lambda v: fmt_xof(v, signed=True)
                ).values,
                "Variation unitaire": _top_raw["variation"].map(
                    lambda v: fmt_xof(v, signed=True)
                ).values,
            })
            render_table(
                _top_mov,
                color_cols=["+/- value jour", "Variation unitaire"],
            )

        _xlsx_btn(
            "📊 Exporter le tableau (.xlsx)", rows,
            f"{fcp.replace(' ', '_')}_{as_of_ts.date()}.xlsx",
            key="dl_dashboard", sheet_name="Positions",
        )


# ---------------------------------------------------------------------------
# Page: Récap variations
# ---------------------------------------------------------------------------
elif page == "📊 Récap variations":
    st.header("Récap variations — tous les FCPs")
    as_of_ts = pd.Timestamp(as_of)

    tx_all = _tx_all_global
    prices = _prices_global
    all_fcps = _all_fcps_global
    tx_h, pr_h = len(tx_all), len(prices)
    recap = _cached_recap_data(as_of_ts.isoformat(), tx_h, pr_h)
    divs_by_fcp = _build_divs_by_fcp(all_fcps, as_of_ts)

    tab_day, tab_period = st.tabs(["📅 Variation du jour / YTD", "📆 Analyse sur période"])

    # ── Tab 1: Day / YTD ─────────────────────────────────────────────────────
    with tab_day:
        st.caption(
            f"Variations à la date du **{as_of_ts.strftime('%d/%m/%Y')}** — "
            f"YTD calculé depuis le 01/01/{as_of_ts.year}"
        )

        if recap.empty:
            st.info("Aucune donnée à afficher.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            total_valo_all = float(recap["Valorisation"].sum())
            total_var_jour = float(recap["Var. jour"].sum())
            total_var_ytd  = float(recap["Var. YTD"].sum())
            col1.metric("Valorisation totale", fmt_xof(total_valo_all))
            col2.metric("Variation jour cumulée", fmt_xof(total_var_jour, signed=True))
            col3.metric("Variation YTD cumulée", fmt_xof(total_var_ytd, signed=True))
            col4.metric("FCPs actifs",
                        f"{(recap['Valorisation']>0).sum()} / {len(recap)}")

            st.divider()

            display = recap.copy()
            display["Valorisation"]   = display["Valorisation"].map(fmt_xof)
            display["Var. jour"]      = display["Var. jour"].map(lambda v: fmt_xof(v, signed=True))
            display["Var. jour %"]    = display["Var. jour %"].map(fmt_pct)
            display["Valo début année"] = display["Valo début année"].map(fmt_xof)
            display["Var. YTD"]       = display["Var. YTD"].map(lambda v: fmt_xof(v, signed=True))
            display["Var. YTD %"]     = display["Var. YTD %"].map(fmt_pct)

            render_table(display, height=None,
                         color_cols=["Var. jour","Var. jour %",
                                     "Var. YTD","Var. YTD %"])

            col_csv, col_pdf = st.columns([1, 1])
            with col_csv:
                _xlsx_btn(
                    "📊 Exporter le récap (.xlsx)", recap,
                    f"recap_variations_{as_of_ts.date()}.xlsx",
                    key="dl_recap", sheet_name="Récap variations",
                )
            with col_pdf:
                if st.button("📄 Générer le rapport PDF", type="primary",
                             key="gen_pdf"):
                    with st.spinner("Analyse des drivers de performance…"):
                        try:
                            from report_pdf import generate_report
                            from portfolio import (
                                compute_action_drivers,
                                aggregate_drivers_by_ticker,
                            )
                            from sectors import _load as _load_sectors
                            drivers = compute_action_drivers(
                                tx_all, prices, all_fcps, as_of_ts, divs_by_fcp,
                            )
                            by_ticker = aggregate_drivers_by_ticker(drivers)
                            sectors_map = _load_sectors()
                            pdf_bytes = generate_report(
                                drivers=drivers, by_ticker=by_ticker,
                                as_of=as_of_ts, sectors_map=sectors_map,
                                recap=recap,
                            )
                            st.download_button(
                                "⬇️ Télécharger le rapport",
                                data=pdf_bytes,
                                file_name=f"rapport_analytics_actions_{as_of_ts.date()}.pdf",
                                mime="application/pdf",
                                key="dl_pdf_report",
                            )
                        except Exception as e:
                            st.error(f"Génération PDF impossible : {e}")

            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Top variations du jour")
                top_day = (
                    recap.assign(abs_=recap["Var. jour"].abs())
                    .nlargest(10, "abs_")[["FCP","Var. jour","Var. jour %"]]
                )
                top_day["Var. jour"]   = top_day["Var. jour"].map(lambda v: fmt_xof(v, signed=True))
                top_day["Var. jour %"] = top_day["Var. jour %"].map(fmt_pct)
                render_table(top_day, height=None,
                             color_cols=["Var. jour","Var. jour %"])
            with col_b:
                st.subheader("Top variations YTD")
                top_ytd = (
                    recap.assign(abs_=recap["Var. YTD"].abs())
                    .nlargest(10, "abs_")[["FCP","Var. YTD","Var. YTD %"]]
                )
                top_ytd["Var. YTD"]   = top_ytd["Var. YTD"].map(lambda v: fmt_xof(v, signed=True))
                top_ytd["Var. YTD %"] = top_ytd["Var. YTD %"].map(fmt_pct)
                render_table(top_ytd, height=None,
                             color_cols=["Var. YTD","Var. YTD %"])

    # ── Tab 2: Analyse sur période ────────────────────────────────────────────
    with tab_period:
        st.subheader("Analyse de performance sur une période personnalisée")
        st.caption(
            "Calcul basé sur les positions réelles à chaque date "
            "(transactions jusqu'à la date de début, transactions jusqu'à la date de fin). "
            "La **variation nette** déduit les flux d'investissement (achats − ventes) "
            "pour isoler la performance pure du marché."
        )

        col_d1, col_d2, col_mode = st.columns([1, 1, 2])
        with col_d1:
            default_start = pd.Timestamp(year=as_of_ts.year, month=1, day=1).date()
            period_start = st.date_input(
                "Date de début",
                value=default_start,
                key="period_start",
            )
        with col_d2:
            period_end = st.date_input(
                "Date de fin",
                value=as_of_ts.date(),
                key="period_end",
            )
        with col_mode:
            show_mode = st.radio(
                "Affichage",
                ["Variation nette (performance pure)",
                 "Variation brute",
                 "Les deux"],
                horizontal=True,
                key="period_mode",
            )

        if period_start >= period_end:
            st.warning("La date de début doit être strictement antérieure à la date de fin.")
        else:
            with st.spinner(
                f"Calcul des performances "
                f"{pd.Timestamp(period_start).strftime('%d/%m/%Y')} → "
                f"{pd.Timestamp(period_end).strftime('%d/%m/%Y')}…"
            ):
                from portfolio import compute_period_perf
                period_result = compute_period_perf(
                    tx_all, prices, all_fcps,
                    pd.Timestamp(period_start),
                    pd.Timestamp(period_end),
                    divs_by_fcp,
                )

            if period_result.empty:
                st.info("Aucune donnée disponible.")
            else:
                n_days = (pd.Timestamp(period_end) - pd.Timestamp(period_start)).days
                period_label = (
                    f"{pd.Timestamp(period_start).strftime('%d/%m/%Y')} "
                    f"→ {pd.Timestamp(period_end).strftime('%d/%m/%Y')} "
                    f"({n_days} jours)"
                )
                st.caption(f"**Période :** {period_label}")

                # KPIs
                total_valo_fin = float(period_result["Valo fin"].sum())
                total_var_nette = float(period_result["Variation nette"].sum())
                total_var_brute = float(period_result["Variation brute"].sum())
                total_flux = float(period_result["Flux nets"].sum())
                n_pos = int((period_result["Variation nette"] > 0).sum())
                n_neg = int((period_result["Variation nette"] < 0).sum())

                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("Valo fin de période", fmt_xof(total_valo_fin))
                c2.metric("Var. nette cumulée", fmt_xof(total_var_nette, signed=True))
                c3.metric("Var. brute cumulée", fmt_xof(total_var_brute, signed=True))
                c4.metric("Flux nets période", fmt_xof(total_flux, signed=True))
                c5.metric("FCPs ↑ / ↓", f"{n_pos} / {n_neg}")

                st.divider()

                # Build display table according to mode
                pr = period_result.copy()
                base_cols = {
                    "FCP":         pr["FCP"],
                    "Valo début":  pr["Valo début"].map(fmt_xof),
                    "Valo fin":    pr["Valo fin"].map(fmt_xof),
                    "Flux nets":   pr["Flux nets"].map(lambda v: fmt_xof(v, signed=True)),
                }
                nette_cols = {
                    "Var. nette":   pr["Variation nette"].map(lambda v: fmt_xof(v, signed=True)),
                    "Var. nette %": pr["Variation nette %"].map(lambda v: fmt_pct(v, signed=True)),
                }
                brute_cols = {
                    "Var. brute":   pr["Variation brute"].map(lambda v: fmt_xof(v, signed=True)),
                    "Var. brute %": pr["Variation brute %"].map(lambda v: fmt_pct(v, signed=True)),
                }

                color_cols = []
                if show_mode == "Variation nette (performance pure)":
                    disp = pd.DataFrame({**base_cols, **nette_cols})
                    color_cols = ["Var. nette","Var. nette %"]
                elif show_mode == "Variation brute":
                    disp = pd.DataFrame({**base_cols, **brute_cols})
                    color_cols = ["Var. brute","Var. brute %"]
                else:  # Les deux
                    disp = pd.DataFrame({**base_cols, **nette_cols, **brute_cols})
                    color_cols = ["Var. nette","Var. nette %","Var. brute","Var. brute %"]

                render_table(disp, height=None, color_cols=color_cols)

                # Top performers
                st.divider()
                col_p, col_n = st.columns(2)
                with col_p:
                    st.subheader("Top performances nettes")
                    top_p = pr.nlargest(5, "Variation nette")[
                        ["FCP","Variation nette","Variation nette %"]
                    ].rename(columns={
                        "Variation nette":   "Var. nette",
                        "Variation nette %": "Var. nette %",
                    })
                    top_p["Var. nette"]   = top_p["Var. nette"].map(lambda v: fmt_xof(v, signed=True))
                    top_p["Var. nette %"] = top_p["Var. nette %"].map(lambda v: fmt_pct(v, signed=True))
                    render_table(top_p, height=None, color_cols=["Var. nette","Var. nette %"])
                with col_n:
                    st.subheader("Moins bonnes performances")
                    top_n = pr.nsmallest(5, "Variation nette")[
                        ["FCP","Variation nette","Variation nette %"]
                    ].rename(columns={
                        "Variation nette":   "Var. nette",
                        "Variation nette %": "Var. nette %",
                    })
                    top_n["Var. nette"]   = top_n["Var. nette"].map(lambda v: fmt_xof(v, signed=True))
                    top_n["Var. nette %"] = top_n["Var. nette %"].map(lambda v: fmt_pct(v, signed=True))
                    render_table(top_n, height=None, color_cols=["Var. nette","Var. nette %"])

                _xlsx_btn(
                    "📊 Exporter la performance (.xlsx)", period_result,
                    f"perf_periode_{pd.Timestamp(period_start).strftime('%Y%m%d')}_{pd.Timestamp(period_end).strftime('%Y%m%d')}.xlsx",
                    key="dl_period", sheet_name="Performance période",
                )

        # ── Tableau d'attribution de performance ─────────────────────────
        st.divider()
        st.subheader("Tableau d'attribution de performance")
        st.caption(
            "Décomposition de la variation de valeur du portefeuille actions "
            "entre la date de début et la date de fin choisies. "
            "**Effet marché** = Ptf. fin − Ptf. début − Achats + Ventes − Dividendes"
        )

        if not (period_start >= period_end):
            with st.spinner("Calcul du tableau d'attribution…"):
                from portfolio import compute_attribution
                divs_dated = _cached_dividends_dated()
                attr = compute_attribution(
                    tx_all, prices, all_fcps,
                    pd.Timestamp(period_start),
                    pd.Timestamp(period_end),
                    divs_dated,
                )

            if attr.empty:
                st.info("Aucune donnée disponible.")
            else:
                # Totals row
                totals = attr[[c for c in attr.columns if c != "FCP"]].sum()
                totals_row = pd.DataFrame([{
                    "FCP": "TOTAL",
                    **{c: totals[c] for c in totals.index}
                }])
                attr_with_total = pd.concat([attr, totals_row], ignore_index=True)

                # Format for display
                money_cols = [
                    "Ptf. Actions début", "Achats", "Ventes",
                    "Dividendes", "Effet marché", "Ptf. Actions fin"
                ]
                disp_attr = attr_with_total.copy()
                for col in money_cols:
                    disp_attr[col] = disp_attr[col].map(
                        lambda v: fmt_xof(v, signed=(col == "Effet marché"))
                    )

                render_table(
                    disp_attr, height=None,
                    color_cols=["Effet marché"],
                )

                # Export buttons
                col_xl, col_pdf_attr = st.columns(2)

                with col_xl:
                    # Excel export
                    try:
                        import io as _io
                        import openpyxl
                        from openpyxl.styles import (
                            Font, PatternFill, Alignment, Border, Side
                        )
                        wb = openpyxl.Workbook()
                        ws = wb.active
                        ws.title = "Attribution"

                        # Header
                        headers = list(attr_with_total.columns)
                        header_fill = PatternFill(
                            "solid", fgColor="004977"
                        )
                        for col_i, h in enumerate(headers, 1):
                            cell = ws.cell(row=1, column=col_i, value=h)
                            cell.font = Font(
                                bold=True, color="FFFFFF", size=10
                            )
                            cell.fill = header_fill
                            cell.alignment = Alignment(horizontal="center")

                        # Data rows
                        for row_i, row_data in enumerate(
                            attr_with_total.itertuples(index=False), 2
                        ):
                            is_total = row_data.FCP == "TOTAL"
                            for col_i, val in enumerate(row_data, 1):
                                cell = ws.cell(row=row_i, column=col_i, value=val)
                                if is_total:
                                    cell.font = Font(bold=True, size=10)
                                    cell.fill = PatternFill(
                                        "solid", fgColor="E6EFF5"
                                    )
                                elif isinstance(val, float):
                                    cell.number_format = '#,##0'

                        # Column widths
                        ws.column_dimensions["A"].width = 35
                        for col_l in "BCDEFG":
                            ws.column_dimensions[col_l].width = 22

                        xl_buf = _io.BytesIO()
                        wb.save(xl_buf)
                        xl_buf.seek(0)

                        st.download_button(
                            "📊 Télécharger Excel (.xlsx)",
                            data=xl_buf.read(),
                            file_name=(
                                f"attribution_"
                                f"{pd.Timestamp(period_start).strftime('%Y%m%d')}_"
                                f"{pd.Timestamp(period_end).strftime('%Y%m%d')}.xlsx"
                            ),
                            mime=(
                                "application/vnd.openxmlformats-officedocument"
                                ".spreadsheetml.sheet"
                            ),
                            key="dl_attr_xl",
                        )
                    except Exception as e:
                        st.error(f"Export Excel impossible : {e}")

                with col_pdf_attr:
                    if st.button(
                        "📄 Générer PDF Attribution",
                        key="gen_attr_pdf"
                    ):
                        with st.spinner("Génération PDF…"):
                            try:
                                from report_pdf import (
                                    generate_attribution_pdf
                                )
                                pdf_bytes = generate_attribution_pdf(
                                    attr=attr_with_total,
                                    date_debut=pd.Timestamp(period_start),
                                    date_fin=pd.Timestamp(period_end),
                                )
                                st.download_button(
                                    "⬇️ Télécharger le PDF",
                                    data=pdf_bytes,
                                    file_name=(
                                        f"attribution_"
                                        f"{pd.Timestamp(period_start).strftime('%Y%m%d')}_"
                                        f"{pd.Timestamp(period_end).strftime('%Y%m%d')}.pdf"
                                    ),
                                    mime="application/pdf",
                                    key="dl_attr_pdf",
                                )
                            except Exception as e:
                                st.error(
                                    f"Génération PDF impossible : {e}"
                                )


# ---------------------------------------------------------------------------
# Page: Expositions
# ---------------------------------------------------------------------------
elif page == "🎯 Expositions":
    st.header("Expositions cross-portefeuille")
    as_of_ts = pd.Timestamp(as_of)

    # ── Filtre FCP ──────────────────────────────────────────────────────────
    tx_all = _tx_all_global
    prices = _prices_global
    all_fcps = _all_fcps_global
    divs_by_fcp = _build_divs_by_fcp(all_fcps, as_of_ts)
    filtre_options = ["Tous les FCPs"] + all_fcps
    filtre_fcp = st.selectbox(
        "Périmètre d'analyse",
        options=filtre_options,
        index=0,
        key="exp_filtre_fcp",
    )

    # Compute exposures for the selected scope
    if filtre_fcp == "Tous les FCPs":
        fcps_scope = all_fcps
        scope_label = "tous les FCPs"
    else:
        fcps_scope = [filtre_fcp]
        scope_label = filtre_fcp

    st.caption(
        f"Photographie des positions au **{as_of_ts.strftime('%d/%m/%Y')}** — "
        f"périmètre : **{scope_label}**."
    )

    with st.spinner(f"Calcul des expositions — {scope_label}…"):
        tx_h2, pr_h2 = len(tx_all), len(prices)
        exp = _cached_exposures_data(
            as_of_ts.isoformat(), tx_h2, pr_h2,
            "ALL" if filtre_fcp == "Tous les FCPs" else filtre_fcp,
        )

    if exp.empty:
        st.info("Aucune position à afficher pour ce périmètre.")
    else:
        total_global = float(exp["valorisation"].sum())

        # --- KPIs ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Valorisation", fmt_xof(total_global))
        c2.metric("Tickers distincts", f"{exp['ticker'].nunique()}")
        c3.metric(
            "FCPs" if filtre_fcp == "Tous les FCPs" else "FCP",
            f"{exp['fcp'].nunique()}" if filtre_fcp == "Tous les FCPs" else filtre_fcp,
        )
        c4.metric("Lignes totales", f"{len(exp)}")

        st.divider()

        # --- Section 1: Top tickers ---
        st.subheader(
            "Top expositions — tous FCPs confondus"
            if filtre_fcp == "Tous les FCPs"
            else f"Top expositions — {filtre_fcp}"
        )

        global_by_ticker = (
            exp.groupby("ticker")["valorisation"].sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        global_by_ticker["Poids"] = global_by_ticker["valorisation"] / total_global

        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.caption("Top 15 — barres en FCFA")
            top15 = global_by_ticker.head(15).set_index("ticker")["valorisation"]
            st.bar_chart(top15)

        with col_b:
            st.caption("Concentration top 10 vs reste")
            top10 = global_by_ticker.head(10)
            others_val = total_global - top10["valorisation"].sum()
            pie_df = pd.concat([
                top10[["ticker", "valorisation"]],
                pd.DataFrame([{"ticker": "Autres", "valorisation": others_val}]),
            ]).set_index("ticker")
            st.bar_chart(pie_df)

        # Full ranked table (collapsed by default)
        with st.expander(f"Voir tous les tickers ({len(global_by_ticker)})"):
            display_global = global_by_ticker.rename(columns={
                "ticker": "Symbole",
                "valorisation": "Valorisation",
            }).copy()
            display_global["Valorisation"] = display_global["Valorisation"].map(fmt_xof)
            display_global["Poids"] = display_global["Poids"].map(fmt_pct)
            render_table(display_global, height=None)

            st.download_button(
                "📊 Exporter (.xlsx)",
                data=_to_xlsx(global_by_ticker, sheet_name="Top expositions"),
                file_name=f"expositions_globales_{as_of_ts.date()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_global",
            )

        st.divider()

        # --- Section 1bis: Répartition sectorielle ---
        st.subheader("Répartition sectorielle")

        exp_with_sector = annotate_sectors(exp)
        sector_global = (
            exp_with_sector.groupby("secteur")["valorisation"].sum()
            .sort_values(ascending=False)
        )
        sector_global_pct = sector_global / sector_global.sum()

        col_s1, col_s2 = st.columns([3, 2])
        with col_s1:
            st.caption("Exposition par secteur (FCFA)")
            st.bar_chart(sector_global)
        with col_s2:
            st.caption("Indicateurs sectoriels")
            top_sector = sector_global.index[0]
            top_sector_pct = sector_global_pct.iloc[0]
            n_sectors = len(sector_global)
            st.metric("Secteur dominant", top_sector)
            st.metric("Poids du secteur dominant", fmt_pct(top_sector_pct))
            st.metric("Nb secteurs représentés", f"{n_sectors}")

        # Detailed sector table with weights
        sector_table = pd.DataFrame({
            "Secteur": sector_global.index,
            "Valorisation": sector_global.values,
            "Poids global": sector_global_pct.values,
            "Nb tickers": [
                exp_with_sector[exp_with_sector["secteur"] == s]["ticker"].nunique()
                for s in sector_global.index
            ],
        })
        display_sector = sector_table.copy()
        display_sector["Valorisation"] = display_sector["Valorisation"].map(fmt_xof)
        display_sector["Poids global"] = display_sector["Poids global"].map(fmt_pct)
        render_table(display_sector, height=None)

        # Sector × FCP matrix
        with st.expander("Voir la matrice secteurs × FCPs"):
            sec_view_mode = st.radio(
                "Mode d'affichage",
                ["Montant (FCFA)", "Poids dans le FCP", "Poids global"],
                horizontal=True,
                key="exp_sec_view_mode",
            )

            sector_fcp = exp_with_sector.pivot_table(
                index="secteur", columns="fcp", values="valorisation",
                aggfunc="sum", fill_value=0,
            )
            sector_fcp = sector_fcp.loc[sector_global.index]  # keep order

            if sec_view_mode == "Montant (FCFA)":
                m = sector_fcp.copy()
                m["TOTAL"] = m.sum(axis=1)
                total_row = m.sum(axis=0)
                total_row.name = "TOTAL"
                m = pd.concat([m, total_row.to_frame().T])
                formatted = m.reset_index().rename(
                    columns={"secteur": "Secteur"}
                ).apply(lambda col: col.map(
                    lambda v: f"{v:,.0f}".replace(",", " ")
                    if isinstance(v, (int, float)) and v > 0 else
                    ("-" if isinstance(v, (int, float)) else str(v))
                ))
                export_sec_df = m
            elif sec_view_mode == "Poids dans le FCP":
                fcp_totals = sector_fcp.sum(axis=0)
                m = sector_fcp.div(fcp_totals, axis=1).fillna(0)
                formatted = m.reset_index().rename(
                    columns={"secteur": "Secteur"}
                ).apply(lambda col: col.map(
                    lambda v: f"{v*100:.1f}%"
                    if isinstance(v, (int, float)) and v > 0 else
                    ("-" if isinstance(v, (int, float)) else str(v))
                ))
                export_sec_df = m
            else:  # Poids global
                m = sector_fcp / total_global
                m["TOTAL"] = m.sum(axis=1)
                formatted = m.reset_index().rename(
                    columns={"secteur": "Secteur"}
                ).apply(lambda col: col.map(
                    lambda v: f"{v*100:.2f}%"
                    if isinstance(v, (int, float)) and v > 0 else
                    ("-" if isinstance(v, (int, float)) else str(v))
                ))
                export_sec_df = m

            render_table(formatted, height=None)

            st.download_button(
                "📊 Exporter la matrice sectorielle (.xlsx)",
                data=_to_xlsx(export_sec_df.reset_index(), sheet_name="Secteurs"),
                file_name=f"matrice_secteurs_{as_of_ts.date()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_sec_matrix",
            )

        st.divider()

        # --- Section 2: Matrice ticker × FCP ---
        st.subheader(
            "Matrice détaillée tickers × FCPs"
            if filtre_fcp == "Tous les FCPs"
            else f"Détail des positions — {filtre_fcp}"
        )

        if filtre_fcp != "Tous les FCPs":
            # Single FCP view: simple table, no pivot needed
            view_mode_single = st.radio(
                "Mode d'affichage",
                ["Montant (FCFA)", "Poids dans le FCP"],
                horizontal=True,
                key="exp_view_mode",
            )
            single = global_by_ticker.copy().rename(
                columns={"ticker": "Ticker", "valorisation": "Valorisation"}
            )
            if view_mode_single == "Montant (FCFA)":
                single["Valorisation"] = single["Valorisation"].map(fmt_xof)
                single["Poids"] = single["Poids"].map(fmt_pct)
                render_table(single, height=500)
                export_df = global_by_ticker
            else:
                single["Valorisation"] = single["Valorisation"].map(fmt_xof)
                single["Poids"] = single["Poids"].map(fmt_pct)
                render_table(single, height=500)
                export_df = global_by_ticker
            _xlsx_btn(
                "📊 Exporter (.xlsx)", export_df,
                f"positions_{filtre_fcp.replace(' ','_')}_{as_of_ts.date()}.xlsx",
                key="dl_matrix", sheet_name="Expositions",
            )

        else:
            # All FCPs: full pivot matrix
            view_mode = st.radio(
                "Mode d'affichage",
                ["Montant (FCFA)", "Poids dans le FCP", "Poids global"],
                horizontal=True,
                key="exp_view_mode",
            )

            matrix_val = exp.pivot_table(
                index="ticker", columns="fcp", values="valorisation",
                aggfunc="sum", fill_value=0,
            )
            matrix_val = matrix_val.loc[global_by_ticker["ticker"].tolist()]

            if view_mode == "Montant (FCFA)":
                matrix_display = matrix_val.copy()
                matrix_display["TOTAL"] = matrix_display.sum(axis=1)
                total_row = matrix_display.sum(axis=0)
                total_row.name = "TOTAL"
                matrix_display = pd.concat([matrix_display, total_row.to_frame().T])
                formatted = matrix_display.reset_index().rename(
                    columns={"ticker": "Ticker"}
                ).apply(lambda col: col.map(
                    lambda v: f"{v:,.0f}".replace(",", " ")
                    if isinstance(v, (int, float)) and v > 0 else
                    ("-" if isinstance(v, (int, float)) else str(v))
                ))
                render_table(formatted, height=600)
                export_df = matrix_display

            elif view_mode == "Poids dans le FCP":
                fcp_totals = matrix_val.sum(axis=0)
                matrix_pct = matrix_val.div(fcp_totals, axis=1).fillna(0)
                formatted = matrix_pct.reset_index().rename(
                    columns={"ticker": "Ticker"}
                ).apply(lambda col: col.map(
                    lambda v: f"{v*100:.1f}%"
                    if isinstance(v, (int, float)) and v > 0 else
                    ("-" if isinstance(v, (int, float)) else str(v))
                ))
                render_table(formatted, height=600)
                export_df = matrix_pct

            else:  # Poids global
                matrix_pct = matrix_val / total_global
                matrix_pct["TOTAL"] = matrix_pct.sum(axis=1)
                formatted = matrix_pct.reset_index().rename(
                    columns={"ticker": "Ticker"}
                ).apply(lambda col: col.map(
                    lambda v: f"{v*100:.2f}%"
                    if isinstance(v, (int, float)) and v > 0 else
                    ("-" if isinstance(v, (int, float)) else str(v))
                ))
                render_table(formatted, height=600)
                export_df = matrix_pct

            st.download_button(
                "📊 Exporter la matrice (.xlsx)",
                data=_to_xlsx(export_df.reset_index(), sheet_name="Matrice expositions"),
                file_name=f"matrice_expositions_{as_of_ts.date()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_matrix",
            )

        st.divider()

        # --- Section 3: Concentration par FCP ---
        st.subheader(
            "Concentration par FCP"
            if filtre_fcp == "Tous les FCPs"
            else f"Concentration — {filtre_fcp}"
        )
        st.caption(
            "Poids cumulé des plus grosses positions. "
            "Permet d'identifier les portefeuilles les plus concentrés."
        )

        conc = concentration_metrics(exp)
        if not conc.empty:
            display_conc = conc.copy()
            display_conc["Valorisation"] = display_conc["Valorisation"].map(fmt_xof)
            for c in ["Top 1 %", "Top 3 %", "Top 5 %", "Top 10 %"]:
                display_conc[c] = display_conc[c].map(fmt_pct)
            render_table(display_conc, height=None)

            st.download_button(
                "📊 Exporter (.xlsx)",
                data=_to_xlsx(conc, sheet_name="Concentration"),
                file_name=f"concentration_{as_of_ts.date()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_conc",
            )

    # ── Section 4: Dividendes perçus par titre et par FCP ──────────────────
    st.divider()
    st.subheader("💰 Dividendes perçus")
    st.caption("Montant total reçu par titre et par FCP selon l'année choisie.")

    current_year = as_of_ts.year
    div_year = st.selectbox(
        "Année",
        options=list(range(current_year, current_year - 6, -1)),
        index=0,
        key="div_year_select",
    )

    all_divs_dated = _cached_dividends_dated()
    # Filter dividends for selected year
    year_divs = [
        d for d in all_divs_dated
        if pd.Timestamp(d["payment_date"]).year == div_year
    ]

    if not year_divs:
        st.info(f"Aucun dividende enregistré pour l'année {div_year}.")
    else:
        # Build recap: for each dividend, compute amount per FCP
        from portfolio import compute_positions as _cp

        recap_rows = []
        tx_all_div = _tx_all_global
        all_fcps_div = _all_fcps_global

        for d in year_divs:
            ticker_d = str(d["ticker"]).strip().upper()
            amount_d = float(d["amount"])
            pay_date = pd.Timestamp(d["payment_date"])

            for fcp_name in all_fcps_div:
                pos = _cp(tx_all_div, fcp_name, pay_date)
                if pos.empty:
                    continue
                row_t = pos[pos["ticker"] == ticker_d]
                if row_t.empty:
                    continue
                qty = float(row_t.iloc[0]["quantite"])
                if qty <= 0:
                    continue
                total_div = qty * amount_d
                recap_rows.append({
                    "FCP":           fcp_name,
                    "Titre":         ticker_d,
                    "Date paiement": pay_date.strftime("%d/%m/%Y"),
                    "Div./action":   amount_d,
                    "Qté détenue":   qty,
                    "Total reçu":    total_div,
                })

        if not recap_rows:
            st.info(
                f"Aucune position détenue aux dates de paiement "
                f"des dividendes de {div_year}."
            )
        else:
            div_df = pd.DataFrame(recap_rows)

            # Global KPIs
            total_all = float(div_df["Total reçu"].sum())
            n_tickers = div_df["Titre"].nunique()
            n_fcps_d  = div_df["FCP"].nunique()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total dividendes perçus", fmt_xof(total_all))
            c2.metric("Titres concernés", str(n_tickers))
            c3.metric("FCPs concernés", str(n_fcps_d))

            st.divider()

            # ── Matrice Titres × FCPs ───────────────────────────────────────
            st.subheader("Matrice dividendes — Titres × FCPs")
            st.caption("Montant total reçu (FCFA) par titre (lignes) et par FCP (colonnes).")

            # Pivot: rows=ticker, cols=FCP, values=Total reçu
            matrix = div_df.pivot_table(
                index=["Titre", "Date paiement", "Div./action"],
                columns="FCP",
                values="Total reçu",
                aggfunc="sum",
                fill_value=0,
            ).reset_index()

            # Add a TOTAL column
            fcp_cols = [c for c in matrix.columns
                        if c not in ["Titre", "Date paiement", "Div./action"]]
            matrix["TOTAL"] = matrix[fcp_cols].sum(axis=1)

            # Sort by TOTAL descending
            matrix = matrix.sort_values("TOTAL", ascending=False).reset_index(drop=True)

            # Add TOTAL row at bottom
            total_row = {"Titre": "TOTAL", "Date paiement": "", "Div./action": ""}
            for col in fcp_cols + ["TOTAL"]:
                total_row[col] = float(matrix[col].sum())
            matrix = pd.concat(
                [matrix, pd.DataFrame([total_row])],
                ignore_index=True,
            )

            # Format for display
            disp_matrix = matrix.copy()
            disp_matrix["Div./action"] = disp_matrix["Div./action"].map(
                lambda v: f"{v:,.0f} FCFA".replace(",", " ")
                if isinstance(v, (int, float)) and v else str(v or "")
            )
            for col in fcp_cols + ["TOTAL"]:
                disp_matrix[col] = disp_matrix[col].map(
                    lambda v: fmt_xof(v) if isinstance(v, (int, float)) and v > 0
                    else ("—" if isinstance(v, (int, float)) else str(v or ""))
                )

            render_table(disp_matrix, height=500)

            # Export
            st.download_button(
                "📊 Exporter la matrice dividendes (.xlsx)",
                data=_to_xlsx(disp_matrix, sheet_name=f"Dividendes {div_year}"),
                file_name=f"dividendes_matrice_{div_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_dividendes",
            )

    # ── Section 5: Dividendes à percevoir ────────────────────────────────────
    st.divider()
    st.subheader("📅 Dividendes à percevoir")
    st.caption(
        "Estimation basée sur les positions à la **date de valorisation** sélectionnée "
        "dans la sidebar. Les quantités peuvent évoluer avant la date de paiement. "
        "Seuls les dividendes dont la date de paiement est **postérieure** à la date "
        "de valorisation sont affichés."
    )

    all_divs_dated_fut = _cached_dividends_dated()
    # Dividends with payment_date AFTER as_of_ts (future payments)
    future_divs = [
        d for d in all_divs_dated_fut
        if pd.Timestamp(d["payment_date"]).normalize() > as_of_ts.normalize()
    ]

    if not future_divs:
        st.info(
            "Aucun dividende futur enregistré. "
            "Ajoutez des dividendes avec une date de paiement future "
            "dans l'onglet **⚙️ Paramètres**."
        )
    else:
        st.warning(
            f"⚠️ **{len(future_divs)} dividende(s) futur(s)** — estimation basée sur "
            f"les positions au {as_of_ts.strftime('%d/%m/%Y')}. "
            "Les acquisitions/cessions futures modifieront ces montants."
        )

        from portfolio import compute_positions as _cp_fut

        fut_rows = []
        tx_all_fut = _tx_all_global
        all_fcps_fut = _all_fcps_global

        for d in future_divs:
            ticker_d = str(d["ticker"]).strip().upper()
            amount_d = float(d["amount"])
            pay_date = pd.Timestamp(d["payment_date"])

            for fcp_name in all_fcps_fut:
                # Use positions at as_of_ts (current snapshot)
                pos = _cp_fut(tx_all_fut, fcp_name, as_of_ts)
                if pos.empty:
                    continue
                row_t = pos[pos["ticker"] == ticker_d]
                if row_t.empty:
                    continue
                qty = float(row_t.iloc[0]["quantite"])
                if qty <= 0:
                    continue
                fut_rows.append({
                    "FCP":              fcp_name,
                    "Titre":            ticker_d,
                    "Date paiement":    pay_date.strftime("%d/%m/%Y"),
                    "Jours restants":   (pay_date - as_of_ts).days,
                    "Div./action":      amount_d,
                    "Qté actuelle":     qty,
                    "Estimation reçu":  qty * amount_d,
                })

        if not fut_rows:
            st.info("Aucune position actuelle sur les titres avec dividendes futurs.")
        else:
            fut_df = pd.DataFrame(fut_rows)

            # KPIs
            total_fut = float(fut_df["Estimation reçu"].sum())
            n_tick_fut = fut_df["Titre"].nunique()
            n_fcp_fut  = fut_df["FCP"].nunique()
            c1, c2, c3 = st.columns(3)
            c1.metric("Estimation totale", fmt_xof(total_fut))
            c2.metric("Titres concernés", str(n_tick_fut))
            c3.metric("FCPs concernés", str(n_fcp_fut))

            st.divider()

            # ── Matrice Titres × FCPs ─────────────────────────────────────
            st.subheader("Matrice dividendes à percevoir — Titres × FCPs")
            st.caption(
                "Montant estimé (FCFA) basé sur les positions actuelles. "
                "🔸 = paiement dans moins de 30 jours."
            )

            # Pivot
            mat_fut = fut_df.pivot_table(
                index=["Titre", "Date paiement", "Jours restants", "Div./action"],
                columns="FCP",
                values="Estimation reçu",
                aggfunc="sum",
                fill_value=0,
            ).reset_index()

            fcp_cols_fut = [
                c for c in mat_fut.columns
                if c not in ["Titre", "Date paiement", "Jours restants", "Div./action"]
            ]
            mat_fut["TOTAL"] = mat_fut[fcp_cols_fut].sum(axis=1)
            mat_fut = mat_fut.sort_values(
                ["Date paiement", "TOTAL"], ascending=[True, False]
            ).reset_index(drop=True)

            # TOTAL row
            total_row_fut = {
                "Titre": "TOTAL", "Date paiement": "",
                "Jours restants": "", "Div./action": "",
            }
            for col in fcp_cols_fut + ["TOTAL"]:
                total_row_fut[col] = float(mat_fut[col].sum())
            mat_fut = pd.concat(
                [mat_fut, pd.DataFrame([total_row_fut])],
                ignore_index=True,
            )

            # Format
            disp_fut = mat_fut.copy()
            disp_fut["Div./action"] = disp_fut["Div./action"].map(
                lambda v: f"{v:,.0f} FCFA".replace(",", " ")
                if isinstance(v, (int, float)) and v else str(v or "")
            )
            disp_fut["Jours restants"] = disp_fut["Jours restants"].map(
                lambda v: f"🔸 {int(v)}j" if isinstance(v, (int, float)) and 0 < v <= 30
                else (f"{int(v)}j" if isinstance(v, (int, float)) and v > 0 else "")
            )
            for col in fcp_cols_fut + ["TOTAL"]:
                disp_fut[col] = disp_fut[col].map(
                    lambda v: fmt_xof(v) if isinstance(v, (int, float)) and v > 0
                    else ("—" if isinstance(v, (int, float)) else str(v or ""))
                )

            render_table(disp_fut, height=400)

            st.download_button(
                "📊 Exporter les dividendes à percevoir (.xlsx)",
                data=_to_xlsx(disp_fut, sheet_name="Dividendes à percevoir"),
                file_name=f"dividendes_a_percevoir_{as_of_ts.date()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_dividendes_fut",
            )


# ---------------------------------------------------------------------------
# Page: Suivi des cibles
# ---------------------------------------------------------------------------
elif page == "📋 Suivi des cibles":
    st.header(f"Suivi des cibles — {fcp}")
    as_of_ts = pd.Timestamp(as_of)
    st.caption(
        f"Comparaison positions actuelles vs pondérations cibles "
        f"au **{as_of_ts.strftime('%d/%m/%Y')}**."
    )

    tx_all = _cached_transactions()
    prices = _cached_prices()
    divs = _build_divs_by_fcp(list(fcps), as_of_ts).get(fcp, {})

    # ── Import de cibles ────────────────────────────────────────────────────
    with st.expander("📥 Importer les cibles (CSV ou Excel)", expanded=False):

        # FCPs exclus de l'import global
        EXCLUDED_FCPS = {"FCP AL BARAKA", "FCP AL BARAKA 2"}

        st.caption(
            "**Format attendu** : colonnes **Name** (A), **Ticker** (B), "
            "**Weight** (C, en décimal — ex: 0.0762 = 7.62%). "
            "Les pondérations s'appliquent à **tous les FCPs** "
            "sauf **FCP AL BARAKA** et **FCP AL BARAKA 2**."
        )

        template_ntw = (
            "Name,Ticker,Weight\n"
            "NESTLE CI,NTLC,0.0762\n"
            "SONATEL SN,SNTS,0.0857\n"
            "ETI TG,ETIT,0.0810\n"
        )
        st.download_button(
            "📄 Télécharger le modèle",
            data=template_ntw.encode(),
            file_name="modele_ponderations_cibles.csv",
            mime="text/csv",
            key="dl_tpl_ntw",
        )

        up = st.file_uploader(
            "Fichier de pondérations (.xlsx ou .csv)",
            type=["csv", "xlsx"],
            key="targets_upload",
        )

        if up is not None:
            try:
                if up.name.endswith(".xlsx"):
                    raw = pd.read_excel(up, header=None)
                else:
                    raw = pd.read_csv(up, header=None)

                # Detect if first row is a header
                first_row = [str(v).strip().lower() for v in raw.iloc[0]]
                if any(w in first_row for w in ["name","ticker","weight","symbole","poids"]):
                    raw.columns = [str(v).strip().lower() for v in raw.iloc[0]]
                    raw = raw.iloc[1:].reset_index(drop=True)
                else:
                    raw.columns = [f"col{i}" for i in range(raw.shape[1])]
                    raw.columns = ["name","ticker","weight"][:raw.shape[1]]

                # Normalize to 3 columns
                raw.columns = list(raw.columns)[:3]
                raw.columns = ["name","ticker","weight"]
                raw = raw.dropna(subset=["ticker"])
                raw["ticker"] = raw["ticker"].astype(str).str.strip().str.upper()
                raw["weight_pct"] = pd.to_numeric(raw["weight"], errors="coerce")
                # Auto-convert if weights look like percentages (> 1)
                if raw["weight_pct"].dropna().max() > 1.5:
                    raw["weight_pct"] = raw["weight_pct"] / 100
                raw["amount_fcfa"] = None
                raw = raw[raw["weight_pct"].notna() & (raw["weight_pct"] > 0)]

                total_w = float(raw["weight_pct"].sum())
                target_fcps = [f for f in _cached_fcps() if f not in EXCLUDED_FCPS]

                # Preview
                st.write(
                    f"**{len(raw)}** titres détectés  ·  "
                    f"Poids total : **{total_w*100:.1f}%**  ·  "
                    f"Application sur **{len(target_fcps)} FCPs**"
                )
                if abs(total_w - 1.0) > 0.01:
                    st.warning(
                        f"⚠️ Le poids total est {total_w*100:.2f}% "
                        f"(attendu 100%). Vérifie les données avant de confirmer."
                    )

                preview_df = raw[["name","ticker","weight_pct"]].copy()
                preview_df["weight_pct"] = preview_df["weight_pct"].map(
                    lambda v: f"{v*100:.2f}%"
                )
                preview_df.columns = ["Nom", "Ticker", "Poids cible"]
                render_table(preview_df, height=350)

                st.caption(
                    "FCPs exclus : " + ", ".join(sorted(EXCLUDED_FCPS))
                    + " — ces fonds conservent leurs cibles actuelles."
                )

                if st.button(
                    "✅ Confirmer l'import sur tous les FCPs",
                    key="targets_import_confirm",
                    type="primary",
                ):
                    total_lines = 0
                    errors = []
                    for fcp_name in target_fcps:
                        try:
                            n = db.replace_targets_for_fcp(fcp_name, raw)
                            total_lines += n
                        except Exception as e_fcp:
                            errors.append(f"{fcp_name}: {e_fcp}")
                    _clear_data_cache()
                    if errors:
                        st.warning(
                            f"Import partiel : {total_lines} lignes chargées. "
                            f"Erreurs : {'; '.join(errors)}"
                        )
                    else:
                        st.success(
                            f"✅ {len(raw)} pondérations appliquées sur "
                            f"{len(target_fcps)} FCPs "
                            f"({total_lines} entrées au total)."
                        )
                    st.rerun()

            except Exception as e:
                st.error(f"Lecture impossible : {e}")
    # ── Saisie manuelle ─────────────────────────────────────────────────────
    with st.expander("✏️ Saisie / modification manuelle des cibles", expanded=False):
        st.caption(
            "Ajoute ou modifie une cible pour ce FCP. "
            "Renseigne au moins l'une des deux colonnes cibles."
        )

        known_tickers = _cached_known_tickers()
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            m_ticker = st.selectbox(
                "Titre", options=[""] + known_tickers, key="target_ticker"
            )
        with col_t2:
            m_weight = st.number_input(
                "Poids cible (%)", min_value=0.0, max_value=100.0,
                step=0.01, value=0.0, key="target_weight",
            )
        with col_t3:
            m_amount = st.number_input(
                "Montant cible (FCFA)", min_value=0.0, step=1000.0,
                value=0.0, key="target_amount",
            )

        col_save, col_del = st.columns([1, 1])
        with col_save:
            if st.button("💾 Enregistrer", key="target_save"):
                if not m_ticker:
                    st.warning("Sélectionne un ticker.")
                else:
                    db.upsert_target(
                        fcp, m_ticker,
                        m_weight if m_weight > 0 else None,
                        m_amount if m_amount > 0 else None,
                    )
                    _clear_data_cache()
                    st.success(f"✅ Cible enregistrée pour {m_ticker}.")
                    st.rerun()
        with col_del:
            if st.button("🗑️ Supprimer cette cible", key="target_del"):
                if not m_ticker:
                    st.warning("Sélectionne un ticker.")
                else:
                    db.delete_target(fcp, m_ticker)
                    _clear_data_cache()
                    st.success(f"Cible supprimée pour {m_ticker}.")
                    st.rerun()

    st.divider()

    # ── Sélecteur de périmètre ───────────────────────────────────────────────
    EXCLUDED_FROM_TARGETS = {"FCP AL BARAKA", "FCP AL BARAKA 2"}
    eligible_fcps = [f for f in fcps if f not in EXCLUDED_FROM_TARGETS]

    scope_options = ["Tous les FCPs"] + eligible_fcps
    scope = st.selectbox(
        "Périmètre d'analyse",
        options=scope_options,
        index=0,
        key="tracking_scope",
    )

    # ── Vue globale — tous les FCPs ──────────────────────────────────────────
    if scope == "Tous les FCPs":
        st.subheader("Suivi des cibles — Vue globale nettée")
        st.caption(
            "Agrégation par titre sur tous les FCPs éligibles. "
            "L'écart net est la somme des écarts individuels — "
            "un écart ≈ 0 signale une transaction interne possible entre FCPs."
        )

        # Compute tracking for each eligible FCP
        all_tracking_rows = []
        for fcp_name in eligible_fcps:
            targets_df_f = db.get_targets(fcp_name)
            if targets_df_f.empty:
                continue
            divs_f = _build_divs_by_fcp(fcps, as_of_ts).get(fcp_name, {})
            t = compute_tracking(
                tx_all, prices, targets_df_f, fcp_name, as_of_ts, divs_f
            )
            if not t.empty:
                t["FCP"] = fcp_name
                all_tracking_rows.append(t)

        if not all_tracking_rows:
            st.info("Aucune cible définie. Utilisez le panneau d'import ci-dessus.")
        else:
            gt = pd.concat(all_tracking_rows, ignore_index=True)

            # ── Netting par ticker ──────────────────────────────────────────
            agg = (
                gt.groupby("ticker")
                .agg(
                    qty_totale=("quantite_actuelle", "sum"),
                    valo_totale=("valo_actuelle", "sum"),
                    ecart_fcfa_net=("ecart_fcfa", "sum"),
                    qty_ecart_net=("quantite_ecart", "sum"),
                    cible_pct=("cible_pct", "first"),
                    cours=("cours", "first"),
                    n_fcps=("FCP", "nunique"),
                )
                .reset_index()
            )

            # Global totals for weights
            total_valo_global = float(agg["valo_totale"].sum())
            agg["poids_actuel_global"] = agg["valo_totale"] / total_valo_global \
                if total_valo_global else 0.0

            # Derive cible_fcfa at global level from cible_pct × total valo
            agg["cible_fcfa_global"] = agg.apply(
                lambda r: r["cible_pct"] * total_valo_global
                if pd.notna(r["cible_pct"]) else None,
                axis=1,
            )
            agg["ecart_pct_net"] = agg.apply(
                lambda r: r["cible_pct"] - r["poids_actuel_global"]
                if pd.notna(r["cible_pct"]) else None,
                axis=1,
            )

            # Determine net sens — threshold = half a share value
            def _net_sens(row) -> str:
                ef = row["ecart_fcfa_net"]
                if pd.isna(ef):
                    return "—"
                cours_v = row["cours"] or 1
                if abs(ef) < cours_v * 0.5:
                    return "OK"
                return "ACHAT" if ef > 0 else "VENTE"

            agg["sens_net"] = agg.apply(_net_sens, axis=1)

            # Detect inter-FCP opportunities: ticker has both ACHAT and VENTE
            # in different FCPs → internal cross could offset
            ticker_sens = gt.groupby("ticker")["sens"].apply(set)
            agg["inter_fcp"] = agg["ticker"].map(
                lambda t: "🔁" if {"ACHAT","VENTE"} <= ticker_sens.get(t, set())
                else ""
            )

            # Sort: VENTE first, then ACHAT, then OK
            _order = {"VENTE": 0, "ACHAT": 1, "OK": 2, "—": 3}
            agg = agg.sort_values(
                ["sens_net", "ecart_fcfa_net"],
                key=lambda col: col.map(_order) if col.name == "sens_net"
                                else col.abs(),
                ascending=[True, False],
            ).reset_index(drop=True)

            # ── KPIs ──────────────────────────────────────────────────────
            n_a  = int((agg["sens_net"] == "ACHAT").sum())
            n_v  = int((agg["sens_net"] == "VENTE").sum())
            n_ok = int((agg["sens_net"] == "OK").sum())
            n_if = int((agg["inter_fcp"] == "🔁").sum())
            tot_a = float(agg.loc[agg["sens_net"]=="ACHAT","ecart_fcfa_net"].sum())
            tot_v = float(agg.loc[agg["sens_net"]=="VENTE","ecart_fcfa_net"].sum())

            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("Lignes ACHAT net", n_a)
            c2.metric("Montant net à acheter", fmt_xof(tot_a))
            c3.metric("Lignes VENTE nette", n_v)
            c4.metric("Montant net à vendre", fmt_xof(abs(tot_v)))
            c5.metric("Lignes OK ✓", n_ok)
            c6.metric("🔁 Transactions inter-FCP", n_if)

            st.divider()

            # ── Tableau agrégé ─────────────────────────────────────────────
            display_g = pd.DataFrame({
                "": agg["sens_net"].map(
                    {"ACHAT":"🟢","VENTE":"🔴","OK":"✅","—":"⚪"}
                ),
                "🔁": agg["inter_fcp"],
                "Ticker":       agg["ticker"],
                "Qté totale":   agg["qty_totale"].map(
                    lambda v: f"{v:,.0f}".replace(",", " ") if v else "—"
                ),
                "Cours":        agg["cours"].map(
                    lambda v: fmt_xof(v) if pd.notna(v) and v else "—"
                ),
                "Valo totale":  agg["valo_totale"].map(fmt_xof),
                "Poids actuel": agg["poids_actuel_global"].map(fmt_pct),
                "Cible %":      agg["cible_pct"].map(
                    lambda v: fmt_pct(v) if pd.notna(v) else "—"
                ),
                "Cible FCFA":   agg["cible_fcfa_global"].map(
                    lambda v: fmt_xof(v) if pd.notna(v) else "—"
                ),
                "Écart net FCFA": agg["ecart_fcfa_net"].map(
                    lambda v: fmt_xof(v, signed=True) if pd.notna(v) else "—"
                ),
                "Écart net %":  agg["ecart_pct_net"].map(
                    lambda v: fmt_pct(v, signed=True) if pd.notna(v) else "—"
                ),
                "Qté écart nette": agg["qty_ecart_net"].map(
                    lambda v: f"{v:+,.0f}".replace(",", " ")
                    if pd.notna(v) else "—"
                ),
                "Nb FCPs": agg["n_fcps"].astype(str),
                "Sens net": agg["sens_net"],
            })
            render_table(display_g, height=600,
                         color_cols=["Écart net FCFA","Écart net %",
                                     "Qté écart nette"])

            # ── Transactions inter-FCP potentielles ────────────────────────
            inter_tickers = agg[agg["inter_fcp"] == "🔁"]["ticker"].tolist()
            if inter_tickers:
                st.divider()
                st.subheader("🔁 Transactions inter-FCP potentielles")
                st.caption(
                    "Ces titres ont des FCPs qui veulent acheter **et** d'autres "
                    "qui veulent vendre. Une transaction interne entre FCPs "
                    "pourrait solder tout ou partie de ces écarts sans passer "
                    "par le marché."
                )
                inter_rows = []
                for ticker in inter_tickers:
                    rows_t = gt[gt["ticker"] == ticker].copy()
                    rows_t = rows_t[rows_t["sens"].isin(["ACHAT","VENTE"])]
                    rows_t = rows_t.sort_values("sens")
                    cours_t = float(rows_t["cours"].iloc[0]) if not rows_t.empty else 0

                    buy_fcps  = rows_t[rows_t["sens"]=="ACHAT"]
                    sell_fcps = rows_t[rows_t["sens"]=="VENTE"]
                    qty_buy   = float(buy_fcps["quantite_ecart"].sum())
                    qty_sell  = abs(float(sell_fcps["quantite_ecart"].sum()))
                    qty_match = min(qty_buy, qty_sell)
                    valo_match = qty_match * cours_t

                    inter_rows.append({
                        "Ticker": ticker,
                        "FCPs acheteurs": ", ".join(buy_fcps["FCP"].tolist()),
                        "FCPs vendeurs":  ", ".join(sell_fcps["FCP"].tolist()),
                        "Qté nettable":   f"{qty_match:,.0f}".replace(",", " "),
                        "Montant nettable": fmt_xof(valo_match),
                        "Cours":          fmt_xof(cours_t),
                    })

                inter_df = pd.DataFrame(inter_rows)
                render_table(inter_df, height=None)

            # ── Exports ────────────────────────────────────────────────────
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                _xlsx_btn(
                    "📊 Exporter la vue nettée (.xlsx)", display_g,
                    f"suivi_cibles_net_{as_of_ts.date()}.xlsx",
                    key="dl_tracking_global", sheet_name="Cibles global",
                )
            with col_e2:
                if inter_tickers:
                    _xlsx_btn(
                        "📊 Exporter inter-FCP (.xlsx)", inter_df,
                        f"inter_fcp_{as_of_ts.date()}.xlsx",
                        key="dl_inter_fcp", sheet_name="Inter-FCP",
                    )

    # ── Vue par FCP ──────────────────────────────────────────────────────────
    else:
        fcp_scope = scope
        targets_df = db.get_targets(fcp_scope)
        divs_scope = _build_divs_by_fcp(fcps, as_of_ts).get(fcp_scope, {})

        if targets_df.empty:
            st.info(
                f"Aucune cible définie pour **{fcp_scope}**. "
                "Utilisez le panneau d'import ci-dessus."
            )
        else:
            tracking = compute_tracking(
                tx_all, prices, targets_df, fcp_scope, as_of_ts, divs_scope
            )

            if tracking.empty:
                st.info("Aucune position ni cible à afficher.")
            else:
                n_achat = int((tracking["sens"] == "ACHAT").sum())
                n_vente = int((tracking["sens"] == "VENTE").sum())
                n_ok    = int((tracking["sens"] == "OK").sum())
                total_a_acheter = tracking.loc[
                    tracking["sens"] == "ACHAT", "ecart_fcfa"
                ].sum()
                total_a_vendre = tracking.loc[
                    tracking["sens"] == "VENTE", "ecart_fcfa"
                ].sum()

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Lignes ACHAT", n_achat)
                c2.metric("Montant à acheter", fmt_xof(total_a_acheter))
                c3.metric("Lignes VENTE", n_vente)
                c4.metric("Montant à vendre", fmt_xof(abs(total_a_vendre)))
                c5.metric("Lignes OK ✓", n_ok)

                st.divider()

                display = pd.DataFrame({
                    "": tracking["sens"].map(
                        lambda s: {"ACHAT":"🟢","VENTE":"🔴","OK":"✅","—":"⚪"}.get(s,"")
                    ),
                    "Ticker":       tracking["ticker"],
                    "Qté actuelle": tracking["quantite_actuelle"].map(
                        lambda v: f"{v:,.0f}".replace(",", " ") if v else "—"
                    ),
                    "Cours":        tracking["cours"].map(
                        lambda v: fmt_xof(v) if pd.notna(v) and v else "—"
                    ),
                    "Valo actuelle":tracking["valo_actuelle"].map(fmt_xof),
                    "Poids actuel": tracking["poids_actuel"].map(fmt_pct),
                    "Cible %":      tracking["cible_pct"].map(
                        lambda v: fmt_pct(v) if pd.notna(v) and v is not None else "—"
                    ),
                    "Cible FCFA":   tracking["cible_fcfa"].map(
                        lambda v: fmt_xof(v) if pd.notna(v) and v is not None else "—"
                    ),
                    "Écart FCFA":   tracking["ecart_fcfa"].map(
                        lambda v: fmt_xof(v, signed=True) if pd.notna(v) and v is not None else "—"
                    ),
                    "Écart %":      tracking["ecart_pct"].map(
                        lambda v: fmt_pct(v, signed=True) if pd.notna(v) and v is not None else "—"
                    ),
                    "Qté écart":    tracking["quantite_ecart"].map(
                        lambda v: f"{v:+,.0f}".replace(",", " ") if pd.notna(v) and v is not None else "—"
                    ),
                    "Sens":         tracking["sens"],
                })
                render_table(display, height=600,
                             color_cols=["Écart FCFA","Écart %","Qté écart"])

                export_cols = [
                    "ticker","quantite_actuelle","cours","valo_actuelle",
                    "poids_actuel","cible_pct","cible_fcfa",
                    "ecart_fcfa","ecart_pct","quantite_ecart","sens",
                ]
                _xlsx_btn(
                    "📊 Exporter le suivi (.xlsx)", display,
                    f"suivi_cibles_{fcp_scope.replace(' ','_')}_{as_of_ts.date()}.xlsx",
                    key="dl_tracking", sheet_name=f"Cibles {fcp_scope[:20]}",
                )

                with st.expander(f"📋 Cibles enregistrées — {fcp_scope}"):
                    raw_disp = targets_df.copy()
                    raw_disp["weight_pct"] = raw_disp["weight_pct"].map(
                        lambda v: f"{v*100:.2f}%" if pd.notna(v) else "—"
                    )
                    raw_disp["amount_fcfa"] = raw_disp["amount_fcfa"].map(
                        lambda v: fmt_xof(v) if pd.notna(v) else "—"
                    )
                    render_table(
                        raw_disp[["ticker","weight_pct","amount_fcfa","updated_at"]],
                        height=None,
                    )
                    if st.button(
                        f"🗑️ Effacer toutes les cibles de {fcp_scope}",
                        key="clear_all_targets",
                    ):
                        db.replace_targets_for_fcp(
                            fcp_scope,
                            pd.DataFrame(columns=["ticker","weight_pct","amount_fcfa"]),
                        )
                        _clear_data_cache()
                        st.success("Toutes les cibles effacées.")
                        st.rerun()


# ---------------------------------------------------------------------------
# Page: Transactions
# ---------------------------------------------------------------------------
elif page == "💼 Transactions":
    st.header(f"Transactions — {fcp}")

    with st.expander("🔄 Synchronisation SharePoint CGF GESTION", expanded=False):
        st.caption(
            "Source : `PTF ACTIONS CGF GESTION v2.xlsm` (boursecgf.sharepoint.com). "
            "Le bouton ci-dessous télécharge le fichier et écrase les transactions de "
            "l'application avec la feuille **Transactions** (colonnes A à K). "
            "Les cours du jour ne sont **pas** touchés ici — utilisez l'onglet "
            "🌐 Cours BRVM pour cela."
        )

        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            sync_clicked = st.button("🔄 Synchroniser", type="primary", key="sp_sync")
        with col_status:
            st.caption("⚠️ Cette action remplace toutes les transactions stockées.")

        if sync_clicked:
            try:
                from sharepoint_sync import (
                    DEFAULT_URL,
                    download_workbook,
                    read_transactions_from_bytes,
                )
                with st.spinner("Téléchargement depuis SharePoint…"):
                    xlsm = download_workbook(DEFAULT_URL, timeout=60)
                st.success(f"✅ Fichier téléchargé ({len(xlsm)/1024/1024:.1f} Mo)")

                with st.spinner("Lecture de la feuille Transactions…"):
                    new_tx = read_transactions_from_bytes(xlsm)
                st.success(f"✅ {len(new_tx):,} transactions lues".replace(",", " "))

                with st.spinner("Écriture en base…"):
                    n = db.replace_transactions(new_tx)
                st.success(f"✅ {n:,} transactions chargées en base".replace(",", " "))

                st.balloons()
                _clear_data_cache()
                st.rerun()

            except Exception as e:
                st.error(f"❌ Échec de la synchronisation : {e}")
                st.info(
                    "Vérifiez que le lien SharePoint anonyme est toujours actif. "
                    "En cas de blocage, utilisez l'import manuel ci-dessous."
                )

        st.divider()
        with st.expander("📥 Import manuel (.xlsm) — fallback"):
            up = st.file_uploader(
                "Téléverser le fichier `PTF ACTIONS CGF GESTION v2.xlsm`",
                type=["xlsm", "xlsx"],
                key="sp_manual_upload",
            )
            if up is not None:
                try:
                    from sharepoint_sync import read_transactions_from_bytes
                    xlsm = up.getvalue()
                    new_tx = read_transactions_from_bytes(xlsm)
                    st.write(
                        f"**{len(new_tx):,}** transactions détectées".replace(",", " ")
                    )
                    if st.button("Confirmer l'import", key="sp_manual_confirm"):
                        db.replace_transactions(new_tx)
                        st.success("Import réussi.")
                        _clear_data_cache()
                        st.rerun()
                except Exception as e:
                    st.error(f"Lecture impossible : {e}")

    st.divider()

    with st.expander("➕ Nouvelle transaction", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            t_date = st.date_input("Date", value=date.today(), key="tx_date")
            t_sens = st.selectbox("Sens", ["ACHAT", "VENTE"], key="tx_sens")
        with col2:
            known_tickers = _cached_known_tickers()
            ticker_input = st.selectbox(
                "Titre (ou tapez)",
                options=[""] + known_tickers,
                index=0,
                key="tx_ticker_select",
            )
            ticker_manual = st.text_input(
                "...ou nouveau symbole",
                value="",
                key="tx_ticker_manual",
            ).strip().upper()
            t_ticker = ticker_manual or ticker_input
        with col3:
            t_qte = st.number_input("Quantité", min_value=0.0, step=1.0, key="tx_qte")
            t_prix = st.number_input("Prix unitaire (FCFA)", min_value=0.0, step=1.0, key="tx_prix")
            t_frais = st.number_input("Frais (FCFA)", min_value=0.0, step=1.0,
                                       key="tx_frais", value=0.0)

        valeur_preview = t_qte * t_prix + (t_frais if t_sens == "ACHAT" else -t_frais)
        st.caption(f"Valeur calculée : **{fmt_xof(valeur_preview)}**")

        if st.button("💾 Enregistrer", type="primary"):
            if not t_ticker:
                st.error("Symbole requis.")
            elif t_qte <= 0 or t_prix <= 0:
                st.error("Quantité et prix doivent être > 0.")
            else:
                tx_id = db.add_transaction(
                    str(t_date), fcp, t_ticker, t_sens, t_qte, t_prix, t_frais
                )
                st.success(f"Transaction #{tx_id} enregistrée.")
                _clear_data_cache()
                st.rerun()

    st.divider()
    st.subheader("Historique")
    tx = db.get_transactions(fcp)
    if tx.empty:
        st.info("Aucune transaction pour ce FCP.")
    else:
        col_l, col_r = st.columns([3, 1])
        col_l.metric("Total transactions", f"{len(tx)}")
        with col_r:
            _xlsx_btn(
                "📊 Exporter (.xlsx)", tx,
                f"transactions_{fcp.replace(' ', '_')}.xlsx",
                key="dl_tx", sheet_name="Transactions",
            )

        tx_display = tx[["id","date","fcp","ticker","sens",
                          "quantite","prix","valeur","frais",
                          "cost_in","cost_out"]].copy()
        tx_display = tx_display.rename(columns={
            "id":       "ID",
            "date":     "Date",
            "fcp":      "FCP",
            "ticker":   "Ticker",
            "sens":     "Sens",
            "quantite": "Quantité",
            "prix":     "Prix",
            "valeur":   "Valeur",
            "frais":    "Frais",
            "cost_in":  "Coût achat",
            "cost_out": "Coût cession",
        })
        for col in ["Prix","Valeur","Frais","Coût achat","Coût cession"]:
            tx_display[col] = pd.to_numeric(
                tx_display[col], errors="coerce"
            ).map(lambda v: "-" if pd.isna(v) or v == 0
                  else f"{v:,.0f}".replace(",", " "))
        tx_display["Quantité"] = pd.to_numeric(
            tx_display["Quantité"], errors="coerce"
        ).map(lambda v: f"{v:,.0f}".replace(",", " ") if pd.notna(v) else "-")
        render_table(tx_display, height=500)

        with st.expander("🗑️ Supprimer une transaction"):
            tx_id_del = st.number_input("ID à supprimer", min_value=0, step=1, key="tx_del_id")
            if st.button("Supprimer"):
                if tx_id_del > 0:
                    db.delete_transaction(int(tx_id_del))
                    st.success(f"Transaction #{tx_id_del} supprimée.")
                _clear_data_cache()
                st.rerun()


# ---------------------------------------------------------------------------
# Page: BRVM live quotes
# ---------------------------------------------------------------------------
elif page == "🌐 Cours BRVM":
    st.header("Cours BRVM — Mise à jour automatique")

    # ── Helpers de timing ────────────────────────────────────────────────────
    import datetime as _dt

    # Fuseau Dakar = UTC+0 (pas de décalage, même heure que UTC)
    MARKET_START = _dt.time(9, 50)   # premier refresh à 9h50
    MARKET_END   = _dt.time(17, 30)  # clôture BRVM
    INTERVAL_MIN = 15                 # minutes entre refreshs

    def _next_slot_seconds() -> int | None:
        """Calcule les secondes jusqu'au prochain slot de 15 min aligné sur 9h50.

        Retourne None si on est en dehors des heures de marché.
        Les slots sont : 9h50, 10h05, 10h20, ..., 17h20, 17h35 (dernier avant 17h30).
        """
        now = _dt.datetime.utcnow()  # Dakar = UTC+0
        t   = now.time()

        if t < MARKET_START:
            # Pas encore ouvert → prochain slot = 9h50 aujourd'hui
            target = _dt.datetime.combine(now.date(), MARKET_START)
            return max(1, int((target - now).total_seconds()))

        if t >= MARKET_END:
            # Marché fermé → pas de refresh automatique
            return None

        # Dans les heures de marché → calculer le prochain slot
        # Nombre de minutes depuis l'ouverture (9h50)
        open_dt = _dt.datetime.combine(now.date(), MARKET_START)
        elapsed_min = (now - open_dt).total_seconds() / 60
        slots_passed = int(elapsed_min // INTERVAL_MIN)
        next_slot_dt = open_dt + _dt.timedelta(minutes=(slots_passed + 1) * INTERVAL_MIN)

        # Si le prochain slot dépasse la fermeture, stop
        if next_slot_dt.time() > MARKET_END:
            return None

        secs = max(1, int((next_slot_dt - now).total_seconds()))
        return secs

    def _is_market_open() -> bool:
        t = _dt.datetime.utcnow().time()
        return MARKET_START <= t < MARKET_END

    # ── Fonction de refresh BRVM ──────────────────────────────────────────────
    def _do_brvm_refresh() -> tuple[int, str, object] | None:
        """Scrape et enregistre les cours. Retourne (n_cours, source, session_date)."""
        from scraper import fetch_with_session_date
        quotes_df, sess = fetch_with_session_date(timeout=25)
        for _, row in quotes_df.iterrows():
            db.upsert_quote_today(row.to_dict())
        close_rows = quotes_df[["ticker", "close"]].dropna().copy()
        close_rows["date"] = sess.isoformat()
        close_rows = close_rows.rename(columns={"close": "price"})[["date", "ticker", "price"]]
        n = db.upsert_prices(close_rows, source="brvm")
        _clear_data_cache()
        src = quotes_df.get("source_url", pd.Series(["?"])).iloc[0] if len(quotes_df) else "?"
        return len(quotes_df), str(src), sess

    # ── Fragment auto-rafraîchissant ──────────────────────────────────────────
    next_secs = _next_slot_seconds()
    market_open = _is_market_open()

    # run_every = secondes jusqu'au prochain slot si marché ouvert, sinon None
    @st.fragment(run_every=next_secs)
    def _brvm_fragment():
        now_utc = _dt.datetime.utcnow()
        t = now_utc.time()
        is_open = MARKET_START <= t < MARKET_END

        # Statut du marché
        if is_open:
            secs = _next_slot_seconds()
            if secs is not None:
                mins, sec = divmod(secs, 60)
                st.info(
                    f"🟢 Marché ouvert — prochain rafraîchissement automatique dans "
                    f"**{mins}m {sec:02d}s** "
                    f"(slots : 9h50, 10h05, 10h20… à intervalles de 15 min)"
                )
            else:
                st.info("🟢 Marché ouvert — dernier slot de la journée atteint.")
        else:
            if t < MARKET_START:
                st.warning(
                    f"⏳ Marché pas encore ouvert — premier refresh automatique à **9h50**."
                )
            else:
                st.warning(
                    f"🔴 Marché fermé (après 17h30) — pas de refresh automatique jusqu'à demain 9h50."
                )

        last_refresh = db.get_last_brvm_refresh()
        if last_refresh:
            st.caption(f"📅 Dernier rafraîchissement enregistré : **{last_refresh}**")

        col1, col2 = st.columns([1, 3])
        with col1:
            do_fetch = st.button("🔄 Rafraîchir maintenant", type="primary",
                                 key="brvm_manual_refresh")
        with col2:
            st.caption(
                "Le bouton force un rafraîchissement immédiat. "
                "En heures de marché, le refresh automatique toutes les 15 min "
                "est actif même sans interaction."
            )

        # Refresh automatique en heures de marché (déclenché par run_every)
        # ou manuel
        should_refresh = do_fetch
        if is_open and not do_fetch:
            # Fragment rerun automatique → refresh les cours
            should_refresh = True

        if should_refresh and not do_fetch:
            # Auto-refresh silencieux (pas de spinner visible)
            try:
                n_q, src, sess = _do_brvm_refresh()
                st.caption(
                    f"🔄 Auto-refresh {now_utc.strftime('%H:%M:%S')} UTC — "
                    f"{n_q} cours ({src}, séance {sess.strftime('%d/%m/%Y')})"
                )
            except Exception as e:
                st.caption(f"⚠️ Auto-refresh échoué : {e}")

        elif do_fetch:
            try:
                with st.spinner("Récupération des cours…"):
                    n_q, src, sess = _do_brvm_refresh()
                st.success(
                    f"✅ {n_q} cours récupérés depuis **{src}** "
                    f"(séance du {sess.strftime('%d/%m/%Y')})."
                )
            except Exception as e:
                st.error(f"Échec du rafraîchissement : {e}")

        st.divider()

        quotes = db.get_quotes_today()
        if quotes.empty:
            st.info("Pas encore de cours en mémoire — cliquez sur Rafraîchir.")
        else:
            st.subheader(f"Snapshot ({len(quotes)} titres)")
            if "fetched_at" in quotes.columns and not quotes["fetched_at"].isna().all():
                st.caption(f"Capturé à : {quotes['fetched_at'].max()}")
            display = quotes.copy()
            for c in ["volume", "prev_close", "open", "close"]:
                if c in display.columns:
                    display[c] = display[c].map(
                        lambda v: f"{v:,.0f}".replace(",", " ") if pd.notna(v) else "-"
                    )
            if "variation_pct" in display.columns:
                display["variation_pct"] = display["variation_pct"].map(
                    lambda v: f"{v:+.2f}%" if pd.notna(v) else "-"
                )
            render_table(
                display.drop(columns=["fetched_at"], errors="ignore"),
                height=None,
            )

    _brvm_fragment()

    st.divider()
    with st.expander("📥 Import manuel CSV (fallback si scraping bloqué)"):
        st.caption("Format attendu : ticker,name,volume,prev_close,open,close,variation_pct")
        f = st.file_uploader("Fichier CSV", type=["csv"], key="quotes_csv")
        if f is not None:
            df_csv = pd.read_csv(f)
            session_d = st.date_input("Date de séance", value=date.today(),
                                      key="manual_sess")
            if st.button("Importer ce CSV"):
                for _, r in df_csv.iterrows():
                    db.upsert_quote_today(r.to_dict())
                close_rows = df_csv[["ticker", "close"]].dropna().copy()
                close_rows["date"] = session_d.isoformat()
                close_rows = close_rows.rename(columns={"close": "price"})
                close_rows = close_rows[["date", "ticker", "price"]]
                n = db.upsert_prices(close_rows, source="manual_csv")
                st.success(f"Importé : {len(df_csv)} cours, {n} archivés.")
                _clear_data_cache()
                st.rerun()


# ---------------------------------------------------------------------------
# Page: Price history
# ---------------------------------------------------------------------------
elif page == "📚 Historique cours":
    st.header("Historique des cours")

    with st.expander("🔄 Synchronisation SharePoint CGF GESTION", expanded=False):
        st.caption(
            "Source : `PTF ACTIONS CGF GESTION v2.xlsm` (boursecgf.sharepoint.com). "
            "Le bouton ci-dessous télécharge le fichier et écrase **tout l'historique** "
            "des cours en base avec la feuille **Cours** (colonnes A à BJ). "
            "Les transactions ne sont **pas** touchées ici."
        )

        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            sync_clicked = st.button("🔄 Synchroniser", type="primary", key="sp_sync_cours")
        with col_status:
            st.caption(
                "⚠️ Cette action remplace tout l'historique des cours stocké, "
                "y compris les rafraîchissements BRVM récents."
            )

        if sync_clicked:
            try:
                from sharepoint_sync import (
                    DEFAULT_URL,
                    download_workbook,
                    read_full_cours_history,
                )
                with st.spinner("Téléchargement depuis SharePoint…"):
                    xlsm = download_workbook(DEFAULT_URL, timeout=60)
                st.success(f"✅ Fichier téléchargé ({len(xlsm)/1024/1024:.1f} Mo)")

                with st.spinner("Lecture de la feuille Cours (A:BJ)…"):
                    new_prices = read_full_cours_history(xlsm)
                st.success(
                    f"✅ {len(new_prices):,} cours lus "
                    f"({new_prices['ticker'].nunique()} titres, "
                    f"de {new_prices['date'].min()} à {new_prices['date'].max()})"
                    .replace(",", " ")
                )

                with st.spinner("Écriture en base…"):
                    n = db.replace_prices_history(new_prices, source="sharepoint")
                st.success(f"✅ {n:,} cours chargés en base".replace(",", " "))

                st.balloons()
                _clear_data_cache()
                st.rerun()

            except Exception as e:
                st.error(f"❌ Échec de la synchronisation : {e}")
                st.info(
                    "Vérifiez que le lien SharePoint anonyme est toujours actif. "
                    "En cas de blocage, utilisez l'import manuel ci-dessous."
                )

        st.divider()
        with st.expander("📥 Import manuel (.xlsm) — fallback"):
            up = st.file_uploader(
                "Téléverser le fichier `PTF ACTIONS CGF GESTION v2.xlsm`",
                type=["xlsm", "xlsx"],
                key="sp_manual_upload_cours",
            )
            if up is not None:
                try:
                    from sharepoint_sync import read_full_cours_history
                    xlsm = up.getvalue()
                    new_prices = read_full_cours_history(xlsm)
                    st.write(
                        f"**{len(new_prices):,}** cours détectés "
                        f"({new_prices['ticker'].nunique()} titres, "
                        f"de {new_prices['date'].min()} à {new_prices['date'].max()})"
                        .replace(",", " ")
                    )
                    if st.button("Confirmer l'import", key="sp_manual_confirm_cours"):
                        db.replace_prices_history(new_prices, source="sharepoint")
                        st.success("Import réussi.")
                        _clear_data_cache()
                        st.rerun()
                except Exception as e:
                    st.error(f"Lecture impossible : {e}")

    st.divider()

    prices = _cached_prices()
    if prices.empty:
        st.info("Base vide.")
    else:
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"])
        tickers = sorted(prices["ticker"].unique())
        sel = st.multiselect(
            "Sélectionnez 1 à 5 titres", tickers,
            default=tickers[:1], max_selections=5,
        )
        if sel:
            sub = prices[prices["ticker"].isin(sel)]
            pivot = sub.pivot_table(index="date", columns="ticker", values="price")
            st.line_chart(pivot)

            with st.expander("Voir les données brutes"):
                render_table(sub.sort_values(["ticker", "date"]), height=None)


# ---------------------------------------------------------------------------
# Page: Settings
# ---------------------------------------------------------------------------
elif page == "⚙️ Paramètres":
    st.header("Paramètres")

    st.subheader("État de la base")
    tx = _cached_transactions()
    prices = _cached_prices()
    quotes = db.get_quotes_today()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FCPs", len(_cached_fcps()))
    c2.metric("Transactions", f"{len(tx):,}".replace(",", " "))
    c3.metric("Cours archivés", f"{len(prices):,}".replace(",", " "))
    c4.metric("Cours du jour", f"{len(quotes)}")

    st.divider()
    st.subheader("Dividendes")
    st.caption(
        "Les dividendes sont stockés avec leur date de paiement. "
        "Le montant est ajouté au cours de clôture **uniquement** à la date de paiement. "
        "Les jours suivants, le cours est utilisé sans dividende."
    )

    all_tickers_held = sorted(set(tx["ticker"].dropna().tolist()))
    all_divs_dated = _cached_dividends_dated()

    # ── Saisie d'un nouveau dividende ────────────────────────────────────────
    with st.expander("➕ Saisir un dividende", expanded=True):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1:
            div_ticker = st.selectbox(
                "Titre", options=all_tickers_held or [""],
                key="div_ticker"
            )
        with c2:
            div_amount = st.number_input(
                "Montant (FCFA / action)",
                min_value=0.0, step=1.0, value=0.0,
                key="div_amount"
            )
        with c3:
            div_date = st.date_input(
                "Date de paiement",
                value=date.today(),
                key="div_date"
            )
        with c4:
            st.write("")
            st.write("")
            if st.button("💾 Enregistrer", key="div_save", type="primary"):
                if div_amount > 0 and div_ticker:
                    db.add_dividend_dated(
                        ticker=div_ticker,
                        amount=float(div_amount),
                        payment_date=str(div_date),
                    )
                    _clear_data_cache()
                    st.success(
                        f"✅ Dividende {div_ticker} : "
                        f"{div_amount:,.0f} FCFA/action le {div_date.strftime('%d/%m/%Y')}"
                    )
                    st.rerun()
                else:
                    st.warning("Montant doit être > 0 et un titre doit être sélectionné.")

    # ── Liste des dividendes enregistrés ─────────────────────────────────────
    if all_divs_dated:
        st.write(f"**{len(all_divs_dated)} dividende(s) enregistré(s) :**")
        divs_df = pd.DataFrame(all_divs_dated)
        divs_df["payment_date"] = pd.to_datetime(divs_df["payment_date"]).dt.strftime("%d/%m/%Y")
        divs_df["amount"] = divs_df["amount"].map(lambda v: f"{v:,.0f} FCFA".replace(",", " "))
        render_table(
            divs_df[["id", "ticker", "amount", "payment_date"]].rename(columns={
                "id": "ID", "ticker": "Titre",
                "amount": "Montant / action",
                "payment_date": "Date paiement",
            }),
            height=300,
        )

        # Suppression par ID
        col_del1, col_del2 = st.columns(2)
        with col_del1:
            del_id = st.number_input(
                "Supprimer par ID", min_value=0, step=1,
                key="div_del_id"
            )
            if st.button("🗑️ Supprimer ce dividende", key="div_del_one"):
                if del_id > 0:
                    db.delete_dividend_dated(int(del_id))
                    _clear_data_cache()
                    st.success(f"Dividende #{del_id} supprimé.")
                    st.rerun()
        with col_del2:
            st.write("")
            st.write("")
            if st.button(
                "🗑️ Effacer tous les dividendes",
                key="div_del_all",
                type="secondary",
            ):
                db.clear_all_dividends_dated()
                _clear_data_cache()
                st.success("Tous les dividendes supprimés.")
                st.rerun()
    else:
        st.info("Aucun dividende enregistré.")

    st.divider()
    st.subheader("Maintenance")
    st.info(
        "💡 Les données sont stockées dans Postgres (Supabase) et persistent "
        "automatiquement. Pour ré-initialiser depuis zéro, utilisez les boutons "
        "de synchronisation SharePoint dans les onglets **Transactions** et "
        "**Historique cours**, qui écrasent les tables avec la source officielle."
    )
