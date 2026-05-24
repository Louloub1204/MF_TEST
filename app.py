import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import json
import os
import warnings
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Assure que le dossier du script est dans sys.path
# (nécessaire sur Streamlit Cloud où le CWD peut différer)
import sys, os
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

try:
    from charia_screening import (parse_screening_file as parse_charia,
                                   screen_all_from_fin_data,
                                   get_charia_label,
                                   get_charia_compatible_tickers)
    CHARIA_AVAILABLE = True
except ImportError:
    CHARIA_AVAILABLE = False

try:
    from fs_parser import (parse_financial_file, merge_financial_data,
                           save_financial_db, load_financial_db,
                           validate_and_fix_units)
    from valuation_models import (valuation_pe, valuation_pb, valuation_ddm,
                                   valuation_dcf, combined_price, compute_beta,
                                   calibrate_params)
    VALUATION_AVAILABLE = True
except ImportError as _e:
    VALUATION_AVAILABLE = False
    _VALUATION_ERROR = str(_e)
warnings.filterwarnings("ignore")

st.set_page_config(page_title="CGF Gestion · SMF BRVM", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
:root{--bg:#0b0f1a;--surface:#111827;--card:#161d2e;--border:#1e2d45;
      --accent:#3b82f6;--accent2:#06b6d4;--gold:#f59e0b;--green:#10b981;
      --red:#ef4444;--text:#e2e8f0;--muted:#64748b;}
html,body,[class*="css"]{background:var(--bg);color:var(--text);font-family:'JetBrains Mono',monospace;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
h1,h2,h3{font-family:'Syne',sans-serif!important;}
[data-testid="metric-container"]{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;}
[data-testid="metric-container"] label{color:var(--muted)!important;font-size:11px!important;text-transform:uppercase;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:var(--accent)!important;font-family:'JetBrains Mono'!important;}
.stButton>button{background:var(--accent)!important;color:white!important;border:none!important;border-radius:6px!important;transition:all .2s;}
.stButton>button:hover{background:var(--accent2)!important;transform:translateY(-1px);}
.stTabs [data-baseweb="tab-list"]{background:var(--surface);border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;font-family:'JetBrains Mono'!important;font-size:13px;padding:10px 18px;}
.stTabs [aria-selected="true"]{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;}
.stInfo{background:rgba(59,130,246,.1)!important;border-left:3px solid var(--accent)!important;}
.stSuccess{background:rgba(16,185,129,.1)!important;border-left:3px solid var(--green)!important;}
.stWarning{background:rgba(245,158,11,.1)!important;border-left:3px solid var(--gold)!important;}
hr{border-color:var(--border)!important;}
.pill{display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;
      letter-spacing:.08em;text-transform:uppercase;background:rgba(59,130,246,.15);
      color:var(--accent);border:1px solid rgba(59,130,246,.3);margin-bottom:8px;}
.sh{font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:#e2e8f0;margin:0 0 4px 0;}
.ss{font-family:'JetBrains Mono',monospace;font-size:12px;color:#64748b;margin-bottom:20px;}
.fbox{background:#0d1526;border:1px solid #1e3a5f;border-radius:8px;padding:14px 18px;
      margin:10px 0;font-family:'JetBrains Mono',monospace;font-size:13px;color:#93c5fd;}
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────
BENCH_COLS = ['Unnamed: 0','ANNEE','JOUR','Unnamed: 62','.BRVMCI','BRVM30',
              'BRVM PREST','BRVM-PRINC','BRVM-C TR','BRVM-CB','BRVM-CD',
              'BRVM-ENER','BRVM-SFIN','BRVM-SPUB']

# ─── DATA LOADING ─────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_bytes):
    xl = pd.ExcelFile(io.BytesIO(uploaded_bytes))

    # COURS
    cours_raw = pd.read_excel(xl, sheet_name="Cours")
    cours = cours_raw.drop(columns=[c for c in BENCH_COLS if c in cours_raw.columns], errors='ignore')
    cours['Date'] = pd.to_datetime(cours['Date'], errors='coerce')
    cours = cours.dropna(subset=['Date']).set_index('Date').sort_index()
    tickers = [c for c in cours.columns]

    # VOLUMES
    vol_raw = pd.read_excel(xl, sheet_name="Volume moyen")
    vol_raw['Date'] = pd.to_datetime(vol_raw['Date'], errors='coerce')
    vol = vol_raw.dropna(subset=['Date']).set_index('Date').sort_index()
    # Keep only ticker columns that are numeric (exclude formula/text columns)
    vol_ticker_cols = [c for c in tickers if c in vol.columns]
    vol = vol[vol_ticker_cols].apply(pd.to_numeric, errors='coerce')

    # DIVIDENDES
    div_raw = pd.read_excel(xl, sheet_name="Historique_dividende")
    div = div_raw[['Date'] + [c for c in tickers if c in div_raw.columns]].copy()
    div = div.dropna(subset=['Date'])
    div['Date'] = div['Date'].astype(int)

    # MOYENNE COURS
    moy_raw = pd.read_excel(xl, sheet_name="Moyenne_cours")
    moy = moy_raw[['Date'] + [c for c in tickers if c in moy_raw.columns]].copy()
    moy = moy.dropna(subset=['Date'])
    moy['Date'] = moy['Date'].astype(int)

    # NB TITRES
    # Structure réelle (header=None) :
    #   Ligne 0 : [NaN, 'Date', 'CABC', 'FTSC', ...]
    #   Ligne 1 : [NaN, 'Titres flottants', val, val, ...]
    #   Ligne 2 : [NaN, 'Nombre de titres', val, val, ...]
    nb_raw = pd.read_excel(xl, sheet_name="Nb_titres", header=None)
    nb_titres, nb_flottant = {}, {}
    nb_ticker_col = {}
    for col_idx in range(2, nb_raw.shape[1]):
        val = nb_raw.iloc[0, col_idx]
        if pd.notna(val):
            nb_ticker_col[str(val).strip()] = col_idx
    for row_idx in range(1, nb_raw.shape[0]):
        label = str(nb_raw.iloc[row_idx, 1]).strip().lower()
        for t, col_idx in nb_ticker_col.items():
            v = nb_raw.iloc[row_idx, col_idx]
            try:
                v = float(v)
                if not np.isnan(v) and v > 0:
                    if 'flottant' in label:
                        nb_flottant[t] = v
                    elif 'nombre' in label:
                        nb_titres[t] = v
            except (TypeError, ValueError):
                pass

    # TABLEAU DE BORD (fondamentaux)
    # Structure réelle (header=None) :
    #   Ligne 0 : [NaN, 1000000, 'CABC', 'FTSC', ...]  ← tickers col 2+
    #   Col 1   : années (2019..2024) ou NaN
    #   Col 2   : label métrique ('CAPITAUX PROPRES') ou valeur numérique
    #   Lignes label : col1=NaN, col2='CAPITAUX PROPRES'
    #   Lignes data  : col1=année, col2+=valeurs par ticker
    tb_raw = pd.read_excel(xl, sheet_name="Tableau de bord", header=None)

    ticker_col = {}
    for col_idx in range(2, tb_raw.shape[1]):
        val = tb_raw.iloc[0, col_idx]
        if pd.notna(val) and str(val).strip() not in ["nan", ""]:
            ticker_col[str(val).strip()] = col_idx

    def extract_metric(label_keyword):
        result = {}
        in_block = False
        for row_idx in range(1, tb_raw.shape[0]):
            cell2 = str(tb_raw.iloc[row_idx, 2]).strip().upper()
            cell1 = tb_raw.iloc[row_idx, 1]

            # Détection du label de la métrique
            if label_keyword.upper() in cell2 and pd.isna(cell1):
                in_block = True
                continue

            if in_block:
                try:
                    year = int(float(cell1)) if pd.notna(cell1) else None
                except (ValueError, TypeError):
                    year = None

                if year is None:
                    # Ligne de tickers répétée (ex: row avec 'CABC','FTSC'...) → ignorer
                    if cell2 in ticker_col:
                        continue
                    # Nouveau label métrique → fin du bloc
                    if pd.isna(cell1) and cell2 not in ["NAN", ""]:
                        break
                    continue

                # Lire les valeurs pour chaque ticker (col 2+ de tb_raw)
                for t, col_idx in ticker_col.items():
                    if col_idx < tb_raw.shape[1]:
                        v = tb_raw.iloc[row_idx, col_idx]
                        try:
                            v = float(v)
                            if not np.isnan(v):
                                result.setdefault(t, {})[year] = v
                        except (TypeError, ValueError):
                            pass
        return result

    # Années disponibles dans les fondamentaux (exclut valeurs parasites > 9999)
    fundamental_years = sorted(
        {y for metric in [
            extract_metric("CAPITAUX PROPRES"),
            extract_metric("RESULTAT NET"),
        ] for t_data in metric.values() for y in t_data.keys()
         if isinstance(y, int) and 1990 <= y <= 2100},
        reverse=True
    )

    return {
        "cours": cours, "volumes": vol, "dividendes": div,
        "moyenne_cours": moy, "nb_titres": nb_titres, "nb_flottant": nb_flottant,
        "tickers": tickers,
        "fundamental_years": fundamental_years,
        "capitaux_propres": extract_metric("CAPITAUX PROPRES"),
        "resultat_net":     extract_metric("RESULTAT NET"),
        "flux_treso":       extract_metric("FLUX DE TRESORERIE"),
        "chiffre_affaires": extract_metric("CHIFFRE D'AFFAIRES"),
        "resultat_expl":    extract_metric("RESULTAT D'EXPLOITATION"),
    }

# ─── NORMALISATION HELPER ─────────────────────────────────────
# Note technique : F_i(t,T) = Σ w_ij * m_ij(t,T) / max_{E_t}(m_ij(t,T))
# La division est inconditionnelle — max peut être négatif (ex: rendements tous négatifs).
# Seul cas exclu : max == 0 exactement (division par zéro).
# Volatilité (cas inversé) : F_i = Σ w_ij * min_{E_t}(m_ij) / m_ij(t,T)

def _normalise_standard(col: pd.Series) -> pd.Series:
    """m_ij / max_{E_t}(m_ij)  — formule générale de la note technique."""
    col = col.replace([np.inf, -np.inf], np.nan)
    mx  = col.max()
    if pd.isna(mx) or mx == 0:
        return pd.Series(np.nan, index=col.index)
    return col / mx

def _normalise_volatility(col: pd.Series) -> pd.Series:
    """min_{E_t}(m_ij) / m_ij  — formule inversée pour la volatilité."""
    col = col.replace([np.inf, -np.inf], np.nan)
    mn  = col.min()
    if pd.isna(mn):
        return pd.Series(np.nan, index=col.index)
    # avoid division by zero for any individual m_ij == 0
    return (mn / col.replace(0, np.nan))

# ─── FACTOR FUNCTIONS ─────────────────────────────────────────

def compute_value_factor(data, year, weights, date_start=None, date_end=None):
    """
    F_value(t,T) = Σ w_j * m_j(t,T) / max_{E_t}(m_j(t,T))

    Métriques (toutes exprimées par action, divisées par le cours moyen) :

      1. B/P  = (Capitaux propres / Nb titres) / Cours moyen      [Book-to-Price]
                 → plus élevé = plus sous-évalué  ✓ déjà dans le bon sens

      2. E/P  = (Résultat net / Nb titres) / Cours moyen          [Earnings-to-Price]
                 → inverse de P/E : score élevé = titre bon marché ✓

      3. FCF/P  = (Flux nets trésorerie / Nb titres) / Cours moyen [FCF yield]

      4. CA/P   = (Chiffre d'affaires / Nb titres) / Cours moyen   [Sales yield]

      5. EBIT/P = (Résultat d'exploitation / Nb titres) / Cours moyen
                  → proxy de EV/EBIT inversé, rapporté au cours unitaire

    Le cours moyen est calculé sur la plage [date_start, date_end] à partir
    des cours journaliers (feuille Cours), ou sur l'année calendaire si non fournie.
    """
    nb = data["nb_titres"]

    # ── 1. Cours moyen sur la plage de dates choisie ──────────────
    cours = data["cours"].apply(pd.to_numeric, errors="coerce")
    if date_start is not None and date_end is not None:
        mask = (cours.index >= pd.to_datetime(date_start)) & \
               (cours.index <= pd.to_datetime(date_end))
        cours_period = cours.loc[mask]
        period_label = f"{date_start} → {date_end}"
    else:
        # fallback : année calendaire
        cours_period = cours[cours.index.year == year]
        period_label = str(year)

    if cours_period.empty:
        return None

    cours_moy = cours_period.mean(numeric_only=True)  # Series : ticker → cours moyen

    # ── 2. Fondamentaux annuels (année sélectionnée) ───────────────
    detail = {}
    for t in data["tickers"]:
        if t not in nb or nb[t] <= 0:
            continue
        cm = cours_moy.get(t, np.nan)
        if pd.isna(cm) or cm <= 0:
            continue

        n   = nb[t]                                               # nombre de titres
        cp  = data["capitaux_propres"].get(t, {}).get(year, np.nan)
        rn  = data["resultat_net"].get(t, {}).get(year, np.nan)
        fcf = data["flux_treso"].get(t, {}).get(year, np.nan)
        ca  = data["chiffre_affaires"].get(t, {}).get(year, np.nan)
        rex = data["resultat_expl"].get(t, {}).get(year, np.nan)

        # Ratios par action / cours moyen
        bp_val   = (cp  / n) / cm if pd.notna(cp)  else np.nan   # B/P  = (CP/n) / cours
        ep_val   = (rn  / n) / cm if pd.notna(rn)  else np.nan   # E/P  = (RN/n) / cours
        fcfp     = (fcf / n) / cm if pd.notna(fcf) else np.nan   # FCF/P
        cap_val  = (ca  / n) / cm if pd.notna(ca)  else np.nan   # CA/P
        ebitp    = (rex / n) / cm if pd.notna(rex) else np.nan   # EBIT/P

        detail[t] = {
            "B/P  (CP/n÷cours)":    bp_val,
            "E/P  (RN/n÷cours)":    ep_val,
            "FCF/P (FCF/n÷cours)":  fcfp,
            "CA/P  (CA/n÷cours)":   cap_val,
            "EBIT/P (Rex/n÷cours)": ebitp,
            "Cours moyen":          cm,
            "Nb titres":            n,
        }

    if not detail:
        return None

    df = pd.DataFrame(detail).T
    # Force float — DataFrame.T depuis un dict mixte donne dtype=object
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    metrics = ["B/P  (CP/n÷cours)", "E/P  (RN/n÷cours)",
               "FCF/P (FCF/n÷cours)", "CA/P  (CA/n÷cours)",
               "EBIT/P (Rex/n÷cours)"]

    # Garder uniquement les titres qui ont AU MOINS une métrique valide
    has_any = df[metrics].notna().any(axis=1)
    df = df[has_any]
    if df.empty:
        return None

    # ── 3. Normalisation : m_ij / max_{E_t}(m_ij) ─────────────────
    norm_df = pd.DataFrame({m: _normalise_standard(df[m]) for m in metrics})

    # ── 4. Score pondéré F_value = Σ w_j * (m_j / max) ───────────
    score = sum(norm_df[m].fillna(0) * weights.get(m, 0) for m in metrics)

    res = pd.DataFrame({
        "Score Value":           score,
        **{m: df[m] for m in metrics},
        "Cours moyen":           df["Cours moyen"],
        "Période":               period_label,
    })
    # Exclure les titres avec score strictement nul (aucune métrique disponible)
    res = res[res["Score Value"] > 0].sort_values("Score Value", ascending=False)
    return res if not res.empty else None


def compute_momentum_factor(data, horizon_ranges, weights):
    """
    F_mom(t,T) = Σ w_h * rdt_moyen_h(t,T) / max_{E_t}(rdt_moyen_h)

    horizon_ranges : dict  {label: (date_start, date_end)}
    Pour chaque horizon activé, le rendement moyen journalier est calculé
    sur la plage [date_start, date_end] choisie par l'utilisateur.
    """
    returns = data["cours"].apply(pd.to_numeric, errors='coerce').pct_change().dropna(how='all')

    mom_cols = {}
    active_horizons = []

    for h, (d_start, d_end) in horizon_ranges.items():
        if d_start is None or d_end is None:
            continue
        try:
            sd = pd.to_datetime(d_start)
            ed = pd.to_datetime(d_end)
        except Exception:
            continue
        sub = returns[(returns.index >= sd) & (returns.index <= ed)]
        if sub.empty:
            continue
        rdt_moy = sub.mean(numeric_only=True).replace([np.inf, -np.inf], np.nan)
        mom_cols[h] = rdt_moy
        active_horizons.append(h)

    if not active_horizons:
        return None

    mom_df = pd.DataFrame(mom_cols)

    # Normalisation : m_ij / max_{E_t}(m_ij)
    norm_df = pd.DataFrame({h: _normalise_standard(mom_df[h]) for h in active_horizons})

    # Score pondéré — renormalise les poids sur les horizons actifs
    active_weights = {h: weights.get(h, 0) for h in active_horizons}
    w_sum = sum(active_weights.values())
    if w_sum > 0:
        active_weights = {h: w/w_sum for h, w in active_weights.items()}

    score = sum(norm_df[h].fillna(0) * active_weights[h] for h in active_horizons)

    res = pd.DataFrame({
        "Score Momentum": score,
        **{f"Rdt {h}": mom_df[h] for h in active_horizons},
    })
    return res.dropna(subset=["Score Momentum"]).sort_values("Score Momentum", ascending=False)


def compute_volatility_factor(data, end_date, window=252):
    """
    F_vol(t,T) = w * min_{E_t}(σ) / σ(T)
    Cas particulier inversé — titre le moins volatile reçoit le score le plus élevé.
    """
    returns = data["cours"].apply(pd.to_numeric, errors='coerce').pct_change().dropna(how='all')
    try:
        ed = pd.to_datetime(end_date)
    except Exception:
        ed = returns.index[-1]

    # m_ij = écart-type des rendements sur la fenêtre
    vol = returns[returns.index <= ed].tail(window).std(numeric_only=True).dropna()

    # F_vol = min_{E_t}(σ) / σ(T)  — w = 1 (métrique unique)
    score = _normalise_volatility(vol).replace([np.inf, -np.inf], np.nan).dropna()

    return pd.DataFrame({
        "Score Volatilité": score,
        "Écart-type σ":     vol,
        "σ annualisée":     vol * np.sqrt(252),
    }).sort_values("Score Volatilité", ascending=False)


def compute_dividend_factor(data, year, date_start=None, date_end=None):
    """
    F_div(t,T) = 1.0 * DY(t,T) / max_{E_t}(DY(t,T))

    Métrique unique (w = 100%) : Dividend Yield = Dividende annuel / Cours moyen
    Le cours moyen est calculé sur la plage [date_start, date_end].
    """
    div = data["dividendes"]
    dr  = div[div['Date'] == year]
    if dr.empty:
        return None

    # Cours moyen sur la plage choisie
    cours = data["cours"].apply(pd.to_numeric, errors="coerce")
    if date_start is not None and date_end is not None:
        mask = (cours.index >= pd.to_datetime(date_start)) & \
               (cours.index <= pd.to_datetime(date_end))
        cours_period = cours.loc[mask]
    else:
        cours_period = cours[cours.index.year == year]

    if cours_period.empty:
        return None

    cours_moy = cours_period.mean(numeric_only=True)

    yields = {}
    for t in data["tickers"]:
        if t not in dr.columns:
            continue
        d = dr[t].values[0]
        p = cours_moy.get(t, np.nan)
        if pd.notna(d) and pd.notna(p) and p > 0 and d >= 0:
            yields[t] = float(d) / float(p)

    if not yields:
        return None

    s = pd.Series(yields)
    # F_div = (100% × DY) / max_{E_t}(DY)   [100% = poids unique w=1]
    score = _normalise_standard(s)

    return pd.DataFrame({
        "Score Dividende": score,
        "Dividend Yield":  s,
        "Cours moyen":     cours_moy.reindex(s.index),
    }).sort_values("Score Dividende", ascending=False)


def compute_liquidity_factor(data, date_start=None, date_end=None):
    """
    F_liq(t,T) = 1.0 * Vol_moy(T) / max_{E_t}(Vol_moy)

    Métrique unique (w = 100%) : volume moyen transigé sur la plage choisie.
    """
    vol_df = data["volumes"].apply(pd.to_numeric, errors="coerce")

    if date_start is not None and date_end is not None:
        mask = (vol_df.index >= pd.to_datetime(date_start)) & \
               (vol_df.index <= pd.to_datetime(date_end))
        vol_period = vol_df.loc[mask]
        period_label = f"{date_start} → {date_end}"
    else:
        vol_period = vol_df
        period_label = "Historique complet"

    if vol_period.empty:
        return None

    avg = vol_period.mean(numeric_only=True).replace([np.inf, -np.inf], np.nan).dropna()

    # F_liq = Vol_moy / max_{E_t}(Vol_moy)
    score = _normalise_standard(avg)

    return pd.DataFrame({
        "Score Liquidité": score,
        "Volume moyen":    avg,
        "Période":         period_label,
    }).sort_values("Score Liquidité", ascending=False)

# ─── OPTIMISATION DE L'UNIVERS ────────────────────────────────

def filter_liquidity(mf_scores, liq_factor_result, min_vol_pct):
    """
    Filtre 1 — Liquidité minimale.
    Garde les titres dont le volume moyen dépasse min_vol_pct% du volume max.
    """
    if liq_factor_result is None or min_vol_pct <= 0:
        return mf_scores.index.tolist()
    vol = liq_factor_result["Volume moyen"]
    threshold = vol.max() * (min_vol_pct / 100)
    liquid = vol[vol >= threshold].index.tolist()
    return [t for t in mf_scores.index if t in liquid]


def filter_mf_percentile(mf_scores, universe, top_pct):
    """
    Filtre 2 — Seuil Score MF.
    Garde les titres dans le top top_pct% de l'univers courant.
    """
    sub = mf_scores.reindex(universe).dropna()
    if sub.empty or top_pct >= 100:
        return universe
    threshold = sub.quantile(1 - top_pct / 100)
    return sub[sub >= threshold].index.tolist()


def filter_correlation(mf_scores, universe, cours_df, max_corr, window=252):
    """
    Filtre 3 — Diversification par corrélation.
    Clustering hiérarchique sur la matrice de corrélations.
    Dans chaque cluster, seul le titre avec le meilleur score MF est retenu.
    max_corr : seuil de corrélation (0→1). Plus bas = plus diversifié.
    """
    if len(universe) < 3 or max_corr >= 1.0:
        return universe

    ret = cours_df[universe].apply(pd.to_numeric, errors="coerce").pct_change().dropna(how="all")
    ret = ret.tail(window).dropna(axis=1, thresh=int(window * 0.5))
    valid = [t for t in universe if t in ret.columns]
    if len(valid) < 3:
        return universe

    corr = ret[valid].corr().fillna(0)
    dist = np.clip(1 - corr.values, 0, 2)
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist)

    Z = linkage(condensed, method="ward")
    # Nombre de clusters : distance seuil = 1 - max_corr
    labels = fcluster(Z, t=(1 - max_corr), criterion="distance")

    # Pour chaque cluster, garder le titre au meilleur score MF
    selected = []
    sub_mf = mf_scores.reindex(valid)
    for cluster_id in np.unique(labels):
        members = [valid[i] for i, l in enumerate(labels) if l == cluster_id]
        best = sub_mf.reindex(members).idxmax()
        if pd.notna(best):
            selected.append(best)
    return selected


def optimize_markowitz(mf_scores, universe, cours_df,
                       window=252, risk_aversion=1.0,
                       mf_weight=0.5, min_w=0.0, max_w=1.0):
    """
    Filtre 4 — Optimisation Mean-Variance avec intégration du score MF.
    Maximise : mf_weight * MF_score_portefeuille - (1-mf_weight) * λ * variance
    Retourne les poids optimaux (Series ticker→poids).
    """
    ret = cours_df[universe].apply(pd.to_numeric, errors="coerce").pct_change().dropna(how="all")
    ret = ret.tail(window).dropna(axis=1, thresh=int(window * 0.5))
    valid = [t for t in universe if t in ret.columns]

    if len(valid) < 2:
        # Pas assez de données → revenir aux poids MF rank
        return None

    ret_valid = ret[valid]
    mu  = ret_valid.mean().values       # rendements moyens
    Sigma = ret_valid.cov().values      # matrice de covariance
    mf_vec = mf_scores.reindex(valid).fillna(0).values  # scores MF normalisés

    n = len(valid)
    w0 = np.ones(n) / n                # départ équipondéré

    def neg_utility(w):
        port_mf  = mf_weight * (w @ mf_vec)
        port_var = (1 - mf_weight) * risk_aversion * (w @ Sigma @ w)
        return -(port_mf - port_var)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(min_w, max_w)] * n

    res = minimize(neg_utility, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 500, "ftol": 1e-9})

    if not res.success:
        return None

    w_opt = pd.Series(res.x, index=valid)
    w_opt = w_opt[w_opt > 1e-4]        # drop poids négligeables
    w_opt = w_opt / w_opt.sum()        # renormalise
    return w_opt.sort_values(ascending=False)


def compute_portfolio_weights(mf, included=None):
    """
    α(T,t) = (n − r(T,t) + 1) / (n·(n+1)/2)
    Si included est fourni, seuls ces titres entrent dans le portefeuille.
    """
    if included:
        mf = mf.reindex(included).dropna()
    n = len(mf)
    if n == 0:
        return pd.Series(dtype=float)
    ranks = mf.rank(ascending=False, method='min')
    w = (n - ranks + 1) / (n * (n+1) / 2)
    return w.sort_values(ascending=False)


def run_optimization_pipeline(mf_scores, data, factor_results,
                              min_vol_pct, top_pct, max_corr,
                              use_markowitz, risk_aversion, mf_weight,
                              min_w, max_w, window):
    """
    Pipeline séquentiel des 4 filtres.
    Retourne (universe_final, weights, étapes_log).
    """
    log = []
    universe = mf_scores.index.tolist()
    log.append(("Univers initial", len(universe), universe))

    # Filtre 1 — Liquidité
    liq = factor_results.get("Liquidité")
    universe = filter_liquidity(mf_scores, liq, min_vol_pct)
    log.append(("① Filtre Liquidité", len(universe), universe))

    # Filtre 2 — Score MF
    universe = filter_mf_percentile(mf_scores, universe, top_pct)
    log.append(("② Filtre Score MF", len(universe), universe))

    # Filtre 3 — Corrélation
    universe = filter_correlation(mf_scores, universe,
                                  data["cours"], max_corr, window)
    log.append(("③ Filtre Corrélation", len(universe), universe))

    if len(universe) == 0:
        return [], None, log

    # Filtre 4 — Poids optimaux
    if use_markowitz:
        weights = optimize_markowitz(mf_scores, universe, data["cours"],
                                     window=window, risk_aversion=risk_aversion,
                                     mf_weight=mf_weight, min_w=min_w, max_w=max_w)
        if weights is None:
            # Fallback : poids MF rank
            weights = compute_portfolio_weights(mf_scores, included=universe)
            log.append(("④ Markowitz (fallback → rang MF)", len(weights), weights.index.tolist()))
        else:
            log.append(("④ Markowitz", len(weights), weights.index.tolist()))
    else:
        weights = compute_portfolio_weights(mf_scores, included=universe)
        log.append(("④ Poids rang MF", len(weights), weights.index.tolist()))

    return universe, weights, log


# ─── MULTIFACTOR & PORTFOLIO ──────────────────────────────────

def compute_multifactor(factor_results, betas):
    all_t = set()
    for df in factor_results.values():
        if df is not None:
            all_t.update(df.index)
    mf = pd.Series(0.0, index=list(all_t))
    for fname, df in factor_results.items():
        if df is None:
            continue
        sc = [c for c in df.columns if "Score" in c]
        if not sc:
            continue
        mf = mf.add(df[sc[0]] * betas.get(fname, 0), fill_value=0)
    return mf.sort_values(ascending=False)

def optimize_betas_ols(data, train_start, train_end,
                       target_start, target_end, year):
    """
    Approche 1 — Régression OLS.
    β_i = coefficients de régression linéaire qui minimisent :
    ||Rendement_T - Σ β_i * F_i(T)||²
    Les β obtenus représentent la contribution marginale de chaque facteur
    pour expliquer les rendements historiques.
    """
    FACTOR_NAMES = ["Value","Momentum","Volatilité","Dividende","Liquidité"]
    X, y, tickers = build_ml_dataset(
        data, train_start, train_end, target_start, target_end, year
    )
    if X is None or len(X) < 6:
        return None, None

    from numpy.linalg import lstsq
    X_arr = X.values
    y_arr = y.values

    # OLS : β = (XᵀX)⁻¹ Xᵀy
    coeffs, _, _, _ = lstsq(X_arr, y_arr, rcond=None)
    coeffs_s = pd.Series(coeffs, index=FACTOR_NAMES)

    # Normalisation : on prend les valeurs absolues et on renormalise
    # (un coefficient négatif = facteur inversement lié → on garde l'info mais β ≥ 0)
    abs_coeffs = coeffs_s.abs().clip(lower=0)
    total = abs_coeffs.sum()
    if total <= 0:
        return None, None
    betas = (abs_coeffs / total).to_dict()

    # R² manuel
    y_pred = X_arr @ coeffs
    ss_res = ((y_arr - y_pred)**2).sum()
    ss_tot = ((y_arr - y_arr.mean())**2).sum()
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0

    return betas, {
        "coefficients": coeffs_s.to_dict(),
        "r2": float(r2),
        "n_obs": len(y),
    }


def optimize_betas_walkforward(data, train_start, train_end,
                                target_start, target_end, year,
                                n_windows=4):
    """
    Approche 2 — Walk-forward (fenêtres glissantes + maximisation Sharpe).
    Divise [train_start → train_end] en n_windows sous-périodes.
    Pour chaque sous-période, calcule les scores factoriels et les rendements
    de la période suivante. Retourne les β moyens qui ont donné le meilleur
    Sharpe ratio cumulé.
    """
    FACTOR_NAMES = ["Value","Momentum","Volatilité","Dividende","Liquidité"]

    ts = pd.to_datetime(train_start)
    te = pd.to_datetime(train_end)
    tgs = pd.to_datetime(target_start)
    tge = pd.to_datetime(target_end)

    total_days = (tge - ts).days
    if total_days < 365 or n_windows < 2:
        return None, None

    # Découper en n_windows fenêtres glissantes
    window_days = total_days // (n_windows + 1)
    all_betas = []
    window_results = []

    for i in range(n_windows):
        w_train_start = ts + pd.Timedelta(days=i * window_days)
        w_train_end   = w_train_start + pd.Timedelta(days=window_days)
        w_target_start= w_train_end
        w_target_end  = w_target_start + pd.Timedelta(days=window_days)

        if w_target_end > tge:
            break

        X, y, _ = build_ml_dataset(
            data,
            w_train_start.date(), w_train_end.date(),
            w_target_start.date(), w_target_end.date(),
            year
        )
        if X is None or len(X) < 5:
            continue

        # Sharpe simple de chaque facteur sur la fenêtre
        sharpes = {}
        for f in FACTOR_NAMES:
            if f not in X.columns:
                sharpes[f] = 0
                continue
            scores = X[f]
            # Rendement du facteur = corrélation score × rendement réalisé
            corr = scores.corr(y)
            sharpes[f] = max(float(corr), 0)

        total_sh = sum(sharpes.values())
        if total_sh > 0:
            betas_w = {f: v/total_sh for f, v in sharpes.items()}
            all_betas.append(betas_w)
            window_results.append({
                "fenetre": i+1,
                "train": f"{w_train_start.date()} → {w_train_end.date()}",
                "target": f"{w_target_start.date()} → {w_target_end.date()}",
                "sharpes": sharpes,
            })

    if not all_betas:
        return None, None

    # Moyenne des β sur toutes les fenêtres
    avg_betas = {}
    for f in FACTOR_NAMES:
        avg_betas[f] = float(np.mean([b.get(f, 0) for b in all_betas]))

    total = sum(avg_betas.values())
    if total > 0:
        avg_betas = {f: v/total for f, v in avg_betas.items()}

    return avg_betas, {"n_windows": len(all_betas), "detail": window_results}


def vote_majority_betas(results_dict):
    """
    Vote majoritaire par facteur entre les 3 approches.
    Pour chaque facteur : β_final = médiane des β des approches disponibles.
    Puis renormalisation pour que Σβ = 1.
    """
    FACTOR_NAMES = ["Value","Momentum","Volatilité","Dividende","Liquidité"]
    available = {name: betas for name, betas in results_dict.items()
                 if betas is not None}
    if not available:
        return {f: 1/5 for f in FACTOR_NAMES}

    median_betas = {}
    for f in FACTOR_NAMES:
        vals = [b.get(f, 0) for b in available.values()]
        median_betas[f] = float(np.median(vals))

    total = sum(median_betas.values())
    if total > 0:
        median_betas = {f: v/total for f, v in median_betas.items()}
    return median_betas
    """
    α(T,t) = (n − r(T,t) + 1) / (n·(n+1)/2)
    Si included est fourni, seuls ces titres entrent dans le portefeuille.
    """
    if included:
        mf = mf.reindex(included).dropna()
    n = len(mf)
    if n == 0:
        return pd.Series(dtype=float)
    ranks = mf.rank(ascending=False, method='min')
    w = (n - ranks + 1) / (n * (n+1) / 2)
    return w.sort_values(ascending=False)


# ─── ML BETA OPTIMISATION ─────────────────────────────────────

def compute_scores_on_window(data, train_start, train_end, year):
    """
    Recalcule les 5 scores factoriels sur la plage [train_start, train_end]
    spécifiquement pour le ML — indépendant des onglets facteurs.

    Retourne un DataFrame (tickers × facteurs) avec les scores normalisés.
    """
    FACTOR_NAMES = ["Value", "Momentum", "Volatilité", "Dividende", "Liquidité"]
    scores = {}

    cours  = data["cours"].apply(pd.to_numeric, errors="coerce")
    ts, te = pd.to_datetime(train_start), pd.to_datetime(train_end)
    mask   = (cours.index >= ts) & (cours.index <= te)
    sub    = cours.loc[mask]
    if sub.empty:
        return None

    tickers = [t for t in data["tickers"] if t in sub.columns]

    # ── VALUE ─────────────────────────────────────────────────
    nb  = data["nb_titres"]
    cms = sub.mean(numeric_only=True)
    val_scores = {}
    for t in tickers:
        if t not in nb or nb[t] <= 0:
            continue
        cm = cms.get(t, np.nan)
        if pd.isna(cm) or cm <= 0:
            continue
        n   = nb[t]
        cp  = data["capitaux_propres"].get(t, {}).get(year, np.nan)
        rn  = data["resultat_net"].get(t, {}).get(year, np.nan)
        fcf = data["flux_treso"].get(t, {}).get(year, np.nan)
        ca  = data["chiffre_affaires"].get(t, {}).get(year, np.nan)
        rex = data["resultat_expl"].get(t, {}).get(year, np.nan)
        metrics = {
            "bp":  (cp/n)/cm  if pd.notna(cp)  else np.nan,
            "ep":  (rn/n)/cm  if pd.notna(rn)  else np.nan,
            "fp":  (fcf/n)/cm if pd.notna(fcf) else np.nan,
            "cp":  (ca/n)/cm  if pd.notna(ca)  else np.nan,
            "ebp": (rex/n)/cm if pd.notna(rex) else np.nan,
        }
        val_scores[t] = metrics
    if val_scores:
        vdf = pd.DataFrame(val_scores).T
        vdf = vdf.apply(pd.to_numeric, errors="coerce")
        norm_v = {}
        for m in vdf.columns:
            mx = vdf[m].max()
            norm_v[m] = vdf[m]/mx if (pd.notna(mx) and mx != 0) else vdf[m]*0
        norm_vdf = pd.DataFrame(norm_v)
        scores["Value"] = norm_vdf.mean(axis=1).fillna(0)

    # ── MOMENTUM (rendement moyen sur la fenêtre) ──────────────
    ret = sub.pct_change().dropna(how="all")
    if not ret.empty:
        mom = ret.mean(numeric_only=True).replace([np.inf,-np.inf], np.nan).dropna()
        mx  = mom.max()
        if pd.notna(mx) and mx != 0:
            scores["Momentum"] = (mom/mx).clip(0)
        else:
            scores["Momentum"] = mom*0

    # ── VOLATILITÉ (inversée) ───────────────────────────────────
    vol = ret.std(numeric_only=True).dropna()
    if not vol.empty:
        mn = vol.min()
        if pd.notna(mn) and mn > 0:
            scores["Volatilité"] = (mn/vol).replace([np.inf,-np.inf], np.nan).fillna(0)

    # ── DIVIDENDE ──────────────────────────────────────────────
    div = data["dividendes"]
    dr  = div[div["Date"] == year]
    if not dr.empty:
        yields = {}
        for t in tickers:
            if t not in dr.columns:
                continue
            d = dr[t].values[0]
            p = cms.get(t, np.nan)
            if pd.notna(d) and pd.notna(p) and p > 0:
                yields[t] = float(d)/float(p)
        if yields:
            s  = pd.Series(yields)
            mx = s.max()
            scores["Dividende"] = (s/mx if mx > 0 else s).clip(0)

    # ── LIQUIDITÉ ──────────────────────────────────────────────
    vol_df = data["volumes"].apply(pd.to_numeric, errors="coerce")
    vmask  = (vol_df.index >= ts) & (vol_df.index <= te)
    vsub   = vol_df.loc[vmask]
    if not vsub.empty:
        avg = vsub.mean(numeric_only=True).replace([np.inf,-np.inf], np.nan).dropna()
        mx  = avg.max()
        if pd.notna(mx) and mx > 0:
            scores["Liquidité"] = (avg/mx).clip(0)

    if not scores:
        return None

    score_df = pd.DataFrame(scores).reindex(columns=FACTOR_NAMES).fillna(0)
    return score_df


def build_ml_dataset(data, train_start, train_end, target_start, target_end, year):
    """
    Construit le dataset ML avec recalcul complet des scores sur la fenêtre ML :

      Features X  : F_i(T) recalculés sur [train_start → train_end]
                    → indépendant des plages définies dans les onglets facteurs
      Cible Y     : rendement total du titre T sur [target_start → target_end]

    Retourne (X_df, y_series, tickers) ou (None, None, None).
    """
    # ── Features : recalcul des scores sur la fenêtre d'entraînement ──
    X_df = compute_scores_on_window(data, train_start, train_end, year)
    if X_df is None or X_df.empty:
        return None, None, None

    # ── Cible : rendement total sur la fenêtre cible ────────────────
    cours = data["cours"].apply(pd.to_numeric, errors="coerce")
    mask  = (cours.index >= pd.to_datetime(target_start)) & \
            (cours.index <= pd.to_datetime(target_end))
    sub   = cours.loc[mask]

    if sub.empty or len(sub) < 2:
        return None, None, None

    returns = {}
    for t in X_df.index:
        if t not in sub.columns:
            continue
        s = sub[t].dropna()
        if len(s) < 2:
            continue
        returns[t] = (s.iloc[-1] - s.iloc[0]) / s.iloc[0]

    if len(returns) < 5:
        return None, None, None

    y      = pd.Series(returns)
    common = X_df.index.intersection(y.index)
    if len(common) < 5:
        return None, None, None

    return X_df.loc[common], y.loc[common], list(common)


def optimize_betas_ml(data, train_start, train_end,
                      target_start, target_end, year, n_estimators=200):
    """
    Pipeline complet :
      1. Recalcule les scores factoriels sur [train_start → train_end]
      2. Cible Y = rendements réalisés sur [target_start → target_end]
      3. Entraîne RF + GB, retourne les importances normalisées comme β_i
    """
    FACTOR_NAMES = ["Value", "Momentum", "Volatilité", "Dividende", "Liquidité"]

    X, y, tickers = build_ml_dataset(
        data, train_start, train_end,
        target_start, target_end, year
    )
    if X is None:
        return None, None, None

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    results = {}

    # ── Random Forest ──────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_sc, y)
    rf_imp = pd.Series(rf.feature_importances_, index=FACTOR_NAMES)
    results["Random Forest"] = {
        "importances": rf_imp,
        "r2": rf.score(X_sc, y),
    }

    # ── Gradient Boosting ──────────────────────────────────────
    gb = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_sc, y)
    gb_imp = pd.Series(gb.feature_importances_, index=FACTOR_NAMES)
    results["Gradient Boosting"] = {
        "importances": gb_imp,
        "r2": gb.score(X_sc, y),
    }

    # ── β combinés ────────────────────────────────────────────
    combined  = (rf_imp * 0.5 + gb_imp * 0.5).clip(lower=0)
    total     = combined.sum()
    betas_opt = (combined / total).to_dict() if total > 0 \
                else {f: 1/5 for f in FACTOR_NAMES}

    return betas_opt, results, {
        "X": X, "y": y, "tickers": tickers,
        "train_window": f"{train_start} → {train_end}",
        "target_window": f"{target_start} → {target_end}",
        "year_fundamentals": year,
    }

# ─── PLOT HELPERS ─────────────────────────────────────────────
PLOT_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   font=dict(color="#94a3b8", family="JetBrains Mono"),
                   margin=dict(l=10, r=10, t=20, b=60),
                   xaxis=dict(gridcolor="#1e2d45", tickangle=-45),
                   yaxis=dict(gridcolor="#1e2d45"))

def score_bar(series, color_end, height=360, fmt=".4f", yformat=None):
    top = series.head(25)
    fig = go.Figure(go.Bar(
        x=top.index, y=top.values,
        marker=dict(color=top.values,
                    colorscale=[[0,"#1e2d45"],[1, color_end]], line=dict(width=0)),
        text=[f"{v:{fmt}}" for v in top.values],
        textposition="outside",
        textfont=dict(size=9, color="#94a3b8"),
    ))
    layout = {**PLOT_LAYOUT, "height": height}
    if yformat:
        layout["yaxis"] = dict(gridcolor="#1e2d45", tickformat=yformat)
    fig.update_layout(**layout)
    return fig

# ─── SESSION STATE ────────────────────────────────────────────
for k in ["data","factor_results","mf_scores","pw"]:
    if k not in st.session_state:
        st.session_state[k] = {} if k == "factor_results" else None

# β sliders — initialisés inconditionnellement au démarrage
for k, default in [("sv_val",0.20),("sv_mom",0.20),("sv_vol",0.20),
                   ("sv_div",0.20),("sv_liq",0.20)]:
    if k not in st.session_state:
        st.session_state[k] = default

# Screening Charia
if "charia_results" not in st.session_state:
    st.session_state.charia_results = {}

# Valuation DB path
VALUATION_DB_PATH = "financial_db.json"
if "fin_data" not in st.session_state:
    st.session_state.fin_data = load_financial_db(VALUATION_DB_PATH) if VALUATION_AVAILABLE else {}

# ── Chargement de la classification sectorielle ───────────────
@st.cache_data
def load_sector_mapping():
    """Charge sectors.json et retourne {ticker: secteur}."""
    candidates = [
        os.path.join(_APP_DIR, "sectors.json"),
        "sectors.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            mapping = {}
            for sector, tickers in raw.items():
                for t in tickers:
                    mapping[t.strip().upper()] = sector
            return mapping
    return {}

SECTOR_MAP = load_sector_mapping()

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:14px 0 8px 0;'>
      <div style='font-family:Syne,sans-serif;font-size:20px;font-weight:800;
           background:linear-gradient(135deg,#3b82f6,#06b6d4);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>CGF GESTION</div>
      <div style='font-size:10px;color:#475569;letter-spacing:.12em;'>STRATÉGIE MULTIFACTORIELLE BRVM</div>
    </div><hr style='margin:8px 0 14px 0;'>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("📁 Base_de_données_-_SMF.xlsx",
                                 type=["xlsx","xls"], label_visibility="visible")
    if uploaded:
        try:
            st.session_state.data = load_data(uploaded.read())
            st.success(f"✅ {len(st.session_state.data['tickers'])} titres chargés")
        except Exception as e:
            st.error(f"Erreur : {e}")

    data = st.session_state.data
    if data:
        st.markdown("---")
        # Années fondamentaux disponibles (CP, RN, etc.) — exclut 2025 si absent
        fund_years = data.get("fundamental_years", [])
        if fund_years:
            selected_year = st.selectbox(
                "📅 Année fondamentaux (Value / Dividende)",
                fund_years,
                help="Années pour lesquelles les données de Capitaux propres et Résultat net sont disponibles dans le Tableau de bord."
            )
            st.caption(f"Données fondamentaux disponibles : {', '.join(str(y) for y in fund_years)}")
        else:
            selected_year = 2024
            st.warning("Aucune année de fondamentaux détectée.")
        date_ref = st.date_input("📅 Date ref. (Momentum / Vol.)", value=data["cours"].index[-1].date())
        st.markdown("---")
        st.markdown("**⚖️ Poids des facteurs β_i**")
        st.caption("Calibrez chaque facteur · Σβ doit = 1.0")

        b_val = st.slider("💰 Value",      0.0, 1.0, st.session_state.sv_val, 0.01, key="b_val")
        b_mom = st.slider("🚀 Momentum",   0.0, 1.0, st.session_state.sv_mom, 0.01, key="b_mom")
        b_vol = st.slider("📉 Volatilité", 0.0, 1.0, st.session_state.sv_vol, 0.01, key="b_vol")
        b_div = st.slider("💸 Dividende",  0.0, 1.0, st.session_state.sv_div, 0.01, key="b_div")
        b_liq = st.slider("💧 Liquidité",  0.0, 1.0, st.session_state.sv_liq, 0.01, key="b_liq")

        bs = round(b_val + b_mom + b_vol + b_div + b_liq, 4)
        if abs(bs - 1.0) <= 0.01:
            st.success(f"✅ Σβ = {bs:.2f}")
        else:
            st.warning(f"⚠️ Σβ = {bs:.2f} ≠ 1.0")

        if st.button("⚖️ Égaliser (20% chacun)"):
            # Écriture dans les clés de stockage (pas les clés widget)
            for k in ["sv_val","sv_mom","sv_vol","sv_div","sv_liq"]:
                st.session_state[k] = 0.20
            st.rerun()

        betas = {
            "Value":      b_val,
            "Momentum":   b_mom,
            "Volatilité": b_vol,
            "Dividende":  b_div,
            "Liquidité":  b_liq,
        }
        st.session_state.betas = betas
    else:
        selected_year, date_ref = 2024, None
        betas = st.session_state.get("betas",
                {"Value":0.20,"Momentum":0.20,"Volatilité":0.20,"Dividende":0.20,"Liquidité":0.20})

    st.markdown("---")
    st.markdown("""<div style='font-size:10px;color:#475569;line-height:1.6;'>
    Note Technique CGF Gestion · 10/05/2024<br>Auteur : A.B.A.M. Gueye</div>""", unsafe_allow_html=True)

    # ── Manuel d'utilisation intégré ──────────────────────────
    st.markdown("---")
    st.markdown("**📖 Manuel d'utilisation**")

    @st.cache_data(show_spinner=False)
    def generate_manual_pdf():
        """Génère le PDF du manuel et retourne les bytes."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, HRFlowable, KeepTogether
        )
        import datetime, io as _io

        BLUE   = colors.HexColor("#3b82f6")
        CYAN   = colors.HexColor("#06b6d4")
        GREEN  = colors.HexColor("#10b981")
        AMBER  = colors.HexColor("#f59e0b")
        RED    = colors.HexColor("#ef4444")
        PURPLE = colors.HexColor("#8b5cf6")
        LIGHT  = colors.HexColor("#f0f4ff")
        MUTED  = colors.HexColor("#64748b")
        DARK   = colors.HexColor("#1e293b")
        WHITE  = colors.white
        PW     = A4[0]

        def sty(name, **kw):
            return ParagraphStyle(name, **kw)

        S = {
            "cover_title": sty("ct", fontName="Helvetica-Bold", fontSize=26,
                               textColor=DARK, alignment=TA_CENTER, spaceAfter=6, leading=32),
            "cover_sub":   sty("cs", fontName="Helvetica", fontSize=12,
                               textColor=BLUE, alignment=TA_CENTER, spaceAfter=4),
            "cover_date":  sty("cd", fontName="Helvetica", fontSize=9,
                               textColor=MUTED, alignment=TA_CENTER, spaceAfter=10),
            "h1":  sty("h1", fontName="Helvetica-Bold", fontSize=15,
                       textColor=BLUE, spaceBefore=16, spaceAfter=7, leading=19),
            "h2":  sty("h2", fontName="Helvetica-Bold", fontSize=12,
                       textColor=DARK, spaceBefore=12, spaceAfter=5, leading=15),
            "h3":  sty("h3", fontName="Helvetica-Bold", fontSize=10,
                       textColor=MUTED, spaceBefore=8, spaceAfter=4, leading=13),
            "body": sty("body", fontName="Helvetica", fontSize=9.5,
                        textColor=DARK, alignment=TA_JUSTIFY, spaceAfter=5, leading=14),
            "bl":   sty("bl", fontName="Helvetica", fontSize=9.5,
                        textColor=DARK, spaceAfter=4, leading=14, leftIndent=14),
            "cap":  sty("cap", fontName="Helvetica-Oblique", fontSize=8.5,
                        textColor=MUTED, alignment=TA_CENTER, spaceAfter=4),
            "frm":  sty("frm", fontName="Courier-Bold", fontSize=9.5,
                        textColor=colors.HexColor("#1e3a5f"),
                        backColor=colors.HexColor("#dbeafe"),
                        alignment=TA_CENTER, spaceAfter=7, leading=15, borderPad=8),
            "toc":  sty("toc", fontName="Helvetica", fontSize=10.5,
                        textColor=DARK, spaceAfter=5, leading=17),
            "tocs": sty("tocs", fontName="Helvetica", fontSize=9.5,
                        textColor=MUTED, spaceAfter=4, leading=15, leftIndent=18),
        }

        def sp(n=1): return Spacer(1, n*0.32*cm)
        def hr(c=colors.HexColor("#e2e8f0"), w=0.5):
            return HRFlowable(width="100%", thickness=w, color=c, spaceAfter=6, spaceBefore=3)

        def badge(text, col=BLUE):
            t = Table([[Paragraph(f'<font color="white"><b>{text}</b></font>', S["bl"])]],
                      colWidths=[PW-4*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,-1),col),
                ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
                ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
            ]))
            return t

        def ibox(text, bg=LIGHT, border=BLUE):
            t = Table([[Paragraph(text, S["bl"])]], colWidths=[PW-4*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,-1),bg),
                ("LINEBEFORE",(0,0),(0,-1),3,border),
                ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
                ("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),10),
            ]))
            return t

        def tbl(headers, rows, col_w=None):
            data = [headers] + rows
            n = len(headers)
            cw = col_w or ([(PW-4*cm)/n]*n)
            t = Table(data, colWidths=cw, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),BLUE),
                ("TEXTCOLOR",(0,0),(-1,0),WHITE),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                ("FONTSIZE",(0,0),(-1,-1),8.5),
                ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,LIGHT]),
                ("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#cbd5e1")),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
                ("LEFTPADDING",(0,0),(-1,-1),7),
            ]))
            return t

        def ph(text): return Paragraph(f"<b>{text}</b>", S["bl"])
        def pb(text): return Paragraph(text, S["bl"])

        buf = _io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2.2*cm, bottomMargin=2.2*cm,
            title="Manuel CGF Gestion — SMF BRVM")

        story = []
        today = datetime.date.today().strftime("%d/%m/%Y")

        # ── COVER ────────────────────────────────────────────────
        story += [
            Spacer(1, 2.5*cm),
            Paragraph("MANUEL D'UTILISATION", S["cover_title"]),
            sp(0.3),
            Paragraph("Moteur d'Allocation Multifactoriel BRVM", S["cover_sub"]),
            Paragraph(f"CGF Gestion · Note Technique 10/05/2024 · v1.0 · {today}", S["cover_date"]),
            sp(),
            hr(BLUE, 1.5), sp(0.5),
        ]
        cov = Table([[pb("Application : Streamlit"), pb("Auteur : A.B.A.M. Gueye"),
                      pb("Marché : BRVM — 48 titres"), pb("Facteurs : 5 (Value · Mom · Vol · Div · Liq)")]],
                    colWidths=[(PW-4*cm)/4]*4)
        cov.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),LIGHT),
            ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#cbd5e1")),
            ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
            ("LEFTPADDING",(0,0),(-1,-1),8),("VALIGN",(0,0),(-1,-1),"TOP"),
            ("FONTSIZE",(0,0),(-1,-1),8.5)]))
        story += [cov, sp(2), hr(BLUE,1.5), Spacer(1,3*cm),
            Paragraph("Ce document décrit l'utilisation complète de l'application de gestion "
                "de portefeuille multifactorielle CGF Gestion, couvrant le chargement des "
                "données, le calcul des 5 facteurs, l'indice MF et l'optimisation du "
                "portefeuille selon la Note Technique du 10/05/2024.", S["body"]),
            PageBreak()]

        # ── TOC ──────────────────────────────────────────────────
        story += [Paragraph("TABLE DES MATIÈRES", S["h1"]), hr(), sp(0.3)]
        toc = [
            ("1.", "Présentation générale et workflow", False),
            ("2.", "Démarrage et chargement des données", False),
            ("2.1", "Chargement du fichier Excel", True),
            ("2.2", "Structure des 6 feuilles", True),
            ("2.3", "Mise à jour des données", True),
            ("3.", "Barre latérale — Paramètres globaux", False),
            ("4.", "Onglet Value — Facteur de valorisation", False),
            ("4.1", "Formules et 5 métriques", True),
            ("4.2", "Paramétrage et interprétation", True),
            ("5.", "Onglet Momentum — Dynamique des cours", False),
            ("5.1", "Formule et 6 horizons", True),
            ("5.2", "Plages de dates libres par horizon", True),
            ("6.", "Onglet Volatilité — Facteur de risque", False),
            ("7.", "Onglet Dividende — Rendement", False),
            ("8.", "Onglet Liquidité — Volume transigé", False),
            ("9.", "Onglet Indice Multifactoriel", False),
            ("10.", "Onglet Portefeuille Cible", False),
            ("10.1", "Pipeline automatique — 4 filtres", True),
            ("10.2", "Mode sélection manuelle", True),
            ("10.3", "Pondération finale (Rang MF / Markowitz)", True),
            ("11.", "Export des résultats", False),
            ("12.", "Référence des formules mathématiques", False),
            ("13.", "Résolution des problèmes fréquents", False),
        ]
        for num, title, sub in toc:
            style = S["tocs"] if sub else S["toc"]
            indent = "    " if sub else ""
            story.append(Paragraph(f"{indent}<b>{num}</b>  {title}", style))
        story.append(PageBreak())

        # ── 1. PRÉSENTATION ──────────────────────────────────────
        story += [badge("1. PRÉSENTATION GÉNÉRALE ET WORKFLOW", BLUE), sp(),
            Paragraph("L'application implémente la Note Technique CGF Gestion du 10/05/2024 "
                "sur la stratégie multifactorielle. Elle suit un processus en 3 étapes :", S["body"])]

        steps = [
            ("1", BLUE, "Calcul des indices factoriels (Étape 1)",
             "Calculer F_i(t,T) pour chaque facteur et chaque titre. "
             "5 onglets dédiés : Value, Momentum, Volatilité, Dividende, Liquidité."),
            ("2", PURPLE, "Calcul de l'indice multifactoriel (Étape 2)",
             "Agréger : MF(t,T) = Σ β_i · F_i(t,T). Les β sont calibrés dans la sidebar."),
            ("3", GREEN, "Construction du portefeuille cible (Étape 3)",
             "Pondération par rang MF ou optimisation Markowitz sur l'univers filtré."),
        ]
        for num, col, title, desc in steps:
            r = Table([[Paragraph(f'<font color="white"><b>{num}</b></font>', S["bl"]),
                        Paragraph(f'<b>{title}</b><br/>{desc}', S["bl"])]],
                      colWidths=[0.9*cm, PW-4*cm-0.9*cm])
            r.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(0,0),col),("BACKGROUND",(1,0),(1,0),LIGHT),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("ALIGN",(0,0),(0,0),"CENTER"),
                ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
                ("LEFTPADDING",(0,0),(-1,-1),9),
            ]))
            story += [sp(0.3), r]
        story += [sp(), PageBreak()]

        # ── 2. DONNÉES ───────────────────────────────────────────
        story += [badge("2. DÉMARRAGE ET CHARGEMENT DES DONNÉES", BLUE), sp(),
            Paragraph("2.1  Chargement du fichier Excel", S["h2"]),
            Paragraph("Dans la barre latérale, cliquez sur Browse files et sélectionnez "
                "Base_de_données_-_SMF.xlsx. Le message ✅ 48 titres chargés confirme "
                "le succès.", S["body"]),
            sp(0.3),
            Paragraph("2.2  Structure des 6 feuilles requises", S["h2"]),
            tbl([ph("Feuille"), ph("Contenu"), ph("Utilisée pour")],
                [[pb("Cours"), pb("Cours journaliers de clôture"), pb("Momentum · Volatilité")],
                 [pb("Volume moyen"), pb("Volumes journaliers transigés"), pb("Liquidité")],
                 [pb("Historique_dividende"), pb("Dividendes annuels versés"), pb("Dividende")],
                 [pb("Moyenne_cours"), pb("Cours moyen annuel"), pb("Référence interne")],
                 [pb("Nb_titres"), pb("Nombre de titres en circulation"), pb("Value (unitaire)")],
                 [pb("Tableau de bord"), pb("5 blocs : CP · RN · FCF · CA · Rex"), pb("Value")]],
                col_w=[(PW-4*cm)*0.28, (PW-4*cm)*0.42, (PW-4*cm)*0.30]),
            sp(0.5),
            Paragraph("2.3  Mise à jour des données (ex. ajout 2025)", S["h2"]),
            Paragraph("Ajoutez une ligne 2025 dans chacun des 5 blocs du Tableau de bord "
                "(CAPITAUX PROPRES, RESULTAT NET, FLUX DE TRESORERIE, CHIFFRE D'AFFAIRES, "
                "RESULTAT D'EXPLOITATION) en respectant le format des années 2019–2024. "
                "L'année 2025 apparaîtra automatiquement dans le sélecteur de la sidebar.", S["body"]),
            sp(0.3),
            ibox("⚠️  Les années proposées dans la sidebar sont limitées aux années disposant "
                "simultanément de Capitaux propres ET de Résultat net dans le Tableau de bord.",
                colors.HexColor("#fef3c7"), AMBER),
            PageBreak()]

        # ── 3. SIDEBAR ───────────────────────────────────────────
        story += [badge("3. BARRE LATÉRALE — PARAMÈTRES GLOBAUX", MUTED), sp(),
            Paragraph("La sidebar est permanente et accessible depuis tous les onglets.", S["body"]),
            Paragraph("Année fondamentaux", S["h2"]),
            Paragraph("Sélectionne l'année des données CP/RN/FCF/CA/Rex pour Value et Dividende. "
                "Limitée aux années réellement disponibles dans votre fichier.", S["body"]),
            Paragraph("Poids des facteurs β_i  (Σβ = 1.0)", S["h2"]),
            tbl([ph("Curseur"), ph("Facteur"), ph("Rôle")],
                [[pb("💰 β Value"), pb("Valorisation"), pb("Poids du score Value dans MF")],
                 [pb("🚀 β Momentum"), pb("Dynamique"), pb("Poids du score Momentum dans MF")],
                 [pb("📉 β Volatilité"), pb("Risque"), pb("Poids du score Volatilité dans MF")],
                 [pb("💸 β Dividende"), pb("Rendement"), pb("Poids du score Dividende dans MF")],
                 [pb("💧 β Liquidité"), pb("Négociabilité"), pb("Poids du score Liquidité dans MF")]]),
            sp(0.3),
            ibox("💡  Le bouton ⚖️ Égaliser remet tous les β à 0.20. Les poids sont persistés "
                "entre les onglets via le session_state de Streamlit.", LIGHT, BLUE),
            PageBreak()]

        # ── 4. VALUE ─────────────────────────────────────────────
        story += [badge("4. ONGLET VALUE — FACTEUR DE VALORISATION", BLUE), sp(),
            Paragraph("4.1  Formules et 5 métriques", S["h2"]),
            Paragraph("F_value(t,T) = w1·(B/P) + w2·(E/P) + w3·(FCF/P) + w4·(CA/P) + w5·(EBIT/P)", S["frm"]),
            Paragraph("Chaque métrique est normalisée : m_ij / max_E(m_ij). Score élevé = titre sous-évalué.", S["cap"]),
            sp(0.3),
            tbl([ph("Métrique"), ph("Formule"), ph("Poids défaut"), ph("Interprétation")],
                [[pb("B/P"), pb("(CP/n) / Cours moyen"), pb("20%"), pb("Valeur comptable/prix — ↑ = sous-évalué")],
                 [pb("E/P"), pb("(RN/n) / Cours moyen"), pb("20%"), pb("Inverse du P/E — ↑ = bon marché")],
                 [pb("FCF/P"), pb("(FCF/n) / Cours moyen"), pb("20%"), pb("Rendement des flux — ↑ = générateur de cash")],
                 [pb("CA/P"), pb("(CA/n) / Cours moyen"), pb("20%"), pb("Revenus/prix — ↑ = chiffre d'affaires élevé")],
                 [pb("EBIT/P"), pb("(Rex/n) / Cours moyen"), pb("20%"), pb("Profitabilité opérationnelle/prix")]],
                col_w=[(PW-4*cm)*0.12,(PW-4*cm)*0.30,(PW-4*cm)*0.16,(PW-4*cm)*0.42]),
            sp(0.5),
            Paragraph("4.2  Paramétrage et interprétation", S["h2"]),
            Paragraph("→  Période du cours moyen : choisissez Date début et Date fin. "
                "Le cours moyen est la moyenne arithmétique des cours journaliers sur la période. "
                "Les fondamentaux proviennent de l'année sélectionnée en sidebar.", S["bl"]),
            Paragraph("→  Pondérations : ajustez les 5 poids (somme = 1.0). "
                "Un poids à 0 exclut la métrique du calcul.", S["bl"]),
            Paragraph("→  Cliquez ⚙️ Calculer le Facteur Value pour lancer. "
                "Les résultats incluent KPIs, graphique de classement et table exportable.", S["bl"]),
            sp(0.3),
            ibox("⚠️  Score Value = 0.0000 : aucune métrique disponible pour ce titre "
                "sur l'année sélectionnée. Vérifiez le Tableau de bord dans votre fichier Excel.",
                colors.HexColor("#fef3c7"), AMBER),
            PageBreak()]

        # ── 5. MOMENTUM ──────────────────────────────────────────
        story += [badge("5. ONGLET MOMENTUM — DYNAMIQUE DES COURS", PURPLE), sp(),
            Paragraph("5.1  Formule", S["h2"]),
            Paragraph("F_mom(t,T) = Σ w_h · rdt_moy_h(t,T) / max_E(rdt_moy_h)", S["frm"]),
            Paragraph("rdt_moy_h = rendement journalier moyen sur la plage [debut_h, fin_h]", S["cap"]),
            sp(0.3),
            Paragraph("5.2  Plages de dates libres par horizon", S["h2"]),
            Paragraph("Chaque horizon dispose de son propre sélecteur de plage de dates :", S["body"]),
            sp(0.2),
            tbl([ph("Horizon"), ph("Plage par défaut"), ph("Exemple avancé")],
                [[pb("Journalier"), pb("J-1 → Aujourd'hui"), pb("Toujours 1 jour")],
                 [pb("Hebdo"), pb("J-7 → Aujourd'hui"), pb("Semaine de référence")],
                 [pb("Mensuel"), pb("J-30 → Aujourd'hui"), pb("Mois de référence")],
                 [pb("Trimestriel"), pb("J-91 → Aujourd'hui"), pb("Dernier trimestre")],
                 [pb("Semestriel"), pb("J-182 → Aujourd'hui"), pb("6 derniers mois")],
                 [pb("Annuel"), pb("J-365 → Aujourd'hui"), pb("5 ans : 20/05/2021 → 20/05/2026")]]),
            sp(0.3),
            ibox("💡  Pour désactiver un horizon, mettez son poids à 0. Les poids restants "
                "sont automatiquement renormalisés pour que leur somme reste égale à 1.0.", LIGHT, BLUE),
            PageBreak()]

        # ── 6·7·8 ────────────────────────────────────────────────
        story += [badge("6. VOLATILITÉ  |  7. DIVIDENDE  |  8. LIQUIDITÉ", MUTED), sp(),
            Paragraph("Facteur Volatilité (onglet 📉)", S["h2"]),
            Paragraph("F_vol(t,T) = min_E(sigma) / sigma(T)    [formule inversée]", S["frm"]),
            Paragraph("sigma = ecart-type des rendements journaliers sur la fenetre choisie (60–504 jours)", S["cap"]),
            Paragraph("Un titre peu volatil reçoit un score élevé. La fenêtre est ajustable "
                "via un curseur (défaut = 252 jours). Le graphique compare la volatilité "
                "annualisée (sigma × sqrt(252)) de tous les titres.", S["body"]),
            sp(0.5),
            Paragraph("Facteur Dividende (onglet 💸)", S["h2"]),
            Paragraph("F_div(t,T) = 1.0 x DY(t,T) / max_E(DY)    [poids unique = 100%]", S["frm"]),
            Paragraph("DY = Dividende annuel verse / Cours moyen sur la periode choisie", S["cap"]),
            Paragraph("Paramètres : année du dividende (issu de Historique_dividende) + "
                "plage de dates pour le cours moyen (indépendante). "
                "Un titre non-payeur reçoit DY = 0 et score = 0.", S["body"]),
            sp(0.5),
            Paragraph("Facteur Liquidité (onglet 💧)", S["h2"]),
            Paragraph("F_liq(t,T) = 1.0 x Vol_moy(T) / max_E(Vol_moy)    [poids unique = 100%]", S["frm"]),
            Paragraph("Vol_moy = volume moyen journalier sur la periode choisie", S["cap"]),
            Paragraph("Plage de dates entièrement libre sur l'historique disponible. "
                "Par défaut : historique complet.", S["body"]),
            PageBreak()]

        # ── 9. INDICE MF ─────────────────────────────────────────
        story += [badge("9. ONGLET INDICE MULTIFACTORIEL", PURPLE), sp(),
            Paragraph("MF(t,T) = beta_Value·F_Value + beta_Mom·F_Mom + beta_Vol·F_Vol + beta_Div·F_Div + beta_Liq·F_Liq", S["frm"]),
            Paragraph("Avec Σ beta_i = 1  ·  Les beta sont calibres dans la barre laterale", S["cap"]),
            sp(0.3),
            Paragraph("L'onglet affiche les β actifs avec leur statut (✓ calculé / ✗ non calculé). "
                "Il faut cliquer sur 🔢 Calculer l'Indice MF après chaque modification des β. "
                "Résultats : classement en barres, décomposition factorielle empilée Top 15 "
                "(contribution β_i·F_i de chaque facteur), table exportable.", S["body"]),
            sp(0.3),
            ibox("💡  Recalculez l'indice MF après toute modification des β en sidebar ou "
                "après recalcul d'un facteur dans ses onglets dédiés.", LIGHT, BLUE),
            PageBreak()]

        # ── 10. PORTEFEUILLE ─────────────────────────────────────
        story += [badge("10. ONGLET PORTEFEUILLE CIBLE", GREEN), sp(),
            Paragraph("10.1  Pipeline automatique — 4 filtres séquentiels", S["h2"]),
            tbl([ph("Filtre"), ph("Paramètre"), ph("Objectif"), ph("Défaut")],
                [[pb("① Liquidité"), pb("Volume ≥ X% du max"), pb("Exclut les titres peu liquides"), pb("0%")],
                 [pb("② Score MF"), pb("Top X% des scores"), pb("Ne garde que les mieux classés"), pb("60%")],
                 [pb("③ Corrélation"), pb("Corr. max entre titres"), pb("Clustering — 1 par groupe"), pb("0.75")],
                 [pb("④ Poids"), pb("Rang MF ou Markowitz"), pb("Pondération optimale"), pb("Rang MF")]],
                col_w=[(PW-4*cm)*0.18,(PW-4*cm)*0.27,(PW-4*cm)*0.33,(PW-4*cm)*0.22]),
            sp(0.3),
            Paragraph("La trace du pipeline s'affiche sous forme de cartes montrant le "
                "nombre de titres survivant à chaque filtre. Si le résultat est 0 titres, "
                "élargissez les seuils (Liquidité → 0%, Top MF → 100%, Corrélation → 1.0).", S["body"]),
            sp(0.5),
            Paragraph("10.2  Mode sélection manuelle", S["h2"]),
            Paragraph("Activez l'expander 🔧 Sélection manuelle pour choisir les titres "
                "titre par titre, court-circuitant les 3 premiers filtres. "
                "La pondération finale reste au choix.", S["body"]),
            sp(0.5),
            Paragraph("10.3  Pondération finale — Rang MF vs Markowitz", S["h2"]),
            tbl([ph("Méthode"), ph("Formule"), ph("Quand l'utiliser")],
                [[pb("Rang MF (défaut)"),
                  pb("alpha(T) = (n-r+1) / (n(n+1)/2)"),
                  pb("Approche conforme à la Note Technique · Simple · Reproductible")],
                 [pb("Markowitz MF-augmenté"),
                  pb("max{ lambda·Score_MF - (1-lambda)·variance }"),
                  pb("Optimise rendement/risque · Prend en compte les corrélations")]],
                col_w=[(PW-4*cm)*0.25,(PW-4*cm)*0.38,(PW-4*cm)*0.37]),
            sp(0.3),
            Paragraph("Markowitz : paramètres supplémentaires = poids Score MF dans l'utilité "
                "(0→1), aversion au risque λ, poids min/max par titre.", S["body"]),
            PageBreak()]

        # ── 11. EXPORT ───────────────────────────────────────────
        story += [badge("11. EXPORT DES RÉSULTATS", BLUE), sp(),
            tbl([ph("Onglet"), ph("Fichier Excel exporté"), ph("Contenu")],
                [[pb("Value"), pb("value_{annee}_{debut}_{fin}.xlsx"), pb("Scores + 5 métriques + cours moyen")],
                 [pb("Momentum"), pb("momentum.xlsx"), pb("Scores + rendements moyens par horizon")],
                 [pb("Dividende"), pb("dividende_{annee}.xlsx"), pb("Scores + Dividend Yield + cours moyen")],
                 [pb("Liquidité"), pb("liquidite_{debut}_{fin}.xlsx"), pb("Scores + volumes moyens")],
                 [pb("Indice MF"), pb("classement_MF.xlsx"), pb("Rang + Score MF par titre")],
                 [pb("Portefeuille"), pb("portefeuille_optimal_BRVM.xlsx"), pb("Rang MF + poids alpha + poids %")]],
                col_w=[(PW-4*cm)*0.18,(PW-4*cm)*0.40,(PW-4*cm)*0.42]),
            PageBreak()]

        # ── 12. FORMULES ─────────────────────────────────────────
        story += [badge("12. RÉFÉRENCE DES FORMULES MATHÉMATIQUES", DARK), sp(),
            Paragraph("Indice factoriel — formule générale", S["h2"]),
            Paragraph("F_i(t,T) = SUM(j=1..k_i) w_ij * m_ij(t,T) / max_Et(m_ij(t,T))", S["frm"]),
            Paragraph("Indice factoriel — cas Volatilité (inversé)", S["h2"]),
            Paragraph("F_vol(t,T) = SUM(j=1..k_i) w_ij * min_Et(m_ij(t,T)) / m_ij(t,T)", S["frm"]),
            Paragraph("Indice Multifactoriel", S["h2"]),
            Paragraph("MF(t,T) = SUM(i=1..5) beta_i * F_i(t,T)    avec SUM(beta_i) = 1", S["frm"]),
            Paragraph("Pondération portefeuille cible", S["h2"]),
            Paragraph("alpha(T,t) = (n(t) - r(T,t) + 1) / ( n(t) * (n(t)+1) / 2 )", S["frm"]),
            sp(0.3),
            tbl([ph("Variable"), ph("Définition")],
                [[pb("k_i"), pb("Nombre de métriques composant le facteur i")],
                 [pb("m_ij(t,T)"), pb("Valeur de la métrique j du facteur i pour le titre T à la date t")],
                 [pb("w_ij"), pb("Poids de la métrique j dans le facteur i  (Σw_ij = 1)")],
                 [pb("E_t"), pb("Ensemble des titres admissibles à la date t")],
                 [pb("beta_i"), pb("Poids du facteur i dans l'indice MF  (Σβ_i = 1)")],
                 [pb("r(T,t)"), pb("Rang du titre T selon MF (r=1 = meilleur score)")],
                 [pb("n(t)"), pb("Nombre total de titres admissibles à la date t")],
                 [pb("alpha(T,t)"), pb("Pondération du titre T dans le portefeuille cible")]],
                col_w=[(PW-4*cm)*0.20, (PW-4*cm)*0.80]),
            PageBreak()]

        # ── 13. PROBLÈMES ────────────────────────────────────────
        story += [badge("13. RÉSOLUTION DES PROBLÈMES FRÉQUENTS", RED), sp()]
        problems = [
            ("❌ \"Données insuffisantes\" dans Value",
             ['L\'année choisie (ex: 2025) n\'a pas de fondamentaux dans le Tableau de bord.',
              'Solution : choisissez une année disponible (2019–2024) ou renseignez les données 2025.']),
            ("❌ Score Value = 0.0000 pour tous les titres",
             ['Toutes les métriques affichent None : les fondamentaux ne sont pas lus.',
              'Solution : vérifiez la structure du Tableau de bord (5 blocs, tickers en ligne 0 col 2+).']),
            ("❌ Erreur TypeError sur la Liquidité",
             ['La feuille Volume moyen contient des colonnes non numériques (totaux, formules).',
              'Solution : supprimez les colonnes parasites et rechargez le fichier.']),
            ("❌ \"Aucun horizon valide\" dans Momentum",
             ['Toutes les plages de Momentum ont Date début >= Date fin.',
              'Solution : vérifiez que la date début est strictement antérieure à la date fin.']),
            ("❌ Pipeline Portefeuille retourne 0 titres",
             ['Les filtres sont trop restrictifs.',
              'Solution : réduisez Liquidité → 0%, montez Top MF → 100%, Corrélation → 1.0.']),
            ("❌ StreamlitAPIException sur les sélecteurs de dates",
             ['La valeur par défaut dépasse les bornes min/max de la date.',
              'Solution : rechargez la page (F5) — ce cas est normalement géré automatiquement.']),
        ]
        for title, items in problems:
            story += [KeepTogether(
                [Paragraph(title, S["h3"])] +
                [Paragraph(f"→  {item}", S["bl"]) for item in items] +
                [sp(0.4)]
            )]

        story += [hr(BLUE,1), sp(0.3),
            Paragraph(f"CGF Gestion · Moteur Multifactoriel BRVM · v1.0 · {today}", S["cap"])]

        doc.build(story)
        buf.seek(0)
        return buf.read()

    # Génère le PDF (mis en cache)
    pdf_bytes = generate_manual_pdf()

    # Bouton de téléchargement
    st.download_button(
        label="⬇️ Télécharger le manuel (PDF)",
        data=pdf_bytes,
        file_name="Manuel_CGF_SMF_BRVM.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    # Viewer inline dans un expander
    with st.expander("👁️ Consulter le manuel", expanded=False):
        import base64
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            f'width="100%" height="600px" style="border:1px solid #1e2d45;border-radius:6px;">'
            f'</iframe>',
            unsafe_allow_html=True,
        )

# ─── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div style='padding:6px 0 20px 0;'>
  <span class='pill'>Smart Beta · BRVM · 48 Titres</span>
  <p class='sh'>Moteur d'Allocation Multifactoriel</p>
  <p class='ss'>CGF Gestion · Note Technique 10/05/2024 · Facteurs : Value · Momentum · Volatilité · Dividende · Liquidité</p>
</div>""", unsafe_allow_html=True)

if not data:
    st.warning("⬅️ Veuillez charger **Base_de_données_-_SMF.xlsx** dans la barre latérale.")
    st.stop()

fr = st.session_state.factor_results

# ─── TABS ─────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
    "💰 Value", "🚀 Momentum", "📉 Volatilité", "💸 Dividende",
    "💧 Liquidité", "🔢 Indice MF", "📂 Portefeuille",
    "📈 Valorisation", "ℹ️ Données"])

# ══ VALUE ══════════════════════════════════════════════════════
with t1:
    st.markdown("<span class='pill'>Étape 1 · Facteur Value</span><p class='sh'>Valorisation relative au cours</p>", unsafe_allow_html=True)
    st.markdown("""<div class='fbox'>
    F_value(t,T) = Σ w_j · m_j(t,T) / max_{E_t}(m_j(t,T))<br><br>
    B/P  = (Capitaux propres / Nb titres) / Cours moyen &nbsp;→&nbsp; score ↑ = sous-évalué ✓<br>
    E/P  = (Résultat net / Nb titres) / Cours moyen &nbsp;&nbsp;&nbsp;&nbsp;→&nbsp; inverse de P/E  ✓<br>
    FCF/P = (FCF / Nb titres) / Cours moyen<br>
    CA/P  = (CA / Nb titres) / Cours moyen
    </div>""", unsafe_allow_html=True)

    # ── Paramètres de période ──────────────────────────────────
    st.markdown("**📅 Période de calcul du cours moyen**")
    vd1, vd2, vd3 = st.columns([1,1,1])
    with vd1:
        cours_min = data["cours"].index.min().date()
        cours_max = data["cours"].index.max().date()
        v_start_default = max(cours_min, min(cours_max,
                          pd.Timestamp(f"{selected_year}-01-01").date()))
        v_date_start = st.date_input("Date début", value=v_start_default,
                                      min_value=cours_min, max_value=cours_max, key="v_d1")
    with vd2:
        v_end_default = max(cours_min, min(cours_max,
                        pd.Timestamp(f"{selected_year}-12-31").date()))
        v_date_end   = st.date_input("Date fin", value=v_end_default,
                                      min_value=cours_min, max_value=cours_max, key="v_d2")
    with vd3:
        st.markdown(f"<br><span style='font-size:11px;color:#64748b;'>"
                    f"Fondamentaux année : <b>{selected_year}</b></span>", unsafe_allow_html=True)

    # ── Pondérations ───────────────────────────────────────────
    st.markdown("**⚖️ Pondérations des métriques**")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    w_bp   = mc1.number_input("B/P  (CP/n÷cours)",    0.0, 1.0, 0.20, 0.05, key="wbp")
    w_eps  = mc2.number_input("E/P  (RN/n÷cours)",    0.0, 1.0, 0.20, 0.05, key="wep")
    w_fcf  = mc3.number_input("FCF/P (FCF/n÷cours)",  0.0, 1.0, 0.20, 0.05, key="wfcf")
    w_ca   = mc4.number_input("CA/P  (CA/n÷cours)",   0.0, 1.0, 0.20, 0.05, key="wca")
    w_ebit = mc5.number_input("EBIT/P (Rex/n÷cours)", 0.0, 1.0, 0.20, 0.05, key="webit")

    ws = round(w_bp + w_eps + w_fcf + w_ca + w_ebit, 4)
    if abs(ws - 1.0) > 0.01:
        st.warning(f"⚠️ Somme des poids = {ws:.2f} ≠ 1.0")
    else:
        st.success(f"✅ Somme des poids = {ws:.2f}")

    if st.button("⚙️ Calculer le Facteur Value", type="primary"):
        metrics_v = ["B/P  (CP/n÷cours)", "E/P  (RN/n÷cours)",
                     "FCF/P (FCF/n÷cours)", "CA/P  (CA/n÷cours)",
                     "EBIT/P (Rex/n÷cours)"]
        weights_v = dict(zip(metrics_v, [w_bp, w_eps, w_fcf, w_ca, w_ebit]))
        res = compute_value_factor(data, selected_year, weights_v,
                                   date_start=v_date_start, date_end=v_date_end)
        if res is not None:
            fr["Value"] = res
            st.success(f"✅ {len(res)} titres scorés · cours moyen {v_date_start} → {v_date_end} · fondamentaux {selected_year}")
        else:
            st.error("Données insuffisantes pour la période et l'année sélectionnées.")

    if "Value" in fr:
        res = fr["Value"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Titres scorés",  len(res))
        m2.metric("Score max",      f"{res['Score Value'].max():.4f}")
        m3.metric("N°1 Value",      res.index[0])
        m4.metric("Score moyen",    f"{res['Score Value'].mean():.4f}")

        st.plotly_chart(score_bar(res["Score Value"], "#06b6d4"), width="stretch")

        metrics_v = ["B/P  (CP/n÷cours)", "E/P  (RN/n÷cours)",
                     "FCF/P (FCF/n÷cours)", "CA/P  (CA/n÷cours)",
                     "EBIT/P (Rex/n÷cours)"]
        all_possible = ["Score Value"] + metrics_v + ["Cours moyen", "Période"]
        show_cols = [c for c in all_possible if c in res.columns]
        disp = res[show_cols].reset_index().rename(columns={"index": "Ticker"})
        disp.insert(0, "Rang", range(1, len(disp) + 1))
        fmt = {"Score Value": "{:.4f}", "Cours moyen": "{:,.1f}"}
        fmt.update({m: "{:.6f}" for m in metrics_v})
        st.dataframe(disp.style.format(fmt).bar(subset=["Score Value"],
                     color=["#1e3a5f", "#3b82f6"]),
                     width="stretch", hide_index=True)

        buf = io.BytesIO()
        disp.to_excel(buf, index=False)
        st.download_button("⬇️ Exporter Value (Excel)", buf.getvalue(),
                           f"value_{selected_year}_{v_date_start}_{v_date_end}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══ MOMENTUM ═══════════════════════════════════════════════════
with t2:
    st.markdown("<span class='pill'>Étape 1 · Facteur Momentum</span><p class='sh'>Dynamique des cours — 6 horizons</p>", unsafe_allow_html=True)
    st.markdown("""<div class='fbox'>
    F_mom(t,T) = Σ w_h · rdt_moyen_h(t,T) / max_{E_t}(rdt_moyen_h)<br>
    Une seule plage globale · les 6 horizons calculent leur rendement moyen sur cette fenêtre
    </div>""", unsafe_allow_html=True)

    c_min = data["cours"].index.min().date()
    c_max = data["cours"].index.max().date()

    # ── Plage globale unique ──────────────────────────────────
    st.markdown("**📅 Plage de dates globale**")
    st.caption("Tous les horizons (Journalier, Hebdo, Mensuel...) calculent leur rendement moyen sur cette fenêtre.")
    mg1, mg2, mg3 = st.columns([1, 1, 1])
    with mg1:
        mom_global_start = st.date_input(
            "Date début", key="mom_global_start",
            value=max(c_min, min(c_max, (pd.Timestamp(c_max) - pd.Timedelta(days=365)).date())),
            min_value=c_min, max_value=c_max
        )
    with mg2:
        mom_global_end = st.date_input(
            "Date fin", key="mom_global_end",
            value=c_max, min_value=c_min, max_value=c_max
        )
    with mg3:
        if mom_global_start < mom_global_end:
            n_cal = (mom_global_end - mom_global_start).days
            # Nombre de jours de trading sur la période
            cours_tmp = data["cours"]
            n_trd = ((cours_tmp.index >= pd.to_datetime(mom_global_start)) &
                     (cours_tmp.index <= pd.to_datetime(mom_global_end))).sum()
            st.markdown("<br>", unsafe_allow_html=True)
            st.success(f"✅ **{n_cal}** jours cal. · **{n_trd}** jours trading")
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            st.error("⚠️ Date début ≥ Date fin")

    # ── Pondérations par horizon ──────────────────────────────
    st.markdown("**⚖️ Poids par horizon**")
    HORIZON_LABELS = ["Journalier","Hebdo","Mensuel","Trimestriel","Semestriel","Annuel"]
    wh_cols = st.columns(6)
    horizon_weights = {}
    for i, h in enumerate(HORIZON_LABELS):
        w = wh_cols[i].number_input(h, 0.0, 1.0, round(1/6, 4), 0.01, key=f"mom_w_{h}")
        horizon_weights[h] = w

    w_mom_sum = round(sum(horizon_weights.values()), 4)
    if abs(w_mom_sum - 1.0) > 0.01:
        st.warning(f"⚠️ Σ poids = {w_mom_sum:.2f} ≠ 1.0 — renormalisés automatiquement")
    else:
        st.success(f"✅ Σ poids = {w_mom_sum:.2f}")

    if st.button("⚙️ Calculer le Momentum", type="primary"):
        if mom_global_start >= mom_global_end:
            st.error("Plage de dates invalide.")
        else:
            # La même plage pour tous les horizons
            horizon_ranges = {h: (mom_global_start, mom_global_end) for h in HORIZON_LABELS}
            res = compute_momentum_factor(data, horizon_ranges, horizon_weights)
            if res is not None:
                fr["Momentum"] = res
                st.success(f"✅ {len(res)} titres scorés · plage {mom_global_start} → {mom_global_end}")
            else:
                st.error("Données insuffisantes pour la période sélectionnée.")

    if "Momentum" in fr:
        res = fr["Momentum"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Titres",    len(res))
        m2.metric("N°1",       res.index[0])
        m3.metric("Score max", f"{res['Score Momentum'].max():.4f}")

        st.plotly_chart(score_bar(res["Score Momentum"], "#8b5cf6"), width="stretch")

        rdt_c = [c for c in res.columns if "Rdt" in c]
        disp2 = res[["Score Momentum"] + rdt_c].head(20).reset_index().rename(columns={"index": "Ticker"})
        disp2.insert(0, "Rang", range(1, len(disp2)+1))
        st.dataframe(disp2.style.format(
            {"Score Momentum": "{:.4f}"} | {c: "{:.4%}" for c in rdt_c}
        ), width="stretch", hide_index=True)

        buf = io.BytesIO()
        disp2.to_excel(buf, index=False)
        st.download_button("⬇️ Exporter Momentum (Excel)", buf.getvalue(),
                           "momentum.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══ VOLATILITÉ ═════════════════════════════════════════════════
with t3:
    st.markdown("<span class='pill'>Étape 1 · Facteur Volatilité</span><p class='sh'>Faible volatilité — Inversion</p>", unsafe_allow_html=True)
    st.markdown("""<div class='fbox'>F_vol(t,T) = min_{E}(σ) / σ(T) &nbsp;·&nbsp; Score élevé = titre stable</div>""", unsafe_allow_html=True)
    window_v = st.slider("Fenêtre (jours de trading)", 60, 504, 252, 21)

    if st.button("⚙️ Calculer la Volatilité", type="primary"):
        res = compute_volatility_factor(data, str(date_ref), window_v)
        fr["Volatilité"] = res
        st.success(f"✅ {len(res)} titres scorés")

    if "Volatilité" in fr:
        res = fr["Volatilité"]
        m1,m2,m3 = st.columns(3)
        m1.metric("Titre le + stable", res.index[0])
        m2.metric("σ min", f"{res['Écart-type σ'].min():.4%}")
        m3.metric("σ moy", f"{res['Écart-type σ'].mean():.4%}")

        l,r = st.columns(2)
        with l:
            st.markdown("**Score (inversé) — Top 25**")
            st.plotly_chart(score_bar(res["Score Volatilité"], "#ef4444"), width="stretch")
        with r:
            st.markdown("**Volatilité annualisée (σ×√252)**")
            rv = res.sort_values("Écart-type σ")
            fig2 = go.Figure(go.Bar(x=rv.index, y=rv["σ annualisée"],
                                    marker=dict(color=rv["σ annualisée"],
                                                colorscale=[[0,"#10b981"],[0.5,"#f59e0b"],[1,"#ef4444"]])))
            fig2.update_layout(**{**PLOT_LAYOUT,"height":360,"yaxis":dict(gridcolor="#1e2d45",tickformat=".1%")})
            st.plotly_chart(fig2, width="stretch")

# ══ DIVIDENDE ══════════════════════════════════════════════════
with t4:
    st.markdown("<span class='pill'>Étape 1 · Facteur Dividende</span><p class='sh'>Dividend Yield = Dividende annuel / Cours moyen</p>", unsafe_allow_html=True)
    st.markdown("""<div class='fbox'>
    F_div(t,T) = 1.0 × DY(t,T) / max_{E_t}(DY(t,T))<br>
    Métrique unique (w = 100%) : DY = Dividende annuel versé / Cours moyen sur la période
    </div>""", unsafe_allow_html=True)

    div_years = sorted(data["dividendes"]["Date"].dropna().astype(int).unique(), reverse=True)
    dd1, dd2, dd3 = st.columns([1, 1, 1])
    with dd1:
        div_yr = st.selectbox("Année du dividende", div_years, key="div_yr")
    with dd2:
        d_cours_min = data["cours"].index.min().date()
        d_cours_max = data["cours"].index.max().date()
        d_start_default = max(d_cours_min, min(d_cours_max,
                          pd.Timestamp(f"{div_years[0]}-01-01").date()))
        d_date_start = st.date_input("Cours moyen — Date début",
                                      value=d_start_default,
                                      min_value=d_cours_min,
                                      max_value=d_cours_max, key="dd1")
    with dd3:
        d_end_default = max(d_cours_min, min(d_cours_max,
                        pd.Timestamp(f"{div_years[0]}-12-31").date()))
        d_date_end = st.date_input("Cours moyen — Date fin",
                                    value=d_end_default,
                                    min_value=d_cours_min,
                                    max_value=d_cours_max, key="dd2")

    if st.button("⚙️ Calculer le Dividende", type="primary"):
        res = compute_dividend_factor(data, div_yr,
                                       date_start=d_date_start, date_end=d_date_end)
        if res is not None:
            fr["Dividende"] = res
            paying = len(res[res["Dividend Yield"] > 0])
            st.success(f"✅ {len(res)} titres · {paying} payeurs · cours moyen {d_date_start} → {d_date_end}")
        else:
            st.error("Données insuffisantes pour la période sélectionnée.")

    if "Dividende" in fr:
        res = fr["Dividende"]
        paying = res[res["Dividend Yield"] > 0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Payeurs de dividende", len(paying))
        m2.metric("Yield max", f"{res['Dividend Yield'].max():.2%}")
        m3.metric("N°1 Dividende", res.index[0])

        fig_d = go.Figure(go.Bar(
            x=res.index, y=res["Dividend Yield"],
            marker=dict(color=res["Dividend Yield"],
                        colorscale=[[0,"#1e2d45"],[1,"#f59e0b"]]),
            text=[f"{v:.2%}" for v in res["Dividend Yield"]],
            textposition="outside", textfont=dict(size=9, color="#94a3b8")))
        fig_d.update_layout(**{**PLOT_LAYOUT, "height": 360,
                               "yaxis": dict(gridcolor="#1e2d45", tickformat=".1%")})
        st.plotly_chart(fig_d, width="stretch")

        disp_d = res.reset_index().rename(columns={"index": "Ticker"})
        disp_d.insert(0, "Rang", range(1, len(disp_d)+1))
        st.dataframe(disp_d.style.format({
            "Score Dividende": "{:.4f}",
            "Dividend Yield":  "{:.4%}",
            "Cours moyen":     "{:,.1f}",
        }), width="stretch", hide_index=True)

        buf = io.BytesIO()
        disp_d.to_excel(buf, index=False)
        st.download_button("⬇️ Exporter Dividende (Excel)", buf.getvalue(),
                           f"dividende_{div_yr}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══ LIQUIDITÉ ══════════════════════════════════════════════════
with t5:
    st.markdown("<span class='pill'>Étape 1 · Facteur Liquidité</span><p class='sh'>Volume moyen transigé sur la période</p>", unsafe_allow_html=True)
    st.markdown("""<div class='fbox'>
    F_liq(t,T) = 1.0 × Vol_moy(T) / max_{E_t}(Vol_moy)<br>
    Métrique unique (w = 100%) : volume moyen transigé calculé sur la plage de dates choisie
    </div>""", unsafe_allow_html=True)

    ll1, ll2 = st.columns(2)
    with ll1:
        l_date_start = st.date_input("Date début",
                                      value=data["volumes"].index.min().date(),
                                      min_value=data["volumes"].index.min().date(),
                                      max_value=data["volumes"].index.max().date(), key="ld1")
    with ll2:
        l_date_end   = st.date_input("Date fin",
                                      value=data["volumes"].index.max().date(),
                                      min_value=data["volumes"].index.min().date(),
                                      max_value=data["volumes"].index.max().date(), key="ld2")

    if st.button("⚙️ Calculer la Liquidité", type="primary"):
        res = compute_liquidity_factor(data, date_start=l_date_start, date_end=l_date_end)
        if res is not None:
            fr["Liquidité"] = res
            st.success(f"✅ {len(res)} titres scorés · période {l_date_start} → {l_date_end}")
        else:
            st.error("Données insuffisantes pour la période sélectionnée.")

    if "Liquidité" in fr:
        res = fr["Liquidité"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Titre le + liquide", res.index[0])
        m2.metric("Vol. moyen max",  f"{res['Volume moyen'].max()/1e6:.1f} M FCFA")
        m3.metric("Score max",       f"{res['Score Liquidité'].max():.4f}")

        fig_l = go.Figure(go.Bar(
            x=res.index, y=res["Volume moyen"] / 1e6,
            marker=dict(color=res["Score Liquidité"],
                        colorscale=[[0,"#1e2d45"],[1,"#06b6d4"]]),
            text=(res["Volume moyen"]/1e6).round(1),
            textposition="outside", textfont=dict(size=9, color="#94a3b8"),
        ))
        fig_l.update_layout(**{**PLOT_LAYOUT, "height": 360,
                               "yaxis": dict(gridcolor="#1e2d45", title="Volume moy. (M FCFA)")})
        st.plotly_chart(fig_l, width="stretch")

        disp_l = res[["Score Liquidité","Volume moyen"]].reset_index().rename(columns={"index":"Ticker"})
        disp_l.insert(0, "Rang", range(1, len(disp_l)+1))
        st.dataframe(disp_l.style.format({
            "Score Liquidité": "{:.4f}",
            "Volume moyen":    "{:,.0f}",
        }), width="stretch", hide_index=True)

        buf = io.BytesIO()
        disp_l.to_excel(buf, index=False)
        st.download_button("⬇️ Exporter Liquidité (Excel)", buf.getvalue(),
                           f"liquidite_{l_date_start}_{l_date_end}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══ INDICE MULTIFACTORIEL ══════════════════════════════════════
with t6:
    st.markdown("<span class='pill'>Étape 2</span><p class='sh'>Indice Multifactoriel</p>", unsafe_allow_html=True)
    st.markdown("""<div class='fbox'>MF(t,T) = Σ_{i=1}^{5} β_i · F_i(t,T) &nbsp;·&nbsp; Σ β_i = 1<br>
    Calibrez les β manuellement via la sidebar, ou laissez le ML les optimiser automatiquement</div>""",
    unsafe_allow_html=True)

    betas_mf = st.session_state.get("betas",
               {"Value":0.20,"Momentum":0.20,"Volatilité":0.20,"Dividende":0.20,"Liquidité":0.20})
    computed = list(fr.keys())
    CLRS  = {"Value":"#3b82f6","Momentum":"#8b5cf6","Volatilité":"#ef4444",
             "Dividende":"#f59e0b","Liquidité":"#06b6d4"}
    ICONS = {"Value":"💰","Momentum":"🚀","Volatilité":"📉","Dividende":"💸","Liquidité":"💧"}

    if not computed:
        st.info("👈 Calculez au moins un facteur (onglets 💰 🚀 📉 💸 💧) avant de continuer.")
    else:
        st.success(f"✅ Facteurs calculés : {', '.join(computed)}")

        # ── Choix du mode de calibration des β ────────────────────
        st.markdown("---")
        st.markdown("**⚖️ Mode de calibration des β_i**")
        beta_mode = st.radio(
            "Mode β",
            ["🎛️ Manuel — curseurs sidebar", "🤖 Automatique — optimisation ML"],
            horizontal=True,
            label_visibility="collapsed",
        )

        # ══════════════════════════════════════════════════════════
        # MODE A — MANUEL
        # ══════════════════════════════════════════════════════════
        if beta_mode == "🎛️ Manuel — curseurs sidebar":
            st.caption("Les β sont ceux définis dans la barre latérale ←")
            bc = st.columns(5)
            for i, fname in enumerate(["Value","Momentum","Volatilité","Dividende","Liquidité"]):
                with bc[i]:
                    st.markdown(
                        f"<div style='background:#161d2e;border:1px solid {CLRS[fname]};"
                        f"border-radius:8px;padding:10px;text-align:center;'>"
                        f"<div style='font-size:16px'>{ICONS[fname]}</div>"
                        f"<div style='font-size:10px;color:#94a3b8;'>{fname}</div>"
                        f"<div style='font-size:20px;font-weight:800;color:{CLRS[fname]};'>"
                        f"{betas_mf.get(fname,0):.2f}</div>"
                        f"<div style='font-size:10px;color:#64748b;'>"
                        f"{'✓' if fname in computed else '✗'}</div></div>",
                        unsafe_allow_html=True
                    )
            bs_mf = round(sum(betas_mf.values()), 4)
            if abs(bs_mf - 1.0) > 0.01:
                st.warning(f"⚠️ Σβ = {bs_mf:.2f} — ajustez les curseurs dans la sidebar")

        # ══════════════════════════════════════════════════════════
        # MODE B — OPTIMISATION ML
        # ══════════════════════════════════════════════════════════
        else:
            if not ML_AVAILABLE:
                st.error("❌ scikit-learn non installé. Ajoutez `scikit-learn>=1.3.0` dans requirements.txt.")
            else:
                st.markdown("""<div class='fbox' style='margin-top:8px;'>
                <b>Principe :</b> les scores factoriels sont recalculés sur la fenêtre d'entraînement,
                indépendamment des onglets facteurs. Le modèle apprend quels facteurs ont le mieux
                prédit les rendements réels sur la fenêtre cible.<br><br>
                <b>Features X</b> : F_i(T) recalculés sur [Début features → Fin features]<br>
                <b>Cible Y</b> : rendement total du titre T sur [Début cible → Fin cible]
                </div>""", unsafe_allow_html=True)

                ml_c1, ml_c2 = st.columns(2)
                c_min = data["cours"].index.min().date()
                c_max = data["cours"].index.max().date()

                with ml_c1:
                    st.markdown("**📐 Fenêtre Features X — Scores factoriels**")
                    st.caption("Les 5 scores F_i(T) sont recalculés sur cette plage de dates")
                    ml_ts = st.date_input("Début features", key="ml_ts",
                        value=max(c_min, min(c_max, pd.Timestamp("2019-01-01").date())),
                        min_value=c_min, max_value=c_max)
                    ml_te = st.date_input("Fin features", key="ml_te",
                        value=max(c_min, min(c_max, pd.Timestamp("2023-12-31").date())),
                        min_value=c_min, max_value=c_max)
                    fund_years = data.get("fundamental_years", [2024])
                    ml_year = st.selectbox(
                        "Année fondamentaux (Value/Dividende)",
                        fund_years, key="ml_year",
                        help="Année des CP, RN, FCF, CA, Rex utilisés pour recalculer les scores Value et Dividende"
                    )

                with ml_c2:
                    st.markdown("**🎯 Fenêtre Cible Y — Rendements réalisés**")
                    st.caption("Variable à prédire : rendement total (P_fin - P_deb) / P_deb")
                    ml_tgs = st.date_input("Début cible", key="ml_tgs",
                        value=max(c_min, min(c_max, pd.Timestamp("2024-01-01").date())),
                        min_value=c_min, max_value=c_max)
                    ml_tge = st.date_input("Fin cible", key="ml_tge",
                        value=c_max, min_value=c_min, max_value=c_max)
                    st.markdown("<br>", unsafe_allow_html=True)
                    if ml_tgs < ml_tge:
                        n_days = (ml_tge - ml_tgs).days
                        st.caption(f"↳ Fenêtre cible : **{n_days} jours** calendaires")

                # Validation visuelle des fenêtres
                if ml_ts < ml_te and ml_tgs < ml_tge:
                    if ml_te >= ml_tgs:
                        st.warning("⚠️ Les fenêtres se chevauchent — la fin des features "
                                   "est postérieure au début de la cible. "
                                   "Idéalement : Fin features < Début cible.")
                    else:
                        gap = (ml_tgs - ml_te).days
                        st.success(f"✅ Fenêtres valides · Écart entre les deux : {gap} jours")

                ml_p1, ml_p2 = st.columns(2)
                with ml_p1:
                    n_trees = st.slider("Nombre d'arbres", 50, 500, 200, 50, key="ml_ntrees")
                with ml_p2:
                    apply_auto = st.toggle("Appliquer β ML à la sidebar automatiquement",
                                           value=True, key="ml_auto")

                if st.button("🤖 Lancer les 3 approches + Vote majoritaire", type="primary"):
                    if ml_ts >= ml_te:
                        st.error("Fenêtre features invalide (début ≥ fin).")
                    elif ml_tgs >= ml_tge:
                        st.error("Fenêtre cible invalide (début ≥ fin).")
                    else:
                        results_3 = {}
                        infos_3   = {}
                        with st.spinner("Approche 1/3 — Régression OLS..."):
                            b1, i1 = optimize_betas_ols(
                                data, ml_ts, ml_te, ml_tgs, ml_tge, ml_year)
                            results_3["① OLS"] = b1
                            infos_3["① OLS"]   = i1

                        with st.spinner("Approche 2/3 — Walk-forward Sharpe..."):
                            b2, i2 = optimize_betas_walkforward(
                                data, ml_ts, ml_te, ml_tgs, ml_tge, ml_year)
                            results_3["② Walk-forward"] = b2
                            infos_3["② Walk-forward"]   = i2

                        with st.spinner("Approche 3/3 — ML (Random Forest + Gradient Boosting)..."):
                            b3, ml_res, ml_ds = optimize_betas_ml(
                                data=data,
                                train_start=ml_ts, train_end=ml_te,
                                target_start=ml_tgs, target_end=ml_tge,
                                year=ml_year, n_estimators=n_trees
                            )
                            results_3["③ ML (RF+GB)"] = b3

                        # Vote majoritaire = médiane par facteur
                        betas_opt = vote_majority_betas(results_3)
                        n_ok = sum(1 for b in results_3.values() if b is not None)

                        st.session_state.ml_betas     = betas_opt
                        st.session_state.ml_all_betas = results_3
                        st.session_state.ml_results   = ml_res
                        st.session_state.ml_dataset   = ml_ds

                        if apply_auto and betas_opt:
                            km = {"Value":"sv_val","Momentum":"sv_mom",
                                  "Volatilité":"sv_vol","Dividende":"sv_div","Liquidité":"sv_liq"}
                            for f, b in betas_opt.items():
                                if f in km:
                                    st.session_state[km[f]] = round(float(b), 4)
                            st.success(
                                f"✅ {n_ok}/3 approches réussies · "
                                f"Vote majoritaire appliqué à la sidebar"
                            )
                            st.rerun()
                        else:
                            st.success(f"✅ {n_ok}/3 approches réussies · Cliquez 📥 pour appliquer")

                # ── Résultats 3 approches ──────────────────────────────
                if "ml_betas" in st.session_state and st.session_state.ml_betas:
                    betas_opt    = st.session_state.ml_betas
                    all_betas    = st.session_state.get("ml_all_betas", {})
                    ml_res       = st.session_state.get("ml_results")
                    ml_ds        = st.session_state.get("ml_dataset")

                    st.markdown("---")
                    st.markdown("**📊 Comparaison des 3 approches — β par facteur**")

                    FACTORS = ["Value","Momentum","Volatilité","Dividende","Liquidité"]
                    APPROACHES = list(all_betas.keys()) if all_betas else []
                    A_COLORS = ["#3b82f6","#10b981","#8b5cf6"]

                    # Graphique comparatif
                    fig_3a = go.Figure()
                    for idx, (approach, betas_a) in enumerate(all_betas.items()):
                        if betas_a is None:
                            continue
                        fig_3a.add_trace(go.Bar(
                            name=approach,
                            x=[f"{ICONS.get(f,'')} {f}" for f in FACTORS],
                            y=[betas_a.get(f, 0) for f in FACTORS],
                            marker_color=A_COLORS[idx % len(A_COLORS)],
                            opacity=0.8,
                        ))
                    # Vote majoritaire en ligne
                    fig_3a.add_trace(go.Scatter(
                        name="🗳️ Vote majoritaire",
                        x=[f"{ICONS.get(f,'')} {f}" for f in FACTORS],
                        y=[betas_opt.get(f, 0) for f in FACTORS],
                        mode="lines+markers",
                        line=dict(color="#f59e0b", width=2.5, dash="dash"),
                        marker=dict(size=10, symbol="diamond"),
                    ))
                    fig_3a.update_layout(
                        barmode="group",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94a3b8", family="JetBrains Mono"), height=340,
                        legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
                        margin=dict(l=10, r=10, t=50, b=30),
                        xaxis=dict(gridcolor="#1e2d45"),
                        yaxis=dict(gridcolor="#1e2d45", title="β_i"),
                    )
                    st.plotly_chart(fig_3a, width="stretch")

                    # Tableau de synthèse
                    st.markdown("**β par facteur — Tableau de synthèse**")
                    tbl_rows = []
                    for f in FACTORS:
                        row = {"Facteur": f"{ICONS.get(f,'')} {f}"}
                        for ap, betas_a in all_betas.items():
                            row[ap] = f"{betas_a.get(f,0):.3f}" if betas_a else "✗"
                        row["🗳️ Vote final"] = f"**{betas_opt.get(f,0):.3f}**"
                        tbl_rows.append(row)
                    st.dataframe(pd.DataFrame(tbl_rows), width="stretch", hide_index=True)

                    # β finaux en cards
                    st.markdown("**β finaux (vote majoritaire) — appliqués à la sidebar**")
                    b_cols = st.columns(5)
                    for i, fname in enumerate(FACTORS):
                        bv = betas_opt.get(fname, 0)
                        with b_cols[i]:
                            st.markdown(
                                f"<div style='background:#161d2e;border:1px solid {CLRS[fname]};"
                                f"border-radius:8px;padding:10px;text-align:center;'>"
                                f"<div style='font-size:16px'>{ICONS[fname]}</div>"
                                f"<div style='font-size:10px;color:#94a3b8;'>{fname}</div>"
                                f"<div style='font-size:22px;font-weight:800;color:{CLRS[fname]};'>"
                                f"{bv:.3f}</div>"
                                f"<div style='font-size:10px;color:#64748b;'>{bv*100:.1f}%</div>"
                                f"</div>", unsafe_allow_html=True
                            )

                    # Graphique comparaison RF vs GB
                    st.markdown("**Comparaison Random Forest vs Gradient Boosting**")
                    factors_ord = ["Value","Momentum","Volatilité","Dividende","Liquidité"]
                    rf_imp = ml_res["Random Forest"]["importances"]
                    gb_imp = ml_res["Gradient Boosting"]["importances"]
                    comb   = (rf_imp*0.5 + gb_imp*0.5)
                    x_lbl  = [f"{ICONS.get(f,'')} {f}" for f in factors_ord]

                    fig_ml = go.Figure()
                    fig_ml.add_trace(go.Bar(name="Random Forest", x=x_lbl,
                        y=[rf_imp.get(f,0) for f in factors_ord],
                        marker_color="#3b82f6", opacity=0.8))
                    fig_ml.add_trace(go.Bar(name="Gradient Boosting", x=x_lbl,
                        y=[gb_imp.get(f,0) for f in factors_ord],
                        marker_color="#8b5cf6", opacity=0.8))
                    fig_ml.add_trace(go.Scatter(name="β combiné", x=x_lbl,
                        y=[comb.get(f,0) for f in factors_ord],
                        mode="lines+markers",
                        line=dict(color="#06b6d4", width=2, dash="dash"),
                        marker=dict(size=8, symbol="diamond")))
                    fig_ml.update_layout(barmode="group",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94a3b8", family="JetBrains Mono"), height=300,
                        legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
                        margin=dict(l=10,r=10,t=45,b=30),
                        xaxis=dict(gridcolor="#1e2d45"),
                        yaxis=dict(gridcolor="#1e2d45", title="Importance"))
                    st.plotly_chart(fig_ml, width="stretch")

                    # Tableau comparatif β ML vs sidebar
                    st.markdown("**β ML optimaux vs β sidebar actuels**")
                    cur = st.session_state.get("betas",{f:0.20 for f in factors_ord})
                    rows_cmp = []
                    for f in factors_ord:
                        b_ml  = betas_opt.get(f, 0)
                        b_cur = cur.get(f, 0)
                        diff  = b_ml - b_cur
                        arr   = "▲" if diff > 0.01 else ("▼" if diff < -0.01 else "≈")
                        rows_cmp.append({
                            "Facteur": f"{ICONS.get(f,'')} {f}",
                            "β ML": f"{b_ml:.4f}",
                            "β sidebar": f"{b_cur:.4f}",
                            "Δ": f"{arr} {diff:+.4f}",
                            "R² RF": f"{ml_res['Random Forest']['r2']:.4f}",
                            "R² GB": f"{ml_res['Gradient Boosting']['r2']:.4f}",
                        })
                    st.dataframe(pd.DataFrame(rows_cmp), width="stretch", hide_index=True)

                    # Infos dataset
                    st.markdown("**ℹ️ Résumé du dataset ML**")
                    di1, di2, di3, di4 = st.columns(4)
                    di1.metric("Titres", len(ml_ds["tickers"]))
                    di2.metric("Features X", ml_ds["X"].shape[1])
                    di3.metric("Rdt cible max", f"{ml_ds['y'].max():.2%}")
                    di4.metric("Rdt cible min", f"{ml_ds['y'].min():.2%}")
                    st.caption(
                        f"Fenêtre features : **{ml_ds.get('train_window','—')}** "
                        f"(fondamentaux {ml_ds.get('year_fundamentals','—')})  ·  "
                        f"Fenêtre cible : **{ml_ds.get('target_window','—')}**"
                    )

                    if not apply_auto or "ml_betas" in st.session_state:
                        if st.button("📥 Appliquer les β ML à la sidebar", key="apply_ml_btn"):
                            km = {"Value":"sv_val","Momentum":"sv_mom",
                                  "Volatilité":"sv_vol","Dividende":"sv_div","Liquidité":"sv_liq"}
                            for f, b in betas_opt.items():
                                if f in km:
                                    st.session_state[km[f]] = round(float(b), 4)
                            st.rerun()

                    # Mise à jour de betas_mf pour le calcul MF ci-dessous
                    betas_mf = betas_opt

        # ── Calcul MF (commun aux deux modes) ─────────────────────
        st.markdown("---")
        if st.button("🔢 Calculer l'Indice MF", type="primary"):
            mf = compute_multifactor(fr, betas_mf)
            st.session_state.mf_scores = mf
            st.success(f"✅ {len(mf)} titres classés · β = { {k: round(v,3) for k,v in betas_mf.items()} }")

        if st.session_state.mf_scores is not None:
            mf = st.session_state.mf_scores
            st.markdown("---")
            st.plotly_chart(score_bar(mf, "#3b82f6", height=380), width="stretch")

            st.markdown("**Décomposition factorielle — Top 15**")
            top15 = mf.head(15).index
            fig_s = go.Figure()
            for fname in computed:
                sc = [c for c in fr[fname].columns if "Score" in c][0]
                vals = [fr[fname].loc[t, sc] * betas_mf.get(fname, 0)
                        if t in fr[fname].index else 0 for t in top15]
                fig_s.add_trace(go.Bar(
                    name=f"{ICONS.get(fname,'')} {fname}",
                    x=list(top15), y=vals,
                    marker_color=CLRS.get(fname, "#64748b"), opacity=0.85
                ))
            fig_s.update_layout(barmode="stack",
                **{**PLOT_LAYOUT, "height":340,
                   "legend":dict(orientation="h",y=1.08,bgcolor="rgba(0,0,0,0)"),
                   "margin":dict(l=10,r=10,t=50,b=50)})
            st.plotly_chart(fig_s, width="stretch")

            tbl_mf = mf.reset_index()
            tbl_mf.columns = ["Ticker","Score MF"]
            tbl_mf.insert(0,"Rang",range(1,len(tbl_mf)+1))
            tbl_mf["Score MF"] = tbl_mf["Score MF"].round(6)
            st.dataframe(tbl_mf, width="stretch", hide_index=True)

            buf = io.BytesIO()
            tbl_mf.to_excel(buf, index=False)
            st.download_button("⬇️ Exporter classement MF", buf.getvalue(),
                "classement_MF.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══ PORTEFEUILLE ═══════════════════════════════════════════
with t7:
    st.markdown("<span class='pill'>Étape 3</span><p class='sh'>Construction du Portefeuille Cible</p>", unsafe_allow_html=True)

    if st.session_state.mf_scores is None:
        st.info("👈 Calculez d'abord l'Indice MF (onglet 🔢).")
    else:
        mf = st.session_state.mf_scores
        all_tickers = mf.index.tolist()

        # ── Choix du mode ─────────────────────────────────────────
        st.markdown("**🎯 Mode de construction du portefeuille**")
        mode = st.radio(
            label="Mode",
            options=[
                "🤖 Optimisation automatique (pipeline de filtres)",
                "🖐️ Sélection manuelle des titres",
            ],
            label_visibility="collapsed",
            horizontal=True,
        )
        st.markdown("---")

        # ══════════════════════════════════════════════════════════
        # MODE A — PIPELINE AUTOMATIQUE
        # ══════════════════════════════════════════════════════════
        if mode == "🤖 Optimisation automatique (pipeline de filtres)":
            st.markdown("""<div class='fbox'>
            Pipeline séquentiel — chaque filtre réduit l'univers :<br>
            Univers complet → ① Liquidité → ② Score MF → ③ Corrélation → ④ Poids finaux
            </div>""", unsafe_allow_html=True)

            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                st.markdown("**① Filtre Liquidité**")
                liq_ok = "Liquidité" in fr
                min_vol_pct = st.slider("Volume moyen ≥ X% du max", 0, 50, 10, 1,
                    help="0% = pas de filtre", disabled=not liq_ok)
                if not liq_ok:
                    st.caption("⚠️ Calculez le facteur Liquidité pour activer")
            with col_f2:
                st.markdown("**② Filtre Score MF**")
                top_pct = st.slider("Garder le top X% des scores MF", 10, 100, 60, 5,
                    help="60% = garde les 60% de titres avec les meilleurs scores")
            with col_f3:
                st.markdown("**③ Filtre Corrélation**")
                max_corr = st.slider("Corrélation max entre titres", 0.3, 1.0, 0.75, 0.05,
                    help="Deux titres corrélés au-delà de ce seuil → on garde le meilleur")
                window_corr = st.number_input("Fenêtre (jours)", 60, 504, 252, 21, key="win_corr_a")

            st.markdown("---")
            st.markdown("**④ Pondération finale**")
            use_mkz = st.toggle("Optimisation Markowitz MF-augmentée", value=False,
                help="OFF = poids par rang MF (note technique) · ON = Mean-Variance avec score MF")
            if use_mkz:
                mk1,mk2,mk3,mk4 = st.columns(4)
                mf_weight     = mk1.slider("Poids Score MF",    0.0,1.0,0.50,0.05)
                risk_aversion = mk2.slider("Aversion risque λ", 0.1,10.0,2.0,0.1)
                min_w         = mk3.slider("Poids min/titre (%)",0,20,0,1)/100
                max_w         = mk4.slider("Poids max/titre (%)",5,100,40,5)/100
            else:
                mf_weight,risk_aversion,min_w,max_w = 0.5,2.0,0.0,1.0

            if st.button("🚀 Lancer le pipeline", type="primary"):
                with st.spinner("Optimisation en cours..."):
                    inc,pw,pipeline_log = run_optimization_pipeline(
                        mf_scores=mf, data=data, factor_results=fr,
                        min_vol_pct=min_vol_pct if liq_ok else 0,
                        top_pct=top_pct, max_corr=max_corr,
                        use_markowitz=use_mkz, risk_aversion=risk_aversion,
                        mf_weight=mf_weight, min_w=min_w, max_w=max_w,
                        window=int(window_corr),
                    )
                if pw is None or len(pw)==0:
                    st.error("❌ Aucun titre retenu — élargissez les seuils des filtres.")
                else:
                    st.session_state.pw         = pw
                    st.session_state.pw_log     = pipeline_log
                    st.session_state.pw_use_mkz = use_mkz
                    st.session_state.pw_mode    = "auto"
                    st.success(f"✅ {len(pw)} titres retenus · Σα = {pw.sum():.6f}")

        # ══════════════════════════════════════════════════════════
        # MODE B — SÉLECTION MANUELLE
        # ══════════════════════════════════════════════════════════
        else:
            st.markdown("""<div class='fbox'>
            Vous choisissez librement les titres à inclure, en vous appuyant sur le classement MF.<br>
            Les pondérations α sont ensuite calculées sur cet univers restreint.
            </div>""", unsafe_allow_html=True)

            with st.expander("📊 Classement MF — aide à la sélection", expanded=True):
                mf_disp = mf.reset_index()
                mf_disp.columns = ["Ticker","Score MF"]
                mf_disp.insert(0,"Rang",range(1,len(mf_disp)+1))
                mf_disp["Score MF"] = mf_disp["Score MF"].round(6)
                st.dataframe(
                    mf_disp.style.format({"Score MF":"{:.6f}"})
                           .bar(subset=["Score MF"],color=["#1e3a5f","#3b82f6"]),
                    width="stretch", hide_index=True, height=300
                )

            included = st.multiselect(
                "✅ Titres à inclure dans le portefeuille",
                options=all_tickers, default=all_tickers,
                help="Consultez le classement MF ci-dessus pour guider votre sélection"
            )
            n_inc = len(included)
            if n_inc == 0:
                st.warning("⚠️ Sélectionnez au moins un titre.")
            elif n_inc < len(all_tickers):
                st.info(f"ℹ️ {n_inc} titres sélectionnés sur {len(all_tickers)}")
            else:
                st.success(f"✅ Univers complet ({n_inc} titres)")

            st.markdown("---")
            st.markdown("**④ Pondération finale**")
            use_mkz_m = st.toggle("Optimisation Markowitz MF-augmentée", value=False,
                key="mkz_manual",
                help="OFF = poids par rang MF · ON = Mean-Variance avec score MF")
            if use_mkz_m:
                mm1,mm2,mm3,mm4 = st.columns(4)
                mf_weight_m     = mm1.slider("Poids Score MF",    0.0,1.0,0.50,0.05,key="mfw_m")
                risk_aversion_m = mm2.slider("Aversion risque λ", 0.1,10.0,2.0,0.1,key="rav_m")
                min_w_m         = mm3.slider("Poids min/titre (%)",0,20,0,1,key="mnw_m")/100
                max_w_m         = mm4.slider("Poids max/titre (%)",5,100,40,5,key="mxw_m")/100
            else:
                mf_weight_m,risk_aversion_m,min_w_m,max_w_m = 0.5,2.0,0.0,1.0

            if st.button("📂 Construire le portefeuille", type="primary") and n_inc > 0:
                with st.spinner("Calcul des poids..."):
                    if use_mkz_m:
                        pw = optimize_markowitz(mf, included, data["cours"],
                            window=252, risk_aversion=risk_aversion_m,
                            mf_weight=mf_weight_m, min_w=min_w_m, max_w=max_w_m)
                        if pw is None:
                            pw = compute_portfolio_weights(mf, included=included)
                            st.warning("⚠️ Markowitz indisponible — poids rang MF appliqués")
                    else:
                        pw = compute_portfolio_weights(mf, included=included)
                st.session_state.pw         = pw
                st.session_state.pw_log     = [("Sélection manuelle",n_inc,included),
                                               ("Poids finaux",len(pw),pw.index.tolist())]
                st.session_state.pw_use_mkz = use_mkz_m
                st.session_state.pw_mode    = "manuel"
                st.success(f"✅ {len(pw)} titres · Σα = {pw.sum():.6f}")

        # ══════════════════════════════════════════════════════════
        # RÉSULTATS COMMUNS AUX DEUX MODES
        # ══════════════════════════════════════════════════════════
        if st.session_state.pw is not None:
            pw        = st.session_state.pw
            log       = st.session_state.get("pw_log",[])
            mode_used = st.session_state.get("pw_mode","")

            st.markdown("---")

            # Trace pipeline (mode auto uniquement)
            if log and mode_used == "auto":
                st.markdown("**🔍 Trace du pipeline**")
                log_cols  = st.columns(len(log))
                clr_steps = ["#475569","#3b82f6","#8b5cf6","#06b6d4","#10b981"]
                for i,(step,n_step,_) in enumerate(log):
                    with log_cols[i]:
                        st.markdown(
                            f"<div style='background:#161d2e;border:1px solid #1e2d45;"
                            f"border-radius:8px;padding:10px;text-align:center;'>"
                            f"<div style='font-size:10px;color:{clr_steps[i%len(clr_steps)]};font-weight:600;"
                            f"letter-spacing:.08em;text-transform:uppercase;'>{step}</div>"
                            f"<div style='font-size:24px;font-weight:800;color:#e2e8f0;'>{n_step}</div>"
                            f"<div style='font-size:10px;color:#64748b;'>titres</div></div>",
                            unsafe_allow_html=True)
                st.markdown("")

            # KPIs
            k1,k2,k3,k4,k5 = st.columns(5)
            k1.metric("Nb titres",         len(pw))
            k2.metric("Poids max",         f"{pw.max()*100:.2f}%")
            k3.metric("Poids min",         f"{pw.min()*100:.2f}%")
            k4.metric("HHI concentration", f"{(pw**2).sum():.4f}")
            k5.metric("Mode", "🤖 Auto" if mode_used=="auto" else "🖐️ Manuel")

            pl,pr = st.columns(2)
            with pl:
                st.markdown("**Répartition du portefeuille**")
                dpw = pw.copy()
                if len(dpw)>15:
                    dpw = pd.concat([dpw.head(15),pd.Series({"Autres":dpw.iloc[15:].sum()})])
                fig_p = go.Figure(go.Pie(
                    labels=dpw.index,values=dpw.values,hole=0.45,
                    textfont=dict(size=10),
                    marker=dict(line=dict(color="#0b0f1a",width=2)),
                    hovertemplate="<b>%{label}</b><br>%{percent:.2%}<extra></extra>"))
                fig_p.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",height=380,
                    font=dict(color="#94a3b8",family="JetBrains Mono"),
                    legend=dict(font=dict(size=10),bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=10,r=10,t=20,b=10),
                    annotations=[dict(text=f"<b>{len(pw)}</b><br>titres",
                        x=0.5,y=0.5,font_size=14,showarrow=False,
                        font=dict(color="#94a3b8"))])
                st.plotly_chart(fig_p,width="stretch")

            with pr:
                st.markdown("**Poids par titre**")
                pt = pw.head(25)
                fig_h = go.Figure(go.Bar(
                    x=pt.values*100,y=pt.index,orientation="h",
                    marker=dict(color=np.linspace(0.9,0.2,len(pt)),colorscale="Blues"),
                    text=[f"{v*100:.2f}%" for v in pt.values],
                    textposition="outside",textfont=dict(size=9,color="#94a3b8")))
                fig_h.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                    height=480,font=dict(color="#94a3b8",family="JetBrains Mono"),
                    margin=dict(l=10,r=80,t=20,b=30),
                    xaxis=dict(gridcolor="#1e2d45",ticksuffix="%"),
                    yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_h,width="stretch")

            # Table
            ranks_final = mf.reindex(pw.index).rank(ascending=False,method="min")
            alloc = pd.DataFrame({
                "Ticker":       pw.index,
                "Rang MF":      ranks_final.loc[pw.index].astype(int),
                "Score MF":     mf.loc[pw.index].round(6),
                "Poids α(T,t)": pw.values,
                "Poids (%)":    (pw.values*100).round(4),
            }).reset_index(drop=True)
            st.markdown("**Table d'allocation complète**")
            st.dataframe(
                alloc.style.format({"Score MF":"{:.6f}","Poids α(T,t)":"{:.6f}","Poids (%)":"{:.4f}%"})
                     .bar(subset=["Poids (%)"],color=["#1e3a5f","#3b82f6"]),
                width="stretch",hide_index=True)

            buf = io.BytesIO(); alloc.to_excel(buf,index=False)
            st.download_button("⬇️ Exporter le portefeuille (Excel)", buf.getvalue(),
                "portefeuille_BRVM.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # ── Portefeuille Charia ────────────────────────────────
            if CHARIA_AVAILABLE and st.session_state.charia_results:
                st.markdown("---")
                st.markdown(
                    "<span class='pill' style='background:#10b981;'>☪️</span>"
                    "<p class='sh'>Allocation Charia Compatible</p>",
                    unsafe_allow_html=True)
                st.markdown("""<div class='fbox'>
                Sous-ensemble du portefeuille filtré sur les titres ≥ 3/4 standards Charia<br>
                DJIM · FTSE · S&amp;P · AAOIFI — poids recalculés sur cet univers restreint
                </div>""", unsafe_allow_html=True)

                charia_tickers = get_charia_compatible_tickers(
                    st.session_state.charia_results)
                charia_in_ptf  = [t for t in pw.index if t in charia_tickers]

                ck1, ck2, ck3 = st.columns(3)
                ck1.metric("Compatibles Charia (BRVM)", len(charia_tickers))
                ck2.metric("Dans le portefeuille cible", len(charia_in_ptf))
                ck3.metric("Σα Charia", f"{compute_portfolio_weights(mf, included=charia_in_ptf).sum():.4f}"
                           if charia_in_ptf else "—")

                if charia_in_ptf:
                    pw_charia = compute_portfolio_weights(mf, included=charia_in_ptf)

                    ch_l, ch_r = st.columns(2)
                    with ch_l:
                        st.markdown("**Répartition Charia**")
                        dpwc = pw_charia.copy()
                        if len(dpwc) > 12:
                            dpwc = pd.concat([dpwc.head(12),
                                             pd.Series({"Autres": dpwc.iloc[12:].sum()})])
                        fig_pc = go.Figure(go.Pie(
                            labels=dpwc.index, values=dpwc.values, hole=0.45,
                            textfont=dict(size=10),
                            marker=dict(color=["#10b981"]*len(dpwc),
                                        line=dict(color="#0b0f1a", width=2)),
                            hovertemplate="<b>%{label}</b><br>%{percent:.2%}<extra></extra>"
                        ))
                        fig_pc.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", height=320,
                            font=dict(color="#94a3b8", family="JetBrains Mono"),
                            legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
                            margin=dict(l=10,r=10,t=20,b=10),
                            annotations=[dict(text=f"<b>{len(pw_charia)}</b><br>titres",
                                             x=0.5,y=0.5,font_size=13,showarrow=False,
                                             font=dict(color="#10b981"))]
                        )
                        st.plotly_chart(fig_pc, width="stretch")

                    with ch_r:
                        st.markdown("**Poids par titre**")
                        fig_hc = go.Figure(go.Bar(
                            x=pw_charia.values*100, y=pw_charia.index,
                            orientation="h",
                            marker=dict(color="#10b981", opacity=0.85),
                            text=[f"{v*100:.2f}%" for v in pw_charia.values],
                            textposition="outside",
                            textfont=dict(size=9, color="#94a3b8")
                        ))
                        fig_hc.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            height=360, font=dict(color="#94a3b8", family="JetBrains Mono"),
                            margin=dict(l=10,r=80,t=20,b=30),
                            xaxis=dict(gridcolor="#1e2d45", ticksuffix="%"),
                            yaxis=dict(autorange="reversed")
                        )
                        st.plotly_chart(fig_hc, width="stretch")

                    ranks_c = mf.reindex(pw_charia.index).rank(
                        ascending=False, method="min")
                    alloc_c = pd.DataFrame({
                        "Ticker":       pw_charia.index,
                        "Rang MF":      ranks_c.loc[pw_charia.index].astype(int),
                        "Score MF":     mf.loc[pw_charia.index].round(6),
                        "☪️ Charia":   [get_charia_label(t,st.session_state.charia_results)
                                         for t in pw_charia.index],
                        "Poids α(T,t)": pw_charia.values,
                        "Poids (%)":    (pw_charia.values*100).round(4),
                    }).reset_index(drop=True)

                    st.dataframe(
                        alloc_c.style.format({
                            "Score MF":     "{:.6f}",
                            "Poids α(T,t)": "{:.6f}",
                            "Poids (%)":    "{:.4f}%"
                        }), width="stretch", hide_index=True
                    )
                    buf_c = io.BytesIO(); alloc_c.to_excel(buf_c, index=False)
                    st.download_button("⬇️ Exporter portefeuille Charia (Excel)",
                        buf_c.getvalue(), "portefeuille_charia_BRVM.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.info("☪️ Aucun titre Charia compatible dans le portefeuille cible. "
                            "Élargissez les filtres ou chargez le fichier de screening.")


# ══ VALORISATION ═══════════════════════════════════════════════
with t8:
    st.markdown("<span class='pill'>Valorisation · Prix Cible</span><p class='sh'>Modèles de valorisation théorique</p>", unsafe_allow_html=True)
    st.markdown("""<div class='fbox'>
    DDM · P/E relatif · P/B · DCF simplifié → Prix cible combiné par titre<br>
    Chargez votre fichier d'états financiers · La base est persistée entre les sessions
    </div>""", unsafe_allow_html=True)

    if not VALUATION_AVAILABLE:
        st.error("❌ Modules fs_parser.py et valuation_models.py introuvables.")
        st.code(_VALUATION_ERROR if '_VALUATION_ERROR' in dir() else "Erreur inconnue")
        st.info("Vérifiez que fs_parser.py et valuation_models.py sont à la racine du dépôt GitHub (même niveau que app.py).")
        st.stop()

    # ── Chargement du fichier états financiers ─────────────────
    st.markdown("**📁 États Financiers**")
    col_up, col_db = st.columns([2, 1])

    with col_up:
        fs_file = st.file_uploader(
            "Charger Etats_Financiers_BRVM.xlsx (nouveau chargement = mise à jour cumulative)",
            type=["xlsx","xls"], key="fs_uploader"
        )
        if fs_file:
            with st.spinner("Parsing des états financiers..."):
                new_data = parse_financial_file(fs_file)
                nb_titres_tmp = st.session_state.data.get("nb_titres", {}) \
                                if st.session_state.data else {}
                cours_tmp     = st.session_state.data.get("cours", pd.DataFrame()) \
                                if st.session_state.data else pd.DataFrame()
                moy_tmp       = st.session_state.data.get("moyenne_cours", pd.DataFrame()) \
                                if st.session_state.data else pd.DataFrame()
                if nb_titres_tmp and not cours_tmp.empty:
                    from fs_parser import validate_and_fix_units
                    new_data, corrections = validate_and_fix_units(
                        new_data, nb_titres_tmp, cours_tmp
                    )
                    if corrections:
                        st.warning(
                            f"⚠️ Correction automatique d'unités : "
                            + ", ".join(f"{t} (PE {v['pe_avant']}x→{v['pe_apres']}x)"
                                        for t, v in corrections.items())
                        )
                existing = load_financial_db(VALUATION_DB_PATH)
                merged   = merge_financial_data(existing, new_data)
                save_financial_db(merged, VALUATION_DB_PATH)
                st.session_state.fin_data = merged

                # ── Screening Charia automatique depuis fin_data ──────
                if CHARIA_AVAILABLE and nb_titres_tmp:
                    charia_auto = screen_all_from_fin_data(
                        merged, nb_titres_tmp, moy_tmp
                    )
                    # Fusion avec screening existant (priorité au fichier statique)
                    existing_charia = st.session_state.charia_results
                    for t, r in charia_auto.items():
                        if t not in existing_charia:
                            existing_charia[t] = r
                    st.session_state.charia_results = existing_charia

            tickers_new = set(new_data.keys()) - set(existing.keys())
            years_new   = {yr for t in new_data.values() for yr in t.keys()}
            n_charia    = len(get_charia_compatible_tickers(
                              st.session_state.charia_results)) \
                          if CHARIA_AVAILABLE else 0
            st.success(
                f"✅ {len(new_data)} sociétés · Années : {sorted(years_new)} · "
                f"Base : {len(merged)} · ☪️ Charia compatibles : {n_charia}"
            )

    # ── Chargement fichier screening Charia statique ───────────
    if CHARIA_AVAILABLE:
        with st.expander("☪️ Charger le fichier de screening Charia (optionnel)", expanded=False):
            charia_file = st.file_uploader(
                "CHARIA_SCREENING_MODEL.xlsx — enrichit le screening automatique",
                type=["xlsx","xls"], key="charia_uploader"
            )
            if charia_file:
                with st.spinner("Parsing screening Charia..."):
                    charia_static = parse_charia(charia_file)
                    # Le fichier statique a priorité sur le calcul auto
                    merged_charia = {**st.session_state.charia_results, **charia_static}
                    st.session_state.charia_results = merged_charia
                n_ok = len(get_charia_compatible_tickers(merged_charia))
                st.success(f"✅ {len(charia_static)} tickers screenés · "
                           f"☪️ {n_ok} compatibles Charia")

    with col_db:
        fin_data = st.session_state.fin_data
        if fin_data:
            all_years = sorted({yr for t in fin_data.values() for yr in t.keys()})
            st.metric("Sociétés en base", len(fin_data))
            st.metric("Années disponibles", f"{min(all_years)}–{max(all_years)}")
            if os.path.exists(VALUATION_DB_PATH):
                db_data = json.load(open(VALUATION_DB_PATH))
                st.caption(f"Dernière mise à jour : {db_data.get('metadata',{}).get('last_updated','—')[:10]}")
        else:
            st.info("Aucune donnée en base. Chargez un fichier.")

    _has_fin_data = bool(fin_data)
    if not _has_fin_data:
        st.info("📂 Aucune donnée en base. Chargez votre fichier **Etats_Financiers_BRVM.xlsx** ci-dessus pour commencer.")

    if _has_fin_data:
        st.markdown("---")

        # ── Info calibrage automatique ─────────────────────────────
        st.markdown("""<div class='fbox'>
        🤖 <b>Calibrage 100% automatique</b> — Zéro saisie manuelle<br>
        β calculé depuis cours historiques · ke/WACC via CAPM · kd depuis états financiers<br>
        P/E et P/B calibrés sur les multiples historiques observés du titre · g_FCF = CAGR Rex/RN
        </div>""", unsafe_allow_html=True)

        st.markdown("**📅 Fenêtre de calcul du Beta (β)**")
        st.caption("Le beta mesure la sensibilité du titre au marché BRVM sur la période choisie. "
                   "Une fenêtre longue (3–5 ans) donne un beta stable · courte (6–12 mois) capture le beta récent.")

        c_min_v = data["cours"].index.min().date() if st.session_state.data else pd.Timestamp("2014-01-01").date()
        c_max_v = data["cours"].index.max().date() if st.session_state.data else pd.Timestamp("2026-01-01").date()

        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            beta_start = st.date_input(
                "Début fenêtre β", key="beta_start",
                value=max(c_min_v, min(c_max_v, pd.Timestamp("2022-01-01").date())),
                min_value=c_min_v, max_value=c_max_v,
                help="Date de début pour estimer le beta"
            )
        with bc2:
            beta_end = st.date_input(
                "Fin fenêtre β", key="beta_end",
                value=c_max_v,
                min_value=c_min_v, max_value=c_max_v,
                help="Date de fin pour estimer le beta (en général = aujourd'hui)"
            )
        with bc3:
            if beta_start < beta_end and st.session_state.data:
                cours_tmp  = st.session_state.data["cours"]
                mask_beta  = (cours_tmp.index >= pd.to_datetime(beta_start)) & \
                             (cours_tmp.index <= pd.to_datetime(beta_end))
                n_jours = mask_beta.sum()
                st.markdown("<br>", unsafe_allow_html=True)
                st.success(f"✅ **{n_jours}** jours de trading\n\n"
                           f"≈ {n_jours//252} an(s) {(n_jours%252)//21} mois")
            elif beta_start >= beta_end:
                st.markdown("<br>", unsafe_allow_html=True)
                st.error("⚠️ Date début ≥ Date fin")

        st.markdown("---")

        ac1, ac2 = st.columns(2)
        with ac1:
            dcf_horizon = st.slider(
                "Horizon DCF (ans)", 3, 10, 5, 1, key="dcf_h",
                help="Nombre d'années de projection explicite. La valeur terminale couvre l'infini."
            )
        with ac2:
            st.markdown("**Poids des modèles (renormalisés si modèle absent)**")
            wp1, wp2, wp3, wp4 = st.columns(4)
            w_ddm = wp1.number_input("DDM",  0.0,1.0,0.20,0.05,key="w_ddm")
            w_pe  = wp2.number_input("P/E",  0.0,1.0,0.30,0.05,key="w_pe")
            w_pb  = wp3.number_input("P/B",  0.0,1.0,0.20,0.05,key="w_pb")
            w_dcf = wp4.number_input("DCF",  0.0,1.0,0.30,0.05,key="w_dcf")
        model_weights = {"DDM": w_ddm, "P/E": w_pe, "P/B": w_pb, "DCF": w_dcf}

        st.markdown("---")

        # ── Sélection des titres ────────────────────────────────────
        av_tickers = sorted(fin_data.keys())
        if st.session_state.data:
            cours_tickers = st.session_state.data.get("tickers", [])
            common   = [t for t in av_tickers if t in cours_tickers]
            only_fs  = [t for t in av_tickers if t not in cours_tickers]
        else:
            common, only_fs = av_tickers, []

        sel_tickers = st.multiselect(
            "Titres à valoriser",
            options=av_tickers,
            default=common[:15] if len(common) >= 15 else common,
            help="Titres disponibles dans les états financiers chargés"
        )

        if st.button("📈 Calculer les valorisations", type="primary"):
            if not sel_tickers:
                st.warning("Sélectionnez au moins un titre.")
            else:
                nb_titres  = st.session_state.data.get("nb_titres",  {}) if st.session_state.data else {}
                dividendes = st.session_state.data.get("dividendes", pd.DataFrame()) if st.session_state.data else pd.DataFrame()
                cours_df   = st.session_state.data.get("cours",      pd.DataFrame()) if st.session_state.data else pd.DataFrame()
                moy_cours  = st.session_state.data.get("moyenne_cours", pd.DataFrame()) if st.session_state.data else pd.DataFrame()

                # Dividendes → {ticker: {year: montant}}
                div_hist = {}
                if not dividendes.empty and "Date" in dividendes.columns:
                    for _, row in dividendes.iterrows():
                        yr_d = int(row["Date"])
                        for t in sel_tickers:
                            if t in dividendes.columns:
                                v = row.get(t)
                                if pd.notna(v) and v > 0:
                                    div_hist.setdefault(t, {})[yr_d] = float(v)

                results_all = {}
                params_all  = {}
                with st.spinner(f"Calibrage auto + valorisation de {len(sel_tickers)} titres..."):
                    for ticker in sel_tickers:
                        # ── Calibrage auto de tous les paramètres ──────
                        try:
                            from valuation_models import calibrate_params
                            p = calibrate_params(
                                ticker, fin_data, nb_titres,
                                cours_df, moy_cours,
                                div_history=div_hist.get(ticker),
                                beta_date_start=beta_start,
                                beta_date_end=beta_end,
                            )
                        except Exception:
                            p = None
                        params_all[ticker] = p

                        res = {}
                        res["DDM"] = valuation_ddm(
                            ticker, fin_data, div_hist.get(ticker, {}),
                            nb_titres, cours_df, params=p
                        )
                        res["P/E"] = valuation_pe(
                            ticker, fin_data, nb_titres,
                            cours_df, moy_cours, params=p
                        )
                        res["P/B"] = valuation_pb(
                            ticker, fin_data, nb_titres,
                            cours_df, moy_cours, params=p
                        )
                        res["DCF"] = valuation_dcf(
                            ticker, fin_data, nb_titres,
                            cours_df, horizon=dcf_horizon, params=p
                        )
                        p_comb, prices_ok = combined_price(res, model_weights)
                        results_all[ticker] = {
                            "modeles":      res,
                            "prix_cible":   p_comb,
                            "prix_modeles": prices_ok,
                            "params":       p,
                        }

                st.session_state.val_results = results_all
                st.success(f"✅ Valorisation terminée — {len(results_all)} titres")

        # ── Affichage des résultats ────────────────────────────────
        if "val_results" in st.session_state and st.session_state.val_results:
            val      = st.session_state.val_results
            cours_df = st.session_state.data.get("cours", pd.DataFrame()) if st.session_state.data else pd.DataFrame()

            st.markdown("---")
            st.markdown("**📋 Tableau de synthèse — Prix cibles et potentiels**")

            rows = []
            for ticker, v in val.items():
                p_comb = v.get("prix_cible")
                if not p_comb:
                    continue

                # Cours actuel
                cours_act = np.nan
                if not cours_df.empty and ticker in cours_df.columns:
                    s = cours_df[ticker].dropna()
                    if not s.empty:
                        cours_act = float(s.iloc[-1])

                pot_num = potentiel if not np.isnan(potentiel) else 0
                charia_lbl = get_charia_label(ticker, st.session_state.charia_results) \
                             if CHARIA_AVAILABLE else "—"

                rows.append({
                    "Ticker":      ticker,
                    "Secteur":     sect,
                    "☪️ Charia":   charia_lbl,
                    "Cours actuel": f"{cours_act:,.0f}" if not np.isnan(cours_act) else "—",
                    "DDM":          f"{pm.get('DDM',0):,.0f}"  if "DDM" in pm  else "—",
                    "P/E":          f"{pm.get('P/E',0):,.0f}"  if "P/E" in pm  else "—",
                    "P/B":          f"{pm.get('P/B',0):,.0f}"  if "P/B" in pm  else "—",
                    "DCF":          f"{pm.get('DCF',0):,.0f}"  if "DCF" in pm  else "—",
                    "Prix cible":   f"{p_comb:,.0f}",
                    "Potentiel":    f"{potentiel:+.1f}%" if not np.isnan(potentiel) else "—",
                    "Signal":       signal,
                    "_pot_num":     pot_num,
                })

            if not rows:
                st.warning("Aucun résultat disponible — vérifiez que les nb_titres sont chargés.")
            else:
                df_res = pd.DataFrame(rows).sort_values("_pot_num", ascending=False)
                display_cols = ["Ticker","Secteur","☪️ Charia","Cours actuel",
                            "DDM","P/E","P/B","DCF","Prix cible","Potentiel","Signal"]
                st.dataframe(df_res[display_cols], width="stretch", hide_index=True)

                # Graphique potentiels
                st.markdown("**Potentiels de hausse / baisse par titre**")
                df_chart = df_res[df_res["_pot_num"] != 0].copy()
                colors_bar = ["#10b981" if v > 0 else "#ef4444" for v in df_chart["_pot_num"]]
                fig_pot = go.Figure(go.Bar(
                    x=df_chart["Ticker"],
                    y=df_chart["_pot_num"],
                    marker=dict(color=colors_bar, line=dict(width=0)),
                    text=[f"{v:+.1f}%" for v in df_chart["_pot_num"]],
                    textposition="outside",
                    textfont=dict(size=9, color="#94a3b8"),
                ))
                fig_pot.add_hline(y=0, line_width=1, line_color="#475569")
                fig_pot.add_hline(y=10,  line_dash="dash", line_color="#10b981", line_width=0.8)
                fig_pot.add_hline(y=-10, line_dash="dash", line_color="#ef4444", line_width=0.8)
                fig_pot.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8", family="JetBrains Mono"), height=380,
                    margin=dict(l=10, r=10, t=20, b=50),
                    xaxis=dict(gridcolor="#1e2d45", tickangle=-45),
                    yaxis=dict(gridcolor="#1e2d45", ticksuffix="%", title="Potentiel (%)"),
                )
                st.plotly_chart(fig_pot, width="stretch")

                # Répartition des signaux par secteur
                if SECTOR_MAP:
                    st.markdown("**📊 Potentiel moyen par secteur**")
                    df_sect = df_res[df_res["_pot_num"] != 0].copy()
                    df_sect["Secteur_map"] = df_sect["Ticker"].apply(
                        lambda t: SECTOR_MAP.get(t.upper(), "Autre")
                    )
                    sect_avg = df_sect.groupby("Secteur_map")["_pot_num"].mean().sort_values(ascending=False)
                    fig_sect = go.Figure(go.Bar(
                        x=sect_avg.index,
                        y=sect_avg.values,
                        marker=dict(
                            color=["#10b981" if v > 0 else "#ef4444" for v in sect_avg.values],
                            line=dict(width=0)
                        ),
                        text=[f"{v:+.1f}%" for v in sect_avg.values],
                        textposition="outside",
                        textfont=dict(size=10, color="#94a3b8"),
                    ))
                    fig_sect.add_hline(y=0, line_width=1, line_color="#475569")
                    fig_sect.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94a3b8", family="JetBrains Mono"), height=320,
                        margin=dict(l=10, r=10, t=20, b=80),
                        xaxis=dict(gridcolor="#1e2d45", tickangle=-30),
                        yaxis=dict(gridcolor="#1e2d45", ticksuffix="%"),
                    )
                    st.plotly_chart(fig_sect, width="stretch")

                # Tableau transparence des paramètres calibrés
                st.markdown("---")
                st.markdown("**🔬 Transparence — Paramètres calibrés automatiquement**")
                st.caption("Tous dérivés des données réelles · aucune saisie manuelle")
                param_rows = []
                for ticker_p, v_p in val.items():
                    p = v_p.get("params")
                    if not p:
                        continue
                    param_rows.append({
                        "Ticker":    ticker_p,
                        "Secteur":   SECTOR_MAP.get(ticker_p.upper(), p.get("secteur","—")),
                        "Type":      "Banque" if p.get("is_bank") else "Société",
                        "β":         f"{p.get('beta',1):.2f}",
                        "ke":        f"{p.get('ke',0)*100:.1f}%",
                        "kd":        f"{p.get('kd',0)*100:.1f}%",
                        "WACC":      f"{p.get('wacc',0)*100:.1f}%",
                        "g_FCF":     f"{p.get('g_fcf',0)*100:.1f}%",
                        "g_div":     f"{p.get('g_div',0)*100:.1f}%",
                        "P/E cible": f"{p.get('pe_target',0):.1f}x",
                        "P/E méth.": p.get("pe_method","—"),
                        "P/B cible": f"{p.get('pb_target',0):.2f}x",
                        "Taux IS":   f"{p.get('tax_rate',0)*100:.1f}%",
                        "Année":     str(p.get("annee_ref","—")),
                    })
                if param_rows:
                    st.dataframe(pd.DataFrame(param_rows), width="stretch", hide_index=True)

                # Détail par titre
                st.markdown("**🔍 Détail par titre — DCF**")
                ticker_detail = st.selectbox("Choisir un titre", [r["Ticker"] for r in rows])
                if ticker_detail and ticker_detail in val:
                    dcf_res = val[ticker_detail]["modeles"].get("DCF")
                    if dcf_res and dcf_res.get("fcf_table"):
                        st.markdown(f"**{ticker_detail} — Projection DCF** · WACC={dcf_res['wacc']:.2%} · g_FCF={dcf_res['g_fcf']:.2%} · g_TV={dcf_res['g_tv']:.2%}")
                        dcf_df = pd.DataFrame(dcf_res["fcf_table"])
                        dcf_df.columns = ["Année","FCF projeté (M FCFA)","PV (M FCFA)"]
                        dcf_df = dcf_df.round(0)

                        col_dcf1, col_dcf2 = st.columns(2)
                        with col_dcf1:
                            st.dataframe(dcf_df, width="stretch", hide_index=True)
                        with col_dcf2:
                            kd1, kd2, kd3 = st.columns(3)
                            kd1.metric("PV FCF (M FCFA)", f"{dcf_res['pv_fcf_m']:,.0f}")
                            kd2.metric("PV TV (M FCFA)", f"{dcf_res['pv_tv_m']:,.0f}")
                            kd3.metric("EV (M FCFA)", f"{dcf_res['ev_m']:,.0f}")
                            st.metric("Prix cible DCF (FCFA)", f"{dcf_res['prix_cible']:,.0f}")
                    else:
                        st.info("DCF non disponible pour ce titre.")

                # Export
                buf = io.BytesIO()
                df_res[display_cols].to_excel(buf, index=False)
                st.download_button("⬇️ Exporter valorisations (Excel)", buf.getvalue(),
                                   "valorisation_BRVM.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══ DONNÉES ════════════════════════════════════════════════════
with t9:
    st.markdown("<span class='pill'>Sources</span><p class='sh'>Aperçu des données chargées</p>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Observations cours", len(data["cours"]))
    c2.metric("Première date", str(data["cours"].index[0].date()))
    c3.metric("Dernière date",  str(data["cours"].index[-1].date()))
    st.markdown(f"**{len(data['tickers'])} titres :** {', '.join(data['tickers'])}")
    st.markdown("---")

    ti1, ti2, ti3 = st.tabs(["📈 Cours historiques", "💸 Dividendes", "🏢 Résultats nets"])
    with ti1:
        sel = st.multiselect("Titres", data["tickers"], default=data["tickers"][:6])
        if sel:
            fig_c = go.Figure()
            for t in sel:
                s = data["cours"][t].dropna()
                fig_c.add_trace(go.Scatter(x=s.index, y=s, name=t, mode="lines", line=dict(width=1.5)))
            fig_c.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 font=dict(color="#94a3b8",family="JetBrains Mono"), height=400,
                                 legend=dict(bgcolor="rgba(0,0,0,0)"),
                                 xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"),
                                 hovermode="x unified")
            st.plotly_chart(fig_c, width="stretch")
    with ti2:
        st.dataframe(data["dividendes"].set_index("Date"), width="stretch")
    with ti3:
        rn = data["resultat_net"]
        rows2 = [{"Ticker":t,"Année":y,"Résultat net (MFCFA)":v/1e6}
                 for t,yrs in rn.items() for y,v in yrs.items()]
        if rows2:
            df_rn = pd.DataFrame(rows2).pivot(index="Ticker",columns="Année",values="Résultat net (MFCFA)")
            st.dataframe(df_rn.style.format("{:,.0f}"), width="stretch")
