import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import warnings
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
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
    nb_raw = pd.read_excel(xl, sheet_name="Nb_titres")
    nb_titres, nb_flottant = {}, {}
    for _, row in nb_raw.iterrows():
        label = str(row.get('Date', '')).strip().lower()
        for t in tickers:
            if t in nb_raw.columns:
                v = row.get(t, np.nan)
                if pd.notna(v):
                    if 'flottant' in label:
                        nb_flottant[t] = float(v)
                    elif 'nombre' in label:
                        nb_titres[t] = float(v)

    # TABLEAU DE BORD
    tb = pd.read_excel(xl, sheet_name="Tableau de bord")
    tb_tickers = [c for c in tickers if c in tb.columns]

    def extract_metric(label):
        result = {}
        header = None
        for idx, row in tb.iterrows():
            val = str(row.get(tb.columns[2], '')).strip().upper()
            if label.upper() in val:
                header = idx
                break
        if header is None:
            return result
        for idx2 in range(header + 1, min(header + 8, len(tb))):
            row = tb.iloc[idx2]
            yr = row.get(tb.columns[1], np.nan)
            if pd.isna(yr):
                break
            year = int(yr)
            for t in tb_tickers:
                v = row.get(t, np.nan)
                if pd.notna(v):
                    result.setdefault(t, {})[year] = float(v)
        return result

    return {
        "cours": cours, "volumes": vol, "dividendes": div,
        "moyenne_cours": moy, "nb_titres": nb_titres, "nb_flottant": nb_flottant,
        "tickers": tickers,
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

    df      = pd.DataFrame(detail).T
    metrics = ["B/P  (CP/n÷cours)", "E/P  (RN/n÷cours)",
               "FCF/P (FCF/n÷cours)", "CA/P  (CA/n÷cours)",
               "EBIT/P (Rex/n÷cours)"]

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
    return res.dropna(subset=["Score Value"]).sort_values("Score Value", ascending=False)


def compute_momentum_factor(data, end_date, weights):
    """
    F_mom(t,T) = Σ w_h * rdt_h(t,T) / max_{E_t}(rdt_h(t,T))
    6 horizons : J · 5J · 21J · 63J · 126J · 252J
    """
    returns = data["cours"].apply(pd.to_numeric, errors='coerce').pct_change().dropna(how='all')
    try:
        ed = pd.to_datetime(end_date)
    except Exception:
        ed = returns.index[-1]
    sub = returns[returns.index <= ed]

    horizons = {
        "Journalier":  1,
        "Hebdo":       5,
        "Mensuel":     21,
        "Trimestriel": 63,
        "Semestriel":  126,
        "Annuel":      252,
    }
    # m_ij = rendement moyen sur l'horizon h pour le titre T
    mom_df = pd.DataFrame(
        {h: sub.tail(n).mean(numeric_only=True) for h, n in horizons.items()}
    ).replace([np.inf, -np.inf], np.nan)

    # Normalisation standard : m_ij / max_{E_t}(m_ij)
    norm_df = pd.DataFrame({h: _normalise_standard(mom_df[h]) for h in horizons})

    # Score pondéré
    score = sum(norm_df[h].fillna(0) * weights.get(h, 1/6) for h in horizons)

    res = pd.DataFrame({
        "Score Momentum": score,
        **{f"Rdt {h}": mom_df[h] for h in horizons},
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
        avail_years = sorted(data["moyenne_cours"]["Date"].dropna().astype(int).unique(), reverse=True)
        selected_year = st.selectbox("📅 Année (Value / Dividende)", avail_years)
        date_ref = st.date_input("📅 Date ref. (Momentum / Vol.)", value=data["cours"].index[-1].date())
        st.markdown("---")
        st.markdown("**⚖️ Poids des facteurs β_i**")
        st.caption("Calibrez chaque facteur · Σβ doit = 1.0")

        b_val = st.slider("💰 Value",      0.0, 1.0,
                          st.session_state.get("b_val", 0.20), 0.01, key="b_val")
        b_mom = st.slider("🚀 Momentum",   0.0, 1.0,
                          st.session_state.get("b_mom", 0.20), 0.01, key="b_mom")
        b_vol = st.slider("📉 Volatilité", 0.0, 1.0,
                          st.session_state.get("b_vol", 0.20), 0.01, key="b_vol")
        b_div = st.slider("💸 Dividende",  0.0, 1.0,
                          st.session_state.get("b_div", 0.20), 0.01, key="b_div")
        b_liq = st.slider("💧 Liquidité",  0.0, 1.0,
                          st.session_state.get("b_liq", 0.20), 0.01, key="b_liq")

        bs = round(b_val + b_mom + b_vol + b_div + b_liq, 4)
        if abs(bs - 1.0) <= 0.01:
            st.success(f"✅ Σβ = {bs:.2f}")
        else:
            st.warning(f"⚠️ Σβ = {bs:.2f} ≠ 1.0")

        if st.button("⚖️ Égaliser (20% chacun)"):
            for k in ["b_val","b_mom","b_vol","b_div","b_liq"]:
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
t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
    "💰 Value", "🚀 Momentum", "📉 Volatilité", "💸 Dividende",
    "💧 Liquidité", "🔢 Indice MF", "📂 Portefeuille", "ℹ️ Données"])

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
        v_date_start = st.date_input("Date début", value=pd.Timestamp(f"{selected_year}-01-01").date(),
                                      min_value=cours_min, max_value=cours_max, key="v_d1")
    with vd2:
        v_date_end   = st.date_input("Date fin",   value=pd.Timestamp(f"{selected_year}-12-31").date(),
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
    st.markdown("""<div class='fbox'>F_mom = Σ w_h · rdt_moyen_h / max(rdt_moyen_h) &nbsp;·&nbsp; h ∈ {J, 5J, 21J, 63J, 126J, 252J}</div>""", unsafe_allow_html=True)

    hl = ["Journalier","Hebdo","Mensuel","Trimestriel","Semestriel","Annuel"]
    mc = st.columns(6)
    w_mom = {h: mc[i].number_input(h, 0.0, 1.0, round(1/6,4), 0.01, key=f"wm{i}") for i,h in enumerate(hl)}

    if st.button("⚙️ Calculer le Momentum", type="primary"):
        res = compute_momentum_factor(data, str(date_ref), w_mom)
        fr["Momentum"] = res
        st.success(f"✅ {len(res)} titres scorés")

    if "Momentum" in fr:
        res = fr["Momentum"]
        m1,m2,m3 = st.columns(3)
        m1.metric("Titres", len(res)); m2.metric("N°1", res.index[0]); m3.metric("Score max", f"{res['Score Momentum'].max():.4f}")
        st.plotly_chart(score_bar(res["Score Momentum"], "#8b5cf6"), width="stretch")
        rdt_c = [c for c in res.columns if "Rdt" in c]
        disp2 = res[["Score Momentum"]+rdt_c].head(20).reset_index().rename(columns={"index":"Ticker"})
        st.dataframe(disp2.style.format({"Score Momentum":"{:.4f}"} | {c:"{:.4%}" for c in rdt_c}),
                     width="stretch", hide_index=True)

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
        m2.metric("σ min", f"{res['Écart-type'].min():.4%}")
        m3.metric("σ moy", f"{res['Écart-type'].mean():.4%}")

        l,r = st.columns(2)
        with l:
            st.markdown("**Score (inversé) — Top 25**")
            st.plotly_chart(score_bar(res["Score Volatilité"], "#ef4444"), width="stretch")
        with r:
            st.markdown("**Volatilité annualisée (σ×√252)**")
            rv = res.sort_values("Écart-type")
            fig2 = go.Figure(go.Bar(x=rv.index, y=rv["Écart-type"]*np.sqrt(252),
                                    marker=dict(color=rv["Écart-type"]*np.sqrt(252),
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
        d_date_start = st.date_input("Cours moyen — Date début",
                                      value=pd.Timestamp(f"{div_years[0]}-01-01").date(),
                                      min_value=data["cours"].index.min().date(),
                                      max_value=data["cours"].index.max().date(), key="dd1")
    with dd3:
        d_date_end   = st.date_input("Cours moyen — Date fin",
                                      value=pd.Timestamp(f"{div_years[0]}-12-31").date(),
                                      min_value=data["cours"].index.min().date(),
                                      max_value=data["cours"].index.max().date(), key="dd2")

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
    Calibrez les β_i dans la barre latérale ← puis cliquez Calculer</div>""", unsafe_allow_html=True)

    # Lire les betas depuis session_state (toujours à jour)
    betas_mf = st.session_state.get("betas",
               {"Value":0.20,"Momentum":0.20,"Volatilité":0.20,"Dividende":0.20,"Liquidité":0.20})

    computed = list(fr.keys())

    # Affichage live des β actuels
    st.markdown("**β actifs (modifiables dans la barre latérale ←)**")
    CLRS = {"Value":"#3b82f6","Momentum":"#8b5cf6","Volatilité":"#ef4444",
            "Dividende":"#f59e0b","Liquidité":"#06b6d4"}
    ICONS = {"Value":"💰","Momentum":"🚀","Volatilité":"📉","Dividende":"💸","Liquidité":"💧"}
    bc = st.columns(5)
    for i, (fname, beta) in enumerate(betas_mf.items()):
        status = "✓ calculé" if fname in computed else "✗ non calculé"
        bc[i].metric(
            label=f"{ICONS.get(fname,'')} β {fname}",
            value=f"{beta:.2f}",
            delta=status
        )

    bs_mf = round(sum(betas_mf.values()), 4)
    if abs(bs_mf - 1.0) > 0.01:
        st.warning(f"⚠️ Σβ = {bs_mf:.2f} ≠ 1.0 — ajustez les curseurs dans la barre latérale")

    st.markdown("---")

    if not computed:
        st.info("👈 Calculez au moins un facteur (onglets 💰 🚀 📉 💸 💧) avant de continuer.")
    else:
        st.success(f"Facteurs calculés : {', '.join(computed)}")

        if st.button("🔢 Calculer l'Indice MF", type="primary"):
            mf = compute_multifactor(fr, betas_mf)
            st.session_state.mf_scores = mf
            st.success(f"✅ {len(mf)} titres classés · β = {betas_mf}")

        if st.session_state.mf_scores is not None:
            mf = st.session_state.mf_scores
            st.markdown("---")
            st.plotly_chart(score_bar(mf, "#3b82f6", height=380), width="stretch")

            # Décomposition factorielle
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
            fig_s.update_layout(
                barmode="stack", **{**PLOT_LAYOUT, "height": 340,
                "legend": dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
                "margin": dict(l=10, r=10, t=50, b=50)}
            )
            st.plotly_chart(fig_s, width="stretch")

            tbl = mf.reset_index()
            tbl.columns = ["Ticker", "Score MF"]
            tbl.insert(0, "Rang", range(1, len(tbl) + 1))
            tbl["Score MF"] = tbl["Score MF"].round(6)
            st.dataframe(tbl, width="stretch", hide_index=True)

            buf = io.BytesIO()
            tbl.to_excel(buf, index=False)
            st.download_button("⬇️ Exporter classement MF", buf.getvalue(),
                               "classement_MF.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══ PORTEFEUILLE ═══════════════════════════════════════════════
with t7:
    st.markdown("<span class='pill'>Étape 3</span><p class='sh'>Construction du Portefeuille Cible</p>", unsafe_allow_html=True)
    st.markdown("""<div class='fbox'>
    Pipeline d'optimisation séquentiel :<br>
    Univers complet → ① Liquidité → ② Score MF → ③ Corrélation → ④ Poids optimaux<br>
    Pondération finale : α(T,t) = (n−r+1)/(n(n+1)/2) &nbsp;ou&nbsp; Markowitz MF-augmenté
    </div>""", unsafe_allow_html=True)

    if st.session_state.mf_scores is None:
        st.info("👈 Calculez d'abord l'Indice MF (onglet 🔢).")
    else:
        mf = st.session_state.mf_scores
        all_tickers = mf.index.tolist()

        # ── Sélection manuelle (override) ─────────────────────────
        with st.expander("🔧 Sélection manuelle des titres (optionnel — écrase les filtres)", expanded=False):
            manual_mode = st.checkbox("Activer la sélection manuelle", value=False)
            if manual_mode:
                manual_included = st.multiselect(
                    "✅ Titres à inclure",
                    options=all_tickers,
                    default=all_tickers,
                )

        st.markdown("---")

        # ── Paramètres des filtres ─────────────────────────────────
        st.markdown("**⚙️ Paramètres du pipeline d'optimisation**")

        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            st.markdown("**① Filtre Liquidité**")
            min_vol_pct = st.slider(
                "Volume moyen ≥ X% du max",
                0, 50, 10, 1,
                help="0% = pas de filtre · 20% = garde les titres avec au moins 20% du volume du titre le plus liquide"
            )
            liq_ok = "Liquidité" in fr
            if not liq_ok:
                st.caption("⚠️ Calculez le facteur Liquidité pour activer ce filtre")

        with col_f2:
            st.markdown("**② Filtre Score MF**")
            top_pct = st.slider(
                "Garder le top X% des scores MF",
                10, 100, 60, 5,
                help="60% = garde les 60% de titres avec les meilleurs scores MF"
            )

        with col_f3:
            st.markdown("**③ Filtre Corrélation**")
            max_corr = st.slider(
                "Corrélation max entre titres",
                0.3, 1.0, 0.75, 0.05,
                help="0.75 = deux titres corrélés à plus de 75% → on garde le meilleur"
            )
            window_corr = st.number_input("Fenêtre (jours)", 60, 504, 252, 21,
                                          key="win_corr")

        st.markdown("---")
        st.markdown("**④ Pondération finale**")
        use_mkz = st.toggle("Optimisation Markowitz MF-augmentée", value=False,
                             help="OFF = poids par rang MF (formule de la note technique)\nON = optimisation Mean-Variance avec score MF intégré dans l'utilité")

        if use_mkz:
            mk1, mk2, mk3, mk4 = st.columns(4)
            mf_weight     = mk1.slider("Poids Score MF dans l'utilité",  0.0, 1.0, 0.50, 0.05,
                                        help="0 = pure minimisation variance · 1 = pure maximisation score MF")
            risk_aversion = mk2.slider("Aversion au risque λ",           0.1, 10.0, 2.0, 0.1)
            min_w         = mk3.slider("Poids min par titre (%)",         0, 20, 0, 1) / 100
            max_w         = mk4.slider("Poids max par titre (%)",         5, 100, 40, 5) / 100
        else:
            mf_weight, risk_aversion, min_w, max_w = 0.5, 2.0, 0.0, 1.0

        st.markdown("---")

        if st.button("🚀 Lancer l'optimisation", type="primary"):
            with st.spinner("Optimisation en cours..."):
                if manual_mode:
                    # Mode manuel : bypass des filtres
                    inc = manual_included if manual_included else all_tickers
                    if use_mkz:
                        pw = optimize_markowitz(
                            mf, inc, data["cours"],
                            window=int(window_corr),
                            risk_aversion=risk_aversion,
                            mf_weight=mf_weight,
                            min_w=min_w, max_w=max_w
                        )
                        if pw is None:
                            pw = compute_portfolio_weights(mf, included=inc)
                    else:
                        pw = compute_portfolio_weights(mf, included=inc)
                    pipeline_log = [("Sélection manuelle", len(inc), inc),
                                    ("Poids optimaux", len(pw), pw.index.tolist())]
                else:
                    inc, pw, pipeline_log = run_optimization_pipeline(
                        mf_scores     = mf,
                        data          = data,
                        factor_results= fr,
                        min_vol_pct   = min_vol_pct if liq_ok else 0,
                        top_pct       = top_pct,
                        max_corr      = max_corr,
                        use_markowitz = use_mkz,
                        risk_aversion = risk_aversion,
                        mf_weight     = mf_weight,
                        min_w         = min_w,
                        max_w         = max_w,
                        window        = int(window_corr),
                    )

            if pw is None or len(pw) == 0:
                st.error("❌ Le pipeline n'a retenu aucun titre. Élargissez les seuils des filtres.")
            else:
                st.session_state.pw        = pw
                st.session_state.pw_log    = pipeline_log
                st.session_state.pw_use_mkz= use_mkz
                st.success(f"✅ Portefeuille optimal : {len(pw)} titres · Σα = {pw.sum():.6f}")

        # ── Résultats ──────────────────────────────────────────────
        if st.session_state.pw is not None:
            pw  = st.session_state.pw
            log = st.session_state.get("pw_log", [])

            # Pipeline log
            if log:
                st.markdown("**🔍 Trace du pipeline**")
                log_cols = st.columns(len(log))
                colors_log = ["#475569","#3b82f6","#8b5cf6","#06b6d4","#10b981"]
                for i, (step, n_step, _) in enumerate(log):
                    with log_cols[i]:
                        st.markdown(
                            f"<div style='background:#161d2e;border:1px solid #1e2d45;"
                            f"border-radius:8px;padding:10px;text-align:center;'>"
                            f"<div style='font-size:10px;color:{colors_log[i % len(colors_log)]};font-weight:600;"
                            f"letter-spacing:.08em;text-transform:uppercase;'>{step}</div>"
                            f"<div style='font-size:24px;font-weight:800;color:#e2e8f0;'>{n_step}</div>"
                            f"<div style='font-size:10px;color:#64748b;'>titres</div></div>",
                            unsafe_allow_html=True
                        )

            st.markdown("---")

            # KPIs
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Nb titres",         len(pw))
            k2.metric("Poids max",         f"{pw.max()*100:.2f}%")
            k3.metric("Poids min",         f"{pw.min()*100:.2f}%")
            k4.metric("HHI concentration", f"{(pw**2).sum():.4f}")
            k5.metric("Méthode",
                      "Markowitz" if st.session_state.get("pw_use_mkz") else "Rang MF")

            pl, pr = st.columns(2)
            with pl:
                st.markdown("**Répartition du portefeuille**")
                dpw = pw.copy()
                if len(dpw) > 15:
                    dpw = pd.concat([dpw.head(15),
                                     pd.Series({"Autres": dpw.iloc[15:].sum()})])
                fig_p = go.Figure(go.Pie(
                    labels=dpw.index, values=dpw.values, hole=0.45,
                    textfont=dict(size=10),
                    marker=dict(line=dict(color="#0b0f1a", width=2)),
                    hovertemplate="<b>%{label}</b><br>%{percent:.2%}<extra></extra>"
                ))
                fig_p.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", height=380,
                    font=dict(color="#94a3b8", family="JetBrains Mono"),
                    legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=10, r=10, t=20, b=10),
                    annotations=[dict(text=f"<b>{len(pw)}</b><br>titres",
                                      x=0.5, y=0.5, font_size=14, showarrow=False,
                                      font=dict(color="#94a3b8"))]
                )
                st.plotly_chart(fig_p, width="stretch")

            with pr:
                st.markdown("**Poids par titre**")
                pt = pw.head(25)
                fig_h = go.Figure(go.Bar(
                    x=pt.values * 100, y=pt.index, orientation="h",
                    marker=dict(color=np.linspace(0.9, 0.2, len(pt)),
                                colorscale="Blues"),
                    text=[f"{v*100:.2f}%" for v in pt.values],
                    textposition="outside",
                    textfont=dict(size=9, color="#94a3b8")
                ))
                fig_h.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=480, font=dict(color="#94a3b8", family="JetBrains Mono"),
                    margin=dict(l=10, r=80, t=20, b=30),
                    xaxis=dict(gridcolor="#1e2d45", ticksuffix="%"),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_h, width="stretch")

            # Table d'allocation
            ranks_final = mf.reindex(pw.index).rank(ascending=False, method='min')
            alloc = pd.DataFrame({
                "Ticker":       pw.index,
                "Rang MF":      ranks_final.loc[pw.index].astype(int),
                "Score MF":     mf.loc[pw.index].round(6),
                "Poids α(T,t)": pw.values,
                "Poids (%)":    (pw.values * 100).round(4),
            }).reset_index(drop=True)

            st.markdown("**Table d'allocation complète**")
            st.dataframe(
                alloc.style.format({
                    "Score MF":     "{:.6f}",
                    "Poids α(T,t)": "{:.6f}",
                    "Poids (%)":    "{:.4f}%"
                }).bar(subset=["Poids (%)"], color=["#1e3a5f", "#3b82f6"]),
                width="stretch", hide_index=True
            )

            buf = io.BytesIO()
            alloc.to_excel(buf, index=False)
            st.download_button(
                "⬇️ Exporter le portefeuille (Excel)", buf.getvalue(),
                "portefeuille_optimal_BRVM.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# ══ DONNÉES ════════════════════════════════════════════════════
with t8:
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
