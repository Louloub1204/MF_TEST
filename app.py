import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CGF Gestion · BRVM Multifactorial",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  — refined dark finance aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg:        #0b0f1a;
    --surface:   #111827;
    --card:      #161d2e;
    --border:    #1e2d45;
    --accent:    #3b82f6;
    --accent2:   #06b6d4;
    --gold:      #f59e0b;
    --green:     #10b981;
    --red:       #ef4444;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

html, body, [class*="css"] {
    background-color: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: var(--font-mono) !important; }

/* Headers */
h1, h2, h3 { font-family: var(--font-head) !important; letter-spacing: -0.02em; }

/* Metric boxes */
[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.1em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--accent) !important; font-family: var(--font-mono) !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em;
    transition: all .2s;
}
.stButton > button:hover { background: var(--accent2) !important; transform: translateY(-1px); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--surface); border-bottom: 1px solid var(--border); gap: 0; }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; font-family: var(--font-mono) !important; font-size: 13px; padding: 10px 20px; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }

/* Slider */
.stSlider [data-baseweb="slider"] div { background: var(--accent) !important; }

/* Input */
.stNumberInput input, .stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    border-radius: 6px !important;
}

/* Select */
.stSelectbox [data-baseweb="select"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
}

/* Info/warning boxes */
.stInfo { background: rgba(59,130,246,0.1) !important; border-left: 3px solid var(--accent) !important; }
.stSuccess { background: rgba(16,185,129,0.1) !important; border-left: 3px solid var(--green) !important; }
.stWarning { background: rgba(245,158,11,0.1) !important; border-left: 3px solid var(--gold) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Section pill badge */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: rgba(59,130,246,0.15);
    color: var(--accent);
    border: 1px solid rgba(59,130,246,0.3);
    margin-bottom: 8px;
}
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 4px 0;
}
.section-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #64748b;
    margin-bottom: 20px;
}
.formula-box {
    background: #0d1526;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #93c5fd;
}
.rank-badge {
    display: inline-block;
    width: 28px; height: 28px;
    border-radius: 50%;
    background: var(--accent);
    color: white;
    font-size: 12px;
    font-weight: 700;
    text-align: center;
    line-height: 28px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS — factor definitions from the note
# ─────────────────────────────────────────────

FACTOR_DEFINITIONS = {
    "Value": {
        "metrics": ["Book-to-Price (B/P)", "EPS/Price (P/E inv.)", "FCF/Price", "Sales/Price", "EBIT/EV"],
        "weights": [0.10, 0.50, 0.20, 0.10, 0.10],
        "invert": False,
        "color": "#3b82f6",
        "icon": "💰",
        "description": "Valorisation relative au cours de marché",
    },
    "Qualité": {
        "metrics": ["Levier financier", "Ratio qualité résultat", "ROA", "ROE"],
        "weights": [0.25, 0.25, 0.25, 0.25],
        "invert": False,
        "color": "#10b981",
        "icon": "🏅",
        "description": "Solidité des fondamentaux",
    },
    "Dividende": {
        "metrics": ["Dividend Yield"],
        "weights": [1.0],
        "invert": False,
        "color": "#f59e0b",
        "icon": "💸",
        "description": "Rendement du dividende",
    },
    "Volatilité": {
        "metrics": ["Écart-type des rendements"],
        "weights": [1.0],
        "invert": True,
        "color": "#ef4444",
        "icon": "📉",
        "description": "Faible volatilité (min/métrique)",
    },
    "Momentum": {
        "metrics": ["Rdt moyen journalier", "Rdt moyen hebdo", "Rdt moyen mensuel",
                    "Rdt moyen trimestriel", "Rdt moyen semestriel", "Rdt moyen annuel"],
        "weights": [1/6]*6,
        "invert": False,
        "color": "#8b5cf6",
        "icon": "🚀",
        "description": "Dynamique des cours (6 horizons)",
    },
    "Liquidité": {
        "metrics": ["Volume moyen transigé"],
        "weights": [1.0],
        "invert": False,
        "color": "#06b6d4",
        "icon": "💧",
        "description": "Facilité de transaction",
    },
    "Taille": {
        "metrics": ["Capitalisation boursière"],
        "weights": [1.0],
        "invert": False,
        "color": "#ec4899",
        "icon": "🏢",
        "description": "Taille de l'entreprise",
    },
}

FACTOR_NAMES = list(FACTOR_DEFINITIONS.keys())


# ─────────────────────────────────────────────
# CORE COMPUTATION FUNCTIONS  (per technical note)
# ─────────────────────────────────────────────

def compute_factor_index(df: pd.DataFrame, factor: str) -> pd.Series:
    """
    Étape 1 — Compute factoriel index F_i(t, T) for each security T.

    Standard:  F_i = Σ w_ij * m_ij / max(m_ij)
    Volatility: F_i = Σ w_ij * min(m_ij) / m_ij
    """
    defn = FACTOR_DEFINITIONS[factor]
    metrics = defn["metrics"]
    weights = defn["weights"]
    invert = defn["invert"]

    scores = pd.Series(0.0, index=df.index)
    for metric, weight in zip(metrics, weights):
        if metric not in df.columns:
            continue
        col = df[metric].astype(float)
        if invert:
            min_val = col.min()
            # avoid div by zero
            factor_score = weight * (min_val / col.replace(0, np.nan)).fillna(0)
        else:
            max_val = col.max()
            factor_score = weight * (col / max_val if max_val != 0 else 0)
        scores += factor_score
    return scores.clip(lower=0)


def compute_multifactor_index(factor_scores: pd.DataFrame, betas: dict) -> pd.Series:
    """
    Étape 2 — MF(t,T) = Σ β_i * F_i(t,T)
    """
    mf = pd.Series(0.0, index=factor_scores.index)
    for factor, beta in betas.items():
        if factor in factor_scores.columns:
            mf += beta * factor_scores[factor]
    return mf


def compute_portfolio_weights(mf_scores: pd.Series, excluded: list = None) -> pd.Series:
    """
    Étape 3 — α(T,t) = (n - r(T,t) + 1) / (n(n+1)/2)
    Rank 1 = best score → highest weight.
    """
    if excluded:
        mf_scores = mf_scores.drop(labels=excluded, errors="ignore")

    n = len(mf_scores)
    if n == 0:
        return pd.Series(dtype=float)

    # rank: 1 = highest MF score
    ranks = mf_scores.rank(ascending=False, method="min")
    weights = (n - ranks + 1) / (n * (n + 1) / 2)
    return weights.sort_values(ascending=False)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "securities_df" not in st.session_state:
    st.session_state.securities_df = None
if "factor_scores" not in st.session_state:
    st.session_state.factor_scores = None
if "mf_scores" not in st.session_state:
    st.session_state.mf_scores = None
if "portfolio_weights" not in st.session_state:
    st.session_state.portfolio_weights = None
if "betas" not in st.session_state:
    st.session_state.betas = {f: round(1/7, 4) for f in FACTOR_NAMES}


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <div style='font-family:Syne,sans-serif; font-size:20px; font-weight:800;
                    background:linear-gradient(135deg,#3b82f6,#06b6d4);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            CGF GESTION
        </div>
        <div style='font-size:10px; color:#475569; letter-spacing:0.12em; margin-top:2px;'>
            BRVM MULTIFACTORIAL ENGINE
        </div>
    </div>
    <hr style='margin:8px 0 16px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("**📐 Poids des Facteurs (β_i)**")
    st.caption("Doit sommer à 1.0")

    betas = {}
    beta_sum = 0.0
    for factor in FACTOR_NAMES:
        defn = FACTOR_DEFINITIONS[factor]
        val = st.slider(
            f"{defn['icon']} {factor}",
            min_value=0.0, max_value=1.0,
            value=st.session_state.betas[factor],
            step=0.01,
            key=f"beta_{factor}"
        )
        betas[factor] = val
        beta_sum += val

    beta_sum_rounded = round(beta_sum, 4)
    if abs(beta_sum_rounded - 1.0) > 0.01:
        st.warning(f"⚠️ Somme β = {beta_sum_rounded:.3f} ≠ 1.0")
    else:
        st.success(f"✅ Somme β = {beta_sum_rounded:.3f}")

    if st.button("⚖️ Égaliser les poids"):
        equal = round(1/7, 4)
        for f in FACTOR_NAMES:
            st.session_state.betas[f] = equal
        st.rerun()

    st.session_state.betas = betas

    st.markdown("---")
    st.markdown("**📋 Référence BRVM**")
    benchmark = st.selectbox("Indice de référence", ["BRVM Composite", "BRVM Prestige"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:10px; color:#475569; line-height:1.6;'>
        Note technique · CGF Gestion<br>
        v1.0 · Mai 2024<br>
        Auteur : A.B.A.M. Gueye
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='padding: 8px 0 24px 0;'>
    <span class='pill'>Stratégie Smart Beta</span>
    <p class='section-header'>Moteur d'Allocation Multifactoriel BRVM</p>
    <p class='section-sub'>Basé sur la Note Technique CGF Gestion · 10/05/2024 · Étapes 1→2→3</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_data, tab_factors, tab_mf, tab_portfolio, tab_formula = st.tabs([
    "📥 Données",
    "📊 Indices Factoriels",
    "🔢 Indice Multifactoriel",
    "📂 Portefeuille Cible",
    "📐 Formules"
])


# ════════════════════════════════════════════
# TAB 1 — DATA INPUT
# ════════════════════════════════════════════
with tab_data:
    st.markdown("""
    <span class='pill'>Étape préliminaire</span>
    <p class='section-header'>Saisie des données de marché</p>
    <p class='section-sub'>Importez un fichier Excel/CSV ou utilisez les données de démonstration BRVM</p>
    """, unsafe_allow_html=True)

    col_load, col_demo = st.columns([1, 1])

    with col_load:
        st.markdown("**📂 Importer vos données**")
        uploaded = st.file_uploader(
            "Fichier Excel ou CSV",
            type=["xlsx", "xls", "csv"],
            help="Le fichier doit contenir une colonne 'Ticker' et les métriques financières."
        )
        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)
                st.session_state.securities_df = df
                st.success(f"✅ {len(df)} titres chargés — {df.shape[1]} colonnes")
            except Exception as e:
                st.error(f"Erreur : {e}")

    with col_demo:
        st.markdown("**🎲 Données de démonstration BRVM**")
        st.caption("Génère des titres fictifs avec toutes les métriques requises")
        n_stocks = st.number_input("Nombre de titres", min_value=5, max_value=50, value=20, step=1)

        if st.button("🚀 Générer les données de démo"):
            np.random.seed(42)
            tickers = [
                "SNTS", "ONTBV", "SGBC", "BICC", "ECOBANK",
                "SOLIBRA", "SPHC", "PALM", "CABC", "NSIA",
                "TOTALSE", "CFAO", "NESTLE", "UNILEVER", "BOABF",
                "CORIS", "SAPH", "SIFCA", "SICOR", "FILTISAC",
                "SITAB", "SODECI", "CIE", "CILU", "SETAO",
                "SODE", "BERNABE", "UNIWAX", "TRITURAF", "VIVO",
                "AIRTEL", "MTN", "ORANGE", "CROWN", "BICI",
                "SGCI", "BHCI", "SMBC", "BSIC", "SIB",
                "ALIOS", "BOA", "UBA", "ATLANTIC", "NSIA2",
                "BLOHORN", "VERSUS", "CAPEC", "CELTEL", "ICON"
            ][:n_stocks]

            demo = pd.DataFrame({
                "Ticker": tickers,
                "Société": [f"Société {t}" for t in tickers],
                "Secteur": np.random.choice(["Banque", "Industrie", "Télécom", "Agro", "Distribution"], n_stocks),
                # Value metrics
                "Book-to-Price (B/P)": np.random.uniform(0.1, 3.0, n_stocks).round(3),
                "EPS/Price (P/E inv.)": np.random.uniform(0.02, 0.25, n_stocks).round(4),
                "FCF/Price": np.random.uniform(-0.05, 0.20, n_stocks).round(4),
                "Sales/Price": np.random.uniform(0.1, 5.0, n_stocks).round(3),
                "EBIT/EV": np.random.uniform(0.01, 0.20, n_stocks).round(4),
                # Quality
                "Levier financier": np.random.uniform(0.1, 0.8, n_stocks).round(3),
                "Ratio qualité résultat": np.random.uniform(0.3, 1.5, n_stocks).round(3),
                "ROA": np.random.uniform(0.01, 0.20, n_stocks).round(4),
                "ROE": np.random.uniform(0.05, 0.35, n_stocks).round(4),
                # Dividende
                "Dividend Yield": np.random.uniform(0.0, 0.10, n_stocks).round(4),
                # Volatilité
                "Écart-type des rendements": np.random.uniform(0.005, 0.05, n_stocks).round(5),
                # Momentum
                "Rdt moyen journalier": np.random.uniform(-0.002, 0.003, n_stocks).round(5),
                "Rdt moyen hebdo": np.random.uniform(-0.01, 0.015, n_stocks).round(5),
                "Rdt moyen mensuel": np.random.uniform(-0.03, 0.06, n_stocks).round(4),
                "Rdt moyen trimestriel": np.random.uniform(-0.05, 0.15, n_stocks).round(4),
                "Rdt moyen semestriel": np.random.uniform(-0.10, 0.25, n_stocks).round(4),
                "Rdt moyen annuel": np.random.uniform(-0.15, 0.50, n_stocks).round(4),
                # Liquidité
                "Volume moyen transigé": np.random.uniform(1e4, 5e6, n_stocks).round(0),
                # Taille
                "Capitalisation boursière": np.random.uniform(1e9, 5e11, n_stocks).round(0),
            })
            demo.set_index("Ticker", inplace=True)
            st.session_state.securities_df = demo
            st.success(f"✅ {n_stocks} titres BRVM générés avec toutes les métriques")

    # Show loaded data
    if st.session_state.securities_df is not None:
        df = st.session_state.securities_df
        st.markdown("---")
        st.markdown(f"**Univers des titres — {len(df)} valeurs**")

        # Coverage check
        all_metrics = []
        for defn in FACTOR_DEFINITIONS.values():
            all_metrics.extend(defn["metrics"])
        all_metrics = list(set(all_metrics))
        found = [m for m in all_metrics if m in df.columns]
        missing = [m for m in all_metrics if m not in df.columns]

        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Métriques disponibles", f"{len(found)}/{len(all_metrics)}")
        mcol2.metric("Titres dans l'univers", len(df))

        if missing:
            st.warning(f"Métriques manquantes : {', '.join(missing)}")

        st.dataframe(df.style.format(precision=4), use_container_width=True, height=350)

        # Download template
        template_cols = ["Ticker"] + all_metrics
        template_df = pd.DataFrame(columns=template_cols)
        buf = io.BytesIO()
        template_df.to_excel(buf, index=False)
        st.download_button(
            "⬇️ Télécharger le template Excel",
            data=buf.getvalue(),
            file_name="template_brvm_multifactorial.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ════════════════════════════════════════════
# TAB 2 — FACTOR INDICES
# ════════════════════════════════════════════
with tab_factors:
    st.markdown("""
    <span class='pill'>Étape 1</span>
    <p class='section-header'>Calcul des Indices Factoriels</p>
    <p class='section-sub'>F_i(t,T) = Σ w_ij · m_ij / max(m_ij) &nbsp;|&nbsp; Inversion pour la Volatilité</p>
    """, unsafe_allow_html=True)

    if st.session_state.securities_df is None:
        st.info("👈 Chargez vos données dans l'onglet **Données** pour commencer.")
    else:
        df = st.session_state.securities_df.copy()

        if st.button("⚙️ Calculer les indices factoriels", type="primary"):
            factor_scores = pd.DataFrame(index=df.index)
            for factor in FACTOR_NAMES:
                factor_scores[factor] = compute_factor_index(df, factor)
            st.session_state.factor_scores = factor_scores
            st.success("✅ Indices factoriels calculés avec succès")

        if st.session_state.factor_scores is not None:
            fs = st.session_state.factor_scores

            # Summary metrics
            cols = st.columns(7)
            for i, factor in enumerate(FACTOR_NAMES):
                defn = FACTOR_DEFINITIONS[factor]
                with cols[i]:
                    best = fs[factor].idxmax()
                    st.metric(
                        label=f"{defn['icon']} {factor}",
                        value=f"{fs[factor].mean():.3f}",
                        delta=f"Leader: {best}"
                    )

            st.markdown("---")

            # Heatmap
            st.markdown("**Heatmap des scores factoriels**")
            fig_heat = go.Figure(data=go.Heatmap(
                z=fs.values,
                x=fs.columns.tolist(),
                y=fs.index.tolist(),
                colorscale=[
                    [0, "#0b0f1a"],
                    [0.3, "#1e3a5f"],
                    [0.6, "#3b82f6"],
                    [1, "#06b6d4"]
                ],
                text=fs.round(3).values,
                texttemplate="%{text}",
                textfont={"size": 9, "family": "JetBrains Mono"},
                hoverongaps=False,
                showscale=True,
                colorbar=dict(
                    tickfont=dict(color="#94a3b8", family="JetBrains Mono"),
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="#1e2d45",
                )
            ))
            fig_heat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="JetBrains Mono"),
                height=max(350, len(fs) * 22),
                margin=dict(l=80, r=20, t=20, b=40),
                xaxis=dict(tickfont=dict(size=11), side="top"),
                yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # Factor distribution (radar per top-5 stocks)
            st.markdown("**Profil factoriel — Top 5 titres (meilleur MF brut)**")
            raw_mf = sum(fs[f] * (1/7) for f in FACTOR_NAMES)
            top5 = raw_mf.nlargest(5).index.tolist()

            fig_radar = go.Figure()
            cats = FACTOR_NAMES + [FACTOR_NAMES[0]]
            for ticker in top5:
                vals = [fs.loc[ticker, f] for f in FACTOR_NAMES]
                vals_closed = vals + [vals[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_closed, theta=cats,
                    fill="toself", name=ticker,
                    opacity=0.7,
                    line=dict(width=2)
                ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1],
                                   tickfont=dict(color="#475569", size=9, family="JetBrains Mono"),
                                   gridcolor="#1e2d45", linecolor="#1e2d45"),
                    angularaxis=dict(tickfont=dict(color="#94a3b8", size=11, family="Syne"),
                                     gridcolor="#1e2d45", linecolor="#1e2d45")
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(color="#94a3b8", family="JetBrains Mono"),
                            bgcolor="rgba(0,0,0,0)"),
                height=420,
                margin=dict(l=60, r=60, t=40, b=40),
                font=dict(color="#94a3b8"),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Table
            st.markdown("**Table des scores factoriels**")
            display_fs = fs.copy().round(4)
            st.dataframe(display_fs.style.background_gradient(
                cmap="Blues", axis=0
            ), use_container_width=True)

            buf = io.BytesIO()
            fs.reset_index().to_excel(buf, index=False)
            st.download_button("⬇️ Exporter les scores factoriels",
                               buf.getvalue(),
                               "scores_factoriels.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ════════════════════════════════════════════
# TAB 3 — MULTIFACTOR INDEX
# ════════════════════════════════════════════
with tab_mf:
    st.markdown("""
    <span class='pill'>Étape 2</span>
    <p class='section-header'>Calcul de l'Indice Multifactoriel</p>
    <p class='section-sub'>MF(t,T) = Σ β_i · F_i(t,T) &nbsp;·&nbsp; Ajustez les β dans la barre latérale</p>
    """, unsafe_allow_html=True)

    if st.session_state.factor_scores is None:
        st.info("👈 Calculez d'abord les **Indices Factoriels** (Étape 1).")
    else:
        fs = st.session_state.factor_scores

        # Show beta summary
        beta_cols = st.columns(7)
        for i, factor in enumerate(FACTOR_NAMES):
            defn = FACTOR_DEFINITIONS[factor]
            with beta_cols[i]:
                st.metric(f"{defn['icon']} β_{factor[:3]}", f"{betas[factor]:.2f}")

        if st.button("🔢 Calculer l'indice multifactoriel", type="primary"):
            mf = compute_multifactor_index(fs, betas)
            st.session_state.mf_scores = mf
            st.success("✅ Indice multifactoriel calculé")

        if st.session_state.mf_scores is not None:
            mf = st.session_state.mf_scores.sort_values(ascending=False)

            # Ranking bar chart
            st.markdown("**Classement MF — tous les titres**")
            colors = [FACTOR_DEFINITIONS["Value"]["color"] if i < 5 else
                      ("#1e3a5f" if i >= len(mf)-5 else "#1e2d45")
                      for i in range(len(mf))]

            fig_bar = go.Figure(go.Bar(
                x=mf.index,
                y=mf.values,
                marker=dict(
                    color=colors,
                    line=dict(width=0)
                ),
                text=mf.round(4).values,
                textposition="outside",
                textfont=dict(size=9, color="#94a3b8", family="JetBrains Mono"),
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="JetBrains Mono"),
                height=380,
                margin=dict(l=20, r=20, t=20, b=60),
                xaxis=dict(tickfont=dict(size=10), gridcolor="#1e2d45", linecolor="#1e2d45"),
                yaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", title="Score MF"),
                bargap=0.3,
            )
            # annotation for top/bottom zone
            fig_bar.add_annotation(
                x=mf.index[2], y=mf.max()*1.05,
                text="▲ Zone d'achat", showarrow=False,
                font=dict(color="#3b82f6", size=10, family="JetBrains Mono"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Factor contribution decomposition (top 10)
            st.markdown("**Décomposition factorielle — Top 10**")
            top10 = mf.head(10).index
            decomp = pd.DataFrame(
                {f: betas[f] * fs.loc[top10, f] for f in FACTOR_NAMES},
                index=top10
            )
            colors_stack = [FACTOR_DEFINITIONS[f]["color"] for f in FACTOR_NAMES]
            fig_stack = go.Figure()
            for factor, color in zip(FACTOR_NAMES, colors_stack):
                fig_stack.add_trace(go.Bar(
                    name=FACTOR_DEFINITIONS[factor]["icon"] + " " + factor,
                    x=decomp.index,
                    y=decomp[factor],
                    marker_color=color,
                    opacity=0.85,
                ))
            fig_stack.update_layout(
                barmode="stack",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="JetBrains Mono"),
                height=380,
                legend=dict(orientation="h", y=1.08, font=dict(size=10),
                            bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=20, r=20, t=50, b=50),
                xaxis=dict(tickfont=dict(size=11), gridcolor="#1e2d45"),
                yaxis=dict(gridcolor="#1e2d45", title="Contribution β·F"),
                bargap=0.25,
            )
            st.plotly_chart(fig_stack, use_container_width=True)

            # Full table with ranking
            st.markdown("**Table de classement MF complet**")
            mf_table = mf.reset_index()
            mf_table.columns = ["Ticker", "Score MF"]
            mf_table.insert(0, "Rang", range(1, len(mf_table)+1))
            mf_table["Score MF"] = mf_table["Score MF"].round(6)
            st.dataframe(mf_table, use_container_width=True, hide_index=True)

            buf = io.BytesIO()
            mf_table.to_excel(buf, index=False)
            st.download_button("⬇️ Exporter le classement MF",
                               buf.getvalue(),
                               "classement_multifactoriel.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ════════════════════════════════════════════
# TAB 4 — PORTFOLIO CONSTRUCTION
# ════════════════════════════════════════════
with tab_portfolio:
    st.markdown("""
    <span class='pill'>Étape 3</span>
    <p class='section-header'>Construction du Portefeuille Cible</p>
    <p class='section-sub'>α(T,t) = (n − r(T,t) + 1) / (n(n+1)/2) &nbsp;·&nbsp; Pondération par rang MF</p>
    """, unsafe_allow_html=True)

    if st.session_state.mf_scores is None:
        st.info("👈 Calculez d'abord l'**Indice Multifactoriel** (Étape 2).")
    else:
        mf = st.session_state.mf_scores
        df = st.session_state.securities_df

        pcol1, pcol2 = st.columns([2, 1])
        with pcol1:
            all_tickers = mf.index.tolist()
            excluded = st.multiselect(
                "🚫 Exclure des titres (choix motivé du Gestionnaire)",
                options=all_tickers,
                help="Titres exclus du portefeuille cible selon la note technique"
            )
        with pcol2:
            top_n = st.number_input("🔝 Top N titres seulement (0 = tous)", min_value=0, max_value=len(mf), value=0)

        if st.button("📂 Construire le portefeuille cible", type="primary"):
            pw = compute_portfolio_weights(mf, excluded=excluded)
            if top_n > 0:
                pw = pw.head(top_n)
                pw = pw / pw.sum()  # renormalize
            st.session_state.portfolio_weights = pw
            st.success(f"✅ Portefeuille construit : {len(pw)} titres · Σα = {pw.sum():.6f}")

        if st.session_state.portfolio_weights is not None:
            pw = st.session_state.portfolio_weights

            # KPI row
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Nombre de titres", len(pw))
            kpi2.metric("Poids max", f"{pw.max()*100:.2f}%")
            kpi3.metric("Poids min", f"{pw.min()*100:.2f}%")
            kpi4.metric("HHI (concentration)", f"{(pw**2).sum():.4f}")

            st.markdown("---")

            left, right = st.columns([1, 1])

            with left:
                # Pie chart
                st.markdown("**Répartition du portefeuille**")
                display_pw = pw.copy()
                if len(display_pw) > 15:
                    top15 = display_pw.head(15)
                    rest = pd.Series({"Autres": display_pw.iloc[15:].sum()})
                    display_pw = pd.concat([top15, rest])

                fig_pie = go.Figure(go.Pie(
                    labels=display_pw.index,
                    values=display_pw.values,
                    hole=0.45,
                    textfont=dict(size=10, family="JetBrains Mono"),
                    marker=dict(
                        colors=px.colors.sequential.Blues_r[:len(display_pw)] +
                               ["#1e2d45"] * max(0, len(display_pw) - 9),
                        line=dict(color="#0b0f1a", width=2)
                    ),
                    hovertemplate="<b>%{label}</b><br>%{percent:.2%}<extra></extra>"
                ))
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8", family="JetBrains Mono"),
                    height=360,
                    legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=10, r=10, t=20, b=10),
                    annotations=[dict(
                        text=f"<b>{len(pw)}</b><br>titres",
                        x=0.5, y=0.5, font_size=14,
                        showarrow=False,
                        font=dict(color="#94a3b8", family="Syne")
                    )]
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with right:
                # Horizontal bar of weights
                st.markdown("**Poids par titre — top 20**")
                pw_top = pw.head(20)
                fig_horiz = go.Figure(go.Bar(
                    x=pw_top.values * 100,
                    y=pw_top.index,
                    orientation="h",
                    marker=dict(
                        color=np.linspace(0.8, 0.2, len(pw_top)),
                        colorscale="Blues",
                        line=dict(width=0)
                    ),
                    text=[f"{v*100:.2f}%" for v in pw_top.values],
                    textposition="outside",
                    textfont=dict(size=9, color="#94a3b8", family="JetBrains Mono"),
                ))
                fig_horiz.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8", family="JetBrains Mono"),
                    height=420,
                    margin=dict(l=10, r=70, t=20, b=30),
                    xaxis=dict(title="Poids (%)", gridcolor="#1e2d45", ticksuffix="%"),
                    yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                    bargap=0.25,
                )
                st.plotly_chart(fig_horiz, use_container_width=True)

            # Full allocation table
            st.markdown("**Table d'allocation complète**")
            n = len(mf.drop(labels=excluded, errors="ignore"))
            ranks = mf.drop(labels=excluded, errors="ignore").rank(ascending=False, method="min")

            alloc_table = pd.DataFrame({
                "Ticker": pw.index,
                "Rang r(T,t)": ranks.loc[pw.index].astype(int),
                "Score MF": mf.loc[pw.index].round(6),
                "Poids α(T,t)": pw.values,
                "Poids (%)": (pw.values * 100).round(4),
            }).reset_index(drop=True)

            if "Secteur" in df.columns:
                alloc_table["Secteur"] = df.loc[alloc_table["Ticker"]]["Secteur"].values

            st.dataframe(
                alloc_table.style.format({
                    "Score MF": "{:.6f}",
                    "Poids α(T,t)": "{:.6f}",
                    "Poids (%)": "{:.4f}%"
                }).bar(subset=["Poids (%)"], color=["#1e3a5f", "#3b82f6"]),
                use_container_width=True,
                hide_index=True,
            )

            # Sector breakdown if available
            if "Secteur" in df.columns:
                st.markdown("**Répartition sectorielle**")
                sector_weights = alloc_table.groupby("Secteur")["Poids α(T,t)"].sum().sort_values(ascending=False)
                fig_sec = go.Figure(go.Bar(
                    x=sector_weights.index,
                    y=sector_weights.values * 100,
                    marker_color=["#3b82f6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"],
                    text=[f"{v*100:.1f}%" for v in sector_weights.values],
                    textposition="outside",
                    textfont=dict(size=11, color="#e2e8f0", family="JetBrains Mono"),
                ))
                fig_sec.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8", family="JetBrains Mono"),
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=40),
                    xaxis=dict(gridcolor="#1e2d45"),
                    yaxis=dict(gridcolor="#1e2d45", ticksuffix="%", title="Poids (%)"),
                    bargap=0.3,
                )
                st.plotly_chart(fig_sec, use_container_width=True)

            # Download
            buf = io.BytesIO()
            alloc_table.to_excel(buf, index=False)
            st.download_button(
                "⬇️ Exporter le portefeuille cible (Excel)",
                data=buf.getvalue(),
                file_name="portefeuille_cible_BRVM.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# ════════════════════════════════════════════
# TAB 5 — FORMULAS REFERENCE
# ════════════════════════════════════════════
with tab_formula:
    st.markdown("""
    <span class='pill'>Référence</span>
    <p class='section-header'>Formulaire Mathématique</p>
    <p class='section-sub'>Extrait de la Note Technique CGF Gestion · 10/05/2024</p>
    """, unsafe_allow_html=True)

    with st.expander("📐 Étape 1 · Calcul des indices factoriels", expanded=True):
        st.markdown("""
        <div class='formula-box'>
        <b>Cas général :</b><br><br>
        &nbsp;&nbsp;&nbsp;&nbsp;F_i(t,T) = Σ_{j=1}^{k_i} w_ij · m_ij(t,T) / max_{E_t}( m_ij(t,T) )<br><br>
        <b>Cas Volatilité (inversion) :</b><br><br>
        &nbsp;&nbsp;&nbsp;&nbsp;F_i(t,T) = Σ_{j=1}^{k_i} w_ij · min_{E_t}( m_ij(t,T) ) / m_ij(t,T)<br><br>
        où :<br>
        &nbsp;&nbsp;k_i     = nombre de métriques du facteur i<br>
        &nbsp;&nbsp;m_ij    = valeur de la métrique j du facteur i pour le titre T à la date t<br>
        &nbsp;&nbsp;w_ij    = poids de la métrique j dans le facteur i<br>
        &nbsp;&nbsp;E_t     = univers des titres à la date t
        </div>
        """, unsafe_allow_html=True)

        # Factor detail table
        rows = []
        for fname, defn in FACTOR_DEFINITIONS.items():
            for m, w in zip(defn["metrics"], defn["weights"]):
                rows.append({
                    "Facteur": defn["icon"] + " " + fname,
                    "Métrique": m,
                    "Poids w_ij": f"{w*100:.2f}%",
                    "Inversion": "✓" if defn["invert"] else "—"
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("📐 Étape 2 · Calcul de l'indice multifactoriel"):
        st.markdown("""
        <div class='formula-box'>
        MF(t,T) = Σ_{i=1}^{7} β_i · F_i(t,T)<br><br>
        où :<br>
        &nbsp;&nbsp;β_i  = poids attribué au facteur i (ajustable dans la barre latérale)<br>
        &nbsp;&nbsp;F_i  = score factoriel calculé à l'étape 1<br>
        &nbsp;&nbsp;Σ β_i = 1  (contrainte de normalisation)
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📐 Étape 3 · Construction du portefeuille cible"):
        st.markdown("""
        <div class='formula-box'>
        α(T,t) = ( n(t) − r(T,t) + 1 ) / ( n(t) · (n(t)+1) / 2 )<br><br>
        où :<br>
        &nbsp;&nbsp;α(T,t)  = pondération du titre T dans le portefeuille cible à la date t<br>
        &nbsp;&nbsp;r(T,t)  = rang du titre T selon MF (1 = meilleur score)<br>
        &nbsp;&nbsp;n(t)    = nombre total de titres admissibles à la date t<br><br>
        Propriété : Σ_T α(T,t) = 1  (les poids somment à 1)<br>
        Propriété : α décroît linéairement avec le rang → le meilleur titre reçoit le poids max.
        </div>
        """, unsafe_allow_html=True)

        # Numeric illustration
        st.markdown("**Illustration numérique (n=5)**")
        n_ex = 5
        ex_data = []
        for r in range(1, n_ex+1):
            w = (n_ex - r + 1) / (n_ex * (n_ex+1) / 2)
            ex_data.append({"Rang r": r, "Poids α": f"{w:.4f}", "Poids (%)": f"{w*100:.2f}%"})
        st.dataframe(pd.DataFrame(ex_data), use_container_width=True, hide_index=True)

    with st.expander("📋 Indices de référence BRVM"):
        st.markdown("""
        **BRVM Composite** — Indice de référence par défaut pour les sous-portefeuilles actions.  
        Construit comme la moyenne pondérée par capitalisation de l'ensemble des titres cotés.  
        Variation = variation de capitalisation boursière (hors effet volume).  
        Ajusté à chaque introduction en bourse ou augmentation de capital.

        **BRVM Prestige** — Regroupe les valeurs du Compartiment Prestige.  
        Révision annuelle selon les critères d'éligibilité.

        En l'absence d'indices multifactoriels sur le marché BRVM, le **BRVM Composite** est 
        l'indice de référence par défaut.
        """)

    with st.expander("📋 Détail des 7 facteurs et métriques"):
        for fname, defn in FACTOR_DEFINITIONS.items():
            st.markdown(f"""
            **{defn['icon']} Facteur {fname}** · *{defn['description']}*
            """)
            detail_rows = [
                {"Métrique": m, "Poids": f"{w*100:.2f}%"}
                for m, w in zip(defn["metrics"], defn["weights"])
            ]
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
            st.markdown("")
