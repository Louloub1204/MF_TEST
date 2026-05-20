import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import warnings
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
    vol = vol[[c for c in tickers if c in vol.columns]]

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

# ─── FACTOR FUNCTIONS ─────────────────────────────────────────

def compute_value_factor(data, year, weights):
    moy = data["moyenne_cours"]
    nb  = data["nb_titres"]
    year_row = moy[moy['Date'] == year]
    if year_row.empty:
        return None
    detail = {}
    for t in data["tickers"]:
        if t not in nb:
            continue
        cm = year_row[t].values[0] if t in year_row.columns else np.nan
        if pd.isna(cm) or cm <= 0:
            continue
        cap = cm * nb[t]
        if cap <= 0:
            continue
        bp  = data["capitaux_propres"].get(t, {}).get(year, np.nan)
        eps = data["resultat_net"].get(t, {}).get(year, np.nan)
        fcf = data["flux_treso"].get(t, {}).get(year, np.nan)
        ca  = data["chiffre_affaires"].get(t, {}).get(year, np.nan)
        detail[t] = {
            "Book/Price": bp/cap  if pd.notna(bp)  else np.nan,
            "EPS/Price":  eps/cap if pd.notna(eps) else np.nan,
            "FCF/Price":  fcf/cap if pd.notna(fcf) else np.nan,
            "CA/Price":   ca/cap  if pd.notna(ca)  else np.nan,
            "Cours moy":  cm,
            "Cap. mché (MFCFA)": cap/1e6,
        }
    if not detail:
        return None
    df = pd.DataFrame(detail).T
    metrics = ["Book/Price","EPS/Price","FCF/Price","CA/Price"]
    # normalise: m / max(m)
    norm = {}
    for m in metrics:
        col = df[m].replace([np.inf,-np.inf], np.nan)
        mx  = col.max()
        norm[m] = col/mx if (pd.notna(mx) and mx > 0) else col*0
    norm_df = pd.DataFrame(norm)
    score = sum(norm_df[m].fillna(0) * weights.get(m, 0) for m in metrics)
    res = pd.DataFrame({"Score Value": score, **{m: df[m] for m in metrics},
                        "Cours moy": df["Cours moy"],
                        "Cap. mché (MFCFA)": df["Cap. mché (MFCFA)"]})
    return res.dropna(subset=["Score Value"]).sort_values("Score Value", ascending=False)

def compute_momentum_factor(data, end_date, weights):
    returns = data["cours"].pct_change().dropna(how='all')
    try:
        ed = pd.to_datetime(end_date)
    except Exception:
        ed = returns.index[-1]
    sub = returns[returns.index <= ed]
    horizons = {"Journalier":1,"Hebdo":5,"Mensuel":21,"Trimestriel":63,"Semestriel":126,"Annuel":252}
    mom = {h: sub.tail(n).mean() for h, n in horizons.items()}
    mom_df = pd.DataFrame(mom).replace([np.inf,-np.inf], np.nan)
    norm = {}
    for h in horizons:
        col = mom_df[h].dropna()
        mx  = col.max()
        norm[h] = mom_df[h]/mx if (pd.notna(mx) and mx > 0) else mom_df[h]*0
    norm_df = pd.DataFrame(norm)
    score = sum(norm_df[h].fillna(0) * weights.get(h, 1/6) for h in horizons)
    res = pd.DataFrame({"Score Momentum": score,
                        **{f"Rdt {h}": mom_df[h] for h in horizons}})
    return res.dropna(subset=["Score Momentum"]).sort_values("Score Momentum", ascending=False)

def compute_volatility_factor(data, end_date, window=252):
    returns = data["cours"].pct_change().dropna(how='all')
    try:
        ed = pd.to_datetime(end_date)
    except Exception:
        ed = returns.index[-1]
    vol = returns[returns.index <= ed].tail(window).std().dropna()
    min_v = vol.min()
    score = (min_v / vol).replace([np.inf,-np.inf], np.nan).dropna()
    return pd.DataFrame({"Score Volatilité": score, "Écart-type": vol}
                        ).sort_values("Score Volatilité", ascending=False)

def compute_dividend_factor(data, year):
    div = data["dividendes"]
    moy = data["moyenne_cours"]
    dr  = div[div['Date'] == year]
    mr  = moy[moy['Date'] == year]
    if dr.empty or mr.empty:
        return None
    yields = {}
    for t in data["tickers"]:
        if t not in dr.columns or t not in mr.columns:
            continue
        d = dr[t].values[0]
        p = mr[t].values[0]
        if pd.notna(d) and pd.notna(p) and p > 0:
            yields[t] = d/p
    if not yields:
        return None
    s = pd.Series(yields)
    mx = s.max()
    score = s/mx if mx > 0 else s
    return pd.DataFrame({"Score Dividende": score, "Dividend Yield": s}
                        ).sort_values("Score Dividende", ascending=False)

def compute_liquidity_factor(data):
    avg = data["volumes"].mean().replace([np.inf,-np.inf], np.nan).dropna()
    mx  = avg.max()
    score = avg/mx if mx > 0 else avg
    return pd.DataFrame({"Score Liquidité": score, "Volume moyen": avg}
                        ).sort_values("Score Liquidité", ascending=False)

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

def compute_portfolio_weights(mf, excluded=None):
    if excluded:
        mf = mf.drop(labels=excluded, errors='ignore')
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
        st.markdown("**⚖️ Poids des facteurs (β_i)**")
        b_val = st.slider("💰 Value",      0.0, 1.0, 0.20, 0.01)
        b_mom = st.slider("🚀 Momentum",   0.0, 1.0, 0.20, 0.01)
        b_vol = st.slider("📉 Volatilité", 0.0, 1.0, 0.20, 0.01)
        b_div = st.slider("💸 Dividende",  0.0, 1.0, 0.20, 0.01)
        b_liq = st.slider("💧 Liquidité",  0.0, 1.0, 0.20, 0.01)
        bs = round(b_val+b_mom+b_vol+b_div+b_liq, 4)
        st.success(f"✅ Σβ = {bs:.2f}") if abs(bs-1.0) <= 0.01 else st.warning(f"⚠️ Σβ = {bs:.2f}")
        betas = {"Value":b_val,"Momentum":b_mom,"Volatilité":b_vol,"Dividende":b_div,"Liquidité":b_liq}
    else:
        selected_year, date_ref, betas = 2024, None, {}

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
    st.markdown("""<div class='fbox'>F_value(t,T) = Σ w_i · m_i(t,T) / max_E(m_i(t,T))<br>
    Métriques : Book/Price · EPS/Price · FCF/Price · CA/Price</div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    w_bp  = c1.number_input("Book/Price",  0.0,1.0,0.25,0.05)
    w_eps = c2.number_input("EPS/Price",   0.0,1.0,0.25,0.05)
    w_fcf = c3.number_input("FCF/Price",   0.0,1.0,0.25,0.05)
    w_ca  = c4.number_input("CA/Price",    0.0,1.0,0.25,0.05)
    ws = round(w_bp+w_eps+w_fcf+w_ca,4)
    if abs(ws-1.0) > 0.01:
        st.warning(f"Somme poids = {ws:.2f} ≠ 1.0")

    if st.button("⚙️ Calculer le Facteur Value", type="primary"):
        weights_v = {"Book/Price":w_bp,"EPS/Price":w_eps,"FCF/Price":w_fcf,"CA/Price":w_ca}
        res = compute_value_factor(data, selected_year, weights_v)
        if res is not None:
            fr["Value"] = res
            st.success(f"✅ {len(res)} titres scorés — Année {selected_year}")
        else:
            st.error("Données insuffisantes.")

    if "Value" in fr:
        res = fr["Value"]
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Titres", len(res))
        m2.metric("Score max", f"{res['Score Value'].max():.4f}")
        m3.metric("N°1 Value", res.index[0])
        m4.metric("Score moyen", f"{res['Score Value'].mean():.4f}")

        st.plotly_chart(score_bar(res["Score Value"], "#06b6d4"), use_container_width=True)

        metrics_v = ["Book/Price","EPS/Price","FCF/Price","CA/Price"]
        disp = res[[c for c in ["Score Value"]+metrics_v+["Cours moy","Cap. mché (MFCFA)"] if c in res.columns]].reset_index()
        disp.columns = ["Ticker"] + [c for c in ["Score Value"]+metrics_v+["Cours moy","Cap. mché (MFCFA)"] if c in res.columns]
        disp.insert(0,"Rang", range(1, len(disp)+1))
        st.dataframe(disp.style.format({c:"{:.4f}" for c in metrics_v} | {"Score Value":"{:.4f}","Cours moy":"{:,.0f}","Cap. mché (MFCFA)":"{:,.0f}"}
                     ).bar(subset=["Score Value"], color=["#1e3a5f","#3b82f6"]),
                     use_container_width=True, hide_index=True)

        buf = io.BytesIO(); disp.to_excel(buf, index=False)
        st.download_button("⬇️ Exporter", buf.getvalue(), f"value_{selected_year}.xlsx",
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
        st.plotly_chart(score_bar(res["Score Momentum"], "#8b5cf6"), use_container_width=True)
        rdt_c = [c for c in res.columns if "Rdt" in c]
        disp2 = res[["Score Momentum"]+rdt_c].head(20).reset_index().rename(columns={"index":"Ticker"})
        st.dataframe(disp2.style.format({"Score Momentum":"{:.4f}"} | {c:"{:.4%}" for c in rdt_c}),
                     use_container_width=True, hide_index=True)

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
            st.plotly_chart(score_bar(res["Score Volatilité"], "#ef4444"), use_container_width=True)
        with r:
            st.markdown("**Volatilité annualisée (σ×√252)**")
            rv = res.sort_values("Écart-type")
            fig2 = go.Figure(go.Bar(x=rv.index, y=rv["Écart-type"]*np.sqrt(252),
                                    marker=dict(color=rv["Écart-type"]*np.sqrt(252),
                                                colorscale=[[0,"#10b981"],[0.5,"#f59e0b"],[1,"#ef4444"]])))
            fig2.update_layout(**{**PLOT_LAYOUT,"height":360,"yaxis":dict(gridcolor="#1e2d45",tickformat=".1%")})
            st.plotly_chart(fig2, use_container_width=True)

# ══ DIVIDENDE ══════════════════════════════════════════════════
with t4:
    st.markdown("<span class='pill'>Étape 1 · Facteur Dividende</span><p class='sh'>Dividend Yield = Dividende / Cours moyen</p>", unsafe_allow_html=True)

    div_years = sorted(data["dividendes"]["Date"].dropna().astype(int).unique(), reverse=True)
    div_yr = st.selectbox("Année dividende", div_years)

    if st.button("⚙️ Calculer le Dividende", type="primary"):
        res = compute_dividend_factor(data, div_yr)
        if res is not None:
            fr["Dividende"] = res
            paying = len(res[res["Dividend Yield"]>0])
            st.success(f"✅ {len(res)} titres · {paying} payeurs de dividende")
        else:
            st.error("Données insuffisantes.")

    if "Dividende" in fr:
        res = fr["Dividende"]
        paying = res[res["Dividend Yield"]>0]
        m1,m2,m3 = st.columns(3)
        m1.metric("Payeurs", len(paying))
        m2.metric("Yield max", f"{res['Dividend Yield'].max():.2%}")
        m3.metric("N°1 Dividende", res.index[0])

        fig_d = go.Figure(go.Bar(x=res.index, y=res["Dividend Yield"],
                                 marker=dict(color=res["Dividend Yield"],
                                             colorscale=[[0,"#1e2d45"],[1,"#f59e0b"]]),
                                 text=[f"{v:.2%}" for v in res["Dividend Yield"]],
                                 textposition="outside", textfont=dict(size=9,color="#94a3b8")))
        fig_d.update_layout(**{**PLOT_LAYOUT,"height":360,"yaxis":dict(gridcolor="#1e2d45",tickformat=".1%")})
        st.plotly_chart(fig_d, use_container_width=True)

        st.dataframe(res.reset_index().rename(columns={"index":"Ticker"}).style.format(
            {"Score Dividende":"{:.4f}","Dividend Yield":"{:.4%}"}),
            use_container_width=True, hide_index=True)

# ══ LIQUIDITÉ ══════════════════════════════════════════════════
with t5:
    st.markdown("<span class='pill'>Étape 1 · Facteur Liquidité</span><p class='sh'>Volume moyen transigé</p>", unsafe_allow_html=True)

    if st.button("⚙️ Calculer la Liquidité", type="primary"):
        res = compute_liquidity_factor(data)
        fr["Liquidité"] = res
        st.success(f"✅ {len(res)} titres scorés")

    if "Liquidité" in fr:
        res = fr["Liquidité"]
        m1,m2 = st.columns(2)
        m1.metric("Titre + liquide", res.index[0])
        m2.metric("Vol. moyen max", f"{res['Volume moyen'].max()/1e6:.0f} M FCFA")

        fig_l = go.Figure(go.Bar(x=res.index, y=res["Volume moyen"]/1e6,
                                 marker=dict(color=res["Score Liquidité"],
                                             colorscale=[[0,"#1e2d45"],[1,"#06b6d4"]])))
        fig_l.update_layout(**{**PLOT_LAYOUT,"height":360,"yaxis":dict(gridcolor="#1e2d45",title="Volume moy. (M FCFA)")})
        st.plotly_chart(fig_l, use_container_width=True)

# ══ INDICE MULTIFACTORIEL ══════════════════════════════════════
with t6:
    st.markdown("<span class='pill'>Étape 2</span><p class='sh'>Indice Multifactoriel</p>", unsafe_allow_html=True)
    st.markdown("""<div class='fbox'>MF(t,T) = Σ_{i=1}^{5} β_i · F_i(t,T) &nbsp;·&nbsp; Σ β_i = 1<br>
    Ajustez les β_i dans la barre latérale gauche</div>""", unsafe_allow_html=True)

    computed = list(fr.keys())
    if not computed:
        st.info("👈 Calculez au moins un facteur avant de continuer.")
    else:
        st.success(f"Facteurs disponibles : {', '.join(computed)}")
        bc = st.columns(5)
        for i,(fname,beta) in enumerate(betas.items()):
            bc[i].metric(f"β {fname}", f"{beta:.2f}", "✓" if fname in computed else "✗")

        if st.button("🔢 Calculer l'Indice MF", type="primary"):
            mf = compute_multifactor(fr, betas)
            st.session_state.mf_scores = mf
            st.success(f"✅ {len(mf)} titres classés")

        if st.session_state.mf_scores is not None:
            mf = st.session_state.mf_scores
            st.markdown("---")
            st.plotly_chart(score_bar(mf, "#3b82f6", height=380), use_container_width=True)

            # Stacked contribution
            st.markdown("**Décomposition factorielle — Top 15**")
            top15 = mf.head(15).index
            clrs = {"Value":"#3b82f6","Momentum":"#8b5cf6","Volatilité":"#ef4444",
                    "Dividende":"#f59e0b","Liquidité":"#06b6d4"}
            fig_s = go.Figure()
            for fname in computed:
                sc = [c for c in fr[fname].columns if "Score" in c][0]
                vals = [fr[fname].loc[t,sc]*betas.get(fname,0) if t in fr[fname].index else 0 for t in top15]
                fig_s.add_trace(go.Bar(name=fname,x=list(top15),y=vals,
                                       marker_color=clrs.get(fname,"#64748b"),opacity=0.85))
            fig_s.update_layout(barmode="stack",**{**PLOT_LAYOUT,"height":340,
                                "legend":dict(orientation="h",y=1.08,bgcolor="rgba(0,0,0,0)"),
                                "margin":dict(l=10,r=10,t=50,b=50)})
            st.plotly_chart(fig_s, use_container_width=True)

            tbl = mf.reset_index()
            tbl.columns = ["Ticker","Score MF"]
            tbl.insert(0,"Rang",range(1,len(tbl)+1))
            st.dataframe(tbl, use_container_width=True, hide_index=True)

            buf=io.BytesIO(); tbl.to_excel(buf,index=False)
            st.download_button("⬇️ Exporter classement MF", buf.getvalue(), "classement_MF.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══ PORTEFEUILLE ═══════════════════════════════════════════════
with t7:
    st.markdown("<span class='pill'>Étape 3</span><p class='sh'>Construction du Portefeuille Cible</p>", unsafe_allow_html=True)
    st.markdown("""<div class='fbox'>α(T,t) = (n − r(T,t) + 1) / (n·(n+1)/2)<br>
    Propriété : Σ α = 1 · le meilleur titre MF reçoit le poids maximum</div>""", unsafe_allow_html=True)

    if st.session_state.mf_scores is None:
        st.info("👈 Calculez d'abord l'Indice MF (onglet 🔢).")
    else:
        mf = st.session_state.mf_scores
        p1,p2 = st.columns([2,1])
        with p1:
            excluded = st.multiselect("🚫 Exclure des titres", mf.index.tolist())
        with p2:
            top_n = st.number_input("🔝 Top N (0 = tous)", 0, len(mf), 0)

        if st.button("📂 Construire le portefeuille", type="primary"):
            pw = compute_portfolio_weights(mf, excluded)
            if top_n > 0:
                pw = pw.head(top_n); pw = pw / pw.sum()
            st.session_state.pw = pw
            st.success(f"✅ {len(pw)} titres · Σα = {pw.sum():.6f}")

        if st.session_state.pw is not None:
            pw = st.session_state.pw
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Nb titres", len(pw))
            k2.metric("Poids max", f"{pw.max()*100:.2f}%")
            k3.metric("Poids min", f"{pw.min()*100:.2f}%")
            k4.metric("HHI", f"{(pw**2).sum():.4f}")
            st.markdown("---")

            pl,pr = st.columns(2)
            with pl:
                st.markdown("**Répartition**")
                dpw = pw.copy()
                if len(dpw) > 15:
                    dpw = pd.concat([dpw.head(15), pd.Series({"Autres": dpw.iloc[15:].sum()})])
                fig_p = go.Figure(go.Pie(labels=dpw.index, values=dpw.values, hole=0.45,
                                         textfont=dict(size=10),
                                         marker=dict(line=dict(color="#0b0f1a",width=2)),
                                         hovertemplate="<b>%{label}</b><br>%{percent:.2%}<extra></extra>"))
                fig_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=380,
                                    font=dict(color="#94a3b8",family="JetBrains Mono"),
                                    legend=dict(font=dict(size=10),bgcolor="rgba(0,0,0,0)"),
                                    margin=dict(l=10,r=10,t=20,b=10),
                                    annotations=[dict(text=f"<b>{len(pw)}</b><br>titres",
                                                      x=0.5,y=0.5,font_size=14,showarrow=False,
                                                      font=dict(color="#94a3b8"))])
                st.plotly_chart(fig_p, use_container_width=True)

            with pr:
                st.markdown("**Poids — Top 20**")
                pt = pw.head(20)
                fig_h = go.Figure(go.Bar(x=pt.values*100, y=pt.index, orientation="h",
                                         marker=dict(color=np.linspace(0.9,0.2,len(pt)),
                                                     colorscale="Blues"),
                                         text=[f"{v*100:.2f}%" for v in pt.values],
                                         textposition="outside", textfont=dict(size=9,color="#94a3b8")))
                fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                    height=440, font=dict(color="#94a3b8",family="JetBrains Mono"),
                                    margin=dict(l=10,r=70,t=20,b=30),
                                    xaxis=dict(gridcolor="#1e2d45",ticksuffix="%"),
                                    yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_h, use_container_width=True)

            ranks = mf.drop(labels=excluded, errors='ignore').rank(ascending=False, method='min')
            alloc = pd.DataFrame({
                "Ticker": pw.index,
                "Rang r(T,t)": ranks.loc[pw.index].astype(int),
                "Score MF": mf.loc[pw.index].round(6),
                "Poids α(T,t)": pw.values,
                "Poids (%)": (pw.values*100).round(4),
            }).reset_index(drop=True)
            st.dataframe(alloc.style.format({"Score MF":"{:.6f}","Poids α(T,t)":"{:.6f}","Poids (%)":"{:.4f}%"}
                         ).bar(subset=["Poids (%)"],color=["#1e3a5f","#3b82f6"]),
                         use_container_width=True, hide_index=True)

            buf = io.BytesIO(); alloc.to_excel(buf, index=False)
            st.download_button("⬇️ Exporter le portefeuille (Excel)", buf.getvalue(),
                               "portefeuille_BRVM.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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
            st.plotly_chart(fig_c, use_container_width=True)
    with ti2:
        st.dataframe(data["dividendes"].set_index("Date"), use_container_width=True)
    with ti3:
        rn = data["resultat_net"]
        rows2 = [{"Ticker":t,"Année":y,"Résultat net (MFCFA)":v/1e6}
                 for t,yrs in rn.items() for y,v in yrs.items()]
        if rows2:
            df_rn = pd.DataFrame(rows2).pivot(index="Ticker",columns="Année",values="Résultat net (MFCFA)")
            st.dataframe(df_rn.style.format("{:,.0f}"), use_container_width=True)
