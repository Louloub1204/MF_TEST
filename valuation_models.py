"""
Modèles de valorisation BRVM — v3 Full Auto
Tous les paramètres sont calibrés depuis les données réelles.
Zéro saisie manuelle requise.
"""
import numpy as np
import pandas as pd

# ── Constantes marché BRVM/UEMOA ──────────────────────────────
RF              = 0.06    # Taux sans risque BCEAO 10 ans
MARKET_PREMIUM  = 0.06    # Prime de risque marché BRVM estimée
TERMINAL_GROWTH = 0.045   # Croissance PIB UEMOA long terme
TAX_DEFAULT     = 0.25    # IS moyen UEMOA
SCALE           = 1_000_000  # DB en M FCFA → FCFA


# ══════════════════════════════════════════════════════════════
# BLOC 1 — PARAMÈTRES AUTO (calibrage depuis données réelles)
# ══════════════════════════════════════════════════════════════

def compute_beta(cours_df, ticker, window=252):
    """Beta vs marché BRVM (proxy = moyenne équipondérée des titres)."""
    ret = cours_df.apply(pd.to_numeric, errors="coerce").pct_change().dropna(how="all")
    sub = ret.tail(window)
    if ticker not in sub.columns or sub.empty:
        return 1.0
    mkt    = sub.mean(axis=1)
    t_ret  = sub[ticker].dropna()
    common = t_ret.index.intersection(mkt.index)
    if len(common) < 30:
        return 1.0
    cov = t_ret.loc[common].cov(mkt.loc[common])
    var = mkt.loc[common].var()
    return float(np.clip(cov / var, 0.2, 3.0)) if var > 0 else 1.0


def auto_ke(cours_df, ticker):
    """ke = Rf + β × prime_risque (CAPM)."""
    beta = compute_beta(cours_df, ticker)
    return RF + beta * MARKET_PREMIUM, beta


def auto_kd(postes):
    """kd = Intérêts / Dette observée (coût réel de la dette)."""
    debt = abs(postes.get("dette_financiere") or 0)
    int_ = abs(postes.get("interets_charges") or 0)
    if debt > 0 and int_ > 0:
        return float(np.clip(int_ / debt, 0.03, 0.20))
    return 0.07  # défaut marché UEMOA


def auto_tax_rate(postes):
    """Taux d'imposition effectif observé."""
    rex = abs(postes.get("rex") or postes.get("rai") or 0)
    tax = abs(postes.get("impots") or 0)
    if rex > 0 and tax > 0:
        return float(np.clip(tax / rex, 0.10, 0.40))
    return TAX_DEFAULT


def auto_wacc(cours_df, ticker, postes):
    """
    WACC = ke × E/(D+E) + kd×(1-t) × D/(D+E)
    Tous les paramètres viennent des données réelles.
    """
    ke, beta = auto_ke(cours_df, ticker)
    kd       = auto_kd(postes)
    tax      = auto_tax_rate(postes)
    debt_m   = abs(postes.get("dette_financiere") or 0)
    equity_m = abs(postes.get("capitaux_propres") or
                   postes.get("capitaux_propres_mere") or 1)
    total = debt_m + equity_m
    if total <= 0:
        return ke, ke, kd, tax, beta, 0, 1
    w_e = equity_m / total
    w_d = debt_m   / total
    wacc = ke * w_e + kd * (1 - tax) * w_d
    return float(np.clip(wacc, 0.06, 0.28)), ke, kd, tax, beta, w_d, w_e


def auto_growth(data_dict, key, fallback=0.05):
    """
    CAGR calculé sur toutes les années disponibles.
    Plus il y a d'années, plus le CAGR est fiable.
    """
    vals = sorted([
        (yr, v.get(key))
        for yr, v in data_dict.items()
        if isinstance(v, dict) and v.get(key) and v[key] > 0
    ])
    if len(vals) < 2:
        return fallback
    v0, v1 = vals[0][1], vals[-1][1]
    n = vals[-1][0] - vals[0][0]
    if n <= 0 or v0 <= 0:
        return fallback
    return float(np.clip((v1/v0)**(1/n) - 1, -0.10, 0.30))


def auto_pe_target(cours_df, data_dict, ticker, nb_titres, moy_cours=None):
    """
    P/E cible = médiane des P/E historiques observés du titre.
    Si insuffisant, fallback sectoriel.
    """
    n = nb_titres.get(ticker, 0)
    if n <= 0:
        return _sector_pe(data_dict), "sectoriel (nb_titres manquant)"

    pe_obs = []
    for yr, postes in data_dict.items():
        rn = postes.get("rn")
        if not rn or rn <= 0:
            continue
        eps = (rn * SCALE) / n

        # Cours moyen annuel depuis historique
        cours_yr = None
        if moy_cours is not None and not moy_cours.empty:
            row = moy_cours[moy_cours["Date"] == yr]
            if not row.empty and ticker in row.columns:
                cours_yr = float(row[ticker].values[0])

        # Fallback : cours depuis DataFrame cours
        if cours_yr is None and cours_df is not None and not cours_df.empty:
            if ticker in cours_df.columns:
                sub = cours_df[cours_df.index.year == yr][ticker].dropna()
                if not sub.empty:
                    cours_yr = float(sub.mean())

        if cours_yr and cours_yr > 0 and eps > 0:
            pe = cours_yr / eps
            if 1 < pe < 100:  # filtrage des valeurs aberrantes
                pe_obs.append(pe)

    if len(pe_obs) >= 2:
        pe_med = float(np.median(pe_obs))
        return float(np.clip(pe_med, 4, 40)), f"médiane historique ({len(pe_obs)} obs.)"

    # Fallback sectoriel
    return _sector_pe(data_dict), "sectoriel (historique insuffisant)"


def auto_pb_target(cours_df, data_dict, ticker, nb_titres, moy_cours=None):
    """P/B cible = médiane des P/B historiques observés."""
    n = nb_titres.get(ticker, 0)
    if n <= 0:
        return _sector_pb(data_dict), "sectoriel"

    pb_obs = []
    for yr, postes in data_dict.items():
        cp = postes.get("capitaux_propres") or postes.get("capitaux_propres_mere")
        if not cp or cp <= 0:
            continue
        bvps = (cp * SCALE) / n

        cours_yr = None
        if moy_cours is not None and not moy_cours.empty:
            row = moy_cours[moy_cours["Date"] == yr]
            if not row.empty and ticker in row.columns:
                cours_yr = float(row[ticker].values[0])
        if cours_yr is None and cours_df is not None and not cours_df.empty:
            if ticker in cours_df.columns:
                sub = cours_df[cours_df.index.year == yr][ticker].dropna()
                if not sub.empty:
                    cours_yr = float(sub.mean())

        if cours_yr and bvps > 0:
            pb = cours_yr / bvps
            if 0.1 < pb < 20:
                pb_obs.append(pb)

    if len(pb_obs) >= 2:
        pb_med = float(np.median(pb_obs))
        return float(np.clip(pb_med, 0.3, 8.0)), f"médiane historique ({len(pb_obs)} obs.)"

    return _sector_pb(data_dict), "sectoriel"


def _sector_pe(data_dict):
    """P/E sectoriel par défaut."""
    yr = max(data_dict.keys()) if data_dict else 2024
    t  = data_dict.get(yr, {}).get("type", "")
    s  = data_dict.get(yr, {}).get("secteur", "")
    if t == "banque":
        return 8.0
    if "Télécom" in str(s):
        return 15.0
    if "Agro" in str(s):
        return 10.0
    return 11.0


def _sector_pb(data_dict):
    """P/B sectoriel par défaut."""
    yr = max(data_dict.keys()) if data_dict else 2024
    t  = data_dict.get(yr, {}).get("type", "")
    return 1.2 if t == "banque" else 1.5


# ══════════════════════════════════════════════════════════════
# BLOC 2 — CALIBRAGE GLOBAL (tous paramètres auto pour 1 titre)
# ══════════════════════════════════════════════════════════════

def calibrate_params(ticker, fin_data, nb_titres, cours_df,
                     moy_cours=None, div_history=None):
    """
    Calcule automatiquement tous les paramètres de valorisation
    pour un titre donné. Retourne un dict complet des paramètres.
    """
    d = fin_data.get(ticker, {})
    if not d:
        return None

    yr      = max(d.keys())
    postes  = d[yr]
    is_bank = postes.get("type") == "banque"

    # ── Paramètres marché ──────────────────────────────────────
    wacc, ke, kd, tax, beta, w_d, w_e = auto_wacc(cours_df, ticker, postes)
    g_rn  = auto_growth(d, "rn")
    g_rex = auto_growth(d, "rex") if not is_bank else g_rn
    g_ca  = auto_growth(d, "ca")
    g_fcf = g_rn if is_bank else g_rex

    # ── P/E et P/B historiques ─────────────────────────────────
    pe_target, pe_method = auto_pe_target(cours_df, d, ticker, nb_titres, moy_cours)
    pb_target, pb_method = auto_pb_target(cours_df, d, ticker, nb_titres, moy_cours)

    # ── Croissance dividendes ──────────────────────────────────
    g_div = 0.0
    if div_history:
        g_div = auto_growth(
            {yr: {"div": v} for yr, v in div_history.items()}, "div"
        )

    # ── Taux d'imposition effectif ─────────────────────────────
    tax_eff = auto_tax_rate(postes)

    return {
        # Marché
        "beta":      beta,
        "ke":        ke,
        "kd":        kd,
        "wacc":      wacc,
        "w_dette":   w_d,
        "w_equity":  w_e,
        # Croissance
        "g_rn":      g_rn,
        "g_rex":     g_rex,
        "g_ca":      g_ca,
        "g_fcf":     g_fcf,
        "g_div":     g_div,
        "g_tv":      TERMINAL_GROWTH,
        # Multiples
        "pe_target": pe_target,
        "pe_method": pe_method,
        "pb_target": pb_target,
        "pb_method": pb_method,
        # Fondamentaux
        "tax_rate":  tax_eff,
        "is_bank":   is_bank,
        "annee_ref": yr,
        "secteur":   postes.get("secteur", "—"),
    }


# ══════════════════════════════════════════════════════════════
# BLOC 3 — MODÈLES DE VALORISATION (100% auto)
# ══════════════════════════════════════════════════════════════

def valuation_ddm(ticker, fin_data, div_history, nb_titres, cours_df, params=None):
    """DDM Gordon-Shapiro — ke et g 100% auto."""
    if not div_history:
        return None
    recent_yr = max(div_history.keys())
    d0 = div_history.get(recent_yr)
    if not d0 or d0 <= 0:
        return None

    p = params or calibrate_params(ticker, fin_data, nb_titres, cours_df)
    if not p:
        return None

    g_div = float(np.clip(p["g_div"] if p["g_div"] > 0 else 0.04, 0.0, 0.12))
    ke    = p["ke"]
    if ke <= g_div:
        ke = g_div + 0.02

    d1 = d0 * (1 + g_div)
    price = d1 / (ke - g_div)

    return {
        "modele": "DDM",
        "prix_cible": float(price),
        "d0": d0, "d1": d1,
        "ke": ke, "g": g_div,
        "beta": p["beta"],
        "annee_div": recent_yr,
        "methode_params": "auto",
    }


def valuation_pe(ticker, fin_data, nb_titres, cours_df=None,
                 moy_cours=None, params=None):
    """P/E relatif — multiple calibré sur historique observé du titre."""
    d = fin_data.get(ticker, {})
    if not d or not nb_titres.get(ticker):
        return None

    yr  = max(d.keys())
    rn  = d[yr].get("rn")
    if not rn or rn <= 0:
        return None

    n   = nb_titres[ticker]
    eps = (rn * SCALE) / n

    p = params or calibrate_params(ticker, fin_data, nb_titres,
                                   cours_df or pd.DataFrame(), moy_cours)
    if not p:
        return None

    pe  = p["pe_target"]
    price = eps * pe

    return {
        "modele": "P/E",
        "prix_cible": float(price),
        "eps": eps, "pe_cible": pe,
        "pe_methode": p["pe_method"],
        "rn_m": rn, "annee": yr,
        "methode_params": "auto",
    }


def valuation_pb(ticker, fin_data, nb_titres, cours_df=None,
                 moy_cours=None, params=None):
    """P/B — multiple calibré sur historique observé du titre."""
    d = fin_data.get(ticker, {})
    if not d or not nb_titres.get(ticker):
        return None

    yr = max(d.keys())
    cp = d[yr].get("capitaux_propres") or d[yr].get("capitaux_propres_mere")
    if not cp or cp <= 0:
        return None

    n    = nb_titres[ticker]
    bvps = (cp * SCALE) / n

    p = params or calibrate_params(ticker, fin_data, nb_titres,
                                   cours_df or pd.DataFrame(), moy_cours)
    if not p:
        return None

    pb    = p["pb_target"]
    price = bvps * pb

    return {
        "modele": "P/B",
        "prix_cible": float(price),
        "bvps": bvps, "pb_cible": pb,
        "pb_methode": p["pb_method"],
        "cp_m": cp, "annee": yr,
        "methode_params": "auto",
    }


def valuation_dcf(ticker, fin_data, nb_titres, cours_df,
                  horizon=5, params=None):
    """
    DCF simplifié — WACC, g_FCF, g_TV tous auto.
    FCF proxy : Rex*(1-t) + Amort - CAPEX_est   (sociétés)
                RN                               (banques)
    """
    d = fin_data.get(ticker, {})
    if not d or not nb_titres.get(ticker):
        return None

    yr     = max(d.keys())
    postes = d[yr]
    n      = nb_titres[ticker]

    p = params or calibrate_params(ticker, fin_data, nb_titres, cours_df)
    if not p:
        return None

    is_bank = p["is_bank"]
    rex     = postes.get("rex") or 0
    rn      = postes.get("rn")  or 0
    amort   = abs(postes.get("amortissements") or 0)
    tax     = p["tax_rate"]

    # FCF proxy
    if is_bank:
        fcf0 = max(float(rn), 0)
    else:
        nopat = float(rex) * (1 - tax)
        fcf0  = nopat + amort - amort * 0.80  # CAPEX ≈ 80% amort
    if fcf0 <= 0:
        fcf0 = max(float(rn), 0)
    if fcf0 <= 0:
        return None

    g_fcf = float(np.clip(p["g_fcf"], 0.0, 0.20))
    wacc  = p["wacc"]
    g_tv  = float(min(p["g_tv"], wacc - 0.01))

    # Projection sur l'horizon
    pv_fcf, fcf_table = 0.0, []
    for t in range(1, horizon + 1):
        ft  = fcf0 * (1 + g_fcf) ** t
        pvt = ft   / (1 + wacc)  ** t
        fcf_table.append({"annee": yr + t, "fcf_m": round(ft, 1), "pv_m": round(pvt, 1)})
        pv_fcf += pvt

    # Valeur terminale
    fcf_tv = fcf0 * (1 + g_fcf) ** horizon * (1 + g_tv)
    tv     = fcf_tv / (wacc - g_tv)
    pv_tv  = tv / (1 + wacc) ** horizon

    # Bridge EV → equity
    ev_m     = pv_fcf + pv_tv
    debt_m   = abs(postes.get("dette_financiere") or 0)
    cash_m   = abs(postes.get("tresorerie") or 0)
    eq_val_m = max(ev_m - debt_m + cash_m, 0)
    price    = (eq_val_m * SCALE) / n

    return {
        "modele": "DCF",
        "prix_cible": float(price),
        "ev_m": ev_m, "pv_fcf_m": pv_fcf, "pv_tv_m": pv_tv,
        "fcf0_m": fcf0, "g_fcf": g_fcf, "g_tv": g_tv,
        "wacc": wacc, "ke": p["ke"], "kd": p["kd"],
        "beta": p["beta"],
        "horizon": horizon, "is_bank": is_bank,
        "fcf_table": fcf_table, "annee_base": yr,
        "methode_params": "auto",
    }


# ══════════════════════════════════════════════════════════════
# BLOC 4 — PRIX CIBLE COMBINÉ + POIDS AUTO
# ══════════════════════════════════════════════════════════════

def combined_price(results, weights=None):
    """
    Prix cible = moyenne pondérée des modèles disponibles.
    Poids auto : DDM=0.20, P/E=0.30, P/B=0.20, DCF=0.30
    (si un modèle est absent son poids est redistribué aux autres)
    """
    DEFAULT_W = {"DDM": 0.20, "P/E": 0.30, "P/B": 0.20, "DCF": 0.30}
    prices = {
        r["modele"]: r["prix_cible"]
        for r in results.values()
        if r and r.get("prix_cible", 0) > 0
    }
    if not prices:
        return None, {}

    w = weights or DEFAULT_W
    active_w = {m: w.get(m, 0.25) for m in prices}
    tw = sum(active_w.values())
    if tw <= 0:
        return None, {}

    # Renormalisation sur les modèles actifs
    active_w = {m: v/tw for m, v in active_w.items()}
    p_comb = sum(v * active_w[m] for m, v in prices.items())
    return float(p_comb), prices
