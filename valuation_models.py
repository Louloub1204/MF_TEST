"""
Modèles de valorisation BRVM v2 — avec scaling correct (M FCFA → FCFA)
"""
import numpy as np
import pandas as pd

RISK_FREE_RATE  = 0.06
MARKET_PREMIUM  = 0.06
TERMINAL_GROWTH = 0.04
TAX_RATE_DEFAULT= 0.25
SCALE           = 1_000_000  # Les valeurs en DB sont en M FCFA → multiplier par 1M


def compute_beta(cours_df, ticker, window=252):
    ret = cours_df.apply(pd.to_numeric, errors="coerce").pct_change().dropna(how="all")
    sub = ret.tail(window)
    if ticker not in sub.columns or sub.empty:
        return 1.0
    mkt   = sub.mean(axis=1)
    t_ret = sub[ticker].dropna()
    common = t_ret.index.intersection(mkt.index)
    if len(common) < 30:
        return 1.0
    cov = t_ret.loc[common].cov(mkt.loc[common])
    var = mkt.loc[common].var()
    return float(np.clip(cov / var, 0.2, 3.0)) if var > 0 else 1.0


def compute_wacc(beta, debt_m, equity_m, interest_m, tax=TAX_RATE_DEFAULT):
    ke    = RISK_FREE_RATE + beta * MARKET_PREMIUM
    total = abs(debt_m) + abs(equity_m)
    if total <= 0:
        return ke
    w_e = abs(equity_m) / total
    w_d = abs(debt_m) / total
    if abs(debt_m) > 0 and interest_m:
        kd = min(abs(interest_m) / abs(debt_m), 0.25)
    else:
        kd = 0.08
    return ke * w_e + kd * (1 - tax) * w_d


def cagr(data_dict, key):
    vals = sorted([(yr, v.get(key)) for yr, v in data_dict.items()
                   if v.get(key) and v[key] > 0])
    if len(vals) < 2:
        return 0.05
    v0, v1 = vals[0][1], vals[-1][1]
    n = vals[-1][0] - vals[0][0]
    if n <= 0 or v0 <= 0:
        return 0.05
    return float(np.clip((v1/v0)**(1/n) - 1, -0.10, 0.30))


def valuation_pe(ticker, fin_data, nb_titres, pe_target=None):
    d = fin_data.get(ticker, {})
    if not d or not nb_titres.get(ticker):
        return None
    yr = max(d.keys())
    rn = d[yr].get("rn")
    if not rn or rn <= 0:
        return None
    n   = nb_titres[ticker]
    eps = (rn * SCALE) / n
    # PE par défaut : banques 8x, industriels 12x, télécom 15x
    sector = d[yr].get("secteur", "")
    if pe_target is None:
        if d[yr].get("type") == "banque":
            pe_target = 8.0
        elif "Télécom" in str(sector):
            pe_target = 15.0
        else:
            pe_target = 11.0
    return {
        "modele": "P/E", "prix_cible": eps * pe_target,
        "eps": eps, "pe_cible": pe_target,
        "rn_m": rn, "annee": yr,
    }


def valuation_pb(ticker, fin_data, nb_titres, pb_target=None):
    d = fin_data.get(ticker, {})
    if not d or not nb_titres.get(ticker):
        return None
    yr = max(d.keys())
    cp = d[yr].get("capitaux_propres") or d[yr].get("capitaux_propres_mere")
    if not cp or cp <= 0:
        return None
    n    = nb_titres[ticker]
    bvps = (cp * SCALE) / n
    if pb_target is None:
        pb_target = 1.2 if d[yr].get("type") == "banque" else 1.5
    return {
        "modele": "P/B", "prix_cible": bvps * pb_target,
        "bvps": bvps, "pb_cible": pb_target,
        "cp_m": cp, "annee": yr,
    }


def valuation_ddm(ticker, fin_data, div_history, nb_titres,
                  cours_df, ke=None, g=None):
    if not div_history or not nb_titres.get(ticker):
        return None
    # Dividende le plus récent (déjà en FCFA/action dans historique_dividende)
    recent_yr = max(div_history.keys())
    d0 = div_history[recent_yr]
    if not d0 or d0 <= 0:
        return None
    g_div = g if g else cagr(
        {yr: {"div": v} for yr, v in div_history.items()}, "div"
    )
    g_div = min(float(g_div), 0.12)
    ke_val = ke if ke else (RISK_FREE_RATE + compute_beta(cours_df, ticker) * MARKET_PREMIUM)
    if ke_val <= g_div:
        ke_val = g_div + 0.02
    d1 = d0 * (1 + g_div)
    p  = d1 / (ke_val - g_div)
    return {
        "modele": "DDM", "prix_cible": p,
        "d0": d0, "d1": d1, "ke": ke_val, "g": g_div,
        "annee_div": recent_yr,
    }


def valuation_dcf(ticker, fin_data, nb_titres, cours_df,
                  horizon=5, g_tv=None, wacc_override=None, g_fcf_override=None):
    d = fin_data.get(ticker, {})
    if not d or not nb_titres.get(ticker):
        return None
    yr      = max(d.keys())
    postes  = d[yr]
    n       = nb_titres[ticker]
    is_bank = postes.get("type") == "banque"

    rex   = (postes.get("rex") or 0)
    rn    = (postes.get("rn")  or 0)
    amort = abs(postes.get("amortissements") or 0)
    tax   = abs(postes.get("impots") or 0)

    # FCF proxy (M FCFA)
    if is_bank:
        fcf0 = max(rn, 0)
    else:
        t_eff = min(abs(tax) / abs(rex), 0.40) if rex > 0 else TAX_RATE_DEFAULT
        fcf0  = rex * (1 - t_eff) + amort - amort * 0.80  # amort * 0.2 net
    if fcf0 <= 0:
        fcf0 = max(rn, 0)
    if fcf0 <= 0:
        return None

    g_fcf = g_fcf_override if g_fcf_override else cagr(d, "rn" if is_bank else "rex")
    g_fcf = float(np.clip(g_fcf, 0.0, 0.20))

    if wacc_override:
        wacc = wacc_override
    else:
        beta   = compute_beta(cours_df, ticker)
        debt_m = abs(postes.get("dette_financiere") or 0)
        eq_m   = abs(postes.get("capitaux_propres") or 1)
        int_m  = abs(postes.get("interets_charges") or 0)
        wacc   = compute_wacc(beta, debt_m, eq_m, int_m)
    wacc = float(np.clip(wacc, 0.06, 0.28))

    g_terminal = float(min(g_tv if g_tv else TERMINAL_GROWTH, wacc - 0.01))

    # Projection
    pv_fcf = 0.0
    fcf_table = []
    for t in range(1, horizon + 1):
        ft  = fcf0 * (1 + g_fcf) ** t
        pvt = ft   / (1 + wacc)  ** t
        fcf_table.append({"annee": yr + t, "fcf_m": ft, "pv_m": pvt})
        pv_fcf += pvt

    fcf_tv = fcf0 * (1 + g_fcf) ** horizon * (1 + g_terminal)
    tv     = fcf_tv / (wacc - g_terminal)
    pv_tv  = tv / (1 + wacc) ** horizon

    ev_m    = pv_fcf + pv_tv
    debt_m  = abs(postes.get("dette_financiere") or 0)
    cash_m  = abs(postes.get("tresorerie") or 0)
    eq_val_m= max(ev_m - debt_m + cash_m, 0)

    p_cible = (eq_val_m * SCALE) / n

    return {
        "modele": "DCF", "prix_cible": p_cible,
        "ev_m": ev_m, "pv_fcf_m": pv_fcf, "pv_tv_m": pv_tv,
        "fcf0_m": fcf0, "g_fcf": g_fcf, "g_tv": g_terminal,
        "wacc": wacc, "horizon": horizon,
        "fcf_table": fcf_table, "annee_base": yr, "is_bank": is_bank,
    }


def combined_price(results, weights=None):
    prices = {r["modele"]: r["prix_cible"]
              for r in results.values() if r and r.get("prix_cible", 0) > 0}
    if not prices:
        return None, {}
    if weights is None:
        weights = {m: 1.0 for m in prices}
    tw = sum(weights.get(m, 1.0) for m in prices)
    if tw <= 0:
        return None, {}
    return sum(v * weights.get(m, 1.0) for m, v in prices.items()) / tw, prices
