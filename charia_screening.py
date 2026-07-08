"""
charia_screening.py — Screening Chariatique BRVM
4 standards : DJIM · FTSE · S&P · AAOIFI/Malaisie
Compatibilité : ≥ 3 standards sur 4 passés

Les ratios sont calculés depuis le fichier de screening (Feuil1)
ou recalculés automatiquement depuis les états financiers chargés.

Exclusions sectorielles :
  - Banques conventionnelles : incompatibles structurellement (modèle Riba)
  - Revenus illicites : revenus_financiers / CA > SEUIL_ILLICITE (5%)
"""
import pandas as pd
import numpy as np

MIN_STANDARDS_PASS = 3
SEUIL_ILLICITE     = 0.05   # 5% — seuil de tolérance AAOIFI

# Tickers BRVM du secteur bancaire conventionnel — exclus automatiquement
BANK_TICKERS = {
    "SGBC","SIBC","NSBC","ECOC","BICC","BOAB","BOAS",
    "BOABF","BOAM","BOAC","BOAN","CBIBF","ETI","ETIT",
    "ORGT","SAFC","BICB","BNBC",
}

# Tickers exclus pour activités sectorielles illicites
ILLICIT_SECTOR_TICKERS = {
    "STBC": "Industrie du tabac",
    "LNBB": "Jeux de hasard / loterie",
    "SLBC": "Production / distribution de boissons alcoolisées",
}

# Seuils par standard
SEUILS = {
    "DJIM":  {"RE": 0.33, "RC_actif": 0.50, "RL": 0.33},
    "FTSE":  {"RE": 0.33, "RC_cap24": 0.33, "RL_cap24": 0.33},
    "S&P":   {"RE_cap36": 0.33, "RC_cap36": 0.49, "RL_cap36": 0.33},
    "AAOIFI":{"RE": 0.33, "RL": 0.33},
}


def _r(n, d):
    """Ratio sécurisé."""
    try:
        n, d = float(n), float(d)
        if np.isnan(n) or np.isnan(d) or d == 0:
            return None
        return n / d
    except Exception:
        return None


def _chk(v, seuil):
    """True si v < seuil (None → True : donnée manquante ne pénalise pas)."""
    return v is None or v < seuil


def _screen_ratios(re, rc_a, rl, re_c24, rc_c24, rl_c24,
                   re_c36, rc_c36, rl_c36, halal):
    """Évalue les 4 standards et retourne (résultats dict, n_pass)."""
    djim   = halal and _chk(re,0.33)    and _chk(rc_a,0.50)  and _chk(rl,0.33)
    ftse   = halal and _chk(re,0.33)    and _chk(rc_c24,0.33) and _chk(rl_c24,0.33)
    sp     = halal and _chk(re_c36,0.33)and _chk(rc_c36,0.49) and _chk(rl_c36,0.33)
    aaoifi = halal and _chk(re,0.33)    and _chk(rl,0.33)

    n_pass = sum([djim, ftse, sp, aaoifi])
    return {
        "DJIM":   {"pass": djim},
        "FTSE":   {"pass": ftse},
        "S&P":    {"pass": sp},
        "AAOIFI": {"pass": aaoifi},
    }, n_pass


def parse_screening_file(filepath):
    """
    Parse le fichier Excel de screening.
    Calcule les ratios DIRECTEMENT depuis les données brutes de Feuil1
    (dette, créances, cash, actif, cap moy 24/36) — ticker et données
    sur la même ligne, donc aucun risque de désalignement entre feuilles.

    Retourne dict {ticker: screening_result}.
    """
    df = pd.read_excel(filepath, sheet_name="Feuil1", header=None)
    results = {}

    for i in range(len(df)):
        ticker = str(df.iloc[i, 3]).strip()
        # Ligne valide = ticker alphanumérique court en col 3 + actif en col 7
        if ticker in ["nan", "NaN", ""] or pd.isna(df.iloc[i, 3]):
            continue
        if not (2 <= len(ticker) <= 6 and ticker.isalnum() and ticker.isupper()):
            continue

        def sf(col):
            try:
                v = float(df.iloc[i, col])
                return None if np.isnan(v) else v
            except Exception:
                return None

        dette = sf(4); crean = sf(5); cash = sf(6)
        actif = sf(7); cap36 = sf(12); cap24 = sf(13)

        if not actif or actif <= 0:
            continue

        # Ratios sur actif (DJIM, AAOIFI)
        re_a = _r(dette, actif)
        rc_a = _r(crean, actif)
        rl_a = _r(cash,  actif)
        # Ratios sur capitalisation moyenne (FTSE=24m, S&P=36m)
        re24 = _r(dette, cap24) if cap24 else re_a
        rc24 = _r(crean, cap24) if cap24 else rc_a
        rl24 = _r(cash,  cap24) if cap24 else rl_a
        re36 = _r(dette, cap36) if cap36 else re_a
        rc36 = _r(crean, cap36) if cap36 else rc_a
        rl36 = _r(cash,  cap36) if cap36 else rl_a

        # Exclusions fixes prioritaires
        if ticker.upper() in BANK_TICKERS or \
           ticker.upper() in ILLICIT_SECTOR_TICKERS:
            continue  # gérées par l'initialisation au démarrage

        stds, n_pass = _screen_ratios(
            re_a, rc_a, rl_a,
            re24, rc24, rl24,
            re36, rc36, rl36,
            halal=True
        )
        results[ticker] = {
            "compatible":   n_pass >= MIN_STANDARDS_PASS,
            "n_standards":  n_pass,
            "halal_sector": True,
            "standards":    stds,
            "source":       "screening_file",
            "ratios": {
                "RE_actif": round(re_a, 4) if re_a is not None else None,
                "RC_actif": round(rc_a, 4) if rc_a is not None else None,
                "RL_actif": round(rl_a, 4) if rl_a is not None else None,
            },
        }

    return results


def screen_from_fin_data(ticker, fin_data, nb_titres, cours_moy=None):
    """
    Recalcule le screening depuis les états financiers chargés.

    Exclusions :
      1. Banques conventionnelles → non compatible (halal=False)
      2. Revenus d'intérêts > 5% du CA → non compatible (Riba)

    cap24 = moyenne des CP sur les 2 dernières années disponibles
    cap36 = moyenne des CP sur les 3 dernières années disponibles
    """
    d = fin_data.get(ticker, {})
    if not d:
        return None

    yr    = max(d.keys())
    p     = d[yr]
    actif = abs(p.get("total_actif") or 0)
    if actif == 0:
        return None

    # ── 1. Exclusion banques conventionnelles ─────────────────
    is_bank = (p.get("type") == "banque") or (ticker.upper() in BANK_TICKERS)
    if is_bank:
        return {
            "compatible":    False,
            "n_standards":   0,
            "halal_sector":  False,
            "excluded":      True,
            "raison":        "Banque conventionnelle — incompatible Charia (modèle Riba)",
            "standards":     {s: {"pass": False} for s in ["DJIM","FTSE","S&P","AAOIFI"]},
            "annee":         yr,
            "source":        "fin_data",
            "ratios":        {},
        }

    # ── 2. Exclusion secteurs illicites (tabac, jeux, alcool) ─
    illicit_sector = ILLICIT_SECTOR_TICKERS.get(ticker.upper())
    if illicit_sector:
        return {
            "compatible":    False,
            "n_standards":   0,
            "halal_sector":  False,
            "excluded":      True,
            "raison":        f"Secteur illicite — {illicit_sector}",
            "standards":     {s: {"pass": False} for s in ["DJIM","FTSE","S&P","AAOIFI"]},
            "annee":         yr,
            "source":        "fin_data",
            "ratios":        {},
        }

    dette  = abs(p.get("dette_financiere") or 0)
    crean  = abs(p.get("creances_clientele") or 0)
    cash   = abs(p.get("tresorerie") or 0)

    # ── 2. Détection revenus illicites (Riba sur sociétés) ────
    ca           = abs(p.get("ca") or p.get("pnb") or 0)
    rev_fin      = abs(p.get("revenus_financiers") or 0)
    interets     = abs(p.get("interets_charges") or 0)
    # Proxy : revenus financiers (intérêts reçus) / CA total
    illicit_ratio = rev_fin / ca if ca > 0 else 0
    halal = illicit_ratio <= SEUIL_ILLICITE

    # ── 3. Capitaux propres moyens ────────────────────────────
    cp_hist = {}
    for yr_h, postes_h in d.items():
        cp_h = abs(postes_h.get("capitaux_propres") or
                   postes_h.get("capitaux_propres_mere") or 0)
        if cp_h > 0:
            cp_hist[yr_h] = cp_h

    cp_sorted = [cp_hist[y] for y in sorted(cp_hist.keys())]
    cap24 = float(np.mean(cp_sorted[-2:])) if len(cp_sorted) >= 2 else \
            (cp_sorted[-1] if cp_sorted else None)
    cap36 = float(np.mean(cp_sorted[-3:])) if len(cp_sorted) >= 3 else \
            (float(np.mean(cp_sorted[-2:])) if len(cp_sorted) >= 2 else
             (cp_sorted[-1] if cp_sorted else None))

    # ── 4. Ratios financiers ──────────────────────────────────
    re     = _r(dette, actif)
    rc_a   = _r(crean, actif)
    rl     = _r(cash,  actif)
    re_c24 = _r(dette, cap24) if cap24 else re
    rc_c24 = _r(crean, cap24) if cap24 else rc_a
    rl_c24 = _r(cash,  cap24) if cap24 else rl
    re_c36 = _r(dette, cap36) if cap36 else re
    rc_c36 = _r(crean, cap36) if cap36 else rc_a
    rl_c36 = _r(cash,  cap36) if cap36 else rl

    stds, n_pass = _screen_ratios(
        re, rc_a, rl,
        re_c24, rc_c24, rl_c24,
        re_c36, rc_c36, rl_c36,
        halal=halal
    )
    return {
        "compatible":    n_pass >= MIN_STANDARDS_PASS,
        "n_standards":   n_pass,
        "halal_sector":  halal,
        "excluded":      False,
        "illicit_ratio": round(illicit_ratio, 4),
        "standards":     stds,
        "annee":         yr,
        "source":        "fin_data",
        "cap24_source":  f"moy CP {len(cp_sorted[-2:])} ans" if cap24 else "actif (fallback)",
        "cap36_source":  f"moy CP {min(len(cp_sorted),3)} ans" if cap36 else "actif (fallback)",
        "ratios": {
            "RE_actif":  round(re,4)     if re     is not None else None,
            "RC_actif":  round(rc_a,4)   if rc_a   is not None else None,
            "RL_actif":  round(rl,4)     if rl     is not None else None,
            "RE_cap24":  round(re_c24,4) if re_c24 is not None else None,
            "RE_cap36":  round(re_c36,4) if re_c36 is not None else None,
            "Rev_fin/CA":round(illicit_ratio,4),
        },
    }


def screen_all_from_fin_data(fin_data, nb_titres, cours_moy=None):
    """Screening de tous les tickers depuis fin_data."""
    return {t: screen_from_fin_data(t, fin_data, nb_titres, cours_moy)
            for t in fin_data
            if screen_from_fin_data(t, fin_data, nb_titres, cours_moy)}


def get_charia_label(ticker, screening_results):
    """Label court pour affichage tableau."""
    r = screening_results.get(ticker)
    if r is None:
        return "—"
    if r.get("excluded"):
        raison = r.get("raison", "")
        if "Banque" in raison:
            return "🏦 Exclu (banque)"
        if "tabac" in raison.lower():
            return "🚬 Exclu (tabac)"
        if "jeux" in raison.lower():
            return "🎲 Exclu (jeux)"
        if "alcool" in raison.lower():
            return "🍺 Exclu (alcool)"
        return f"⛔ Exclu"
    n = r.get("n_standards", 0)
    if not r.get("halal_sector"):
        return f"☽ Riba {n}/4"
    return f"☪️ {n}/4" if r.get("compatible") else f"✗ {n}/4"


def get_charia_compatible_tickers(screening_results):
    """Retourne la liste des tickers compatibles Charia."""
    return [t for t, r in screening_results.items() if r and r.get("compatible")]
