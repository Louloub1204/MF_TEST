"""
charia_screening.py — Screening Chariatique BRVM
4 standards : DJIM · FTSE · S&P · AAOIFI/Malaisie
Compatibilité : ≥ 3 standards sur 4 passés

Les ratios sont calculés depuis le fichier de screening (Feuil1)
ou recalculés automatiquement depuis les états financiers chargés.
"""
import pandas as pd
import numpy as np

MIN_STANDARDS_PASS = 3

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
    Utilise les ratios précalculés du feuillet TEST SCREENING UMOA
    et les tickers du feuillet Feuil1 (même ordre).

    Retourne dict {ticker: screening_result}.
    """
    df_test  = pd.read_excel(filepath, sheet_name="TEST SCREENING UMOA", header=None)
    df_feuil = pd.read_excel(filepath, sheet_name="Feuil1", header=None)

    # Extraction ordonnée des tickers depuis Feuil1
    feuil_tickers = []
    for i in range(9, len(df_feuil) - 3):
        row = df_feuil.iloc[i]
        ticker = str(row[3]).strip()
        if ticker not in ["nan", "NaN", ""] and not pd.isna(row[3]):
            feuil_tickers.append(ticker)

    # Extraction ordonnée des ratios depuis TEST SCREENING UMOA
    test_rows = []
    for i in range(5, 40):
        row = df_test.iloc[i]
        name = str(row[0]).strip()
        if pd.isna(row[1]) or name in ["", "nan"]:
            continue
        test_rows.append((i, row))

    results = {}
    for idx, (ticker, (row_i, row)) in enumerate(
            zip(feuil_tickers, test_rows)):

        def sf(col):
            try:
                v = float(row[col])
                return None if np.isnan(v) else v
            except Exception:
                return None

        # Ratios du TEST sheet (déjà normalisés)
        re     = sf(9);  rc_a  = sf(10); rl     = sf(11)  # DJIM  (actif)
        re_c24 = sf(13); rc_c24= sf(14); rl_c24 = sf(15)  # FTSE  (cap24)
        re_c36 = sf(17); rc_c36= sf(18); rl_c36 = sf(19)  # S&P   (cap36)
        ca_ill = sf(4) or 0
        halal  = (ca_ill == 0)

        stds, n_pass = _screen_ratios(
            re, rc_a, rl,
            re_c24 or re, rc_c24 or rc_a, rl_c24 or rl,
            re_c36 or re, rc_c36 or rc_a, rl_c36 or rl,
            halal
        )
        compatible = n_pass >= MIN_STANDARDS_PASS

        results[ticker] = {
            "compatible":   compatible,
            "n_standards":  n_pass,
            "halal_sector": halal,
            "standards":    stds,
            "ratios": {
                "RE_actif": round(re, 4)    if re    is not None else None,
                "RC_actif": round(rc_a, 4)  if rc_a  is not None else None,
                "RL_actif": round(rl, 4)    if rl    is not None else None,
            },
        }

    return results


def screen_from_fin_data(ticker, fin_data, nb_titres, cours_moy=None):
    """
    Recalcule le screening depuis les états financiers chargés
    (automatique lors du chargement d'une nouvelle année).

    Utilisé quand l'année des états financiers diffère de l'année
    du fichier de screening statique.
    """
    d = fin_data.get(ticker, {})
    if not d:
        return None
    yr     = max(d.keys())
    p      = d[yr]
    actif  = abs(p.get("total_actif") or 0)
    if actif == 0:
        return None

    dette  = abs(p.get("dette_financiere") or 0)
    crean  = abs(p.get("creances_clientele") or 0)
    cash   = abs(p.get("tresorerie") or 0)

    # Capitalisation moyenne depuis cours_moy + nb_titres
    cap24 = cap36 = None
    n     = nb_titres.get(ticker, 0)
    if n > 0 and cours_moy is not None and not cours_moy.empty \
            and "Date" in cours_moy.columns and ticker in cours_moy.columns:
        yrs = sorted(cours_moy["Date"].unique())
        p24  = [float(cours_moy.loc[cours_moy["Date"]==y, ticker].values[0]) * n
                for y in yrs[-2:] if pd.notna(
                cours_moy.loc[cours_moy["Date"]==y, ticker].values[0]
                if len(cours_moy.loc[cours_moy["Date"]==y, ticker]) else np.nan)]
        p36  = [float(cours_moy.loc[cours_moy["Date"]==y, ticker].values[0]) * n
                for y in yrs[-3:] if pd.notna(
                cours_moy.loc[cours_moy["Date"]==y, ticker].values[0]
                if len(cours_moy.loc[cours_moy["Date"]==y, ticker]) else np.nan)]
        if p24: cap24 = np.mean(p24)
        if p36: cap36 = np.mean(p36)

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
        halal=True  # filtre sectoriel BRVM : pas de sociétés illicites cotées
    )
    return {
        "compatible":   n_pass >= MIN_STANDARDS_PASS,
        "n_standards":  n_pass,
        "halal_sector": True,
        "standards":    stds,
        "annee":        yr,
        "source":       "fin_data",
        "ratios": {
            "RE_actif": round(re,4)   if re   is not None else None,
            "RC_actif": round(rc_a,4) if rc_a is not None else None,
            "RL_actif": round(rl,4)   if rl   is not None else None,
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
    n = r.get("n_standards", 0)
    return f"☪️ {n}/4" if r.get("compatible") else f"✗ {n}/4"


def get_charia_compatible_tickers(screening_results):
    """Retourne la liste des tickers compatibles Charia."""
    return [t for t, r in screening_results.items() if r and r.get("compatible")]
