"""
Parser des États Financiers BRVM
Supporte : banques (BCEAO), sociétés industrielles (SYSCOA/IFRS)
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ── Unités de conversion vers FCFA ────────────────────────────
UNIT_FACTORS = {
    "M FCFA":    1_000_000,
    "K FCFA":    1_000,
    "Mds FCFA":  1_000_000_000,
    "FCFA":      1,
}

# ── Postes à extraire (label partiel → clé standard) ──────────
# Banques
BANK_ROWS = {
    "PRODUIT NET BANCAIRE":                "pnb",
    "RÉSULTAT BRUT D'EXPLOITATION":        "rbe",
    "RÉSULTAT D'EXPLOITATION":             "rex",
    "RÉSULTAT AVANT IMPÔT":                "rai",
    "RÉSULTAT NET":                        "rn",
    "Impôts sur les bénéfices":            "impots",
    "Charges générales d'exploitation":    "charges_generales",
    "Dotations aux amortissements":        "amortissements",
    "Intérêts et charges assimilées":      "interets_charges",
    "Coût net du risque":                  "cout_risque",
    # Bilan
    "TOTAL ACTIF":                         "total_actif",
    "Créances sur la clientèle":           "creances_clientele",
    "Immobilisations corporelles":         "immo_corpo",
    "Capitaux propres et ressources":      "capitaux_propres",
    "Dettes à l'égard de la clientèle":   "dettes_clientele",
    "Dettes interbancaires":               "dette_financiere",
    "Emprunts et titres émis":             "dettes_lp",
}

# Sociétés (SYSCOA + IFRS)
CORP_ROWS = {
    "Chiffre d'affaires":                  "ca",
    "CHIFFRE D'AFFAIRES":                  "ca",
    "Ventes marchandises":                 "ventes_march",
    "Ventes produits fabriqués":           "ventes_prod",
    "Travaux, services vendus":            "services",
    "RÉSULTAT D'EXPLOITATION":             "rex",
    "Résultat d'exploitation":             "rex",
    "RÉSULTAT NET":                        "rn",
    "RÉSULTAT NET CONSOLIDÉ":              "rn",
    "Résultat net":                        "rn",
    "Impôt sur les sociétés":              "impots",
    "Impôts sur les bénéfices":            "impots",
    "Dotations aux amortissements":        "amortissements",
    "Résultat financier":                  "resultat_financier",
    "Intérêts et charges":                 "interets_charges",
    # Bilan
    "TOTAL ACTIF":                         "total_actif",
    "Immobilisations corporelles":         "immo_corpo",
    "Immobilisations incorporelles":       "immo_incorpo",
    "TOTAL CAPITAUX PROPRES":              "capitaux_propres",
    "Capitaux propres et ressources":      "capitaux_propres",
    "Capitaux propres mère":               "capitaux_propres_mere",
    "Passifs financiers non courants":     "dette_lp",
    "Passifs financiers courants":         "dette_cp",
    "Dettes représentées par un titre":    "obligations",
    "RN/action de base":                   "eps",
    "Stocks":                              "stocks",
    "Disponibilités":                      "tresorerie",
}

BANK_TICKERS = {"SGBC","SIBC","NSBC","ECOC","BICC","BOAB","BOAS","BOABF","BOAM"}


def detect_unit(header_text):
    """
    Détecte l'unité depuis la ligne de titre de la feuille.
    Gère les variantes françaises : 'millions', 'milliers', 'milliards'.
    Retourne (label, facteur_vers_FCFA).
    """
    t = str(header_text).lower()
    if "milliard" in t:
        return "Mds FCFA", 1_000_000_000
    if "million" in t:
        return "M FCFA", 1_000_000
    if "millier" in t:
        return "K FCFA", 1_000
    # Fallback : cherche les labels exacts
    for unit, factor in UNIT_FACTORS.items():
        if unit in str(header_text):
            return unit, factor
    return "FCFA", 1


def extract_sheet(df_raw, ticker):
    """
    Extrait les postes financiers d'une feuille individuelle.
    Retourne dict {year: {poste: valeur_en_FCFA}}
    """
    unit_str, factor = detect_unit(df_raw.iloc[0, 0])

    is_bank  = ticker in BANK_TICKERS
    row_map  = BANK_ROWS if is_bank else CORP_ROWS

    # Colonnes d'années (ligne 1)
    year_cols = {}
    for col in range(1, df_raw.shape[1]):
        val = df_raw.iloc[1, col]
        try:
            yr = int(float(val))
            if 2000 <= yr <= 2100:
                year_cols[yr] = col
        except (ValueError, TypeError):
            pass

    if not year_cols:
        return {}

    result = {yr: {} for yr in year_cols}

    for row_idx in range(2, df_raw.shape[0]):
        label = str(df_raw.iloc[row_idx, 0]).strip()

        # Match partiel sur les labels cibles
        for pattern, key in row_map.items():
            if pattern.upper() in label.upper():
                for yr, col in year_cols.items():
                    val = df_raw.iloc[row_idx, col]
                    try:
                        v = float(val)
                        if not np.isnan(v):
                            # Stocker en FCFA
                            if key not in result[yr]:  # premier match gagne
                                result[yr][key] = v * factor
                    except (TypeError, ValueError):
                        pass
                break  # un seul match par ligne

    # Calculs dérivés
    for yr in result:
        d = result[yr]

        # CA total (banque = PNB, société = somme ventes)
        if is_bank:
            if "pnb" in d:
                d["ca"] = d["pnb"]
        else:
            if "ca" not in d:
                ca_comp = sum(d.get(k, 0) for k in ["ventes_march","ventes_prod","services"])
                if ca_comp > 0:
                    d["ca"] = ca_comp

        # Dette financière totale (sociétés)
        if not is_bank and "dette_financiere" not in d:
            d["dette_financiere"] = d.get("dette_lp", 0) + d.get("dette_cp", 0)

        # Ajoute le type de société
        d["type"] = "banque" if is_bank else "societe"
        d["unite"] = unit_str

    return result


def parse_financial_file(filepath):
    """
    Parse l'ensemble du fichier Excel états financiers.
    Retourne dict {ticker: {year: {postes}}}
    """
    xl = pd.ExcelFile(filepath)
    skip_sheets = {"SYNTHÈSE"}
    data = {}

    # Lire la synthèse pour les secteurs et unités
    synth = pd.read_excel(xl, sheet_name="SYNTHÈSE", header=None)
    sectors = {}
    for _, row in synth.iterrows():
        t = str(row.get(0, "")).strip()
        s = str(row.get(1, "")).strip()
        u = str(row.get(2, "")).strip()
        if t and s and t not in ["Société","NaN","nan"] and len(t) <= 6:
            sectors[t] = {"secteur": s, "unite_synth": u}

    for sheet in xl.sheet_names:
        if sheet in skip_sheets:
            continue
        ticker = sheet.strip()
        try:
            df_raw = pd.read_excel(xl, sheet_name=sheet, header=None)
            extracted = extract_sheet(df_raw, ticker)
            if extracted:
                data[ticker] = extracted
                for yr in data[ticker]:
                    data[ticker][yr]["secteur"] = sectors.get(ticker, {}).get("secteur", "Autre")
        except Exception as e:
            print(f"⚠️ Erreur {ticker}: {e}")

    return data


def merge_financial_data(existing: dict, new_data: dict) -> dict:
    """
    Fusionne les nouvelles données dans la base existante.
    Logique : nouvelles années ajoutées, années existantes enrichies
    (nouvelles clés ajoutées, valeurs existantes conservées sauf si None).
    """
    merged = {t: dict(yrs) for t, yrs in existing.items()}

    for ticker, years in new_data.items():
        if ticker not in merged:
            merged[ticker] = {}
        for year, postes in years.items():
            if year not in merged[ticker]:
                merged[ticker][year] = postes
            else:
                for k, v in postes.items():
                    if k not in merged[ticker][year] or merged[ticker][year][k] is None:
                        merged[ticker][year][k] = v
    return merged


def validate_and_fix_units(data, nb_titres, cours_df,
                           pe_min=0.5, pe_max=150):
    """
    Valide la cohérence des unités en vérifiant que le P/E implicite
    (cours / EPS) est dans une plage raisonnable [pe_min, pe_max].

    Si le P/E implicite est hors plage et que diviser les valeurs par 1000
    donne un P/E raisonnable → corrige toutes les valeurs numériques du ticker.

    Retourne data corrigé + dict des corrections appliquées.
    """
    corrections = {}
    BENCH = {'Unnamed: 0','ANNEE','JOUR','Unnamed: 62','.BRVMCI','BRVM30',
             'BRVM PREST','BRVM-PRINC','BRVM-C TR','BRVM-CB','BRVM-CD',
             'BRVM-ENER','BRVM-SFIN','BRVM-SPUB','Date'}
    ticker_cols = [c for c in cours_df.columns if c not in BENCH]
    cours_clean  = cours_df[ticker_cols].apply(pd.to_numeric, errors="coerce")

    for ticker, years in data.items():
        yr  = max(years.keys())
        rn  = years[yr].get("rn", 0)
        n   = nb_titres.get(ticker, 0)
        if not n or not rn or rn <= 0:
            continue

        # Cours actuel
        if ticker not in cours_clean.columns:
            continue
        s = cours_clean[ticker].dropna()
        if s.empty:
            continue
        cours_act = float(s.iloc[-1])
        if cours_act <= 0:
            continue

        eps = rn / n
        pe  = cours_act / eps if eps != 0 else 0

        if pe_min <= pe <= pe_max:
            continue  # valeurs cohérentes

        # Essai division par 1000
        pe2 = cours_act / (eps / 1000) if eps != 0 else 0
        if pe_min <= pe2 <= pe_max:
            # Corriger toutes les valeurs numériques de toutes les années
            for yr2, postes in data[ticker].items():
                for k, v in postes.items():
                    if isinstance(v, float) and not np.isnan(v) and \
                       k not in ("type", "unite", "secteur"):
                        data[ticker][yr2][k] = v / 1000
            corrections[ticker] = {
                "action":   "÷1000",
                "pe_avant": round(pe, 1),
                "pe_apres": round(pe2, 1),
            }

    return data, corrections
    """
    Fusionne les nouvelles données dans la base existante.
    Logique : nouvelles années ajoutées, années existantes enrichies
    (nouvelles clés ajoutées, valeurs existantes conservées sauf si None).
    """
    merged = {t: dict(yrs) for t, yrs in existing.items()}

    for ticker, years in new_data.items():
        if ticker not in merged:
            merged[ticker] = {}
        for year, postes in years.items():
            if year not in merged[ticker]:
                merged[ticker][year] = postes
            else:
                # Enrichissement : ajoute les nouvelles clés, ne remplace pas
                for k, v in postes.items():
                    if k not in merged[ticker][year] or merged[ticker][year][k] is None:
                        merged[ticker][year][k] = v
    return merged


def save_financial_db(data: dict, db_path: str):
    """Sauvegarde la base JSON avec metadata."""
    payload = {
        "metadata": {
            "last_updated": datetime.now().isoformat(),
            "tickers": sorted(data.keys()),
            "years": sorted({yr for t in data.values() for yr in t.keys()}),
            "count": len(data),
        },
        "data": data
    }
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_financial_db(db_path: str) -> dict:
    """Charge la base JSON. Retourne dict vide si absent."""
    if not os.path.exists(db_path):
        return {}
    with open(db_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload.get("data", payload)
    # Convertir les clés d'années en int
    result = {}
    for t, years in raw.items():
        result[t] = {int(yr): postes for yr, postes in years.items()}
    return result


# ── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    data = parse_financial_file('/mnt/user-data/uploads/Etats_Financiers_2023_2024_2025.xlsx')
    print(f"Tickers parsés : {sorted(data.keys())}")
    print(f"\nSGBC 2024 :")
    for k, v in data.get("SGBC", {}).get(2024, {}).items():
        if isinstance(v, float):
            print(f"  {k}: {v:,.0f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nSNTS 2024 :")
    for k, v in data.get("SNTS", {}).get(2024, {}).items():
        if isinstance(v, float):
            print(f"  {k}: {v:,.0f}")
        else:
            print(f"  {k}: {v}")
