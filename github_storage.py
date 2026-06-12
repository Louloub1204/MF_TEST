"""
github_storage.py — Persistance des données via l'API GitHub
Lit/écrit les fichiers de données directement dans le dépôt GitHub.

Fichiers persistés :
  data/smf_data.parquet       ← cours, nb_titres, dividendes, moyenne_cours
  data/financial_db.json      ← états financiers parsés
  data/charia_results.json    ← résultats screening Charia

Configuration requise dans .streamlit/secrets.toml :
  [github]
  token  = "ghp_..."
  repo   = "username/smf-brvm"
  branch = "main"
"""
import json
import base64
import io
import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime


# ── Constantes ────────────────────────────────────────────────
DATA_FILES = {
    "smf":     "data/smf_data.parquet",
    "fin_db":  "data/financial_db.json",
    "charia":  "data/charia_results.json",
}


def _get_config():
    """Récupère la config GitHub depuis les Streamlit Secrets."""
    try:
        cfg = st.secrets.get("github", {})
        token  = cfg.get("token")
        repo   = cfg.get("repo")
        branch = cfg.get("branch", "main")
        if not token or not repo:
            return None, None, None
        return token, repo, branch
    except Exception:
        return None, None, None


def _headers(token):
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }


def _get_file_sha(token, repo, path, branch):
    """Récupère le SHA d'un fichier existant (nécessaire pour le mettre à jour)."""
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    r = requests.get(url, headers=_headers(token),
                     params={"ref": branch}, timeout=10)
    if r.status_code == 200:
        return r.json().get("sha")
    return None


def push_file(content_bytes: bytes, path: str, commit_msg: str) -> bool:
    """
    Pousse un fichier binaire vers GitHub.
    Crée ou met à jour selon l'existence du fichier.
    Retourne True si succès.
    """
    token, repo, branch = _get_config()
    if not token:
        return False

    sha = _get_file_sha(token, repo, path, branch)
    payload = {
        "message": commit_msg,
        "content": base64.b64encode(content_bytes).decode(),
        "branch":  branch,
    }
    if sha:
        payload["sha"] = sha

    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    r   = requests.put(url, headers=_headers(token),
                       json=payload, timeout=15)
    return r.status_code in (200, 201)


def pull_file(path: str) -> bytes | None:
    """
    Télécharge un fichier depuis GitHub.
    Retourne les bytes ou None si absent.
    """
    token, repo, branch = _get_config()
    if not token:
        return None

    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    r   = requests.get(url, headers=_headers(token),
                       params={"ref": branch}, timeout=10)
    if r.status_code == 200:
        return base64.b64decode(r.json()["content"])
    return None


# ── SMF Data (cours, nb_titres, dividendes) ───────────────────

def save_smf_data(data_dict: dict) -> bool:
    """
    Sérialise et pousse les données SMF vers GitHub.
    data_dict : le dict retourné par load_data()
    """
    try:
        buf = io.BytesIO()
        # Sauvegarder chaque DataFrame comme feuille Parquet séparée
        # On encode en JSON les DataFrames puis on les empaquète
        payload = {}

        for key in ["cours", "moyenne_cours", "dividendes"]:
            df = data_dict.get(key)
            if df is not None and not df.empty:
                df_clean = df.copy()
                if hasattr(df_clean.index, 'strftime'):
                    df_clean.index = df_clean.index.strftime('%Y-%m-%d')
                payload[key] = df_clean.to_json(orient="split", date_format="iso")

        # nb_titres : dict simple
        nb = data_dict.get("nb_titres", {})
        payload["nb_titres"] = json.dumps({k: float(v) for k, v in nb.items()})

        # tickers : liste
        payload["tickers"] = json.dumps(data_dict.get("tickers", []))

        # Serialiser en JSON
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        content_bytes = content.encode("utf-8")

        ok = push_file(
            content_bytes,
            DATA_FILES["smf"],
            f"data: mise à jour SMF {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        return ok
    except Exception as e:
        st.warning(f"⚠️ Sauvegarde SMF GitHub : {e}")
        return False


def load_smf_data() -> dict | None:
    """
    Charge les données SMF depuis GitHub.
    Retourne None si absent ou erreur.
    """
    try:
        raw = pull_file(DATA_FILES["smf"])
        if not raw:
            return None

        payload = json.loads(raw.decode("utf-8"))
        result  = {}

        for key in ["cours", "moyenne_cours", "dividendes"]:
            if key in payload:
                df = pd.read_json(io.StringIO(payload[key]), orient="split")
                if key == "cours":
                    df.index = pd.to_datetime(df.index, errors="coerce")
                    df = df.sort_index()
                result[key] = df

        if "nb_titres" in payload:
            result["nb_titres"] = json.loads(payload["nb_titres"])

        if "tickers" in payload:
            result["tickers"] = json.loads(payload["tickers"])

        return result if result else None

    except Exception as e:
        st.warning(f"⚠️ Chargement SMF GitHub : {e}")
        return None


# ── Financial DB (états financiers) ──────────────────────────

def save_financial_db_github(data: dict) -> bool:
    """Pousse financial_db.json vers GitHub."""
    try:
        payload = {
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "tickers": sorted(data.keys()),
                "count": len(data),
            },
            "data": data
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        return push_file(
            content.encode("utf-8"),
            DATA_FILES["fin_db"],
            f"data: états financiers {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
    except Exception as e:
        st.warning(f"⚠️ Sauvegarde états financiers GitHub : {e}")
        return False


def load_financial_db_github() -> dict:
    """Charge financial_db.json depuis GitHub. Retourne {} si absent."""
    try:
        raw = pull_file(DATA_FILES["fin_db"])
        if not raw:
            return {}
        payload = json.loads(raw.decode("utf-8"))
        raw_data = payload.get("data", payload)
        return {t: {int(yr): postes for yr, postes in years.items()}
                for t, years in raw_data.items()}
    except Exception as e:
        st.warning(f"⚠️ Chargement états financiers GitHub : {e}")
        return {}


# ── Charia Results ─────────────────────────────────────────────

def save_charia_results(results: dict) -> bool:
    """Pousse charia_results.json vers GitHub."""
    try:
        # Nettoyer les valeurs non-sérialisables
        clean = {}
        for ticker, r in results.items():
            if r is None:
                continue
            clean[ticker] = {
                k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                    int(v)  if isinstance(v, (np.integer,)) else
                    float(v) if isinstance(v, (np.floating,)) else v)
                for k, v in r.items()
                if k not in ("_alpha","_beta","_cagr","_vol","_sharpe","_maxdd")
            }
        content = json.dumps({
            "last_updated": datetime.now().isoformat(),
            "results": clean
        }, ensure_ascii=False, indent=2)
        return push_file(
            content.encode("utf-8"),
            DATA_FILES["charia"],
            f"data: charia screening {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
    except Exception as e:
        st.warning(f"⚠️ Sauvegarde Charia GitHub : {e}")
        return False


def load_charia_results() -> dict:
    """Charge charia_results.json depuis GitHub. Retourne {} si absent."""
    try:
        raw = pull_file(DATA_FILES["charia"])
        if not raw:
            return {}
        payload = json.loads(raw.decode("utf-8"))
        return payload.get("results", {})
    except Exception as e:
        st.warning(f"⚠️ Chargement Charia GitHub : {e}")
        return {}


def is_github_configured() -> bool:
    """Vérifie si la config GitHub est disponible."""
    token, repo, _ = _get_config()
    return bool(token and repo)
