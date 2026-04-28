# CGF Gestion · BRVM Multifactorial Portfolio Engine

Application Streamlit implémentant fidèlement la **Note Technique de calcul de la stratégie multifactorielle**
de CGF Gestion (10/05/2024) — Auteur : A.B.A.M. Gueye.

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fonctionnalités

### Étape 1 — Calcul des indices factoriels
Implémente les formules de la note :
- Cas général : `F_i(t,T) = Σ w_ij · m_ij / max(m_ij)`
- Cas Volatilité (inversion) : `F_i(t,T) = Σ w_ij · min(m_ij) / m_ij`

**7 facteurs** avec leurs métriques et poids exacts :
| Facteur | Métriques | Poids |
|---------|-----------|-------|
| Value | B/P, EPS/P, FCF/P, Sales/P, EBIT/EV | 10%, 50%, 20%, 10%, 10% |
| Qualité | Levier, Qual. résultat, ROA, ROE | 25% chacun |
| Dividende | Dividend Yield | 100% |
| Volatilité | Écart-type rendements | 100% (inversé) |
| Momentum | Rdt J/H/M/T/S/A | 16.67% chacun |
| Liquidité | Volume moyen | 100% |
| Taille | Cap. boursière | 100% |

### Étape 2 — Indice multifactoriel
`MF(t,T) = Σ β_i · F_i(t,T)` — poids β ajustables interactivement.

### Étape 3 — Portefeuille cible
`α(T,t) = (n − r(T,t) + 1) / (n(n+1)/2)` — pondération par rang MF.

## Structure des données

Le fichier d'entrée (Excel/CSV) doit contenir une colonne **Ticker** + les 19 métriques.
Un template téléchargeable est disponible dans l'onglet Données.

## Indices de référence
- **BRVM Composite** (défaut)
- **BRVM Prestige**
