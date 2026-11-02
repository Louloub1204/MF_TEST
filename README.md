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

## Moteur commun de valeur fondamentale (aligné avec l'application des fiches de valeur)

L'onglet 📈 Valorisation utilise `valorisation_commune.py`, **fichier identique** à
`fiche_valeur/valorisation_commune.py` de l'application CGF-Fiche-Valeurs. Les deux applications lisent la base SMF et le
classeur des états financiers avec le même lecteur et calculent la valeur fondamentale avec la même méthode et les
mêmes paramètres : à fichiers et date d'arrêté identiques, elles affichent la même valeur. L'empreinte du moteur est
affichée dans l'onglet pour le vérifier ; toute modification du fichier doit être recopiée dans l'autre dépôt.

Le classeur des états financiers est désormais lu par ce lecteur commun (unités du titre de chaque feuille, sous-totaux
calculés par formule, part du groupe pour les comptes consolidés) ; une nouvelle publication remplace l'exercice
correspondant dans la base au lieu de seulement la compléter. L'ancienne correction automatique des unités par le PER
est abandonnée. La base SMF et le dernier classeur chargé sont conservés sur GitHub (`data/base_commune.json`,
`data/etats_financiers_commun.json`).
