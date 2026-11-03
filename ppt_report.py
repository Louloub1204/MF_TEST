"""
ppt_report.py — Générateur de rapport PowerPoint CGF Gestion
Reproduit fidèlement la charte du « Modèle de présentation - Stratégie Actions ».

10 diapositives :
  1  Sommaire
  2  Cadre méthodologique (pondération des facteurs)
  3  Facteur Value
  4  Facteur Momentum
  5  Facteurs Volatilité & Dividende
  6  Facteur Liquidité
  7  Portefeuille cible
  8  Screening Charia
  9  Backtesting de la stratégie          [slide complémentaire]
  10 Risque, corrélations & diversification [slide complémentaire]

Toutes les diapositives sont construites en positionnement absolu, comme le
modèle source. Aucun fichier .pptx externe n'est requis.
"""
import io
import datetime
import numpy as np
import pandas as pd

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION


# ══════════════════════════════════════════════════════════════
# CHARTE GRAPHIQUE CGF GESTION
# ══════════════════════════════════════════════════════════════
BLEU        = RGBColor(0x01, 0x4B, 0x75)   # bleu CGF principal
CYAN        = RGBColor(0x00, 0xB0, 0xF0)   # sous-titres
OR          = RGBColor(0xC4, 0x9A, 0x22)   # accent numéros
CARTE_BG    = RGBColor(0xEA, 0xF2, 0xF8)   # fond carte claire
CARTE_BORD  = RGBColor(0xE2, 0xE8, 0xF0)   # bordure carte
TEXTE       = RGBColor(0x1A, 0x2E, 0x3E)   # corps de texte
MUET        = RGBColor(0x64, 0x74, 0x8B)   # texte secondaire
CLAIR       = RGBColor(0x9E, 0xC9, 0xE8)   # texte sur fond bleu
BLANC       = RGBColor(0xFF, 0xFF, 0xFF)
LIGNE_A     = RGBColor(0xF7, 0xFB, 0xFF)   # ligne de tableau paire
LIGNE_B     = RGBColor(0xFF, 0xFF, 0xFF)   # ligne de tableau impaire
VERT        = RGBColor(0x1B, 0x7F, 0x4B)   # positif
ROUGE       = RGBColor(0xC0, 0x39, 0x2B)   # négatif

PALETTE = [
    RGBColor(0x01, 0x4B, 0x75), RGBColor(0x02, 0x77, 0xBD),
    RGBColor(0x02, 0x88, 0xD1), RGBColor(0xC4, 0x9A, 0x22),
    RGBColor(0x00, 0xAC, 0xC1), RGBColor(0xF5, 0x7F, 0x17),
    RGBColor(0x4F, 0xC3, 0xF7), RGBColor(0x7E, 0x57, 0xC2),
]

F_TITRE = "Cambria"
F_TEXTE = "Calibri"

# Géométrie (pouces)
SL_W, SL_H   = 13.333, 7.5
MARGE        = 0.60
LARGEUR      = 12.13
Y_TITRE      = 0.29
Y_SOUSTITRE  = 0.97
Y_REGLE      = 1.40
Y_CONTENU    = 1.60
H_CONTENU    = 5.20

PT_TITRE     = 29.33
PT_SOUSTITRE = 14.67
PT_CARTE_T   = 14.67
PT_SECTION   = 13.33
PT_CORPS     = 12.67
PT_TABLE     = 12.0
PT_KPI       = 37.33
PT_PETIT     = 11.0

FACTEURS = ["Value", "Momentum", "Volatilité", "Dividende", "Liquidité"]


# ══════════════════════════════════════════════════════════════
# PRIMITIVES DE MISE EN PAGE
# ══════════════════════════════════════════════════════════════

def _I(v):
    return Inches(v)


def _zone_texte(slide, x, y, w, h, texte, *, taille=PT_CORPS, police=F_TEXTE,
                gras=False, couleur=TEXTE, align=PP_ALIGN.LEFT,
                interligne=1.15, ancre=MSO_ANCHOR.TOP, italique=False):
    """Zone de texte simple, marges nulles pour un alignement exact."""
    tb = slide.shapes.add_textbox(_I(x), _I(y), _I(w), _I(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = ancre
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0

    lignes = str(texte).split("\n")
    for i, ligne in enumerate(lignes):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = interligne
        r = p.add_run()
        r.text = ligne
        r.font.name = police
        r.font.size = Pt(taille)
        r.font.bold = gras
        r.font.italic = italique
        r.font.color.rgb = couleur
    return tb


def _rect(slide, x, y, w, h, *, fond=CARTE_BG, bord=CARTE_BORD,
          ep_bord=1.0, arrondi=False):
    """Rectangle de fond (carte)."""
    forme = MSO_SHAPE.ROUNDED_RECTANGLE if arrondi else MSO_SHAPE.RECTANGLE
    sh = slide.shapes.add_shape(forme, _I(x), _I(y), _I(w), _I(h))
    if fond is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fond
    if bord is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = bord
        sh.line.width = Pt(ep_bord)
    sh.shadow.inherit = False
    sh.text_frame.text = ""
    return sh


def _entete(slide, titre, sous_titre):
    """Bandeau titre + sous-titre + filet, identique sur toutes les pages."""
    _zone_texte(slide, MARGE, Y_TITRE, LARGEUR, 0.69, titre,
                taille=PT_TITRE, police=F_TITRE, gras=True, couleur=BLEU,
                ancre=MSO_ANCHOR.MIDDLE)
    _zone_texte(slide, MARGE, Y_SOUSTITRE, LARGEUR, 0.37, sous_titre,
                taille=PT_SOUSTITRE, police=F_TEXTE, couleur=CYAN,
                ancre=MSO_ANCHOR.MIDDLE)
    ligne = slide.shapes.add_connector(
        1, _I(MARGE), _I(Y_REGLE), _I(MARGE + LARGEUR), _I(Y_REGLE))
    ligne.line.color.rgb = BLEU
    ligne.line.width = Pt(1.2)


def _pied(slide, numero, mention=""):
    """Pied de page : mention à gauche, numéro à droite."""
    if mention:
        _zone_texte(slide, MARGE, 6.95, 7.0, 0.40, mention,
                    taille=9, couleur=MUET, ancre=MSO_ANCHOR.MIDDLE)
    _zone_texte(slide, 8.73, 6.95, 4.0, 0.40, str(numero),
                taille=10, couleur=MUET, align=PP_ALIGN.RIGHT,
                ancre=MSO_ANCHOR.MIDDLE)


def _page(prs, titre, sous_titre, numero, mention=""):
    """Crée une diapositive vierge habillée de l'en-tête et du pied."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _entete(slide, titre, sous_titre)
    _pied(slide, numero, mention)
    return slide


def _carte(slide, x, y, w, h, titre=None, corps=None, *,
           fond=CARTE_BG, c_titre=BLEU, c_corps=TEXTE, pt_corps=PT_CORPS):
    """Carte avec titre et corps de texte."""
    _rect(slide, x, y, w, h, fond=fond,
          bord=CARTE_BORD if fond == CARTE_BG else fond)
    y_int = y + 0.16
    if titre:
        _zone_texte(slide, x + 0.27, y_int, w - 0.54, 0.40, titre,
                    taille=PT_CARTE_T, police=F_TITRE, gras=True,
                    couleur=c_titre, ancre=MSO_ANCHOR.MIDDLE)
        y_int += 0.45
    if corps:
        _zone_texte(slide, x + 0.27, y_int, w - 0.54, y + h - y_int - 0.16,
                    corps, taille=pt_corps, couleur=c_corps, interligne=1.25)


def _kpi(slide, x, y, w, valeur, libelle, *, h=1.27, couleur=BLEU):
    """Bloc chiffre-clé."""
    _rect(slide, x, y, w, h)
    _zone_texte(slide, x, y + 0.03, w, 0.67, str(valeur),
                taille=PT_KPI, police=F_TITRE, gras=True, couleur=couleur,
                align=PP_ALIGN.CENTER, ancre=MSO_ANCHOR.MIDDLE)
    _zone_texte(slide, x, y + 0.69, w, 0.46, libelle,
                taille=PT_TABLE, couleur=MUET, align=PP_ALIGN.CENTER,
                ancre=MSO_ANCHOR.TOP, interligne=1.0)


def _tableau(slide, x, y, w, h, entetes, lignes, *, largeurs=None,
             pt=PT_TABLE, aligns=None, couleurs_valeur=None):
    """
    Tableau à en-tête bleu et lignes alternées.
    couleurs_valeur : dict {(ligne, colonne): RGBColor} pour colorer des cellules.
    """
    n_l, n_c = len(lignes) + 1, len(entetes)
    forme = slide.shapes.add_table(n_l, n_c, _I(x), _I(y), _I(w), _I(h))
    tbl = forme.table
    tbl.first_row = False
    tbl.horz_banding = False

    if largeurs:
        total = sum(largeurs)
        for i, lw in enumerate(largeurs):
            tbl.columns[i].width = Emu(int(_I(w) * lw / total))

    h_ligne = h / n_l
    for r in range(n_l):
        tbl.rows[r].height = _I(h_ligne)

    def _remplir(cell, texte, *, gras, couleur, fond, align):
        cell.fill.solid()
        cell.fill.fore_color.rgb = fond
        cell.margin_left = _I(0.08)
        cell.margin_right = _I(0.06)
        cell.margin_top = cell.margin_bottom = 0
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = align
        r = p.add_run()
        r.text = "" if texte is None else str(texte)
        r.font.name = F_TEXTE
        r.font.size = Pt(pt)
        r.font.bold = gras
        r.font.color.rgb = couleur

    aligns = aligns or [PP_ALIGN.LEFT] * n_c
    for c, titre in enumerate(entetes):
        _remplir(tbl.cell(0, c), titre, gras=True, couleur=BLANC,
                 fond=BLEU, align=aligns[c])

    for i, ligne in enumerate(lignes):
        fond = LIGNE_A if i % 2 == 0 else LIGNE_B
        for c, valeur in enumerate(ligne):
            coul = TEXTE
            if couleurs_valeur and (i, c) in couleurs_valeur:
                coul = couleurs_valeur[(i, c)]
            _remplir(tbl.cell(i + 1, c), valeur, gras=False, couleur=coul,
                     fond=fond, align=aligns[c])
    return tbl


def _style_axes(chart, *, grille_valeur=True, pt=10):
    """Habillage commun des axes : police, couleur, grille discrète."""
    chart.font.name = F_TEXTE
    chart.font.size = Pt(pt)
    chart.font.color.rgb = MUET
    try:
        ca = chart.category_axis
        ca.has_major_gridlines = False
        ca.tick_labels.font.size = Pt(pt)
        ca.tick_labels.font.name = F_TEXTE
        ca.tick_labels.font.color.rgb = TEXTE
        ca.format.line.color.rgb = CARTE_BORD
    except Exception:
        pass
    try:
        va = chart.value_axis
        va.has_major_gridlines = grille_valeur
        if grille_valeur:
            va.major_gridlines.format.line.color.rgb = CARTE_BORD
            va.major_gridlines.format.line.width = Pt(0.5)
        va.tick_labels.font.size = Pt(pt - 1)
        va.tick_labels.font.name = F_TEXTE
        va.tick_labels.font.color.rgb = MUET
        va.format.line.fill.background()
    except Exception:
        pass


def _barres(slide, x, y, w, h, categories, valeurs, *,
            couleur=BLEU, fmt="0.000", pt=10):
    """Barres horizontales — rang 1 en haut."""
    cats = list(categories)[::-1]
    vals = list(valeurs)[::-1]
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series("Score", tuple(float(v) for v in vals))
    gf = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED,
                                _I(x), _I(y), _I(w), _I(h), cd)
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = False
    plot = ch.plots[0]
    plot.gap_width = 45
    serie = plot.series[0]
    serie.format.fill.solid()
    serie.format.fill.fore_color.rgb = couleur
    serie.format.line.fill.background()
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(pt - 1)
    dl.font.name = F_TEXTE
    dl.font.color.rgb = MUET
    _style_axes(ch, grille_valeur=False, pt=pt)
    return ch


def _anneau(slide, x, y, w, h, categories, valeurs, *, fmt="0%"):
    """
    Anneau (doughnut) à pastilles colorées.
    Le pourcentage est intégré au libellé de légende : certains moteurs de
    rendu n'affichent pas les étiquettes internes d'un anneau.
    """
    cd = CategoryChartData()
    cd.categories = [f"{c} · {v:.0%}" for c, v in zip(categories, valeurs)]
    cd.add_series("Poids", tuple(float(v) for v in valeurs))
    gf = slide.shapes.add_chart(XL_CHART_TYPE.DOUGHNUT,
                                _I(x), _I(y), _I(w), _I(h), cd)
    ch = gf.chart
    ch.has_title = False
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(10)
    ch.legend.font.name = F_TEXTE
    ch.legend.font.color.rgb = TEXTE
    plot = ch.plots[0]
    for i, pt_ in enumerate(plot.series[0].points):
        pt_.format.fill.solid()
        pt_.format.fill.fore_color.rgb = PALETTE[i % len(PALETTE)]
        pt_.format.line.color.rgb = BLANC
        pt_.format.line.width = Pt(1.5)
    plot.has_data_labels = False
    ch.font.name = F_TEXTE
    return ch


def _secteurs(slide, x, y, w, h, categories, valeurs, *, fmt="0.0%"):
    """Camembert sectoriel."""
    cd = CategoryChartData()
    cd.categories = list(categories)
    cd.add_series("Poids", tuple(float(v) for v in valeurs))
    gf = slide.shapes.add_chart(XL_CHART_TYPE.PIE,
                                _I(x), _I(y), _I(w), _I(h), cd)
    ch = gf.chart
    ch.has_title = False
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.RIGHT
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(10)
    ch.legend.font.name = F_TEXTE
    ch.legend.font.color.rgb = TEXTE
    plot = ch.plots[0]
    for i, pt_ in enumerate(plot.series[0].points):
        pt_.format.fill.solid()
        pt_.format.fill.fore_color.rgb = PALETTE[i % len(PALETTE)]
        pt_.format.line.color.rgb = BLANC
        pt_.format.line.width = Pt(1.5)
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(9)
    dl.font.name = F_TEXTE
    dl.font.color.rgb = TEXTE
    ch.font.name = F_TEXTE
    return ch


def _colonnes_groupees(slide, x, y, w, h, categories, series, *,
                       fmt="0.0%", pt=9):
    """Colonnes groupées — series = [(nom, [valeurs]), ...]."""
    cd = CategoryChartData()
    cd.categories = list(categories)
    for nom, vals in series:
        cd.add_series(nom, tuple(float(v) for v in vals))
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,
                                _I(x), _I(y), _I(w), _I(h), cd)
    ch = gf.chart
    ch.has_title = False
    ch.has_legend = len(series) > 1
    if ch.has_legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(10)
        ch.legend.font.name = F_TEXTE
        ch.legend.font.color.rgb = TEXTE
    plot = ch.plots[0]
    plot.gap_width = 60
    for i, s in enumerate(plot.series):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = PALETTE[i % len(PALETTE)]
        s.format.line.fill.background()
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(pt)
    dl.font.name = F_TEXTE
    dl.font.color.rgb = MUET
    _style_axes(ch, pt=10)
    return ch


def _lignes(slide, x, y, w, h, categories, series, *, pt=9):
    """Courbes — series = [(nom, [valeurs]), ...]."""
    cd = CategoryChartData()
    cd.categories = list(categories)
    for nom, vals in series:
        cd.add_series(nom, tuple(float(v) for v in vals))
    gf = slide.shapes.add_chart(XL_CHART_TYPE.LINE,
                                _I(x), _I(y), _I(w), _I(h), cd)
    ch = gf.chart
    ch.has_title = False
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.TOP
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(10)
    ch.legend.font.name = F_TEXTE
    ch.legend.font.color.rgb = TEXTE
    for i, s in enumerate(ch.plots[0].series):
        s.format.line.color.rgb = PALETTE[i % len(PALETTE)]
        s.format.line.width = Pt(2.0)
        s.smooth = False
    ch.plots[0].has_data_labels = False
    _style_axes(ch, pt=pt + 1)
    return ch


# ══════════════════════════════════════════════════════════════
# EXTRACTION DES DONNÉES
# ══════════════════════════════════════════════════════════════

def _colonne_score(df):
    if df is None:
        return None
    cols = [c for c in df.columns if "Score" in c]
    return cols[0] if cols else None


def _top(df, n=10):
    """Top n (ticker, score) d'un DataFrame de facteur."""
    col = _colonne_score(df)
    if col is None:
        return [], []
    s = pd.to_numeric(df[col], errors="coerce").dropna().nlargest(n)
    return list(s.index), [float(v) for v in s.values]


def _cours_actuel(cours_df, ticker):
    try:
        if cours_df is None or ticker not in cours_df.columns:
            return np.nan
        s = pd.to_numeric(cours_df[ticker], errors="coerce").dropna()
        return float(s.iloc[-1]) if len(s) else np.nan
    except Exception:
        return np.nan


def _repartition_sectorielle(poids, sector_map):
    """Series {secteur: poids}, triée décroissante."""
    if poids is None or len(poids) == 0 or not sector_map:
        return pd.Series(dtype=float)
    agg = {}
    for t, w in poids.items():
        sec = sector_map.get(str(t).upper(), "Autre")
        agg[sec] = agg.get(sec, 0.0) + float(w)
    return pd.Series(agg).sort_values(ascending=False)


def _abrege(secteur):
    """Libellé sectoriel compact pour les graphiques."""
    corr = {
        "SERVICES FINANCIERS":          "Serv. Financiers",
        "TELECOMMUNICATIONS":           "Télécom",
        "CONSOMMATION DE BASE":         "Conso. Base",
        "CONSOMMATION DISCRETIONNAIRE": "Conso. Discrét.",
        "SERVICES PUBLICS":             "Serv. Publics",
        "INDUSTRIELS":                  "Industriels",
        "ENERGIE":                      "Énergie",
    }
    return corr.get(str(secteur).upper(), str(secteur).title())


# ══════════════════════════════════════════════════════════════
# TEXTES PAR DÉFAUT
# ══════════════════════════════════════════════════════════════

RATIONNELS = {
    "Value": (
        "La logique de valorisation retenue postule que les titres dont le prix "
        "est inférieur à leur valeur intrinsèque surperforment à moyen terme. "
        "Sur un marché comme la BRVM, où la couverture analytique reste limitée, "
        "les écarts de valorisation persistent plus longtemps qu'ailleurs et "
        "offrent une prime de rendement à l'investisseur patient."
    ),
    "Momentum": (
        "Le facteur Momentum exploite la persistance des tendances de cours. "
        "Les rendements moyens sont mesurés sur six horizons — du journalier à "
        "l'annuel — puis agrégés. Un score élevé signale une dynamique installée "
        "que les flux acheteurs tendent à prolonger."
    ),
    "Volatilité": (
        "Un score de 1 désigne le titre le moins volatil de l'univers. "
        "L'indice est construit de façon inversée afin de récompenser la "
        "stabilité. Sur la BRVM, où la liquidité est hétérogène, ce facteur "
        "écarte les valeurs dont les cours décrochent brutalement."
    ),
    "Dividende": (
        "Mesuré par le dividend yield — dividende par action rapporté au cours "
        "moyen de la période. Ce facteur rémunère les titres distribuant un "
        "revenu régulier, dimension centrale pour les FCP à horizon de "
        "distribution."
    ),
    "Liquidité": (
        "La liquidité conditionne l'exécutabilité des décisions. Les "
        "portefeuilles doivent être bâtis autour de titres négociables dans des "
        "proportions permettant de dénouer les positions sans impact de marché "
        "excessif. Ce facteur sert également de filtre d'entrée au portefeuille."
    ),
}

METRIQUES_VALUE = [
    ("B/P",      "Capitaux propres / Cours"),
    ("E/P",      "Résultat net / Cours (inverse du PER)"),
    ("FCF/P",    "Free Cash-Flow / Cours"),
    ("CA/P",     "Chiffre d'affaires / Cours"),
    ("EBIT/P",   "Résultat d'exploitation / Cours"),
]

STANDARDS_CHARIA = [
    ("FTSE",     "FTSE Shariah — ratios rapportés à l'actif total "
                 "(dette < 33%, créances < 50%, cash < 33%)"),
    ("DJIM",     "Dow Jones Islamic Market — ratios rapportés à la "
                 "capitalisation moyenne 24 mois (seuils à 33%)"),
    ("S&P",      "S&P Shariah — ratios rapportés à la capitalisation "
                 "moyenne 36 mois (dette < 33%, créances < 49%)"),
    ("Malaysia", "Malaysia Islamic Index — dette et trésorerie rapportées "
                 "à l'actif total (seuils à 33%)"),
]


# ══════════════════════════════════════════════════════════════
# CONSTRUCTION DES DIAPOSITIVES
# ══════════════════════════════════════════════════════════════

def _slide_sommaire(prs, ctx):
    slide = _page(prs, ctx["titre"], "Structure de la présentation", 1)
    sections = [
        ("01", "Cadre méthodologique",
         "Architecture factorielle et pondération des facteurs"),
        ("02", "Analyse des facteurs",
         "Value · Momentum · Volatilité · Dividende · Liquidité"),
        ("03", "Portefeuille cible",
         f"Construction, pondérations et ventilation sectorielle "
         f"({ctx['n_ptf']} titres)"),
        ("04", "Screening Charia",
         f"Filtres islamiques et portefeuille conforme "
         f"({ctx['n_charia_ptf']} titres)"),
        ("05", "Backtesting de la stratégie",
         "Performance historique et métriques de risque"),
        ("06", "Risque et diversification",
         "Corrélations, clusters et concentration sectorielle"),
    ]
    y = 1.60
    h = 0.80
    ecart = 0.88
    for i, (num, titre, soustitre) in enumerate(sections):
        fonce = (i == 0)
        _rect(slide, MARGE, y, LARGEUR, h,
              fond=BLEU if fonce else CARTE_BG,
              bord=BLEU if fonce else CARTE_BORD)
        _zone_texte(slide, MARGE + 0.20, y + 0.05, 0.80, h - 0.10, num,
                    taille=26, police=F_TITRE, gras=True,
                    couleur=OR if fonce else BLEU,
                    align=PP_ALIGN.CENTER, ancre=MSO_ANCHOR.MIDDLE)
        _zone_texte(slide, MARGE + 1.20, y + 0.09, 6.00, 0.36, titre,
                    taille=16, police=F_TITRE, gras=True,
                    couleur=BLANC if fonce else TEXTE,
                    ancre=MSO_ANCHOR.MIDDLE)
        _zone_texte(slide, MARGE + 1.20, y + 0.44, 10.40, 0.30, soustitre,
                    taille=12, couleur=CLAIR if fonce else MUET,
                    ancre=MSO_ANCHOR.MIDDLE)
        y += ecart
    return slide


def _slide_methodologie(prs, ctx):
    slide = _page(prs, "Cadre méthodologique",
                  "Une approche quantitative multifactorielle", 2,
                  ctx["mention"])
    betas = ctx["betas"]
    actifs = [f for f in FACTEURS if betas.get(f, 0) > 0.001]

    corps = (
        "L'indice multifactoriel agrège cinq primes de risque documentées, "
        "chacune normalisée par la valeur maximale observée sur l'univers BRVM "
        "à la date de calcul.\n\n"
        "MF(t,T)  =  Σ  βᵢ · Fᵢ(t,T)          avec  Σ βᵢ = 1\n\n"
        "Chaque indice factoriel Fᵢ résulte lui-même d'une combinaison pondérée "
        "de métriques élémentaires. La pondération βᵢ arbitre entre les facteurs "
        "et constitue le principal levier de pilotage de la stratégie.\n\n"
        "Le classement obtenu alimente la formule de pondération par rang :\n"
        "α(T,t)  =  (n − r + 1) / [ n(n+1)/2 ]"
    )
    _carte(slide, MARGE, Y_CONTENU, 6.00, H_CONTENU,
           "Architecture de la stratégie", corps)

    _zone_texte(slide, 6.93, Y_CONTENU, 5.80, 0.40,
                "Pondération des facteurs", taille=PT_CARTE_T,
                police=F_TITRE, gras=True, couleur=BLEU,
                ancre=MSO_ANCHOR.MIDDLE)
    if actifs:
        _anneau(slide, 6.80, 2.05, 6.00, 4.10,
                actifs, [betas.get(f, 0) for f in actifs])
    _zone_texte(slide, 6.93, 6.30, 5.73, 0.42,
                f"{len(actifs)} facteurs actifs dans cette configuration — "
                f"{ctx['periode']}",
                taille=11, couleur=MUET, italique=True)
    return slide


def _slide_value(prs, ctx):
    slide = _page(prs, "Facteur 1 — Value",
                  "Identifier les titres décotés par rapport à leurs fondamentaux",
                  3, ctx["mention"])
    _rect(slide, MARGE, Y_CONTENU, 7.33, H_CONTENU)
    _zone_texte(slide, 0.87, 1.76, 6.80, 0.40, "Rationale d'investissement",
                taille=PT_CARTE_T, police=F_TITRE, gras=True, couleur=BLEU,
                ancre=MSO_ANCHOR.MIDDLE)
    _zone_texte(slide, 0.87, 2.20, 6.80, 1.17, RATIONNELS["Value"],
                taille=PT_CORPS, couleur=TEXTE, interligne=1.25)
    _zone_texte(slide, 0.87, 3.44, 6.80, 0.37,
                "Métriques composites et pondérations retenues",
                taille=PT_SECTION, police=F_TITRE, gras=True, couleur=BLEU,
                ancre=MSO_ANCHOR.MIDDLE)

    poids = ctx.get("poids_value") or {}
    lignes = [[nom, desc,
               f"{poids.get(nom, 0.20):.0%}" if poids else "20%"]
              for nom, desc in METRIQUES_VALUE]
    _tableau(slide, 0.87, 3.87, 6.67, 2.70,
             ["Ratio", "Description", "Poids"], lignes,
             largeurs=[2.4, 3.07, 1.2],
             aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.RIGHT])

    _rect(slide, 8.20, Y_CONTENU, 4.53, H_CONTENU)
    _zone_texte(slide, 8.40, 1.76, 4.13, 0.40, "Top 10 — Scores Value",
                taille=PT_CARTE_T, police=F_TITRE, gras=True, couleur=BLEU,
                ancre=MSO_ANCHOR.MIDDLE)
    tick, vals = _top(ctx["facteurs"].get("Value"), 10)
    if tick:
        lignes = [[str(i + 1), t, f"{v:.3f}"]
                  for i, (t, v) in enumerate(zip(tick, vals))]
        _tableau(slide, 8.33, 2.21, 4.27, min(4.30, 0.39 * (len(lignes) + 1)),
                 ["#", "Ticker", "Score"], lignes,
                 largeurs=[0.7, 1.7, 1.7],
                 aligns=[PP_ALIGN.CENTER, PP_ALIGN.LEFT, PP_ALIGN.RIGHT])
    else:
        _zone_texte(slide, 8.40, 2.40, 4.13, 0.60,
                    "Facteur non calculé dans cette configuration.",
                    taille=PT_CORPS, couleur=MUET, italique=True)
    return slide


def _slide_momentum(prs, ctx):
    slide = _page(prs, "Facteur 2 — Momentum",
                  "Exploiter la persistance des tendances de cours", 4,
                  ctx["mention"])
    _zone_texte(slide, MARGE, Y_CONTENU, 7.33, 0.40, "Top 10 — Scores Momentum",
                taille=PT_CARTE_T, police=F_TITRE, gras=True, couleur=BLEU,
                ancre=MSO_ANCHOR.MIDDLE)
    tick, vals = _top(ctx["facteurs"].get("Momentum"), 10)
    if tick:
        _barres(slide, MARGE, 2.03, 7.33, 4.73, tick, vals, couleur=BLEU)

    _rect(slide, 8.20, Y_CONTENU, 4.53, H_CONTENU)
    _zone_texte(slide, 8.40, 1.76, 4.13, 0.40, "Lecture analytique",
                taille=PT_CARTE_T, police=F_TITRE, gras=True, couleur=BLEU,
                ancre=MSO_ANCHOR.MIDDLE)

    notes = ctx["commentaires"].get("momentum") or _notes_momentum(tick, vals, ctx)
    y = 2.20
    for titre, corps in notes[:3]:
        _rect(slide, 8.33, y, 4.13, 1.40, fond=BLANC, bord=CARTE_BORD)
        _zone_texte(slide, 8.47, y + 0.05, 3.87, 0.37, titre,
                    taille=12, police=F_TITRE, gras=True, couleur=BLEU,
                    ancre=MSO_ANCHOR.MIDDLE)
        _zone_texte(slide, 8.47, y + 0.42, 3.87, 0.92, corps,
                    taille=11, couleur=TEXTE, interligne=1.15)
        y += 1.57
    return slide


def _notes_momentum(tick, vals, ctx):
    """Commentaire automatique du facteur Momentum."""
    if not tick:
        return [("Facteur non calculé",
                 "Lancez le calcul du facteur Momentum dans l'application.")]
    sm = ctx.get("sector_map") or {}
    notes = []
    t1, v1 = tick[0], vals[0]
    notes.append((
        f"{t1} en tête du classement",
        f"Avec un score de {v1:.3f}, {t1} ({_abrege(sm.get(t1.upper(),'—'))}) "
        f"affiche la dynamique de cours la plus marquée de l'univers sur "
        f"l'ensemble des horizons retenus."))
    if len(tick) >= 3:
        trio = ", ".join(tick[:3])
        notes.append((
            "Concentration du haut de tableau",
            f"{trio} occupent les trois premières places. L'écart entre le "
            f"premier et le dixième score ({v1:.3f} contre {vals[-1]:.3f}) "
            f"mesure la dispersion des tendances sur la période."))
    secteurs = [_abrege(sm.get(t.upper(), "Autre")) for t in tick]
    if secteurs:
        dominant = max(set(secteurs), key=secteurs.count)
        part = secteurs.count(dominant)
        notes.append((
            f"Poids du secteur {dominant}",
            f"{part} des 10 premières valeurs relèvent du secteur {dominant}, "
            f"signe d'une rotation sectorielle à surveiller lors du "
            f"rééquilibrage."))
    return notes


def _slide_vol_div(prs, ctx):
    slide = _page(prs, "Facteurs 3 & 4 — Volatilité & Dividende",
                  "Privilégier la stabilité · Rémunérer la distribution", 5,
                  ctx["mention"])
    for x, w, nom, titre in [
        (MARGE, 5.73, "Volatilité", "Volatilité — Faible risque, signal défensif"),
        (6.60, 6.13, "Dividende", "Dividende — Rendement locatif attractif"),
    ]:
        _rect(slide, x, Y_CONTENU, w, H_CONTENU)
        _zone_texte(slide, x + 0.20, 1.76, w - 0.40, 0.40, titre,
                    taille=PT_CARTE_T, police=F_TITRE, gras=True, couleur=BLEU,
                    ancre=MSO_ANCHOR.MIDDLE)
        _zone_texte(slide, x + 0.20, 2.21, w - 0.40, 1.15, RATIONNELS[nom],
                    taille=11.5, couleur=TEXTE, interligne=1.2)
        tick, vals = _top(ctx["facteurs"].get(nom), 8)
        if tick:
            coul = PALETTE[2] if nom == "Volatilité" else PALETTE[3]
            _barres(slide, x + 0.07, 3.49, w - 0.20, 3.13, tick, vals,
                    couleur=coul, pt=9)
        else:
            _zone_texte(slide, x + 0.20, 3.70, w - 0.40, 0.5,
                        "Facteur non calculé.", taille=PT_CORPS,
                        couleur=MUET, italique=True)
    return slide


def _slide_liquidite(prs, ctx):
    slide = _page(prs, "Facteur 5 — Liquidité",
                  "Garantir l'exécutabilité des décisions d'investissement", 6,
                  ctx["mention"])
    tick, vals = _top(ctx["facteurs"].get("Liquidité"), 10)
    if tick:
        _barres(slide, MARGE, Y_CONTENU, 7.73, 5.20, tick, vals,
                couleur=PALETTE[4])

    _rect(slide, 8.60, Y_CONTENU, 4.13, H_CONTENU)
    _zone_texte(slide, 8.80, 1.76, 3.73, 0.40, "Pourquoi ce facteur est critique",
                taille=PT_CARTE_T, police=F_TITRE, gras=True, couleur=BLEU,
                ancre=MSO_ANCHOR.MIDDLE)

    notes = ctx["commentaires"].get("liquidite") or _notes_liquidite(tick, vals, ctx)
    y = 2.21
    for titre, corps in notes[:3]:
        _zone_texte(slide, 8.80, y, 3.71, 0.37, titre,
                    taille=12, police=F_TITRE, gras=True, couleur=BLEU,
                    ancre=MSO_ANCHOR.MIDDLE)
        _zone_texte(slide, 8.80, y + 0.38, 3.71, 1.05, corps,
                    taille=11, couleur=TEXTE, interligne=1.15)
        y += 1.55
    return slide


def _notes_liquidite(tick, vals, ctx):
    if not tick:
        return [("Facteur non calculé",
                 "Lancez le calcul du facteur Liquidité dans l'application.")]
    t1, v1 = tick[0], vals[0]
    notes = [(
        f"{t1} — dominance des volumes",
        f"Avec un score de {v1:.3f}, {t1} concentre l'essentiel des échanges "
        f"de l'univers retenu. Sa liquidité réduit les coûts d'impact lors "
        f"des rééquilibrages.")]
    if len(vals) >= 2 and vals[1] > 0:
        ratio = v1 / vals[1]
        notes.append((
            "Écart structurel de liquidité",
            f"Un facteur {ratio:.1f} sépare le premier du deuxième titre, "
            f"illustrant la fragmentation de la liquidité sur la BRVM — "
            f"paramètre critique pour les fonds de taille significative."))
    seuil = ctx.get("filtre_liquidite")
    notes.append((
        "Filtre de construction",
        (f"Seuil minimal appliqué au pipeline : {seuil}. Les titres situés "
         f"sous ce niveau sont écartés avant pondération, limitant le risque "
         f"d'exécution.") if seuil else
        ("Ce facteur sert également de filtre d'entrée : les titres sous le "
         "seuil de volume retenu sont écartés avant pondération.")))
    return notes


def _slide_portefeuille(prs, ctx):
    pw = ctx["pw"]
    n = len(pw) if pw is not None else 0
    slide = _page(prs, f"Portefeuille cible — {n} titres",
                  "Construction par pondération selon le rang multifactoriel",
                  7, ctx["mention"])
    if n == 0:
        _zone_texte(slide, MARGE, 3.2, LARGEUR, 0.6,
                    "Portefeuille non encore construit — lancez le pipeline "
                    "d'optimisation dans l'application.",
                    taille=14, couleur=MUET, align=PP_ALIGN.CENTER)
        return slide

    mf = ctx["mf"]
    items = list(pw.items())
    moitie = (len(items) + 1) // 2
    blocs = [items[:moitie], items[moitie:]]
    for bi, bloc in enumerate(blocs):
        if not bloc:
            continue
        depart = 1 if bi == 0 else moitie + 1
        lignes = [[str(depart + i), str(t), f"{float(w):.2%}"]
                  for i, (t, w) in enumerate(bloc)]
        x = MARGE if bi == 0 else 3.44
        hauteur = min(5.44, 0.42 * (len(lignes) + 1))
        _tableau(slide, x, 1.57, 2.69, hauteur,
                 ["#", "Ticker", "Poids"], lignes,
                 largeurs=[0.55, 1.15, 1.0],
                 aligns=[PP_ALIGN.CENTER, PP_ALIGN.LEFT, PP_ALIGN.RIGHT],
                 pt=11)

    _zone_texte(slide, 6.47, 1.57, 6.27, 0.40, "Répartition sectorielle",
                taille=PT_CARTE_T, police=F_TITRE, gras=True, couleur=BLEU,
                ancre=MSO_ANCHOR.MIDDLE)
    sect = _repartition_sectorielle(pw, ctx.get("sector_map"))
    if len(sect):
        _secteurs(slide, 6.40, 2.00, 6.33, 3.75,
                  [_abrege(s) for s in sect.index], list(sect.values))

    texte = ctx["commentaires"].get("portefeuille")
    if not texte:
        if len(sect):
            tete = _abrege(sect.index[0])
            part = sect.iloc[0]
            texte = (
                f"{n} titres composent le portefeuille final. Le secteur "
                f"{tete} pèse {part:.1%} de l'allocation, reflet de son poids "
                f"dans la capitalisation de la cote. Le premier titre "
                f"({pw.index[0]}) porte {float(pw.iloc[0]):.2%} et le dernier "
                f"({pw.index[-1]}) {float(pw.iloc[-1]):.2%}, conformément à la "
                f"décroissance linéaire de la pondération par rang.")
        else:
            texte = (f"{n} titres composent le portefeuille final, pondérés "
                     f"par leur rang dans le classement multifactoriel.")
    _rect(slide, 6.40, 5.82, 6.33, 1.02)
    _zone_texte(slide, 6.60, 5.88, 5.93, 0.92, texte,
                taille=11, couleur=TEXTE, interligne=1.15)
    return slide


def _slide_charia(prs, ctx):
    slide = _page(prs, "Screening Charia",
                  "Évaluation sur quatre standards islamiques reconnus", 8,
                  ctx["mention"])
    ch = ctx.get("charia") or {}
    conformes = [t for t, r in ch.items() if r and r.get("compatible")]
    exclus    = [t for t, r in ch.items() if r and r.get("excluded")]
    non_conf  = [t for t, r in ch.items()
                 if r and not r.get("compatible") and not r.get("excluded")]
    pw_ch = ctx.get("pw_charia")
    n_ptf_ch = len(pw_ch) if pw_ch is not None else 0

    for i, (val, lib) in enumerate([
        (len(conformes), "Titres Charia conformes"),
        (len(exclus),    "Titres exclus (secteur ou activité)"),
        (len(non_conf),  "Non conformes (ratios financiers)"),
        (n_ptf_ch,       "Titres du portefeuille Charia"),
    ]):
        _kpi(slide, MARGE + i * 3.07, Y_CONTENU, 2.73, val, lib)

    _zone_texte(slide, MARGE, 3.04, 6.00, 0.37,
                "Grille d'évaluation — compatibilité à partir de 3 standards sur 4",
                taille=PT_SECTION, police=F_TITRE, gras=True, couleur=BLEU,
                ancre=MSO_ANCHOR.MIDDLE)
    y = 3.49
    for code, desc in STANDARDS_CHARIA:
        _rect(slide, MARGE, y, 6.00, 0.70, fond=BLANC, bord=CARTE_BORD)
        _zone_texte(slide, 0.73, y + 0.06, 1.00, 0.58, code,
                    taille=12, police=F_TITRE, gras=True, couleur=BLEU,
                    ancre=MSO_ANCHOR.MIDDLE)
        _zone_texte(slide, 1.80, y + 0.06, 4.66, 0.58, desc,
                    taille=10.5, couleur=TEXTE, interligne=1.1,
                    ancre=MSO_ANCHOR.MIDDLE)
        y += 0.78

    _rect(slide, 6.87, 2.96, 5.86, 3.84, fond=BLEU, bord=BLEU)
    libelle = f"Portefeuille Charia — {n_ptf_ch} titres (intersection)"
    if n_ptf_ch > 10:
        libelle += " · 10 premiers"
    _zone_texte(slide, 7.06, 3.10, 5.48, 0.32, libelle,
                taille=12.5, police=F_TITRE, gras=True, couleur=BLANC,
                ancre=MSO_ANCHOR.MIDDLE)
    if n_ptf_ch:
        rebase = pw_ch / pw_ch.sum()
        lignes = [[str(t), f"{float(w):.2%}"] for t, w in rebase.items()][:10]
        _tableau(slide, 7.06, 3.49, 5.48, min(3.19, 0.40 * (len(lignes) + 1)),
                 ["Titre", "Poids rebasé"], lignes,
                 largeurs=[3.0, 2.48],
                 aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT], pt=11)
    else:
        _zone_texte(slide, 7.06, 3.60, 5.48, 0.60,
                    "Aucun titre conforme retenu dans le portefeuille cible.",
                    taille=11, couleur=CLAIR, italique=True)
    return slide


# ── Diapositive complémentaire 1 ──────────────────────────────
def _slide_backtest(prs, ctx):
    slide = _page(prs, "Backtesting de la stratégie",
                  "Performance historique et métriques de risque", 9,
                  ctx["mention"])
    bt = ctx.get("backtest")
    if not bt:
        _zone_texte(slide, MARGE, 3.2, LARGEUR, 0.8,
                    "Backtesting non lancé — exécutez la simulation dans "
                    "l'onglet Backtesting de l'application pour alimenter "
                    "cette page.",
                    taille=14, couleur=MUET, align=PP_ALIGN.CENTER)
        return slide

    perf = bt["perf"]
    mets = bt["metrics"]

    # Échantillonnage mensuel pour un graphique lisible
    try:
        ech = perf.resample("ME").last().dropna(how="all")
    except Exception:
        ech = perf
    if len(ech) > 60:
        pas = max(1, len(ech) // 60)
        ech = ech.iloc[::pas]
    cats = [d.strftime("%m/%y") for d in ech.index]
    series = [(c, list(pd.to_numeric(ech[c], errors="coerce").fillna(100)))
              for c in ech.columns]

    _zone_texte(slide, MARGE, Y_CONTENU, 8.00, 0.40,
                "Performance cumulée — base 100",
                taille=PT_CARTE_T, police=F_TITRE, gras=True, couleur=BLEU,
                ancre=MSO_ANCHOR.MIDDLE)
    _lignes(slide, MARGE, 2.03, 8.00, 3.35, cats, series)

    # Métriques
    cles = ["Rendement total", "CAGR", "Volatilité annuelle",
            "Ratio de Sharpe", "Max Drawdown", "Alpha annualisé"]
    colonnes = list(perf.columns)
    entetes = ["Métrique"] + [c.replace("Portefeuille ", "Ptf ")
                              for c in colonnes]
    lignes, couleurs = [], {}
    for i, k in enumerate(cles):
        ligne = [k]
        for j, c in enumerate(colonnes):
            v = mets.get(c, {}).get(k, "—")
            ligne.append(v)
            if isinstance(v, str) and v.startswith("+"):
                couleurs[(i, j + 1)] = VERT
            elif isinstance(v, str) and v.startswith("-") and k != "Max Drawdown":
                couleurs[(i, j + 1)] = ROUGE
        lignes.append(ligne)
    _tableau(slide, MARGE, 5.52, 8.00, 1.30, entetes, lignes,
             largeurs=[2.2] + [1.5] * len(colonnes),
             aligns=[PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * len(colonnes),
             pt=10, couleurs_valeur=couleurs)

    _rect(slide, 8.90, Y_CONTENU, 3.83, H_CONTENU)
    _zone_texte(slide, 9.10, 1.76, 3.43, 0.40, "Enseignements",
                taille=PT_CARTE_T, police=F_TITRE, gras=True, couleur=BLEU,
                ancre=MSO_ANCHOR.MIDDLE)
    notes = ctx["commentaires"].get("backtest") or _notes_backtest(mets, perf)
    y = 2.21
    for titre, corps in notes[:3]:
        _zone_texte(slide, 9.10, y, 3.43, 0.34, titre,
                    taille=12, police=F_TITRE, gras=True, couleur=BLEU,
                    ancre=MSO_ANCHOR.MIDDLE)
        _zone_texte(slide, 9.10, y + 0.36, 3.43, 1.10, corps,
                    taille=10.5, couleur=TEXTE, interligne=1.15)
        y += 1.56
    return slide


def _notes_backtest(mets, perf):
    ptf = mets.get("Portefeuille MF", {})
    bm  = mets.get("BRVM Composite", {})
    ch  = mets.get("Portefeuille Charia", {})
    notes = []
    c_ptf, c_bm = ptf.get("_cagr"), bm.get("_cagr")
    if c_ptf is not None and c_bm is not None:
        ecart = (c_ptf - c_bm) * 100
        sens = "surperforme" if ecart > 0 else "sous-performe"
        notes.append((
            "Performance relative",
            f"Le portefeuille multifactoriel {sens} le BRVM Composite de "
            f"{abs(ecart):.1f} points de CAGR sur la période simulée "
            f"({c_ptf:+.1%} contre {c_bm:+.1%})."))
    sh, dd = ptf.get("_sharpe"), ptf.get("_maxdd")
    if sh is not None and dd is not None:
        notes.append((
            "Profil rendement-risque",
            f"Un ratio de Sharpe de {sh:.2f} pour un drawdown maximal de "
            f"{dd:.1%}. Le rééquilibrage trimestriel limite l'exposition aux "
            f"retournements de tendance sans multiplier les coûts de "
            f"transaction."))
    c_ch = ch.get("_cagr")
    if c_ch is not None and c_ptf is not None:
        delta = (c_ch - c_ptf) * 100
        sens = "au-dessus" if delta > 0 else "en deçà"
        notes.append((
            "Coût du filtre Charia",
            f"La version conforme Charia ressort {abs(delta):.1f} points "
            f"{sens} du portefeuille complet ({c_ch:+.1%} de CAGR), l'univers "
            f"restreint concentrant mécaniquement le risque."))
    if not notes:
        notes.append(("Simulation disponible",
                      "Les métriques détaillées figurent dans le tableau "
                      "ci-contre."))
    return notes


# ── Diapositive complémentaire 2 ──────────────────────────────
def _slide_risque(prs, ctx):
    slide = _page(prs, "Risque, corrélations et diversification",
                  "Mesurer la diversification effective du portefeuille", 10,
                  ctx["mention"])
    pw = ctx["pw"]
    sm = ctx.get("sector_map") or {}

    _zone_texte(slide, MARGE, Y_CONTENU, 7.40, 0.40,
                "Concentration sectorielle — portefeuille contre marché",
                taille=PT_CARTE_T, police=F_TITRE, gras=True, couleur=BLEU,
                ancre=MSO_ANCHOR.MIDDLE)

    sect_pf = _repartition_sectorielle(pw, sm)
    sect_bm = ctx.get("sect_benchmark")
    if sect_bm is None or not len(sect_bm):
        sect_bm = pd.Series(dtype=float)

    if len(sect_pf):
        ordre = list(sect_pf.index)
        for s in sect_bm.index:
            if s not in ordre:
                ordre.append(s)
        ordre = ordre[:7]
        cats = [_abrege(s) for s in ordre]
        series = [("Portefeuille", [float(sect_pf.get(s, 0)) for s in ordre])]
        if len(sect_bm):
            series.append(("Marché BRVM",
                           [float(sect_bm.get(s, 0)) for s in ordre]))
        _colonnes_groupees(slide, MARGE, 2.03, 7.40, 2.85, cats, series)

        lignes, couleurs = [], {}
        for i, s in enumerate(ordre):
            p = float(sect_pf.get(s, 0)) * 100
            b = float(sect_bm.get(s, 0)) * 100 if len(sect_bm) else np.nan
            if np.isnan(b):
                ecart, fleche = "—", None
            else:
                d = p - b
                fleche = "▲" if d > 0.5 else ("▼" if d < -0.5 else "≈")
                ecart = f"{fleche} {d:+.1f} pt"
                couleurs[(i, 3)] = VERT if d > 0.5 else (ROUGE if d < -0.5 else MUET)
            lignes.append([_abrege(s), f"{p:.1f}%",
                           "—" if np.isnan(b) else f"{b:.1f}%", ecart])
        _tableau(slide, MARGE, 5.05, 7.40, min(1.80, 0.34 * (len(lignes) + 1)),
                 ["Secteur", "Portefeuille", "Marché", "Écart"], lignes,
                 largeurs=[2.9, 1.6, 1.4, 1.5],
                 aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT,
                         PP_ALIGN.RIGHT, PP_ALIGN.RIGHT],
                 pt=10, couleurs_valeur=couleurs)
    else:
        _zone_texte(slide, MARGE, 2.60, 7.40, 0.6,
                    "Portefeuille non construit — lancez le pipeline "
                    "d'optimisation.",
                    taille=13, couleur=MUET, italique=True)

    _rect(slide, 8.60, Y_CONTENU, 4.13, H_CONTENU)
    _zone_texte(slide, 8.80, 1.76, 3.73, 0.40, "Corrélations et clusters",
                taille=PT_CARTE_T, police=F_TITRE, gras=True, couleur=BLEU,
                ancre=MSO_ANCHOR.MIDDLE)
    notes = ctx["commentaires"].get("risque") or _notes_risque(ctx)
    y = 2.21
    for titre, corps in notes[:3]:
        _zone_texte(slide, 8.80, y, 3.71, 0.37, titre,
                    taille=12, police=F_TITRE, gras=True, couleur=BLEU,
                    ancre=MSO_ANCHOR.MIDDLE)
        _zone_texte(slide, 8.80, y + 0.38, 3.71, 1.05, corps,
                    taille=10.5, couleur=TEXTE, interligne=1.15)
        y += 1.55
    return slide


def _notes_risque(ctx):
    notes = []
    corr = ctx.get("corr_mat")
    clusters = ctx.get("clusters")
    pw = ctx["pw"]

    if corr is not None and pw is not None and len(pw) > 1:
        tick = [t for t in pw.index if t in corr.columns]
        if len(tick) > 1:
            sub = corr.loc[tick, tick]
            masque = ~np.eye(len(sub), dtype=bool)
            moy = float(np.nanmean(sub.values[masque]))
            notes.append((
                "Corrélation moyenne du portefeuille",
                f"La corrélation moyenne entre les {len(tick)} titres retenus "
                f"s'établit à {moy:.2f}. Plus elle est faible, plus la "
                f"diversification réduit effectivement la volatilité "
                f"d'ensemble."))
    if clusters:
        n_cl = len(set(clusters.values()))
        couverts = (len({clusters[t] for t in pw.index if t in clusters})
                    if pw is not None else 0)
        notes.append((
            "Couverture des clusters",
            f"L'univers se répartit en {n_cl} groupes de titres corrélés. "
            f"Le portefeuille en couvre {couverts}, ce qui mesure sa capacité "
            f"à capter des dynamiques de marché distinctes."))
    sect = _repartition_sectorielle(pw, ctx.get("sector_map"))
    if len(sect):
        hhi = float((sect ** 2).sum())
        equiv = 1 / hhi if hhi > 0 else 0
        notes.append((
            "Concentration sectorielle",
            f"L'indice de Herfindahl sectoriel ressort à {hhi:.3f}, soit "
            f"l'équivalent de {equiv:.1f} secteurs également pondérés. "
            f"Le secteur dominant pèse {sect.iloc[0]:.1%} de l'allocation."))
    if not notes:
        notes.append((
            "Analyse non disponible",
            "Lancez le calcul des corrélations dans l'onglet dédié pour "
            "alimenter cette page."))
    return notes


# ══════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════

def build_ppt_report(*, data=None, factor_results=None, mf_scores=None,
                     pw=None, betas=None, charia_results=None,
                     sector_map=None, backtest=None, corr_mat=None,
                     clusters=None, sect_benchmark=None,
                     poids_value=None, filtre_liquidite=None,
                     commentaires=None, titre=None, periode=None,
                     slides=None):
    """
    Construit le rapport PowerPoint et retourne les octets du fichier.

    slides : liste des numéros de diapositives à inclure (1-10).
             None = toutes.
    """
    factor_results = factor_results or {}
    betas = betas or {f: 0.20 for f in FACTEURS}
    commentaires = commentaires or {}
    aujourd_hui = datetime.date.today()

    # Portefeuille Charia = intersection portefeuille cible / titres conformes
    pw_charia = None
    if pw is not None and len(pw) and charia_results:
        conformes = [t for t, r in charia_results.items()
                     if r and r.get("compatible")]
        inter = [t for t in pw.index if t in conformes]
        if inter:
            pw_charia = pw.loc[inter]

    ctx = {
        "titre":        titre or f"La stratégie multifactorielle en {aujourd_hui.year}",
        "periode":      periode or f"situation au {aujourd_hui.strftime('%d/%m/%Y')}",
        "mention":      f"CGF Gestion · Stratégie Actions BRVM · "
                        f"{aujourd_hui.strftime('%d/%m/%Y')}",
        "data":         data,
        "facteurs":     factor_results,
        "mf":           mf_scores,
        "pw":           pw,
        "pw_charia":    pw_charia,
        "betas":        betas,
        "charia":       charia_results or {},
        "sector_map":   sector_map or {},
        "backtest":     backtest,
        "corr_mat":     corr_mat,
        "clusters":     clusters,
        "sect_benchmark": sect_benchmark,
        "poids_value":  poids_value,
        "filtre_liquidite": filtre_liquidite,
        "commentaires": commentaires,
        "n_ptf":        len(pw) if pw is not None else 0,
        "n_charia_ptf": len(pw_charia) if pw_charia is not None else 0,
    }

    prs = Presentation()
    prs.slide_width = _I(SL_W)
    prs.slide_height = _I(SL_H)

    constructeurs = {
        1:  _slide_sommaire,
        2:  _slide_methodologie,
        3:  _slide_value,
        4:  _slide_momentum,
        5:  _slide_vol_div,
        6:  _slide_liquidite,
        7:  _slide_portefeuille,
        8:  _slide_charia,
        9:  _slide_backtest,
        10: _slide_risque,
    }
    choisies = sorted(slides) if slides else sorted(constructeurs)
    for num in choisies:
        fn = constructeurs.get(num)
        if fn:
            fn(prs, ctx)

    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf.getvalue()


TITRES_SLIDES = {
    1:  "Sommaire",
    2:  "Cadre méthodologique",
    3:  "Facteur Value",
    4:  "Facteur Momentum",
    5:  "Facteurs Volatilité & Dividende",
    6:  "Facteur Liquidité",
    7:  "Portefeuille cible",
    8:  "Screening Charia",
    9:  "Backtesting de la stratégie",
    10: "Risque, corrélations & diversification",
}
