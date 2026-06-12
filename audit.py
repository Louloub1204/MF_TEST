"""
audit.py — Script d'audit automatique post-modification
À exécuter après chaque changement sur app.py, fs_parser.py ou valuation_models.py
"""
import sys, os, ast, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "
results = []

def check(label, ok, detail=""):
    status = PASS if ok else FAIL
    results.append((status, label, detail))
    print(f"{status} {label}" + (f"  →  {detail}" if detail else ""))

def warn(label, detail=""):
    results.append((WARN, label, detail))
    print(f"{WARN} {label}" + (f"  →  {detail}" if detail else ""))

print("=" * 60)
print("AUDIT POST-MODIFICATION — CGF SMF BRVM")
print("=" * 60)

# ── 1. SYNTAXE ──────────────────────────────────────────────
print("\n── 1. Syntaxe Python ──")
import py_compile, tempfile
for fname in ["app.py", "fs_parser.py", "valuation_models.py"]:
    try:
        py_compile.compile(fname, doraise=True)
        check(f"Syntaxe {fname}", True)
    except py_compile.PyCompileError as e:
        check(f"Syntaxe {fname}", False, str(e))

# ── 2. IMPORTS EXTERNES ─────────────────────────────────────
print("\n── 2. Imports externes ──")
FS_FUNCTIONS = [
    "parse_financial_file", "merge_financial_data",
    "save_financial_db", "load_financial_db", "validate_and_fix_units",
    "detect_unit", "extract_sheet",
]
VM_FUNCTIONS = [
    "valuation_pe", "valuation_pb", "valuation_ddm", "valuation_dcf",
    "combined_price", "compute_beta", "calibrate_params",
    "auto_wacc", "auto_ke", "auto_growth",
]
try:
    import fs_parser
    for f in FS_FUNCTIONS:
        check(f"fs_parser.{f}", hasattr(fs_parser, f))
except ImportError as e:
    check("Import fs_parser", False, str(e))

try:
    import valuation_models
    for f in VM_FUNCTIONS:
        check(f"valuation_models.{f}", hasattr(valuation_models, f))
except ImportError as e:
    check("Import valuation_models", False, str(e))

# ── 3. FONCTIONS INTERNES APP.PY ────────────────────────────
print("\n── 3. Fonctions internes app.py ──")
APP_FUNCTIONS = [
    "load_data",
    "_normalise_standard", "_normalise_volatility",
    "compute_value_factor", "compute_momentum_factor",
    "compute_volatility_factor", "compute_dividend_factor",
    "compute_liquidity_factor", "compute_multifactor",
    "compute_portfolio_weights",
    "filter_liquidity", "filter_mf_percentile",
    "filter_correlation", "optimize_markowitz",
    "run_optimization_pipeline",
    "compute_scores_on_window", "build_ml_dataset",
    "optimize_betas_ml", "optimize_betas_ols",
    "optimize_betas_walkforward", "vote_majority_betas",
    "generate_manual_pdf", "load_sector_mapping",
]
with open("app.py") as f:
    src = f.read()
tree = ast.parse(src)
defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
for fn in APP_FUNCTIONS:
    check(f"def {fn}", fn in defined)

# ── 4. TABS ─────────────────────────────────────────────────
print("\n── 4. Tabs ──")
tabs_decl = re.findall(r"(t\d+(?:,\s*t\d+)*)\s*=\s*st\.tabs\(", src)
tabs_used  = re.findall(r"^with (t\d+):", src, re.MULTILINE)
if tabs_decl:
    declared = [t.strip() for t in tabs_decl[0].split(",")]
    check(f"Tabs déclarés ({len(declared)})", True, ", ".join(declared))
    check(f"Tabs utilisés ({len(tabs_used)})", len(tabs_used) == len(declared),
          f"déclarés={len(declared)} utilisés={len(tabs_used)}")
else:
    check("Tabs déclarés", False, "Déclaration non trouvée")

# ── 5. SESSION STATE ─────────────────────────────────────────
print("\n── 5. Session State ──")
REQUIRED_KEYS = [
    "data", "factor_results", "mf_scores", "pw",
    "sv_val", "sv_mom", "sv_vol", "sv_div", "sv_liq", "fin_data",
]
for key in REQUIRED_KEYS:
    initialized = any([
        f'"{key}" not in st.session_state' in src,
        f"'{key}' not in st.session_state" in src,
        f'["{key}"] =' in src,
        f"['{key}'] =" in src,
        # for-loop pattern: for k in ["data","factor_results"...]
        f'"{key}"' in src and "not in st.session_state" in src,
        f"'{key}'" in src and "not in st.session_state" in src,
        # sv_* pattern: for k, default in [("sv_val",...)...]
        f'("{key}",' in src,
        f"('{key}'," in src,
    ])
    check(f"session_state['{key}'] initialisé", initialized)

# ── 6. FICHIERS REQUIS ──────────────────────────────────────
print("\n── 6. Fichiers requis ──")
REQUIRED_FILES = [
    "app.py", "fs_parser.py", "valuation_models.py",
    "requirements.txt", "sectors.json",
]
for fname in REQUIRED_FILES:
    check(f"Fichier {fname}", os.path.exists(fname))

# ── 7. REQUIREMENTS ─────────────────────────────────────────
print("\n── 7. requirements.txt ──")
REQUIRED_PKGS = [
    "streamlit", "pandas", "numpy", "plotly",
    "openpyxl", "scipy", "scikit-learn", "reportlab",
]
with open("requirements.txt") as f:
    reqs = f.read().lower()
for pkg in REQUIRED_PKGS:
    check(f"Package {pkg}", pkg.lower() in reqs)

# ── 8. INTÉGRATIONS CLÉS ────────────────────────────────────
print("\n── 8. Intégrations clés ──")
checks_8 = [
    ("SECTOR_MAP chargé",        "SECTOR_MAP = load_sector_mapping()" in src),
    ("sys.path _APP_DIR",        "_APP_DIR" in src and "sys.path" in src),
    ("VALUATION_AVAILABLE guard","VALUATION_AVAILABLE" in src),
    ("ML_AVAILABLE guard",       "ML_AVAILABLE" in src),
    ("json importé",             "import json" in src),
    ("os importé",               "import os" in src),
    ("sectors.json chargé",      "sectors.json" in src),
    ("financial_db.json path",   "VALUATION_DB_PATH" in src),
    ("Beta bench_cols exclus",   "BENCH_COLS" in open("valuation_models.py").read()),
    ("detect_unit million",      "million" in open("fs_parser.py").read()),
    ("merge_financial_data",     "def merge_financial_data" in open("fs_parser.py").read()),
    ("validate_and_fix_units",   "def validate_and_fix_units" in open("fs_parser.py").read()),
    ("vote_majority_betas",      "def vote_majority_betas" in src),
    ("3 approches OLS+WF+ML",    "optimize_betas_ols" in src and "optimize_betas_walkforward" in src),
    ("Momentum plage unique",    "mom_global_start" in src),
    ("Backtesting tab",          "run_backtest" in src and "bt_result" in src),
    ("Corrélations tab",         "compute_correlation_matrix" in src),
    ("Clustering function",       "cluster_tickers" in src),
    ("Sector allocation",         "sector_allocation" in src),
    ("Sector performance",        "sector_performance" in src),
    ("charia_screening importé",  "CHARIA_AVAILABLE" in src),
    ("github_storage importé",    "GITHUB_STORAGE" in src),
    ("github auto-save SMF",       "save_smf_data" in src),
    ("github auto-save fin_data",  "save_financial_db_github" in src),
    ("github startup load",        "github_loaded" in src),
    ("parse_charia appelé",        "parse_charia" in src),
    ("charia_results session",     "charia_results" in src),
    ("Charia portefeuille tab7",   "get_charia_compatible_tickers" in src),
    ("Charia col valorisation",    "☪️ Charia" in src),
]
for label, ok in checks_8:
    check(label, ok)

# ── RÉSUMÉ ──────────────────────────────────────────────────
print("\n" + "=" * 60)
n_ok   = sum(1 for r in results if r[0] == PASS)
n_fail = sum(1 for r in results if r[0] == FAIL)
n_warn = sum(1 for r in results if r[0] == WARN)
print(f"RÉSULTAT : {n_ok} ✅  {n_fail} ❌  {n_warn} ⚠️")
if n_fail == 0:
    print("🎉 Audit réussi — prêt pour git push")
else:
    print("🚨 Corriger les ❌ avant de pusher")
print("=" * 60)
sys.exit(0 if n_fail == 0 else 1)
