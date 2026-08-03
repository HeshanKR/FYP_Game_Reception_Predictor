# src/config.py
# ─────────────────────────────────────────────────────────────────────────────
# PPRS — Pre-Launch Player Reception System
# Single source of truth for all constants, column names, feature tier
# definitions, model paths, and hyperparameters.
#
# Every module that needs to know which features belong to which tier,
# or where model files live, imports from here.
# Never hardcode column names or feature lists in notebooks or src modules.
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent
DATA_RAW       = ROOT_DIR / 'data' / 'raw'
DATA_PROCESSED = ROOT_DIR / 'data' / 'processed'
MODELS_DIR     = ROOT_DIR / 'models'
FIGURES_DIR    = ROOT_DIR / 'outputs' / 'figures'
RESULTS_DIR    = ROOT_DIR / 'outputs' / 'results'

# ── Raw dataset ───────────────────────────────────────────────────────────────
RAW_CSV = DATA_RAW / 'Dataset-2.csv'

# ── Label creation constants ──────────────────────────────────────────────────
MIN_REVIEW_COUNT   = 100     # games with fewer reviews are excluded (unreliable label)
POSITIVE_THRESHOLD = 0.75    # pct_pos_total / 100 >= 0.75 → label = 1
TARGET_COL         = 'reception_label'

# ── Confirmed column names (from notebook 01 exploration) ─────────────────────
# These are the actual names as they appear in Dataset-2.csv.
# COL_LABEL and COL_FILTER are not features — used only for label creation
# and data quality filtering respectively.
COL_LABEL        = 'pct_pos_total'       # 0-100 integer → /100 → binary label
COL_FILTER       = 'num_reviews_total'   # filter gate — uses -1 sentinel for no data

# Raw structured features
COL_PRICE        = 'price'
COL_AGE          = 'required_age'        # -1 sentinel present → clip(0, 18) in nb02
COL_DLC          = 'dlc_count'           # used only as input to price_to_dlc_ratio

# Multi-value string columns → parsed and one-hot encoded in notebook 02
COL_GENRES       = 'genres'              # stored as string-encoded list
COL_TAGS         = 'tags'               # stored as dict with vote counts (not a list)
COL_CATS         = 'categories'          # stored as string-encoded list

# Platform flags → combined into platform_coverage derived feature
# windows is NOT included — 100% of games support it (zero variance, dropped)
COL_MAC          = 'mac'
COL_LINUX        = 'linux'

# Columns parsed to integer counts in notebook 02
COL_LANGS        = 'supported_languages'  # → supported_languages_count
COL_SCREENSHOTS  = 'screenshots'          # → screenshot_count (parse list length)
COL_MOVIES       = 'movies'               # → movie_count (parse list length)

# Columns parsed to binary flags in notebook 02
COL_ACHIEVEMENTS = 'achievements'         # → has_achievements = (achievements > 0)
COL_WEBSITE      = 'website'              # → has_website = (website not empty)

# NLP source for Model E
COL_DESC         = 'short_description'    # SBERT encoding source; 37 games missing

# ── Outlier caps (applied in notebook 02 before feature engineering) ──────────
# price      : NO cap — tree models are robust; only 41 games above $60 (0.2%)
# achievements: NO cap — converted to binary (achievements > 0), raw value irrelevant
DLC_CAP          = 50    # prevents dlc_count=3427 distorting price_to_dlc_ratio
AGE_SENTINEL     = -1    # replace -1 with 0 (means "no age restriction set")
SCREENSHOT_CAP   = 20    # Steam hard limit is 20; values above are parsing errors
MOVIE_CAP        = 10    # max=47 is an outlier; 75th percentile is only 2
LANG_COUNT_CAP   = 30    # max=103 but 75th percentile=9; 30 covers all real cases

# ── Top 10 genres (confirmed from notebook 01) ────────────────────────────────
# These become one-hot columns: genre_Indie, genre_Action, etc.
# Ordered by frequency in the filtered dataset.
TOP_10_GENRES = [
    'Indie',                  # n=13,600  well_received=74.8%
    'Adventure',              # n=8,896   well_received=73.4%
    'Action',                 # n=8,392   well_received=68.6%
    'Casual',                 # n=6,771   well_received=75.9% (highest)
    'Simulation',             # n=5,340   well_received=65.9%
    'RPG',                    # n=5,076   well_received=67.1%
    'Strategy',               # n=4,515   well_received=65.2%
    'Free To Play',           # n=3,127   well_received=64.3%
    'Early Access',           # n=1,653   well_received=61.9%
    'Massively Multiplayer',  # n=898     well_received=36.2% (most discriminative)
]

# ── Top 20 tags (confirmed from notebook 01) ──────────────────────────────────
# Post-launch tags excluded: Walking Simulator, Classic, Memes
# These become one-hot columns: tag_Singleplayer, tag_Indie, etc.
TOP_20_TAGS = [
    'Singleplayer',   # n=11,869
    'Indie',          # n=10,091
    'Adventure',      # n=8,598
    'Action',         # n=8,158
    'Casual',         # n=6,327
    '2D',             # n=5,302
    'Simulation',     # n=4,891
    'RPG',            # n=4,733
    'Atmospheric',    # n=4,651
    'Story Rich',     # n=4,507
    'Strategy',       # n=4,493
    'Multiplayer',    # n=4,320
    'Puzzle',         # n=3,270
    'Exploration',    # n=3,147
    'First-Person',   # n=3,138
    'Anime',          # n=3,015
    'Funny',          # n=2,832
    '3D',             # n=2,817
    'Cute',           # n=2,803
    'Fantasy',        # n=2,799
]

# ── Key categories to one-hot encode (confirmed from notebook 01) ─────────────
# In-App Purchases is the most discriminative: only 43.2% well-received
# Full controller support and Steam Cloud signal higher production quality
KEY_CATEGORIES = [
    'Single-player',
    'Multi-player',
    'Co-op',
    'Steam Achievements',
    'Steam Cloud',
    'Full controller support',
    'Partial Controller Support',
    'Online Co-op',
    'Online PvP',
    'In-App Purchases',         # 43.2% well-received — most discriminative category
    'Steam Workshop',
    'Remote Play Together',
]

# ── NLP / embedding ───────────────────────────────────────────────────────────
SBERT_MODEL    = 'all-MiniLM-L6-v2'
PCA_COMPONENTS = 50
DESC_FALLBACK_MODEL = 'D'   # 37 games with no description fall back to Model D

# ── Training constants ────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
CV_FOLDS     = 5
RANDOM_STATE = 42

# ── CatBoost hyperparameters ──────────────────────────────────────────────────
# FIX 1 (added): Was missing entirely from the original config.
# Identical across all five specialist models (A–E).
# Confirmed in NB03 tuning. scale_pos_weight=0.5 upweights the minority
# class (Not Well Received). allow_writing_files=False prevents CatBoost
# writing training artefacts to disk during inference.
CATBOOST_PARAMS = {
    'depth':               8,
    'learning_rate':       0.05,
    'iterations':          1000,
    'l2_leaf_reg':         3,
    'scale_pos_weight':    0.5,
    'random_seed':         RANDOM_STATE,
    'verbose':             0,
    'allow_writing_files': False,
}

# ── Routing thresholds ────────────────────────────────────────────────────────
# FIX 2 (added): UNCERTAINTY_THRESHOLD was missing entirely from the original config.
# If Model D's max predicted probability falls below this value, the game is
# in an ambiguous zone and Model E (SBERT) is invoked as a tiebreaker.
# Selected by CV search over [0.50, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65, 0.70]
# in NB04 — best CV Macro F1 at 0.63. Used by router.py and the app.
UNCERTAINTY_THRESHOLD = 0.63

# NB03 standalone benchmark threshold only — NOT used in the routing pipeline.
DECISION_THRESHOLD = 0.40

# ── Optimisation target ───────────────────────────────────────────────────────
PRIMARY_METRIC = 'f1_macro'   # equal weight to both classes
CLASS_WEIGHT   = 'balanced'   # upweight minority class during training

# ── Feature tier definitions ──────────────────────────────────────────────────
# Built programmatically from the lists above to avoid duplication.
# Exact column names match what notebook 02 produces after encoding.

# Helper: convert a raw name to a safe column identifier
def _col(prefix, name):
    return f'{prefix}_{name.replace(" ", "_").replace("-", "_")}'

# T1 — Core (13 features)
# FIX 3 (comment only): original said "14 features" — correct count is 13.
# price, required_age, 10 genre one-hots, genre_concentration
TIER1_FEATURES = (
    ['price', 'required_age']
    + [_col('genre', g) for g in TOP_10_GENRES]
    + ['genre_concentration']
)
# len check: 2 + 10 + 1 = 13 ✓

# T2 — Core + Monetisation (15 features)
# FIX 3 (comment only): original said "16 features" — correct count is 15.
# Adds is_free and price_to_dlc_ratio
TIER2_FEATURES = TIER1_FEATURES + [
    'is_free',
    'price_to_dlc_ratio',
]
# len check: 13 + 2 = 15 ✓

# T3 — Core + Monetisation + Community labels (47 features)
# FIX 4: Original TIER3_FEATURES incorrectly included the 6 effort signals.
# T3 is community labels only: tags and categories. Effort signals belong
# exclusively in T4. model_c.pkl was trained on exactly these 47 features.
TIER3_FEATURES = TIER2_FEATURES + (
    [_col('tag', t) for t in TOP_20_TAGS]
    + [_col('cat', c) for c in KEY_CATEGORIES]
)
# len check: 15 + 20 + 12 = 47 ✓

# Effort signals — the 6 derived developer-effort features that separate T4 from T3.
# These test whether observable pre-launch effort (platforms, media count, languages)
# adds predictive signal on top of community labels alone.
# T2→T3 was the largest single gain (+0.053 Macro F1). T3→T4 added +0.022.
EFFORT_SIGNALS = [
    'platform_coverage',          # (mac + linux) / 2 — r=+0.107, strongest feature
    'has_achievements',           # binary (achievements > 0) — r=+0.095
    'has_website',                # binary (website not empty) — r=-0.021
    'supported_languages_count',  # parsed integer count — r=+0.015
    'screenshot_count',           # parsed integer count — r=-0.060
    'movie_count',                # parsed integer count — r=-0.021
]

# T4 — Full structured (53 features)
# FIX 4 (continued): T4 is T3 + effort signals. Was missing entirely from
# original config — TIER_FEATURES[4] just aliased TIER3_FEATURES, which
# made T3 and T4 identical. model_d.pkl was trained on exactly these 53 features.
TIER4_FEATURES = TIER3_FEATURES + EFFORT_SIGNALS
# len check: 47 + 6 = 53 ✓

# T5 — Full structured + SBERT PCA50 (103 features)
# FIX 5: Original built T5 from TIER3_FEATURES instead of TIER4_FEATURES.
# T5 must be T4 + text dims. model_e.pkl was trained on exactly these 103 features.
TIER5_TEXT_DIMS = [f'text_dim_{i}' for i in range(PCA_COMPONENTS)]
TIER5_FEATURES  = TIER4_FEATURES + TIER5_TEXT_DIMS
# len check: 53 + 50 = 103 ✓

# Tier → feature list lookup
# Used by router.py and the Streamlit app to build the correct feature vector.
TIER_FEATURES = {
    1: TIER1_FEATURES,    # 13 features — Model A
    2: TIER2_FEATURES,    # 15 features — Model B
    3: TIER3_FEATURES,    # 47 features — Model C
    4: TIER4_FEATURES,    # 53 features — Model D  (FIX 4: was TIER3_FEATURES)
    5: TIER5_FEATURES,    # 103 features — Model E (FIX 5: was TIER3_FEATURES + text)
}

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_PATHS = {
    'A': MODELS_DIR / 'model_a.pkl',   # T1 specialist — 13 features
    'B': MODELS_DIR / 'model_b.pkl',   # T2 specialist — 15 features
    'C': MODELS_DIR / 'model_c.pkl',   # T3 specialist — 47 features
    'D': MODELS_DIR / 'model_d.pkl',   # T4 full structured — 53 features
    'E': MODELS_DIR / 'model_e.pkl',   # T5 SBERT fusion — 103 features
}
PCA_PATH = MODELS_DIR / 'pca_transformer.pkl'

# Tier → model letter lookup
TIER_TO_MODEL = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

# ── Processed data paths ──────────────────────────────────────────────────────
FILTERED_CSV    = DATA_PROCESSED / 'games_filtered.csv'
FEATURES_T4_CSV = DATA_PROCESSED / 'games_features_t4.csv'
SBERT_RAW_NPY   = DATA_PROCESSED / 'sbert_embeddings_raw.npy'
SBERT_PCA_NPY   = DATA_PROCESSED / 'sbert_embeddings_pca50.npy'
