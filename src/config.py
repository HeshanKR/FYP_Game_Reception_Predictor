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

# ── Feature tier definitions ──────────────────────────────────────────────────
# Built programmatically from the lists above to avoid duplication.
# Exact column names match what notebook 02 produces after encoding.

# Helper: convert a raw name to a safe column identifier
def _col(prefix, name):
    return f'{prefix}_{name.replace(" ", "_").replace("-", "_")}'

# T1 — Core (14 features)
# price, required_age, 10 genre one-hots, genre_concentration
TIER1_FEATURES = (
    ['price', 'required_age']
    + [_col('genre', g) for g in TOP_10_GENRES]
    + ['genre_concentration']
)

# T2 — Core + Monetisation (16 features)
# Adds is_free and price_to_dlc_ratio
TIER2_FEATURES = TIER1_FEATURES + [
    'is_free',
    'price_to_dlc_ratio',
]

# T3 — Core + Monetisation + Content (varies — finalised after nb02 encoding)
# Adds top 20 tag one-hots, key category one-hots, and derived content features
TIER3_FEATURES = TIER2_FEATURES + (
    [_col('tag', t) for t in TOP_20_TAGS]
    + [_col('cat', c) for c in KEY_CATEGORIES]
    + [
        'platform_coverage',          # (mac + linux) / 2  — r=0.107 strongest feature
        'supported_languages_count',  # r=+0.015 ✓ weak but significant
        'screenshot_count',           # r=-0.060 ✓ negative (more ≠ better quality)
        'movie_count',                # r=-0.021 ✓ negative
        'has_website',                # r=-0.021 ✓ negative
        'has_achievements',           # r=+0.095 ✓ binary (achievements>0)
    ]
)

# T5 — Full tabular + SBERT PCA50 (T3 + 50 text dims)
TIER5_TEXT_DIMS = [f'text_dim_{i}' for i in range(PCA_COMPONENTS)]
TIER5_FEATURES  = TIER3_FEATURES + TIER5_TEXT_DIMS

# Tier → feature list lookup
TIER_FEATURES = {
    1: TIER1_FEATURES,
    2: TIER2_FEATURES,
    3: TIER3_FEATURES,
    4: TIER3_FEATURES,   # T4 = T3 (full tabular, same features as T3)
    5: TIER5_FEATURES,
}

# ── Optimisation target ───────────────────────────────────────────────────────
PRIMARY_METRIC    = 'f1_macro'     # equal weight to both classes
CLASS_WEIGHT      = 'balanced'     # upweight minority class during training
DECISION_THRESHOLD = 0.40          # starting point — tuned per model in nb04

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_PATHS = {
    'A': MODELS_DIR / 'model_a.pkl',   # T1 specialist
    'B': MODELS_DIR / 'model_b.pkl',   # T2 specialist
    'C': MODELS_DIR / 'model_c.pkl',   # T3 specialist
    'D': MODELS_DIR / 'model_d.pkl',   # T4 full tabular
    'E': MODELS_DIR / 'model_e.pkl',   # T5 SBERT fusion
}
PCA_PATH = MODELS_DIR / 'pca_transformer.pkl'

# Tier → model letter lookup
TIER_TO_MODEL = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

# ── Processed data paths ──────────────────────────────────────────────────────
FILTERED_CSV        = DATA_PROCESSED / 'games_filtered.csv'
FEATURES_T4_CSV     = DATA_PROCESSED / 'games_features_t4.csv'
SBERT_RAW_NPY       = DATA_PROCESSED / 'sbert_embeddings_raw.npy'
SBERT_PCA_NPY       = DATA_PROCESSED / 'sbert_embeddings_pca50.npy'