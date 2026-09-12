"""
src/data/preprocessor.py
─────────────────────────
Parse raw string columns, fix sentinels, derive engineered features,
and one-hot encode genres, tags, and categories.

All logic mirrors NB02 (cells 7–21) exactly — column names, sentinel
handling, cap values, and parsing behaviour are replicated from the
executed notebook to guarantee that this module's output matches
the pre-trained models (A–E) that were fitted on that same processed data.

Public API (call in this order on raw data)
-------------------------------------------
drop_unwanted_columns(df)    → keep only the 16 raw columns needed
fix_sentinels_and_cap(df)    → fix required_age -1 sentinel; cap dlc_count
derive_simple_features(df)   → is_free, price_to_dlc_ratio,
                                platform_coverage, has_achievements, has_website
parse_counts(df)             → supported_languages_count, screenshot_count,
                                movie_count  (with caps applied)
encode_genres(df)            → 10 genre one-hot columns + genre_concentration
encode_tags(df)              → 20 tag one-hot columns
encode_categories(df)        → 12 category one-hot columns
drop_raw_sources(df)         → remove raw source columns after encoding
"""

import ast
import pandas as pd

from src.config import (
    TARGET_COL,
    COL_PRICE, COL_AGE, COL_DLC,
    COL_GENRES, COL_TAGS, COL_CATS,
    COL_MAC, COL_LINUX,
    COL_LANGS, COL_SCREENSHOTS, COL_MOVIES,
    COL_ACHIEVEMENTS, COL_WEBSITE, COL_DESC,
    AGE_SENTINEL, DLC_CAP,
    SCREENSHOT_CAP, MOVIE_CAP, LANG_COUNT_CAP,
    TOP_10_GENRES, TOP_20_TAGS, KEY_CATEGORIES,
)


# ── Private parsing helpers ───────────────────────────────────────────────────
# These are row-level functions applied via .apply().
# They are prefixed with _ to signal that callers should use the public API.

def _parse_list(val) -> list:
    """
    Parse a single string-encoded list value into a Python list of strings.

    Input formats handled:
      "['Action', 'RPG']"      → ['Action', 'RPG']
      "Action, RPG"            → ['Action', 'RPG']  (comma-split fallback)
      NaN / '' / '[]' / '{}'  → []

    Mirrors parse_list_col() from NB02 cell 15.
    Used for genres and categories (which use list format).
    """
    if pd.isna(val) or str(val).strip() in ('', '[]', '{}'):
        return []
    try:
        result = ast.literal_eval(str(val))
        if isinstance(result, list):
            return [str(x).strip() for x in result]
        elif isinstance(result, str):
            return [result.strip()]
        return []
    except Exception:
        # Fallback: split by comma and strip surrounding quotes
        return [x.strip().strip("'\"") for x in str(val).split(',') if x.strip()]


def _parse_tags(val) -> list:
    """
    Parse the tags column (stored as a dict with vote counts) into tag names.

    Input:  "{'FPS': 90857, 'Shooter': 65397}"
    Output: ['FPS', 'Shooter']

    Vote counts are discarded — only the dict keys (tag names) are used.
    Also handles list format if present.

    Mirrors parse_tags_col() from NB02 cell 17.
    """
    if pd.isna(val) or str(val).strip() in ('', '[]', '{}'):
        return []
    try:
        result = ast.literal_eval(str(val))
        if isinstance(result, dict):
            return list(result.keys())
        elif isinstance(result, list):
            return [str(x).strip() for x in result]
        return []
    except Exception:
        return []


def _safe_list_len(val) -> int:
    """
    Safely parse a string-encoded list or dict and return its length.

    Returns 0 for null, empty, or unparseable values.
    Used for screenshots and movies columns.

    Mirrors safe_list_len() from NB02 cell 13.
    """
    if pd.isna(val) or str(val).strip() in ('', '[]', '{}'):
        return 0
    try:
        result = ast.literal_eval(str(val))
        if isinstance(result, (list, dict)):
            return len(result)
        return 0
    except Exception:
        return 0


def _count_languages(val) -> int:
    """
    Parse the supported_languages column and return the language count.

    Handles both list format ['English', 'French'] and comma-separated
    plain text — both appear across different versions of the Steam dataset.

    Mirrors count_languages() from NB02 cell 13.
    """
    if pd.isna(val) or str(val).strip() in ('', '[]'):
        return 0
    try:
        result = ast.literal_eval(str(val))
        return len(result) if isinstance(result, list) else 1
    except Exception:
        # Fallback: count comma-separated tokens
        return len([x for x in str(val).split(',') if x.strip()])


# ── Public API ────────────────────────────────────────────────────────────────

def parse_list_column(series: pd.Series) -> pd.Series:
    """
    Apply list parsing to an entire column, returning a Series of Python lists.

    Wrapper around _parse_list for use with genre and category columns.

    Parameters
    ----------
    series : pd.Series
        A string-encoded list column (genres or categories).

    Returns
    -------
    pd.Series
        Each element is a Python list of strings.
    """
    return series.apply(_parse_list)


def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the 16 columns required for feature engineering and drop all others.

    Uses an explicit keep-list (not a drop-list) to prevent accidentally
    carrying forward new columns if the raw dataset is updated.

    Mirrors NB02 cell 7 (KEEP_COLS) exactly.
    Expected output: 16 columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with label already created (from loader.create_label()).

    Returns
    -------
    pd.DataFrame
        Sliced DataFrame with exactly 16 columns.

    Raises
    ------
    ValueError
        If any expected column is absent from df.
    """
    keep_cols = [
        TARGET_COL,          # reception_label (already created)
        'positive_ratio',    # kept temporarily; dropped in drop_raw_sources()
        COL_PRICE,           # price
        COL_AGE,             # required_age  (has -1 sentinel — fixed next)
        COL_DLC,             # dlc_count     (used only for price_to_dlc_ratio)
        COL_GENRES,          # genres        (parsed to one-hot in encode_genres)
        COL_TAGS,            # tags          (parsed to one-hot in encode_tags)
        COL_CATS,            # categories    (parsed to one-hot in encode_categories)
        COL_MAC,             # mac           (feeds platform_coverage)
        COL_LINUX,           # linux         (feeds platform_coverage)
        COL_LANGS,           # supported_languages  (parsed to count in parse_counts)
        COL_SCREENSHOTS,     # screenshots   (parsed to count in parse_counts)
        COL_MOVIES,          # movies        (parsed to count in parse_counts)
        COL_ACHIEVEMENTS,    # achievements  (converted to binary flag)
        COL_WEBSITE,         # website       (converted to binary flag)
        COL_DESC,            # short_description (kept for Model E in NB05)
    ]

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f'Expected columns not found in DataFrame: {missing}. '
            'Ensure loader.create_label() was called before this function.'
        )

    return df[keep_cols].copy()


def fix_sentinels_and_cap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the required_age sentinel value and cap dlc_count outliers.

    Also ensures price, required_age, dlc_count, and achievements are numeric
    (they can be object dtype in the raw CSV).

    Mirrors NB02 cell 9 exactly.

    Sentinel fix:
      required_age == -1  →  0  (means "no age restriction set by developer")
      Only 1 row is affected in the canonical dataset.

    Cap:
      dlc_count: clipped to [0, 50]
      The raw max is 3,427 — an extreme outlier that would collapse
      price_to_dlc_ratio to near-zero on legitimate games.

    Note: screenshot_count, movie_count, and supported_languages_count
    are capped inside parse_counts() immediately after creation, matching
    the order in NB02 cell 13.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # Ensure numeric dtype — raw CSV columns can arrive as object
    for col in [COL_PRICE, COL_AGE, COL_DLC, COL_ACHIEVEMENTS]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Fix required_age sentinel: -1 → 0
    df[COL_AGE] = df[COL_AGE].replace(AGE_SENTINEL, 0)

    # Cap dlc_count
    df[COL_DLC] = df[COL_DLC].clip(0, DLC_CAP)

    return df


def derive_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the five simple derived features from raw columns.

    Mirrors NB02 cell 11 exactly.

    Features created:
      is_free              price == 0 → 1, else 0
                           (free games: 65.5% well-received vs 73.4% overall)
      price_to_dlc_ratio   price / (dlc_count + 1)
                           (+1 prevents division by zero when dlc_count == 0)
      platform_coverage    (mac + linux) / 2
                           windows excluded — 100% support = zero variance
                           strongest individual feature (r = +0.107)
      has_achievements     achievements > 0 → 1, else 0
                           raw count not significant (r = -0.006, p = 0.430);
                           binary flag IS significant (r = +0.095, p < 0.001)
      has_website          website not empty → 1, else 0
                           (r = -0.021; marketing investment proxy)

    Note: genre_concentration is created in encode_genres() because it
    requires the parsed genres list, which is built there.

    The mac and linux columns are converted to 0/1 integers here but are
    NOT dropped — drop_raw_sources() removes them at the end.

    Parameters
    ----------
    df : pd.DataFrame
        After fix_sentinels_and_cap() has been called.

    Returns
    -------
    pd.DataFrame
        Input df with five new columns added.
    """
    df = df.copy()

    # is_free
    df['is_free'] = (df[COL_PRICE] == 0.0).astype(int)

    # price_to_dlc_ratio
    df['price_to_dlc_ratio'] = df[COL_PRICE] / (df[COL_DLC] + 1)

    # platform_coverage
    # mac and linux are stored as boolean-like strings ('True'/'False')
    # or integers in the raw CSV — normalise both formats to 0/1
    df[COL_MAC] = df[COL_MAC].map(
        lambda x: 1 if str(x).strip().lower() in ('true', '1', 'yes') else 0
    )
    df[COL_LINUX] = df[COL_LINUX].map(
        lambda x: 1 if str(x).strip().lower() in ('true', '1', 'yes') else 0
    )
    df['platform_coverage'] = (df[COL_MAC] + df[COL_LINUX]) / 2

    # has_achievements
    df['has_achievements'] = (df[COL_ACHIEVEMENTS] > 0).astype(int)

    # has_website
    df['has_website'] = df[COL_WEBSITE].apply(
        lambda x: 0 if (pd.isna(x) or str(x).strip() == '') else 1
    )

    return df


def parse_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse string-encoded list columns into integer counts and apply caps.

    Mirrors NB02 cell 13 exactly.

    Columns created (with caps applied immediately after parsing):
      supported_languages_count  capped at LANG_COUNT_CAP = 30
                                 (raw max 103; 75th percentile is 9)
      screenshot_count           capped at SCREENSHOT_CAP = 20
                                 (Steam hard limit is 20; values above are
                                 parsing artefacts from the dataset collector)
      movie_count                capped at MOVIE_CAP = 10
                                 (raw max 47; 75th percentile is 2)

    The raw source columns (COL_LANGS, COL_SCREENSHOTS, COL_MOVIES) are
    NOT dropped here — drop_raw_sources() removes them at the end.

    Parameters
    ----------
    df : pd.DataFrame
        After derive_simple_features() has been called.

    Returns
    -------
    pd.DataFrame
        Input df with three new integer count columns.
    """
    df = df.copy()

    df['supported_languages_count'] = (
        df[COL_LANGS].apply(_count_languages).clip(0, LANG_COUNT_CAP)
    )

    # screenshot_count — cap at 20 (Steam's hard display limit)
    df['screenshot_count'] = (
        df[COL_SCREENSHOTS].apply(_safe_list_len).clip(0, SCREENSHOT_CAP)
    )

    # movie_count
    df['movie_count'] = (
        df[COL_MOVIES].apply(_safe_list_len).clip(0, MOVIE_CAP)
    )

    return df


def encode_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the genres column and create 10 binary one-hot columns plus genre_concentration.

    Mirrors NB02 cells 14–15 exactly.

    Column naming rule:
      spaces → underscores, hyphens → underscores
      'Free To Play' → genre_Free_To_Play
      'Massively Multiplayer' → genre_Massively_Multiplayer

    genre_concentration:
      total_genres_assigned / 10, clipped to [0.0, 1.0]
      Uses the TOTAL genre count (not just top-10 matches) — a game with
      15 genres gets 1.0, a game with 3 genres gets 0.3.

    Note: 'Free To Play' in TOP_10_GENRES uses capital T in 'To' — this
    matches the case used in the raw dataset. The _parse_list function
    preserves the original case.

    Parameters
    ----------
    df : pd.DataFrame
        After parse_counts() has been called.

    Returns
    -------
    pd.DataFrame
        Input df with 11 new columns (10 genre one-hot + genre_concentration).
    """
    df = df.copy()

    genres_parsed = df[COL_GENRES].apply(_parse_list)

    # genre_concentration — based on total genre count, normalised and clipped
    df['genre_concentration'] = (genres_parsed.apply(len) / 10).clip(0, 1.0)

    # One-hot columns for top 10 genres
    for genre in TOP_10_GENRES:
        col_name = f'genre_{genre.replace(" ", "_").replace("-", "_")}'
        df[col_name] = genres_parsed.apply(
            lambda lst, g=genre: 1 if g in lst else 0
        )

    return df


def encode_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the tags column (dict format) and create 20 binary one-hot columns.

    Mirrors NB02 cells 16–17 exactly.

    tags is stored as a dict {tag_name: vote_count} — vote counts are discarded.
    Only the dict keys (tag names) are used.

    Column naming rule:
      spaces → underscores, hyphens → underscores
      'Story Rich'   → tag_Story_Rich
      'First-Person' → tag_First_Person
      '2D'           → tag_2D  (no transformation needed)

    Known limitation: 2,639 games (12.9%) have no tags — all their tag
    columns will be 0. This is documented in NB02 and accepted as a
    dataset limitation.

    Parameters
    ----------
    df : pd.DataFrame
        After encode_genres() has been called.

    Returns
    -------
    pd.DataFrame
        Input df with 20 new binary tag columns.
    """
    df = df.copy()

    tags_parsed = df[COL_TAGS].apply(_parse_tags)

    for tag in TOP_20_TAGS:
        col_name = f'tag_{tag.replace(" ", "_").replace("-", "_")}'
        df[col_name] = tags_parsed.apply(
            lambda lst, t=tag: 1 if t in lst else 0
        )

    return df


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the categories column and create 12 binary one-hot columns.

    Mirrors NB02 cells 18–19 exactly.

    Column naming rule:
      spaces → underscores, hyphens → underscores, slashes → underscores
      'Single-player'       → cat_Single_player
      'In-App Purchases'    → cat_In_App_Purchases
      'Full controller support' → cat_Full_controller_support

    Most discriminative category by reception rate:
      In-App Purchases: 43.2% well-received (28pp below baseline)

    Parameters
    ----------
    df : pd.DataFrame
        After encode_tags() has been called.

    Returns
    -------
    pd.DataFrame
        Input df with 12 new binary category columns.
    """
    df = df.copy()

    cats_parsed = df[COL_CATS].apply(_parse_list)

    for cat in KEY_CATEGORIES:
        col_name = (
            f'cat_{cat.replace(" ", "_").replace("-", "_").replace("/", "_")}'
        )
        df[col_name] = cats_parsed.apply(
            lambda lst, c=cat: 1 if c in lst else 0
        )

    return df


def drop_raw_sources(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove raw source columns after they have been encoded into features.

    Mirrors NB02 cell 21 exactly.

    Columns dropped:
      COL_GENRES, COL_TAGS, COL_CATS          → encoded to one-hot columns
      COL_MAC, COL_LINUX                       → consumed by platform_coverage
      COL_LANGS, COL_SCREENSHOTS, COL_MOVIES   → consumed by count columns
      COL_ACHIEVEMENTS, COL_WEBSITE            → consumed by binary flags
      positive_ratio                           → used only for label creation

    short_description (COL_DESC) is intentionally KEPT — it is used by
    NB05 for SBERT encoding (Model E).

    Only columns actually present in df are dropped (safety check prevents
    errors if this function is called more than once).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Final feature matrix, 56 columns when called on the full pipeline.
    """
    cols_to_drop = [
        COL_GENRES, COL_TAGS, COL_CATS,
        COL_MAC, COL_LINUX,
        COL_LANGS, COL_SCREENSHOTS, COL_MOVIES,
        COL_ACHIEVEMENTS, COL_WEBSITE,
        'positive_ratio',
    ]
    cols_present = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=cols_present)
