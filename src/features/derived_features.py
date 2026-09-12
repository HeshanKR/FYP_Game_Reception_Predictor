"""
src/features/derived_features.py
─────────────────────────────────
Compute the six derived engineered features from clean, already-parsed inputs.

These functions operate on DataFrames whose numeric columns are already clean —
no string parsing, no sentinel fixing. They are the inference-time counterparts
to the transformations in src/data/preprocessor.py.

The preprocessor handles the batch training pipeline (raw CSV → clean DataFrame).
This module handles the same derivations for:
  1. The Streamlit app — a single-row DataFrame built from form inputs
  2. Any code that needs to derive features from already-normalised data

All formulas mirror NB02 cell 11 exactly.

Feature reference (from NB02 ablation and NB07):
  is_free              Free games are less well-received (64.3% vs 73.4% baseline)
  price_to_dlc_ratio   Separates monetisation strategies; +1 avoids division by zero
  platform_coverage    Strongest single feature (r=+0.107); mac+linux / 2
  has_achievements     10pp reception gap (74.4% with vs 64.7% without)
  has_website          r=-0.021; marketing investment proxy (larger studios)
  genre_concentration  How spread the game's genres are; computed in encode_genres

Public API
----------
derive_monetisation_features(df) → is_free, price_to_dlc_ratio
derive_platform_features(df)     → platform_coverage
derive_content_features(df)      → has_achievements, has_website
derive_all(df)                   → all of the above in one call
"""

import pandas as pd

from src.config import (
    COL_PRICE, COL_DLC,
    COL_MAC, COL_LINUX,
    COL_ACHIEVEMENTS, COL_WEBSITE,
)


def derive_monetisation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive is_free and price_to_dlc_ratio from price and dlc_count.

    Mirrors NB02 cell 11 exactly.

    Preconditions
    -------------
    df[COL_PRICE]  must be numeric (float), already clean.
    df[COL_DLC]    must be numeric (int), already capped at DLC_CAP.

    Columns added
    -------------
    is_free              int  1 if price == 0.0, else 0
    price_to_dlc_ratio   float  price / (dlc_count + 1)
                         The +1 in the denominator prevents division by zero
                         when dlc_count == 0 (the majority case).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Input df with two new columns added.
    """
    df = df.copy()
    df['is_free'] = (df[COL_PRICE] == 0.0).astype(int)
    df['price_to_dlc_ratio'] = df[COL_PRICE] / (df[COL_DLC] + 1)
    return df


def derive_platform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive platform_coverage from mac and linux support flags.

    Mirrors NB02 cell 11 exactly.

    windows is excluded — 100% of games in the dataset support Windows,
    giving it zero variance and zero predictive value.

    mac and linux each correlate positively with reception (r≈0.097 and 0.094).
    Their combination (platform_coverage) is the strongest individual structured
    feature at r=+0.107.

    Preconditions
    -------------
    df[COL_MAC] and df[COL_LINUX] must be numeric or boolean (0/1 or True/False).
    They are cast to int internally.

    Columns added
    -------------
    platform_coverage   float  (mac + linux) / 2  → values: 0.0, 0.5, 1.0

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Input df with platform_coverage added.
    """
    df = df.copy()
    mac_vals   = df[COL_MAC].astype(int)   if COL_MAC   in df.columns else 0
    linux_vals = df[COL_LINUX].astype(int) if COL_LINUX in df.columns else 0
    df['platform_coverage'] = (mac_vals + linux_vals) / 2
    return df


def derive_content_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive has_achievements and has_website from their raw source columns.

    Mirrors NB02 cell 11 exactly.

    This function only adds a derived column if the raw source column exists
    AND the derived column does not already exist. This makes it safe to call
    on DataFrames from both the training pipeline (where raw columns are present)
    and the app (where derived values may have already been set directly from
    form checkboxes).

    Reception gap (from NB01 analysis):
      has_achievements: 0 achievements → 64.7% well-received;
                        >0 achievements → 74.4% well-received (10pp gap)
                        Raw achievement count: r=-0.006, p=0.430 (not significant)
                        Binary flag:           r=+0.095, p<0.001  (significant)
      has_website:      r=-0.021 (games with websites slightly less well-received —
                        larger studios face higher user expectations)

    Columns added (if source column exists and derived column is not yet present)
    -------------
    has_achievements   int  1 if COL_ACHIEVEMENTS > 0, else 0
    has_website        int  1 if COL_WEBSITE is non-empty, else 0

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Input df with has_achievements and/or has_website columns added,
        depending on what source columns are available.
    """
    df = df.copy()

    if COL_ACHIEVEMENTS in df.columns and 'has_achievements' not in df.columns:
        df['has_achievements'] = (df[COL_ACHIEVEMENTS] > 0).astype(int)

    if COL_WEBSITE in df.columns and 'has_website' not in df.columns:
        df['has_website'] = df[COL_WEBSITE].apply(
            lambda x: 0 if (pd.isna(x) or str(x).strip() == '') else 1
        )

    return df


def derive_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all three derivation functions in the correct order.

    Convenience wrapper that calls:
      1. derive_monetisation_features  (is_free, price_to_dlc_ratio)
      2. derive_platform_features      (platform_coverage)
      3. derive_content_features       (has_achievements, has_website)

    Note: genre_concentration is NOT computed here. It requires the parsed
    genre list and is handled inside preprocessor.encode_genres() because
    it needs the raw genres column to count total genres before one-hot
    encoding limits to the top 10.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Input df with up to five new derived columns added.
    """
    df = derive_monetisation_features(df)
    df = derive_platform_features(df)
    df = derive_content_features(df)
    return df
