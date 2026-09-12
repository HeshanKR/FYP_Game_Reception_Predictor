"""
src/features/raw_features.py
────────────────────────────
Build tier feature lists and validate feature vectors.

get_tier_feature_lists() is the single most important function in this module.
It is called by router.py to determine exactly which columns — and in which
order — are passed to each specialist model's predict_proba. The column order
must match the order used during training, or CatBoost will silently use the
wrong features for the wrong columns.

REVISION NOTE (fix applied): this module previously built the T4/T5 column
order dynamically from whatever order the input DataFrame's columns happened
to be in, on the theory that this would avoid drift from config.py. In
practice this made correctness depend entirely on src/data/preprocessor.py
producing columns in exactly the training order on every call — including at
live inference time in the Streamlit app. Verification (via
verify_tier_order.py, run directly against games_features_t4.csv and the
saved models) found that preprocessor.py's documented call order
(parse_counts() before encode_genres()) does NOT reproduce the actual
training column order — genre_concentration and the three count columns
(supported_languages_count, screenshot_count, movie_count) are swapped
relative to each other. Rather than fix preprocessor.py's call order (which
would need re-verifying every time either module changes), this module now
uses config.py's TIER4_FEATURES directly, which has itself been verified
against the actual trained models. Any DataFrame with the right 53 column
NAMES present — regardless of what order they arrive in — will now be
reordered correctly before being passed to a model.

Mirrors NB04 cell 5 in spirit (same exclusion columns), but sources the
authoritative order from config.py rather than from df.columns.
"""

import numpy as np
import pandas as pd

from src.config import (
    TIER1_FEATURES, TIER2_FEATURES, TIER4_FEATURES, TIER5_TEXT_DIMS,
    TARGET_COL, COL_DESC, COL_DLC,
    TOP_20_TAGS, KEY_CATEGORIES,
)

# Expected feature counts per tier — confirmed from NB04 and NB05 outputs.
# Used by validate_feature_vector() for fast pre-predict safety checks.
_EXPECTED_COUNTS = {1: 13, 2: 15, 3: 47, 4: 53, 5: 103}


def get_tier_feature_lists(df: pd.DataFrame) -> dict:
    """
    Build tier feature dictionaries using config.py's verified column order.

    Construction rules:
      Tier 1 : TIER1_FEATURES from config (13)
      Tier 2 : TIER2_FEATURES from config (15)
      Tier 3 : TIER2_FEATURES + tag columns + category columns (47)
      Tier 4 : TIER4_FEATURES from config (53) — verified against the actual
               trained models via verify_tier_order.py, NOT derived from
               whatever order df's columns happen to be in
      Tier 5 : Tier 4 columns (config order) + text_dim_0 … text_dim_49 (103)

    dlc_count and short_description are excluded from every tier's structured
    feature list — dlc_count has no independent predictive value (only used
    to derive price_to_dlc_ratio), and short_description is the raw text
    source for Model E's SBERT embedding, not a structured feature itself.

    IMPORTANT: this function trusts config.py's TIER4_FEATURES for the
    correct COLUMN ORDER. It does not matter what order the columns arrive
    in on `df` — callers should select columns using the returned list
    (e.g. `df[tier_lists[4]]`), which reorders them correctly regardless of
    input order. Do NOT reintroduce `[c for c in df.columns if ...]` here —
    that reintroduces exactly the bug this revision fixes.

    Parameters
    ----------
    df : pd.DataFrame
        The processed feature matrix from loader.load_processed(), or a
        single-row DataFrame built by the app's preprocessing pipeline.
        Must contain all TIER4_FEATURES columns by name — order does not
        matter.

    Returns
    -------
    dict[int, list[str]]
        Keys 1–5. Each value is a list of column name strings in the
        verified training order.

    Raises
    ------
    ValueError
        If any structured column referenced by a tier list is absent from df.
    """
    # Tag and category column names — same replace logic as NB02 cells 17 and 19
    tag_cols = [
        f'tag_{t.replace(" ", "_").replace("-", "_")}'
        for t in TOP_20_TAGS
    ]
    cat_cols = [
        f'cat_{c.replace(" ", "_").replace("-", "_").replace("/", "_")}'
        for c in KEY_CATEGORIES
    ]

    tier_lists = {
        1: TIER1_FEATURES,
        2: TIER2_FEATURES,
        3: TIER2_FEATURES + tag_cols + cat_cols,
        4: TIER4_FEATURES,
        5: TIER4_FEATURES + TIER5_TEXT_DIMS,
    }

    # Verify all structured columns exist in df (order-independent check —
    # this only confirms presence, the returned lists control actual order)
    for tier, features in tier_lists.items():
        structured = [f for f in features if not f.startswith('text_dim_')]
        missing = [f for f in structured if f not in df.columns]
        if missing:
            raise ValueError(
                f'Tier {tier} references columns not found in df: {missing}. '
                f'Ensure the DataFrame was built by the full preprocessor.py '
                f'pipeline (or loader.load_processed() for training data).'
            )

    return tier_lists


def validate_feature_vector(feature_vector: np.ndarray, tier: int) -> bool:
    """
    Assert that a feature vector has the correct length for the given tier.

    Called by router.py immediately before every predict_proba call.
    A wrong-length vector would silently produce garbage predictions from
    CatBoost rather than raising an informative error.

    Note: this checks LENGTH only, not order. Order correctness depends on
    having selected columns via get_tier_feature_lists()'s returned lists
    (e.g. `df[tier_lists[4]].values`), not by taking df's raw column order.

    Expected counts (confirmed from NB04 and NB05 test outputs):
      Tier 1 — Model A :  13 features
      Tier 2 — Model B :  15 features
      Tier 3 — Model C :  47 features
      Tier 4 — Model D :  53 features
      Tier 5 — Model E : 103 features

    Parameters
    ----------
    feature_vector : np.ndarray
        1-D array to validate.
    tier : int
        Tier the vector should correspond to (1–5).

    Returns
    -------
    bool
        Always True if no exception is raised.

    Raises
    ------
    ValueError
        If tier is unknown or vector length does not match.
    """
    if tier not in _EXPECTED_COUNTS:
        raise ValueError(
            f'Unknown tier {tier}. Must be one of '
            f'{sorted(_EXPECTED_COUNTS.keys())}.'
        )

    expected = _EXPECTED_COUNTS[tier]
    actual   = len(feature_vector)

    if actual != expected:
        model_letter = 'ABCDE'[tier - 1]
        raise ValueError(
            f'Tier {tier} (Model {model_letter}) expects {expected} features '
            f'but received {actual}. '
            f'Verify that get_tier_feature_lists() was used to select columns, '
            f'e.g. df[tier_lists[{tier}]].values — not df.values directly.'
        )

    return True
