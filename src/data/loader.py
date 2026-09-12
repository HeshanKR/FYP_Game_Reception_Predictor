"""
src/data/loader.py
──────────────────
Load and filter the raw Steam dataset, create the binary reception label,
and provide fast access to the pre-processed feature matrix.

All path and threshold constants come from src.config — nothing is
hardcoded here. This module mirrors NB02 cells 5 and 7 exactly.
"""

import pandas as pd
from pathlib import Path

from src.config import (
    RAW_CSV, FEATURES_T4_CSV,
    COL_FILTER, COL_LABEL, TARGET_COL,
    MIN_REVIEW_COUNT, POSITIVE_THRESHOLD,
)


def load_raw(path: Path = None) -> pd.DataFrame:
    """
    Load the raw Steam CSV, remove sentinel rows, and apply the ≥100 review filter.

    Mirrors NB02 cell 5 exactly.

    Steps:
      1. Read the raw CSV (low_memory=False — mixed-type columns are common)
      2. Coerce COL_FILTER to numeric; remove rows where COL_FILTER <= 0
         (num_reviews_total uses -1 as a sentinel for "no review data")
      3. Keep only rows with COL_FILTER >= MIN_REVIEW_COUNT (100)

    Parameters
    ----------
    path : Path or None
        Override path to the raw CSV. Defaults to RAW_CSV from config.

    Returns
    -------
    pd.DataFrame
        Filtered raw dataset. 20,383 rows when using the canonical dataset.
    """
    csv_path = Path(path) if path is not None else RAW_CSV
    assert csv_path.exists(), (
        f'Raw dataset not found at {csv_path}. '
        'Download from Kaggle (Steam Games Dataset 2025, Ermilov) '
        'and place the CSV in data/raw/.'
    )

    df = pd.read_csv(csv_path, low_memory=False)

    # Remove -1 sentinel rows — no review data means no reliable label
    df[COL_FILTER] = pd.to_numeric(df[COL_FILTER], errors='coerce')
    df = df[df[COL_FILTER] > 0].copy()

    # Apply the ≥100 review threshold
    # A game with 3 reviews at 100% positive is not comparable to one with
    # 50,000 at 75%. The threshold ensures all labels are statistically meaningful.
    df = df[df[COL_FILTER] >= MIN_REVIEW_COUNT].copy()

    return df


def create_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the binary reception label to a filtered raw DataFrame.

    pct_pos_total is stored as a 0–100 integer (e.g. 86 = 86% positive).
    Well Received (label=1) is defined as positive_ratio >= POSITIVE_THRESHOLD (0.75).

    Mirrors NB02 cell 5 (label creation section) exactly.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered raw DataFrame from load_raw().

    Returns
    -------
    pd.DataFrame
        Input df with two new columns:
          positive_ratio  — float 0.0–1.0
          TARGET_COL      — integer 0 or 1
    """
    df = df.copy()
    df[COL_LABEL] = pd.to_numeric(df[COL_LABEL], errors='coerce')
    df['positive_ratio'] = df[COL_LABEL] / 100.0
    df[TARGET_COL] = (df['positive_ratio'] >= POSITIVE_THRESHOLD).astype(int)
    return df


def load_processed(path: Path = None) -> pd.DataFrame:
    """
    Load the pre-processed feature matrix produced by NB02.

    Runs three hard assertions to fail fast if the file is stale or corrupted:
      1. Shape must be exactly (20,383, 56)
      2. Zero nulls in all structured columns
         (short_description is excluded — 37 games legitimately have no description)
      3. Positive class rate must be 0.718 ± 0.002

    Parameters
    ----------
    path : Path or None
        Override path to the CSV. Defaults to FEATURES_T4_CSV from config.

    Returns
    -------
    pd.DataFrame
        The full 56-column feature matrix, ready for modelling.
    """
    csv_path = Path(path) if path is not None else FEATURES_T4_CSV
    assert csv_path.exists(), (
        f'Processed feature file not found at {csv_path}. '
        'Run notebook 02_data_preparation.ipynb to generate it.'
    )

    df = pd.read_csv(csv_path)

    # ── Assertion 1: correct shape ────────────────────────────────────────────
    assert df.shape == (20383, 56), (
        f'Expected shape (20383, 56), got {df.shape}. '
        'The processed CSV may have been regenerated with different parameters.'
    )

    # ── Assertion 2: zero nulls in structured columns ─────────────────────────
    # short_description is excluded — 37 games have no description (known gap)
    structured_cols = [c for c in df.columns if c != 'short_description']
    null_counts = df[structured_cols].isnull().sum()
    total_nulls = null_counts.sum()
    assert total_nulls == 0, (
        f'Found {total_nulls} null values in structured columns:\n'
        f'{null_counts[null_counts > 0]}'
    )

    # ── Assertion 3: class balance ────────────────────────────────────────────
    pos_rate = df[TARGET_COL].mean()
    assert abs(pos_rate - 0.718) < 0.002, (
        f'Positive class rate = {pos_rate:.3f}, expected 0.718 ± 0.002. '
        'Label threshold may have changed from POSITIVE_THRESHOLD = 0.75.'
    )

    return df
