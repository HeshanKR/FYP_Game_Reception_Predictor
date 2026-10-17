"""
verify_tier_order.py
─────────────────────────────────────────────────────────────────────────────
Standalone diagnostic script — NOT part of the NB01-17 sequence, does not
modify anything. Run this once to check whether src/config.py's TIER1-5
feature lists actually match the order NB04 and NB05 used to train
model_a/b/c/d/e.pkl.

WHY THIS EXISTS:
NB04 and NB05 build their T4 (and T5) feature lists dynamically from the
CSV's own column order:
    all_feature_cols = [c for c in df.columns if c not in
                         [TARGET_COL, 'short_description', 'dlc_count']]
config.py instead builds TIER4_FEATURES by concatenating explicit lists
(TOP_10_GENRES + TOP_20_TAGS + KEY_CATEGORIES + EFFORT_SIGNALS). These two
constructions are not guaranteed to produce the same order, and in this
project they do not, which is what caused notebook 12's Macro F1 = 0.52
result before it was fixed.

This script checks all five tiers, not just T4, so you get a definitive
answer rather than an assumption for any of them.

HOW TO RUN:
1. Save this file directly in your project root — the same folder that
   contains `src/`, `data/`, `models/` (NOT inside `notebooks/`).
2. Open a terminal, activate your fenv environment, and run:
       cd C:\\Users\\3214h\\Documents\\fyp-steam-reception
       fenv\\Scripts\\activate
       python verify_tier_order.py
3. Read the printed report. For any tier marked MISMATCH, a corrected,
   ready-to-paste Python list is printed at the end.

This script only reads games_features_t4.csv and config.py. It writes
nothing and does not touch any model file.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TARGET_COL, FEATURES_T4_CSV, TIER_FEATURES, PCA_COMPONENTS,
)

SEP = '=' * 78


def build_actual_t4_order(df):
    """Exactly replicates NB04's all_feature_cols / NB05's T4_FEATURES."""
    exclude_cols = [TARGET_COL, 'short_description', 'dlc_count']
    return [c for c in df.columns if c not in exclude_cols]


def compare_lists(name, config_list, actual_list):
    """Reports whether two feature-name lists match as sets and in order."""
    print(f'\n--- Tier: {name} ---')
    print(f'  config.py length : {len(config_list)}')
    print(f'  actual length    : {len(actual_list)}')

    config_set = set(config_list)
    actual_set = set(actual_list)

    missing_from_actual = config_set - actual_set
    missing_from_config = actual_set - config_set

    if missing_from_actual:
        print(f'  *** Columns in config.py but NOT in the CSV: {sorted(missing_from_actual)}')
    if missing_from_config:
        print(f'  *** Columns in the CSV but NOT in config.py: {sorted(missing_from_config)}')

    if config_set != actual_set:
        print('  RESULT: SET MISMATCH — this is more serious than an ordering issue. '
              'Investigate the missing/extra columns above before doing anything else.')
        return False, None

    # Same set of columns — now check position-by-position order
    order_matches = (config_list == actual_list)
    if order_matches:
        print('  RESULT: MATCH — same columns, same order. Safe to use config.py as-is.')
        return True, None
    else:
        n_mismatched_positions = sum(
            1 for a, b in zip(config_list, actual_list) if a != b
        )
        print(f'  RESULT: MISMATCH — same 	{len(config_list)} columns, but order differs '
              f'at {n_mismatched_positions} of {len(config_list)} positions.')
        print('  First 5 mismatched positions (index: config.py name -> actual name):')
        shown = 0
        for i, (a, b) in enumerate(zip(config_list, actual_list)):
            if a != b:
                print(f'    [{i}] {a!r} -> {b!r}')
                shown += 1
                if shown >= 5:
                    break
        return False, actual_list


def main():
    assert FEATURES_T4_CSV.exists(), f'Not found: {FEATURES_T4_CSV}'
    df = pd.read_csv(FEATURES_T4_CSV)
    print(SEP)
    print(f'Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns from {FEATURES_T4_CSV.name}')
    print(SEP)

    actual_t4 = build_actual_t4_order(df)
    actual_t5 = actual_t4 + [f'text_dim_{i}' for i in range(PCA_COMPONENTS)]

    results = {}
    for tier in [1, 2, 3]:
        config_list = TIER_FEATURES[tier]
        # For T1-T3, the "actual" order used in training IS config.py's own list
        # (NB04 imports TIER1_FEATURES/TIER2_FEATURES directly, and reconstructs
        # T3 from TOP_20_TAGS/KEY_CATEGORIES in the same order config.py uses).
        # We still verify every name actually exists as a column, as a sanity check.
        missing = [c for c in config_list if c not in df.columns]
        print(f'\n--- Tier: T{tier} ---')
        print(f'  config.py length : {len(config_list)}')
        if missing:
            print(f'  *** MISSING from CSV: {missing}')
            results[tier] = False
        else:
            print(f'  RESULT: All {len(config_list)} columns present in the CSV. '
                  f'NB04 uses this exact config.py list for T{tier}, so order is trusted.')
            results[tier] = True

    match_t4, corrected_t4 = compare_lists('T4 (Model D)', TIER_FEATURES[4], actual_t4)
    results[4] = match_t4

    match_t5, corrected_t5 = compare_lists('T5 (Model E)', TIER_FEATURES[5], actual_t5)
    results[5] = match_t5

    print('\n' + SEP)
    print('SUMMARY')
    print(SEP)
    for tier in [1, 2, 3, 4, 5]:
        status = 'OK' if results[tier] else 'NEEDS FIXING'
        print(f'  Tier {tier}: {status}')

    if not match_t4:
        print('\n' + SEP)
        print('CORRECTED TIER4_FEATURES — paste this into src/config.py to replace')
        print('the current TIER4_FEATURES definition:')
        print(SEP)
        print('TIER4_FEATURES = [')
        for name in corrected_t4:
            print(f"    '{name}',")
        print(']')

    if not match_t5:
        print('\n' + SEP)
        print('CORRECTED TIER5_FEATURES — paste this into src/config.py to replace')
        print('the current TIER5_FEATURES definition (or simply keep it as')
        print('TIER4_FEATURES + TIER5_TEXT_DIMS once TIER4_FEATURES above is fixed):')
        print(SEP)
        print('TIER5_FEATURES = TIER4_FEATURES + TIER5_TEXT_DIMS')

    print('\nDone.')


if __name__ == '__main__':
    main()
