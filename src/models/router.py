# src/models/router.py
# ─────────────────────────────────────────────────────────────────────────────
# PPRS Confidence Router
# Selects which specialist model to use given available feature tier,
# and implements uncertainty-based routing with Model E as tiebreaker.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import joblib
from pathlib import Path

# ── Default model paths (can be overridden at import time) ───────────────────
_ROOT       = Path(__file__).resolve().parent.parent.parent
_MODELS_DIR = _ROOT / 'models'

MODEL_PATHS = {
    'A': _MODELS_DIR / 'model_a.pkl',
    'B': _MODELS_DIR / 'model_b.pkl',
    'C': _MODELS_DIR / 'model_c.pkl',
    'D': _MODELS_DIR / 'model_d.pkl',
    'E': _MODELS_DIR / 'model_e.pkl',
}
PCA_PATH = _MODELS_DIR / 'pca_transformer.pkl'

TIER_TO_MODEL = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}


def load_models(model_paths=None):
    """
    Load all specialist models from disk.

    Parameters
    ----------
    model_paths : dict, optional
        Override default paths. Keys: 'A', 'B', 'C', 'D', 'E'.

    Returns
    -------
    dict  {letter: fitted_model}
    """
    paths = model_paths or MODEL_PATHS
    models = {}
    for letter, path in paths.items():
        if Path(path).exists():
            models[letter] = joblib.load(path)
        else:
            raise FileNotFoundError(
                f"Model {letter} not found at {path}. "
                "Run notebooks 04 and 05 first."
            )
    return models


def load_pca(pca_path=None):
    """Load the fitted PCA transformer for SBERT dimensionality reduction."""
    path = pca_path or PCA_PATH
    if not Path(path).exists():
        raise FileNotFoundError(
            f"PCA transformer not found at {path}. Run notebook 05 first."
        )
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Routing Strategy 1 — Direct Routing
# ─────────────────────────────────────────────────────────────────────────────

def route_direct(features_by_tier, available_tier, models):
    """
    Direct routing: always use the specialist model for the available tier.
    This is the primary routing strategy — confirmed best in ablation study.

    Parameters
    ----------
    features_by_tier : dict {tier_int: 1-D numpy array}
        Feature arrays for each tier up to available_tier.
    available_tier   : int (1, 2, 3, or 4)
        Highest tier of features the developer has provided.
    models           : dict {letter: fitted_model}
        Loaded specialist models.

    Returns
    -------
    prediction  : int   (0 = Not Well Received, 1 = Well Received)
    confidence  : float (max predict_proba)
    model_used  : str   ('Model A', 'Model B', etc.)
    probabilities : 1-D array [p_not_well_received, p_well_received]
    """
    letter  = TIER_TO_MODEL[available_tier]
    model   = models[letter]
    feats   = features_by_tier[available_tier].reshape(1, -1)
    proba   = model.predict_proba(feats)[0]
    pred    = int(np.argmax(proba))
    conf    = float(proba[pred])
    return pred, conf, f'Model {letter}', proba


# ─────────────────────────────────────────────────────────────────────────────
# Routing Strategy 2 — Confidence Routing
# ─────────────────────────────────────────────────────────────────────────────

def route_confidence(features_by_tier, available_tier, models):
    """
    Confidence routing: query all eligible models (T1 to available_tier)
    and select the one with the highest max predict_proba.

    Note: ablation study showed this underperforms direct routing at T3 and T4
    because lower-tier models can express high confidence on wrong predictions.
    Kept for research completeness.

    Parameters
    ----------
    features_by_tier : dict {tier_int: 1-D numpy array}
    available_tier   : int
    models           : dict {letter: fitted_model}

    Returns
    -------
    prediction, confidence, model_used, probabilities
    """
    best_pred  = None
    best_conf  = -1.0
    best_tier  = None
    best_proba = None

    for tier in range(1, available_tier + 1):
        letter = TIER_TO_MODEL[tier]
        model  = models[letter]
        feats  = features_by_tier[tier].reshape(1, -1)
        proba  = model.predict_proba(feats)[0]
        pred   = int(np.argmax(proba))
        conf   = float(proba[pred])

        if conf > best_conf:
            best_conf  = conf
            best_pred  = pred
            best_tier  = tier
            best_proba = proba

    letter = TIER_TO_MODEL[best_tier]
    return best_pred, best_conf, f'Model {letter}', best_proba


# ─────────────────────────────────────────────────────────────────────────────
# Routing Strategy 3 — Uncertainty-Based Routing (Model E as Tiebreaker)
# ─────────────────────────────────────────────────────────────────────────────

def route_uncertainty(
    t4_features,
    description_text,
    models,
    pca,
    sbert_model,
    threshold=0.55
):
    """
    Uncertainty-based routing: use Model D directly when confident,
    invoke Model E (SBERT) to break ties when Model D is uncertain.

    When Model D's max predict_proba is below the threshold, the description
    text is encoded with SBERT, PCA-reduced, and combined with T4 features
    to form a T5 input for Model E. The two probability vectors are averaged.

    Parameters
    ----------
    t4_features      : 1-D numpy array of T4 structured features (53 dims)
    description_text : str  developer-written game description
    models           : dict {letter: fitted_model}
    pca              : fitted sklearn PCA transformer
    sbert_model      : loaded SentenceTransformer model
    threshold        : float confidence cutoff for invoking Model E (default 0.55)

    Returns
    -------
    prediction    : int
    confidence    : float
    model_used    : str  ('Model D (confident)' or 'Model E (uncertainty resolver)')
    probabilities : 1-D array [p_not_well_received, p_well_received]
    """
    model_d = models['D']
    model_e = models['E']

    # Step 1: Query Model D
    prob_d      = model_d.predict_proba(t4_features.reshape(1, -1))[0]
    confidence  = float(np.max(prob_d))

    if confidence >= threshold or not description_text or description_text.strip() == '':
        # Model D is confident, or no description available — use Model D directly
        pred   = int(np.argmax(prob_d))
        source = 'Model D (confident)' if confidence >= threshold \
                 else 'Model D (no description)'
        return pred, confidence, source, prob_d

    # Step 2: Model D is uncertain — encode description and invoke Model E
    embedding   = sbert_model.encode([description_text], convert_to_numpy=True)
    emb_pca     = pca.transform(embedding)           # (1, PCA_COMPONENTS)
    t5_features = np.hstack([t4_features, emb_pca[0]])  # (53 + PCA_COMPONENTS,)

    prob_e   = model_e.predict_proba(t5_features.reshape(1, -1))[0]

    # Average the two probability vectors
    combined = (prob_d + prob_e) / 2
    pred     = int(np.argmax(combined))
    conf_out = float(np.max(combined))

    return pred, conf_out, 'Model E (uncertainty resolver)', combined


# ─────────────────────────────────────────────────────────────────────────────
# Helper — batch evaluation for evaluation notebooks
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_routing_strategy(
    strategy_fn,
    X_test_by_tier,
    y_test,
    models,
    available_tier=4,
    **kwargs
):
    """
    Evaluate any routing strategy function on the full test set.

    Parameters
    ----------
    strategy_fn      : callable  (route_direct, route_confidence, etc.)
    X_test_by_tier   : dict {tier: 2-D numpy array of test features}
    y_test           : 1-D array of true labels
    models           : dict {letter: fitted_model}
    available_tier   : int  which tier to treat as available
    **kwargs         : additional arguments passed to strategy_fn

    Returns
    -------
    dict with keys: predictions, confidences, models_used, macro_f1, minority_f1
    """
    from sklearn.metrics import f1_score

    predictions  = []
    confidences  = []
    models_used  = []

    n_games = len(y_test)
    for i in range(n_games):
        # Build per-game feature dict
        game_features = {tier: X_test_by_tier[tier][i]
                         for tier in range(1, available_tier + 1)
                         if tier in X_test_by_tier}

        pred, conf, used, _ = strategy_fn(
            game_features, available_tier, models, **kwargs
        )
        predictions.append(pred)
        confidences.append(conf)
        models_used.append(used)

    predictions = np.array(predictions)
    macro_f1    = f1_score(y_test, predictions, average='macro')
    minority_f1 = f1_score(y_test, predictions, pos_label=0)

    return {
        'predictions':  predictions,
        'confidences':  confidences,
        'models_used':  models_used,
        'macro_f1':     macro_f1,
        'minority_f1':  minority_f1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main routing entry point — used by the Streamlit app
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    structured_features_by_tier,
    description_text=None,
    available_tier=4,
    models=None,
    pca=None,
    sbert_model=None,
    uncertainty_threshold=0.55,
    strategy='uncertainty'
):
    """
    Main prediction entry point for the Streamlit app.

    Parameters
    ----------
    structured_features_by_tier : dict {tier_int: 1-D numpy array}
    description_text             : str or None
    available_tier               : int (1–4)
    models                       : dict {letter: fitted_model}  (load if None)
    pca                          : fitted PCA transformer        (load if None)
    sbert_model                  : SentenceTransformer           (pass if available)
    uncertainty_threshold        : float (default 0.55)
    strategy                     : 'direct' | 'confidence' | 'uncertainty'

    Returns
    -------
    dict with keys:
      'prediction'    : int  (0 or 1)
      'label'         : str  ('Well Received' or 'Not Well Received')
      'confidence'    : float
      'model_used'    : str
      'probabilities' : dict {'Not Well Received': float, 'Well Received': float}
    """
    if models is None:
        models = load_models()

    if strategy == 'direct':
        pred, conf, used, proba = route_direct(
            structured_features_by_tier, available_tier, models
        )

    elif strategy == 'confidence':
        pred, conf, used, proba = route_confidence(
            structured_features_by_tier, available_tier, models
        )

    elif strategy == 'uncertainty':
        if sbert_model is None or pca is None or description_text is None:
            # Fall back to direct routing if Model E components are unavailable
            pred, conf, used, proba = route_direct(
                structured_features_by_tier, available_tier, models
            )
        else:
            t4_feats = structured_features_by_tier[min(available_tier, 4)]
            pred, conf, used, proba = route_uncertainty(
                t4_feats, description_text,
                models, pca, sbert_model,
                threshold=uncertainty_threshold
            )
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. "
                         "Choose 'direct', 'confidence', or 'uncertainty'.")

    label = 'Well Received' if pred == 1 else 'Not Well Received'

    return {
        'prediction':    pred,
        'label':         label,
        'confidence':    round(conf, 4),
        'model_used':    used,
        'probabilities': {
            'Not Well Received': round(float(proba[0]), 4),
            'Well Received':     round(float(proba[1]), 4),
        }
    }
