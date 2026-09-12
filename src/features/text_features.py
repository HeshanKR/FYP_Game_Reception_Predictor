"""
src/features/text_features.py
──────────────────────────────
SBERT encoding and PCA dimensionality reduction for the short_description field.

This module has two distinct usage contexts:

  Training (Colab / NB05 reference):
    encode_descriptions_batch() — batch-encode 20,383 descriptions on GPU
    fit_pca()                   — fit PCA on training embeddings only (no leakage)
    apply_pca()                 — reduce 384-dim embeddings to 50-dim

  Inference (Streamlit app):
    encode_single_description() — encode one description at prediction time
                                  (uses pre-fitted SBERT model and PCA loaded at startup)

All functions mirror NB05 cells 10–14 exactly. The trained model (model_e.pkl)
expects T5 feature vectors built by:
  np.hstack([T4_structured_features, apply_pca(sbert_embeddings)])

Key findings (NB05 and NB11):
  - SBERT PCA-50 embeddings have mean R² ≈ 0.06 against structured features
    (mean gate value 0.9394 in gated fusion experiment) — descriptions are
    genuinely independent of structured metadata, not just repeating tags
  - However, independent does not mean predictive: descriptions do not improve
    classification over structured features alone
  - 37 games have no description (0.2%) — these fall back to Model D always

SBERT model : all-MiniLM-L6-v2  (22M parameters, 384-dim output)
PCA         : 50 components, 51.1% variance retained
              Chosen empirically (CV macro F1 decreases with more components)
"""

import numpy as np
import pandas as pd

from src.config import (
    SBERT_MODEL, PCA_COMPONENTS,
    SBERT_RAW_NPY, SBERT_PCA_NPY, PCA_PATH,
)


# ── HTML stripping ─────────────────────────────────────────────────────────────

def strip_html(text) -> str:
    """
    Remove HTML tags from a Steam description and return clean plain text.

    Steam descriptions frequently contain HTML markup (<br>, <b>, <i>, <ul>, etc.).
    BeautifulSoup's html.parser handles all common Steam formatting.

    Returns '' for null, NaN, or empty inputs. This return value signals to
    encode_single_description() that no description is available, triggering
    the Model D fallback in the router.

    Mirrors NB05 cell 10 exactly.

    Parameters
    ----------
    text : str or NaN
        Raw description value from the short_description column.

    Returns
    -------
    str
        Clean plain text, or '' if input is null/empty.
    """
    from bs4 import BeautifulSoup

    if pd.isna(text) or str(text).strip() == '':
        return ''
    return BeautifulSoup(str(text), 'html.parser').get_text().strip()


# ── Batch encoding (training reference — Colab/NB05 use only) ─────────────────

def encode_descriptions_batch(
    descriptions: list,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Batch-encode a list of descriptions using SBERT.

    This function is a reference implementation of what NB05 ran on Colab's T4 GPU.
    It is NOT called by the app at inference time — the app uses
    encode_single_description() instead.

    In NB05, games with no description were given the placeholder
    'No description available', then zeroed out afterward. This function
    applies the same approach: empty strings receive zero vectors.

    Mirrors NB05 cell 12 exactly.

    Parameters
    ----------
    descriptions : list of str
        Pre-cleaned descriptions (HTML already stripped by strip_html).
        Empty strings are handled as missing descriptions.
    device : str
        'cuda' for GPU (Colab T4), 'cpu' for local. Default 'cpu'.

    Returns
    -------
    np.ndarray
        Shape (n, 384). Games with empty descriptions have all-zero rows.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(SBERT_MODEL, device=device)

    # Replace empty descriptions with a neutral placeholder
    # Zero these rows out afterward — neutral embedding, not meaningful zero
    desc_for_encoding = [
        'No description available' if d == '' else d
        for d in descriptions
    ]

    embeddings = model.encode(
        desc_for_encoding,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    # Zero out embeddings for games with no description
    empty_mask = np.array([d == '' for d in descriptions])
    embeddings[empty_mask] = 0.0

    return embeddings


# ── PCA fit and transform ──────────────────────────────────────────────────────

def fit_pca(
    embeddings_train: np.ndarray,
    n_components: int = PCA_COMPONENTS,
):
    """
    Fit PCA on training-split embeddings only.

    PCA must be fitted on the training split only — fitting on the full dataset
    before splitting would leak test-set distribution information. In NB05,
    this was enforced explicitly.

    The selected component count (50) was chosen empirically:
    CV Macro F1 decreases monotonically with more components. More SBERT
    dimensions add noise rather than signal to the CatBoost classifier.

    This function is a training-time reference. The fitted PCA is saved as
    pca_transformer.pkl and loaded at inference time via load_pca().

    Parameters
    ----------
    embeddings_train : np.ndarray
        Shape (n_train, 384). Training-split SBERT embeddings only.
    n_components : int
        Number of PCA components. Default PCA_COMPONENTS = 50.

    Returns
    -------
    sklearn.decomposition.PCA
        Fitted PCA transformer. Save to PCA_PATH with joblib.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(embeddings_train)
    variance = pca.explained_variance_ratio_.sum() * 100
    print(f'PCA({n_components}) fitted — {variance:.1f}% variance retained')
    return pca


def apply_pca(embeddings: np.ndarray, pca) -> np.ndarray:
    """
    Apply a fitted PCA transformer to a batch of embeddings.

    Used at both training time (to build the T5 feature matrix) and inference
    time (to transform a single description's embedding).

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n, 384) — raw SBERT embeddings.
    pca : sklearn.decomposition.PCA
        Fitted PCA transformer loaded from pca_transformer.pkl.

    Returns
    -------
    np.ndarray
        Shape (n, PCA_COMPONENTS) — dimensionality-reduced embeddings.
    """
    return pca.transform(embeddings)


# ── Single-description inference (Streamlit app use) ──────────────────────────

def encode_single_description(
    text: str,
    sbert_model,
    pca,
) -> np.ndarray:
    """
    Encode one game description for inference-time prediction.

    Called by router.py when T4 features are available AND the uncertainty
    router has determined that Model D is uncertain (confidence < 0.63).

    Returns a zero vector for null/empty descriptions — this signals to
    router.py that no description is available and Model D should be used
    directly without invoking Model E.

    Steps:
      1. Strip HTML from raw description text
      2. If empty after stripping: return np.zeros(PCA_COMPONENTS)
      3. SBERT encode: text → (1, 384) embedding
      4. PCA transform: (1, 384) → (1, 50)
      5. Return as 1-D array: shape (50,)

    The SBERT model and PCA are passed in as arguments because they are
    loaded once at app startup via @st.cache_resource and should not be
    reloaded per prediction.

    Parameters
    ----------
    text : str
        Raw description text from the app form (may contain HTML).
    sbert_model : SentenceTransformer
        Pre-loaded SBERT model (all-MiniLM-L6-v2).
    pca : sklearn.decomposition.PCA
        Pre-fitted PCA transformer from pca_transformer.pkl.

    Returns
    -------
    np.ndarray
        Shape (PCA_COMPONENTS,) = (50,).
        All zeros if description is empty — triggers Model D fallback.
    """
    # Step 1: strip HTML
    clean_text = strip_html(text)

    # Step 2: return zeros for missing descriptions
    if not clean_text:
        return np.zeros(PCA_COMPONENTS, dtype=np.float32)

    # Step 3: SBERT encode — note the list wrapping required by encode()
    embedding = sbert_model.encode(
        [clean_text],
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    # embedding shape: (1, 384)

    # Step 4: PCA transform
    reduced = pca.transform(embedding)
    # reduced shape: (1, 50)

    # Step 5: return as 1-D array
    return reduced[0].astype(np.float32)


# ── Model loading helpers ──────────────────────────────────────────────────────

def load_sbert_model(device: str = 'cpu'):
    """
    Load the SBERT model from disk (or cache).

    Called once at app startup inside @st.cache_resource.
    Loading takes ~2–3 seconds on first call; subsequent calls use the cache.

    Parameters
    ----------
    device : str
        'cuda' if GPU is available, 'cpu' otherwise.

    Returns
    -------
    SentenceTransformer
        Loaded model ready for encoding.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(SBERT_MODEL, device=device)


def load_pca(path=None):
    """
    Load the fitted PCA transformer from disk.

    Called once at app startup inside @st.cache_resource.

    Parameters
    ----------
    path : Path or None
        Override path. Defaults to PCA_PATH from config.

    Returns
    -------
    sklearn.decomposition.PCA
        Fitted PCA transformer ready for .transform() calls.
    """
    import joblib
    pca_path = path if path is not None else PCA_PATH
    assert pca_path.exists(), (
        f'PCA transformer not found at {pca_path}. '
        'Download pca_transformer.pkl from Colab outputs.'
    )
    return joblib.load(pca_path)
