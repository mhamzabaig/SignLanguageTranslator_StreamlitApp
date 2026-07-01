"""Sign language classifier: loads the trained model and maps features to letters."""

from __future__ import annotations

import os
import pickle
from typing import List, Optional, Tuple

import numpy as np

from utils.feature_extraction import NUM_FEATURES

# Resolve model paths relative to this file so the app works from any CWD.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(_BASE_DIR, "model", "classifier.pkl")
_FALLBACK_MODEL_PATH = os.path.join(_BASE_DIR, "model.p")


class SignLanguageClassifier:
    """Wraps the trained scikit-learn model that maps hand features to A-Z."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Load the pickled classifier.

        Args:
            model_path: Path to the pickled model dict (expects a ``"model"``
                key). Falls back to the legacy ``model.p`` location if the
                default path is missing.

        Raises:
            FileNotFoundError: If no model file can be located.
            KeyError: If the pickle does not contain a ``"model"`` entry.
        """
        path = model_path or DEFAULT_MODEL_PATH
        if not os.path.exists(path) and os.path.exists(_FALLBACK_MODEL_PATH):
            path = _FALLBACK_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at '{path}'.")

        with open(path, "rb") as model_file:
            model_dict = pickle.load(model_file)

        self._model = model_dict["model"]
        # Labels: model classes are strings '0'-'25' mapping to A-Z.
        self._labels = {i: chr(i + 65) for i in range(26)}

    def predict(self, features: List[float]) -> Optional[Tuple[str, float]]:
        """Predict the alphabet letter for a feature vector.

        Args:
            features: A 42-length feature vector from ``extract_features``.

        Returns:
            A ``(letter, confidence)`` tuple, or ``None`` if the feature vector
            has an unexpected length (e.g. malformed detection).
        """
        if len(features) != NUM_FEATURES:
            return None

        sample = np.asarray(features, dtype=np.float32).reshape(1, -1)
        raw_prediction = self._model.predict(sample)[0]
        letter = self._labels[int(raw_prediction)]

        confidence = 1.0
        if hasattr(self._model, "predict_proba"):
            confidence = float(np.max(self._model.predict_proba(sample)))

        return letter, confidence
