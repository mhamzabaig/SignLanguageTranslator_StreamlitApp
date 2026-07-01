"""Feature extraction for hand-landmark based sign language classification.

The trained model expects a 42-dimensional feature vector: for each of the 21
MediaPipe hand landmarks, the (x, y) coordinates are made translation-invariant
by subtracting the minimum x and y of the detected hand. This mirrors the exact
preprocessing used when the model was originally trained, so it must not change.
"""

from __future__ import annotations

from typing import Any, List, Tuple

# A single MediaPipe hand produces 21 landmarks, each contributing an (x, y)
# pair, for a total of 42 features expected by the classifier.
NUM_LANDMARKS: int = 21
NUM_FEATURES: int = NUM_LANDMARKS * 2


def extract_features(hand_landmarks: Any) -> List[float]:
    """Convert a MediaPipe hand-landmark result into the model feature vector.

    Args:
        hand_landmarks: A single MediaPipe ``NormalizedLandmarkList`` (i.e. one
            entry from ``results.multi_hand_landmarks``).

    Returns:
        A list of 42 floats: the normalized ``x`` and ``y`` of every landmark,
        each shifted so the hand's bounding box starts at the origin.
    """
    xs: List[float] = [lm.x for lm in hand_landmarks.landmark]
    ys: List[float] = [lm.y for lm in hand_landmarks.landmark]

    min_x, min_y = min(xs), min(ys)

    features: List[float] = []
    for x, y in zip(xs, ys):
        features.append(x - min_x)
        features.append(y - min_y)

    return features


def get_bounding_box(
    hand_landmarks: Any, width: int, height: int, padding: int = 10
) -> Tuple[int, int, int, int]:
    """Compute a pixel-space bounding box around the detected hand.

    Args:
        hand_landmarks: A single MediaPipe ``NormalizedLandmarkList``.
        width: Frame width in pixels.
        height: Frame height in pixels.
        padding: Extra pixels to pad around the raw landmark extent.

    Returns:
        A ``(x1, y1, x2, y2)`` tuple in pixel coordinates.
    """
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    x1 = int(min(xs) * width) - padding
    y1 = int(min(ys) * height) - padding
    x2 = int(max(xs) * width) + padding
    y2 = int(max(ys) * height) + padding

    return x1, y1, x2, y2
