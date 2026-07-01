"""Thin wrapper around MediaPipe Hands for detection and visualization."""

from __future__ import annotations

from typing import Any, Optional

import mediapipe as mp
import numpy as np


class HandTracker:
    """Detects a single hand in a frame and draws its landmarks.

    The tracker is intentionally limited to one hand because the classifier was
    trained on a 42-dimensional (single-hand) feature vector.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.3,
    ) -> None:
        """Initialize the MediaPipe Hands solution.

        Args:
            static_image_mode: If ``True``, treats every frame independently
                (slower). ``False`` enables tracking across frames for smoother,
                faster real-time performance. Feature values are unaffected.
            max_num_hands: Maximum number of hands to detect (kept at 1).
            min_detection_confidence: Minimum confidence for initial detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
        """
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_rgb: np.ndarray) -> Optional[Any]:
        """Run hand detection on an RGB frame.

        Args:
            frame_rgb: Frame in RGB order (MediaPipe expects RGB, not BGR).

        Returns:
            The first detected hand's landmarks, or ``None`` if no hand found.
        """
        results = self._hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        return None

    def draw_landmarks(self, frame_bgr: np.ndarray, hand_landmarks: Any) -> None:
        """Draw landmarks and connections onto a BGR frame in place.

        Args:
            frame_bgr: The frame to annotate (modified in place).
            hand_landmarks: A single MediaPipe ``NormalizedLandmarkList``.
        """
        self._mp_drawing.draw_landmarks(
            frame_bgr,
            hand_landmarks,
            self._mp_hands.HAND_CONNECTIONS,
            self._mp_drawing_styles.get_default_hand_landmarks_style(),
            self._mp_drawing_styles.get_default_hand_connections_style(),
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()
