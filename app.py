"""Sign Language Translator — a Streamlit real-time ASL alphabet recognizer.

Runs the user's webcam, detects a single hand with MediaPipe, classifies the
alphabet gesture with the pre-trained model, and builds a sentence from the
stabilized predictions. Spaces are inserted automatically after a short pause.

The main loop processes one frame per Streamlit rerun and calls ``st.rerun()``
to continue. This keeps the Start / Stop / Clear buttons responsive, which a
blocking ``while`` loop would not.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st

from inference import SignLanguageClassifier
from utils.feature_extraction import extract_features, get_bounding_box
from utils.hand_tracking import HandTracker

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PAGE_TITLE = "Sign Language Translator"
PAGE_DESCRIPTION = (
    "Translate American Sign Language alphabet gestures into English text in "
    "real time using your webcam."
)
DEFAULT_STABILITY_FRAMES = 12
DEFAULT_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_SPACE_PAUSE_SECONDS = 2.0
DEFAULT_CAMERA_INDEX = 0


# --------------------------------------------------------------------------- #
# Resource loading (cached so the model / tracker load only once)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model and hand tracker...")
def load_resources() -> Tuple[SignLanguageClassifier, HandTracker]:
    """Load the classifier and hand tracker a single time per session."""
    classifier = SignLanguageClassifier()
    tracker = HandTracker()
    return classifier, tracker


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_session_state() -> None:
    """Initialize all persistent state keys with sane defaults."""
    defaults = {
        "running": False,
        "read_failures": 0,
        "sentence": "",
        "candidate_char": None,
        "candidate_count": 0,
        "last_committed": None,
        "last_hand_time": 0.0,
        "space_inserted": True,  # nothing typed yet, so no space owed
        "current_prediction": "-",
        "current_confidence": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_translation_state() -> None:
    """Clear the built-up sentence and per-prediction bookkeeping."""
    st.session_state.sentence = ""
    st.session_state.candidate_char = None
    st.session_state.candidate_count = 0
    st.session_state.last_committed = None
    st.session_state.space_inserted = True
    st.session_state.current_prediction = "-"
    st.session_state.current_confidence = 0.0


# --------------------------------------------------------------------------- #
# Camera helpers
# --------------------------------------------------------------------------- #
# The webcam is held as a cached singleton (like a DB connection) rather than in
# st.session_state. Storing a cv2.VideoCapture in session_state is unreliable
# across st.rerun() and leads to the camera being reopened every frame. With
# cache_resource the SAME capture object is returned on every rerun, so it is
# opened exactly once and only released when the user stops.
@st.cache_resource(show_spinner=False)
def get_camera(index: int) -> Optional[cv2.VideoCapture]:
    """Open the webcam once and reuse it across reruns.

    Uses the DirectShow backend on Windows for faster, quieter startup and a
    single-frame buffer to keep latency low.

    Args:
        index: Camera device index.

    Returns:
        An opened ``VideoCapture``, or ``None`` if the device cannot be opened.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def release_camera(index: int) -> None:
    """Release the cached webcam and clear the cache so it can reopen later.

    Args:
        index: Camera device index used to acquire the capture.
    """
    cap = get_camera(index)
    if cap is not None:
        cap.release()
    get_camera.clear()


# --------------------------------------------------------------------------- #
# Frame processing
# --------------------------------------------------------------------------- #
def process_frame(
    frame_bgr: np.ndarray,
    tracker: HandTracker,
    classifier: SignLanguageClassifier,
) -> Tuple[np.ndarray, Optional[str], float, bool]:
    """Detect a hand, classify it, and annotate the frame.

    Args:
        frame_bgr: Raw BGR frame from the webcam.
        tracker: The shared hand tracker.
        classifier: The shared sign language classifier.

    Returns:
        A tuple ``(annotated_rgb, letter, confidence, hand_present)`` where
        ``annotated_rgb`` is RGB (for ``st.image``), ``letter`` is the predicted
        character or ``None``, ``confidence`` is in ``[0, 1]``, and
        ``hand_present`` indicates whether a hand was detected.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    hand_landmarks = tracker.detect(frame_rgb)

    if hand_landmarks is None:
        return frame_rgb, None, 0.0, False

    tracker.draw_landmarks(frame_bgr, hand_landmarks)
    features = extract_features(hand_landmarks)
    prediction = classifier.predict(features)

    height, width, _ = frame_bgr.shape
    x1, y1, x2, y2 = get_bounding_box(hand_landmarks, width, height)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 0), 4)

    letter: Optional[str] = None
    confidence = 0.0
    if prediction is not None:
        letter, confidence = prediction
        cv2.putText(
            frame_bgr,
            letter,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

    annotated_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, letter, confidence, True


def update_translation(
    letter: Optional[str],
    confidence: float,
    hand_present: bool,
    stability_frames: int,
    confidence_threshold: float,
    space_pause_seconds: float,
) -> None:
    """Update the sentence from a new prediction using stabilization + spacing.

    A letter is committed only after it stays stable for ``stability_frames``
    consecutive frames, and never twice in a row (so "AAAA" becomes "A"). When
    no confident hand is seen for ``space_pause_seconds``, a single space is
    appended, separating words.

    Args:
        letter: The predicted letter this frame, or ``None``.
        confidence: Prediction confidence in ``[0, 1]``.
        hand_present: Whether a hand was detected this frame.
        stability_frames: Frames a letter must persist before being committed.
        confidence_threshold: Minimum confidence to trust a prediction.
        space_pause_seconds: Idle seconds before inserting a space.
    """
    now = time.time()
    confident = (
        hand_present and letter is not None and confidence >= confidence_threshold
    )

    if confident:
        st.session_state.last_hand_time = now
        st.session_state.space_inserted = False
        st.session_state.current_prediction = letter
        st.session_state.current_confidence = confidence

        if letter == st.session_state.candidate_char:
            st.session_state.candidate_count += 1
        else:
            st.session_state.candidate_char = letter
            st.session_state.candidate_count = 1

        stable = st.session_state.candidate_count == stability_frames
        if stable and letter != st.session_state.last_committed:
            st.session_state.sentence += letter
            st.session_state.last_committed = letter
        return

    # No confident hand this frame: reset the stabilization window so the same
    # letter can be typed again after a pause, and consider inserting a space.
    st.session_state.candidate_char = None
    st.session_state.candidate_count = 0
    st.session_state.last_committed = None
    st.session_state.current_prediction = "-"
    st.session_state.current_confidence = 0.0

    elapsed = now - st.session_state.last_hand_time
    should_space = (
        elapsed >= space_pause_seconds
        and not st.session_state.space_inserted
        and st.session_state.sentence
        and not st.session_state.sentence.endswith(" ")
    )
    if should_space:
        st.session_state.sentence += " "
        st.session_state.space_inserted = True


# --------------------------------------------------------------------------- #
# UI sections
# --------------------------------------------------------------------------- #
def render_header() -> None:
    """Render the page title and short description."""
    st.title(PAGE_TITLE)
    st.markdown(f"##### {PAGE_DESCRIPTION}")


def render_instructions() -> None:
    """Render the highlighted instructions panel."""
    st.info(
        "### Instructions\n"
        "1. Press **Start Translation** to begin real-time inference.\n"
        "2. Allow webcam permission when prompted.\n"
        "3. Show one alphabet clearly using your hand.\n"
        "4. Hold the gesture steady until the prediction stabilizes.\n"
        "5. To insert a **SPACE** between words, keep your hand still (or remove "
        "it from view) for 2 seconds.\n"
        "6. Press **Stop Translation** to end webcam inference."
    )


def render_sidebar() -> Tuple[int, float, float, int]:
    """Render tuning controls and return the current settings.

    Returns:
        ``(stability_frames, confidence_threshold, space_pause_seconds,
        camera_index)``.
    """
    st.sidebar.header("Settings")
    stability_frames = st.sidebar.slider(
        "Stability (frames)",
        min_value=3,
        max_value=30,
        value=DEFAULT_STABILITY_FRAMES,
        help="Frames a letter must stay stable before it is added.",
    )
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Minimum model confidence to accept a prediction.",
    )
    space_pause_seconds = st.sidebar.slider(
        "Space pause (seconds)",
        min_value=1.0,
        max_value=5.0,
        value=DEFAULT_SPACE_PAUSE_SECONDS,
        step=0.5,
        help="Idle time before a space is inserted between words.",
    )
    camera_index = st.sidebar.number_input(
        "Camera index",
        min_value=0,
        max_value=10,
        value=DEFAULT_CAMERA_INDEX,
        step=1,
        help="Change if you have multiple cameras.",
    )
    return stability_frames, confidence_threshold, space_pause_seconds, int(camera_index)


def render_controls(camera_index: int) -> None:
    """Render the Start / Stop / Clear control buttons and handle their clicks.

    Args:
        camera_index: Active camera index, needed to release it on Stop.
    """
    col_start, col_stop, col_clear = st.columns(3)

    start_clicked = col_start.button(
        "▶ Start Translation",
        use_container_width=True,
        disabled=st.session_state.running,
    )
    stop_clicked = col_stop.button(
        "⏹ Stop Translation",
        use_container_width=True,
        disabled=not st.session_state.running,
    )
    clear_clicked = col_clear.button(
        "🗑 Clear Text",
        use_container_width=True,
    )

    if start_clicked:
        st.session_state.running = True
        st.session_state.read_failures = 0
        st.session_state.last_hand_time = time.time()
    if stop_clicked:
        st.session_state.running = False
        release_camera(camera_index)
    if clear_clicked:
        reset_translation_state()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    """Application entry point."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🤟", layout="wide")
    init_session_state()

    render_header()
    render_instructions()

    (
        stability_frames,
        confidence_threshold,
        space_pause_seconds,
        camera_index,
    ) = render_sidebar()

    render_controls(camera_index)

    st.divider()

    video_col, text_col = st.columns([2, 1])
    with video_col:
        st.subheader("Live Webcam")
        video_placeholder = st.empty()
    with text_col:
        st.subheader("Prediction")
        prediction_placeholder = st.empty()
        st.subheader("Translated Sentence")
        sentence_placeholder = st.empty()

    try:
        classifier, tracker = load_resources()
    except FileNotFoundError as error:
        st.error(f"Failed to load the model: {error}")
        return

    if not st.session_state.running:
        video_placeholder.info("Webcam is off. Press **Start Translation** to begin.")
        prediction_placeholder.markdown("### `-`")
        sentence_placeholder.markdown(
            f"> {st.session_state.sentence or '_Your translated text will appear here._'}"
        )
        return

    # --- Active translation: process a single frame, then rerun to loop. ---
    # The camera is a cached singleton, so this returns the same object every
    # rerun and does NOT reopen the device.
    cap = get_camera(camera_index)

    if cap is None or not cap.isOpened():
        st.session_state.running = False
        release_camera(camera_index)
        video_placeholder.error(
            "Could not access the webcam. Check that it is connected, not in use "
            "by another app, and that browser/OS permissions are granted."
        )
        return

    ret, frame = cap.read()
    if not ret or frame is None:
        # DirectShow cameras often fail the first few reads while warming up.
        # Tolerate transient failures instead of tearing the camera down.
        st.session_state.read_failures += 1
        if st.session_state.read_failures > 30:
            st.session_state.running = False
            release_camera(camera_index)
            video_placeholder.error("Failed to read frames from the webcam.")
            return
        video_placeholder.info("Warming up the webcam...")
        time.sleep(0.05)
        st.rerun()

    st.session_state.read_failures = 0

    annotated_rgb, letter, confidence, hand_present = process_frame(
        frame, tracker, classifier
    )
    update_translation(
        letter,
        confidence,
        hand_present,
        stability_frames,
        confidence_threshold,
        space_pause_seconds,
    )

    video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
    prediction_placeholder.markdown(
        f"### `{st.session_state.current_prediction}`  \n"
        f"Confidence: **{st.session_state.current_confidence:.0%}**"
    )
    sentence_placeholder.markdown(
        f"> {st.session_state.sentence or '_Your translated text will appear here._'}"
    )

    # Small yield keeps the UI thread cooperative before looping to next frame.
    time.sleep(0.01)
    st.rerun()


if __name__ == "__main__":
    main()
