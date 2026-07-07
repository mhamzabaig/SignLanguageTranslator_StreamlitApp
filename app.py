"""Sign Language Translator — a browser-based real-time ASL alphabet recognizer.

Uses ``streamlit-webrtc`` so the video is captured from the *visitor's own*
browser (desktop or mobile) and streamed to the server over WebRTC. This is what
makes the app work when deployed to the internet (Streamlit Community Cloud,
Hugging Face Spaces, a VPS, etc.), where there is no server-side webcam --
unlike ``cv2.VideoCapture``, which only ever sees a camera attached to the host.

Frame processing runs inside ``VideoProcessorBase.recv``, which streamlit-webrtc
calls on a **background worker thread** -- not the Streamlit script thread. That
has two consequences the code below is built around:

1. ``st.session_state`` must NOT be touched from ``recv`` (it belongs to the
   script thread). All per-prediction state therefore lives on the processor
   instance and is guarded by a ``threading.Lock``.
2. The main thread reads a thread-safe ``snapshot()`` of that state to render the
   live prediction and sentence, and pushes the sidebar settings down to the
   processor the same way.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

from inference import SignLanguageClassifier
from utils.feature_extraction import extract_features, get_bounding_box
from utils.hand_tracking import HandTracker

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PAGE_TITLE = "Sign Language Translator"
PAGE_DESCRIPTION = (
    "Deploy This App at Front Desks of Hospitals, Schools, Banks, Government Offices, and Other Public Places to Help Deaf People Communicate in Real time using your webcam."
)
DEFAULT_STABILITY_FRAMES = 12
DEFAULT_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_SPACE_PAUSE_SECONDS = 2.0

# End-to-end lag is dominated by how much work happens per frame on the server
# (decode -> hand detection -> re-encode), not by raw bandwidth: streamlit-webrtc
# already drops stale frames and only processes the newest one. So the biggest
# levers are (1) asking the browser for a smaller, slower stream and (2) never
# processing a frame larger than we need. 640x480 keeps hand detection accurate
# while cutting the pixels to decode/detect/encode by ~4x vs 720p.
#
# These are "ideal" (best-effort) hints only -- NOT hard "max" limits. A hard max
# makes getUserMedia raise OverconstrainedError on cameras that can't hit it, so
# the stream never starts. The server-side clamp below (PROCESS_MAX_WIDTH) is the
# real guarantee, so ideal hints here are enough.
VIDEO_CONSTRAINTS = {
    "width": {"ideal": 640},
    "height": {"ideal": 480},
    "frameRate": {"ideal": 15},
}
# Safety net: some cameras ignore the request above and send larger frames, so
# we also clamp on the server before doing any work.
PROCESS_MAX_WIDTH = 640

# A public STUN server lets the browser and server negotiate a peer-to-peer
# connection across NATs/firewalls, which is what makes the stream work for a
# remote visitor (including on mobile networks) rather than only on localhost.
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# --------------------------------------------------------------------------- #
# Resource loading (cached so the model loads only once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def load_classifier() -> SignLanguageClassifier:
    """Load the pickled classifier a single time and share it across reruns.

    The classifier is read-only at inference time, so the same instance is safe
    to use from the WebRTC worker thread. The ``HandTracker`` (MediaPipe) is
    deliberately NOT cached here -- it is created per processor instance so it is
    only ever driven from that processor's single worker thread.
    """
    return SignLanguageClassifier()


# --------------------------------------------------------------------------- #
# Frame processing (stateless -- safe to call from the worker thread)
# --------------------------------------------------------------------------- #
def process_frame(
    frame_bgr: np.ndarray,
    tracker: HandTracker,
    classifier: SignLanguageClassifier,
) -> Tuple[np.ndarray, Optional[str], float, bool]:
    """Detect a hand, classify it, and annotate the frame.

    Args:
        frame_bgr: Raw BGR frame from the browser.
        tracker: The processor's hand tracker.
        classifier: The shared sign language classifier.

    Returns:
        A tuple ``(annotated_bgr, letter, confidence, hand_present)`` where
        ``letter`` is the predicted character or ``None``, ``confidence`` is in
        ``[0, 1]``, and ``hand_present`` indicates whether a hand was detected.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    hand_landmarks = tracker.detect(frame_rgb)

    if hand_landmarks is None:
        return frame_bgr, None, 0.0, False

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

    return frame_bgr, letter, confidence, True


# --------------------------------------------------------------------------- #
# WebRTC video processor
# --------------------------------------------------------------------------- #
class SignLanguageProcessor(VideoProcessorBase):
    """Processes each browser video frame and builds up the translated sentence.

    All translation state lives on the instance (not ``st.session_state``)
    because ``recv`` runs on a WebRTC worker thread. A single lock guards every
    read/write so the Streamlit script thread can safely read a ``snapshot`` and
    push new settings while frames are being processed.
    """

    def __init__(self, classifier: SignLanguageClassifier) -> None:
        self._classifier = classifier
        self._tracker = HandTracker()
        self._lock = threading.Lock()

        # Tuning settings -- overwritten from the main thread via update_settings.
        self._stability_frames = DEFAULT_STABILITY_FRAMES
        self._confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        self._space_pause_seconds = DEFAULT_SPACE_PAUSE_SECONDS

        # Translation / stabilization state.
        self._sentence = ""
        self._candidate_char: Optional[str] = None
        self._candidate_count = 0
        self._last_committed: Optional[str] = None
        self._last_hand_time = time.time()
        self._space_inserted = True  # nothing typed yet, so no space owed
        self._current_prediction = "-"
        self._current_confidence = 0.0
        self._clear_requested = False

    # -- Cross-thread API (called from the Streamlit script thread) --------- #
    def update_settings(
        self,
        stability_frames: int,
        confidence_threshold: float,
        space_pause_seconds: float,
    ) -> None:
        """Push the current sidebar settings down to the processor."""
        with self._lock:
            self._stability_frames = stability_frames
            self._confidence_threshold = confidence_threshold
            self._space_pause_seconds = space_pause_seconds

    def request_clear(self) -> None:
        """Ask the worker thread to reset the sentence on its next frame."""
        with self._lock:
            self._clear_requested = True

    def snapshot(self) -> Tuple[str, float, str]:
        """Return ``(prediction, confidence, sentence)`` for live display."""
        with self._lock:
            return self._current_prediction, self._current_confidence, self._sentence

    # -- Internal state updates (called under the lock) --------------------- #
    def _reset(self) -> None:
        """Clear the built-up sentence and per-prediction bookkeeping."""
        self._sentence = ""
        self._candidate_char = None
        self._candidate_count = 0
        self._last_committed = None
        self._space_inserted = True
        self._current_prediction = "-"
        self._current_confidence = 0.0

    def _update_translation(
        self, letter: Optional[str], confidence: float, hand_present: bool
    ) -> None:
        """Update the sentence from a new prediction (stabilization + spacing).

        A letter is committed only after it stays stable for ``stability_frames``
        consecutive frames, and never twice in a row (so "AAAA" becomes "A").
        When no confident hand is seen for ``space_pause_seconds``, a single
        space is appended, separating words.
        """
        now = time.time()
        confident = (
            hand_present
            and letter is not None
            and confidence >= self._confidence_threshold
        )

        if confident:
            self._last_hand_time = now
            self._space_inserted = False
            self._current_prediction = letter
            self._current_confidence = confidence

            if letter == self._candidate_char:
                self._candidate_count += 1
            else:
                self._candidate_char = letter
                self._candidate_count = 1

            stable = self._candidate_count == self._stability_frames
            if stable and letter != self._last_committed:
                self._sentence += letter
                self._last_committed = letter
            return

        # No confident hand this frame: reset the stabilization window so the
        # same letter can be typed again after a pause, and consider a space.
        self._candidate_char = None
        self._candidate_count = 0
        self._last_committed = None
        self._current_prediction = "-"
        self._current_confidence = 0.0

        elapsed = now - self._last_hand_time
        should_space = (
            elapsed >= self._space_pause_seconds
            and not self._space_inserted
            and self._sentence
            and not self._sentence.endswith(" ")
        )
        if should_space:
            self._sentence += " "
            self._space_inserted = True

    # -- WebRTC callback (runs on the worker thread) ------------------------ #
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process one incoming browser frame and return the annotated frame."""
        img = frame.to_ndarray(format="bgr24")

        # Downscale oversized frames before any processing. Cheaper detection and
        # a smaller frame to re-encode both directly reduce the visible lag.
        height, width = img.shape[:2]
        if width > PROCESS_MAX_WIDTH:
            scale = PROCESS_MAX_WIDTH / width
            img = cv2.resize(
                img, (PROCESS_MAX_WIDTH, int(height * scale)), interpolation=cv2.INTER_AREA
            )

        annotated, letter, confidence, hand_present = process_frame(
            img, self._tracker, self._classifier
        )

        with self._lock:
            if self._clear_requested:
                self._reset()
                self._clear_requested = False
            self._update_translation(letter, confidence, hand_present)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# --------------------------------------------------------------------------- #
# UI sections
# --------------------------------------------------------------------------- #
def render_header() -> None:
    """Render the page title and short description."""
    st.title(PAGE_TITLE)
    st.markdown("### **A Smart Sign Language Translator for Public Service Areas**")
    st.markdown(f"##### {PAGE_DESCRIPTION}")


def render_instructions() -> None:
    """Render the highlighted instructions panel."""
    st.info(
        "### Instructions\n"
        "1. Press **START** on the video panel and allow camera access when your "
        "browser prompts you.\n"
        "2. Show one alphabet clearly using your hand.\n"
        "3. Hold the gesture steady until the prediction stabilizes.\n"
        "4. To insert a **SPACE** between words, keep your hand still (or remove "
        "it from view) for 2 seconds.\n"
        "5. Press **STOP** on the video panel to end the session and free the "
        "server (nothing is processed while stopped).\n"
        "6. Use **Clear Text** to start a new sentence without stopping."
    )


def render_sidebar() -> Tuple[int, float, float]:
    """Render tuning controls and return the current settings.

    Returns:
        ``(stability_frames, confidence_threshold, space_pause_seconds)``.
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
    return stability_frames, confidence_threshold, space_pause_seconds


@st.fragment(run_every=0.15)
def render_output(ctx) -> None:
    """Poll the processor for the latest prediction/sentence and render them.

    ``run_every`` re-executes *only this fragment* on a timer, so the live text
    refreshes smoothly without re-rendering (or re-mounting) the video stream.
    While the stream is active the sentence is mirrored into ``st.session_state``
    so it survives on screen after the user stops the stream (the processor --
    and its state -- is destroyed on stop).

    The prediction/sentence elements are created *inside* the fragment (rather
    than writing into ``st.empty`` placeholders defined in the main script).
    A timer fragment owns and redraws its own output region on each tick, and
    recent Streamlit raises ``StreamlitAPIException`` if a fragment enqueues into
    a container created during an earlier full-script run.
    """
    vp = ctx.video_processor
    if vp is not None:
        prediction, confidence, sentence = vp.snapshot()
        st.session_state.sentence = sentence
    else:
        prediction, confidence = "-", 0.0
        sentence = st.session_state.get("sentence", "")

    st.subheader("Prediction")
    st.markdown(f"### `{prediction}`  \nConfidence: **{confidence:.0%}**")
    st.subheader("Translated Sentence")
    st.markdown(f"> {sentence or '_Your translated text will appear here._'}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    """Application entry point."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🤟", layout="wide")

    render_header()
    render_instructions()

    stability_frames, confidence_threshold, space_pause_seconds = render_sidebar()

    try:
        classifier = load_classifier()
    except FileNotFoundError as error:
        st.error(f"Failed to load the model: {error}")
        return

    video_col, text_col = st.columns([2, 1])

    with video_col:
        st.subheader("Live Webcam")

        # SENDRECV: the browser sends its camera track and receives the annotated
        # track back. async_processing keeps frame handling off the event loop so
        # the stream stays smooth. The factory injects the shared classifier.
        #
        # The component renders its own START / STOP buttons. STOP fully closes
        # the peer connection and destroys the video worker, so no frames are
        # sent or processed and server load drops to idle -- and nothing connects
        # until START is pressed, so visitors just browsing cost the server
        # nothing. (An earlier attempt to drive this from custom buttons via
        # desired_playing_state stopped the feed from starting, so we rely on the
        # built-in controls, which are reliable and do exactly what we need.)
        ctx = webrtc_streamer(
            key="sign-language",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: SignLanguageProcessor(classifier),
            media_stream_constraints={"video": VIDEO_CONSTRAINTS, "audio": False},
            async_processing=True,
        )

    with text_col:
        # The live prediction/sentence are drawn by the timer fragment, which
        # creates its own elements (see render_output) so it can safely redraw
        # them on every tick without touching main-script containers.
        render_output(ctx)
        if st.button("🗑 Clear Text", use_container_width=True):
            st.session_state.sentence = ""
            if ctx.video_processor is not None:
                ctx.video_processor.request_clear()

    # Push the current sidebar settings down to the running processor (no-op when
    # the stream is stopped and there is no processor).
    if ctx.video_processor is not None:
        ctx.video_processor.update_settings(
            stability_frames, confidence_threshold, space_pause_seconds
        )


if __name__ == "__main__":
    main()
