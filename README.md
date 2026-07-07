# 🤟 Sign Language Translator

Translate American Sign Language (ASL) alphabet gestures into English text in
real time using your webcam. Built with **Streamlit**, **MediaPipe**, and a
pre-trained **scikit-learn** classifier.

---

## 📋 Project Overview

This application performs **real-time sign language alphabet recognition**. It
captures frames from your webcam **in the browser** via
[`streamlit-webrtc`](https://github.com/whitphd/streamlit-webrtc), detects a hand
with MediaPipe, extracts a normalized landmark feature vector, and classifies the
gesture (A–Z) with a pre-trained Random Forest model. Stabilized predictions are
appended to a sentence, and spaces are inserted automatically when you pause
between words.

> Because the camera is read from the **visitor's browser** (not the server),
> the app works both locally and when deployed to the internet, on desktop and
> mobile.

> The model is **not** retrained by this app — it reuses the existing
> `model/classifier.pkl` (originally `model.p`).

---

## ✨ Features

- 🎥 **Real-time webcam inference** directly in the browser
- 🔤 **Live alphabet prediction** with confidence score
- 📝 **Automatic word construction** from stabilized predictions
- ␣ **Automatic spacing** between words after a short pause
- 🧯 **Prediction stabilization** to avoid repeated-character spam (`AAAA → A`)
- 🎛️ **Tunable settings** (stability, confidence threshold, pause duration)
- 🗑️ **Clear Text** without stopping the webcam
- 🧩 **Modular, PEP8-compliant, type-hinted codebase**
- 🛡️ **Graceful webcam-failure handling**

---

## 🖼️ Screenshots

> _Add your screenshots / demo GIF to the `assets/` folder and reference them here._

| Idle | Translating |
| ---- | ----------- |
| `assets/screenshot_idle.png` | `assets/screenshot_translating.png` |

---

## ⚙️ Installation

**Prerequisites:** Python 3.9–3.11 and a working webcam.

```bash
# 1. Clone / open the project
cd Sign_Language_Detector_Streamlit_app

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running Locally

```bash
streamlit run app.py
```

The app opens in your browser (default `http://localhost:8501`).

1. Press **START** on the video panel.
2. Allow webcam permission when prompted.
3. Show one alphabet clearly with your hand and hold it steady.
4. Pause (or remove your hand) for ~2 seconds to insert a space.
5. Press **STOP** on the video panel to end the session.

---

## 🧠 How the Model Works

1. **Hand detection** — MediaPipe Hands locates 21 landmarks on a single hand.
2. **Feature extraction** — For each landmark, the `(x, y)` coordinates are made
   translation-invariant by subtracting the hand's minimum `x` and `y`,
   producing a **42-dimensional** feature vector (21 landmarks × 2).
3. **Classification** — A pre-trained **Random Forest** predicts a class index
   `0–25`, mapped to letters **A–Z**. `predict_proba` provides a confidence.
4. **Stabilization & spacing** — A letter is committed only after it stays
   stable for several consecutive frames (and never twice in a row). When no
   confident hand is seen for the configured pause, a single space is appended.

This preprocessing exactly matches how the original model was trained, so
accuracy is preserved.

---

## 🗂️ Project Structure

```
Sign_Language_Detector_Streamlit_app/
│
├── app.py                      # Streamlit UI + real-time loop
├── inference.py                # SignLanguageClassifier (loads model, predicts)
├── model/
│      classifier.pkl           # Pre-trained Random Forest model
│
├── utils/
│      hand_tracking.py         # MediaPipe hand detection + drawing
│      feature_extraction.py    # Landmark → feature vector
│
├── assets/                     # Screenshots / demo media
├── requirements.txt            # Python deps (streamlit-webrtc, av, ...)
├── packages.txt                # System libs for Streamlit Cloud (apt)
├── README.md
└── .gitignore
```

---

## 🚀 Deployment

### Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. In **Advanced settings**, set **Python version = 3.11** (see note below).
4. Set **`app.py`** as the entry point and deploy.

The repo already includes everything the deploy needs:

- **`requirements.txt`** — `streamlit-webrtc` + `av` for browser video, and
  `opencv-python-headless` (the GUI-free OpenCV build required on servers).
- **`packages.txt`** — the system libraries (`libgl1`, `libglib2.0-0`) MediaPipe's
  OpenCV needs on Streamlit Cloud's Debian image (they provide `libGL.so.1` and
  `libgthread-2.0.so.0`).

> **⚠️ Python version matters — use 3.11.** `scikit-learn` is pinned to `1.2.0`
> to match the model pickle, and 1.2.0 only publishes wheels up to Python 3.11.
> On Python 3.12+ (Streamlit Cloud may default to a much newer version) pip has
> no wheel and tries to build scikit-learn from source, which fails with
> `ModuleNotFoundError: No module named 'distutils'` (distutils was removed from
> the standard library in Python 3.12). Selecting **Python 3.11** makes every
> pinned package install from a prebuilt wheel and avoids the source build
> entirely.

> **Webcam note:** The camera is captured from the **visitor's browser** over
> WebRTC, so it works on a remote host with no server-side camera. A public
> Google STUN server is configured for NAT traversal. On very restrictive
> networks a peer connection can fail; adding a TURN server to
> `RTC_CONFIGURATION` in `app.py` resolves that.

### Docker (optional)

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 📄 License

Provided for educational and demonstration purposes.
