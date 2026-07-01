# 🤟 Sign Language Translator

Translate American Sign Language (ASL) alphabet gestures into English text in
real time using your webcam. Built with **Streamlit**, **MediaPipe**, and a
pre-trained **scikit-learn** classifier.

---

## 📋 Project Overview

This application performs **real-time sign language alphabet recognition**. It
captures frames from your webcam, detects a hand with MediaPipe, extracts a
normalized landmark feature vector, and classifies the gesture (A–Z) with a
pre-trained Random Forest model. Stabilized predictions are appended to a
sentence, and spaces are inserted automatically when you pause between words.

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

1. Press **Start Translation**.
2. Allow webcam permission when prompted.
3. Show one alphabet clearly with your hand and hold it steady.
4. Pause (or remove your hand) for ~2 seconds to insert a space.
5. Press **Stop Translation** to release the webcam.

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
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Deployment

### Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set **`app.py`** as the entry point and deploy.

> **Webcam note:** This app uses OpenCV (`cv2.VideoCapture`) to access a webcam
> attached to the machine **running** the app, which is ideal for local demos.
> On a remote host (e.g. Streamlit Cloud) there is no server-side camera. To
> capture the **user's browser** camera in a hosted deployment, integrate
> [`streamlit-webrtc`](https://github.com/whitphd/streamlit-webrtc), which
> streams frames from the browser to the same processing pipeline in
> `process_frame`.

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
