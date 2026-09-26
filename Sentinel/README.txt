# SentinelVoice — AI Fake Call Detector
### Built for Brewing Codes 4.0 Hackathon 2026 | MUIT Noida

---

## What this project does
Detects whether an audio clip is a real human voice
or an AI-generated/cloned voice — in real time.

It extracts audio features (MFCC, pitch, jitter) and
runs them through a trained ML model to give a verdict
with a confidence score and plain-English explanation.

---

## Setup (do this once)

### 1. Install Python libraries
Open your terminal / command prompt and run:

    pip install librosa scikit-learn numpy flask flask-cors joblib soundfile

---

## How to run (do this every time)

### Step 1 — Generate training data
    python step1_generate_data.py

This creates a data/ folder with synthetic voice features.

### Step 2 — Train the model
    python step2_train_model.py

This trains a Random Forest classifier and saves it to model/.
Takes about 30 seconds on CPU.

### Step 3 — Start the backend server
    python step3_app.py

Keep this terminal open. You'll see:
    SentinelVoice API is running!
    Open index.html in your browser
    API: http://localhost:5000

### Step 4 — Open the UI
Open index.html in your browser (double-click it).

---

## How to use the app
1. Click the upload zone and pick any WAV/MP3 audio file
   OR click "Record Live" to record from your microphone
2. Click "Analyze Audio"
3. See the verdict: REAL or FAKE, with confidence % and reason

---

## Project structure

    ai_call_detector/
    ├── step1_generate_data.py   ← Creates training dataset
    ├── step2_train_model.py     ← Trains the ML model
    ├── step3_app.py             ← Flask backend API
    ├── index.html               ← Web UI (open in browser)
    ├── requirements.txt         ← Python dependencies
    ├── data/                    ← Created by step 1
    │   ├── X_features.npy
    │   └── y_labels.npy
    └── model/                   ← Created by step 2
        ├── detector.pkl
        └── scaler.pkl

---

## For the hackathon demo — do this for maximum impact

1. Download a real voice sample (record yourself saying something)
2. Generate a fake voice using any free TTS tool
   (ElevenLabs free tier, Murf.ai, or Google TTS)
3. Run both through the app live on stage
4. Show the REAL getting green + FAKE getting red in real time

That 10-second live demo will win the audience over instantly.

---

## Tech stack
- librosa      : audio feature extraction
- scikit-learn : Random Forest classifier
- Flask        : backend REST API
- HTML/CSS/JS  : frontend UI (no framework needed)

---

## Upgrading for production (post-hackathon)
- Replace synthetic data with ASVspoof 2021 dataset
- Swap Random Forest for a CNN on mel-spectrograms
- Add real-time mic streaming (WebRTC chunks every 2 seconds)
- Add Indian language fine-tuning (Hindi, Tamil, Telugu)
