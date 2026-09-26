# SentinelVoice — AI Fake Call Detector
### Built for Brewing Codes 4.0 Hackathon 2026 | MUIT Noida

---

## What this project does
Screens audio, images, sampled video frames, and text documents for signals
associated with AI-generated content.

It extracts audio features (MFCC, pitch, jitter) and
runs them through a trained ML model to give a verdict
with a confidence score and plain-English explanation.

---

## Setup (do this once)

### 1. Install Python libraries
Open a terminal in the project folder and run this in the same Python
environment you will use to start the backend:

    python -m pip install -r requirements.txt

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
1. Choose Audio, Photo, Video, or Text.
2. Select a supported file, or record audio in Audio mode.
3. Review the experimental screening score and its limitations.

Supported media: WAV, MP3, M4A, OGG, FLAC, WEBM; PNG, JPG, WEBP, BMP;
MP4, MOV, AVI, MKV, WEBM; TXT, MD, CSV, JSON, HTML, LOG, XML, YAML, PDF,
and DOCX. Uploads are limited to 100 MB.

Image and text model weights download from Hugging Face on first use, so an
internet connection is needed the first time each model is selected. Video
screening extracts at most eight frames from the first minute and runs the
image model on those frames; it does not check audio/video lip-sync or prove
that a face is authentic.

## Detection limitations
- Scores are estimates, not proof, and no mode is 100% accurate.
- Image screening uses `Organika/sdxl-detector`, which is limited to SDXL-like
    generated images and is licensed CC BY-NC 3.0 (non-commercial use only).
- Text screening uses `Hello-SimpleAI/chatgpt-detector-roberta`, trained on
    English ChatGPT-era text. Hindi, edited text, short samples, and newer models
    may be misclassified. At least 60 words are required.
- The audio model was trained on a very small number of real recordings plus
    synthetic feature samples; its confidence is not calibrated for general use.

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
