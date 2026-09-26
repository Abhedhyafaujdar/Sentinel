# step3_app.py - FINAL VERSION with webm support
import numpy as np
import joblib
import librosa
import os
import sys
import socket
import shutil
import warnings
import subprocess
import tempfile
import soundfile as sf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from multimodal_detection import analyze_image, analyze_text, analyze_video

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

if not os.path.exists("model/detector.pkl"):
    print("ERROR: Run step2_train_model.py first!")
    exit()

print("Loading model...")
model  = joblib.load("model/detector.pkl")
scaler = joblib.load("model/scaler.pkl")
print("Model loaded!")

# -------------------------------------------------------
# Convert ANY audio format to WAV using ffmpeg if available
# Falls back to librosa audioread for mp3/webm
# -------------------------------------------------------
def convert_to_wav(input_path, output_path):
    """Try ffmpeg first, then pydub, then just use librosa directly."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, OSError, RuntimeError):
        ffmpeg = shutil.which("ffmpeg")

    if ffmpeg:
        try:
            result = subprocess.run(
                [ffmpeg, "-y", "-i", input_path, "-ar", "16000",
                 "-ac", "1", "-f", "wav", output_path],
                capture_output=True, timeout=30
            )
            if result.returncode == 0 and os.path.exists(output_path):
                return True
        except Exception:
            pass

    # Try pydub
    try:
        from pydub import AudioSegment
        ext = os.path.splitext(input_path)[1].lower().replace(".", "")
        if ext == "webm":
            audio = AudioSegment.from_file(input_path, format="webm")
        elif ext == "mp3":
            audio = AudioSegment.from_mp3(input_path)
        elif ext == "m4a":
            audio = AudioSegment.from_file(input_path, format="m4a")
        elif ext == "ogg":
            audio = AudioSegment.from_ogg(input_path)
        else:
            audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
        return True
    except Exception:
        pass

    return False


def load_audio_any_format(path):
    """Load audio from any format, returning (y, sr)."""
    ext = os.path.splitext(path)[1].lower()

    # For WAV — use soundfile directly (fastest)
    if ext == ".wav":
        try:
            raw, sr_native = sf.read(path, always_2d=False)
            if len(raw.shape) > 1:
                raw = raw.mean(axis=1)
            y = librosa.resample(raw.astype(np.float32),
                                 orig_sr=sr_native, target_sr=16000)
            return y, 16000
        except Exception:
            pass

    # For webm/mp3/m4a/ogg — convert to wav first
    tmp_wav = path + "_converted.wav"
    converted = convert_to_wav(path, tmp_wav)

    if converted and os.path.exists(tmp_wav):
        try:
            raw, sr_native = sf.read(tmp_wav, always_2d=False)
            if len(raw.shape) > 1:
                raw = raw.mean(axis=1)
            y = librosa.resample(raw.astype(np.float32),
                                 orig_sr=sr_native, target_sr=16000)
            os.remove(tmp_wav)
            return y, 16000
        except Exception:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

    # Last resort — librosa with audioread (handles mp3, webm sometimes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(path, sr=16000, mono=True)
    return y, sr


def extract_deep_features(path):
    y, sr = load_audio_any_format(path)

    if y is None or len(y) < sr * 0.3:
        raise ValueError("Audio too short or unreadable. Need at least 0.3 seconds.")

    print(f"  Audio loaded: {len(y)/sr:.2f}s at {sr}Hz")
    feats = []

    # 40 MFCC mean + 40 std + 40 delta = 120
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    feats.extend(np.mean(mfcc, axis=1).tolist())
    feats.extend(np.std(mfcc, axis=1).tolist())
    feats.extend(np.mean(mfcc_delta, axis=1).tolist())

    # Spectral features = 7
    feats.append(float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))))
    feats.append(float(np.std(librosa.feature.spectral_centroid(y=y, sr=sr))))
    feats.append(float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))))
    feats.append(float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))))
    feats.append(float(np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr))))
    feats.append(float(np.mean(librosa.feature.spectral_flatness(y=y))))
    feats.append(float(np.std(librosa.feature.spectral_flatness(y=y))))

    # Pitch features = 6
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pv = pitches[pitches > 0]
    if len(pv) > 10:
        pm   = float(np.mean(pv))
        ps   = float(np.std(pv))
        pj   = ps / (pm + 1e-6)
        pvs  = np.sort(pv)[:500]
        pdm  = float(np.mean(np.abs(np.diff(pvs))))
        pds  = float(np.std(np.diff(pvs)))
        pr   = float(np.max(pv) - np.min(pv))
    else:
        pm=ps=pj=pdm=pds=pr=0.0
    feats.extend([pm, ps, pj, pdm, pds, pr])

    # Energy = 4
    rms = librosa.feature.rms(y=y)[0]
    feats.extend([float(np.mean(rms)), float(np.std(rms)),
                  float(np.max(rms)),
                  float(np.min(rms)/(np.mean(rms)+1e-6))])

    # ZCR = 2
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats.extend([float(np.mean(zcr)), float(np.std(zcr))])

    # Chroma = 4
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.extend([float(np.mean(chroma)), float(np.std(chroma)),
                  float(np.max(chroma)), float(np.min(chroma))])

    # Mel spectrogram stats = 10
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    feats.extend([float(np.mean(mel_db)), float(np.std(mel_db)),
                  float(np.median(mel_db)),
                  float(np.percentile(mel_db, 25)),
                  float(np.percentile(mel_db, 75))])
    band = 128 // 5
    for i in range(5):
        feats.append(float(np.mean(mel_db[i*band:(i+1)*band, :])))

    return np.array(feats, dtype=np.float32).reshape(1, -1)


def build_reason(verdict, fake_conf, feats):
    # Fixed index targets based on actual appended array position
    pj     = float(feats[0, 129])  # pitch jitter (Index 129)
    sf_val = float(feats[0, 125])  # spectral flatness (Index 125)
    es     = float(feats[0, 134])  # energy std (Index 134)

    if verdict == "FAKE":
        clues = []
        if pj  < 0.05:  clues.append("unnaturally stable pitch")
        if sf_val < 0.05: clues.append("low spectral flatness (AI smoothing)")
        if es  < 0.006: clues.append("suspiciously uniform energy levels")
        if not clues:   clues.append("multiple subtle AI synthesis patterns")
        prefix = "High-confidence" if fake_conf > 85 else "Likely" if fake_conf > 65 else "Possible"
        return f"{prefix} AI-generated voice: {', '.join(clues)} detected."
    else:
        if fake_conf < 20:
            return "Strong natural human voice: healthy pitch variation, organic spectral texture, and natural energy fluctuations confirmed."
        return "Appears human but some patterns are slightly unusual — possibly background noise or compressed recording."


@app.route("/")
@app.route("/index.html")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    f = request.files["audio"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    return analyze_audio_upload(f)


def analyze_audio_upload(upload):
    ext = os.path.splitext(upload.filename)[1].lower() or ".wav"
    handle = tempfile.NamedTemporaryFile(prefix="tmp_upload_", suffix=ext, delete=False)
    tmp = handle.name
    handle.close()
    upload.save(tmp)
    print(f"[UPLOAD] {upload.filename} ({ext}) saved for analysis")

    try:
        feats   = extract_deep_features(tmp)
        feats_s = scaler.transform(feats)
        pred    = model.predict(feats_s)[0]
        proba   = model.predict_proba(feats_s)[0]

        fake_conf = round(float(proba[1]) * 100, 1)
        real_conf = round(float(proba[0]) * 100, 1)
        verdict   = "FAKE" if pred == 1 else "REAL"
        risk      = "HIGH" if fake_conf > 75 else "MEDIUM" if fake_conf > 50 else "LOW"
        reason    = build_reason(verdict, fake_conf, feats)

        result = {
            "verdict":         verdict,
            "verdict_label":   "AI-generated voice" if verdict == "FAKE" else "Human voice",
            "fake_confidence": fake_conf,
            "real_confidence": real_conf,
            "reason":          reason,
            "risk_level":      risk,
            "modality":        "audio",
            "experimental":    True,
        }
        print(f"[RESULT] {verdict} | Fake:{fake_conf}% Real:{real_conf}% Risk:{risk}")
        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@app.route("/analyze", methods=["POST"])
def analyze_media():
    if "file" not in request.files:
        return jsonify({"error": "Choose a file to analyze."}), 400
    upload = request.files["file"]
    if not upload.filename:
        return jsonify({"error": "The uploaded file has no filename."}), 400

    mode = request.form.get("mode", "").lower()
    extension = os.path.splitext(upload.filename)[1].lower()
    allowed = {
        "audio": {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"},
        "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp"},
        "video": {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"},
        "text": {".txt", ".md", ".csv", ".json", ".html", ".log", ".xml", ".yaml", ".yml", ".pdf", ".docx"},
    }
    if mode not in allowed:
        return jsonify({"error": "Choose audio, image, video, or text mode."}), 400
    if extension not in allowed[mode]:
        return jsonify({"error": f"{extension or 'This file'} is not supported in {mode} mode."}), 415
    if mode == "audio":
        return analyze_audio_upload(upload)

    handle = tempfile.NamedTemporaryFile(prefix="tmp_upload_", suffix=extension, delete=False)
    tmp = handle.name
    handle.close()
    upload.save(tmp)
    try:
        if mode == "image":
            result = analyze_image(tmp)
        elif mode == "video":
            result = analyze_video(tmp)
        else:
            result = analyze_text(tmp)
        return jsonify(result)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except ImportError as error:
        return jsonify({"error": f"Required file reader is missing: {error}. Install dependencies from requirements.txt."}), 503
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503
    except Exception as error:
        print(f"[ERROR] {error}")
        return jsonify({"error": "The file could not be analyzed. Check its format and try another file."}), 500
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"error": "Files must be smaller than 100 MB."}), 413


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "loaded"})


if __name__ == "__main__":
    server_port = 5000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_probe:
        try:
            port_probe.bind(("127.0.0.1", server_port))
        except OSError:
            server_port = 5001

    print("\n" + "="*50)
    print("  SentinelVoice API running!")
    print(f"  Open: http://127.0.0.1:{server_port}/")
    print("="*50 + "\n")
    app.run(debug=False, host="127.0.0.1", port=server_port)
    