"""Best-effort image, video-frame, and English text screening."""

import gc
import os
from pathlib import Path
import subprocess
import tempfile


IMAGE_MODEL = "Organika/sdxl-detector"
TEXT_MODEL = "Hello-SimpleAI/chatgpt-detector-roberta"
_loaded_classifier_key = None
_loaded_classifier = None


def _get_classifier(task, model_name):
    global _loaded_classifier_key, _loaded_classifier

    key = (task, model_name)
    if key != _loaded_classifier_key:
        _loaded_classifier = None
        _loaded_classifier_key = None
        gc.collect()
        _loaded_classifier = _load_classifier(task, model_name)
        _loaded_classifier_key = key
    return _loaded_classifier


def _image_classifier():
    return _get_classifier("image-classification", IMAGE_MODEL)


def _text_classifier():
    return _get_classifier("text-classification", TEXT_MODEL)


def _load_classifier(task, model_name):
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Install multimodal dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    try:
        return pipeline(task, model=model_name, device=-1)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {model_name}. Check your internet connection "
            f"and installed PyTorch/Transformers packages. Details: {exc}"
        ) from exc


def _ai_score(prediction):
    label = str(prediction["label"]).strip().lower()
    score = float(prediction["score"])
    if any(term in label for term in ("artificial", "chatgpt", "ai", "fake", "synthetic")):
        return score
    if any(term in label for term in ("human", "real", "original")):
        return 1.0 - score
    raise RuntimeError(f"The selected detector returned an unknown label: {label}")


def _result(score, modality, verdict_label, reason):
    ai_confidence = round(max(0.0, min(1.0, score)) * 100, 1)
    human_confidence = round(100.0 - ai_confidence, 1)
    verdict = "FAKE" if ai_confidence >= 50 else "REAL"
    risk = "HIGH" if ai_confidence >= 75 else "MEDIUM" if ai_confidence >= 55 else "LOW"
    return {
        "verdict": verdict,
        "verdict_label": verdict_label,
        "fake_confidence": ai_confidence,
        "real_confidence": human_confidence,
        "risk_level": risk,
        "reason": reason,
        "modality": modality,
        "experimental": True,
    }


def analyze_image(path):
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    score = _ai_score(_image_classifier()(image)[0])
    return _result(
        score,
        "image",
        "Likely AI-generated image" if score >= 0.5 else "Likely human-created image",
        "Experimental SDXL-image screening score; this is not a face-swap or deepfake verification.",
    )


def analyze_video(path):
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "Video frame extraction requires imageio-ffmpeg. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    with tempfile.TemporaryDirectory(prefix="sentinel_frames_") as frame_dir:
        pattern = os.path.join(frame_dir, "frame-%02d.jpg")
        result = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                "-y", "-i", path, "-t", "60", "-vf", "fps=1/2",
                "-frames:v", "8", pattern,
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        frames = sorted(Path(frame_dir).glob("frame-*.jpg"))
        if result.returncode != 0 or not frames:
            detail = result.stderr.strip()[-500:]
            raise ValueError(f"Could not extract frames from this video. {detail}")

        from PIL import Image

        classifier = _image_classifier()
        scores = []
        for frame_path in frames:
            with Image.open(frame_path) as source:
                scores.append(_ai_score(classifier(source.convert("RGB"))[0]))

    average_score = sum(scores) / len(scores)
    return _result(
        average_score,
        "video",
        "AI-like frames detected" if average_score >= 0.5 else "Mostly human-created frames",
        f"Experimental image-model score averaged over {len(scores)} frames sampled from the first minute; this does not verify face swaps or lip-sync deepfakes.",
    )


def _read_text(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix == ".docx":
        from docx import Document

        document = Document(path)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    return Path(path).read_text(encoding="utf-8-sig", errors="replace")


def analyze_text(path):
    text = _read_text(path)
    words = text.split()
    if len(words) < 60:
        raise ValueError("Text screening needs at least 60 words; shorter samples are especially unreliable.")

    chunks = [" ".join(words[index:index + 180]) for index in range(0, len(words), 180)][:8]
    classifier = _text_classifier()
    scores = []
    for chunk in chunks:
        prediction = classifier(chunk, truncation=True, max_length=510)[0]
        scores.append(_ai_score(prediction))

    average_score = sum(scores) / len(scores)
    return _result(
        average_score,
        "text",
        "AI-writing signal (experimental)" if average_score >= 0.5 else "Human-writing signal (experimental)",
        f"English ChatGPT-text classifier score averaged over {len(scores)} text chunk(s). AI-text detection is uncertain and this model does not reliably cover Hindi or every writing model.",
    )