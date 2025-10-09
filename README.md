ASR System — Speech-to-Text Pipeline

Overview
- Real-time audio capture, streaming transcription, and enhanced preprocessing.
- Latency and accuracy evaluation toolkit with benchmarking across Whisper and Wav2Vec2.
- Live Gradio UI for microphone transcription and a benchmark script for quick comparisons.

Setup
- Python 3.11 recommended.
- Install dependencies: `python -m pip install -r requirements.txt`

Run Live Demo (UI)
- Start: `python app.py`
- Open: `http://127.0.0.1:7860/`
- Choose model (Whisper tiny/base or Wav2Vec2 by name) and speak.

Run Benchmark (Comparison)
- Record ~15s and compare configurations: `python benchmark.py`
- Outputs table in console and saves `reports/evaluation_report_YYYYMMDD_HHMMSS.json` (ignored by Git).

Project Structure
- `src/` core modules: audio capture, preprocessing, models, streaming, evaluation.
- `tests/` basic validations for capture, preprocessing, models, and streaming.

Notes
- `.gitignore` excludes generated reports and test WAVs.