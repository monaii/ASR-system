"""
Step 9: Performance Comparison Script

Records a short microphone sample and runs the comprehensive benchmark
across Whisper and Wav2Vec2 configurations using the evaluation module.
Outputs a concise table and saves a JSON report in `reports/`.
"""

import os
import time
import json
from datetime import datetime
from typing import List

import numpy as np

try:
    from src.audio_processing import AudioCapture
    from src.evaluation_metrics import ComprehensiveEvaluator, EvaluationResult
except ImportError:
    from audio_processing import AudioCapture
    from evaluation_metrics import ComprehensiveEvaluator, EvaluationResult


def record_microphone(duration_sec: int = 15, sample_rate: int = 16000, chunk_duration: float = 0.5) -> np.ndarray:
    """Record microphone audio for `duration_sec` and return mono float32 array."""
    cap = AudioCapture(sample_rate=sample_rate, chunk_duration=chunk_duration)
    cap.start_recording()
    audio_chunks: List[np.ndarray] = []
    start = time.time()
    try:
        while time.time() - start < duration_sec:
            chunk = cap.get_audio_chunk(timeout=1.0)
            if chunk is None:
                continue
            if chunk.ndim > 1:
                chunk = chunk[:, 0]
            audio_chunks.append(chunk.astype(np.float32))
    finally:
        cap.stop_recording()

    if not audio_chunks:
        return np.zeros(int(sample_rate * duration_sec), dtype=np.float32)

    return np.concatenate(audio_chunks)


def save_report(results: List[EvaluationResult], sample_rate: int) -> str:
    """Save benchmark results to JSON under `reports/` and return path."""
    os.makedirs("reports", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("reports", f"evaluation_report_{ts}.json")

    serializable = []
    for r in results:
        entry = {
            "test_name": r.test_name,
            "model_type": r.model_type,
            "preprocessing_type": r.preprocessing_type,
            "latency": {
                "preprocessing_time": r.latency.preprocessing_time,
                "inference_time": r.latency.inference_time,
                "total_time": r.latency.total_time,
                "audio_duration": r.latency.audio_duration,
                "real_time_factor": r.latency.real_time_factor,
            },
            "accuracy": (
                {
                    "word_error_rate": r.accuracy.word_error_rate,
                    "character_error_rate": r.accuracy.character_error_rate,
                    "bleu_score": r.accuracy.bleu_score,
                    "confidence_score": r.accuracy.confidence_score,
                }
                if r.accuracy is not None
                else None
            ),
            "metadata": r.metadata,
        }
        serializable.append(entry)

    with open(path, "w") as f:
        json.dump({"sample_rate": sample_rate, "results": serializable}, f, indent=2)
    return path


def print_table(results: List[EvaluationResult]) -> None:
    """Print a simple summary table to stdout."""
    cols = [
        "test_name",
        "model_type",
        "preprocessing",
        "prep_ms",
        "infer_ms",
        "total_ms",
        "RTF",
    ]
    widths = [26, 10, 12, 9, 9, 9, 6]
    header = " ".join([c.ljust(w) for c, w in zip(cols, widths)])
    print(header)
    print("-" * len(header))
    for r in results:
        row = [
            r.test_name,
            r.model_type,
            r.preprocessing_type,
            f"{r.latency.preprocessing_time*1000:.1f}",
            f"{r.latency.inference_time*1000:.1f}",
            f"{r.latency.total_time*1000:.1f}",
            f"{r.latency.real_time_factor:.2f}",
        ]
        print(" ".join([str(v).ljust(w) for v, w in zip(row, widths)]))


def main():
    sample_rate = 16000
    print("Recording microphone sample for benchmark...")
    audio = record_microphone(duration_sec=15, sample_rate=sample_rate, chunk_duration=0.5)
    print(f"Captured {len(audio)/sample_rate:.1f}s of audio. Running benchmark...")

    evaluator = ComprehensiveEvaluator()
    results = evaluator.run_comprehensive_benchmark(test_audio=audio, sample_rate=sample_rate, reference_text=None)

    print("\nBenchmark results:")
    print_table(results)

    path = save_report(results, sample_rate)
    print(f"\nSaved JSON report to: {path} (ignored by Git)")


if __name__ == "__main__":
    main()