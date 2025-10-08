"""
ASR Live Demo UI (Step 8)
Gradio interface that starts our StreamingASR pipeline and shows live transcription.
"""

import time
import queue
from typing import Generator

import gradio as gr

try:
    # Package relative imports
    from src.models import get_model
    from src.streaming_asr import StreamingASR
except ImportError:
    # Fallback for direct execution if src is not a package in sys.path
    from models import get_model
    from streaming_asr import StreamingASR


def record_and_transcribe(
    model_type: str,
    whisper_size: str,
    wav2_name: str,
    sample_rate: int,
    chunk_duration: float,
    window_duration: float,
    overlap_duration: float,
    min_speech_duration: float,
    silence_timeout: float,
    duration: int,
) -> Generator[str, None, None]:
    """
    Start microphone capture and stream transcriptions for `duration` seconds.
    Yields incremental transcription text updates for Gradio to display.
    """

    # Initialize model
    if model_type == "whisper":
        model = get_model("whisper", model_size=whisper_size)
    else:
        model = get_model("wav2vec2", model_name=wav2_name)
    model.load_model()

    # Set up streaming ASR
    asr = StreamingASR(
        model,
        sample_rate=sample_rate,
        chunk_duration=chunk_duration,
        window_duration=window_duration,
        overlap_duration=overlap_duration,
        min_speech_duration=min_speech_duration,
        silence_timeout=silence_timeout,
    )

    # Queue to receive transcription updates via callback
    updates: queue.Queue[str] = queue.Queue()

    def _on_result(res):
        try:
            updates.put_nowait(res.text)
        except Exception:
            pass

    asr.set_result_callback(_on_result)
    asr.start()

    last_text = ""
    start = time.time()
    try:
        while time.time() - start < duration:
            try:
                text = updates.get(timeout=0.25)
                last_text = text
                yield text
            except queue.Empty:
                # No new text; keep UI responsive with latest
                yield last_text
            time.sleep(0.05)
    finally:
        asr.stop()

    # Final yield after stopping
    yield last_text


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="ASR Live Demo") as demo:
        gr.Markdown(
            "# ASR Live Demo\n"
            "Speak into your microphone and see live transcriptions.\n"
            "Choose model settings, then click Start Recording."
        )

        with gr.Row():
            model_type = gr.Radio(["whisper", "wav2vec2"], value="whisper", label="Model")
            whisper_size = gr.Dropdown(["tiny", "base"], value="tiny", label="Whisper size")
            wav2_name = gr.Textbox(
                value="facebook/wav2vec2-base-960h", label="Wav2Vec2 model name"
            )

        with gr.Row():
            sample_rate = gr.Slider(8000, 48000, value=16000, step=1000, label="Sample rate")
            chunk_duration = gr.Slider(0.1, 1.0, value=0.5, step=0.1, label="Chunk duration (s)")
            window_duration = gr.Slider(1.0, 4.0, value=2.0, step=0.5, label="Window duration (s)")
            overlap_duration = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Overlap (s)")

        with gr.Row():
            min_speech_duration = gr.Slider(0.0, 1.0, value=0.3, step=0.1, label="Min speech (s)")
            silence_timeout = gr.Slider(0.5, 3.0, value=1.0, step=0.1, label="Silence timeout (s)")
            duration = gr.Slider(5, 60, value=20, step=1, label="Run duration (s)")

        start_btn = gr.Button("Start Recording")
        output = gr.Textbox(label="Transcription", lines=8)

        start_btn.click(
            record_and_transcribe,
            inputs=[
                model_type,
                whisper_size,
                wav2_name,
                sample_rate,
                chunk_duration,
                window_duration,
                overlap_duration,
                min_speech_duration,
                silence_timeout,
                duration,
            ],
            outputs=output,
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()