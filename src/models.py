"""
ASR Models Module
Handles loading and managing different ASR models (Whisper, Wav2Vec2)
"""

import torch
import torchaudio
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    pipeline
)
import time
import numpy as np
from typing import Union, Tuple, Optional


class ASRModel:
    """Base class for ASR models"""
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.processor = None
        
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load the model - to be implemented by subclasses"""
        raise NotImplementedError
    
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio - to be implemented by subclasses"""
        raise NotImplementedError


class WhisperModel(ASRModel):
    """Whisper ASR Model wrapper"""
    
    def __init__(self, model_size: str = "base", device: str = "auto"):
        model_name = f"openai/whisper-{model_size}"
        super().__init__(model_name, device)
        self.model_size = model_size
        
    def load_model(self):
        """Load Whisper model and processor"""
        print(f"Loading Whisper {self.model_size} model on {self.device}...")
        
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        # Use pipeline for easier inference
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=0 if self.device == "cuda" else -1,
            return_timestamps=True
        )
        
        print(f"Whisper {self.model_size} loaded successfully!")
        
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio using Whisper"""
        if self.pipe is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            
        # Normalize audio to [-1, 1] range
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
            
        start_time = time.time()
        result = self.pipe(audio, return_timestamps=True)
        inference_time = time.time() - start_time
        
        return result["text"], inference_time


class Wav2Vec2Model(ASRModel):
    """Wav2Vec2 ASR Model wrapper"""
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", device: str = "auto"):
        super().__init__(model_name, device)
        
    def load_model(self):
        """Load Wav2Vec2 model and processor"""
        print(f"Loading Wav2Vec2 model on {self.device}...")
        
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        print("Wav2Vec2 loaded successfully!")
        
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[str, float]:
        """Transcribe audio using Wav2Vec2"""
        if self.model is None or self.processor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            
        start_time = time.time()
        
        # Process audio
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            logits = self.model(**inputs).logits
            
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        inference_time = time.time() - start_time
        
        return transcription, inference_time


def get_model(model_type: str, **kwargs) -> ASRModel:
    """Factory function to get ASR models"""
    if model_type.lower() == "whisper":
        model_size = kwargs.get("model_size", "base")
        return WhisperModel(model_size=model_size, device=kwargs.get("device", "auto"))
    elif model_type.lower() == "wav2vec2":
        model_name = kwargs.get("model_name", "facebook/wav2vec2-base-960h")
        return Wav2Vec2Model(model_name=model_name, device=kwargs.get("device", "auto"))
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model loading
    print("Testing model loading...")
    
    # Test Whisper
    whisper_model = get_model("whisper", model_size="tiny")
    whisper_model.load_model()
    
    # Test Wav2Vec2
    wav2vec2_model = get_model("wav2vec2")
    wav2vec2_model.load_model()
    
    print("All models loaded successfully!")