"""
Streaming ASR Module
Combines audio processing with model inference for real-time transcription
"""

import threading
import time
import queue
import numpy as np
from typing import Callable, Optional, Dict, List
from dataclasses import dataclass
from collections import deque

from .audio_processing import AudioCapture, AudioPreprocessor, StreamingBuffer
from .models import ASRModel


@dataclass
class TranscriptionResult:
    """Container for transcription results"""
    text: str
    confidence: float
    inference_time: float
    timestamp: float
    has_voice_activity: bool


class StreamingASR:
    """Real-time streaming ASR system"""
    
    def __init__(self, 
                 model: ASRModel,
                 sample_rate: int = 16000,
                 chunk_duration: float = 0.5,
                 window_duration: float = 2.0,
                 overlap_duration: float = 0.5,
                 min_speech_duration: float = 0.3,
                 silence_timeout: float = 1.0):
        """
        Initialize streaming ASR system
        
        Args:
            model: ASR model instance
            sample_rate: Audio sample rate
            chunk_duration: Duration of audio chunks
            window_duration: Duration of processing windows
            overlap_duration: Overlap between processing windows
            min_speech_duration: Minimum speech duration to process
            silence_timeout: Timeout for silence detection
        """
        self.model = model
        self.sample_rate = sample_rate
        self.min_speech_duration = min_speech_duration
        self.silence_timeout = silence_timeout
        
        # Initialize components
        self.audio_capture = AudioCapture(
            sample_rate=sample_rate,
            chunk_duration=chunk_duration
        )
        
        self.preprocessor = AudioPreprocessor(
            target_sample_rate=sample_rate,
            enable_vad=True,
            enable_noise_reduction=True
        )
        
        self.streaming_buffer = StreamingBuffer(
            sample_rate=sample_rate,
            window_duration=window_duration,
            overlap_duration=overlap_duration
        )
        
        # Threading and queues
        self.transcription_queue = queue.Queue()
        self.result_callback = None
        self.is_running = False
        self.processing_thread = None
        
        # State tracking
        self.last_transcription = ""
        self.speech_start_time = None
        self.last_voice_time = None
        self.transcription_history = deque(maxlen=10)
        
        # Performance metrics
        self.metrics = {
            'total_chunks': 0,
            'voice_chunks': 0,
            'transcriptions': 0,
            'total_inference_time': 0.0,
            'average_latency': 0.0
        }
        
    def set_result_callback(self, callback: Callable[[TranscriptionResult], None]):
        """Set callback function for transcription results"""
        self.result_callback = callback
        
    def start(self):
        """Start the streaming ASR system"""
        if self.is_running:
            print("Streaming ASR already running!")
            return
            
        print("Starting streaming ASR system...")
        
        # Start audio capture
        self.audio_capture.start_recording()
        
        # Start processing thread
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        print("Streaming ASR started! Speak into your microphone...")
        
    def stop(self):
        """Stop the streaming ASR system"""
        if not self.is_running:
            return
            
        print("Stopping streaming ASR system...")
        
        self.is_running = False
        self.audio_capture.stop_recording()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            
        print("Streaming ASR stopped!")
        
    def _processing_loop(self):
        """Main processing loop running in separate thread"""
        print("Processing loop started...")
        
        while self.is_running:
            try:
                # Get audio chunk
                chunk = self.audio_capture.get_audio_chunk(timeout=0.1)
                if chunk is None:
                    continue
                    
                # Preprocess audio
                processed_chunk, has_voice = self.preprocessor.preprocess(chunk)
                
                # Update metrics
                self.metrics['total_chunks'] += 1
                if has_voice:
                    self.metrics['voice_chunks'] += 1
                    self.last_voice_time = time.time()
                    
                    if self.speech_start_time is None:
                        self.speech_start_time = time.time()
                        
                # Add to streaming buffer
                window = self.streaming_buffer.add_audio(processed_chunk)
                
                if window is not None:
                    self._process_window(window, has_voice)
                    
                # Check for silence timeout
                if (self.last_voice_time is not None and 
                    time.time() - self.last_voice_time > self.silence_timeout):
                    self._handle_silence_timeout()
                    
            except Exception as e:
                print(f"Error in processing loop: {e}")
                continue
                
        print("Processing loop ended.")
        
    def _process_window(self, window: np.ndarray, has_voice: bool):
        """Process audio window for transcription"""
        current_time = time.time()
        
        # Skip processing if no voice activity
        if not has_voice:
            return
            
        # Check minimum speech duration
        if (self.speech_start_time is not None and 
            current_time - self.speech_start_time < self.min_speech_duration):
            return
            
        try:
            # Transcribe audio window
            start_time = time.time()
            transcription, inference_time = self.model.transcribe(window, self.sample_rate)
            total_time = time.time() - start_time
            
            # Clean up transcription
            transcription = transcription.strip()
            
            # Skip empty or very short transcriptions
            if len(transcription) < 2:
                return
                
            # Update metrics
            self.metrics['transcriptions'] += 1
            self.metrics['total_inference_time'] += inference_time
            self.metrics['average_latency'] = (
                self.metrics['total_inference_time'] / self.metrics['transcriptions']
            )
            
            # Create result
            result = TranscriptionResult(
                text=transcription,
                confidence=1.0,  # TODO: Add confidence scoring
                inference_time=inference_time,
                timestamp=current_time,
                has_voice_activity=has_voice
            )
            
            # Add to history
            self.transcription_history.append(result)
            
            # Call result callback
            if self.result_callback:
                self.result_callback(result)
                
            # Update last transcription
            self.last_transcription = transcription
            
            print(f"[{current_time:.1f}] Transcription: '{transcription}' "
                  f"(inference: {inference_time:.3f}s)")
                  
        except Exception as e:
            print(f"Error during transcription: {e}")
            
    def _handle_silence_timeout(self):
        """Handle silence timeout - reset speech tracking"""
        self.speech_start_time = None
        self.last_voice_time = None
        
    def get_latest_transcription(self) -> str:
        """Get the latest transcription"""
        return self.last_transcription
        
    def get_transcription_history(self) -> List[TranscriptionResult]:
        """Get transcription history"""
        return list(self.transcription_history)
        
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        metrics = self.metrics.copy()
        
        if metrics['total_chunks'] > 0:
            metrics['voice_activity_ratio'] = metrics['voice_chunks'] / metrics['total_chunks']
        else:
            metrics['voice_activity_ratio'] = 0.0
            
        return metrics
        
    def clear_history(self):
        """Clear transcription history"""
        self.transcription_history.clear()
        self.last_transcription = ""


class MultiModelASR:
    """ASR system that can compare multiple models"""
    
    def __init__(self, models: Dict[str, ASRModel]):
        """
        Initialize multi-model ASR system
        
        Args:
            models: Dictionary of model_name -> ASRModel instances
        """
        self.models = models
        self.streaming_asrs = {}
        self.results = {name: [] for name in models.keys()}
        
        # Initialize streaming ASR for each model
        for name, model in models.items():
            asr = StreamingASR(model)
            asr.set_result_callback(lambda result, model_name=name: self._collect_result(model_name, result))
            self.streaming_asrs[name] = asr
            
    def _collect_result(self, model_name: str, result: TranscriptionResult):
        """Collect results from different models"""
        self.results[model_name].append(result)
        print(f"[{model_name}] {result.text} (latency: {result.inference_time:.3f}s)")
        
    def start_comparison(self, duration: float = 30.0):
        """Start comparison between models"""
        print(f"Starting {duration}s comparison between models: {list(self.models.keys())}")
        
        # Start all ASR systems
        for asr in self.streaming_asrs.values():
            asr.start()
            
        # Wait for specified duration
        time.sleep(duration)
        
        # Stop all ASR systems
        for asr in self.streaming_asrs.values():
            asr.stop()
            
        return self.get_comparison_results()
        
    def get_comparison_results(self) -> Dict:
        """Get comparison results between models"""
        comparison = {}
        
        for name, asr in self.streaming_asrs.items():
            metrics = asr.get_metrics()
            results = self.results[name]
            
            comparison[name] = {
                'metrics': metrics,
                'transcription_count': len(results),
                'average_inference_time': np.mean([r.inference_time for r in results]) if results else 0,
                'total_text_length': sum(len(r.text) for r in results),
                'sample_transcriptions': [r.text for r in results[-5:]]  # Last 5 transcriptions
            }
            
        return comparison


if __name__ == "__main__":
    # Test streaming ASR
    from .models import get_model
    
    print("Testing streaming ASR...")
    
    # Load a lightweight model for testing
    model = get_model("whisper", model_size="tiny")
    model.load_model()
    
    # Create streaming ASR
    streaming_asr = StreamingASR(model)
    
    # Set up result callback
    def print_result(result: TranscriptionResult):
        print(f"Result: '{result.text}' (confidence: {result.confidence:.2f}, "
              f"latency: {result.inference_time:.3f}s)")
              
    streaming_asr.set_result_callback(print_result)
    
    # Run for 10 seconds
    streaming_asr.start()
    time.sleep(10)
    streaming_asr.stop()
    
    # Print metrics
    metrics = streaming_asr.get_metrics()
    print(f"Metrics: {metrics}")
    
    print("Streaming ASR test completed!")