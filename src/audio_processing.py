"""
Audio Processing Module
Handles real-time audio capture, preprocessing, and feature extraction
"""

import sounddevice as sd
import numpy as np
import librosa
import threading
import queue
import time
from typing import Optional, Callable, Tuple
import scipy.signal
try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    print("webrtcvad not available. Voice activity detection will be disabled.")

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("noisereduce not available. Noise reduction will be disabled.")


class AudioCapture:
    """Real-time audio capture using sounddevice"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_duration: float = 0.5,
                 buffer_duration: float = 3.0):
        """
        Initialize audio capture
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono)
            chunk_duration: Duration of each audio chunk in seconds
            buffer_duration: Duration of audio buffer for processing in seconds
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.buffer_duration = buffer_duration
        
        self.chunk_size = int(sample_rate * chunk_duration)
        self.buffer_size = int(sample_rate * buffer_duration)
        
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None
        
        # Audio buffer for continuous processing
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        
    def _audio_callback(self, indata, frames, time, status):
        """Callback function for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
            
        # Add new audio data to queue
        audio_chunk = indata[:, 0] if self.channels == 1 else indata
        self.audio_queue.put(audio_chunk.copy())
        
    def start_recording(self):
        """Start audio recording"""
        if self.is_recording:
            print("Already recording!")
            return
            
        print(f"Starting audio recording at {self.sample_rate}Hz...")
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=self.chunk_size,
            dtype=np.float32
        )
        
        self.stream.start()
        self.is_recording = True
        print("Recording started!")
        
    def stop_recording(self):
        """Stop audio recording"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
        print("Recording stopped!")
        
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get the next audio chunk from the queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_audio_buffer(self) -> np.ndarray:
        """Get current audio buffer for processing"""
        # Collect all available chunks and update buffer
        while not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get_nowait()
                # Shift buffer and add new chunk
                shift_size = len(chunk)
                self.audio_buffer[:-shift_size] = self.audio_buffer[shift_size:]
                self.audio_buffer[-shift_size:] = chunk
            except queue.Empty:
                break
                
        return self.audio_buffer.copy()


class AudioPreprocessor:
    """Audio preprocessing pipeline"""
    
    def __init__(self, 
                 target_sample_rate: int = 16000,
                 enable_vad: bool = True,
                 enable_noise_reduction: bool = True,
                 vad_aggressiveness: int = 2):
        """
        Initialize audio preprocessor
        
        Args:
            target_sample_rate: Target sample rate for processing
            enable_vad: Enable voice activity detection
            enable_noise_reduction: Enable noise reduction
            vad_aggressiveness: VAD aggressiveness level (0-3)
        """
        self.target_sample_rate = target_sample_rate
        self.enable_vad = enable_vad and WEBRTC_AVAILABLE
        self.enable_noise_reduction = enable_noise_reduction and NOISEREDUCE_AVAILABLE
        
        # Initialize VAD
        if self.enable_vad:
            self.vad = webrtcvad.Vad(vad_aggressiveness)
        else:
            self.vad = None
            
        # Noise profile for noise reduction
        self.noise_profile = None
        
    def resample_audio(self, audio: np.ndarray, original_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if original_sr == self.target_sample_rate:
            return audio
            
        return librosa.resample(audio, orig_sr=original_sr, target_sr=self.target_sample_rate)
        
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
        
    def apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise reduction if enabled"""
        if not self.enable_noise_reduction:
            return audio
            
        try:
            # Use stationary noise reduction
            reduced_audio = nr.reduce_noise(y=audio, sr=self.target_sample_rate, stationary=True)
            return reduced_audio
        except Exception as e:
            print(f"Noise reduction failed: {e}")
            return audio
            
    def detect_voice_activity(self, audio: np.ndarray) -> bool:
        """Detect voice activity in audio chunk"""
        if not self.enable_vad or self.vad is None:
            # Simple energy-based VAD fallback
            energy = np.mean(audio ** 2)
            return energy > 0.001  # Threshold for voice activity
            
        # Convert to 16-bit PCM for webrtcvad
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # webrtcvad requires specific frame sizes (10, 20, or 30ms)
        frame_duration = 30  # ms
        frame_size = int(self.target_sample_rate * frame_duration / 1000)
        
        # Check if we have enough samples
        if len(audio_int16) < frame_size:
            return False
            
        # Take the first frame for VAD
        frame = audio_int16[:frame_size].tobytes()
        
        try:
            return self.vad.is_speech(frame, self.target_sample_rate)
        except Exception as e:
            print(f"VAD error: {e}")
            return False
            
    def preprocess(self, audio: np.ndarray, original_sr: int = None) -> Tuple[np.ndarray, bool]:
        """
        Complete preprocessing pipeline
        
        Returns:
            Tuple of (processed_audio, has_voice_activity)
        """
        if original_sr is None:
            original_sr = self.target_sample_rate
            
        # Resample if needed
        if original_sr != self.target_sample_rate:
            audio = self.resample_audio(audio, original_sr)
            
        # Normalize
        audio = self.normalize_audio(audio)
        
        # Apply noise reduction
        audio = self.apply_noise_reduction(audio)
        
        # Detect voice activity
        has_voice = self.detect_voice_activity(audio)
        
        return audio, has_voice


class StreamingBuffer:
    """Streaming buffer for continuous transcription"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 window_duration: float = 2.0,
                 overlap_duration: float = 0.5):
        """
        Initialize streaming buffer
        
        Args:
            sample_rate: Audio sample rate
            window_duration: Duration of processing window in seconds
            overlap_duration: Overlap between windows in seconds
        """
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.overlap_duration = overlap_duration
        
        self.window_size = int(sample_rate * window_duration)
        self.overlap_size = int(sample_rate * overlap_duration)
        self.step_size = self.window_size - self.overlap_size
        
        self.buffer = np.zeros(self.window_size, dtype=np.float32)
        self.buffer_position = 0
        
    def add_audio(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Add audio chunk to buffer and return window if ready
        
        Returns:
            Audio window if ready for processing, None otherwise
        """
        chunk_size = len(audio_chunk)
        
        # Add chunk to buffer
        if self.buffer_position + chunk_size <= self.window_size:
            self.buffer[self.buffer_position:self.buffer_position + chunk_size] = audio_chunk
            self.buffer_position += chunk_size
        else:
            # Buffer overflow - shift and add
            remaining_space = self.window_size - self.buffer_position
            self.buffer[self.buffer_position:] = audio_chunk[:remaining_space]
            
            # Shift buffer
            self.buffer[:-chunk_size] = self.buffer[chunk_size:]
            self.buffer[-chunk_size:] = audio_chunk
            self.buffer_position = self.window_size
            
        # Return window if buffer is full
        if self.buffer_position >= self.window_size:
            window = self.buffer.copy()
            
            # Shift buffer for next window
            self.buffer[:-self.step_size] = self.buffer[self.step_size:]
            self.buffer[-self.step_size:] = 0
            self.buffer_position = self.window_size - self.step_size
            
            return window
            
        return None


if __name__ == "__main__":
    # Test audio capture
    print("Testing audio capture...")
    
    capture = AudioCapture(sample_rate=16000, chunk_duration=0.5)
    preprocessor = AudioPreprocessor()
    
    print("Starting 5-second recording test...")
    capture.start_recording()
    
    start_time = time.time()
    while time.time() - start_time < 5.0:
        chunk = capture.get_audio_chunk(timeout=0.1)
        if chunk is not None:
            processed_audio, has_voice = preprocessor.preprocess(chunk)
            print(f"Chunk: {len(chunk)} samples, Voice: {has_voice}, Energy: {np.mean(processed_audio**2):.6f}")
            
    capture.stop_recording()
    print("Audio capture test completed!")