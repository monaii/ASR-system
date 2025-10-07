"""
Enhanced Audio Preprocessing Pipeline
Advanced audio processing with feature extraction and optimization
"""

import numpy as np
import librosa
import scipy.signal
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False


class PreprocessingProfile:
    """Predefined preprocessing profiles for different use cases"""
    
    @staticmethod
    def get_profile(profile_name: str) -> Dict:
        """Get preprocessing configuration for specific profile"""
        profiles = {
            'high_quality': {
                'sample_rate': 16000,
                'enable_noise_reduction': True,
                'enable_vad': True,
                'enable_filtering': True,
                'enable_normalization': True,
                'enable_feature_extraction': True,
                'vad_aggressiveness': 2,
                'noise_reduction_strength': 0.8,
                'filter_type': 'butterworth',
                'filter_order': 5,
                'lowpass_freq': 8000,
                'highpass_freq': 80
            },
            'fast_processing': {
                'sample_rate': 16000,
                'enable_noise_reduction': False,
                'enable_vad': True,
                'enable_filtering': True,
                'enable_normalization': True,
                'enable_feature_extraction': False,
                'vad_aggressiveness': 1,
                'filter_type': 'butterworth',
                'filter_order': 3,
                'lowpass_freq': 8000,
                'highpass_freq': 100
            },
            'noise_robust': {
                'sample_rate': 16000,
                'enable_noise_reduction': True,
                'enable_vad': True,
                'enable_filtering': True,
                'enable_normalization': True,
                'enable_feature_extraction': True,
                'vad_aggressiveness': 3,
                'noise_reduction_strength': 1.0,
                'filter_type': 'butterworth',
                'filter_order': 6,
                'lowpass_freq': 7500,
                'highpass_freq': 85
            },
            'minimal': {
                'sample_rate': 16000,
                'enable_noise_reduction': False,
                'enable_vad': False,
                'enable_filtering': False,
                'enable_normalization': True,
                'enable_feature_extraction': False
            }
        }
        
        return profiles.get(profile_name, profiles['fast_processing'])


class AdvancedAudioPreprocessor:
    """Enhanced audio preprocessing with advanced features"""
    
    def __init__(self, profile: str = 'high_quality', custom_config: Optional[Dict] = None):
        """
        Initialize enhanced audio preprocessor
        
        Args:
            profile: Preprocessing profile name
            custom_config: Custom configuration to override profile settings
        """
        # Load profile configuration
        self.config = PreprocessingProfile.get_profile(profile)
        
        # Override with custom config if provided
        if custom_config:
            self.config.update(custom_config)
        
        self.sample_rate = self.config['sample_rate']
        
        # Initialize components based on configuration
        self._init_vad()
        self._init_filters()
        
        # Feature extraction parameters
        self.mfcc_params = {
            'n_mfcc': 13,
            'n_fft': 512,
            'hop_length': 160,
            'n_mels': 40
        }
        
        # Processing statistics
        self.stats = {
            'total_chunks': 0,
            'voice_chunks': 0,
            'avg_energy': 0.0,
            'avg_snr': 0.0
        }
        
    def _init_vad(self):
        """Initialize Voice Activity Detection"""
        if self.config.get('enable_vad', False) and WEBRTC_AVAILABLE:
            aggressiveness = self.config.get('vad_aggressiveness', 2)
            self.vad = webrtcvad.Vad(aggressiveness)
        else:
            self.vad = None
            
    def _init_filters(self):
        """Initialize audio filters"""
        if not self.config.get('enable_filtering', False):
            self.filters = None
            return
            
        filter_type = self.config.get('filter_type', 'butterworth')
        filter_order = self.config.get('filter_order', 5)
        lowpass_freq = self.config.get('lowpass_freq', 8000)
        highpass_freq = self.config.get('highpass_freq', 80)
        
        # Design filters
        nyquist = self.sample_rate / 2
        
        if filter_type == 'butterworth':
            # Ensure frequencies are within valid range (0 < freq < nyquist)
            highpass_norm = max(0.001, min(0.99, highpass_freq / nyquist))
            lowpass_norm = max(0.001, min(0.99, lowpass_freq / nyquist))
            
            # High-pass filter (remove low-frequency noise)
            self.highpass_sos = scipy.signal.butter(
                filter_order, highpass_norm, 
                btype='high', output='sos'
            )
            
            # Low-pass filter (anti-aliasing)
            self.lowpass_sos = scipy.signal.butter(
                filter_order, lowpass_norm, 
                btype='low', output='sos'
            )
        
        self.filters = {
            'highpass': self.highpass_sos,
            'lowpass': self.lowpass_sos
        }
    
    def resample_audio(self, audio: np.ndarray, original_sr: int) -> np.ndarray:
        """Resample audio to target sample rate with high quality"""
        if original_sr == self.sample_rate:
            return audio
            
        # Use high-quality resampling
        return librosa.resample(
            audio, 
            orig_sr=original_sr, 
            target_sr=self.sample_rate,
            res_type='kaiser_best'
        )
    
    def apply_filtering(self, audio: np.ndarray) -> np.ndarray:
        """Apply bandpass filtering to remove noise"""
        if not self.config.get('enable_filtering', False) or self.filters is None:
            return audio
            
        # Apply high-pass filter
        filtered = scipy.signal.sosfilt(self.filters['highpass'], audio)
        
        # Apply low-pass filter
        filtered = scipy.signal.sosfilt(self.filters['lowpass'], filtered)
        
        return filtered
    
    def normalize_audio(self, audio: np.ndarray, method: str = 'peak') -> np.ndarray:
        """Advanced audio normalization"""
        if not self.config.get('enable_normalization', True):
            return audio
            
        if method == 'peak':
            # Peak normalization
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                return audio / max_val
                
        elif method == 'rms':
            # RMS normalization
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                target_rms = 0.1  # Target RMS level
                return audio * (target_rms / rms)
                
        elif method == 'lufs':
            # Loudness normalization (simplified)
            # This is a basic implementation
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                target_lufs = -23  # EBU R128 standard
                current_lufs = 20 * np.log10(rms) - 0.691
                gain_db = target_lufs - current_lufs
                gain_linear = 10 ** (gain_db / 20)
                return audio * gain_linear
                
        return audio
    
    def apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Advanced noise reduction"""
        if not self.config.get('enable_noise_reduction', False) or not NOISEREDUCE_AVAILABLE:
            return audio
            
        try:
            strength = self.config.get('noise_reduction_strength', 0.8)
            
            # Apply noise reduction with configurable strength
            reduced_audio = nr.reduce_noise(
                y=audio, 
                sr=self.sample_rate,
                stationary=True,
                prop_decrease=strength
            )
            
            return reduced_audio
            
        except Exception as e:
            print(f"Noise reduction failed: {e}")
            return audio
    
    def detect_voice_activity(self, audio: np.ndarray) -> Tuple[bool, float]:
        """Enhanced voice activity detection with confidence score"""
        if not self.config.get('enable_vad', False):
            # Energy-based VAD with confidence
            energy = np.mean(audio ** 2)
            threshold = 0.001
            confidence = min(energy / threshold, 1.0) if energy > threshold else 0.0
            return energy > threshold, confidence
            
        if self.vad is None:
            return False, 0.0
            
        # Convert to 16-bit PCM for webrtcvad
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        
        # webrtcvad requires specific frame sizes
        frame_duration = 30  # ms
        frame_size = int(self.sample_rate * frame_duration / 1000)
        
        if len(audio_int16) < frame_size:
            return False, 0.0
            
        # Process multiple frames for better accuracy
        num_frames = len(audio_int16) // frame_size
        voice_frames = 0
        
        for i in range(min(num_frames, 5)):  # Check up to 5 frames
            start_idx = i * frame_size
            end_idx = start_idx + frame_size
            frame = audio_int16[start_idx:end_idx].tobytes()
            
            try:
                if self.vad.is_speech(frame, self.sample_rate):
                    voice_frames += 1
            except Exception:
                continue
                
        # Calculate confidence based on voice frame ratio
        confidence = voice_frames / min(num_frames, 5) if num_frames > 0 else 0.0
        has_voice = confidence > 0.3  # Threshold for voice detection
        
        return has_voice, confidence
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract audio features for ASR models"""
        if not self.config.get('enable_feature_extraction', False):
            return {}
            
        features = {}
        
        try:
            # MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.mfcc_params['n_mfcc'],
                n_fft=self.mfcc_params['n_fft'],
                hop_length=self.mfcc_params['hop_length'],
                n_mels=self.mfcc_params['n_mels']
            )
            features['mfcc'] = mfccs
            
            # Mel-scale spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.mfcc_params['n_fft'],
                hop_length=self.mfcc_params['hop_length'],
                n_mels=self.mfcc_params['n_mels']
            )
            features['mel_spectrogram'] = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate
            )[0]
            features['spectral_centroid'] = spectral_centroids
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zero_crossing_rate'] = zcr
            
            # RMS energy
            rms = librosa.feature.rms(y=audio)[0]
            features['rms_energy'] = rms
            
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            
        return features
    
    def calculate_snr(self, audio: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio"""
        try:
            # Simple SNR estimation using energy distribution
            # This is a basic implementation
            energy = audio ** 2
            
            # Assume noise is in the lower 10% of energy values
            sorted_energy = np.sort(energy)
            noise_threshold = int(0.1 * len(sorted_energy))
            
            if noise_threshold > 0:
                noise_power = np.mean(sorted_energy[:noise_threshold])
                signal_power = np.mean(sorted_energy[noise_threshold:])
                
                if noise_power > 0:
                    snr_db = 10 * np.log10(signal_power / noise_power)
                    return max(snr_db, 0.0)  # Clip negative SNR
                    
        except Exception:
            pass
            
        return 0.0
    
    def update_statistics(self, audio: np.ndarray, has_voice: bool):
        """Update processing statistics"""
        self.stats['total_chunks'] += 1
        
        if has_voice:
            self.stats['voice_chunks'] += 1
            
        # Update running averages
        energy = np.mean(audio ** 2)
        snr = self.calculate_snr(audio)
        
        alpha = 0.1  # Smoothing factor
        self.stats['avg_energy'] = (1 - alpha) * self.stats['avg_energy'] + alpha * energy
        self.stats['avg_snr'] = (1 - alpha) * self.stats['avg_snr'] + alpha * snr
    
    def preprocess(self, audio: np.ndarray, original_sr: int = None) -> Dict:
        """
        Complete enhanced preprocessing pipeline
        
        Returns:
            Dictionary containing processed audio and metadata
        """
        if original_sr is None:
            original_sr = self.sample_rate
            
        result = {
            'audio': audio.copy(),
            'sample_rate': self.sample_rate,
            'has_voice': False,
            'voice_confidence': 0.0,
            'features': {},
            'snr_db': 0.0,
            'energy': 0.0,
            'processing_steps': []
        }
        
        # Step 1: Resampling
        if original_sr != self.sample_rate:
            result['audio'] = self.resample_audio(result['audio'], original_sr)
            result['processing_steps'].append('resampling')
            
        # Step 2: Filtering
        if self.config.get('enable_filtering', False):
            result['audio'] = self.apply_filtering(result['audio'])
            result['processing_steps'].append('filtering')
            
        # Step 3: Noise reduction
        if self.config.get('enable_noise_reduction', False):
            result['audio'] = self.apply_noise_reduction(result['audio'])
            result['processing_steps'].append('noise_reduction')
            
        # Step 4: Normalization
        if self.config.get('enable_normalization', True):
            result['audio'] = self.normalize_audio(result['audio'])
            result['processing_steps'].append('normalization')
            
        # Step 5: Voice activity detection
        has_voice, confidence = self.detect_voice_activity(result['audio'])
        result['has_voice'] = has_voice
        result['voice_confidence'] = confidence
        
        # Step 6: Feature extraction
        if self.config.get('enable_feature_extraction', False):
            result['features'] = self.extract_features(result['audio'])
            result['processing_steps'].append('feature_extraction')
            
        # Step 7: Calculate metrics
        result['snr_db'] = self.calculate_snr(result['audio'])
        result['energy'] = np.mean(result['audio'] ** 2)
        
        # Update statistics
        self.update_statistics(result['audio'], has_voice)
        
        return result
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        stats = self.stats.copy()
        if stats['total_chunks'] > 0:
            stats['voice_activity_ratio'] = stats['voice_chunks'] / stats['total_chunks']
        else:
            stats['voice_activity_ratio'] = 0.0
            
        return stats
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.stats = {
            'total_chunks': 0,
            'voice_chunks': 0,
            'avg_energy': 0.0,
            'avg_snr': 0.0
        }


if __name__ == "__main__":
    # Test the enhanced preprocessing pipeline
    print("Testing Enhanced Audio Preprocessing Pipeline...")
    
    # Test different profiles
    profiles = ['high_quality', 'fast_processing', 'noise_robust', 'minimal']
    
    for profile_name in profiles:
        print(f"\n🔧 Testing {profile_name} profile:")
        
        preprocessor = AdvancedAudioPreprocessor(profile=profile_name)
        
        # Generate test audio (sine wave + noise)
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create test signal: 440Hz tone + noise
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        noise = 0.1 * np.random.randn(len(signal))
        test_audio = signal + noise
        
        # Process audio
        result = preprocessor.preprocess(test_audio, sample_rate)
        
        print(f"   • Processing steps: {', '.join(result['processing_steps'])}")
        print(f"   • Voice detected: {result['has_voice']} (confidence: {result['voice_confidence']:.3f})")
        print(f"   • SNR: {result['snr_db']:.2f} dB")
        print(f"   • Energy: {result['energy']:.6f}")
        print(f"   • Features extracted: {len(result['features'])} types")
        
        # Get statistics
        stats = preprocessor.get_statistics()
        print(f"   • Statistics: {stats['total_chunks']} chunks, {stats['voice_activity_ratio']:.2f} voice ratio")
    
    print("\n✅ Enhanced preprocessing pipeline test completed!")