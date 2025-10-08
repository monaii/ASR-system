"""
Test Enhanced Audio Preprocessing Pipeline
Comprehensive testing of advanced preprocessing features
"""

import numpy as np
import time
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from enhanced_preprocessing import AdvancedAudioPreprocessor, PreprocessingProfile
from audio_processing import AudioCapture, AudioPreprocessor
from models import get_model


def test_preprocessing_profiles():
    """Test different preprocessing profiles"""
    print("🔧 Testing Preprocessing Profiles...")
    
    # Generate test audio
    duration = 3.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create complex test signal
    signal = (0.3 * np.sin(2 * np.pi * 440 * t) +  # 440Hz tone
              0.2 * np.sin(2 * np.pi * 880 * t) +  # 880Hz harmonic
              0.1 * np.sin(2 * np.pi * 220 * t))   # 220Hz sub-harmonic
    
    # Add realistic noise
    noise = 0.05 * np.random.randn(len(signal))
    test_audio = signal + noise
    
    profiles = ['high_quality', 'fast_processing', 'noise_robust', 'minimal']
    
    for profile_name in profiles:
        print(f"\n   📋 Profile: {profile_name}")
        
        preprocessor = AdvancedAudioPreprocessor(profile=profile_name)
        
        start_time = time.time()
        result = preprocessor.preprocess(test_audio, sample_rate)
        processing_time = time.time() - start_time
        
        print(f"      • Processing time: {processing_time:.3f}s")
        print(f"      • Steps: {', '.join(result['processing_steps'])}")
        print(f"      • Voice: {result['has_voice']} (conf: {result['voice_confidence']:.3f})")
        print(f"      • SNR: {result['snr_db']:.2f} dB")
        print(f"      • Energy: {result['energy']:.6f}")
        print(f"      • Features: {len(result['features'])} types")
        
        if result['features']:
            for feature_name, feature_data in result['features'].items():
                print(f"         - {feature_name}: {feature_data.shape}")
    
    print("✅ Profile testing completed!")
    return True


def test_feature_extraction():
    """Test feature extraction capabilities"""
    print("\n🎵 Testing Feature Extraction...")
    
    # Create test audio with speech-like characteristics
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simulate formant frequencies (speech-like)
    formant1 = 0.2 * np.sin(2 * np.pi * 800 * t)   # F1
    formant2 = 0.15 * np.sin(2 * np.pi * 1200 * t) # F2
    formant3 = 0.1 * np.sin(2 * np.pi * 2400 * t)  # F3
    
    # Add envelope modulation (speech rhythm)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))
    speech_like = (formant1 + formant2 + formant3) * envelope
    
    # Add noise
    noise = 0.02 * np.random.randn(len(speech_like))
    test_audio = speech_like + noise
    
    preprocessor = AdvancedAudioPreprocessor(profile='high_quality')
    result = preprocessor.preprocess(test_audio, sample_rate)
    
    print(f"   • Audio length: {len(test_audio)} samples ({duration}s)")
    print(f"   • Voice detected: {result['has_voice']}")
    print(f"   • Voice confidence: {result['voice_confidence']:.3f}")
    
    if result['features']:
        print("   • Extracted features:")
        for feature_name, feature_data in result['features'].items():
            if len(feature_data.shape) == 1:
                print(f"      - {feature_name}: shape {feature_data.shape}, "
                      f"mean={np.mean(feature_data):.3f}, std={np.std(feature_data):.3f}")
            else:
                print(f"      - {feature_name}: shape {feature_data.shape}, "
                      f"mean={np.mean(feature_data):.3f}")
    
    print("✅ Feature extraction test completed!")
    return True


def test_performance_comparison():
    """Compare performance between basic and enhanced preprocessing"""
    print("\n⚡ Testing Performance Comparison...")
    
    # Generate test data
    sample_rate = 16000
    duration = 5.0
    num_chunks = 10
    chunk_duration = duration / num_chunks
    
    # Create test chunks
    test_chunks = []
    for i in range(num_chunks):
        t = np.linspace(0, chunk_duration, int(sample_rate * chunk_duration))
        
        # Vary signal characteristics
        freq = 440 + i * 50  # Varying frequency
        amplitude = 0.3 + 0.2 * np.sin(i)  # Varying amplitude
        
        signal = amplitude * np.sin(2 * np.pi * freq * t)
        noise = 0.05 * np.random.randn(len(signal))
        
        test_chunks.append(signal + noise)
    
    # Test basic preprocessor
    print("   📊 Basic AudioPreprocessor:")
    basic_preprocessor = AudioPreprocessor()
    
    basic_times = []
    basic_results = []
    
    for chunk in test_chunks:
        start_time = time.time()
        processed_audio, has_voice = basic_preprocessor.preprocess(chunk, sample_rate)
        processing_time = time.time() - start_time
        
        basic_times.append(processing_time)
        basic_results.append((processed_audio, has_voice))
    
    basic_avg_time = np.mean(basic_times)
    basic_voice_ratio = sum(1 for _, has_voice in basic_results if has_voice) / len(basic_results)
    
    print(f"      • Average processing time: {basic_avg_time:.4f}s")
    print(f"      • Voice detection ratio: {basic_voice_ratio:.2f}")
    
    # Test enhanced preprocessor (fast profile)
    print("   🚀 Enhanced AudioPreprocessor (fast_processing):")
    enhanced_preprocessor = AdvancedAudioPreprocessor(profile='fast_processing')
    
    enhanced_times = []
    enhanced_results = []
    
    for chunk in test_chunks:
        start_time = time.time()
        result = enhanced_preprocessor.preprocess(chunk, sample_rate)
        processing_time = time.time() - start_time
        
        enhanced_times.append(processing_time)
        enhanced_results.append(result)
    
    enhanced_avg_time = np.mean(enhanced_times)
    enhanced_voice_ratio = sum(1 for result in enhanced_results if result['has_voice']) / len(enhanced_results)
    enhanced_avg_confidence = np.mean([result['voice_confidence'] for result in enhanced_results])
    enhanced_avg_snr = np.mean([result['snr_db'] for result in enhanced_results])
    
    print(f"      • Average processing time: {enhanced_avg_time:.4f}s")
    print(f"      • Voice detection ratio: {enhanced_voice_ratio:.2f}")
    print(f"      • Average confidence: {enhanced_avg_confidence:.3f}")
    print(f"      • Average SNR: {enhanced_avg_snr:.2f} dB")
    
    # Test enhanced preprocessor (high quality profile)
    print("   🎯 Enhanced AudioPreprocessor (high_quality):")
    hq_preprocessor = AdvancedAudioPreprocessor(profile='high_quality')
    
    hq_times = []
    hq_results = []
    
    for chunk in test_chunks:
        start_time = time.time()
        result = hq_preprocessor.preprocess(chunk, sample_rate)
        processing_time = time.time() - start_time
        
        hq_times.append(processing_time)
        hq_results.append(result)
    
    hq_avg_time = np.mean(hq_times)
    hq_voice_ratio = sum(1 for result in hq_results if result['has_voice']) / len(hq_results)
    hq_avg_confidence = np.mean([result['voice_confidence'] for result in hq_results])
    hq_avg_snr = np.mean([result['snr_db'] for result in hq_results])
    hq_feature_count = sum(len(result['features']) for result in hq_results) / len(hq_results)
    
    print(f"      • Average processing time: {hq_avg_time:.4f}s")
    print(f"      • Voice detection ratio: {hq_voice_ratio:.2f}")
    print(f"      • Average confidence: {hq_avg_confidence:.3f}")
    print(f"      • Average SNR: {hq_avg_snr:.2f} dB")
    print(f"      • Average features per chunk: {hq_feature_count:.1f}")
    
    # Performance summary
    print("\n   📈 Performance Summary:")
    print(f"      • Speed comparison (vs basic):")
    print(f"         - Fast processing: {enhanced_avg_time/basic_avg_time:.2f}x")
    print(f"         - High quality: {hq_avg_time/basic_avg_time:.2f}x")
    print(f"      • Quality improvements:")
    print(f"         - Enhanced confidence scoring")
    print(f"         - SNR estimation")
    print(f"         - Feature extraction (high quality)")
    
    print("✅ Performance comparison completed!")
    return True


def test_integration_with_asr():
    """Test integration with ASR models"""
    print("\n🤖 Testing ASR Integration...")
    
    try:
        # Load ASR model
        print("   Loading Whisper model...")
        model = get_model('whisper', model_size='tiny')
        model.load_model()
        print("   ✅ Model loaded successfully!")
        
        # Create test audio with speech content
        sample_rate = 16000
        duration = 3.0
        
        # Generate synthetic speech-like audio
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Multiple frequency components (speech-like)
        speech_signal = (
            0.3 * np.sin(2 * np.pi * 200 * t) +   # Fundamental
            0.2 * np.sin(2 * np.pi * 400 * t) +   # First harmonic
            0.15 * np.sin(2 * np.pi * 600 * t) +  # Second harmonic
            0.1 * np.sin(2 * np.pi * 800 * t)     # Third harmonic
        )
        
        # Add speech-like modulation
        modulation = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))  # 3Hz modulation
        speech_signal *= modulation
        
        # Add noise
        noise = 0.03 * np.random.randn(len(speech_signal))
        test_audio = speech_signal + noise
        
        # Test with different preprocessing profiles
        profiles = ['minimal', 'fast_processing', 'high_quality']
        
        for profile_name in profiles:
            print(f"\n   🔧 Testing with {profile_name} profile:")
            
            preprocessor = AdvancedAudioPreprocessor(profile=profile_name)
            
            # Preprocess audio
            start_time = time.time()
            result = preprocessor.preprocess(test_audio, sample_rate)
            preprocessing_time = time.time() - start_time
            
            # Transcribe with ASR model
            if result['has_voice']:
                transcription_start = time.time()
                transcription_result = model.transcribe(result['audio'])
                transcription_time = time.time() - transcription_start
                
                # Handle the tuple return from Whisper model
                if isinstance(transcription_result, tuple):
                    transcription_text, inference_time = transcription_result
                else:
                    transcription_text = transcription_result
                    inference_time = transcription_time
                
                print(f"      • Preprocessing: {preprocessing_time:.3f}s")
                print(f"      • Transcription: {transcription_time:.3f}s")
                print(f"      • Total time: {preprocessing_time + transcription_time:.3f}s")
                print(f"      • Voice confidence: {result['voice_confidence']:.3f}")
                print(f"      • SNR: {result['snr_db']:.2f} dB")
                print(f"      • Transcription: '{transcription_text.strip()}'")
                
                if result['features']:
                    print(f"      • Features available: {list(result['features'].keys())}")
            else:
                print(f"      • No voice detected (confidence: {result['voice_confidence']:.3f})")
        
        print("✅ ASR integration test completed!")
        return True
        
    except Exception as e:
        print(f"   ❌ ASR integration test failed: {e}")
        return False


def test_custom_configuration():
    """Test custom preprocessing configuration"""
    print("\n⚙️ Testing Custom Configuration...")
    
    # Create custom configuration
    custom_config = {
        'sample_rate': 16000,
        'enable_noise_reduction': True,
        'enable_vad': True,
        'enable_filtering': True,
        'enable_normalization': True,
        'enable_feature_extraction': True,
        'vad_aggressiveness': 3,  # Maximum aggressiveness
        'noise_reduction_strength': 0.9,  # Strong noise reduction
        'filter_order': 8,  # Higher order filters
        'lowpass_freq': 7000,  # Lower cutoff
        'highpass_freq': 100   # Higher cutoff
    }
    
    print("   Custom configuration:")
    for key, value in custom_config.items():
        print(f"      • {key}: {value}")
    
    # Test with custom configuration
    preprocessor = AdvancedAudioPreprocessor(
        profile='high_quality', 
        custom_config=custom_config
    )
    
    # Generate noisy test audio
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Signal with noise
    signal = 0.4 * np.sin(2 * np.pi * 440 * t)
    noise = 0.15 * np.random.randn(len(signal))  # Higher noise level
    test_audio = signal + noise
    
    result = preprocessor.preprocess(test_audio, sample_rate)
    
    print(f"\n   Results with custom configuration:")
    print(f"      • Voice detected: {result['has_voice']}")
    print(f"      • Voice confidence: {result['voice_confidence']:.3f}")
    print(f"      • SNR improvement: {result['snr_db']:.2f} dB")
    print(f"      • Processing steps: {', '.join(result['processing_steps'])}")
    
    # Get statistics
    stats = preprocessor.get_statistics()
    print(f"      • Statistics: {stats}")
    
    print("✅ Custom configuration test completed!")
    return True


def main():
    """Run all enhanced preprocessing tests"""
    print("🚀 Enhanced Audio Preprocessing Pipeline Tests")
    print("=" * 60)
    
    tests = [
        ("Preprocessing Profiles", test_preprocessing_profiles),
        ("Feature Extraction", test_feature_extraction),
        ("Performance Comparison", test_performance_comparison),
        ("ASR Integration", test_integration_with_asr),
        ("Custom Configuration", test_custom_configuration)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with error: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All enhanced preprocessing tests completed successfully!")
        print("🚀 Step 6: Enhanced audio preprocessing pipeline is ready!")
        return True
    else:
        print(f"⚠️  {total_tests - passed_tests} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)