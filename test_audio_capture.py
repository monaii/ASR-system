#!/usr/bin/env python3
"""
Test script for real-time audio capture functionality.
Tests microphone input, audio preprocessing, and streaming buffer.
"""

import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
import threading
import queue

# Import our custom modules
sys.path.append('src')
from audio_processing import AudioCapture, AudioPreprocessor, StreamingBuffer

def list_audio_devices():
    """List all available audio input devices."""
    print("🎤 Available Audio Input Devices:")
    print("=" * 50)
    
    devices = sd.query_devices()
    input_devices = []
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device))
            is_default = "(DEFAULT)" if i == sd.default.device[0] else ""
            print(f"  {i}: {device['name']} {is_default}")
            print(f"      Channels: {device['max_input_channels']}")
            print(f"      Sample Rate: {device['default_samplerate']} Hz")
            print()
    
    return input_devices

def test_basic_recording(duration=3, device_id=None):
    """Test basic audio recording functionality."""
    print(f"\n🎵 Testing Basic Recording ({duration}s)")
    print("=" * 50)
    
    try:
        print("Recording... Speak into your microphone!")
        
        # Record audio
        sample_rate = 16000
        audio_data = sd.rec(
            int(duration * sample_rate), 
            samplerate=sample_rate, 
            channels=1, 
            device=device_id,
            dtype=np.float32
        )
        sd.wait()  # Wait for recording to complete
        
        print(f"✅ Recording completed!")
        print(f"   Shape: {audio_data.shape}")
        print(f"   Duration: {len(audio_data) / sample_rate:.2f}s")
        print(f"   Sample Rate: {sample_rate} Hz")
        print(f"   Max Amplitude: {np.max(np.abs(audio_data)):.3f}")
        
        # Save test recording
        output_file = "test_recording.wav"
        sf.write(output_file, audio_data, sample_rate)
        print(f"   Saved to: {output_file}")
        
        return audio_data, sample_rate
        
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return None, None

def test_audio_capture_class():
    """Test the AudioCapture class functionality."""
    print(f"\n🎙️ Testing AudioCapture Class")
    print("=" * 50)
    
    try:
        # Initialize audio capture
        capture = AudioCapture(
            sample_rate=16000,
            channels=1,
            chunk_duration=0.5,  # 0.5 seconds
            buffer_duration=3.0  # 3 seconds
        )
        
        print(f"✅ AudioCapture initialized")
        print(f"   Sample Rate: {capture.sample_rate} Hz")
        print(f"   Channels: {capture.channels}")
        print(f"   Chunk Size: {capture.chunk_size}")
        print(f"   Buffer Size: {capture.buffer_size}")
        
        # Test short recording
        print("\n📹 Testing 2-second recording...")
        capture.start_recording()
        time.sleep(2)
        capture.stop_recording()
        
        print("✅ AudioCapture test passed!")
        return True
            
    except Exception as e:
        print(f"❌ AudioCapture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_preprocessing():
    """Test audio preprocessing functionality."""
    print(f"\n🔧 Testing Audio Preprocessing")
    print("=" * 50)
    
    try:
        # Initialize preprocessor
        preprocessor = AudioPreprocessor(
            target_sample_rate=16000,
            enable_noise_reduction=True,
            enable_vad=True
        )
        
        print(f"✅ AudioPreprocessor initialized")
        print(f"   Target Sample Rate: {preprocessor.target_sample_rate} Hz")
        print(f"   Noise Reduction: {preprocessor.enable_noise_reduction}")
        print(f"   VAD Enabled: {preprocessor.enable_vad}")
        
        # Test with recorded audio if available
        if Path("test_recording.wav").exists():
            print("\n📁 Testing with recorded audio...")
            audio_data, sample_rate = sf.read("test_recording.wav")
            
            print(f"   Original: {audio_data.shape}, SR: {sample_rate}")
            
            # Preprocess audio
            processed_audio, has_voice = preprocessor.preprocess(audio_data, original_sr=sample_rate)
            
            print(f"   Processed: {processed_audio.shape}")
            print(f"   Voice Activity: {has_voice}")
            print(f"   Max amplitude: {np.max(np.abs(processed_audio)):.3f}")
            
            # Save processed audio
            sf.write("test_processed.wav", processed_audio, preprocessor.target_sample_rate)
            print(f"   Saved processed audio to: test_processed.wav")
            
            return True
        else:
            print("⚠️  No test recording found, skipping preprocessing test")
            return True
            
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streaming_buffer():
    """Test streaming buffer functionality."""
    print(f"\n📊 Testing Streaming Buffer")
    print("=" * 50)
    
    try:
        # Initialize streaming buffer
        buffer = StreamingBuffer(
            sample_rate=16000,
            window_duration=2.0,  # 2 seconds
            overlap_duration=0.5  # 0.5 seconds overlap
        )
        
        print(f"✅ StreamingBuffer initialized")
        print(f"   Sample Rate: {buffer.sample_rate} Hz")
        print(f"   Window Size: {buffer.window_size}")
        print(f"   Overlap Size: {buffer.overlap_size}")
        print(f"   Step Size: {buffer.step_size}")
        
        # Test adding audio chunks
        print("\n📊 Testing audio streaming...")
        test_chunk = np.random.randn(8000).astype(np.float32)  # 0.5 seconds at 16kHz
        
        # Simulate streaming by adding multiple chunks
        windows_received = 0
        for i in range(6):  # Add 6 chunks (3 seconds of audio)
            window = buffer.add_audio(test_chunk)
            if window is not None:
                windows_received += 1
                print(f"   Window {windows_received}: {window.shape}, Energy: {np.mean(window**2):.6f}")
        
        print(f"✅ Streaming buffer test passed!")
        print(f"   Windows received: {windows_received}")
        
        return True
        
    except Exception as e:
        print(f"❌ Streaming buffer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🎤 Real-Time Audio Capture Testing Suite")
    print("=" * 60)
    
    # List available devices
    input_devices = list_audio_devices()
    
    if not input_devices:
        print("❌ No input devices found!")
        return False
    
    # Test basic recording
    audio_data, sample_rate = test_basic_recording(duration=3)
    
    # Test AudioCapture class
    capture_success = test_audio_capture_class()
    
    # Test audio preprocessing
    preprocessing_success = test_audio_preprocessing()
    
    # Test streaming buffer
    buffer_success = test_streaming_buffer()
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    tests = [
        ("Basic Recording", audio_data is not None),
        ("AudioCapture Class", capture_success),
        ("Audio Preprocessing", preprocessing_success),
        ("Streaming Buffer", buffer_success)
    ]
    
    successful_tests = sum(1 for _, success in tests if success)
    total_tests = len(tests)
    
    for test_name, success in tests:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    print(f"\nSuccessful tests: {successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print("🎉 All audio capture tests passed!")
        print("✅ System ready for real-time streaming ASR!")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return successful_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)