#!/usr/bin/env python3
"""
Test script to load and verify ASR models functionality.
Tests both Whisper and Wav2Vec2 models with sample audio.
"""

import sys
import time
import torch
import torchaudio
import numpy as np
from pathlib import Path

# Import our custom modules
sys.path.append('src')
from models import get_model

def generate_test_audio(duration=3, sample_rate=16000):
    """Generate a simple test audio signal (sine wave)."""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    # Generate a mix of frequencies to simulate speech-like signal
    audio = (torch.sin(2 * np.pi * 440 * t) * 0.3 +  # A4 note
             torch.sin(2 * np.pi * 880 * t) * 0.2 +  # A5 note
             torch.sin(2 * np.pi * 220 * t) * 0.1)   # A3 note
    
    # Add some noise to make it more realistic
    noise = torch.randn_like(audio) * 0.05
    audio = audio + noise
    
    return audio.unsqueeze(0)  # Add batch dimension

def test_model(model_config):
    """Test loading and inference for a specific model."""
    model_type = model_config["type"]
    model_name = model_config.get("name", "default")
    
    print(f"\n{'='*60}")
    print(f"Testing {model_name} ({model_type})")
    print(f"{'='*60}")
    
    try:
        # Load model
        print("📥 Loading model...")
        start_time = time.time()
        if model_type == "whisper":
            model = get_model("whisper", model_size=model_config.get("size", "base"))
        elif model_type == "wav2vec2":
            model = get_model("wav2vec2", model_name=model_config.get("model_name", "facebook/wav2vec2-base-960h"))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model.load_model()  # Load the model
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
        print(f"   Device: {model.device}")
        
        # Generate test audio
        print("🎵 Generating test audio...")
        test_audio = generate_test_audio(duration=2)
        print(f"   Audio shape: {test_audio.shape}")
        print(f"   Sample rate: 16000 Hz")
        
        # Test transcription
        print("🎤 Testing transcription...")
        start_time = time.time()
        # Convert torch tensor to numpy for model input
        test_audio_np = test_audio.squeeze(0).numpy()
        result = model.transcribe(test_audio_np)
        inference_time = time.time() - start_time
        
        print(f"✅ Transcription completed in {inference_time:.2f}s")
        print(f"   Result: '{result}'")
        
        # Test with different audio lengths
        print("📊 Testing different audio lengths...")
        for duration in [1, 3, 5]:
            audio = generate_test_audio(duration=duration)
            audio_np = audio.squeeze(0).numpy()
            start_time = time.time()
            result = model.transcribe(audio_np)
            elapsed = time.time() - start_time
            print(f"   {duration}s audio -> {elapsed:.2f}s inference")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_compatibility():
    """Test device compatibility and performance."""
    print(f"\n{'='*60}")
    print("Device Compatibility Test")
    print(f"{'='*60}")
    
    # Check available devices
    print("Available devices:")
    print(f"  CPU: Always available")
    
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.cuda.get_device_name()}")
        print(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"  MPS: Apple Silicon GPU acceleration")
    
    # Test memory usage
    if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("\n🧠 Memory usage test...")
        try:
            # Load a small model to test memory
            model = get_model("whisper", model_size="tiny")
            model.load_model()
            test_audio = generate_test_audio(duration=1)
            test_audio_np = test_audio.squeeze(0).numpy()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()
                result = model.transcribe(test_audio_np)
                memory_after = torch.cuda.memory_allocated()
                print(f"  CUDA memory used: {(memory_after - memory_before) / 1e6:.1f} MB")
            else:
                result = model.transcribe(test_audio_np)
                print(f"  MPS inference successful")
                
        except Exception as e:
            print(f"  ⚠️  Memory test failed: {e}")

def main():
    """Main test function."""
    print("🚀 ASR Model Testing Suite")
    print("=" * 60)
    
    # Test device compatibility first
    test_device_compatibility()
    
    # Define models to test
    models_to_test = [
        {"type": "whisper", "name": "Whisper Tiny", "size": "tiny"},
        {"type": "whisper", "name": "Whisper Base", "size": "base"},
        {"type": "wav2vec2", "name": "Wav2Vec2 Base", "model_name": "facebook/wav2vec2-base-960h"},
    ]
    
    successful_tests = 0
    total_tests = len(models_to_test)
    
    for model_config in models_to_test:
        if test_model(model_config):
            successful_tests += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Successful tests: {successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print("🎉 All models loaded and tested successfully!")
        print("✅ System is ready for real-time ASR!")
    else:
        print("⚠️  Some models failed to load. Check the errors above.")
        
    return successful_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)