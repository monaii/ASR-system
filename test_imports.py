#!/usr/bin/env python3
"""
Test script to verify all required dependencies are properly installed
and can be imported without errors.
"""

import sys
import traceback

def test_import(module_name, description=""):
    """Test importing a module and report results."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {description}: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name} - {description}: Unexpected error: {e}")
        return False

def main():
    """Test all required imports for the ASR system."""
    print("Testing ASR System Dependencies")
    print("=" * 50)
    
    # Core ML libraries
    success_count = 0
    total_tests = 0
    
    tests = [
        ("torch", "PyTorch for deep learning"),
        ("torchaudio", "PyTorch audio processing"),
        ("transformers", "Hugging Face Transformers"),
        ("datasets", "Hugging Face Datasets"),
        ("sounddevice", "Real-time audio I/O"),
        ("soundfile", "Audio file I/O"),
        ("pyaudio", "Audio capture and playback"),
        ("librosa", "Audio analysis"),
        ("noisereduce", "Noise reduction"),
        ("webrtcvad", "Voice Activity Detection"),
        ("numpy", "Numerical computing"),
        ("scipy", "Scientific computing"),
        ("matplotlib", "Plotting"),
        ("gradio", "Web UI framework"),
        ("streamlit", "Web app framework"),
        ("evaluate", "Model evaluation metrics"),
        ("jiwer", "Word Error Rate calculation"),
    ]
    
    for module_name, description in tests:
        total_tests += 1
        if test_import(module_name, description):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Import Test Results: {success_count}/{total_tests} successful")
    
    if success_count == total_tests:
        print("🎉 All dependencies imported successfully!")
        
        # Test device availability
        print("\nTesting device availability:")
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("✅ MPS (Apple Silicon) available")
            else:
                print("ℹ️  CPU-only mode (no GPU acceleration)")
                
            print(f"PyTorch version: {torch.__version__}")
        except Exception as e:
            print(f"⚠️  Error checking device: {e}")
            
        # Test audio device availability
        print("\nTesting audio devices:")
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            print(f"✅ Found {len(input_devices)} input audio devices")
            if input_devices:
                default_input = sd.default.device[0]
                if default_input is not None:
                    print(f"Default input device: {devices[default_input]['name']}")
        except Exception as e:
            print(f"⚠️  Error checking audio devices: {e}")
            
        return True
    else:
        print("❌ Some dependencies failed to import. Please check the installation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)