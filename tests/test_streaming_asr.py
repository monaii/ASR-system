#!/usr/bin/env python3
"""
Comprehensive test for streaming ASR functionality
Tests real-time transcription with window-based processing
"""

import sys
import time
import threading
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models import get_model
from streaming_asr import StreamingASR, TranscriptionResult, MultiModelASR


def test_streaming_asr_basic():
    """Test basic streaming ASR functionality"""
    print("\n" + "="*60)
    print("🎤 TEST 1: Basic Streaming ASR")
    print("="*60)
    
    try:
        # Load Whisper Tiny model (fastest for testing)
        print("Loading Whisper Tiny model...")
        model = get_model("whisper", model_size="tiny")
        model.load_model()
        print("✅ Model loaded successfully")
        
        # Create streaming ASR
        streaming_asr = StreamingASR(
            model=model,
            chunk_duration=0.5,
            window_duration=2.0,
            overlap_duration=0.5,
            min_speech_duration=0.3,
            silence_timeout=1.5
        )
        
        # Set up result callback
        results = []
        def collect_result(result: TranscriptionResult):
            results.append(result)
            print(f"🔊 Transcription: '{result.text}' "
                  f"(latency: {result.inference_time:.3f}s, "
                  f"confidence: {result.confidence:.2f})")
        
        streaming_asr.set_result_callback(collect_result)
        
        # Test streaming for 8 seconds
        print("\n🎙️  Starting 8-second streaming test...")
        print("💬 Please speak into your microphone!")
        
        streaming_asr.start()
        time.sleep(8)
        streaming_asr.stop()
        
        # Check results
        metrics = streaming_asr.get_metrics()
        print(f"\n📊 Results:")
        print(f"   • Total chunks processed: {metrics['total_chunks']}")
        print(f"   • Voice chunks detected: {metrics['voice_chunks']}")
        print(f"   • Transcriptions generated: {metrics['transcriptions']}")
        print(f"   • Voice activity ratio: {metrics.get('voice_activity_ratio', 0):.2f}")
        print(f"   • Average latency: {metrics.get('average_latency', 0):.3f}s")
        
        if results:
            print(f"   • Sample transcriptions:")
            for i, result in enumerate(results[-3:], 1):
                print(f"     {i}. '{result.text}'")
        
        print("✅ Basic streaming ASR test completed")
        return True
        
    except Exception as e:
        print(f"❌ Basic streaming ASR test failed: {e}")
        return False


def test_streaming_asr_advanced():
    """Test advanced streaming ASR features"""
    print("\n" + "="*60)
    print("🔬 TEST 2: Advanced Streaming Features")
    print("="*60)
    
    try:
        # Load model
        print("Loading Whisper Tiny model...")
        model = get_model("whisper", model_size="tiny")
        model.load_model()
        
        # Create streaming ASR with custom parameters
        streaming_asr = StreamingASR(
            model=model,
            chunk_duration=0.3,  # Smaller chunks for responsiveness
            window_duration=1.5,  # Shorter windows
            overlap_duration=0.3,
            min_speech_duration=0.2,  # Lower threshold
            silence_timeout=1.0
        )
        
        # Advanced result tracking
        transcription_times = []
        voice_segments = []
        
        def advanced_callback(result: TranscriptionResult):
            transcription_times.append(result.inference_time)
            if result.has_voice_activity:
                voice_segments.append({
                    'timestamp': result.timestamp,
                    'text': result.text,
                    'latency': result.inference_time
                })
            print(f"🎯 [{result.timestamp:.1f}s] '{result.text}' "
                  f"(VAD: {result.has_voice_activity}, "
                  f"latency: {result.inference_time:.3f}s)")
        
        streaming_asr.set_result_callback(advanced_callback)
        
        # Test with shorter duration but more detailed analysis
        print("\n🎙️  Starting 6-second advanced test...")
        print("💬 Try speaking with pauses to test silence detection!")
        
        streaming_asr.start()
        time.sleep(6)
        streaming_asr.stop()
        
        # Advanced metrics analysis
        metrics = streaming_asr.get_metrics()
        history = streaming_asr.get_transcription_history()
        
        print(f"\n📈 Advanced Analysis:")
        print(f"   • Processing efficiency: {metrics['voice_chunks']}/{metrics['total_chunks']} chunks had voice")
        print(f"   • Transcription rate: {len(history)} results in 6 seconds")
        
        if transcription_times:
            avg_latency = sum(transcription_times) / len(transcription_times)
            min_latency = min(transcription_times)
            max_latency = max(transcription_times)
            print(f"   • Latency stats: avg={avg_latency:.3f}s, min={min_latency:.3f}s, max={max_latency:.3f}s")
        
        if voice_segments:
            print(f"   • Voice segments detected: {len(voice_segments)}")
            total_text_length = sum(len(seg['text']) for seg in voice_segments)
            print(f"   • Total text generated: {total_text_length} characters")
        
        # Test history and state management
        latest = streaming_asr.get_latest_transcription()
        print(f"   • Latest transcription: '{latest}'")
        
        print("✅ Advanced streaming features test completed")
        return True
        
    except Exception as e:
        print(f"❌ Advanced streaming test failed: {e}")
        return False


def test_multi_model_comparison():
    """Test multi-model ASR comparison (if multiple models available)"""
    print("\n" + "="*60)
    print("⚖️  TEST 3: Multi-Model Comparison")
    print("="*60)
    
    try:
        # Try to load multiple models
        models = {}
        
        # Load Whisper Tiny
        print("Loading Whisper Tiny...")
        whisper_tiny = get_model("whisper", model_size="tiny")
        whisper_tiny.load_model()
        models["whisper_tiny"] = whisper_tiny
        print("✅ Whisper Tiny loaded")
        
        # Try to load Whisper Base (if available)
        try:
            print("Loading Whisper Base...")
            whisper_base = get_model("whisper", model_size="base")
            whisper_base.load_model()
            models["whisper_base"] = whisper_base
            print("✅ Whisper Base loaded")
        except Exception as e:
            print(f"⚠️  Whisper Base not available: {e}")
        
        # Try to load Wav2Vec2 (if available)
        try:
            print("Loading Wav2Vec2...")
            wav2vec2 = get_model("wav2vec2")
            wav2vec2.load_model()
            models["wav2vec2"] = wav2vec2
            print("✅ Wav2Vec2 loaded")
        except Exception as e:
            print(f"⚠️  Wav2Vec2 not available: {e}")
        
        if len(models) < 2:
            print("⚠️  Need at least 2 models for comparison. Skipping multi-model test.")
            return True
        
        # Create multi-model ASR
        print(f"\n🔄 Comparing {len(models)} models: {list(models.keys())}")
        multi_asr = MultiModelASR(models)
        
        # Run comparison for 5 seconds
        print("🎙️  Starting 5-second multi-model comparison...")
        print("💬 Speak clearly to compare model performance!")
        
        comparison_results = multi_asr.start_comparison(duration=5.0)
        
        # Display comparison results
        print(f"\n📊 Multi-Model Comparison Results:")
        for model_name, results in comparison_results.items():
            metrics = results['metrics']
            print(f"\n   🤖 {model_name.upper()}:")
            print(f"      • Transcriptions: {results['transcription_count']}")
            print(f"      • Avg inference time: {results['average_inference_time']:.3f}s")
            print(f"      • Total text length: {results['total_text_length']} chars")
            print(f"      • Voice activity ratio: {metrics.get('voice_activity_ratio', 0):.2f}")
            
            if results['sample_transcriptions']:
                print(f"      • Sample outputs:")
                for i, text in enumerate(results['sample_transcriptions'][-2:], 1):
                    print(f"        {i}. '{text}'")
        
        print("✅ Multi-model comparison completed")
        return True
        
    except Exception as e:
        print(f"❌ Multi-model comparison failed: {e}")
        return False


def test_streaming_performance():
    """Test streaming ASR performance under different conditions"""
    print("\n" + "="*60)
    print("⚡ TEST 4: Performance Stress Test")
    print("="*60)
    
    try:
        # Load model
        model = get_model("whisper", model_size="tiny")
        model.load_model()
        
        # Test with aggressive settings for performance
        streaming_asr = StreamingASR(
            model=model,
            chunk_duration=0.2,  # Very small chunks
            window_duration=1.0,  # Short windows
            overlap_duration=0.2,
            min_speech_duration=0.1,  # Very responsive
            silence_timeout=0.5
        )
        
        # Performance tracking
        performance_data = {
            'chunk_count': 0,
            'transcription_count': 0,
            'latencies': [],
            'start_time': None
        }
        
        def performance_callback(result: TranscriptionResult):
            performance_data['transcription_count'] += 1
            performance_data['latencies'].append(result.inference_time)
            
            # Real-time performance feedback
            if performance_data['transcription_count'] % 3 == 0:
                avg_latency = sum(performance_data['latencies']) / len(performance_data['latencies'])
                print(f"⚡ Performance check: {performance_data['transcription_count']} transcriptions, "
                      f"avg latency: {avg_latency:.3f}s")
        
        streaming_asr.set_result_callback(performance_callback)
        
        print("\n🎙️  Starting 4-second performance stress test...")
        print("💬 Speak continuously to test system responsiveness!")
        
        performance_data['start_time'] = time.time()
        streaming_asr.start()
        time.sleep(4)
        streaming_asr.stop()
        
        # Performance analysis
        total_time = time.time() - performance_data['start_time']
        metrics = streaming_asr.get_metrics()
        
        print(f"\n⚡ Performance Analysis:")
        print(f"   • Total runtime: {total_time:.1f}s")
        print(f"   • Chunks processed: {metrics['total_chunks']}")
        print(f"   • Processing rate: {metrics['total_chunks']/total_time:.1f} chunks/sec")
        print(f"   • Transcriptions: {performance_data['transcription_count']}")
        
        if performance_data['latencies']:
            latencies = performance_data['latencies']
            print(f"   • Latency stats:")
            print(f"     - Average: {sum(latencies)/len(latencies):.3f}s")
            print(f"     - Min: {min(latencies):.3f}s")
            print(f"     - Max: {max(latencies):.3f}s")
            
            # Performance rating
            avg_latency = sum(latencies) / len(latencies)
            if avg_latency < 0.1:
                rating = "🚀 EXCELLENT"
            elif avg_latency < 0.2:
                rating = "✅ GOOD"
            elif avg_latency < 0.5:
                rating = "⚠️  ACCEPTABLE"
            else:
                rating = "❌ NEEDS OPTIMIZATION"
            
            print(f"   • Performance rating: {rating}")
        
        print("✅ Performance stress test completed")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run all streaming ASR tests"""
    print("🎯 STREAMING ASR COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("This will test real-time transcription with window-based processing")
    print("Make sure your microphone is working and ready!")
    
    # Wait for user confirmation
    input("\n📢 Press Enter when ready to start testing...")
    
    # Run all tests
    tests = [
        ("Basic Streaming ASR", test_streaming_asr_basic),
        ("Advanced Features", test_streaming_asr_advanced),
        ("Multi-Model Comparison", test_multi_model_comparison),
        ("Performance Stress Test", test_streaming_performance)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔄 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except KeyboardInterrupt:
            print(f"\n⏹️  Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Final summary
    print("\n" + "="*80)
    print("📋 STREAMING ASR TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All streaming ASR tests completed successfully!")
        print("🚀 System ready for real-time speech-to-text!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)