"""
Evaluation Metrics Module for ASR System

This module provides comprehensive evaluation capabilities including:
- Latency measurement and analysis
- Accuracy evaluation with different metrics
- Performance benchmarking under various conditions
- Statistical analysis and reporting
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import statistics
from dataclasses import dataclass
from collections import defaultdict
import json
import os
from datetime import datetime

# Import our ASR components
try:
    from .models import get_model, ASRModel
    from .audio_processing import AudioPreprocessor
    from .enhanced_preprocessing import AdvancedAudioPreprocessor
    from .streaming_asr import StreamingASR
except ImportError:
    # Fallback for direct execution
    from models import get_model, ASRModel
    from audio_processing import AudioPreprocessor
    from enhanced_preprocessing import AdvancedAudioPreprocessor
    from streaming_asr import StreamingASR


@dataclass
class LatencyMetrics:
    """Container for latency measurements"""
    preprocessing_time: float
    inference_time: float
    total_time: float
    audio_duration: float
    real_time_factor: float  # total_time / audio_duration


@dataclass
class AccuracyMetrics:
    """Container for accuracy measurements"""
    word_error_rate: float
    character_error_rate: float
    bleu_score: float
    confidence_score: float


@dataclass
class EvaluationResult:
    """Container for complete evaluation results"""
    test_name: str
    model_type: str
    preprocessing_type: str
    latency: LatencyMetrics
    accuracy: Optional[AccuracyMetrics]
    metadata: Dict[str, Any]


class LatencyEvaluator:
    """Evaluates latency performance of ASR components"""
    
    def __init__(self):
        self.results = []
        
    def measure_preprocessing_latency(self, preprocessor, audio_data: np.ndarray, 
                                    sample_rate: int = 16000, iterations: int = 10) -> Dict[str, float]:
        """Measure preprocessing latency with multiple iterations"""
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            
            if hasattr(preprocessor, 'process'):
                # Enhanced preprocessor
                result = preprocessor.process(audio_data, sample_rate)
            elif hasattr(preprocessor, 'preprocess'):
                # Both basic and advanced preprocessors have preprocess method
                result = preprocessor.preprocess(audio_data, sample_rate)
            else:
                # Fallback for basic operations
                processed = preprocessor.resample_audio(audio_data, sample_rate)
                processed = preprocessor.normalize_audio(processed)
                result = processed
                
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99)
        }
    
    def measure_inference_latency(self, model: ASRModel, audio_data: np.ndarray,
                                sample_rate: int = 16000, iterations: int = 10) -> Dict[str, float]:
        """Measure model inference latency"""
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            result = model.transcribe(audio_data, sample_rate)
            end_time = time.perf_counter()
            
            # Handle tuple return from some models
            if isinstance(result, tuple):
                inference_time = result[1] if len(result) > 1 else (end_time - start_time)
            else:
                inference_time = end_time - start_time
                
            times.append(inference_time)
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99)
        }
    
    def measure_end_to_end_latency(self, model: ASRModel, preprocessor, 
                                 audio_data: np.ndarray, sample_rate: int = 16000,
                                 iterations: int = 10) -> LatencyMetrics:
        """Measure complete end-to-end latency"""
        preprocessing_times = []
        inference_times = []
        total_times = []
        
        audio_duration = len(audio_data) / sample_rate
        
        for _ in range(iterations):
            # Measure preprocessing
            start_time = time.perf_counter()
            if hasattr(preprocessor, 'process'):
                processed_result = preprocessor.process(audio_data, sample_rate)
                processed_audio = processed_result['audio']
            elif hasattr(preprocessor, 'preprocess'):
                processed_result = preprocessor.preprocess(audio_data, sample_rate)
                if isinstance(processed_result, tuple):
                    processed_audio = processed_result[0]
                elif isinstance(processed_result, dict):
                    processed_audio = processed_result['audio']
                else:
                    processed_audio = processed_result
            else:
                processed_audio = preprocessor.resample_audio(audio_data, sample_rate)
                processed_audio = preprocessor.normalize_audio(processed_audio)
            preprocessing_time = time.perf_counter() - start_time
            
            # Measure inference
            inference_start = time.perf_counter()
            result = model.transcribe(processed_audio, 16000)
            inference_time = time.perf_counter() - inference_start
            
            # Handle tuple return
            if isinstance(result, tuple) and len(result) > 1:
                model_inference_time = result[1]
            else:
                model_inference_time = inference_time
            
            total_time = preprocessing_time + inference_time
            
            preprocessing_times.append(preprocessing_time)
            inference_times.append(model_inference_time)
            total_times.append(total_time)
        
        return LatencyMetrics(
            preprocessing_time=np.mean(preprocessing_times),
            inference_time=np.mean(inference_times),
            total_time=np.mean(total_times),
            audio_duration=audio_duration,
            real_time_factor=np.mean(total_times) / audio_duration
        )


class AccuracyEvaluator:
    """Evaluates accuracy of ASR transcriptions"""
    
    def __init__(self):
        self.results = []
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate"""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        # Simple WER calculation using edit distance
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
        
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j
            
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
        
        return d[len(ref_words)][len(hyp_words)] / len(ref_words) if ref_words else 0.0
    
    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate"""
        ref_chars = list(reference.lower())
        hyp_chars = list(hypothesis.lower())
        
        # Simple CER calculation using edit distance
        d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1))
        
        for i in range(len(ref_chars) + 1):
            d[i][0] = i
        for j in range(len(hyp_chars) + 1):
            d[0][j] = j
            
        for i in range(1, len(ref_chars) + 1):
            for j in range(1, len(hyp_chars) + 1):
                if ref_chars[i-1] == hyp_chars[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
        
        return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars) if ref_chars else 0.0
    
    def calculate_bleu_score(self, reference: str, hypothesis: str) -> float:
        """Simple BLEU score calculation (1-gram only)"""
        ref_words = set(reference.lower().split())
        hyp_words = set(hypothesis.lower().split())
        
        if not hyp_words:
            return 0.0
        
        intersection = ref_words.intersection(hyp_words)
        return len(intersection) / len(hyp_words)
    
    def evaluate_transcription(self, reference: str, hypothesis: str, 
                             confidence: float = 1.0) -> AccuracyMetrics:
        """Evaluate a single transcription"""
        return AccuracyMetrics(
            word_error_rate=self.calculate_wer(reference, hypothesis),
            character_error_rate=self.calculate_cer(reference, hypothesis),
            bleu_score=self.calculate_bleu_score(reference, hypothesis),
            confidence_score=confidence
        )


class ComprehensiveEvaluator:
    """Main evaluation class that combines latency and accuracy evaluation"""
    
    def __init__(self):
        self.latency_evaluator = LatencyEvaluator()
        self.accuracy_evaluator = AccuracyEvaluator()
        self.results = []
    
    def evaluate_model_configuration(self, model_type: str, model_config: Dict,
                                   preprocessor_type: str, preprocessor_config: Dict,
                                   test_audio: np.ndarray, sample_rate: int,
                                   reference_text: str = None,
                                   test_name: str = "default") -> EvaluationResult:
        """Evaluate a specific model and preprocessor configuration"""
        
        # Initialize model
        model = get_model(model_type, **model_config)
        model.load_model()
        
        # Initialize preprocessor
        if preprocessor_type == "enhanced":
            preprocessor = AdvancedAudioPreprocessor(**preprocessor_config)
        else:
            preprocessor = AudioPreprocessor(**preprocessor_config)
        
        # Measure latency
        latency_metrics = self.latency_evaluator.measure_end_to_end_latency(
            model, preprocessor, test_audio, sample_rate
        )
        
        # Measure accuracy if reference text is provided
        accuracy_metrics = None
        if reference_text:
            # Get transcription
            if hasattr(preprocessor, 'process'):
                processed_result = preprocessor.process(test_audio, sample_rate)
                processed_audio = processed_result['audio']
            elif hasattr(preprocessor, 'preprocess'):
                processed_result = preprocessor.preprocess(test_audio, sample_rate)
                if isinstance(processed_result, tuple):
                    processed_audio = processed_result[0]
                elif isinstance(processed_result, dict):
                    processed_audio = processed_result['audio']
                else:
                    processed_audio = processed_result
            else:
                processed_audio = preprocessor.resample_audio(test_audio, sample_rate)
                processed_audio = preprocessor.normalize_audio(processed_audio)
            
            transcription_result = model.transcribe(processed_audio, 16000)
            
            # Handle tuple return
            if isinstance(transcription_result, tuple):
                transcription_text = transcription_result[0]
            else:
                transcription_text = transcription_result
            
            accuracy_metrics = self.accuracy_evaluator.evaluate_transcription(
                reference_text, transcription_text
            )
        
        # Create result
        result = EvaluationResult(
            test_name=test_name,
            model_type=model_type,
            preprocessing_type=preprocessor_type,
            latency=latency_metrics,
            accuracy=accuracy_metrics,
            metadata={
                'model_config': model_config,
                'preprocessor_config': preprocessor_config,
                'audio_duration': len(test_audio) / sample_rate,
                'sample_rate': sample_rate
            }
        )
        
        self.results.append(result)
        return result
    
    def run_comprehensive_benchmark(self, test_audio: np.ndarray, sample_rate: int,
                                  reference_text: str = None) -> List[EvaluationResult]:
        """Run comprehensive benchmark with multiple configurations"""
        
        configurations = [
            # Whisper configurations
            {
                'model_type': 'whisper',
                'model_config': {'model_size': 'tiny'},
                'preprocessor_type': 'basic',
                'preprocessor_config': {'sample_rate': 16000},
                'test_name': 'whisper_tiny_basic'
            },
            {
                'model_type': 'whisper',
                'model_config': {'model_size': 'tiny'},
                'preprocessor_type': 'enhanced',
                'preprocessor_config': {'profile': 'fast_processing'},
                'test_name': 'whisper_tiny_enhanced_fast'
            },
            {
                'model_type': 'whisper',
                'model_config': {'model_size': 'tiny'},
                'preprocessor_type': 'enhanced',
                'preprocessor_config': {'profile': 'high_quality'},
                'test_name': 'whisper_tiny_enhanced_quality'
            },
            {
                'model_type': 'whisper',
                'model_config': {'model_size': 'base'},
                'preprocessor_type': 'enhanced',
                'preprocessor_config': {'profile': 'noise_robust'},
                'test_name': 'whisper_base_enhanced_robust'
            },
            # Wav2Vec2 configurations
            {
                'model_type': 'wav2vec2',
                'model_config': {},
                'preprocessor_type': 'basic',
                'preprocessor_config': {'target_sample_rate': 16000},
                'test_name': 'wav2vec2_basic'
            },
            {
                'model_type': 'wav2vec2',
                'model_config': {},
                'preprocessor_type': 'enhanced',
                'preprocessor_config': {'profile': 'fast_processing'},
                'test_name': 'wav2vec2_enhanced_fast'
            }
        ]
        
        results = []
        for config in configurations:
            try:
                print(f"🔄 Evaluating: {config['test_name']}")
                result = self.evaluate_model_configuration(
                    config['model_type'],
                    config['model_config'],
                    config['preprocessor_type'],
                    config['preprocessor_config'],
                    test_audio,
                    sample_rate,
                    reference_text,
                    config['test_name']
                )
                results.append(result)
                print(f"✅ Completed: {config['test_name']}")
            except Exception as e:
                print(f"❌ Failed: {config['test_name']} - {str(e)}")
        
        return results
    
    def generate_report(self, results: List[EvaluationResult] = None) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        if results is None:
            results = self.results
        
        if not results:
            return {"error": "No evaluation results available", "total_tests": 0}
        
        # Latency analysis
        latency_data = []
        for result in results:
            latency_data.append({
                'test_name': result.test_name,
                'model_type': result.model_type,
                'preprocessing_type': result.preprocessing_type,
                'preprocessing_time': result.latency.preprocessing_time,
                'inference_time': result.latency.inference_time,
                'total_time': result.latency.total_time,
                'real_time_factor': result.latency.real_time_factor,
                'audio_duration': result.latency.audio_duration
            })
        
        # Accuracy analysis (if available)
        accuracy_data = []
        for result in results:
            if result.accuracy:
                accuracy_data.append({
                    'test_name': result.test_name,
                    'model_type': result.model_type,
                    'preprocessing_type': result.preprocessing_type,
                    'word_error_rate': result.accuracy.word_error_rate,
                    'character_error_rate': result.accuracy.character_error_rate,
                    'bleu_score': result.accuracy.bleu_score,
                    'confidence_score': result.accuracy.confidence_score
                })
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(results),
            'latency_analysis': {
                'data': latency_data,
                'summary': {
                    'fastest_total': min(latency_data, key=lambda x: x['total_time']),
                    'slowest_total': max(latency_data, key=lambda x: x['total_time']),
                    'best_rtf': min(latency_data, key=lambda x: x['real_time_factor']),
                    'worst_rtf': max(latency_data, key=lambda x: x['real_time_factor'])
                }
            }
        }
        
        if accuracy_data:
            report['accuracy_analysis'] = {
                'data': accuracy_data,
                'summary': {
                    'best_wer': min(accuracy_data, key=lambda x: x['word_error_rate']),
                    'worst_wer': max(accuracy_data, key=lambda x: x['word_error_rate']),
                    'best_bleu': max(accuracy_data, key=lambda x: x['bleu_score']),
                    'worst_bleu': min(accuracy_data, key=lambda x: x['bleu_score'])
                }
            }
        
        return report
    
    def save_report(self, filename: str = None, results: List[EvaluationResult] = None):
        """Save evaluation report to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.json"
        
        report = self.generate_report(results)
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Evaluation report saved to: {filename}")
        return filename


# Test block
if __name__ == "__main__":
    print("🧪 Testing Evaluation Metrics Module...")
    
    # Generate test audio
    duration = 3.0  # 3 seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Create a simple sine wave with some noise
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    test_audio = test_audio.astype(np.float32)
    
    reference_text = "hello world this is a test"
    
    print(f"📊 Generated test audio: {duration}s, {sample_rate}Hz")
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator()
    
    # Run comprehensive benchmark
    print("\n🚀 Running comprehensive benchmark...")
    results = evaluator.run_comprehensive_benchmark(
        test_audio, sample_rate, reference_text
    )
    
    # Generate and display report
    print("\n📈 Generating evaluation report...")
    report = evaluator.generate_report(results)
    
    print(f"\n📊 EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total tests completed: {report['total_tests']}")
    
    if 'latency_analysis' in report:
        latency = report['latency_analysis']
        print(f"\n⚡ LATENCY ANALYSIS:")
        print(f"Fastest configuration: {latency['summary']['fastest_total']['test_name']}")
        print(f"  • Total time: {latency['summary']['fastest_total']['total_time']:.3f}s")
        print(f"  • Real-time factor: {latency['summary']['fastest_total']['real_time_factor']:.2f}x")
        
        print(f"\nBest real-time factor: {latency['summary']['best_rtf']['test_name']}")
        print(f"  • Real-time factor: {latency['summary']['best_rtf']['real_time_factor']:.2f}x")
        print(f"  • Total time: {latency['summary']['best_rtf']['total_time']:.3f}s")
    
    if 'accuracy_analysis' in report:
        accuracy = report['accuracy_analysis']
        print(f"\n🎯 ACCURACY ANALYSIS:")
        print(f"Best WER: {accuracy['summary']['best_wer']['test_name']}")
        print(f"  • Word Error Rate: {accuracy['summary']['best_wer']['word_error_rate']:.3f}")
        print(f"  • BLEU Score: {accuracy['summary']['best_wer']['bleu_score']:.3f}")
    
    # Save report
    filename = evaluator.save_report(results=results)
    
    print(f"\n✅ Evaluation metrics module test completed!")
    print(f"📄 Report saved as: {filename}")