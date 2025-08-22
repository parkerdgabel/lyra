//! Comprehensive tests for Signal Processing & FFT module
//! 
//! These tests cover all Phase 1.2 requirements including:
//! - Core FFT algorithms (FFT, IFFT, Real FFT)
//! - Signal processing functions (convolution, correlation, filtering, windowing)
//! - Foreign object integration
//! - Performance and mathematical correctness
//! - Error handling

use std::collections::HashMap;

// Mock VM and standard library for testing
#[derive(Debug)]
struct MockVM {
    functions: HashMap<String, fn(&[lyra::vm::Value]) -> lyra::vm::VmResult<lyra::vm::Value>>,
}

impl MockVM {
    fn new() -> Self {
        let mut vm = MockVM {
            functions: HashMap::new(),
        };
        
        // Register our signal processing functions
        let signal_functions = lyra::stdlib::signal::register_signal_functions();
        for (name, function) in signal_functions {
            vm.functions.insert(name, function);
        }
        
        vm
    }
    
    fn call_function(&self, name: &str, args: &[lyra::vm::Value]) -> lyra::vm::VmResult<lyra::vm::Value> {
        match self.functions.get(name) {
            Some(func) => func(args),
            None => Err(lyra::vm::VmError::TypeError {
                expected: format!("function {}", name),
                actual: "function not found".to_string(),
            }),
        }
    }
}

use lyra::vm::Value;

// Helper trait for easier Value access in tests
trait ValueExt {
    fn as_list(&self) -> Option<&Vec<Value>>;
}

impl ValueExt for Value {
    fn as_list(&self) -> Option<&Vec<Value>> {
        match self {
            Value::List(list) => Some(list),
            _ => None,
        }
    }
}

#[cfg(test)]
mod signal_processing_tests {
    use super::*;

    fn setup_vm() -> MockVM {
        MockVM::new()
    }

    // =============================================================================
    // FFT Algorithm Tests
    // =============================================================================

    #[test]
    fn test_fft_basic_functionality() {
        let mut vm = setup_vm();
        
        // Test with simple real signal
        let signal = Value::List(vec![
            Value::Real(1.0), Value::Real(0.0), Value::Real(-1.0), Value::Real(0.0)
        ]);
        
        let result = vm.call_function("FFT", &[signal]).unwrap();
        
        // Should return a SpectralResult Foreign object
        match result {
            Value::LyObj(lyobj) => {
                // Verify it's a SpectralResult
                assert_eq!(lyobj.type_name(), "SpectralResult");
                
                // Test methods
                let frequencies = lyobj.call_method("Frequencies", &[]).unwrap();
                let magnitudes = lyobj.call_method("Magnitudes", &[]).unwrap();
                let method = lyobj.call_method("Method", &[]).unwrap();
                
                assert_eq!(method, Value::String("FFT".to_string()));
                
                match frequencies {
                    Value::List(freqs) => assert_eq!(freqs.len(), 4),
                    _ => panic!("Expected frequency list"),
                }
                
                match magnitudes {
                    Value::List(mags) => assert_eq!(mags.len(), 4),
                    _ => panic!("Expected magnitude list"),
                }
            }
            _ => panic!("Expected SpectralResult Foreign object"),
        }
    }

    #[test]
    fn test_ifft_roundtrip() {
        let mut vm = setup_vm();
        
        // Create a simple signal
        let original_signal = Value::List(vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0), Value::Real(4.0)
        ]);
        
        // FFT -> IFFT should recover original signal
        let fft_result = vm.call_function("FFT", &[original_signal.clone()]).unwrap();
        let ifft_result = vm.call_function("IFFT", &[fft_result]).unwrap();
        
        match ifft_result {
            Value::LyObj(lyobj) => {
                assert_eq!(lyobj.type_name(), "SignalData");
                
                let real_values = lyobj.call_method("ToReal", &[]).unwrap();
                match real_values {
                    Value::List(values) => {
                        assert_eq!(values.len(), 4);
                        // Check approximate equality (allow for floating point errors)
                        for (i, val) in values.iter().enumerate() {
                            if let Value::Real(x) = val {
                                let expected = (i + 1) as f64;
                                assert!((x - expected).abs() < 1e-10, "IFFT roundtrip failed");
                            }
                        }
                    }
                    _ => panic!("Expected real values list"),
                }
            }
            _ => panic!("Expected SignalData Foreign object"),
        }
    }

    #[test]
    fn test_real_fft() {
        let mut vm = setup_vm();
        
        // Test Real FFT with real-valued input
        let signal = Value::List(vec![
            Value::Real(1.0), Value::Real(0.0), Value::Real(-1.0), Value::Real(0.0),
            Value::Real(1.0), Value::Real(0.0), Value::Real(-1.0), Value::Real(0.0)
        ]);
        
        let result = vm.call_function("RealFFT", &[signal]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                assert_eq!(lyobj.type_name(), "SpectralResult");
                
                let magnitudes = lyobj.call_method("Magnitudes", &[]).unwrap();
                match magnitudes {
                    Value::List(mags) => {
                        // Real FFT should return N/2+1 frequency bins
                        assert_eq!(mags.len(), 5); // 8/2 + 1 = 5
                    }
                    _ => panic!("Expected magnitude list"),
                }
            }
            _ => panic!("Expected SpectralResult"),
        }
    }

    #[test]
    fn test_fft_power_of_two_optimization() {
        let mut vm = setup_vm();
        
        // Test with power of 2 length (should use optimized algorithm)
        let signal_pow2 = Value::List((0..16).map(|i| Value::Real(i as f64)).collect());
        let result_pow2 = vm.call_function("FFT", &[signal_pow2]).unwrap();
        
        // Test with non-power of 2 length (should pad and still work)
        let signal_non_pow2 = Value::List((0..15).map(|i| Value::Real(i as f64)).collect());
        let result_non_pow2 = vm.call_function("FFT", &[signal_non_pow2]).unwrap();
        
        // Both should succeed and return SpectralResult
        assert!(matches!(result_pow2, Value::LyObj(_)));
        assert!(matches!(result_non_pow2, Value::LyObj(_)));
    }

    // =============================================================================
    // Signal Processing Function Tests  
    // =============================================================================

    #[test]
    fn test_convolution() {
        let mut vm = setup_vm();
        
        let signal1 = Value::List(vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0)
        ]);
        let signal2 = Value::List(vec![
            Value::Real(0.5), Value::Real(1.0), Value::Real(0.5)
        ]);
        
        let result = vm.call_function("Convolve", &[signal1, signal2]).unwrap();
        
        match result {
            Value::List(values) => {
                assert_eq!(values.len(), 5); // length 3 + 3 - 1 = 5
                // Verify some convolution properties
                if let Value::Real(first) = values[0] {
                    assert_eq!(first, 0.5); // 1.0 * 0.5
                }
            }
            _ => panic!("Expected convolution result list"),
        }
    }

    #[test]
    fn test_circular_convolution() {
        let mut vm = setup_vm();
        
        let signal1 = Value::List(vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0), Value::Real(4.0)
        ]);
        let signal2 = Value::List(vec![
            Value::Real(1.0), Value::Real(0.0), Value::Real(0.0), Value::Real(0.0)
        ]);
        
        let result = vm.call_function("CircularConvolve", &[signal1.clone(), signal2]).unwrap();
        
        match result {
            Value::List(values) => {
                assert_eq!(values.len(), 4); // Same length as input
                // Circular convolution with [1,0,0,0] should return original signal
                if let (Value::Real(a), Value::Real(b)) = (&values[0], &signal1.as_list().unwrap()[0]) {
                    assert!((a - b).abs() < 1e-10);
                }
            }
            _ => panic!("Expected circular convolution result"),
        }
    }

    #[test]
    fn test_cross_correlation() {
        let mut vm = setup_vm();
        
        let signal1 = Value::List(vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0)
        ]);
        let signal2 = Value::List(vec![
            Value::Real(3.0), Value::Real(2.0), Value::Real(1.0)
        ]);
        
        let result = vm.call_function("CrossCorrelate", &[signal1, signal2]).unwrap();
        
        match result {
            Value::List(values) => {
                assert_eq!(values.len(), 5); // 3 + 3 - 1 = 5
                // Cross-correlation should find similarity
            }
            _ => panic!("Expected cross-correlation result"),
        }
    }

    #[test]
    fn test_auto_correlation() {
        let mut vm = setup_vm();
        
        let signal = Value::List(vec![
            Value::Real(1.0), Value::Real(0.0), Value::Real(-1.0), Value::Real(0.0)
        ]);
        
        let result = vm.call_function("AutoCorrelate", &[signal]).unwrap();
        
        match result {
            Value::List(values) => {
                assert_eq!(values.len(), 7); // 4 + 4 - 1 = 7
                // Auto-correlation should be symmetric around center
                let mid = values.len() / 2;
                if let (Value::Real(center), Value::Real(first)) = (&values[mid], &values[0]) {
                    assert!(center >= first); // Peak should be at center
                }
            }
            _ => panic!("Expected auto-correlation result"),
        }
    }

    // =============================================================================
    // Digital Filtering Tests
    // =============================================================================

    #[test]
    fn test_fir_filter() {
        let mut vm = setup_vm();
        
        let signal = Value::List((0..10).map(|i| Value::Real(i as f64)).collect());
        let coefficients = Value::List(vec![
            Value::Real(0.25), Value::Real(0.5), Value::Real(0.25)
        ]);
        
        let result = vm.call_function("FIRFilter", &[signal, coefficients]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                assert_eq!(lyobj.type_name(), "FilterResult");
                
                let filtered_signal = lyobj.call_method("FilteredSignal", &[]).unwrap();
                let filter_type = lyobj.call_method("FilterType", &[]).unwrap();
                
                assert_eq!(filter_type, Value::String("FIR".to_string()));
                
                match filtered_signal {
                    Value::List(values) => {
                        assert!(values.len() > 0);
                    }
                    _ => panic!("Expected filtered signal list"),
                }
            }
            _ => panic!("Expected FilterResult"),
        }
    }

    #[test]
    fn test_iir_filter() {
        let mut vm = setup_vm();
        
        let signal = Value::List(vec![
            Value::Real(1.0), Value::Real(0.0), Value::Real(1.0), Value::Real(0.0),
            Value::Real(1.0), Value::Real(0.0), Value::Real(1.0), Value::Real(0.0)
        ]);
        
        // Simple IIR coefficients (low-pass)
        let b_coeffs = Value::List(vec![Value::Real(0.5), Value::Real(0.5)]);
        let a_coeffs = Value::List(vec![Value::Real(1.0), Value::Real(-0.2)]);
        
        let result = vm.call_function("IIRFilter", &[signal, b_coeffs, a_coeffs]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                assert_eq!(lyobj.type_name(), "FilterResult");
                
                let filter_type = lyobj.call_method("FilterType", &[]).unwrap();
                assert_eq!(filter_type, Value::String("IIR".to_string()));
            }
            _ => panic!("Expected FilterResult"),
        }
    }

    #[test]
    fn test_low_pass_filter() {
        let mut vm = setup_vm();
        
        // Create signal with sample rate
        let signal_data = vm.call_function("SignalData", &[
            Value::List((0..100).map(|i| Value::Real((i as f64 * 0.1).sin())).collect()),
            Value::Real(1000.0) // sample rate
        ]).unwrap();
        
        let cutoff_freq = Value::Real(50.0);
        
        let result = vm.call_function("LowPassFilter", &[signal_data, cutoff_freq]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                assert_eq!(lyobj.type_name(), "FilterResult");
                
                let success = lyobj.call_method("Success", &[]).unwrap();
                assert_eq!(success, Value::Symbol("True".to_string()));
            }
            _ => panic!("Expected FilterResult"),
        }
    }

    #[test]
    fn test_high_pass_filter() {
        let mut vm = setup_vm();
        
        let signal_data = vm.call_function("SignalData", &[
            Value::List((0..50).map(|i| Value::Real(i as f64)).collect()),
            Value::Real(1000.0)
        ]).unwrap();
        
        let cutoff_freq = Value::Real(100.0);
        
        let result = vm.call_function("HighPassFilter", &[signal_data, cutoff_freq]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                assert_eq!(lyobj.type_name(), "FilterResult");
                
                let filter_type = lyobj.call_method("FilterType", &[]).unwrap();
                assert_eq!(filter_type, Value::String("HighPass".to_string()));
            }
            _ => panic!("Expected FilterResult"),
        }
    }

    #[test]
    fn test_band_pass_filter() {
        let mut vm = setup_vm();
        
        let signal_data = vm.call_function("SignalData", &[
            Value::List((0..100).map(|i| Value::Real((i as f64 * 0.1).sin())).collect()),
            Value::Real(1000.0)
        ]).unwrap();
        
        let low_freq = Value::Real(50.0);
        let high_freq = Value::Real(200.0);
        
        let result = vm.call_function("BandPassFilter", &[signal_data, low_freq, high_freq]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                assert_eq!(lyobj.type_name(), "FilterResult");
                
                let parameters = lyobj.call_method("Parameters", &[]).unwrap();
                match parameters {
                    Value::List(params) => assert_eq!(params.len(), 2),
                    _ => panic!("Expected parameter list"),
                }
            }
            _ => panic!("Expected FilterResult"),
        }
    }

    // =============================================================================
    // Window Function Tests
    // =============================================================================

    #[test]
    fn test_hamming_window() {
        let mut vm = setup_vm();
        
        let length = Value::Integer(16);
        let result = vm.call_function("HammingWindow", &[length]).unwrap();
        
        match result {
            Value::List(values) => {
                assert_eq!(values.len(), 16);
                // Verify window properties
                for val in &values {
                    if let Value::Real(x) = val {
                        assert!(*x >= 0.0 && *x <= 1.0, "Window values should be between 0 and 1");
                    }
                }
                // Hamming window should be symmetric
                if let (Value::Real(first), Value::Real(last)) = (&values[0], &values[15]) {
                    assert!((first - last).abs() < 1e-10, "Window should be symmetric");
                }
            }
            _ => panic!("Expected window coefficients list"),
        }
    }

    #[test]
    fn test_hanning_window() {
        let mut vm = setup_vm();
        
        let length = Value::Integer(8);
        let result = vm.call_function("HanningWindow", &[length]).unwrap();
        
        match result {
            Value::List(values) => {
                assert_eq!(values.len(), 8);
                // Hanning window should start and end with 0
                if let (Value::Real(first), Value::Real(last)) = (&values[0], &values[7]) {
                    assert!(first.abs() < 1e-10);
                    assert!(last.abs() < 1e-10);
                }
            }
            _ => panic!("Expected window coefficients"),
        }
    }

    #[test]
    fn test_blackman_window() {
        let mut vm = setup_vm();
        
        let length = Value::Integer(10);
        let result = vm.call_function("BlackmanWindow", &[length]).unwrap();
        
        match result {
            Value::List(values) => {
                assert_eq!(values.len(), 10);
                // Check window properties
                for val in &values {
                    if let Value::Real(x) = val {
                        assert!(*x >= 0.0 && *x <= 1.0);
                    }
                }
            }
            _ => panic!("Expected window coefficients"),
        }
    }

    #[test]
    fn test_kaiser_window() {
        let mut vm = setup_vm();
        
        let length = Value::Integer(12);
        let beta = Value::Real(5.0);
        let result = vm.call_function("KaiserWindow", &[length, beta]).unwrap();
        
        match result {
            Value::List(values) => {
                assert_eq!(values.len(), 12);
                // Kaiser window should be symmetric
                let mid = values.len() / 2;
                if let (Value::Real(early), Value::Real(late)) = (&values[1], &values[values.len()-2]) {
                    assert!((early - late).abs() < 1e-10);
                }
            }
            _ => panic!("Expected Kaiser window coefficients"),
        }
    }

    #[test]
    fn test_apply_window() {
        let mut vm = setup_vm();
        
        let signal = Value::List(vec![
            Value::Real(1.0), Value::Real(1.0), Value::Real(1.0), Value::Real(1.0)
        ]);
        let window = Value::List(vec![
            Value::Real(0.0), Value::Real(0.5), Value::Real(0.5), Value::Real(0.0)
        ]);
        
        let result = vm.call_function("ApplyWindow", &[signal, window]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                assert_eq!(lyobj.type_name(), "SignalData");
                
                let windowed = lyobj.call_method("ToReal", &[]).unwrap();
                match windowed {
                    Value::List(values) => {
                        if let (Value::Real(first), Value::Real(last)) = (&values[0], &values[3]) {
                            assert_eq!(*first, 0.0); // 1.0 * 0.0
                            assert_eq!(*last, 0.0);  // 1.0 * 0.0
                        }
                    }
                    _ => panic!("Expected windowed signal"),
                }
            }
            _ => panic!("Expected SignalData"),
        }
    }

    // =============================================================================
    // Spectral Analysis Tests
    // =============================================================================

    #[test]
    fn test_periodogram() {
        let mut vm = setup_vm();
        
        let signal = Value::List((0..64).map(|i| {
            Value::Real((2.0 * std::f64::consts::PI * 5.0 * i as f64 / 64.0).sin())
        }).collect());
        
        let result = vm.call_function("Periodogram", &[signal]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                assert_eq!(lyobj.type_name(), "SpectralResult");
                
                let method = lyobj.call_method("Method", &[]).unwrap();
                assert_eq!(method, Value::String("Periodogram".to_string()));
                
                let magnitudes = lyobj.call_method("Magnitudes", &[]).unwrap();
                match magnitudes {
                    Value::List(mags) => {
                        assert_eq!(mags.len(), 33); // 64/2 + 1 = 33
                    }
                    _ => panic!("Expected magnitude list"),
                }
            }
            _ => panic!("Expected SpectralResult"),
        }
    }

    #[test]
    fn test_welch_psd_estimate() {
        let mut vm = setup_vm();
        
        let signal = Value::List((0..128).map(|i| Value::Real(i as f64)).collect());
        let segment_length = Value::Integer(32);
        let overlap = Value::Real(0.5);
        
        let result = vm.call_function("WelchPSD", &[signal, segment_length, overlap]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                assert_eq!(lyobj.type_name(), "SpectralResult");
                
                let method = lyobj.call_method("Method", &[]).unwrap();
                assert_eq!(method, Value::String("WelchPSD".to_string()));
            }
            _ => panic!("Expected SpectralResult"),
        }
    }

    #[test]
    fn test_spectrogram() {
        let mut vm = setup_vm();
        
        let signal = Value::List((0..100).map(|i| Value::Real(i as f64)).collect());
        let window_size = Value::Integer(16);
        let hop_size = Value::Integer(8);
        
        let result = vm.call_function("Spectrogram", &[signal, window_size, hop_size]).unwrap();
        
        match result {
            Value::List(time_frames) => {
                assert!(time_frames.len() > 0);
                // Each time frame should be a list of frequency bins
                if let Value::List(first_frame) = &time_frames[0] {
                    assert_eq!(first_frame.len(), 9); // window_size/2 + 1 = 16/2 + 1 = 9
                }
            }
            _ => panic!("Expected spectrogram matrix"),
        }
    }

    // =============================================================================
    // Signal Generation Tests
    // =============================================================================

    #[test]
    fn test_sine_wave_generation() {
        let mut vm = setup_vm();
        
        let frequency = Value::Real(10.0);
        let duration = Value::Real(1.0);
        let sample_rate = Value::Real(100.0);
        let amplitude = Value::Real(1.0);
        
        let result = vm.call_function("SineWave", &[frequency, duration, sample_rate, amplitude]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                assert_eq!(lyobj.type_name(), "SignalData");
                
                let length = lyobj.call_method("Length", &[]).unwrap();
                if let Value::Integer(n) = length {
                    assert_eq!(n, 100); // 1.0 second * 100 Hz = 100 samples
                }
                
                let sample_rate_result = lyobj.call_method("SampleRate", &[]).unwrap();
                assert_eq!(sample_rate_result, Value::Real(100.0));
            }
            _ => panic!("Expected SignalData"),
        }
    }

    #[test]
    fn test_white_noise_generation() {
        let mut vm = setup_vm();
        
        let length = Value::Integer(1000);
        let amplitude = Value::Real(0.5);
        let seed = Value::Integer(42);
        
        let result = vm.call_function("WhiteNoise", &[length, amplitude, seed]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                assert_eq!(lyobj.type_name(), "SignalData");
                
                let signal = lyobj.call_method("ToReal", &[]).unwrap();
                match signal {
                    Value::List(values) => {
                        assert_eq!(values.len(), 1000);
                        // Check that values are within expected range
                        for val in &values {
                            if let Value::Real(x) = val {
                                assert!(x.abs() <= 0.5); // Should be within amplitude bounds
                            }
                        }
                    }
                    _ => panic!("Expected signal values"),
                }
            }
            _ => panic!("Expected SignalData"),
        }
    }

    #[test]
    fn test_chirp_signal() {
        let mut vm = setup_vm();
        
        let start_freq = Value::Real(5.0);
        let end_freq = Value::Real(50.0);
        let duration = Value::Real(2.0);
        let sample_rate = Value::Real(200.0);
        
        let result = vm.call_function("ChirpSignal", &[start_freq, end_freq, duration, sample_rate]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                assert_eq!(lyobj.type_name(), "SignalData");
                
                let length = lyobj.call_method("Length", &[]).unwrap();
                if let Value::Integer(n) = length {
                    assert_eq!(n, 400); // 2.0 seconds * 200 Hz = 400 samples
                }
            }
            _ => panic!("Expected SignalData"),
        }
    }

    // =============================================================================
    // Advanced Signal Processing Tests
    // =============================================================================

    #[test]
    fn test_hilbert_transform() {
        let mut vm = setup_vm();
        
        let signal = Value::List(vec![
            Value::Real(1.0), Value::Real(0.0), Value::Real(-1.0), Value::Real(0.0)
        ]);
        
        let result = vm.call_function("HilbertTransform", &[signal]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                assert_eq!(lyobj.type_name(), "SignalData");
                
                // Analytic signal should have complex values
                let magnitudes = lyobj.call_method("ToMagnitude", &[]).unwrap();
                let phases = lyobj.call_method("ToPhase", &[]).unwrap();
                
                match (magnitudes, phases) {
                    (Value::List(mags), Value::List(phases_vals)) => {
                        assert_eq!(mags.len(), phases_vals.len());
                        assert_eq!(mags.len(), 4);
                    }
                    _ => panic!("Expected magnitude and phase arrays"),
                }
            }
            _ => panic!("Expected SignalData with analytic signal"),
        }
    }

    #[test]
    fn test_envelope_detection() {
        let mut vm = setup_vm();
        
        // Create AM modulated signal
        let signal = Value::List((0..100).map(|i| {
            let t = i as f64 * 0.01;
            let carrier = (2.0 * std::f64::consts::PI * 50.0 * t).sin();
            let envelope = 1.0 + 0.5 * (2.0 * std::f64::consts::PI * 5.0 * t).sin();
            Value::Real(envelope * carrier)
        }).collect());
        
        let result = vm.call_function("EnvelopeDetect", &[signal]).unwrap();
        
        match result {
            Value::List(envelope_values) => {
                assert_eq!(envelope_values.len(), 100);
                // Envelope should be positive and smooth
                for val in &envelope_values {
                    if let Value::Real(x) = val {
                        assert!(*x >= 0.0, "Envelope should be non-negative");
                    }
                }
            }
            _ => panic!("Expected envelope values"),
        }
    }

    #[test]
    fn test_phase_unwrap() {
        let mut vm = setup_vm();
        
        // Create phase values with wrapping
        let phases = Value::List(vec![
            Value::Real(3.0), Value::Real(-3.0), Value::Real(2.8), Value::Real(-2.8)
        ]);
        
        let result = vm.call_function("PhaseUnwrap", &[phases]).unwrap();
        
        match result {
            Value::List(unwrapped) => {
                assert_eq!(unwrapped.len(), 4);
                
                // Check that unwrapped phases don't have large jumps
                let values: Vec<f64> = unwrapped.iter().map(|v| {
                    if let Value::Real(x) = v { *x } else { 0.0 }
                }).collect();
                
                for i in 1..values.len() {
                    let diff = (values[i] - values[i-1]).abs();
                    assert!(diff <= std::f64::consts::PI, "Phase jump too large");
                }
            }
            _ => panic!("Expected unwrapped phase values"),
        }
    }

    // =============================================================================
    // Foreign Object Integration Tests
    // =============================================================================

    #[test]
    fn test_signal_data_foreign_object() {
        let mut vm = setup_vm();
        
        let samples = Value::List(vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0)
        ]);
        let sample_rate = Value::Real(1000.0);
        
        let signal_data = vm.call_function("SignalData", &[samples, sample_rate]).unwrap();
        
        match signal_data {
            Value::LyObj(lyobj) => {
                assert_eq!(lyobj.type_name(), "SignalData");
                
                // Test all methods
                let length = lyobj.call_method("Length", &[]).unwrap();
                assert_eq!(length, Value::Integer(3));
                
                let sr = lyobj.call_method("SampleRate", &[]).unwrap();
                assert_eq!(sr, Value::Real(1000.0));
                
                let duration = lyobj.call_method("Duration", &[]).unwrap();
                assert_eq!(duration, Value::Real(0.003));
                
                let real_vals = lyobj.call_method("ToReal", &[]).unwrap();
                match real_vals {
                    Value::List(vals) => assert_eq!(vals.len(), 3),
                    _ => panic!("Expected real values list"),
                }
                
                let mag_vals = lyobj.call_method("ToMagnitude", &[]).unwrap();
                match mag_vals {
                    Value::List(vals) => assert_eq!(vals.len(), 3),
                    _ => panic!("Expected magnitude values list"),
                }
                
                let phase_vals = lyobj.call_method("ToPhase", &[]).unwrap();
                match phase_vals {
                    Value::List(vals) => assert_eq!(vals.len(), 3),
                    _ => panic!("Expected phase values list"),
                }
            }
            _ => panic!("Expected SignalData Foreign object"),
        }
    }

    #[test]
    fn test_spectral_result_foreign_object() {
        let mut vm = setup_vm();
        
        let signal = Value::List(vec![
            Value::Real(1.0), Value::Real(0.0), Value::Real(-1.0), Value::Real(0.0)
        ]);
        
        let spectral_result = vm.call_function("FFT", &[signal]).unwrap();
        
        match spectral_result {
            Value::LyObj(lyobj) => {
                assert_eq!(lyobj.type_name(), "SpectralResult");
                
                // Test all methods
                let frequencies = lyobj.call_method("Frequencies", &[]).unwrap();
                match frequencies {
                    Value::List(freqs) => assert_eq!(freqs.len(), 4),
                    _ => panic!("Expected frequency list"),
                }
                
                let magnitudes = lyobj.call_method("Magnitudes", &[]).unwrap();
                match magnitudes {
                    Value::List(mags) => assert_eq!(mags.len(), 4),
                    _ => panic!("Expected magnitude list"),
                }
                
                let phases = lyobj.call_method("Phases", &[]).unwrap();
                match phases {
                    Value::List(phases_vals) => assert_eq!(phases_vals.len(), 4),
                    _ => panic!("Expected phase list"),
                }
                
                let power_spectrum = lyobj.call_method("PowerSpectrum", &[]).unwrap();
                match power_spectrum {
                    Value::List(power_vals) => assert_eq!(power_vals.len(), 4),
                    _ => panic!("Expected power spectrum list"),
                }
                
                let sample_rate = lyobj.call_method("SampleRate", &[]).unwrap();
                assert!(matches!(sample_rate, Value::Real(_)));
                
                let method = lyobj.call_method("Method", &[]).unwrap();
                assert_eq!(method, Value::String("FFT".to_string()));
                
                let length = lyobj.call_method("Length", &[]).unwrap();
                assert_eq!(length, Value::Integer(4));
            }
            _ => panic!("Expected SpectralResult Foreign object"),
        }
    }

    #[test]
    fn test_filter_result_foreign_object() {
        let mut vm = setup_vm();
        
        let signal = Value::List(vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0), Value::Real(2.0), Value::Real(1.0)
        ]);
        let window_size = Value::Integer(3);
        
        let filter_result = vm.call_function("MedianFilter", &[signal, window_size]).unwrap();
        
        match filter_result {
            Value::LyObj(lyobj) => {
                assert_eq!(lyobj.type_name(), "FilterResult");
                
                // Test all methods
                let filtered_signal = lyobj.call_method("FilteredSignal", &[]).unwrap();
                match filtered_signal {
                    Value::List(vals) => assert_eq!(vals.len(), 5),
                    _ => panic!("Expected filtered signal list"),
                }
                
                let filter_type = lyobj.call_method("FilterType", &[]).unwrap();
                assert_eq!(filter_type, Value::String("Median".to_string()));
                
                let parameters = lyobj.call_method("Parameters", &[]).unwrap();
                match parameters {
                    Value::List(params) => assert_eq!(params.len(), 1),
                    _ => panic!("Expected parameters list"),
                }
                
                let success = lyobj.call_method("Success", &[]).unwrap();
                assert_eq!(success, Value::Symbol("True".to_string()));
                
                let message = lyobj.call_method("Message", &[]).unwrap();
                assert!(matches!(message, Value::String(_)));
            }
            _ => panic!("Expected FilterResult Foreign object"),
        }
    }

    // =============================================================================
    // Error Handling Tests
    // =============================================================================

    #[test]
    fn test_fft_error_handling() {
        let mut vm = setup_vm();
        
        // Test with empty signal
        let empty_signal = Value::List(vec![]);
        let result = vm.call_function("FFT", &[empty_signal]);
        assert!(result.is_err(), "Should fail with empty signal");
        
        // Test with invalid arguments
        let result = vm.call_function("FFT", &[]);
        assert!(result.is_err(), "Should fail with no arguments");
        
        let result = vm.call_function("FFT", &[Value::String("invalid".to_string())]);
        assert!(result.is_err(), "Should fail with invalid argument type");
    }

    #[test]
    fn test_filter_error_handling() {
        let mut vm = setup_vm();
        
        // Test low-pass filter with invalid cutoff frequency
        let signal_data = vm.call_function("SignalData", &[
            Value::List(vec![Value::Real(1.0)]),
            Value::Real(100.0)
        ]).unwrap();
        
        // Cutoff frequency too high (above Nyquist)
        let invalid_cutoff = Value::Real(200.0);
        let result = vm.call_function("LowPassFilter", &[signal_data.clone(), invalid_cutoff]);
        assert!(result.is_err(), "Should fail with cutoff > Nyquist frequency");
        
        // Negative cutoff frequency
        let negative_cutoff = Value::Real(-10.0);
        let result = vm.call_function("LowPassFilter", &[signal_data, negative_cutoff]);
        assert!(result.is_err(), "Should fail with negative cutoff frequency");
    }

    #[test]
    fn test_window_error_handling() {
        let mut vm = setup_vm();
        
        // Test with zero length window
        let zero_length = Value::Integer(0);
        let result = vm.call_function("HammingWindow", &[zero_length]);
        assert!(result.is_err(), "Should fail with zero length window");
        
        // Test with negative length
        let negative_length = Value::Integer(-5);
        let result = vm.call_function("HanningWindow", &[negative_length]);
        assert!(result.is_err(), "Should fail with negative length window");
    }

    #[test]
    fn test_convolution_error_handling() {
        let mut vm = setup_vm();
        
        // Test with mismatched signal types
        let signal1 = Value::List(vec![Value::Real(1.0)]);
        let signal2 = Value::String("invalid".to_string());
        
        let result = vm.call_function("Convolve", &[signal1, signal2]);
        assert!(result.is_err(), "Should fail with invalid signal type");
        
        // Test with empty signals
        let empty1 = Value::List(vec![]);
        let empty2 = Value::List(vec![]);
        let result = vm.call_function("Convolve", &[empty1, empty2]);
        assert!(result.is_err(), "Should fail with empty signals");
    }

    // =============================================================================
    // Performance Tests
    // =============================================================================

    #[test]
    fn test_fft_performance_large_signal() {
        let mut vm = setup_vm();
        
        // Test with large signal (should complete in reasonable time)
        let large_signal: Vec<Value> = (0..1024)
            .map(|i| Value::Real((i as f64 * 0.01).sin()))
            .collect();
        
        let start = std::time::Instant::now();
        let result = vm.call_function("FFT", &[Value::List(large_signal)]).unwrap();
        let duration = start.elapsed();
        
        assert!(duration.as_millis() < 100, "FFT should complete within 100ms for 1024 samples");
        assert!(matches!(result, Value::LyObj(_)), "Should return valid result");
    }

    #[test]
    fn test_convolution_performance() {
        let mut vm = setup_vm();
        
        // Test convolution performance
        let signal1: Vec<Value> = (0..512).map(|i| Value::Real(i as f64)).collect();
        let signal2: Vec<Value> = (0..128).map(|i| Value::Real(i as f64)).collect();
        
        let start = std::time::Instant::now();
        let result = vm.call_function("Convolve", &[Value::List(signal1), Value::List(signal2)]).unwrap();
        let duration = start.elapsed();
        
        assert!(duration.as_millis() < 50, "Convolution should complete within 50ms");
        assert!(matches!(result, Value::List(_)), "Should return valid result");
    }

    // =============================================================================
    // Mathematical Correctness Tests
    // =============================================================================

    #[test]
    fn test_fft_mathematical_properties() {
        let mut vm = setup_vm();
        
        // Test Parseval's theorem: sum of squares in time domain equals sum of squares in frequency domain
        let signal: Vec<Value> = (0..16)
            .map(|i| Value::Real((2.0 * std::f64::consts::PI * i as f64 / 16.0).sin()))
            .collect();
        
        // Calculate energy in time domain
        let time_energy: f64 = signal.iter().map(|v| {
            if let Value::Real(x) = v { x * x } else { 0.0 }
        }).sum();
        
        let fft_result = vm.call_function("FFT", &[Value::List(signal)]).unwrap();
        
        if let Value::LyObj(lyobj) = fft_result {
            let power_spectrum = lyobj.call_method("PowerSpectrum", &[]).unwrap();
            if let Value::List(power_vals) = power_spectrum {
                let freq_energy: f64 = power_vals.iter().map(|v| {
                    if let Value::Real(x) = v { *x } else { 0.0 }
                }).sum();
                
                // Should satisfy Parseval's theorem (within numerical tolerance)
                let relative_error = (time_energy - freq_energy).abs() / time_energy;
                assert!(relative_error < 1e-10, "Parseval's theorem violation");
            }
        }
    }

    #[test]
    fn test_convolution_commutativity() {
        let mut vm = setup_vm();
        
        let signal1 = Value::List(vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0)
        ]);
        let signal2 = Value::List(vec![
            Value::Real(0.5), Value::Real(1.0), Value::Real(0.5)
        ]);
        
        // Convolution should be commutative: conv(a,b) = conv(b,a)
        let result1 = vm.call_function("Convolve", &[signal1.clone(), signal2.clone()]).unwrap();
        let result2 = vm.call_function("Convolve", &[signal2, signal1]).unwrap();
        
        match (result1, result2) {
            (Value::List(vals1), Value::List(vals2)) => {
                assert_eq!(vals1.len(), vals2.len());
                for (v1, v2) in vals1.iter().zip(vals2.iter()) {
                    if let (Value::Real(x1), Value::Real(x2)) = (v1, v2) {
                        assert!((x1 - x2).abs() < 1e-10, "Convolution commutativity failed");
                    }
                }
            }
            _ => panic!("Expected convolution results"),
        }
    }

    #[test]
    fn test_window_normalization() {
        let mut vm = setup_vm();
        
        // Test that window functions have reasonable normalization
        let length = Value::Integer(32);
        
        let windows = vec![
            ("HammingWindow", vm.call_function("HammingWindow", &[length.clone()]).unwrap()),
            ("HanningWindow", vm.call_function("HanningWindow", &[length.clone()]).unwrap()),
            ("BlackmanWindow", vm.call_function("BlackmanWindow", &[length]).unwrap()),
        ];
        
        for (name, window) in windows {
            if let Value::List(coeffs) = window {
                let sum: f64 = coeffs.iter().map(|v| {
                    if let Value::Real(x) = v { *x } else { 0.0 }
                }).sum();
                
                // Window sum should be reasonable (not zero, not too large)
                assert!(sum > 0.1 && sum < 100.0, "{} window has unreasonable sum: {}", name, sum);
                
                // All coefficients should be non-negative for these windows
                for coeff in &coeffs {
                    if let Value::Real(x) = coeff {
                        assert!(*x >= 0.0, "{} window has negative coefficient", name);
                    }
                }
            }
        }
    }
}