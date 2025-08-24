//! Fast Fourier Transform algorithms for Lyra signal processing
//!
//! This module implements core FFT algorithms using the Foreign object pattern:
//! - FFT/IFFT using Cooley-Tukey algorithm
//! - Real FFT for real-valued signals
//! - DCT (Discrete Cosine Transform)
//!
//! All algorithms are optimized for performance and mathematical accuracy.

use crate::vm::{Value, VmError, VmResult};
use crate::stdlib::common::result::spectral_result;
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::stdlib::common::Complex;
use std::f64::consts::PI;


/// Signal data container with metadata - Foreign object
#[derive(Debug, Clone)]
pub struct SignalData {
    /// Complex-valued signal samples
    pub samples: Vec<Complex>,
    /// Sample rate in Hz
    pub sample_rate: f64,
    /// Signal length
    pub length: usize,
}

impl SignalData {
    pub fn new(samples: Vec<Complex>, sample_rate: f64) -> Self {
        let length = samples.len();
        SignalData { samples, sample_rate, length }
    }

    pub fn from_real(samples: Vec<f64>, sample_rate: f64) -> Self {
        let complex_samples: Vec<Complex> = samples
            .into_iter()
            .map(|x| Complex::new(x, 0.0))
            .collect();
        SignalData::new(complex_samples, sample_rate)
    }

    pub fn to_real(&self) -> Vec<f64> {
        self.samples.iter().map(|c| c.real).collect()
    }

    pub fn to_magnitude(&self) -> Vec<f64> {
        self.samples.iter().map(|c| c.magnitude()).collect()
    }

    pub fn to_phase(&self) -> Vec<f64> {
        self.samples.iter().map(|c| c.phase()).collect()
    }

    pub fn duration(&self) -> f64 {
        self.length as f64 / self.sample_rate
    }
}

/// Spectral analysis result - Foreign object
#[derive(Debug, Clone)]
pub struct SpectralResult {
    /// Frequency bins in Hz
    pub frequencies: Vec<f64>,
    /// Magnitude spectrum
    pub magnitudes: Vec<f64>,
    /// Phase spectrum
    pub phases: Vec<f64>,
    /// Original sample rate
    pub sample_rate: f64,
    /// Analysis method used
    pub method: String,
}

impl SpectralResult {
    pub fn new(frequencies: Vec<f64>, magnitudes: Vec<f64>, phases: Vec<f64>, 
               sample_rate: f64, method: String) -> Self {
        SpectralResult {
            frequencies,
            magnitudes,
            phases,
            sample_rate,
            method,
        }
    }

    pub fn length(&self) -> usize {
        self.frequencies.len()
    }

    pub fn power_spectrum(&self) -> Vec<f64> {
        self.magnitudes.iter().map(|m| m * m).collect()
    }
}

// =============================================================================
// FFT ALGORITHM IMPLEMENTATIONS
// =============================================================================

/// Compute FFT using Cooley-Tukey algorithm with optimizations
pub fn compute_fft(samples: &[Complex]) -> VmResult<Vec<Complex>> {
    let n = samples.len();
    
    if n <= 1 {
        return Ok(samples.to_vec());
    }

    // Check if power of 2 for efficiency
    if !n.is_power_of_two() {
        // Pad with zeros to next power of 2
        let next_pow2 = n.next_power_of_two();
        let mut padded = samples.to_vec();
        padded.resize(next_pow2, Complex::zero());
        return compute_fft_recursive(&padded);
    }

    compute_fft_recursive(samples)
}

fn compute_fft_recursive(samples: &[Complex]) -> VmResult<Vec<Complex>> {
    let n = samples.len();
    
    if n <= 1 {
        return Ok(samples.to_vec());
    }

    // Divide
    let even: Vec<Complex> = samples.iter().step_by(2).cloned().collect();
    let odd: Vec<Complex> = samples.iter().skip(1).step_by(2).cloned().collect();

    // Conquer
    let even_fft = compute_fft_recursive(&even)?;
    let odd_fft = compute_fft_recursive(&odd)?;

    // Combine
    let mut result = vec![Complex::zero(); n];
    for k in 0..n/2 {
        let t = Complex::from_polar(1.0, -2.0 * PI * k as f64 / n as f64) * odd_fft[k];
        result[k] = even_fft[k] + t;
        result[k + n/2] = even_fft[k] - t;
    }

    Ok(result)
}

/// Compute IFFT using conjugate property
pub fn compute_ifft(spectrum: &[Complex]) -> VmResult<Vec<Complex>> {
    // IFFT is computed as conjugate of FFT of conjugated input, divided by N
    let conjugated: Vec<Complex> = spectrum.iter().map(|c| c.conjugate()).collect();
    let fft_result = compute_fft(&conjugated)?;
    
    let n = spectrum.len() as f64;
    let result: Vec<Complex> = fft_result
        .into_iter()
        .map(|c| c.conjugate() * (1.0 / n))
        .collect();
    
    Ok(result)
}

/// Compute Real FFT for real-valued signals (returns one-sided spectrum)
fn compute_real_fft(samples: &[f64]) -> VmResult<SpectralResult> {
    let n = samples.len();
    
    // Convert to complex and compute full FFT
    let complex_samples: Vec<Complex> = samples.iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    
    let fft_result = compute_fft(&complex_samples)?;
    
    // Extract one-sided spectrum (positive frequencies only)
    let num_freqs = n / 2 + 1;
    let mut magnitudes = Vec::with_capacity(num_freqs);
    let mut phases = Vec::with_capacity(num_freqs);
    let mut frequencies = Vec::with_capacity(num_freqs);
    
    for k in 0..num_freqs {
        let magnitude = if k == 0 || (k == n/2 && n % 2 == 0) {
            // DC and Nyquist components (no doubling)
            fft_result[k].magnitude()
        } else {
            // Other frequencies (double for one-sided spectrum)
            2.0 * fft_result[k].magnitude()
        };
        
        magnitudes.push(magnitude);
        phases.push(fft_result[k].phase());
        frequencies.push(k as f64 / n as f64); // Normalized frequencies
    }
    
    Ok(SpectralResult::new(
        frequencies,
        magnitudes,
        phases,
        1.0, // Default normalized sample rate
        "RealFFT".to_string(),
    ))
}

/// Compute DCT Type-II (most common DCT)
fn compute_dct(samples: &[f64]) -> VmResult<Vec<f64>> {
    let n = samples.len();
    let mut result = vec![0.0; n];

    for k in 0..n {
        let mut sum = 0.0;
        for i in 0..n {
            sum += samples[i] * ((PI * k as f64 * (2.0 * i as f64 + 1.0)) / (2.0 * n as f64)).cos();
        }
        result[k] = sum;
    }

    // Apply normalization
    if !result.is_empty() {
        result[0] *= (1.0 / n as f64).sqrt();
        for i in 1..n {
            result[i] *= (2.0 / n as f64).sqrt();
        }
    }

    Ok(result)
}

/// Compute IDCT (Inverse DCT)
fn compute_idct(coeffs: &[f64]) -> VmResult<Vec<f64>> {
    let n = coeffs.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        let mut sum = 0.0;
        
        // DC component
        sum += coeffs[0] * (1.0 / n as f64).sqrt();
        
        // AC components
        for k in 1..n {
            sum += coeffs[k] * (2.0 / n as f64).sqrt() * 
                   ((PI * k as f64 * (2.0 * i as f64 + 1.0)) / (2.0 * n as f64)).cos();
        }
        
        result[i] = sum;
    }

    Ok(result)
}

/// Generate frequency bins for FFT result
fn generate_frequency_bins(n: usize, sample_rate: f64) -> Vec<f64> {
    (0..n)
        .map(|k| k as f64 * sample_rate / n as f64)
        .collect()
}

// =============================================================================
// PUBLIC API FUNCTIONS
// =============================================================================

/// Fast Fourier Transform
/// Usage: FFT[signal] -> Returns Association with frequencies, magnitudes, phases, sampleRate, method
pub fn fft(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (signal)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal_data = parse_signal_input(&args[0])?;
    let spectrum = compute_fft(&signal_data.samples)?;
    
    let frequencies = generate_frequency_bins(spectrum.len(), signal_data.sample_rate);
    let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.magnitude()).collect();
    let phases: Vec<f64> = spectrum.iter().map(|c| c.phase()).collect();
    Ok(spectral_result(frequencies, magnitudes, phases, signal_data.sample_rate, "FFT"))
}

/// Inverse Fast Fourier Transform
/// Usage: IFFT[spectrum] -> Returns SignalData
pub fn ifft(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (spectrum)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let spectrum_data = parse_complex_spectrum(&args[0])?;
    let signal = compute_ifft(&spectrum_data)?;
    
    let signal_result = SignalData::new(signal, 1.0); // Default sample rate
    Ok(Value::LyObj(LyObj::new(Box::new(signal_result))))
}

/// Real FFT for real-valued signals
/// Usage: RealFFT[signal] -> Returns Association with one-sided spectrum
pub fn real_fft(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (signal)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let real_signal = extract_real_signal(&args[0])?;
    let spectral = compute_real_fft(&real_signal)?;
    Ok(spectral_result(spectral.frequencies, spectral.magnitudes, spectral.phases, spectral.sample_rate, &spectral.method))
}

/// Discrete Cosine Transform
/// Usage: DCT[signal] -> Returns DCT coefficients
pub fn dct(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (signal)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal = extract_real_signal(&args[0])?;
    let dct_coeffs = compute_dct(&signal)?;
    
    let result: Vec<Value> = dct_coeffs.into_iter().map(|x| Value::Real(x)).collect();
    Ok(Value::List(result))
}

/// Inverse Discrete Cosine Transform
/// Usage: IDCT[coefficients] -> Returns reconstructed signal
pub fn idct(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (coefficients)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let coeffs = extract_real_signal(&args[0])?;
    let signal = compute_idct(&coeffs)?;
    
    let result: Vec<Value> = signal.into_iter().map(|x| Value::Real(x)).collect();
    Ok(Value::List(result))
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

fn parse_signal_input(value: &Value) -> VmResult<SignalData> {
    match value {
        Value::List(samples) => {
            let real_samples: VmResult<Vec<f64>> = samples
                .iter()
                .map(extract_number)
                .collect();
            Ok(SignalData::from_real(real_samples?, 1.0)) // Default sample rate
        }
        Value::LyObj(lyobj) => {
            if let Some(signal_data) = lyobj.downcast_ref::<SignalData>() {
                Ok(signal_data.clone())
            } else {
                Err(VmError::TypeError {
                    expected: "SignalData or list of numbers".to_string(),
                    actual: format!("LyObj (not SignalData): {}", lyobj.type_name()),
                })
            }
        }
        _ => Err(VmError::TypeError {
            expected: "signal data or list of numbers".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

fn parse_complex_spectrum(value: &Value) -> VmResult<Vec<Complex>> {
    match value {
        Value::LyObj(lyobj) => {
            if let Some(spectral_result) = lyobj.downcast_ref::<SpectralResult>() {
                // Convert magnitude and phase back to complex
                Ok(spectral_result.magnitudes
                    .iter()
                    .zip(spectral_result.phases.iter())
                    .map(|(&mag, &phase)| Complex::from_polar(mag, phase))
                    .collect())
            } else {
                Err(VmError::TypeError {
                    expected: "SpectralResult".to_string(),
                    actual: format!("LyObj (not SpectralResult): {}", lyobj.type_name()),
                })
            }
        }
        Value::Object(m) => {
            // Accept Association with keys: magnitudes, phases
            let mags = m.get("magnitudes").and_then(|v| v.as_list())
                .ok_or_else(|| VmError::TypeError { expected: "Association with 'magnitudes' list".to_string(), actual: format!("{:?}", value) })?;
            let phases = m.get("phases").and_then(|v| v.as_list())
                .ok_or_else(|| VmError::TypeError { expected: "Association with 'phases' list".to_string(), actual: format!("{:?}", value) })?;
            if mags.len() != phases.len() {
                return Err(VmError::Runtime("magnitudes and phases length mismatch".to_string()));
            }
            let mut result = Vec::with_capacity(mags.len());
            for (mv, pv) in mags.iter().zip(phases.iter()) {
                let mag = mv.as_real().ok_or_else(|| VmError::TypeError { expected: "Real".to_string(), actual: format!("{:?}", mv) })?;
                let ph = pv.as_real().ok_or_else(|| VmError::TypeError { expected: "Real".to_string(), actual: format!("{:?}", pv) })?;
                result.push(Complex::from_polar(mag, ph));
            }
            Ok(result)
        }
        _ => Err(VmError::TypeError {
            expected: "spectral result".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

fn extract_real_signal(value: &Value) -> VmResult<Vec<f64>> {
    match value {
        Value::List(items) => {
            items.iter().map(extract_number).collect()
        }
        _ => Err(VmError::TypeError {
            expected: "list of numbers".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

fn extract_number(value: &Value) -> VmResult<f64> {
    match value {
        Value::Integer(n) => Ok(*n as f64),
        Value::Real(r) => Ok(*r),
        _ => Err(VmError::TypeError {
            expected: "number".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

// =============================================================================
// FOREIGN TRAIT IMPLEMENTATIONS
// =============================================================================

impl Foreign for SignalData {
    fn type_name(&self) -> &'static str {
        "SignalData"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Length" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.length as i64))
            }
            "SampleRate" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Real(self.sample_rate))
            }
            "Duration" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Real(self.duration()))
            }
            "ToReal" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let real_values: Vec<Value> = self.to_real().into_iter().map(|x| Value::Real(x)).collect();
                Ok(Value::List(real_values))
            }
            "ToMagnitude" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let mag_values: Vec<Value> = self.to_magnitude().into_iter().map(|x| Value::Real(x)).collect();
                Ok(Value::List(mag_values))
            }
            "ToPhase" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let phase_values: Vec<Value> = self.to_phase().into_iter().map(|x| Value::Real(x)).collect();
                Ok(Value::List(phase_values))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Foreign for SpectralResult {
    fn type_name(&self) -> &'static str {
        "SpectralResult"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Frequencies" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let freq_values: Vec<Value> = self.frequencies.iter().map(|&f| Value::Real(f)).collect();
                Ok(Value::List(freq_values))
            }
            "Magnitudes" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let mag_values: Vec<Value> = self.magnitudes.iter().map(|&m| Value::Real(m)).collect();
                Ok(Value::List(mag_values))
            }
            "Phases" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let phase_values: Vec<Value> = self.phases.iter().map(|&p| Value::Real(p)).collect();
                Ok(Value::List(phase_values))
            }
            "PowerSpectrum" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let power_values: Vec<Value> = self.power_spectrum().into_iter().map(|p| Value::Real(p)).collect();
                Ok(Value::List(power_values))
            }
            "SampleRate" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Real(self.sample_rate))
            }
            "Method" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.method.clone()))
            }
            "Length" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.length() as i64))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_operations() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        
        let sum = a + b;
        assert_eq!(sum.real, 4.0);
        assert_eq!(sum.imag, 6.0);
        
        let product = a * b;
        assert_eq!(product.real, -5.0); // (1*3 - 2*4)
        assert_eq!(product.imag, 10.0); // (1*4 + 2*3)
        
        assert!((a.magnitude() - (5.0_f64).sqrt()).abs() < 1e-10);
        assert!((a.phase() - (2.0_f64).atan2(1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_fft_basic() {
        // Test FFT with simple signal
        let signal = vec![
            Value::Real(1.0), Value::Real(0.0), Value::Real(-1.0), Value::Real(0.0)
        ];
        
        let result = fft(&[Value::List(signal)]).unwrap();
        match result {
            Value::Object(map) => {
                assert_eq!(map.get("method"), Some(&Value::String("FFT".to_string())));
                assert!(map.get("frequencies").is_some());
                assert!(map.get("magnitudes").is_some());
                assert!(map.get("phases").is_some());
            }
            _ => panic!("Expected Association result"),
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let original_signal = vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0), Value::Real(4.0)
        ];
        
        // FFT -> IFFT should recover original signal
        let fft_result = fft(&[Value::List(original_signal.clone())]).unwrap();
        let ifft_result = ifft(&[fft_result]).unwrap();
        
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
            _ => panic!("Expected SignalData"),
        }
    }

    #[test]
    fn test_real_fft() {
        let signal = vec![
            Value::Real(1.0), Value::Real(0.0), Value::Real(-1.0), Value::Real(0.0),
            Value::Real(1.0), Value::Real(0.0), Value::Real(-1.0), Value::Real(0.0)
        ];
        
        let result = real_fft(&[Value::List(signal)]).unwrap();
        match result {
            Value::Object(map) => {
                assert_eq!(map.get("method"), Some(&Value::String("RealFFT".to_string())));
                if let Some(Value::List(mags)) = map.get("magnitudes") {
                    assert_eq!(mags.len(), 5); // 8/2 + 1 = 5
                } else {
                    panic!("Expected magnitudes list in Association");
                }
            }
            _ => panic!("Expected Association result"),
        }
    }

    #[test]
    fn test_dct_idct_roundtrip() {
        let original_signal = vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0), Value::Real(4.0)
        ];
        
        // DCT -> IDCT should recover original signal
        let dct_result = dct(&[Value::List(original_signal.clone())]).unwrap();
        let idct_result = idct(&[dct_result]).unwrap();
        
        match idct_result {
            Value::List(values) => {
                assert_eq!(values.len(), 4);
                // Check approximate equality
                for (i, val) in values.iter().enumerate() {
                    if let Value::Real(x) = val {
                        let expected = (i + 1) as f64;
                        assert!((x - expected).abs() < 1e-10, "DCT/IDCT roundtrip failed");
                    }
                }
            }
            _ => panic!("Expected reconstructed signal list"),
        }
    }

    #[test]
    fn test_signal_data_foreign_object() {
        let samples = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)];
        let signal_data = SignalData::new(samples, 1000.0);
        
        // Test methods
        assert_eq!(signal_data.call_method("Length", &[]).unwrap(), Value::Integer(3));
        assert_eq!(signal_data.call_method("SampleRate", &[]).unwrap(), Value::Real(1000.0));
        assert_eq!(signal_data.call_method("Duration", &[]).unwrap(), Value::Real(0.003));
        
        let real_vals = signal_data.call_method("ToReal", &[]).unwrap();
        match real_vals {
            Value::List(vals) => assert_eq!(vals.len(), 3),
            _ => panic!("Expected real values list"),
        }
    }
}
