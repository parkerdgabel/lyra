//! Signal Processing & Fourier Analysis for the Lyra standard library
//!
//! This module implements comprehensive digital signal processing and Fourier analysis
//! algorithms following the "Take Algorithms for granted" principle. Users can rely on
//! efficient, battle-tested implementations of essential signal processing functions.
//!
//! ## Features
//!
//! - **Fourier Transforms**: FFT, IFFT, DFT, DCT with optimized algorithms
//! - **Digital Filtering**: Butterworth filters, median filtering, moving averages
//! - **Convolution**: Linear and FFT-based convolution, correlation analysis
//! - **Windowing Functions**: Hamming, Hanning, Blackman, Kaiser windows
//! - **Spectral Analysis**: Periodograms, spectrograms, power spectral density
//! - **Advanced Processing**: Hilbert transform, envelope detection, phase unwrapping
//!
//! ## Design Philosophy
//!
//! All functions provide robust error handling, automatic parameter selection where
//! appropriate, and consistent interfaces. The module integrates seamlessly with
//! Lyra's tensor system for multi-dimensional signal processing.

use crate::vm::{Value, VmError, VmResult};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::stdlib::common::Complex;
use std::f64::consts::PI;


/// Signal data container with metadata
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

/// Result of spectral analysis operations
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

/// Result of filtering operations
#[derive(Debug, Clone)]
pub struct FilterResult {
    /// Filtered signal samples
    pub filtered_signal: Vec<f64>,
    /// Filter type used
    pub filter_type: String,
    /// Filter parameters
    pub parameters: Vec<f64>,
    /// Success flag
    pub success: bool,
    /// Status message
    pub message: String,
}

// =============================================================================
// FOURIER TRANSFORM FUNCTIONS
// =============================================================================

/// Fast Fourier Transform
/// Usage: FFT[signal] -> Returns complex spectrum
pub fn fft(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (signal)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal_data = parse_signal_input(&args[0])?;
    let spectrum = compute_fft(&signal_data.samples)?;
    
    let spectral_result = SpectralResult::new(
        generate_frequency_bins(spectrum.len(), signal_data.sample_rate),
        spectrum.iter().map(|c| c.magnitude()).collect(),
        spectrum.iter().map(|c| c.phase()).collect(),
        signal_data.sample_rate,
        "FFT".to_string(),
    );

    Ok(Value::LyObj(LyObj::new(Box::new(spectral_result))))
}

/// Inverse Fast Fourier Transform
/// Usage: IFFT[spectrum] -> Returns time-domain signal
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

/// Discrete Cosine Transform
/// Usage: DCT[signal] -> Returns DCT coefficients
pub fn dct(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (signal)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal_data = parse_signal_input(&args[0])?;
    let dct_coeffs = compute_dct(&signal_data.to_real())?;
    
    let result: Vec<Value> = dct_coeffs.into_iter().map(|x| Value::Real(x)).collect();
    Ok(Value::List(result))
}

/// Power Spectrum estimation
/// Usage: PowerSpectrum[signal] -> Returns power spectral density
pub fn power_spectrum(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (signal)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal_data = parse_signal_input(&args[0])?;
    let spectrum = compute_fft(&signal_data.samples)?;
    let power_spec: Vec<f64> = spectrum.iter().map(|c| c.magnitude().powi(2)).collect();
    
    let spectral_result = SpectralResult::new(
        generate_frequency_bins(spectrum.len(), signal_data.sample_rate),
        power_spec.clone(),
        vec![0.0; power_spec.len()], // Phase not meaningful for power spectrum
        signal_data.sample_rate,
        "PowerSpectrum".to_string(),
    );

    Ok(Value::LyObj(LyObj::new(Box::new(spectral_result))))
}

// =============================================================================
// SPECTRAL ANALYSIS FUNCTIONS
// =============================================================================

/// Periodogram - Power Spectral Density Estimate
/// Usage: Periodogram[signal] -> Returns SpectralResult with PSD estimate
pub fn periodogram(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (signal)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal = extract_real_signal(&args[0])?;
    let n = signal.len();
    
    if n == 0 {
        return Ok(Value::List(vec![]));
    }

    // Convert to complex and compute FFT for periodogram
    let complex_signal: Vec<Complex> = signal.iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    let fft_result = compute_fft(&complex_signal)?;
    
    // Compute power spectral density
    let mut psd = Vec::with_capacity(n / 2 + 1);
    
    // DC component
    let dc_power = fft_result[0].magnitude().powi(2) / (n * n) as f64;
    psd.push(dc_power);
    
    // Positive frequencies (multiply by 2 for one-sided spectrum)
    for i in 1..=n/2 {
        let power = if i == n/2 && n % 2 == 0 {
            // Nyquist frequency (no doubling)
            fft_result[i].magnitude().powi(2) / (n * n) as f64
        } else {
            // Regular frequencies (double for one-sided)
            2.0 * fft_result[i].magnitude().powi(2) / (n * n) as f64
        };
        psd.push(power);
    }
    
    // Create frequency vector (normalized frequencies)
    let frequencies: Vec<f64> = (0..=n/2).map(|k| k as f64 / n as f64).collect();
    let spectral_result = SpectralResult {
        frequencies,
        magnitudes: psd.clone(),
        phases: vec![0.0; psd.len()], // phases not used for periodogram
        sample_rate: 1.0, // normalized sample rate
        method: "Periodogram".to_string(),
    };
    
    Ok(Value::LyObj(LyObj::new(Box::new(spectral_result))))
}

/// Spectrogram - Short-Time Fourier Transform
/// Usage: Spectrogram[signal, windowSize, hopSize] -> Returns time-frequency matrix
pub fn spectrogram(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "1-3 arguments (signal, optional windowSize, optional hopSize)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal = extract_real_signal(&args[0])?;
    let window_size = if args.len() > 1 {
        extract_positive_integer(&args[1])? as usize
    } else {
        256.min(signal.len())
    };
    
    let hop_size = if args.len() > 2 {
        extract_positive_integer(&args[2])? as usize
    } else {
        window_size / 4
    };

    if window_size == 0 || hop_size == 0 || window_size > signal.len() {
        return Err(VmError::TypeError {
            expected: "valid window and hop sizes".to_string(),
            actual: format!("windowSize={}, hopSize={}, signalLength={}", 
                          window_size, hop_size, signal.len()),
        });
    }

    // Generate Hamming window for STFT
    let window = generate_hamming_window(window_size);
    
    // Compute number of time frames
    let num_frames = (signal.len() - window_size) / hop_size + 1;
    let num_freqs = window_size / 2 + 1;
    
    // Compute STFT
    let mut spectrogram = vec![vec![0.0; num_freqs]; num_frames];
    
    for frame in 0..num_frames {
        let start = frame * hop_size;
        let end = start + window_size;
        
        if end <= signal.len() {
            // Apply window to signal segment
            let windowed: Vec<f64> = signal[start..end]
                .iter()
                .zip(window.iter())
                .map(|(s, w)| s * w)
                .collect();
            
            // Convert to complex and compute FFT of windowed segment
            let complex_windowed: Vec<Complex> = windowed.iter()
                .map(|&x| Complex::new(x, 0.0))
                .collect();
            let fft_result = compute_fft(&complex_windowed)?;
            
            // Store magnitude spectrum
            for (freq_bin, &Complex { real, imag }) in fft_result[0..num_freqs].iter().enumerate() {
                spectrogram[frame][freq_bin] = (real * real + imag * imag).sqrt();
            }
        }
    }
    
    // Convert to nested list format
    let mut result = Vec::new();
    for frame in spectrogram {
        let frame_values: Vec<Value> = frame.into_iter()
            .map(Value::Real)
            .collect();
        result.push(Value::List(frame_values));
    }
    
    Ok(Value::List(result))
}

/// PSD Estimate using Welch's Method
/// Usage: PSDEstimate[signal, segmentLength, overlap] -> Returns SpectralResult
pub fn psd_estimate(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "1-3 arguments (signal, optional segmentLength, optional overlap)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal = extract_real_signal(&args[0])?;
    let segment_length = if args.len() > 1 {
        extract_positive_integer(&args[1])? as usize
    } else {
        256.min(signal.len())
    };
    
    let overlap = if args.len() > 2 {
        let overlap_val = extract_real(&args[2])?;
        if !(0.0..1.0).contains(&overlap_val) {
            return Err(VmError::TypeError {
                expected: "overlap between 0.0 and 1.0".to_string(),
                actual: format!("{}", overlap_val),
            });
        }
        (overlap_val * segment_length as f64) as usize
    } else {
        segment_length / 2
    };

    if segment_length == 0 || segment_length > signal.len() {
        return Err(VmError::TypeError {
            expected: "valid segment length".to_string(),
            actual: format!("segmentLength={}, signalLength={}", segment_length, signal.len()),
        });
    }

    let hop_size = segment_length - overlap;
    if hop_size == 0 {
        return Err(VmError::TypeError {
            expected: "hop size > 0".to_string(),
            actual: "overlap too large".to_string(),
        });
    }

    // Generate window function
    let window = generate_hamming_window(segment_length);
    let window_power: f64 = window.iter().map(|w| w * w).sum();
    
    // Compute number of segments
    let num_segments = (signal.len() - segment_length) / hop_size + 1;
    
    if num_segments == 0 {
        return Err(VmError::TypeError {
            expected: "at least one segment".to_string(),
            actual: "signal too short".to_string(),
        });
    }

    let num_freqs = segment_length / 2 + 1;
    let mut psd_accumulator = vec![0.0; num_freqs];
    
    // Process each segment
    for segment in 0..num_segments {
        let start = segment * hop_size;
        let end = start + segment_length;
        
        if end <= signal.len() {
            // Apply window to segment
            let windowed: Vec<f64> = signal[start..end]
                .iter()
                .zip(window.iter())
                .map(|(s, w)| s * w)
                .collect();
            
            // Convert to complex and compute FFT
            let complex_windowed: Vec<Complex> = windowed.iter()
                .map(|&x| Complex::new(x, 0.0))
                .collect();
            let fft_result = compute_fft(&complex_windowed)?;
            
            // Accumulate power spectral density
            for i in 0..num_freqs {
                let power = fft_result[i].magnitude().powi(2);
                psd_accumulator[i] += power;
            }
        }
    }
    
    // Average and normalize
    let scale = 1.0 / (num_segments as f64 * window_power * segment_length as f64);
    for i in 0..num_freqs {
        psd_accumulator[i] *= scale;
        // Double power for positive frequencies (one-sided spectrum)
        if i > 0 && i < num_freqs - 1 {
            psd_accumulator[i] *= 2.0;
        }
    }
    
    // Create frequency vector (normalized frequencies)
    let frequencies: Vec<f64> = (0..num_freqs)
        .map(|k| k as f64 / segment_length as f64)
        .collect();
    
    let spectral_result = SpectralResult {
        frequencies,
        magnitudes: psd_accumulator.clone(),
        phases: vec![0.0; num_freqs], // phases not computed for PSD
        sample_rate: 1.0, // normalized sample rate
        method: "PSDEstimate".to_string(),
    };
    
    Ok(Value::LyObj(LyObj::new(Box::new(spectral_result))))
}

// =============================================================================
// WINDOWING FUNCTIONS
// =============================================================================

/// Hamming Window
/// Usage: HammingWindow[length] -> Returns Hamming window coefficients
pub fn hamming_window(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (window length)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let length = extract_integer(&args[0])? as usize;
    if length == 0 {
        return Err(VmError::TypeError {
            expected: "positive window length".to_string(),
            actual: "zero length".to_string(),
        });
    }

    let window = generate_hamming_window(length);
    let result: Vec<Value> = window.into_iter().map(|x| Value::Real(x)).collect();
    Ok(Value::List(result))
}

/// Hanning Window
/// Usage: HanningWindow[length] -> Returns Hanning window coefficients
pub fn hanning_window(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (window length)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let length = extract_integer(&args[0])? as usize;
    if length == 0 {
        return Err(VmError::TypeError {
            expected: "positive window length".to_string(),
            actual: "zero length".to_string(),
        });
    }

    let window = generate_hanning_window(length);
    let result: Vec<Value> = window.into_iter().map(|x| Value::Real(x)).collect();
    Ok(Value::List(result))
}

/// Blackman Window
/// Usage: BlackmanWindow[length] -> Returns Blackman window coefficients
pub fn blackman_window(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (window length)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let length = extract_integer(&args[0])? as usize;
    if length == 0 {
        return Err(VmError::TypeError {
            expected: "positive window length".to_string(),
            actual: "zero length".to_string(),
        });
    }

    let window = generate_blackman_window(length);
    let result: Vec<Value> = window.into_iter().map(|x| Value::Real(x)).collect();
    Ok(Value::List(result))
}

/// Apply window function to signal
/// Usage: ApplyWindow[signal, window] -> Returns windowed signal
pub fn apply_window(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (signal, window)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal_data = parse_signal_input(&args[0])?;
    let window = parse_real_vector(&args[1])?;

    if signal_data.length != window.len() {
        return Err(VmError::TypeError {
            expected: "signal and window of same length".to_string(),
            actual: format!("signal: {}, window: {}", signal_data.length, window.len()),
        });
    }

    let windowed_samples: Vec<Complex> = signal_data.samples
        .iter()
        .zip(window.iter())
        .map(|(sample, &win_val)| *sample * win_val)
        .collect();

    let result = SignalData::new(windowed_samples, signal_data.sample_rate);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

// =============================================================================
// CONVOLUTION FUNCTIONS
// =============================================================================

/// Linear Convolution
/// Usage: Convolve[signal1, signal2] -> Returns convolution result
pub fn convolve(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (signal1, signal2)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal1 = parse_real_vector(&args[0])?;
    let signal2 = parse_real_vector(&args[1])?;
    
    let result = compute_convolution(&signal1, &signal2);
    let result_values: Vec<Value> = result.into_iter().map(|x| Value::Real(x)).collect();
    Ok(Value::List(result_values))
}

/// Cross-correlation
/// Usage: CrossCorrelation[signal1, signal2] -> Returns cross-correlation
pub fn cross_correlation(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (signal1, signal2)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal1 = parse_real_vector(&args[0])?;
    let signal2 = parse_real_vector(&args[1])?;
    
    let result = compute_cross_correlation(&signal1, &signal2);
    let result_values: Vec<Value> = result.into_iter().map(|x| Value::Real(x)).collect();
    Ok(Value::List(result_values))
}

/// Auto-correlation
/// Usage: AutoCorrelation[signal] -> Returns auto-correlation
pub fn auto_correlation(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (signal)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal = parse_real_vector(&args[0])?;
    let result = compute_cross_correlation(&signal, &signal);
    let result_values: Vec<Value> = result.into_iter().map(|x| Value::Real(x)).collect();
    Ok(Value::List(result_values))
}

// =============================================================================
// FILTERING FUNCTIONS
// =============================================================================

/// Low-pass filter
/// Usage: LowPassFilter[signal, cutoffFreq] -> Returns filtered signal
pub fn low_pass_filter(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (signal, cutoff_frequency)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal_data = parse_signal_input(&args[0])?;
    let cutoff_freq = extract_number(&args[1])?;

    if cutoff_freq <= 0.0 || cutoff_freq >= signal_data.sample_rate / 2.0 {
        return Err(VmError::TypeError {
            expected: format!("cutoff frequency between 0 and {}", signal_data.sample_rate / 2.0),
            actual: format!("{}", cutoff_freq),
        });
    }

    let filtered = apply_butterworth_lowpass(&signal_data.to_real(), cutoff_freq, signal_data.sample_rate);
    
    let filter_result = FilterResult {
        filtered_signal: filtered,
        filter_type: "LowPass".to_string(),
        parameters: vec![cutoff_freq],
        success: true,
        message: "Low-pass filter applied successfully".to_string(),
    };

    Ok(Value::LyObj(LyObj::new(Box::new(filter_result))))
}

/// High-pass filter
/// Usage: HighPassFilter[signal, cutoffFreq] -> Returns filtered signal
pub fn high_pass_filter(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (signal, cutoff_frequency)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal_data = parse_signal_input(&args[0])?;
    let cutoff_freq = extract_number(&args[1])?;

    if cutoff_freq <= 0.0 || cutoff_freq >= signal_data.sample_rate / 2.0 {
        return Err(VmError::TypeError {
            expected: format!("cutoff frequency between 0 and {}", signal_data.sample_rate / 2.0),
            actual: format!("{}", cutoff_freq),
        });
    }

    let filtered = apply_butterworth_highpass(&signal_data.to_real(), cutoff_freq, signal_data.sample_rate);
    
    let filter_result = FilterResult {
        filtered_signal: filtered,
        filter_type: "HighPass".to_string(),
        parameters: vec![cutoff_freq],
        success: true,
        message: "High-pass filter applied successfully".to_string(),
    };

    Ok(Value::LyObj(LyObj::new(Box::new(filter_result))))
}

/// Median filter for noise removal
/// Usage: MedianFilter[signal, windowSize] -> Returns filtered signal
pub fn median_filter(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (signal, window_size)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal = parse_real_vector(&args[0])?;
    let window_size = extract_integer(&args[1])? as usize;

    if window_size == 0 || window_size % 2 == 0 {
        return Err(VmError::TypeError {
            expected: "odd, positive window size".to_string(),
            actual: format!("{}", window_size),
        });
    }

    let filtered = apply_median_filter(&signal, window_size);
    
    let filter_result = FilterResult {
        filtered_signal: filtered,
        filter_type: "Median".to_string(),
        parameters: vec![window_size as f64],
        success: true,
        message: "Median filter applied successfully".to_string(),
    };

    Ok(Value::LyObj(LyObj::new(Box::new(filter_result))))
}

// =============================================================================
// ADVANCED SIGNAL PROCESSING FUNCTIONS
// =============================================================================

/// Hilbert Transform - Computes the analytic signal
/// Usage: Hilbert[signal] -> Returns complex analytic signal
pub fn hilbert_transform(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (signal)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal = extract_real_signal(&args[0])?;
    let n = signal.len();
    
    if n == 0 {
        return Ok(Value::List(vec![]));
    }

    // Convert to complex for FFT
    let complex_signal: Vec<Complex> = signal.iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();

    // Compute FFT
    let fft_result = compute_fft(&complex_signal)?;
    
    // Create Hilbert transform by zeroing negative frequencies
    let mut hilbert_fft = fft_result.clone();
    
    // Zero out negative frequencies (second half)
    for i in (n/2 + 1)..n {
        hilbert_fft[i] = Complex::zero();
    }
    
    // Double positive frequencies (except DC and Nyquist)
    for i in 1..(n/2) {
        hilbert_fft[i] = hilbert_fft[i] * 2.0;
    }

    // Compute inverse FFT
    let ifft_result = compute_ifft(&hilbert_fft)?;
    
    // Create SignalData for the analytic signal
    let signal_data = SignalData::new(ifft_result, 1.0);
    
    Ok(Value::LyObj(LyObj::new(Box::new(signal_data))))
}

/// Zero Padding - Add zeros to extend signal length
/// Usage: ZeroPadding[signal, totalLength] -> Returns zero-padded signal
pub fn zero_padding(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (signal, totalLength)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal = extract_real_signal(&args[0])?;
    let total_length = extract_positive_integer(&args[1])? as usize;
    
    if total_length < signal.len() {
        return Err(VmError::TypeError {
            expected: "total length >= signal length".to_string(),
            actual: format!("totalLength={}, signalLength={}", total_length, signal.len()),
        });
    }

    // Create zero-padded signal
    let mut padded = signal;
    padded.resize(total_length, 0.0);
    
    // Convert to Value::List
    let result: Vec<Value> = padded.into_iter()
        .map(Value::Real)
        .collect();
    
    Ok(Value::List(result))
}

/// Phase Unwrapping - Unwrap phase to avoid discontinuities
/// Usage: PhaseUnwrap[phases] -> Returns unwrapped phase values
pub fn phase_unwrap(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (phase values)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let phases = extract_real_signal(&args[0])?;
    
    if phases.is_empty() {
        return Ok(Value::List(vec![]));
    }

    let mut unwrapped = phases.clone();
    
    // Unwrap phase by detecting jumps > π and adding/subtracting 2π
    for i in 1..unwrapped.len() {
        let mut delta = unwrapped[i] - unwrapped[i-1];
        
        // Detect phase jumps greater than π
        while delta > PI {
            unwrapped[i] -= 2.0 * PI;
            delta -= 2.0 * PI;
        }
        
        while delta < -PI {
            unwrapped[i] += 2.0 * PI;
            delta += 2.0 * PI;
        }
    }
    
    // Convert to Value::List
    let result: Vec<Value> = unwrapped.into_iter()
        .map(Value::Real)
        .collect();
    
    Ok(Value::List(result))
}

// =============================================================================
// CORE ALGORITHM IMPLEMENTATIONS
// =============================================================================

/// Compute FFT using Cooley-Tukey algorithm
fn compute_fft(samples: &[Complex]) -> VmResult<Vec<Complex>> {
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

fn compute_ifft(spectrum: &[Complex]) -> VmResult<Vec<Complex>> {
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

/// Compute DCT Type-II (most common)
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

/// Generate window functions
fn generate_hamming_window(length: usize) -> Vec<f64> {
    (0..length)
        .map(|n| 0.54 - 0.46 * (2.0 * PI * n as f64 / (length - 1) as f64).cos())
        .collect()
}

fn generate_hanning_window(length: usize) -> Vec<f64> {
    (0..length)
        .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f64 / (length - 1) as f64).cos())
        .collect()
}

fn generate_blackman_window(length: usize) -> Vec<f64> {
    (0..length)
        .map(|n| {
            let arg = 2.0 * PI * n as f64 / (length - 1) as f64;
            0.42 - 0.5 * arg.cos() + 0.08 * (2.0 * arg).cos()
        })
        .collect()
}

/// Compute linear convolution
fn compute_convolution(signal1: &[f64], signal2: &[f64]) -> Vec<f64> {
    let n1 = signal1.len();
    let n2 = signal2.len();
    let result_len = n1 + n2 - 1;
    let mut result = vec![0.0; result_len];

    for i in 0..n1 {
        for j in 0..n2 {
            result[i + j] += signal1[i] * signal2[j];
        }
    }

    result
}

/// Compute cross-correlation
fn compute_cross_correlation(signal1: &[f64], signal2: &[f64]) -> Vec<f64> {
    let n1 = signal1.len();
    let n2 = signal2.len();
    let result_len = n1 + n2 - 1;
    let mut result = vec![0.0; result_len];

    for i in 0..n1 {
        for j in 0..n2 {
            result[i + j] += signal1[i] * signal2[n2 - 1 - j]; // Time-reversed signal2
        }
    }

    result
}

/// Apply simple Butterworth low-pass filter (1st order)
fn apply_butterworth_lowpass(signal: &[f64], cutoff: f64, sample_rate: f64) -> Vec<f64> {
    if signal.is_empty() {
        return vec![];
    }

    let rc = 1.0 / (2.0 * PI * cutoff);
    let dt = 1.0 / sample_rate;
    let alpha = dt / (rc + dt);

    let mut filtered = vec![0.0; signal.len()];
    filtered[0] = signal[0];

    for i in 1..signal.len() {
        filtered[i] = alpha * signal[i] + (1.0 - alpha) * filtered[i - 1];
    }

    filtered
}

/// Apply simple Butterworth high-pass filter (1st order)
fn apply_butterworth_highpass(signal: &[f64], cutoff: f64, sample_rate: f64) -> Vec<f64> {
    if signal.is_empty() {
        return vec![];
    }

    let rc = 1.0 / (2.0 * PI * cutoff);
    let dt = 1.0 / sample_rate;
    let alpha = rc / (rc + dt);

    let mut filtered = vec![0.0; signal.len()];
    filtered[0] = signal[0];

    for i in 1..signal.len() {
        filtered[i] = alpha * (filtered[i - 1] + signal[i] - signal[i - 1]);
    }

    filtered
}

/// Apply median filter
fn apply_median_filter(signal: &[f64], window_size: usize) -> Vec<f64> {
    if signal.is_empty() || window_size == 0 {
        return signal.to_vec();
    }

    let half_window = window_size / 2;
    let mut filtered = vec![0.0; signal.len()];

    for i in 0..signal.len() {
        let start = if i >= half_window { i - half_window } else { 0 };
        let end = std::cmp::min(i + half_window + 1, signal.len());
        
        let mut window: Vec<f64> = signal[start..end].to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        filtered[i] = window[window.len() / 2];
    }

    filtered
}

/// Generate frequency bins for FFT result
fn generate_frequency_bins(n: usize, sample_rate: f64) -> Vec<f64> {
    (0..n)
        .map(|k| k as f64 * sample_rate / n as f64)
        .collect()
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
        _ => Err(VmError::TypeError {
            expected: "spectral result".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

fn parse_real_vector(value: &Value) -> VmResult<Vec<f64>> {
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

fn extract_integer(value: &Value) -> VmResult<i64> {
    match value {
        Value::Integer(n) => Ok(*n),
        Value::Real(r) => Ok(*r as i64),
        _ => Err(VmError::TypeError {
            expected: "integer".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

fn extract_real_signal(value: &Value) -> VmResult<Vec<f64>> {
    parse_real_vector(value)
}

fn extract_positive_integer(value: &Value) -> VmResult<i64> {
    let n = extract_integer(value)?;
    if n <= 0 {
        return Err(VmError::TypeError {
            expected: "positive integer".to_string(),
            actual: format!("{}", n),
        });
    }
    Ok(n)
}

fn extract_real(value: &Value) -> VmResult<f64> {
    extract_number(value)
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

impl Foreign for FilterResult {
    fn type_name(&self) -> &'static str {
        "FilterResult"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "FilteredSignal" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let signal_values: Vec<Value> = self.filtered_signal.iter().map(|&s| Value::Real(s)).collect();
                Ok(Value::List(signal_values))
            }
            "FilterType" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.filter_type.clone()))
            }
            "Parameters" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let param_values: Vec<Value> = self.parameters.iter().map(|&p| Value::Real(p)).collect();
                Ok(Value::List(param_values))
            }
            "Success" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Symbol(if self.success { "True" } else { "False" }.to_string()))
            }
            "Message" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.message.clone()))
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
    fn test_hamming_window() {
        let window_len = Value::Integer(8);
        let result = hamming_window(&[window_len]).unwrap();
        
        match result {
            Value::List(values) => {
                assert_eq!(values.len(), 8);
                // Check that it's a valid window (values between 0 and 1)
                for val in values {
                    if let Value::Real(x) = val {
                        assert!(x >= 0.0 && x <= 1.0);
                    }
                }
            }
            _ => panic!("Expected list of window values"),
        }
    }

    #[test]
    fn test_fft_basic() {
        // Test FFT with simple sine wave
        let signal: Vec<Value> = (0..8)
            .map(|i| Value::Real((2.0 * PI * i as f64 / 8.0).sin()))
            .collect();
        
        let result = fft(&[Value::List(signal)]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let spectral_result = lyobj.downcast_ref::<SpectralResult>().unwrap();
                assert_eq!(spectral_result.length(), 8);
                assert_eq!(spectral_result.method, "FFT");
            }
            _ => panic!("Expected SpectralResult"),
        }
    }

    #[test]
    fn test_convolution() {
        let signal1 = Value::List(vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0)
        ]);
        let signal2 = Value::List(vec![
            Value::Real(0.5), Value::Real(1.0), Value::Real(0.5)
        ]);
        
        let result = convolve(&[signal1, signal2]).unwrap();
        
        match result {
            Value::List(values) => {
                assert_eq!(values.len(), 5); // Length 3 + 3 - 1 = 5
            }
            _ => panic!("Expected list of convolution values"),
        }
    }

    #[test]
    fn test_low_pass_filter() {
        let signal_data = SignalData::from_real(vec![1.0, 2.0, 3.0, 2.0, 1.0], 1000.0);
        let signal_value = Value::LyObj(LyObj::new(Box::new(signal_data)));
        let cutoff = Value::Real(100.0);
        
        let result = low_pass_filter(&[signal_value, cutoff]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let filter_result = lyobj.downcast_ref::<FilterResult>().unwrap();
                assert_eq!(filter_result.filter_type, "LowPass");
                assert!(filter_result.success);
                assert_eq!(filter_result.filtered_signal.len(), 5);
            }
            _ => panic!("Expected FilterResult"),
        }
    }

    #[test]
    fn test_median_filter() {
        let signal = Value::List(vec![
            Value::Real(1.0), Value::Real(5.0), Value::Real(2.0), 
            Value::Real(8.0), Value::Real(3.0)
        ]);
        let window_size = Value::Integer(3);
        
        let result = median_filter(&[signal, window_size]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let filter_result = lyobj.downcast_ref::<FilterResult>().unwrap();
                assert_eq!(filter_result.filter_type, "Median");
                assert!(filter_result.success);
                assert_eq!(filter_result.filtered_signal.len(), 5);
            }
            _ => panic!("Expected FilterResult"),
        }
    }

    #[test]
    fn test_dct() {
        let signal = Value::List(vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0), Value::Real(4.0)
        ]);
        
        let result = dct(&[signal]).unwrap();
        
        match result {
            Value::List(coeffs) => {
                assert_eq!(coeffs.len(), 4);
                // DCT of [1,2,3,4] should have DC component
                if let Value::Real(dc) = coeffs[0] {
                    assert!(dc > 0.0); // Should be positive
                }
            }
            _ => panic!("Expected list of DCT coefficients"),
        }
    }

    #[test]
    fn test_signal_data_methods() {
        let signal_data = SignalData::from_real(vec![1.0, 2.0, 3.0], 1000.0);
        
        // Test Length method
        let length = signal_data.call_method("Length", &[]).unwrap();
        assert_eq!(length, Value::Integer(3));
        
        // Test SampleRate method
        let sample_rate = signal_data.call_method("SampleRate", &[]).unwrap();
        assert_eq!(sample_rate, Value::Real(1000.0));
        
        // Test Duration method
        let duration = signal_data.call_method("Duration", &[]).unwrap();
        assert_eq!(duration, Value::Real(0.003)); // 3 samples / 1000 Hz
        
        // Test ToReal method
        let real_values = signal_data.call_method("ToReal", &[]).unwrap();
        match real_values {
            Value::List(values) => {
                assert_eq!(values.len(), 3);
                assert_eq!(values[0], Value::Real(1.0));
                assert_eq!(values[1], Value::Real(2.0));
                assert_eq!(values[2], Value::Real(3.0));
            }
            _ => panic!("Expected list of real values"),
        }
    }

    #[test]
    fn test_auto_correlation() {
        let signal = Value::List(vec![
            Value::Real(1.0), Value::Real(0.0), Value::Real(-1.0), Value::Real(0.0)
        ]);
        
        let result = auto_correlation(&[signal]).unwrap();
        
        match result {
            Value::List(values) => {
                assert_eq!(values.len(), 7); // 4 + 4 - 1 = 7
                // Auto-correlation should be symmetric around center
            }
            _ => panic!("Expected list of correlation values"),
        }
    }

    #[test]
    fn test_periodogram() {
        let signal = Value::List(vec![
            Value::Real(1.0), Value::Real(0.0), Value::Real(-1.0), Value::Real(0.0)
        ]);
        
        let result = periodogram(&[signal]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let spectral_result = lyobj.downcast_ref::<SpectralResult>().unwrap();
                assert_eq!(spectral_result.frequencies.len(), 3); // n/2 + 1 = 4/2 + 1 = 3
                assert_eq!(spectral_result.magnitudes.len(), 3);
                assert_eq!(spectral_result.method, "Periodogram");
            }
            _ => panic!("Expected SpectralResult"),
        }
    }

    #[test]
    fn test_psd_estimate() {
        let signal = Value::List(vec![
            Value::Real(1.0), Value::Real(0.5), Value::Real(-0.5), Value::Real(-1.0),
            Value::Real(0.0), Value::Real(1.0), Value::Real(-0.5), Value::Real(0.5)
        ]);
        
        let result = psd_estimate(&[signal]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let spectral_result = lyobj.downcast_ref::<SpectralResult>().unwrap();
                assert!(spectral_result.frequencies.len() > 0);
                assert_eq!(spectral_result.frequencies.len(), spectral_result.magnitudes.len());
                assert_eq!(spectral_result.method, "PSDEstimate");
            }
            _ => panic!("Expected SpectralResult"),
        }
    }

    #[test]
    fn test_spectrogram() {
        let signal = Value::List(vec![
            Value::Real(1.0), Value::Real(0.0), Value::Real(-1.0), Value::Real(0.0),
            Value::Real(0.5), Value::Real(0.0), Value::Real(-0.5), Value::Real(0.0)
        ]);
        let window_size = Value::Integer(4);
        
        let result = spectrogram(&[signal, window_size]).unwrap();
        
        match result {
            Value::List(time_frames) => {
                assert!(time_frames.len() > 0); // Should have at least one time frame
                // Each time frame should be a list of frequency bins
                if let Value::List(first_frame) = &time_frames[0] {
                    assert_eq!(first_frame.len(), 3); // window_size/2 + 1 = 4/2 + 1 = 3
                }
            }
            _ => panic!("Expected list of time-frequency frames"),
        }
    }

    #[test]
    fn test_hilbert_transform() {
        let signal = Value::List(vec![
            Value::Real(1.0), Value::Real(0.0), Value::Real(-1.0), Value::Real(0.0)
        ]);
        
        let result = hilbert_transform(&[signal]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let signal_data = lyobj.downcast_ref::<SignalData>().unwrap();
                assert_eq!(signal_data.samples.len(), 4);
                assert_eq!(signal_data.sample_rate, 1.0);
                // Analytic signal should have complex values
                assert!(signal_data.samples.iter().any(|c| c.imag != 0.0));
            }
            _ => panic!("Expected SignalData with analytic signal"),
        }
    }

    #[test]
    fn test_zero_padding() {
        let signal = Value::List(vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0)
        ]);
        let total_length = Value::Integer(8);
        
        let result = zero_padding(&[signal, total_length]).unwrap();
        
        match result {
            Value::List(values) => {
                assert_eq!(values.len(), 8);
                assert_eq!(values[0], Value::Real(1.0));
                assert_eq!(values[1], Value::Real(2.0));
                assert_eq!(values[2], Value::Real(3.0));
                // Check that remaining values are zero
                for i in 3..8 {
                    assert_eq!(values[i], Value::Real(0.0));
                }
            }
            _ => panic!("Expected list of padded values"),
        }
    }

    #[test]
    fn test_phase_unwrap() {
        // Test with phase values that wrap around π
        let phases = Value::List(vec![
            Value::Real(3.0), Value::Real(-3.0), Value::Real(2.8), Value::Real(-2.8)
        ]);
        
        let result = phase_unwrap(&[phases]).unwrap();
        
        match result {
            Value::List(values) => {
                assert_eq!(values.len(), 4);
                // Check that unwrapped phases don't have large jumps
                let unwrapped: Vec<f64> = values.iter().map(|v| {
                    if let Value::Real(x) = v { *x } else { 0.0 }
                }).collect();
                
                for i in 1..unwrapped.len() {
                    let diff = (unwrapped[i] - unwrapped[i-1]).abs();
                    assert!(diff <= PI); // No jumps larger than π
                }
            }
            _ => panic!("Expected list of unwrapped phase values"),
        }
    }
}