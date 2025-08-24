//! Signal processing functions for Lyra
//!
//! This module implements comprehensive signal processing capabilities:
//! - Convolution and correlation operations
//! - Digital filtering (FIR, IIR, frequency domain filters)
//! - Window functions (Hamming, Hanning, Blackman, Kaiser)
//! - Spectral analysis (periodogram, spectrogram, PSD estimation)
//! - Signal generation (sine waves, noise, chirp signals)
//! - Advanced processing (Hilbert transform, envelope detection)

use crate::vm::{Value, VmError, VmResult};
use crate::stdlib::common::result::{spectral_result, filter_result};
use crate::foreign::{Foreign, ForeignError, LyObj};
use super::fft::{SignalData, compute_fft, compute_ifft};
use crate::stdlib::common::Complex;
use std::f64::consts::PI;
use rand::prelude::*;

// Filter results now return Associations instead of Foreign objects

// =============================================================================
// CONVOLUTION AND CORRELATION FUNCTIONS
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

    let signal1 = extract_real_signal(&args[0])?;
    let signal2 = extract_real_signal(&args[1])?;
    
    if signal1.is_empty() || signal2.is_empty() {
        return Err(VmError::TypeError {
            expected: "non-empty signals".to_string(),
            actual: "empty signal".to_string(),
        });
    }
    
    let result = compute_convolution(&signal1, &signal2);
    let result_values: Vec<Value> = result.into_iter().map(|x| Value::Real(x)).collect();
    Ok(Value::List(result_values))
}

/// Circular Convolution
/// Usage: CircularConvolve[signal1, signal2] -> Returns circular convolution result
pub fn circular_convolve(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (signal1, signal2)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal1 = extract_real_signal(&args[0])?;
    let signal2 = extract_real_signal(&args[1])?;
    
    if signal1.len() != signal2.len() {
        return Err(VmError::TypeError {
            expected: "signals of same length".to_string(),
            actual: format!("signal1: {}, signal2: {}", signal1.len(), signal2.len()),
        });
    }
    
    let result = compute_circular_convolution(&signal1, &signal2)?;
    let result_values: Vec<Value> = result.into_iter().map(|x| Value::Real(x)).collect();
    Ok(Value::List(result_values))
}

/// Cross-correlation
/// Usage: CrossCorrelate[signal1, signal2] -> Returns cross-correlation
pub fn cross_correlate(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (signal1, signal2)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal1 = extract_real_signal(&args[0])?;
    let signal2 = extract_real_signal(&args[1])?;
    
    let result = compute_cross_correlation(&signal1, &signal2);
    let result_values: Vec<Value> = result.into_iter().map(|x| Value::Real(x)).collect();
    Ok(Value::List(result_values))
}

/// Auto-correlation
/// Usage: AutoCorrelate[signal] -> Returns auto-correlation
pub fn auto_correlate(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (signal)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal = extract_real_signal(&args[0])?;
    let result = compute_cross_correlation(&signal, &signal);
    let result_values: Vec<Value> = result.into_iter().map(|x| Value::Real(x)).collect();
    Ok(Value::List(result_values))
}

// =============================================================================
// DIGITAL FILTERING FUNCTIONS
// =============================================================================

/// FIR Filter
/// Usage: FIRFilter[signal, coefficients] -> Returns FilterResult
pub fn fir_filter(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (signal, coefficients)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal = extract_real_signal(&args[0])?;
    let coefficients = extract_real_signal(&args[1])?;
    
    if coefficients.is_empty() {
        return Err(VmError::TypeError {
            expected: "non-empty coefficient array".to_string(),
            actual: "empty coefficients".to_string(),
        });
    }
    
    let filtered = apply_fir_filter(&signal, &coefficients);
    Ok(filter_result("FIR", coefficients.into_iter().map(Value::Real).collect(), filtered, true, "Filter applied successfully"))
}

/// IIR Filter
/// Usage: IIRFilter[signal, b_coefficients, a_coefficients] -> Returns FilterResult
pub fn iir_filter(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "exactly 3 arguments (signal, b_coefficients, a_coefficients)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal = extract_real_signal(&args[0])?;
    let b_coeffs = extract_real_signal(&args[1])?;
    let a_coeffs = extract_real_signal(&args[2])?;
    
    if b_coeffs.is_empty() || a_coeffs.is_empty() {
        return Err(VmError::TypeError {
            expected: "non-empty coefficient arrays".to_string(),
            actual: "empty coefficients".to_string(),
        });
    }
    
    if a_coeffs[0] == 0.0 {
        return Err(VmError::TypeError {
            expected: "non-zero first a-coefficient".to_string(),
            actual: "zero a[0]".to_string(),
        });
    }
    
    let filtered = apply_iir_filter(&signal, &b_coeffs, &a_coeffs);
    let mut params = b_coeffs.clone();
    params.extend(&a_coeffs);
    Ok(filter_result("IIR", params.into_iter().map(Value::Real).collect(), filtered, true, "Filter applied successfully"))
}

/// Low-pass filter
/// Usage: LowPassFilter[signal_data, cutoff_frequency] -> Returns FilterResult
pub fn low_pass_filter(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (signal_data, cutoff_frequency)".to_string(),
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
    Ok(filter_result("LowPass", vec![Value::Real(cutoff_freq)], filtered, true, "Filter applied successfully"))
}

/// High-pass filter
/// Usage: HighPassFilter[signal_data, cutoff_frequency] -> Returns FilterResult
pub fn high_pass_filter(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (signal_data, cutoff_frequency)".to_string(),
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
    Ok(filter_result("HighPass", vec![Value::Real(cutoff_freq)], filtered, true, "Filter applied successfully"))
}

/// Band-pass filter
/// Usage: BandPassFilter[signal_data, low_frequency, high_frequency] -> Returns FilterResult
pub fn band_pass_filter(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "exactly 3 arguments (signal_data, low_frequency, high_frequency)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal_data = parse_signal_input(&args[0])?;
    let low_freq = extract_number(&args[1])?;
    let high_freq = extract_number(&args[2])?;

    if low_freq <= 0.0 || high_freq <= low_freq || high_freq >= signal_data.sample_rate / 2.0 {
        return Err(VmError::TypeError {
            expected: "0 < low_freq < high_freq < Nyquist".to_string(),
            actual: format!("low: {}, high: {}, Nyquist: {}", low_freq, high_freq, signal_data.sample_rate / 2.0),
        });
    }

    let filtered = apply_butterworth_bandpass(&signal_data.to_real(), low_freq, high_freq, signal_data.sample_rate);
    let mut m = std::collections::HashMap::new();
    m.insert("filterType".to_string(), Value::String("BandPass".to_string()));
    m.insert("parameters".to_string(), Value::List(vec![Value::Real(low_freq), Value::Real(high_freq)]));
    m.insert("success".to_string(), Value::Boolean(true));
    m.insert("message".to_string(), Value::String("Filter applied successfully".to_string()));
    m.insert("filteredSignal".to_string(), Value::List(filtered.into_iter().map(Value::Real).collect()));
    Ok(Value::Object(m))
}

/// Median filter for noise removal
/// Usage: MedianFilter[signal, window_size] -> Returns FilterResult
pub fn median_filter(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (signal, window_size)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal = extract_real_signal(&args[0])?;
    let window_size = extract_integer(&args[1])? as usize;

    if window_size == 0 || window_size % 2 == 0 {
        return Err(VmError::TypeError {
            expected: "odd, positive window size".to_string(),
            actual: format!("{}", window_size),
        });
    }

    let filtered = apply_median_filter(&signal, window_size);
    let mut m = std::collections::HashMap::new();
    m.insert("filterType".to_string(), Value::String("Median".to_string()));
    m.insert("parameters".to_string(), Value::List(vec![Value::Real(window_size as f64)]));
    m.insert("success".to_string(), Value::Boolean(true));
    m.insert("message".to_string(), Value::String("Filter applied successfully".to_string()));
    m.insert("filteredSignal".to_string(), Value::List(filtered.into_iter().map(Value::Real).collect()));
    Ok(Value::Object(m))
}

// =============================================================================
// WINDOW FUNCTION IMPLEMENTATIONS
// =============================================================================

/// Hamming Window
/// Usage: HammingWindow[length] -> Returns window coefficients
pub fn hamming_window(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (window length)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let length = extract_positive_integer(&args[0])? as usize;
    let window = generate_hamming_window(length);
    let result: Vec<Value> = window.into_iter().map(|x| Value::Real(x)).collect();
    Ok(Value::List(result))
}

/// Hanning Window
/// Usage: HanningWindow[length] -> Returns window coefficients
pub fn hanning_window(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (window length)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let length = extract_positive_integer(&args[0])? as usize;
    let window = generate_hanning_window(length);
    let result: Vec<Value> = window.into_iter().map(|x| Value::Real(x)).collect();
    Ok(Value::List(result))
}

/// Blackman Window
/// Usage: BlackmanWindow[length] -> Returns window coefficients
pub fn blackman_window(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (window length)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let length = extract_positive_integer(&args[0])? as usize;
    let window = generate_blackman_window(length);
    let result: Vec<Value> = window.into_iter().map(|x| Value::Real(x)).collect();
    Ok(Value::List(result))
}

/// Kaiser Window
/// Usage: KaiserWindow[length, beta] -> Returns window coefficients
pub fn kaiser_window(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (length, beta)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let length = extract_positive_integer(&args[0])? as usize;
    let beta = extract_number(&args[1])?;
    
    if beta < 0.0 {
        return Err(VmError::TypeError {
            expected: "non-negative beta parameter".to_string(),
            actual: format!("{}", beta),
        });
    }
    
    let window = generate_kaiser_window(length, beta);
    let result: Vec<Value> = window.into_iter().map(|x| Value::Real(x)).collect();
    Ok(Value::List(result))
}

/// Apply window function to signal
/// Usage: ApplyWindow[signal, window] -> Returns windowed SignalData
pub fn apply_window(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (signal, window)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let signal_data = parse_signal_input(&args[0])?;
    let window = extract_real_signal(&args[1])?;

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
    Ok(spectral_result(frequencies, psd, vec![0.0; (n/2)+1], 1.0, "Periodogram"))
}

/// Welch's method for PSD estimation
/// Usage: WelchPSD[signal, segment_length, overlap] -> Returns SpectralResult
pub fn welch_psd(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "1-3 arguments (signal, optional segment_length, optional overlap)".to_string(),
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
        let overlap_val = extract_number(&args[2])?;
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
    
    Ok(spectral_result(frequencies, psd_accumulator, vec![0.0; num_freqs], 1.0, "WelchPSD"))
}

/// Spectrogram - Short-Time Fourier Transform
/// Usage: Spectrogram[signal, window_size, hop_size] -> Returns time-frequency matrix
pub fn spectrogram(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "1-3 arguments (signal, optional window_size, optional hop_size)".to_string(),
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
            for (freq_bin, &complex_val) in fft_result[0..num_freqs].iter().enumerate() {
                spectrogram[frame][freq_bin] = complex_val.magnitude();
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

// =============================================================================
// SIGNAL GENERATION FUNCTIONS
// =============================================================================

/// Create SignalData object
/// Usage: SignalData[samples, sample_rate] -> Returns SignalData
pub fn signal_data(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (samples, sample_rate)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let samples = extract_real_signal(&args[0])?;
    let sample_rate = extract_number(&args[1])?;
    
    if sample_rate <= 0.0 {
        return Err(VmError::TypeError {
            expected: "positive sample rate".to_string(),
            actual: format!("{}", sample_rate),
        });
    }
    
    let signal_data = SignalData::from_real(samples, sample_rate);
    Ok(Value::LyObj(LyObj::new(Box::new(signal_data))))
}

/// Generate sine wave
/// Usage: SineWave[frequency, duration, sample_rate, amplitude] -> Returns SignalData
pub fn sine_wave(args: &[Value]) -> VmResult<Value> {
    if args.len() != 4 {
        return Err(VmError::TypeError {
            expected: "exactly 4 arguments (frequency, duration, sample_rate, amplitude)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let frequency = extract_number(&args[0])?;
    let duration = extract_number(&args[1])?;
    let sample_rate = extract_number(&args[2])?;
    let amplitude = extract_number(&args[3])?;
    
    if frequency < 0.0 || duration <= 0.0 || sample_rate <= 0.0 {
        return Err(VmError::TypeError {
            expected: "positive parameters".to_string(),
            actual: "negative or zero value".to_string(),
        });
    }
    
    let num_samples = (duration * sample_rate) as usize;
    let samples: Vec<f64> = (0..num_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            amplitude * (2.0 * PI * frequency * t).sin()
        })
        .collect();
    
    let signal_data = SignalData::from_real(samples, sample_rate);
    Ok(Value::LyObj(LyObj::new(Box::new(signal_data))))
}

/// Generate white noise
/// Usage: WhiteNoise[length, amplitude, seed] -> Returns SignalData
pub fn white_noise(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "exactly 3 arguments (length, amplitude, seed)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let length = extract_positive_integer(&args[0])? as usize;
    let amplitude = extract_number(&args[1])?;
    let seed = extract_integer(&args[2])? as u64;
    
    let mut rng = StdRng::seed_from_u64(seed);
    let samples: Vec<f64> = (0..length)
        .map(|_| amplitude * (rng.gen::<f64>() - 0.5) * 2.0)
        .collect();
    
    let signal_data = SignalData::from_real(samples, 1.0); // Default sample rate
    Ok(Value::LyObj(LyObj::new(Box::new(signal_data))))
}

/// Generate chirp signal (frequency sweep)
/// Usage: ChirpSignal[start_freq, end_freq, duration, sample_rate] -> Returns SignalData
pub fn chirp_signal(args: &[Value]) -> VmResult<Value> {
    if args.len() != 4 {
        return Err(VmError::TypeError {
            expected: "exactly 4 arguments (start_freq, end_freq, duration, sample_rate)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let start_freq = extract_number(&args[0])?;
    let end_freq = extract_number(&args[1])?;
    let duration = extract_number(&args[2])?;
    let sample_rate = extract_number(&args[3])?;
    
    if start_freq < 0.0 || end_freq < 0.0 || duration <= 0.0 || sample_rate <= 0.0 {
        return Err(VmError::TypeError {
            expected: "positive parameters".to_string(),
            actual: "negative or zero value".to_string(),
        });
    }
    
    let num_samples = (duration * sample_rate) as usize;
    let samples: Vec<f64> = (0..num_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            let instantaneous_freq = start_freq + (end_freq - start_freq) * t / duration;
            let phase = 2.0 * PI * (start_freq * t + 0.5 * (end_freq - start_freq) * t * t / duration);
            phase.sin()
        })
        .collect();
    
    let signal_data = SignalData::from_real(samples, sample_rate);
    Ok(Value::LyObj(LyObj::new(Box::new(signal_data))))
}

// =============================================================================
// ADVANCED PROCESSING FUNCTIONS
// =============================================================================

/// Hilbert Transform - Computes the analytic signal
/// Usage: HilbertTransform[signal] -> Returns complex SignalData
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

/// Envelope Detection using Hilbert transform
/// Usage: EnvelopeDetect[signal] -> Returns envelope values
pub fn envelope_detect(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (signal)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // Get analytic signal using Hilbert transform
    let analytic_result = hilbert_transform(args)?;
    
    match analytic_result {
        Value::LyObj(lyobj) => {
            if let Some(signal_data) = lyobj.downcast_ref::<SignalData>() {
                let envelope: Vec<Value> = signal_data.to_magnitude()
                    .into_iter()
                    .map(|mag| Value::Real(mag))
                    .collect();
                Ok(Value::List(envelope))
            } else {
                Err(VmError::TypeError {
                    expected: "SignalData from Hilbert transform".to_string(),
                    actual: "invalid result".to_string(),
                })
            }
        }
        _ => Err(VmError::TypeError {
            expected: "SignalData from Hilbert transform".to_string(),
            actual: format!("{:?}", analytic_result),
        }),
    }
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

/// Zero Padding - Add zeros to extend signal length
/// Usage: ZeroPadding[signal, total_length] -> Returns zero-padded signal
pub fn zero_padding(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (signal, total_length)".to_string(),
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

// =============================================================================
// CORE ALGORITHM IMPLEMENTATIONS
// =============================================================================

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

/// Compute circular convolution using FFT
fn compute_circular_convolution(signal1: &[f64], signal2: &[f64]) -> VmResult<Vec<f64>> {
    let n = signal1.len();
    
    // Convert to complex
    let complex1: Vec<Complex> = signal1.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let complex2: Vec<Complex> = signal2.iter().map(|&x| Complex::new(x, 0.0)).collect();
    
    // Compute FFTs
    let fft1 = compute_fft(&complex1)?;
    let fft2 = compute_fft(&complex2)?;
    
    // Pointwise multiplication in frequency domain
    let product: Vec<Complex> = fft1.iter().zip(fft2.iter())
        .map(|(a, b)| *a * *b)
        .collect();
    
    // Inverse FFT
    let ifft_result = compute_ifft(&product)?;
    
    // Extract real part
    Ok(ifft_result.iter().map(|c| c.real).collect())
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

/// Apply FIR filter
fn apply_fir_filter(signal: &[f64], coefficients: &[f64]) -> Vec<f64> {
    let signal_len = signal.len();
    let filter_len = coefficients.len();
    let output_len = signal_len + filter_len - 1;
    let mut output = vec![0.0; output_len];

    for n in 0..output_len {
        for k in 0..filter_len {
            if n >= k && (n - k) < signal_len {
                output[n] += coefficients[k] * signal[n - k];
            }
        }
    }

    output
}

/// Apply IIR filter
fn apply_iir_filter(signal: &[f64], b_coeffs: &[f64], a_coeffs: &[f64]) -> Vec<f64> {
    let signal_len = signal.len();
    let b_len = b_coeffs.len();
    let a_len = a_coeffs.len();
    let mut output = vec![0.0; signal_len];

    for n in 0..signal_len {
        // Forward part (numerator)
        for k in 0..b_len {
            if n >= k {
                output[n] += b_coeffs[k] * signal[n - k];
            }
        }
        
        // Feedback part (denominator, skip a[0])
        for k in 1..a_len {
            if n >= k {
                output[n] -= a_coeffs[k] * output[n - k];
            }
        }
        
        // Normalize by a[0]
        output[n] /= a_coeffs[0];
    }

    output
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

/// Apply band-pass filter (combination of high-pass and low-pass)
fn apply_butterworth_bandpass(signal: &[f64], low_freq: f64, high_freq: f64, sample_rate: f64) -> Vec<f64> {
    // First apply high-pass to remove low frequencies
    let highpass_filtered = apply_butterworth_highpass(signal, low_freq, sample_rate);
    
    // Then apply low-pass to remove high frequencies
    apply_butterworth_lowpass(&highpass_filtered, high_freq, sample_rate)
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

fn generate_kaiser_window(length: usize, beta: f64) -> Vec<f64> {
    let i0_beta = modified_bessel_i0(beta);
    
    (0..length)
        .map(|n| {
            let x = 2.0 * n as f64 / (length - 1) as f64 - 1.0;
            let arg = beta * (1.0 - x * x).sqrt();
            modified_bessel_i0(arg) / i0_beta
        })
        .collect()
}

/// Modified Bessel function of the first kind, order 0
fn modified_bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x_half_squared = (x / 2.0).powi(2);
    
    for k in 1..=20 {
        term *= x_half_squared / (k * k) as f64;
        sum += term;
        if term < 1e-12 {
            break;
        }
    }
    
    sum
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

// No Foreign impls needed for filter results; Associations are returned directly

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convolution() {
        let signal1 = vec![Value::Real(1.0), Value::Real(2.0), Value::Real(3.0)];
        let signal2 = vec![Value::Real(0.5), Value::Real(1.0), Value::Real(0.5)];
        
        let result = convolve(&[Value::List(signal1), Value::List(signal2)]).unwrap();
        
        match result {
            Value::List(values) => {
                assert_eq!(values.len(), 5); // Length 3 + 3 - 1 = 5
            }
            _ => panic!("Expected convolution result list"),
        }
    }

    #[test]
    fn test_hamming_window() {
        let length = Value::Integer(8);
        let result = hamming_window(&[length]).unwrap();
        
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
            _ => panic!("Expected window coefficients list"),
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
            Value::Object(m) => {
                assert_eq!(m.get("filterType"), Some(&Value::String("Median".to_string())));
                assert_eq!(m.get("success"), Some(&Value::Boolean(true)));
            }
            _ => panic!("Expected Association"),
        }
    }

    #[test]
    fn test_sine_wave_generation() {
        let frequency = Value::Real(10.0);
        let duration = Value::Real(1.0);
        let sample_rate = Value::Real(100.0);
        let amplitude = Value::Real(1.0);
        
        let result = sine_wave(&[frequency, duration, sample_rate, amplitude]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                assert_eq!(lyobj.type_name(), "SignalData");
                let length = lyobj.call_method("Length", &[]).unwrap();
                assert_eq!(length, Value::Integer(100));
            }
            _ => panic!("Expected SignalData"),
        }
    }

    #[test]
    fn test_periodogram() {
        let signal = Value::List(vec![
            Value::Real(1.0), Value::Real(0.0), Value::Real(-1.0), Value::Real(0.0)
        ]);
        
        let result = periodogram(&[signal]).unwrap();
        
        match result {
            Value::Object(m) => {
                assert_eq!(m.get("method"), Some(&Value::String("Periodogram".to_string())));
                assert!(m.get("frequencies").is_some());
            }
            _ => panic!("Expected Association"),
        }
    }
}
