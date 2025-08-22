//! Signal Processing & FFT Module for Lyra
//!
//! This module provides comprehensive digital signal processing capabilities including:
//! - Fast Fourier Transform algorithms (FFT, IFFT, Real FFT)
//! - Signal processing functions (convolution, correlation, filtering)
//! - Window functions (Hamming, Hanning, Blackman, Kaiser)
//! - Digital filters (FIR, IIR, low-pass, high-pass, band-pass)
//! - Spectral analysis (periodogram, spectrogram, PSD estimation)
//! - Signal generation (sine waves, noise, chirp signals)
//! - Advanced processing (Hilbert transform, envelope detection, phase unwrapping)
//!
//! All complex signal data structures are implemented as Foreign objects following
//! the Foreign object pattern to maintain VM simplicity.

pub mod fft;
pub mod processing;

// Re-export all public functions for easy access
pub use fft::*;
pub use processing::*;

use crate::vm::{Value, VmResult};
use std::collections::HashMap;

/// Registry function for all signal processing functions
pub fn register_signal_functions() -> HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut functions = HashMap::new();
    
    // FFT and Transform Functions
    functions.insert("FFT".to_string(), fft::fft as fn(&[Value]) -> VmResult<Value>);
    functions.insert("IFFT".to_string(), fft::ifft as fn(&[Value]) -> VmResult<Value>);
    functions.insert("RealFFT".to_string(), fft::real_fft as fn(&[Value]) -> VmResult<Value>);
    functions.insert("DCT".to_string(), fft::dct as fn(&[Value]) -> VmResult<Value>);
    functions.insert("IDCT".to_string(), fft::idct as fn(&[Value]) -> VmResult<Value>);
    
    // Convolution and Correlation
    functions.insert("Convolve".to_string(), processing::convolve as fn(&[Value]) -> VmResult<Value>);
    functions.insert("CircularConvolve".to_string(), processing::circular_convolve as fn(&[Value]) -> VmResult<Value>);
    functions.insert("CrossCorrelate".to_string(), processing::cross_correlate as fn(&[Value]) -> VmResult<Value>);
    functions.insert("AutoCorrelate".to_string(), processing::auto_correlate as fn(&[Value]) -> VmResult<Value>);
    
    // Digital Filtering
    functions.insert("FIRFilter".to_string(), processing::fir_filter as fn(&[Value]) -> VmResult<Value>);
    functions.insert("IIRFilter".to_string(), processing::iir_filter as fn(&[Value]) -> VmResult<Value>);
    functions.insert("LowPassFilter".to_string(), processing::low_pass_filter as fn(&[Value]) -> VmResult<Value>);
    functions.insert("HighPassFilter".to_string(), processing::high_pass_filter as fn(&[Value]) -> VmResult<Value>);
    functions.insert("BandPassFilter".to_string(), processing::band_pass_filter as fn(&[Value]) -> VmResult<Value>);
    functions.insert("MedianFilter".to_string(), processing::median_filter as fn(&[Value]) -> VmResult<Value>);
    
    // Window Functions
    functions.insert("HammingWindow".to_string(), processing::hamming_window as fn(&[Value]) -> VmResult<Value>);
    functions.insert("HanningWindow".to_string(), processing::hanning_window as fn(&[Value]) -> VmResult<Value>);
    functions.insert("BlackmanWindow".to_string(), processing::blackman_window as fn(&[Value]) -> VmResult<Value>);
    functions.insert("KaiserWindow".to_string(), processing::kaiser_window as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ApplyWindow".to_string(), processing::apply_window as fn(&[Value]) -> VmResult<Value>);
    
    // Spectral Analysis
    functions.insert("Periodogram".to_string(), processing::periodogram as fn(&[Value]) -> VmResult<Value>);
    functions.insert("WelchPSD".to_string(), processing::welch_psd as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Spectrogram".to_string(), processing::spectrogram as fn(&[Value]) -> VmResult<Value>);
    
    // Signal Generation
    functions.insert("SignalData".to_string(), processing::signal_data as fn(&[Value]) -> VmResult<Value>);
    functions.insert("SineWave".to_string(), processing::sine_wave as fn(&[Value]) -> VmResult<Value>);
    functions.insert("WhiteNoise".to_string(), processing::white_noise as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ChirpSignal".to_string(), processing::chirp_signal as fn(&[Value]) -> VmResult<Value>);
    
    // Advanced Processing
    functions.insert("HilbertTransform".to_string(), processing::hilbert_transform as fn(&[Value]) -> VmResult<Value>);
    functions.insert("EnvelopeDetect".to_string(), processing::envelope_detect as fn(&[Value]) -> VmResult<Value>);
    functions.insert("PhaseUnwrap".to_string(), processing::phase_unwrap as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ZeroPadding".to_string(), processing::zero_padding as fn(&[Value]) -> VmResult<Value>);
    
    functions
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_signal_function_registry() {
        let functions = register_signal_functions();
        
        // Verify core functions are registered
        assert!(functions.contains_key("FFT"));
        assert!(functions.contains_key("IFFT"));
        assert!(functions.contains_key("RealFFT"));
        assert!(functions.contains_key("Convolve"));
        assert!(functions.contains_key("CrossCorrelate"));
        assert!(functions.contains_key("LowPassFilter"));
        assert!(functions.contains_key("HammingWindow"));
        assert!(functions.contains_key("Periodogram"));
        assert!(functions.contains_key("SignalData"));
        assert!(functions.contains_key("HilbertTransform"));
        
        // Should have all expected functions
        assert!(functions.len() >= 25, "Should register at least 25 signal processing functions");
    }
}