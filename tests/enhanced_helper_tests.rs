//! Tests for the enhanced REPL helper architecture
//!
//! This test suite verifies the modular trait-based architecture that provides
//! a foundation for all quality-of-life improvements in the Lyra REPL.

use lyra::repl::{ReplEngine, config::ReplConfig};
use lyra::repl::enhanced_helper::{
    EnhancedLyraHelper, LyraHighlighter, LyraHinter, LyraValidator
};
use rustyline::{
    Helper,
    completion::{Completer, Pair},
    hint::Hinter,
    highlight::Highlighter,
    validate::Validator,
    Context, Result as RustylineResult
};
use std::sync::Arc;
use std::collections::HashMap;

#[test]
fn test_enhanced_helper_creation() {
    // RED: Create test that initially fails
    let repl_engine = ReplEngine::new().expect("Failed to create REPL engine");
    let helper = EnhancedLyraHelper::new(repl_engine.get_config().clone(), repl_engine.create_shared_completer());
    
    // Should be able to create enhanced helper without errors
    assert!(helper.is_ok());
}

#[test]
fn test_enhanced_helper_completion_integration() {
    // RED: Test completion integration
    let repl_engine = ReplEngine::new().expect("Failed to create REPL engine");
    let helper = EnhancedLyraHelper::new(repl_engine.get_config().clone(), repl_engine.create_shared_completer())
        .expect("Failed to create enhanced helper");
    
    let ctx = Context::new();
    
    // Test function completion
    let result = helper.complete("Si", 2, &ctx);
    assert!(result.is_ok());
    let (start, candidates) = result.unwrap();
    assert_eq!(start, 0);
    assert!(candidates.iter().any(|p| p.replacement == "Sin"));
    
    // Test meta command completion
    let result = helper.complete("%h", 2, &ctx);
    assert!(result.is_ok());
    let (start, candidates) = result.unwrap();
    assert_eq!(start, 0);
    assert!(candidates.iter().any(|p| p.replacement == "%help"));
}

#[test]
fn test_highlighter_placeholder() {
    // RED: Test highlighter placeholder functionality
    let highlighter = LyraHighlighter::new();
    
    // Should return the line unchanged for now (placeholder implementation)
    let line = "Sin[Pi/2]";
    let highlighted = highlighter.highlight(line, 0);
    assert_eq!(highlighted.as_ref(), line);
    
    // Character highlighting should be disabled for now
    assert!(!highlighter.highlight_char(line, 0, false));
}

#[test]
fn test_hinter_placeholder() {
    // RED: Test hinter placeholder functionality
    let hinter = LyraHinter::new();
    let ctx = Context::new();
    
    // Should return None for now (placeholder implementation)
    let hint = hinter.hint("Sin[", 4, &ctx);
    assert!(hint.is_none());
}

#[test]
fn test_validator_placeholder() {
    // RED: Test validator placeholder functionality
    let validator = LyraValidator::new();
    let ctx = Context::new();
    
    // Should always validate successfully for now (placeholder implementation)
    let result = validator.validate("incomplete expression [", &ctx);
    assert!(result.is_ok());
}

#[test]
fn test_enhanced_helper_config_integration() {
    // RED: Test configuration integration
    let mut config = ReplConfig::default();
    config.editor.mode = "vi".to_string();
    
    let repl_engine = ReplEngine::new_with_config(config.clone()).expect("Failed to create REPL engine");
    let helper = EnhancedLyraHelper::new(config.clone(), repl_engine.create_shared_completer())
        .expect("Failed to create enhanced helper");
    
    // Configuration should be accessible
    assert_eq!(helper.get_config().editor.mode, "vi");
}

#[test]
fn test_enhanced_helper_runtime_config_changes() {
    // RED: Test runtime configuration changes
    let repl_engine = ReplEngine::new().expect("Failed to create REPL engine");
    let mut helper = EnhancedLyraHelper::new(repl_engine.get_config().clone(), repl_engine.create_shared_completer())
        .expect("Failed to create enhanced helper");
    
    let mut new_config = helper.get_config().clone();
    new_config.repl.show_timing = !new_config.repl.show_timing;
    
    // Should be able to update configuration at runtime
    let result = helper.update_config(new_config.clone());
    assert!(result.is_ok());
    assert_eq!(helper.get_config().repl.show_timing, new_config.repl.show_timing);
}

#[test]
fn test_enhanced_helper_meta_commands() {
    // RED: Test helper management commands
    let repl_engine = ReplEngine::new().expect("Failed to create REPL engine");
    let helper = EnhancedLyraHelper::new(repl_engine.get_config().clone(), repl_engine.create_shared_completer())
        .expect("Failed to create enhanced helper");
    
    // Test helper-info command
    let info = helper.get_helper_info();
    assert!(info.contains("Enhanced Lyra Helper"));
    assert!(info.contains("Completion: enabled"));
    
    // Test helper capabilities
    let capabilities = helper.get_capabilities();
    assert!(capabilities.contains("completion"));
    assert!(capabilities.contains("validation"));
    assert!(capabilities.contains("highlighting"));
    assert!(capabilities.contains("hinting"));
}

#[test]
fn test_enhanced_helper_thread_safety() {
    // RED: Test thread safety
    let repl_engine = ReplEngine::new().expect("Failed to create REPL engine");
    let helper = Arc::new(EnhancedLyraHelper::new(repl_engine.get_config().clone(), repl_engine.create_shared_completer())
        .expect("Failed to create enhanced helper"));
    
    let helper_clone = helper.clone();
    let handle = std::thread::spawn(move || {
        let ctx = Context::new();
        let result = helper_clone.complete("Length", 6, &ctx);
        assert!(result.is_ok());
    });
    
    handle.join().expect("Thread should complete successfully");
}

#[test]
fn test_enhanced_helper_backward_compatibility() {
    // RED: Test backward compatibility with existing functionality
    let repl_engine = ReplEngine::new().expect("Failed to create REPL engine");
    let shared_completer = repl_engine.create_shared_completer();
    
    let helper = EnhancedLyraHelper::new(repl_engine.get_config().clone(), shared_completer.clone())
        .expect("Failed to create enhanced helper");
    
    let ctx = Context::new();
    
    // Enhanced helper should provide same completion results as original
    let enhanced_result = helper.complete("Length", 6, &ctx).unwrap();
    let original_result = shared_completer.complete("Length", 6, &ctx).unwrap();
    
    assert_eq!(enhanced_result.0, original_result.0);
    
    // Should contain same candidates
    let enhanced_replacements: Vec<&str> = enhanced_result.1.iter().map(|p| p.replacement.as_str()).collect();
    let original_replacements: Vec<&str> = original_result.1.iter().map(|p| p.replacement.as_str()).collect();
    
    for replacement in &original_replacements {
        assert!(enhanced_replacements.contains(replacement));
    }
}

#[test]
fn test_enhanced_helper_error_handling() {
    // RED: Test error handling and fallback behavior
    let repl_engine = ReplEngine::new().expect("Failed to create REPL engine");
    
    // Test with invalid configuration
    let mut invalid_config = repl_engine.get_config().clone();
    invalid_config.repl.history_size = 0; // Invalid history size
    
    let helper = EnhancedLyraHelper::new(invalid_config, repl_engine.create_shared_completer());
    assert!(helper.is_err()); // Should fail with invalid config
}

#[test]
fn test_enhanced_helper_vi_emacs_modes() {
    // RED: Test Vi/Emacs mode switching preparation
    let repl_engine = ReplEngine::new().expect("Failed to create REPL engine");
    
    // Test with Emacs mode
    let mut emacs_config = repl_engine.get_config().clone();
    emacs_config.editor.mode = "emacs".to_string();
    let emacs_helper = EnhancedLyraHelper::new(emacs_config, repl_engine.create_shared_completer());
    assert!(emacs_helper.is_ok());
    
    // Test with Vi mode
    let mut vi_config = repl_engine.get_config().clone();
    vi_config.editor.mode = "vi".to_string();
    let vi_helper = EnhancedLyraHelper::new(vi_config, repl_engine.create_shared_completer());
    assert!(vi_helper.is_ok());
}

#[test]
fn test_enhanced_helper_performance() {
    // RED: Test performance characteristics
    let repl_engine = ReplEngine::new().expect("Failed to create REPL engine");
    let helper = EnhancedLyraHelper::new(repl_engine.get_config().clone(), repl_engine.create_shared_completer())
        .expect("Failed to create enhanced helper");
    
    let ctx = Context::new();
    let start = std::time::Instant::now();
    
    // Complete many requests to test performance
    for _ in 0..1000 {
        let _ = helper.complete("S", 1, &ctx);
    }
    
    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() < 1000, "Completion should be fast"); // Should complete 1000 requests in under 1 second
}