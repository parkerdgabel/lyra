//! Comprehensive tests for the REPL configuration system
//! 
//! Following TDD principles - these tests define the expected behavior
//! before implementation.

use lyra::repl::config::{ReplConfig, DisplayConfig, EditorConfig, HistoryConfig};
use lyra::repl::history::{HistoryManager, HistoryEntry};
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};
use tempfile::TempDir;

#[test]
fn test_default_config_creation() {
    let config = ReplConfig::default();
    
    // Test default values match expected configuration
    assert_eq!(config.repl.history_size, 10000);
    assert_eq!(config.repl.remove_duplicates, true);
    assert_eq!(config.repl.show_timing, false);
    assert_eq!(config.repl.show_performance, false);
    assert_eq!(config.repl.auto_complete, true);
    assert_eq!(config.repl.multiline_support, true);
    
    assert_eq!(config.display.colors, true);
    assert_eq!(config.display.unicode_support, true);
    assert_eq!(config.display.max_output_length, 1000);
    
    assert_eq!(config.editor.mode, "emacs");
    assert!(config.editor.external_editor.contains("vim"));
}

#[test]
fn test_config_serialization_roundtrip() {
    let config = ReplConfig::default();
    
    // Serialize to TOML
    let toml_str = toml::to_string(&config).expect("Failed to serialize config");
    
    // Deserialize back
    let deserialized: ReplConfig = toml::from_str(&toml_str)
        .expect("Failed to deserialize config");
    
    // Should be identical
    assert_eq!(config, deserialized);
}

#[test]
fn test_config_file_loading() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config_path = temp_dir.path().join("config.toml");
    
    // Create a test config file
    let config_content = r#"
[repl]
history_size = 5000
remove_duplicates = false
show_timing = true
show_performance = true
auto_complete = false
multiline_support = false

[display]
colors = false
unicode_support = false
max_output_length = 500

[editor]
mode = "vi"
external_editor = "nano"
"#;
    
    fs::write(&config_path, config_content).expect("Failed to write config file");
    
    // Load the config
    let config = ReplConfig::load_from_file(&config_path)
        .expect("Failed to load config from file");
    
    // Verify values
    assert_eq!(config.repl.history_size, 5000);
    assert_eq!(config.repl.remove_duplicates, false);
    assert_eq!(config.repl.show_timing, true);
    assert_eq!(config.repl.show_performance, true);
    assert_eq!(config.repl.auto_complete, false);
    assert_eq!(config.repl.multiline_support, false);
    
    assert_eq!(config.display.colors, false);
    assert_eq!(config.display.unicode_support, false);
    assert_eq!(config.display.max_output_length, 500);
    
    assert_eq!(config.editor.mode, "vi");
    assert_eq!(config.editor.external_editor, "nano");
}

#[test]
fn test_config_partial_override() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config_path = temp_dir.path().join("config.toml");
    
    // Create a partial config file (only some fields)
    let config_content = r#"
[repl]
history_size = 8000
show_timing = true

[display]
colors = false
"#;
    
    fs::write(&config_path, config_content).expect("Failed to write config file");
    
    // Load the config
    let config = ReplConfig::load_from_file(&config_path)
        .expect("Failed to load config from file");
    
    // Overridden values
    assert_eq!(config.repl.history_size, 8000);
    assert_eq!(config.repl.show_timing, true);
    assert_eq!(config.display.colors, false);
    
    // Default values for unspecified fields
    assert_eq!(config.repl.remove_duplicates, true); // default
    assert_eq!(config.display.unicode_support, true); // default
    assert_eq!(config.editor.mode, "emacs"); // default
}

#[test]
fn test_environment_variable_overrides() {
    // Set environment variables
    std::env::set_var("LYRA_HISTORY_SIZE", "15000");
    std::env::set_var("LYRA_SHOW_TIMING", "true");
    std::env::set_var("LYRA_COLORS", "false");
    std::env::set_var("LYRA_EDITOR_MODE", "vi");
    
    let mut config = ReplConfig::default();
    config.apply_env_overrides();
    
    // Check environment variable overrides
    assert_eq!(config.repl.history_size, 15000);
    assert_eq!(config.repl.show_timing, true);
    assert_eq!(config.display.colors, false);
    assert_eq!(config.editor.mode, "vi");
    
    // Clean up environment variables
    std::env::remove_var("LYRA_HISTORY_SIZE");
    std::env::remove_var("LYRA_SHOW_TIMING");
    std::env::remove_var("LYRA_COLORS");
    std::env::remove_var("LYRA_EDITOR_MODE");
}

#[test]
fn test_config_validation() {
    let mut config = ReplConfig::default();
    
    // Valid config should pass validation
    assert!(config.validate().is_ok());
    
    // Invalid history size should fail
    config.repl.history_size = 0;
    assert!(config.validate().is_err());
    
    // Reset and test invalid max output length
    config = ReplConfig::default();
    config.display.max_output_length = 0;
    assert!(config.validate().is_err());
    
    // Reset and test invalid editor mode
    config = ReplConfig::default();
    config.editor.mode = "invalid_mode".to_string();
    assert!(config.validate().is_err());
}

#[test]
fn test_config_file_creation() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config_path = temp_dir.path().join("config.toml");
    
    // Create default config file
    let config = ReplConfig::default();
    config.save_to_file(&config_path).expect("Failed to save config to file");
    
    // Verify file exists and can be loaded
    assert!(config_path.exists());
    
    let loaded_config = ReplConfig::load_from_file(&config_path)
        .expect("Failed to load saved config");
    
    assert_eq!(config, loaded_config);
}

#[test]
fn test_cross_platform_paths() {
    // Test config directory creation
    let config_dir = ReplConfig::get_config_dir().expect("Failed to get config dir");
    
    // Should be a valid path
    assert!(config_dir.is_absolute());
    
    // Test history file path
    let history_path = ReplConfig::get_history_file_path().expect("Failed to get history path");
    assert!(history_path.is_absolute());
    assert!(history_path.file_name().is_some());
}

// History Manager Tests

#[test]
fn test_history_manager_creation() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let history_path = temp_dir.path().join("history.txt");
    
    let config = HistoryConfig {
        size: 1000,
        remove_duplicates: true,
        session_isolation: false,
    };
    
    let manager = HistoryManager::new(history_path.clone(), config)
        .expect("Failed to create history manager");
    
    assert_eq!(manager.max_size(), 1000);
    assert!(manager.remove_duplicates());
}

#[test]
fn test_history_entry_serialization() {
    let entry = HistoryEntry {
        line_number: 42,
        input: "x = 5".to_string(),
        output: "5".to_string(),
        execution_time: Duration::from_millis(150),
        timestamp: SystemTime::now(),
    };
    
    // Test serialization to string format
    let serialized = entry.to_string();
    assert!(serialized.contains("x = 5"));
    assert!(serialized.contains("42"));
    
    // Test parsing from string
    let parsed = HistoryEntry::from_str(&serialized)
        .expect("Failed to parse history entry");
    
    assert_eq!(parsed.line_number, entry.line_number);
    assert_eq!(parsed.input, entry.input);
    assert_eq!(parsed.output, entry.output);
}

#[test]
fn test_history_persistence() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let history_path = temp_dir.path().join("history.txt");
    
    let config = HistoryConfig {
        size: 1000,
        remove_duplicates: true,
        session_isolation: false,
    };
    
    // Create manager and add entries
    {
        let mut manager = HistoryManager::new(history_path.clone(), config.clone())
            .expect("Failed to create history manager");
        
        for i in 1..=5 {
            let entry = HistoryEntry {
                line_number: i,
                input: format!("command{}", i),
                output: format!("result{}", i),
                execution_time: Duration::from_millis(100),
                timestamp: SystemTime::now(),
            };
            manager.add_entry(entry).expect("Failed to add entry");
        }
        
        manager.save().expect("Failed to save history");
    }
    
    // Load in new manager instance
    {
        let manager = HistoryManager::new(history_path, config)
            .expect("Failed to create new history manager");
        
        let entries = manager.get_entries();
        assert_eq!(entries.len(), 5);
        
        for (i, entry) in entries.iter().enumerate() {
            assert_eq!(entry.line_number, (i + 1) as usize);
            assert_eq!(entry.input, format!("command{}", i + 1));
            assert_eq!(entry.output, format!("result{}", i + 1));
        }
    }
}

#[test]
fn test_history_size_limit() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let history_path = temp_dir.path().join("history.txt");
    
    let config = HistoryConfig {
        size: 3, // Small size for testing
        remove_duplicates: false,
        session_isolation: false,
    };
    
    let mut manager = HistoryManager::new(history_path, config)
        .expect("Failed to create history manager");
    
    // Add more entries than the limit
    for i in 1..=5 {
        let entry = HistoryEntry {
            line_number: i,
            input: format!("command{}", i),
            output: format!("result{}", i),
            execution_time: Duration::from_millis(100),
            timestamp: SystemTime::now(),
        };
        manager.add_entry(entry).expect("Failed to add entry");
    }
    
    // Should only keep the last 3 entries (LRU eviction)
    let entries = manager.get_entries();
    assert_eq!(entries.len(), 3);
    
    // Should be the last 3 entries
    assert_eq!(entries[0].input, "command3");
    assert_eq!(entries[1].input, "command4");
    assert_eq!(entries[2].input, "command5");
}

#[test]
fn test_history_duplicate_removal() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let history_path = temp_dir.path().join("history.txt");
    
    let config = HistoryConfig {
        size: 1000,
        remove_duplicates: true,
        session_isolation: false,
    };
    
    let mut manager = HistoryManager::new(history_path, config)
        .expect("Failed to create history manager");
    
    // Add duplicate entries
    for _ in 0..3 {
        let entry = HistoryEntry {
            line_number: 1,
            input: "x = 5".to_string(),
            output: "5".to_string(),
            execution_time: Duration::from_millis(100),
            timestamp: SystemTime::now(),
        };
        manager.add_entry(entry).expect("Failed to add entry");
    }
    
    // Should only have one entry due to duplicate removal
    let entries = manager.get_entries();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].input, "x = 5");
}

#[test]
fn test_history_thread_safety() {
    use std::sync::Arc;
    use std::thread;
    
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let history_path = temp_dir.path().join("history.txt");
    
    let config = HistoryConfig {
        size: 1000,
        remove_duplicates: false,
        session_isolation: false,
    };
    
    let manager = Arc::new(HistoryManager::new(history_path, config)
        .expect("Failed to create history manager"));
    
    let mut handles = vec![];
    
    // Spawn multiple threads adding entries concurrently
    for i in 0..5 {
        let manager_clone = Arc::clone(&manager);
        let handle = thread::spawn(move || {
            for j in 0..10 {
                let entry = HistoryEntry {
                    line_number: i * 10 + j,
                    input: format!("thread{}_command{}", i, j),
                    output: format!("result{}", j),
                    execution_time: Duration::from_millis(100),
                    timestamp: SystemTime::now(),
                };
                manager_clone.add_entry(entry).expect("Failed to add entry");
            }
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }
    
    // Should have 50 entries total (5 threads * 10 entries each)
    let entries = manager.get_entries();
    assert_eq!(entries.len(), 50);
}

#[test]
fn test_atomic_file_operations() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let history_path = temp_dir.path().join("history.txt");
    
    let config = HistoryConfig {
        size: 1000,
        remove_duplicates: false,
        session_isolation: false,
    };
    
    let mut manager = HistoryManager::new(history_path.clone(), config)
        .expect("Failed to create history manager");
    
    // Add some entries
    for i in 1..=3 {
        let entry = HistoryEntry {
            line_number: i,
            input: format!("command{}", i),
            output: format!("result{}", i),
            execution_time: Duration::from_millis(100),
            timestamp: SystemTime::now(),
        };
        manager.add_entry(entry).expect("Failed to add entry");
    }
    
    // Save should be atomic - either succeeds completely or not at all
    manager.save().expect("Failed to save history");
    
    // Verify file exists and is readable
    assert!(history_path.exists());
    
    let content = fs::read_to_string(&history_path)
        .expect("Failed to read history file");
    
    // Should contain all entries
    assert!(content.contains("command1"));
    assert!(content.contains("command2"));
    assert!(content.contains("command3"));
}

#[test]
fn test_config_integration_with_history() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config_path = temp_dir.path().join("config.toml");
    let history_path = temp_dir.path().join("history.txt");
    
    // Create a config file
    let config_content = r#"
[repl]
history_size = 100
remove_duplicates = true
show_timing = true
"#;
    
    fs::write(&config_path, config_content).expect("Failed to write config file");
    
    // Load config and create history manager
    let config = ReplConfig::load_from_file(&config_path)
        .expect("Failed to load config");
    
    let history_config = HistoryConfig::from_repl_config(&config);
    let manager = HistoryManager::new(history_path, history_config)
        .expect("Failed to create history manager");
    
    // Verify the history manager uses the config values
    assert_eq!(manager.max_size(), 100);
    assert!(manager.remove_duplicates());
}