//! Persistent History Management for Lyra REPL
//! 
//! Provides thread-safe, persistent command history with:
//! - Atomic file operations for reliability
//! - Configurable history size with LRU eviction
//! - Duplicate removal options
//! - Session isolation capabilities
//! - Cross-platform file handling

use crate::repl::config::HistoryConfig;
use serde::{Deserialize, Serialize};
use std::collections::{VecDeque, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use regex::Regex;

#[derive(Error, Debug)]
pub enum HistoryError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Lock error: {0}")]
    Lock(String),
}

pub type HistoryResult<T> = std::result::Result<T, HistoryError>;

/// A single history entry representing one REPL interaction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// Line number in the REPL session
    pub line_number: usize,
    /// The input command/expression
    pub input: String,
    /// The output result
    pub output: String,
    /// Execution time for the command
    pub execution_time: Duration,
    /// Timestamp when the command was executed
    pub timestamp: SystemTime,
    /// Tags for categorization
    pub tags: HashSet<String>,
    /// Session identifier
    pub session_id: Option<String>,
    /// Success status of the command
    pub success: bool,
    /// Command category (auto-detected)
    pub category: CommandCategory,
}

/// Command category for automatic classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommandCategory {
    /// Mathematical operations
    Mathematics,
    /// Function definitions
    FunctionDefinition,
    /// Variable assignments
    VariableAssignment,
    /// Pattern matching operations
    PatternMatching,
    /// List/data operations
    DataManipulation,
    /// Help or informational commands
    Information,
    /// System commands
    System,
    /// Unknown or uncategorized
    Unknown,
}

impl Default for CommandCategory {
    fn default() -> Self {
        CommandCategory::Unknown
    }
}

impl HistoryEntry {
    /// Create a new history entry
    pub fn new(
        line_number: usize,
        input: String,
        output: String,
        execution_time: Duration,
    ) -> Self {
        let category = Self::classify_command(&input);
        let success = !output.starts_with("Error:") && !output.starts_with("Parse error:");
        
        Self {
            line_number,
            input,
            output,
            execution_time,
            timestamp: SystemTime::now(),
            tags: HashSet::new(),
            session_id: None,
            success,
            category,
        }
    }
    
    /// Create a new history entry with additional metadata
    pub fn new_with_metadata(
        line_number: usize,
        input: String,
        output: String,
        execution_time: Duration,
        session_id: Option<String>,
        tags: HashSet<String>,
    ) -> Self {
        let category = Self::classify_command(&input);
        let success = !output.starts_with("Error:") && !output.starts_with("Parse error:");
        
        Self {
            line_number,
            input,
            output,
            execution_time,
            timestamp: SystemTime::now(),
            tags,
            session_id,
            success,
            category,
        }
    }
    
    /// Add a tag to this entry
    pub fn add_tag(&mut self, tag: String) {
        self.tags.insert(tag);
    }
    
    /// Remove a tag from this entry
    pub fn remove_tag(&mut self, tag: &str) -> bool {
        self.tags.remove(tag)
    }
    
    /// Check if entry has a specific tag
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.contains(tag)
    }
    
    /// Classify command type based on input
    fn classify_command(input: &str) -> CommandCategory {
        let input_lower = input.to_lowercase();
        
        // Function definitions
        if input.contains("]:=") || input.contains("]=") {
            return CommandCategory::FunctionDefinition;
        }
        
        // Variable assignments
        if input.contains("=") && !input.contains("==") && !input.contains("!=") {
            return CommandCategory::VariableAssignment;
        }
        
        // Pattern matching
        if input.contains("/.") || input.contains("_") {
            return CommandCategory::PatternMatching;
        }
        
        // Mathematical operations
        if input_lower.contains("sin") || input_lower.contains("cos") || 
           input_lower.contains("sqrt") || input_lower.contains("log") ||
           input.contains("+") || input.contains("-") || input.contains("*") || input.contains("/") {
            return CommandCategory::Mathematics;
        }
        
        // Data manipulation
        if input_lower.contains("map") || input_lower.contains("select") ||
           input_lower.contains("length") || input.contains("{") {
            return CommandCategory::DataManipulation;
        }
        
        // Help commands
        if input_lower.starts_with("?") || input_lower.contains("help") {
            return CommandCategory::Information;
        }
        
        CommandCategory::Unknown
    }
    
    /// Serialize entry to string format for file storage
    pub fn to_string(&self) -> String {
        let timestamp_secs = self.timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let execution_time_micros = self.execution_time.as_micros();
        let tags_str = self.tags.iter().cloned().collect::<Vec<_>>().join(",");
        let session_id_str = self.session_id.as_deref().unwrap_or("");
        let category_str = format!("{:?}", self.category);
        
        // Enhanced format with metadata
        // Format: line_number|timestamp|execution_time_micros|success|category|session_id|tags|input_len|input|output
        format!(
            "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
            self.line_number,
            timestamp_secs,
            execution_time_micros,
            self.success,
            category_str,
            session_id_str,
            tags_str,
            self.input.len(),
            self.input,
            self.output
        )
    }
    
    /// Parse entry from string format
    pub fn from_str(s: &str) -> HistoryResult<Self> {
        let parts: Vec<&str> = s.splitn(10, '|').collect();
        
        // Handle both old and new formats for backward compatibility
        if parts.len() == 6 {
            // Old format: line_number|timestamp|execution_time_micros|input_len|input|output
            return Self::parse_old_format(parts);
        } else if parts.len() != 10 {
            return Err(HistoryError::Parse(
                "Invalid history entry format".to_string()
            ));
        }
        
        let line_number = parts[0].parse::<usize>()
            .map_err(|e| HistoryError::Parse(format!("Invalid line number: {}", e)))?;
        
        let timestamp_secs = parts[1].parse::<u64>()
            .map_err(|e| HistoryError::Parse(format!("Invalid timestamp: {}", e)))?;
        
        let execution_time_micros = parts[2].parse::<u128>()
            .map_err(|e| HistoryError::Parse(format!("Invalid execution time: {}", e)))?;
        
        let success = parts[3].parse::<bool>()
            .map_err(|e| HistoryError::Parse(format!("Invalid success flag: {}", e)))?;
        
        let category = match parts[4] {
            "Mathematics" => CommandCategory::Mathematics,
            "FunctionDefinition" => CommandCategory::FunctionDefinition,
            "VariableAssignment" => CommandCategory::VariableAssignment,
            "PatternMatching" => CommandCategory::PatternMatching,
            "DataManipulation" => CommandCategory::DataManipulation,
            "Information" => CommandCategory::Information,
            "System" => CommandCategory::System,
            _ => CommandCategory::Unknown,
        };
        
        let session_id = if parts[5].is_empty() {
            None
        } else {
            Some(parts[5].to_string())
        };
        
        let tags: HashSet<String> = if parts[6].is_empty() {
            HashSet::new()
        } else {
            parts[6].split(',').map(|s| s.to_string()).collect()
        };
        
        let input_len = parts[7].parse::<usize>()
            .map_err(|e| HistoryError::Parse(format!("Invalid input length: {}", e)))?;
        
        let remaining = parts[9];
        if remaining.len() < input_len {
            return Err(HistoryError::Parse(
                "Input length mismatch".to_string()
            ));
        }
        
        let input = remaining[..input_len].to_string();
        let output = remaining[input_len..].to_string();
        
        let timestamp = UNIX_EPOCH + Duration::from_secs(timestamp_secs);
        let execution_time = Duration::from_micros(execution_time_micros as u64);
        
        Ok(Self {
            line_number,
            input,
            output,
            execution_time,
            timestamp,
            tags,
            session_id,
            success,
            category,
        })
    }
    
    /// Parse old format for backward compatibility
    fn parse_old_format(parts: Vec<&str>) -> HistoryResult<Self> {
        let line_number = parts[0].parse::<usize>()
            .map_err(|e| HistoryError::Parse(format!("Invalid line number: {}", e)))?;
        
        let timestamp_secs = parts[1].parse::<u64>()
            .map_err(|e| HistoryError::Parse(format!("Invalid timestamp: {}", e)))?;
        
        let execution_time_micros = parts[2].parse::<u128>()
            .map_err(|e| HistoryError::Parse(format!("Invalid execution time: {}", e)))?;
        
        let input_len = parts[3].parse::<usize>()
            .map_err(|e| HistoryError::Parse(format!("Invalid input length: {}", e)))?;
        
        let remaining = parts[5];
        if remaining.len() < input_len {
            return Err(HistoryError::Parse(
                "Input length mismatch".to_string()
            ));
        }
        
        let input = remaining[..input_len].to_string();
        let output = remaining[input_len..].to_string();
        
        let timestamp = UNIX_EPOCH + Duration::from_secs(timestamp_secs);
        let execution_time = Duration::from_micros(execution_time_micros as u64);
        let category = Self::classify_command(&input);
        let success = !output.starts_with("Error:") && !output.starts_with("Parse error:");
        
        Ok(Self {
            line_number,
            input,
            output,
            execution_time,
            timestamp,
            tags: HashSet::new(),
            session_id: None,
            success,
            category,
        })
    }
    
    /// Check if this entry is a duplicate of another (based on input)
    pub fn is_duplicate(&self, other: &Self) -> bool {
        self.input.trim() == other.input.trim()
    }
}

/// Thread-safe history manager with persistence
pub struct HistoryManager {
    /// Path to the history file
    file_path: PathBuf,
    /// Configuration for history behavior
    config: HistoryConfig,
    /// In-memory history entries (recent first)
    entries: Arc<Mutex<VecDeque<HistoryEntry>>>,
    /// Track if history needs to be saved
    dirty: Arc<Mutex<bool>>,
}

impl HistoryManager {
    /// Create a new history manager
    pub fn new<P: AsRef<Path>>(file_path: P, config: HistoryConfig) -> HistoryResult<Self> {
        let file_path = file_path.as_ref().to_path_buf();
        
        let manager = Self {
            file_path: file_path.clone(),
            config,
            entries: Arc::new(Mutex::new(VecDeque::new())),
            dirty: Arc::new(Mutex::new(false)),
        };
        
        // Load existing history if file exists
        if file_path.exists() {
            manager.load()?;
        } else {
            // Create the file and its directory if they don't exist
            if let Some(parent) = file_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            // Create empty file
            File::create(&file_path)?;
        }
        
        Ok(manager)
    }
    
    // Helper methods for advanced search functionality
    
    /// Check if an entry matches the given search criteria
    fn matches_criteria(&self, entry: &HistoryEntry, criteria: &SearchCriteria) -> bool {
        // Text pattern matching
        if let Some(pattern) = &criteria.text_pattern {
            let matches_text = if criteria.use_regex {
                if let Ok(regex) = Regex::new(pattern) {
                    match criteria.search_scope {
                        SearchScope::InputOnly => regex.is_match(&entry.input),
                        SearchScope::OutputOnly => regex.is_match(&entry.output),
                        SearchScope::Both => regex.is_match(&entry.input) || regex.is_match(&entry.output),
                    }
                } else {
                    false // Invalid regex
                }
            } else {
                let pattern = if criteria.case_sensitive {
                    pattern.clone()
                } else {
                    pattern.to_lowercase()
                };
                
                let input = if criteria.case_sensitive {
                    entry.input.clone()
                } else {
                    entry.input.to_lowercase()
                };
                
                let output = if criteria.case_sensitive {
                    entry.output.clone()
                } else {
                    entry.output.to_lowercase()
                };
                
                match criteria.search_scope {
                    SearchScope::InputOnly => input.contains(&pattern),
                    SearchScope::OutputOnly => output.contains(&pattern),
                    SearchScope::Both => input.contains(&pattern) || output.contains(&pattern),
                }
            };
            
            if !matches_text {
                return false;
            }
        }
        
        // Tags filter
        if !criteria.tags.is_empty() {
            let has_matching_tag = criteria.tags.iter().any(|tag| entry.has_tag(tag));
            if !has_matching_tag {
                return false;
            }
        }
        
        // Categories filter
        if !criteria.categories.is_empty() {
            if !criteria.categories.contains(&entry.category) {
                return false;
            }
        }
        
        // Success status filter
        if let Some(expected_success) = criteria.success_status {
            if entry.success != expected_success {
                return false;
            }
        }
        
        // Session ID filter
        if let Some(expected_session) = &criteria.session_id {
            if entry.session_id.as_deref() != Some(expected_session) {
                return false;
            }
        }
        
        // Time range filter
        if let Some((start, end)) = criteria.time_range {
            if entry.timestamp < start || entry.timestamp > end {
                return false;
            }
        }
        
        // Execution time filters
        if let Some(min_time) = criteria.min_execution_time {
            if entry.execution_time < min_time {
                return false;
            }
        }
        
        if let Some(max_time) = criteria.max_execution_time {
            if entry.execution_time > max_time {
                return false;
            }
        }
        
        true
    }
    
    /// Calculate fuzzy search score for an entry
    fn calculate_fuzzy_score(&self, query: &str, entry: &HistoryEntry) -> f32 {
        let input_score = self.string_similarity(query, &entry.input.to_lowercase());
        let output_score = self.string_similarity(query, &entry.output.to_lowercase());
        
        // Weight input higher than output
        input_score * 0.7 + output_score * 0.3
    }
    
    /// Calculate string similarity using a simple algorithm
    fn string_similarity(&self, s1: &str, s2: &str) -> f32 {
        if s1.is_empty() && s2.is_empty() {
            return 1.0;
        }
        if s1.is_empty() || s2.is_empty() {
            return 0.0;
        }
        
        // Simple substring matching with bonus for exact matches
        if s2.contains(s1) {
            return if s1 == s2 { 1.0 } else { 0.8 };
        }
        
        // Character-based similarity
        let mut common_chars = 0;
        for c in s1.chars() {
            if s2.contains(c) {
                common_chars += 1;
            }
        }
        
        let similarity = common_chars as f32 / s1.len().max(s2.len()) as f32;
        
        // Apply threshold to avoid very low scores
        if similarity < 0.1 {
            0.0
        } else {
            similarity
        }
    }
    
    /// Add a new history entry
    pub fn add_entry(&self, entry: HistoryEntry) -> HistoryResult<()> {
        let mut entries = self.entries.lock()
            .map_err(|e| HistoryError::Lock(e.to_string()))?;
        
        // Check for duplicates if enabled
        if self.config.remove_duplicates {
            // Remove any existing duplicate
            entries.retain(|e| !e.is_duplicate(&entry));
        }
        
        // Add to the front (most recent first)
        entries.push_front(entry);
        
        // Enforce size limit (LRU eviction)
        while entries.len() > self.config.size {
            entries.pop_back();
        }
        
        // Mark as dirty for saving
        let mut dirty = self.dirty.lock()
            .map_err(|e| HistoryError::Lock(e.to_string()))?;
        *dirty = true;
        
        Ok(())
    }
    
    /// Get all history entries (most recent first)
    pub fn get_entries(&self) -> Vec<HistoryEntry> {
        let entries = self.entries.lock().unwrap_or_else(|_| {
            // In case of poison, return empty vector
            std::process::abort();
        });
        entries.iter().cloned().collect()
    }
    
    /// Get recent entries up to a limit
    pub fn get_recent_entries(&self, limit: usize) -> Vec<HistoryEntry> {
        let entries = self.entries.lock().unwrap_or_else(|_| {
            std::process::abort();
        });
        entries.iter().take(limit).cloned().collect()
    }
    
    /// Search history entries by input pattern (exact match)
    pub fn search_entries(&self, pattern: &str) -> Vec<HistoryEntry> {
        let entries = self.entries.lock().unwrap_or_else(|_| {
            std::process::abort();
        });
        
        entries.iter()
            .filter(|entry| entry.input.contains(pattern))
            .cloned()
            .collect()
    }
    
    /// Advanced search with multiple criteria
    pub fn search_entries_advanced(&self, search_criteria: &SearchCriteria) -> Vec<HistoryEntry> {
        let entries = self.entries.lock().unwrap_or_else(|_| {
            std::process::abort();
        });
        
        entries.iter()
            .filter(|entry| self.matches_criteria(entry, search_criteria))
            .cloned()
            .collect()
    }
    
    /// Fuzzy search across input and output
    pub fn fuzzy_search(&self, query: &str, threshold: f32) -> Vec<HistoryEntry> {
        let entries = self.entries.lock().unwrap_or_else(|_| {
            std::process::abort();
        });
        
        let query_lower = query.to_lowercase();
        let mut scored_entries: Vec<(f32, HistoryEntry)> = entries.iter()
            .filter_map(|entry| {
                let score = self.calculate_fuzzy_score(&query_lower, entry);
                if score >= threshold {
                    Some((score, entry.clone()))
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by score (highest first)
        scored_entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        
        scored_entries.into_iter().map(|(_, entry)| entry).collect()
    }
    
    /// Search by tags
    pub fn search_by_tags(&self, tags: &[String]) -> Vec<HistoryEntry> {
        let entries = self.entries.lock().unwrap_or_else(|_| {
            std::process::abort();
        });
        
        entries.iter()
            .filter(|entry| {
                tags.iter().any(|tag| entry.has_tag(tag))
            })
            .cloned()
            .collect()
    }
    
    /// Search by category
    pub fn search_by_category(&self, category: &CommandCategory) -> Vec<HistoryEntry> {
        let entries = self.entries.lock().unwrap_or_else(|_| {
            std::process::abort();
        });
        
        entries.iter()
            .filter(|entry| &entry.category == category)
            .cloned()
            .collect()
    }
    
    /// Search by time range
    pub fn search_by_time_range(&self, start: SystemTime, end: SystemTime) -> Vec<HistoryEntry> {
        let entries = self.entries.lock().unwrap_or_else(|_| {
            std::process::abort();
        });
        
        entries.iter()
            .filter(|entry| entry.timestamp >= start && entry.timestamp <= end)
            .cloned()
            .collect()
    }
    
    /// Search by execution status (success/failure)
    pub fn search_by_status(&self, success: bool) -> Vec<HistoryEntry> {
        let entries = self.entries.lock().unwrap_or_else(|_| {
            std::process::abort();
        });
        
        entries.iter()
            .filter(|entry| entry.success == success)
            .cloned()
            .collect()
    }
    
    /// Get entries by session ID
    pub fn get_session_entries(&self, session_id: &str) -> Vec<HistoryEntry> {
        let entries = self.entries.lock().unwrap_or_else(|_| {
            std::process::abort();
        });
        
        entries.iter()
            .filter(|entry| {
                entry.session_id.as_deref() == Some(session_id)
            })
            .cloned()
            .collect()
    }
    
    /// Add tag to existing entry
    pub fn add_tag_to_entry(&self, line_number: usize, tag: String) -> HistoryResult<()> {
        let mut entries = self.entries.lock()
            .map_err(|e| HistoryError::Lock(e.to_string()))?;
        
        for entry in entries.iter_mut() {
            if entry.line_number == line_number {
                entry.add_tag(tag);
                
                // Mark as dirty for saving
                let mut dirty = self.dirty.lock()
                    .map_err(|e| HistoryError::Lock(e.to_string()))?;
                *dirty = true;
                
                return Ok(());
            }
        }
        
        Err(HistoryError::Parse(format!("Entry with line number {} not found", line_number)))
    }
    
    /// Remove tag from existing entry
    pub fn remove_tag_from_entry(&self, line_number: usize, tag: &str) -> HistoryResult<bool> {
        let mut entries = self.entries.lock()
            .map_err(|e| HistoryError::Lock(e.to_string()))?;
        
        for entry in entries.iter_mut() {
            if entry.line_number == line_number {
                let removed = entry.remove_tag(tag);
                
                if removed {
                    // Mark as dirty for saving
                    let mut dirty = self.dirty.lock()
                        .map_err(|e| HistoryError::Lock(e.to_string()))?;
                    *dirty = true;
                }
                
                return Ok(removed);
            }
        }
        
        Ok(false)
    }
    
    /// Get statistics about history
    pub fn get_statistics(&self) -> HistoryStatistics {
        let entries = self.entries.lock().unwrap_or_else(|_| {
            std::process::abort();
        });
        
        let total_entries = entries.len();
        let successful_entries = entries.iter().filter(|e| e.success).count();
        let failed_entries = total_entries - successful_entries;
        
        let mut category_counts: HashMap<CommandCategory, usize> = HashMap::new();
        let mut tag_counts: HashMap<String, usize> = HashMap::new();
        let mut total_execution_time = Duration::new(0, 0);
        
        for entry in entries.iter() {
            *category_counts.entry(entry.category.clone()).or_insert(0) += 1;
            
            for tag in &entry.tags {
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
            }
            
            total_execution_time += entry.execution_time;
        }
        
        let avg_execution_time = if total_entries > 0 {
            total_execution_time / total_entries as u32
        } else {
            Duration::new(0, 0)
        };
        
        HistoryStatistics {
            total_entries,
            successful_entries,
            failed_entries,
            category_counts,
            tag_counts,
            total_execution_time,
            avg_execution_time,
        }
    }
    
    /// Clear all history entries
    pub fn clear(&self) -> HistoryResult<()> {
        let mut entries = self.entries.lock()
            .map_err(|e| HistoryError::Lock(e.to_string()))?;
        entries.clear();
        
        let mut dirty = self.dirty.lock()
            .map_err(|e| HistoryError::Lock(e.to_string()))?;
        *dirty = true;
        
        Ok(())
    }
    
    /// Save history to file (atomic operation)
    pub fn save(&self) -> HistoryResult<()> {
        let dirty = {
            let dirty_guard = self.dirty.lock()
                .map_err(|e| HistoryError::Lock(e.to_string()))?;
            *dirty_guard
        };
        
        if !dirty {
            return Ok(());
        }
        
        let entries = self.entries.lock()
            .map_err(|e| HistoryError::Lock(e.to_string()))?;
        
        // Use atomic write operation: write to temp file, then rename
        let temp_path = self.file_path.with_extension("tmp");
        
        {
            let temp_file = File::create(&temp_path)?;
            let mut writer = BufWriter::new(temp_file);
            
            // Write entries in reverse order (oldest first for file storage)
            for entry in entries.iter().rev() {
                writeln!(writer, "{}", entry.to_string())?;
            }
            
            writer.flush()?;
        } // File closed here
        
        // Atomic rename
        std::fs::rename(&temp_path, &self.file_path)?;
        
        // Mark as clean
        let mut dirty = self.dirty.lock()
            .map_err(|e| HistoryError::Lock(e.to_string()))?;
        *dirty = false;
        
        Ok(())
    }
    
    /// Load history from file
    pub fn load(&self) -> HistoryResult<()> {
        if !self.file_path.exists() {
            return Ok(());
        }
        
        let file = File::open(&self.file_path)?;
        let reader = BufReader::new(file);
        
        let mut entries = self.entries.lock()
            .map_err(|e| HistoryError::Lock(e.to_string()))?;
        entries.clear();
        
        // Read entries and add them (file has oldest first, we want newest first)
        let mut loaded_entries = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                match HistoryEntry::from_str(&line) {
                    Ok(entry) => loaded_entries.push(entry),
                    Err(e) => {
                        // Log error but continue loading other entries
                        eprintln!("Warning: Failed to parse history entry: {}", e);
                        continue;
                    }
                }
            }
        }
        
        // Add entries in reverse order (newest first in memory)
        for entry in loaded_entries.into_iter().rev() {
            entries.push_back(entry);
        }
        
        // Enforce size limit after loading
        while entries.len() > self.config.size {
            entries.pop_back();
        }
        
        // Mark as clean
        let mut dirty = self.dirty.lock()
            .map_err(|e| HistoryError::Lock(e.to_string()))?;
        *dirty = false;
        
        Ok(())
    }
    
    /// Get the maximum number of entries this manager will keep
    pub fn max_size(&self) -> usize {
        self.config.size
    }
    
    /// Check if duplicate removal is enabled
    pub fn remove_duplicates(&self) -> bool {
        self.config.remove_duplicates
    }
    
    /// Get the current number of entries
    pub fn len(&self) -> usize {
        self.entries.lock().map(|e| e.len()).unwrap_or(0)
    }
    
    /// Check if history is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Export history entries to a different format/file
    pub fn export_to_file<P: AsRef<Path>>(&self, path: P, format: ExportFormat) -> HistoryResult<()> {
        let entries = self.entries.lock()
            .map_err(|e| HistoryError::Lock(e.to_string()))?;
        
        let file = File::create(path.as_ref())?;
        let mut writer = BufWriter::new(file);
        
        match format {
            ExportFormat::Json => {
                let json = serde_json::to_string_pretty(&*entries)
                    .map_err(|e| HistoryError::Serialization(e.to_string()))?;
                write!(writer, "{}", json)?;
            }
            ExportFormat::PlainText => {
                for entry in entries.iter().rev() {
                    writeln!(writer, "[{}] {} => {}", 
                        entry.line_number, entry.input, entry.output)?;
                }
            }
            ExportFormat::Csv => {
                writeln!(writer, "line_number,timestamp,execution_time_ms,success,category,session_id,tags,input,output")?;
                for entry in entries.iter().rev() {
                    let timestamp_secs = entry.timestamp
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let exec_time_ms = entry.execution_time.as_millis();
                    let tags_str = entry.tags.iter().cloned().collect::<Vec<_>>().join(";");
                    let session_id_str = entry.session_id.as_deref().unwrap_or("");
                    
                    writeln!(writer, "{},{},{},{},{:?},{},\"{}\",\"{}\",\"{}\"",
                        entry.line_number,
                        timestamp_secs,
                        exec_time_ms,
                        entry.success,
                        entry.category,
                        session_id_str,
                        tags_str,
                        entry.input.replace("\"", "\"\""), // Escape quotes for CSV
                        entry.output.replace("\"", "\"\"")
                    )?;
                }
            }
            ExportFormat::Markdown => {
                writeln!(writer, "# REPL History Export\n")?;
                for entry in entries.iter().rev() {
                    writeln!(writer, "## Entry {} - {:?}\n", entry.line_number, entry.category)?;
                    writeln!(writer, "**Input:**\n```\n{}\n```\n", entry.input)?;
                    writeln!(writer, "**Output:**\n```\n{}\n```\n", entry.output)?;
                    writeln!(writer, "**Execution Time:** {:?}\n", entry.execution_time)?;
                    if !entry.tags.is_empty() {
                        writeln!(writer, "**Tags:** {}\n", entry.tags.iter().cloned().collect::<Vec<_>>().join(", "))?;
                    }
                    writeln!(writer, "---\n")?;
                }
            }
            ExportFormat::Html => {
                writeln!(writer, "<!DOCTYPE html>")?;
                writeln!(writer, "<html><head><title>REPL History</title>")?;
                writeln!(writer, "<style>")?;
                writeln!(writer, "body {{ font-family: Arial, sans-serif; margin: 40px; }}")?;
                writeln!(writer, ".entry {{ border: 1px solid #ccc; margin: 10px 0; padding: 15px; }}")?;
                writeln!(writer, ".input {{ background: #f0f0f0; padding: 10px; border-radius: 4px; }}")?;
                writeln!(writer, ".output {{ background: #fff; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }}")?;
                writeln!(writer, ".meta {{ color: #666; font-size: 0.9em; }}")?;
                writeln!(writer, "</style></head><body>")?;
                writeln!(writer, "<h1>REPL History Export</h1>")?;
                
                for entry in entries.iter().rev() {
                    writeln!(writer, "<div class='entry'>")?;
                    writeln!(writer, "<h3>Entry {} - {:?}</h3>", entry.line_number, entry.category)?;
                    writeln!(writer, "<div class='input'><strong>Input:</strong><br><code>{}</code></div>", 
                           entry.input.replace("<", "&lt;").replace(">", "&gt;"))?;
                    writeln!(writer, "<div class='output'><strong>Output:</strong><br><code>{}</code></div>", 
                           entry.output.replace("<", "&lt;").replace(">", "&gt;"))?;
                    writeln!(writer, "<div class='meta'>")?;
                    writeln!(writer, "Execution Time: {:?} | Success: {} | ", 
                           entry.execution_time, entry.success)?;
                    writeln!(writer, "Timestamp: {:?}", entry.timestamp)?;
                    if !entry.tags.is_empty() {
                        writeln!(writer, "<br>Tags: {}", entry.tags.iter().cloned().collect::<Vec<_>>().join(", "))?;
                    }
                    writeln!(writer, "</div></div>")?;
                }
                
                writeln!(writer, "</body></html>")?;
            }
        }
        
        writer.flush()?;
        Ok(())
    }
}

/// Drop implementation to auto-save on drop
impl Drop for HistoryManager {
    fn drop(&mut self) {
        // Best effort save on drop
        let _ = self.save();
    }
}

/// Search criteria for advanced search
#[derive(Debug, Clone, Default)]
pub struct SearchCriteria {
    /// Text pattern to search for
    pub text_pattern: Option<String>,
    /// Use regex for text pattern
    pub use_regex: bool,
    /// Search in input only, output only, or both
    pub search_scope: SearchScope,
    /// Tags to match (any of these tags)
    pub tags: Vec<String>,
    /// Categories to match
    pub categories: Vec<CommandCategory>,
    /// Success status filter
    pub success_status: Option<bool>,
    /// Session ID filter
    pub session_id: Option<String>,
    /// Time range filter
    pub time_range: Option<(SystemTime, SystemTime)>,
    /// Minimum execution time
    pub min_execution_time: Option<Duration>,
    /// Maximum execution time
    pub max_execution_time: Option<Duration>,
    /// Case sensitive search
    pub case_sensitive: bool,
}

/// Search scope for text patterns
#[derive(Debug, Clone, PartialEq)]
pub enum SearchScope {
    /// Search in input only
    InputOnly,
    /// Search in output only
    OutputOnly,
    /// Search in both input and output
    Both,
}

impl Default for SearchScope {
    fn default() -> Self {
        SearchScope::Both
    }
}

/// History statistics
#[derive(Debug, Clone)]
pub struct HistoryStatistics {
    /// Total number of entries
    pub total_entries: usize,
    /// Number of successful entries
    pub successful_entries: usize,
    /// Number of failed entries
    pub failed_entries: usize,
    /// Count by category
    pub category_counts: HashMap<CommandCategory, usize>,
    /// Count by tag
    pub tag_counts: HashMap<String, usize>,
    /// Total execution time across all entries
    pub total_execution_time: Duration,
    /// Average execution time per entry
    pub avg_execution_time: Duration,
}

/// Export format options
#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    Json,
    PlainText,
    Csv,
    Markdown,
    Html,
}

/// Thread-safe wrapper for easier integration
#[derive(Clone)]
pub struct SharedHistoryManager {
    inner: Arc<HistoryManager>,
}

impl SharedHistoryManager {
    /// Create a new shared history manager
    pub fn new<P: AsRef<Path>>(file_path: P, config: HistoryConfig) -> HistoryResult<Self> {
        let manager = HistoryManager::new(file_path, config)?;
        Ok(Self {
            inner: Arc::new(manager),
        })
    }
    
    /// Delegate to inner manager
    pub fn add_entry(&self, entry: HistoryEntry) -> HistoryResult<()> {
        self.inner.add_entry(entry)
    }
    
    pub fn get_entries(&self) -> Vec<HistoryEntry> {
        self.inner.get_entries()
    }
    
    pub fn get_recent_entries(&self, limit: usize) -> Vec<HistoryEntry> {
        self.inner.get_recent_entries(limit)
    }
    
    pub fn search_entries(&self, pattern: &str) -> Vec<HistoryEntry> {
        self.inner.search_entries(pattern)
    }
    
    pub fn clear(&self) -> HistoryResult<()> {
        self.inner.clear()
    }
    
    pub fn save(&self) -> HistoryResult<()> {
        self.inner.save()
    }
    
    pub fn max_size(&self) -> usize {
        self.inner.max_size()
    }
    
    pub fn remove_duplicates(&self) -> bool {
        self.inner.remove_duplicates()
    }
    
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    
    pub fn export_to_file<P: AsRef<Path>>(&self, path: P, format: ExportFormat) -> HistoryResult<()> {
        self.inner.export_to_file(path, format)
    }
    
    pub fn search_entries_advanced(&self, criteria: &SearchCriteria) -> Vec<HistoryEntry> {
        self.inner.search_entries_advanced(criteria)
    }
    
    pub fn fuzzy_search(&self, query: &str, threshold: f32) -> Vec<HistoryEntry> {
        self.inner.fuzzy_search(query, threshold)
    }
    
    pub fn search_by_tags(&self, tags: &[String]) -> Vec<HistoryEntry> {
        self.inner.search_by_tags(tags)
    }
    
    pub fn search_by_category(&self, category: &CommandCategory) -> Vec<HistoryEntry> {
        self.inner.search_by_category(category)
    }
    
    pub fn search_by_time_range(&self, start: SystemTime, end: SystemTime) -> Vec<HistoryEntry> {
        self.inner.search_by_time_range(start, end)
    }
    
    pub fn search_by_status(&self, success: bool) -> Vec<HistoryEntry> {
        self.inner.search_by_status(success)
    }
    
    pub fn get_session_entries(&self, session_id: &str) -> Vec<HistoryEntry> {
        self.inner.get_session_entries(session_id)
    }
    
    pub fn add_tag_to_entry(&self, line_number: usize, tag: String) -> HistoryResult<()> {
        self.inner.add_tag_to_entry(line_number, tag)
    }
    
    pub fn remove_tag_from_entry(&self, line_number: usize, tag: &str) -> HistoryResult<bool> {
        self.inner.remove_tag_from_entry(line_number, tag)
    }
    
    pub fn get_statistics(&self) -> HistoryStatistics {
        self.inner.get_statistics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    use tempfile::TempDir;
    use std::collections::HashSet;

    #[test]
    fn test_history_entry_serialization() {
        let entry = HistoryEntry::new(
            42,
            "x = 5".to_string(),
            "5".to_string(),
            Duration::from_millis(150),
        );
        
        let serialized = entry.to_string();
        let parsed = HistoryEntry::from_str(&serialized).unwrap();
        
        assert_eq!(entry.line_number, parsed.line_number);
        assert_eq!(entry.input, parsed.input);
        assert_eq!(entry.output, parsed.output);
        assert_eq!(entry.execution_time, parsed.execution_time);
    }

    #[test]
    fn test_history_manager_basic() {
        let temp_dir = TempDir::new().unwrap();
        let history_path = temp_dir.path().join("history.txt");
        
        let config = HistoryConfig {
            size: 1000,
            remove_duplicates: false,
            session_isolation: false,
        };
        
        let manager = HistoryManager::new(&history_path, config).unwrap();
        
        let entry = HistoryEntry::new(
            1,
            "test".to_string(),
            "result".to_string(),
            Duration::from_millis(100),
        );
        
        manager.add_entry(entry.clone()).unwrap();
        
        let entries = manager.get_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].input, "test");
    }

    #[test]
    fn test_history_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let history_path = temp_dir.path().join("history.txt");
        
        let config = HistoryConfig {
            size: 1000,
            remove_duplicates: false,
            session_isolation: false,
        };
        
        // Create and populate history
        {
            let manager = HistoryManager::new(&history_path, config.clone()).unwrap();
            
            for i in 1..=5 {
                let entry = HistoryEntry::new(
                    i,
                    format!("command{}", i),
                    format!("result{}", i),
                    Duration::from_millis(100),
                );
                manager.add_entry(entry).unwrap();
            }
            
            manager.save().unwrap();
        }
        
        // Load in new manager
        {
            let manager = HistoryManager::new(&history_path, config).unwrap();
            let entries = manager.get_entries();
            
            assert_eq!(entries.len(), 5);
            
            // Should be in reverse order (newest first)
            assert_eq!(entries[0].input, "command5");
            assert_eq!(entries[4].input, "command1");
        }
    }

    #[test]
    fn test_size_limit() {
        let temp_dir = TempDir::new().unwrap();
        let history_path = temp_dir.path().join("history.txt");
        
        let config = HistoryConfig {
            size: 3,
            remove_duplicates: false,
            session_isolation: false,
        };
        
        let manager = HistoryManager::new(&history_path, config).unwrap();
        
        // Add more entries than the limit
        for i in 1..=5 {
            let entry = HistoryEntry::new(
                i,
                format!("command{}", i),
                format!("result{}", i),
                Duration::from_millis(100),
            );
            manager.add_entry(entry).unwrap();
        }
        
        let entries = manager.get_entries();
        assert_eq!(entries.len(), 3);
        
        // Should keep the last 3 entries
        assert_eq!(entries[0].input, "command5");
        assert_eq!(entries[1].input, "command4");
        assert_eq!(entries[2].input, "command3");
    }

    #[test]
    fn test_duplicate_removal() {
        let temp_dir = TempDir::new().unwrap();
        let history_path = temp_dir.path().join("history.txt");
        
        let config = HistoryConfig {
            size: 1000,
            remove_duplicates: true,
            session_isolation: false,
        };
        
        let manager = HistoryManager::new(&history_path, config).unwrap();
        
        // Add duplicate entries
        for i in 1..=3 {
            let entry = HistoryEntry::new(
                i,
                "x = 5".to_string(), // Same input
                "5".to_string(),
                Duration::from_millis(100),
            );
            manager.add_entry(entry).unwrap();
        }
        
        let entries = manager.get_entries();
        assert_eq!(entries.len(), 1); // Only one entry due to deduplication
        assert_eq!(entries[0].input, "x = 5");
    }

    #[test]
    fn test_search_entries() {
        let temp_dir = TempDir::new().unwrap();
        let history_path = temp_dir.path().join("history.txt");
        
        let config = HistoryConfig {
            size: 1000,
            remove_duplicates: false,
            session_isolation: false,
        };
        
        let manager = HistoryManager::new(&history_path, config).unwrap();
        
        // Add various entries
        let commands = vec!["x = 5", "y = x + 1", "Sin[Pi]", "x = 10"];
        for (i, cmd) in commands.iter().enumerate() {
            let entry = HistoryEntry::new(
                i + 1,
                cmd.to_string(),
                "result".to_string(),
                Duration::from_millis(100),
            );
            manager.add_entry(entry).unwrap();
        }
        
        // Search for entries containing "x"
        let results = manager.search_entries("x");
        assert_eq!(results.len(), 3); // "x = 5", "y = x + 1", "x = 10"
        
        // Search for entries containing "Sin"
        let results = manager.search_entries("Sin");
        assert_eq!(results.len(), 1); // "Sin[Pi]"
    }

    #[test]
    fn test_thread_safety() {
        let temp_dir = TempDir::new().unwrap();
        let history_path = temp_dir.path().join("history.txt");
        
        let config = HistoryConfig {
            size: 1000,
            remove_duplicates: false,
            session_isolation: false,
        };
        
        let manager = Arc::new(HistoryManager::new(&history_path, config).unwrap());
        let mut handles = vec![];
        
        // Spawn multiple threads adding entries
        for i in 0..5 {
            let manager_clone = Arc::clone(&manager);
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let entry = HistoryEntry::new(
                        i * 10 + j,
                        format!("thread{}_command{}", i, j),
                        "result".to_string(),
                        Duration::from_millis(100),
                    );
                    manager_clone.add_entry(entry).unwrap();
                }
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        let entries = manager.get_entries();
        assert_eq!(entries.len(), 50); // 5 threads * 10 entries each
    }

    #[test]
    fn test_export_formats() {
        let temp_dir = TempDir::new().unwrap();
        let history_path = temp_dir.path().join("history.txt");
        
        let config = HistoryConfig {
            size: 1000,
            remove_duplicates: false,
            session_isolation: false,
        };
        
        let manager = HistoryManager::new(&history_path, config).unwrap();
        
        // Add test entries
        for i in 1..=3 {
            let entry = HistoryEntry::new(
                i,
                format!("command{}", i),
                format!("result{}", i),
                Duration::from_millis(100),
            );
            manager.add_entry(entry).unwrap();
        }
        
        // Test JSON export
        let json_path = temp_dir.path().join("export.json");
        manager.export_to_file(&json_path, ExportFormat::Json).unwrap();
        assert!(json_path.exists());
        
        // Test plain text export
        let text_path = temp_dir.path().join("export.txt");
        manager.export_to_file(&text_path, ExportFormat::PlainText).unwrap();
        assert!(text_path.exists());
        
        // Test CSV export
        let csv_path = temp_dir.path().join("export.csv");
        manager.export_to_file(&csv_path, ExportFormat::Csv).unwrap();
        assert!(csv_path.exists());
    }
    
    #[test]
    fn test_enhanced_history_search() {
        let temp_dir = TempDir::new().unwrap();
        let history_path = temp_dir.path().join("history.txt");
        
        let config = HistoryConfig {
            size: 1000,
            remove_duplicates: false,
            session_isolation: false,
        };
        
        let manager = HistoryManager::new(&history_path, config).unwrap();
        
        // Add test entries with different categories and tags
        let mut entry1 = HistoryEntry::new(
            1,
            "x = 5".to_string(),
            "5".to_string(),
            Duration::from_millis(100),
        );
        entry1.add_tag("math".to_string());
        entry1.add_tag("assignment".to_string());
        manager.add_entry(entry1).unwrap();
        
        let entry2 = HistoryEntry::new(
            2,
            "Sin[Pi/2]".to_string(),
            "1".to_string(),
            Duration::from_millis(50),
        );
        manager.add_entry(entry2).unwrap();
        
        let mut entry3 = HistoryEntry::new(
            3,
            "Map[Square, {1, 2, 3}]".to_string(),
            "{1, 4, 9}".to_string(),
            Duration::from_millis(200),
        );
        entry3.add_tag("list".to_string());
        manager.add_entry(entry3).unwrap();
        
        // Test tag search
        let results = manager.search_by_tags(&["math".to_string()]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].input, "x = 5");
        
        // Test category search
        let results = manager.search_by_category(&CommandCategory::Mathematics);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].input, "Sin[Pi/2]");
        
        // Test fuzzy search
        let results = manager.fuzzy_search("square", 0.3);
        assert!(!results.is_empty());
        
        // Test advanced search
        let criteria = SearchCriteria {
            text_pattern: Some("Map".to_string()),
            categories: vec![CommandCategory::DataManipulation],
            ..Default::default()
        };
        let results = manager.search_entries_advanced(&criteria);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].input, "Map[Square, {1, 2, 3}]");
        
        // Test statistics
        let stats = manager.get_statistics();
        assert_eq!(stats.total_entries, 3);
        assert_eq!(stats.successful_entries, 3);
        assert_eq!(stats.failed_entries, 0);
    }
    
    #[test]
    fn test_command_classification() {
        // Test function definition
        let entry = HistoryEntry::new(1, "f[x_] := x^2".to_string(), "f".to_string(), Duration::from_millis(10));
        assert_eq!(entry.category, CommandCategory::FunctionDefinition);
        
        // Test variable assignment
        let entry = HistoryEntry::new(2, "x = 5".to_string(), "5".to_string(), Duration::from_millis(10));
        assert_eq!(entry.category, CommandCategory::VariableAssignment);
        
        // Test pattern matching
        let entry = HistoryEntry::new(3, "expr /. x_ -> x^2".to_string(), "result".to_string(), Duration::from_millis(10));
        assert_eq!(entry.category, CommandCategory::PatternMatching);
        
        // Test mathematics
        let entry = HistoryEntry::new(4, "Sin[Pi/2]".to_string(), "1".to_string(), Duration::from_millis(10));
        assert_eq!(entry.category, CommandCategory::Mathematics);
        
        // Test data manipulation
        let entry = HistoryEntry::new(5, "Map[f, {1, 2, 3}]".to_string(), "{1, 4, 9}".to_string(), Duration::from_millis(10));
        assert_eq!(entry.category, CommandCategory::DataManipulation);
    }
    
    #[test]
    fn test_tag_management() {
        let temp_dir = TempDir::new().unwrap();
        let history_path = temp_dir.path().join("history.txt");
        
        let config = HistoryConfig {
            size: 1000,
            remove_duplicates: false,
            session_isolation: false,
        };
        
        let manager = HistoryManager::new(&history_path, config).unwrap();
        
        let entry = HistoryEntry::new(
            1,
            "x = 5".to_string(),
            "5".to_string(),
            Duration::from_millis(100),
        );
        manager.add_entry(entry).unwrap();
        
        // Add tags
        manager.add_tag_to_entry(1, "math".to_string()).unwrap();
        manager.add_tag_to_entry(1, "basic".to_string()).unwrap();
        
        // Verify tags were added
        let entries = manager.get_entries();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].has_tag("math"));
        assert!(entries[0].has_tag("basic"));
        
        // Remove a tag
        let removed = manager.remove_tag_from_entry(1, "basic").unwrap();
        assert!(removed);
        
        // Verify tag was removed
        let entries = manager.get_entries();
        assert!(entries[0].has_tag("math"));
        assert!(!entries[0].has_tag("basic"));
    }
}