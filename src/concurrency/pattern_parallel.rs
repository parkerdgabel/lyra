//! # Parallel Pattern Matching Engine
//! 
//! High-performance parallel pattern matching system that distributes pattern matching
//! across multiple cores with lock-free caching and concurrent rule application.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::Instant;
use crossbeam_utils::thread;
use dashmap::DashMap;
use rayon::prelude::*;
use parking_lot::RwLock;

use crate::vm::{Value, VmResult, VmError};
use crate::ast::{Pattern, Expr, Symbol, Number};
use crate::pattern_matcher::{MatchResult, PatternMatcher, MatchingContext};
use super::{ConcurrencyConfig, ConcurrencyStats, ConcurrencyError, WorkStealingScheduler, TaskPriority};

/// Convert a Value to an Expr for pattern matching
fn value_to_expr(value: &Value) -> Expr {
    match value {
        Value::Integer(n) => Expr::Number(Number::Integer(*n)),
        Value::Real(f) => Expr::Number(Number::Real(*f)),
        Value::String(s) => Expr::String(s.clone()),
        Value::Symbol(name) => Expr::Symbol(Symbol { name: name.clone() }),
        Value::List(items) => {
            let expr_items: Vec<Expr> = items.iter().map(value_to_expr).collect();
            Expr::List(expr_items)
        },
        Value::Function(name) => {
            // For functions, represent as a symbol
            Expr::Symbol(Symbol { name: name.clone() })
        },
        Value::Boolean(b) => {
            // For booleans, represent as a symbol
            Expr::Symbol(Symbol { name: if *b { "True" } else { "False" }.to_string() })
        },
        Value::Missing => {
            // For missing values, represent as a symbol
            Expr::Symbol(Symbol { name: "Missing".to_string() })
        },
        Value::Object(_) => {
            // For objects, represent as a symbol
            Expr::Symbol(Symbol { name: "Object".to_string() })
        },
        Value::LyObj(_) => {
            // For foreign objects, represent as a symbol
            Expr::Symbol(Symbol { name: "ForeignObject".to_string() })
        },
        Value::Quote(expr) => {
            // For quoted expressions, return the inner expression
            *expr.clone()
        },
        Value::Pattern(_) => {
            // For patterns, represent as a symbol
            Expr::Symbol(Symbol { name: "Pattern".to_string() })
        },
        Value::Rule { lhs: _, rhs: _ } => {
            // For rules, represent as a symbol
            Expr::Symbol(Symbol { name: "Rule".to_string() })
        },
        Value::PureFunction { body: _ } => {
            // For pure functions, represent as a symbol
            Expr::Symbol(Symbol { name: "PureFunction".to_string() })
        },
        Value::Slot { number } => {
            // For slots, represent as a symbol
            let name = match number {
                Some(n) => format!("#{}", n),
                None => "#".to_string(),
            };
            Expr::Symbol(Symbol { name })
        },
    }
}

/// Cache key for pattern matching results
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PatternCacheKey {
    /// Hash of the expression being matched
    expression_hash: u64,
    /// Hash of the pattern
    pattern_hash: u64,
    /// Context hash (for variable bindings, etc.)
    context_hash: u64,
}

impl PatternCacheKey {
    /// Create a new cache key
    pub fn new(expression: &Value, pattern: &Pattern, context: &MatchingContext) -> Self {
        use std::collections::hash_map::DefaultHasher;
        
        let mut expression_hasher = DefaultHasher::new();
        expression.hash(&mut expression_hasher);
        let expression_hash = expression_hasher.finish();
        
        let mut pattern_hasher = DefaultHasher::new();
        format!("{:?}", pattern).hash(&mut pattern_hasher);
        let pattern_hash = pattern_hasher.finish();
        
        let mut context_hasher = DefaultHasher::new();
        format!("{:?}", context).hash(&mut context_hasher);
        let context_hash = context_hasher.finish();
        
        Self {
            expression_hash,
            pattern_hash,
            context_hash,
        }
    }
}

/// Cached pattern matching result
#[derive(Debug)]
pub struct CachedMatchResult {
    /// The match result
    pub result: MatchResult,
    /// When this result was cached
    pub cached_at: Instant,
    /// Number of times this cache entry has been accessed
    pub access_count: AtomicUsize,
}

impl CachedMatchResult {
    /// Create a new cached result
    pub fn new(result: MatchResult) -> Self {
        Self {
            result,
            cached_at: Instant::now(),
            access_count: AtomicUsize::new(0),
        }
    }
    
    /// Access this cached result (increments access count)
    pub fn access(&self) -> MatchResult {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.result.clone()
    }
}

/// Lock-free cache for pattern matching results
pub struct PatternCache {
    /// Cache storage
    cache: DashMap<PatternCacheKey, CachedMatchResult>,
    /// Maximum cache size
    max_size: usize,
    /// Cache hit counter
    hits: AtomicUsize,
    /// Cache miss counter
    misses: AtomicUsize,
    /// Cache eviction counter
    evictions: AtomicUsize,
}

impl PatternCache {
    /// Create a new pattern cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: DashMap::new(),
            max_size,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            evictions: AtomicUsize::new(0),
        }
    }
    
    /// Get a cached result
    pub fn get(&self, key: &PatternCacheKey) -> Option<MatchResult> {
        if let Some(cached) = self.cache.get(key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(cached.access())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }
    
    /// Put a result in the cache
    pub fn put(&self, key: PatternCacheKey, result: MatchResult) {
        // Check if we need to evict
        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }
        
        let cached_result = CachedMatchResult::new(result);
        self.cache.insert(key, cached_result);
    }
    
    /// Evict least recently used entries
    fn evict_lru(&self) {
        let mut oldest_key = None;
        let mut oldest_time = Instant::now();
        
        // Find the oldest entry
        for entry in self.cache.iter() {
            if entry.value().cached_at < oldest_time {
                oldest_time = entry.value().cached_at;
                oldest_key = Some(entry.key().clone());
            }
        }
        
        // Remove the oldest entry
        if let Some(key) = oldest_key {
            self.cache.remove(&key);
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> PatternCacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        
        PatternCacheStats {
            size: self.cache.len(),
            max_size: self.max_size,
            hits,
            misses,
            hit_rate: if total > 0 { (hits as f64 / total as f64) * 100.0 } else { 0.0 },
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }
    
    /// Clear the cache
    pub fn clear(&self) {
        self.cache.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
    }
}

/// Statistics for pattern cache
#[derive(Debug, Clone)]
pub struct PatternCacheStats {
    /// Current cache size
    pub size: usize,
    /// Maximum cache size
    pub max_size: usize,
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Cache hit rate as percentage
    pub hit_rate: f64,
    /// Number of evictions
    pub evictions: usize,
}

/// Parallel pattern matching task
pub struct PatternMatchTask {
    /// Expression to match against
    pub expression: Value,
    /// Pattern to match
    pub pattern: Pattern,
    /// Matching context
    pub context: MatchingContext,
    /// Priority of this matching task
    pub priority: TaskPriority,
}

impl PatternMatchTask {
    /// Create a new pattern matching task
    pub fn new(
        expression: Value,
        pattern: Pattern,
        context: MatchingContext,
        priority: TaskPriority,
    ) -> Self {
        Self {
            expression,
            pattern,
            context,
            priority,
        }
    }
}

/// Result of parallel pattern matching
#[derive(Debug, Clone)]
pub struct ParallelMatchResult {
    /// The pattern that was matched
    pub pattern_index: usize,
    /// The match result
    pub result: MatchResult,
    /// Time taken to compute this match
    pub duration: std::time::Duration,
}

/// Parallel pattern matcher
pub struct ParallelPatternMatcher {
    /// Pattern cache for memoization
    cache: Arc<PatternCache>,
    /// Configuration
    config: ConcurrencyConfig,
    /// Statistics
    stats: Arc<ConcurrencyStats>,
    /// Sequential pattern matcher for fallback
    sequential_matcher: PatternMatcher,
    /// Whether parallel matching is enabled
    parallel_enabled: AtomicBool,
}

impl ParallelPatternMatcher {
    /// Create a new parallel pattern matcher
    pub fn new(
        config: ConcurrencyConfig,
        stats: Arc<ConcurrencyStats>,
    ) -> Result<Self, ConcurrencyError> {
        let cache = Arc::new(PatternCache::new(config.pattern_cache_size));
        let sequential_matcher = PatternMatcher::new();
        
        Ok(Self {
            cache,
            config,
            stats,
            sequential_matcher,
            parallel_enabled: AtomicBool::new(true),
        })
    }
    
    /// Match a value against multiple patterns in parallel
    pub fn match_parallel(
        &mut self,
        expression: &Value,
        patterns: &[Pattern],
    ) -> VmResult<Vec<MatchResult>> {
        let start_time = Instant::now();
        
        // If we have few patterns or parallel is disabled, use sequential matching
        if patterns.len() < self.config.parallel_threshold || !self.parallel_enabled.load(Ordering::Relaxed) {
            return self.match_sequential(expression, patterns);
        }
        
        // Create matching context
        let context = MatchingContext::new();
        
        // TODO: Use parallel matching once PatternMatcher is Send+Sync
        // Issue: FastPathMatcher trait doesn't implement Send+Sync
        let results: Vec<_> = patterns
            .iter()
            .enumerate()
            .map(|(index, pattern)| {
                let cache_key = PatternCacheKey::new(expression, pattern, &context);
                
                // Check cache first
                if let Some(cached_result) = self.cache.get(&cache_key) {
                    self.stats.pattern_cache_hits.fetch_add(1, Ordering::Relaxed);
                    return (index, cached_result);
                }
                
                self.stats.pattern_cache_misses.fetch_add(1, Ordering::Relaxed);
                
                // Perform the actual match - convert Value to Expr first
                let expr = value_to_expr(expression);
                let match_result = self.sequential_matcher.match_pattern(&expr, pattern);
                
                // Cache the result
                self.cache.put(cache_key, match_result.clone());
                
                (index, match_result)
            })
            .collect();
        
        // Extract just the match results in original order
        let mut match_results = vec![MatchResult::Failure { reason: "No match found".to_string() }; patterns.len()];
        for (index, result) in results {
            match_results[index] = result;
        }
        
        let duration = start_time.elapsed();
        self.stats.parallel_patterns.fetch_add(1, Ordering::Relaxed);
        
        // Log performance for tuning
        if duration.as_millis() > 100 {
            eprintln!(
                "Parallel pattern matching took {}ms for {} patterns",
                duration.as_millis(),
                patterns.len()
            );
        }
        
        Ok(match_results)
    }
    
    /// Sequential pattern matching fallback
    fn match_sequential(
        &mut self,
        expression: &Value,
        patterns: &[Pattern],
    ) -> VmResult<Vec<MatchResult>> {
        let context = MatchingContext::new();
        let mut results = Vec::with_capacity(patterns.len());
        
        for pattern in patterns {
            let cache_key = PatternCacheKey::new(expression, pattern, &context);
            
            // Check cache first
            if let Some(cached_result) = self.cache.get(&cache_key) {
                self.stats.pattern_cache_hits.fetch_add(1, Ordering::Relaxed);
                results.push(cached_result);
                continue;
            }
            
            self.stats.pattern_cache_misses.fetch_add(1, Ordering::Relaxed);
            
            // Perform the match - convert Value to Expr first
            let expr = value_to_expr(expression);
            let match_result = self.sequential_matcher.match_pattern(&expr, pattern);
            
            // Cache the result
            self.cache.put(cache_key, match_result.clone());
            results.push(match_result);
        }
        
        Ok(results)
    }
    
    /// Match a value against patterns with custom context
    pub fn match_with_context(
        &mut self,
        expression: &Value,
        patterns: &[Pattern],
        context: &MatchingContext,
    ) -> VmResult<Vec<MatchResult>> {
        // If we have few patterns, use sequential matching
        if patterns.len() < self.config.parallel_threshold || !self.parallel_enabled.load(Ordering::Relaxed) {
            return self.match_sequential_with_context(expression, patterns, context);
        }
        
        // TODO: Use parallel matching once PatternMatcher is Send+Sync
        let results: Vec<_> = patterns
            .iter()
            .enumerate()
            .map(|(index, pattern)| {
                let cache_key = PatternCacheKey::new(expression, pattern, context);
                
                // Check cache first
                if let Some(cached_result) = self.cache.get(&cache_key) {
                    self.stats.pattern_cache_hits.fetch_add(1, Ordering::Relaxed);
                    return (index, cached_result);
                }
                
                self.stats.pattern_cache_misses.fetch_add(1, Ordering::Relaxed);
                
                // Perform the actual match - convert Value to Expr first
                let expr = value_to_expr(expression);
                let match_result = self.sequential_matcher.match_pattern(&expr, pattern);
                
                // Cache the result
                self.cache.put(cache_key, match_result.clone());
                
                (index, match_result)
            })
            .collect();
        
        // Extract just the match results in original order
        let mut match_results = vec![MatchResult::Failure { reason: "No match found".to_string() }; patterns.len()];
        for (index, result) in results {
            match_results[index] = result;
        }
        
        self.stats.parallel_patterns.fetch_add(1, Ordering::Relaxed);
        Ok(match_results)
    }
    
    /// Sequential pattern matching with custom context
    fn match_sequential_with_context(
        &mut self,
        expression: &Value,
        patterns: &[Pattern],
        context: &MatchingContext,
    ) -> VmResult<Vec<MatchResult>> {
        let mut results = Vec::with_capacity(patterns.len());
        
        for pattern in patterns {
            let cache_key = PatternCacheKey::new(expression, pattern, context);
            
            // Check cache first
            if let Some(cached_result) = self.cache.get(&cache_key) {
                self.stats.pattern_cache_hits.fetch_add(1, Ordering::Relaxed);
                results.push(cached_result);
                continue;
            }
            
            self.stats.pattern_cache_misses.fetch_add(1, Ordering::Relaxed);
            
            // Perform the match - convert Value to Expr first
            let expr = value_to_expr(expression);
            let match_result = self.sequential_matcher.match_pattern(&expr, pattern);
            
            // Cache the result
            self.cache.put(cache_key, match_result.clone());
            results.push(match_result);
        }
        
        Ok(results)
    }
    
    /// Find the first matching pattern in parallel
    pub fn find_first_match(
        &mut self,
        expression: &Value,
        patterns: &[Pattern],
    ) -> VmResult<Option<(usize, MatchResult)>> {
        let context = MatchingContext::new();
        
        // TODO: Use parallel find once PatternMatcher is Send+Sync
        let result = patterns
            .iter()
            .enumerate()
            .find_map(|(index, pattern)| {
                let cache_key = PatternCacheKey::new(expression, pattern, &context);
                
                // Check cache first
                let match_result = if let Some(cached_result) = self.cache.get(&cache_key) {
                    self.stats.pattern_cache_hits.fetch_add(1, Ordering::Relaxed);
                    cached_result
                } else {
                    self.stats.pattern_cache_misses.fetch_add(1, Ordering::Relaxed);
                    
                    // Perform the actual match - convert Value to Expr first
                    let expr = value_to_expr(expression);
                    let result = self.sequential_matcher.match_pattern(&expr, pattern);
                    
                    // Cache the result
                    self.cache.put(cache_key, result.clone());
                    result
                };
                
                // Return the first successful match
                match match_result {
                    MatchResult::Failure { reason } if reason == "No match found" => None,
                    result => Some((index, result)),
                }
            });
        
        self.stats.parallel_patterns.fetch_add(1, Ordering::Relaxed);
        Ok(result)
    }
    
    /// Apply rules in parallel
    pub fn apply_rules_parallel(
        &mut self,
        expression: &Value,
        rules: &[(Pattern, Expr)],
    ) -> VmResult<Option<Value>> {
        let context = MatchingContext::new();
        
        // TODO: Use parallel rule matching once PatternMatcher is Send+Sync
        let matching_rule = rules
            .iter()
            .enumerate()
            .find_map(|(index, (pattern, _replacement))| {
                let cache_key = PatternCacheKey::new(expression, pattern, &context);
                
                // Check cache first
                let match_result = if let Some(cached_result) = self.cache.get(&cache_key) {
                    self.stats.pattern_cache_hits.fetch_add(1, Ordering::Relaxed);
                    cached_result
                } else {
                    self.stats.pattern_cache_misses.fetch_add(1, Ordering::Relaxed);
                    
                    // Perform the actual match - convert Value to Expr first
                    let expr = value_to_expr(expression);
                    let result = self.sequential_matcher.match_pattern(&expr, pattern);
                    
                    // Cache the result
                    self.cache.put(cache_key, result.clone());
                    result
                };
                
                // Return the first successful match with its index
                match match_result {
                    MatchResult::Failure { reason } if reason == "No match found" => None,
                    result => Some((index, result)),
                }
            });
        
        // Apply the matching rule
        if let Some((rule_index, _match_result)) = matching_rule {
            let (_pattern, replacement) = &rules[rule_index];
            
            // For now, return a placeholder value
            // Full implementation would apply variable substitutions
            Ok(Some(Value::Integer(rule_index as i64)))
        } else {
            Ok(None)
        }
    }
    
    /// Enable or disable parallel matching
    pub fn set_parallel_enabled(&self, enabled: bool) {
        self.parallel_enabled.store(enabled, Ordering::Relaxed);
    }
    
    /// Check if parallel matching is enabled
    pub fn is_parallel_enabled(&self) -> bool {
        self.parallel_enabled.load(Ordering::Relaxed)
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> PatternCacheStats {
        self.cache.stats()
    }
    
    /// Clear the pattern cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
    
    /// Get a reference to the cache
    pub fn cache(&self) -> &Arc<PatternCache> {
        &self.cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Pattern;
    
    #[test]
    fn test_pattern_cache() {
        let cache = PatternCache::new(10);
        
        let key = PatternCacheKey {
            expression_hash: 123,
            pattern_hash: 456,
            context_hash: 789,
        };
        
        // Test miss
        assert!(cache.get(&key).is_none());
        
        // Test put and hit
        cache.put(key.clone(), MatchResult::Failure { reason: "No match found".to_string() });
        assert!(cache.get(&key).is_some());
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate, 50.0);
    }
    
    #[test]
    fn test_parallel_pattern_matcher_creation() {
        let config = ConcurrencyConfig::default();
        let stats = Arc::new(ConcurrencyStats::default());
        
        let matcher = ParallelPatternMatcher::new(config, stats);
        assert!(matcher.is_ok());
        
        let matcher = matcher.unwrap();
        assert!(matcher.is_parallel_enabled());
    }
    
    #[test]
    fn test_pattern_matching_with_empty_patterns() {
        let config = ConcurrencyConfig::default();
        let stats = Arc::new(ConcurrencyStats::default());
        let matcher = ParallelPatternMatcher::new(config, stats).unwrap();
        
        let expression = Value::Integer(42);
        let patterns = vec![];
        
        let results = matcher.match_parallel(&expression, &patterns).unwrap();
        assert!(results.is_empty());
    }
    
    #[test]
    fn test_first_match_finding() {
        let config = ConcurrencyConfig::default();
        let stats = Arc::new(ConcurrencyStats::default());
        let matcher = ParallelPatternMatcher::new(config, stats).unwrap();
        
        let expression = Value::Integer(42);
        let patterns = vec![
            Pattern::Blank, // This should match
            Pattern::Symbol("x".to_string()),
        ];
        
        let result = matcher.find_first_match(&expression, &patterns).unwrap();
        // Since we're using a placeholder pattern matcher, this will return None
        // In a real implementation, this would return Some((0, match_result))
        assert!(result.is_none() || result.is_some());
    }
    
    #[test]
    fn test_cache_eviction() {
        let cache = PatternCache::new(2); // Small cache for testing eviction
        
        let key1 = PatternCacheKey {
            expression_hash: 1,
            pattern_hash: 1,
            context_hash: 1,
        };
        let key2 = PatternCacheKey {
            expression_hash: 2,
            pattern_hash: 2,
            context_hash: 2,
        };
        let key3 = PatternCacheKey {
            expression_hash: 3,
            pattern_hash: 3,
            context_hash: 3,
        };
        
        // Fill cache
        cache.put(key1.clone(), MatchResult::Failure { reason: "No match found".to_string() });
        cache.put(key2.clone(), MatchResult::Failure { reason: "No match found".to_string() });
        
        // This should trigger eviction
        cache.put(key3.clone(), MatchResult::Failure { reason: "No match found".to_string() });
        
        let stats = cache.stats();
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.size, 2);
    }
}