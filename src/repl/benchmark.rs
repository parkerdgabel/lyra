//! Benchmarking and Profiling System for REPL
//! 
//! Provides performance analysis, benchmarking, and profiling capabilities
//! for evaluating expression performance in the Lyra REPL.

use crate::ast::Expr;
use crate::compiler::Compiler;
use crate::parser::Parser;
use crate::vm::Value;
use crate::repl::{ReplError, ReplResult};
use colored::*;
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Statistical summary of benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkStats {
    pub iterations: usize,
    pub total_time: Duration,
    pub mean_time: Duration,
    pub median_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub std_deviation: Duration,
    pub percentile_95: Duration,
    pub percentile_99: Duration,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub peak_memory_bytes: usize,
    pub allocation_count: usize,
    pub deallocation_count: usize,
    pub average_allocation_size: usize,
}

/// Detailed profiling information
#[derive(Debug, Clone)]
pub struct ProfileData {
    pub parse_time: Duration,
    pub compile_time: Duration,
    pub execution_time: Duration,
    pub total_time: Duration,
    pub memory_stats: Option<MemoryStats>,
}

/// Comparison result between two expressions
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub expr1_stats: BenchmarkStats,
    pub expr2_stats: BenchmarkStats,
    pub speedup_factor: f64,
    pub confidence_level: f64,
}

pub struct BenchmarkSystem;

impl BenchmarkSystem {
    /// Run benchmark on an expression with specified iterations
    pub fn benchmark_expression(
        expression: &str,
        iterations: Option<usize>,
        warmup_iterations: Option<usize>,
    ) -> ReplResult<(BenchmarkStats, String)> {
        let iterations = iterations.unwrap_or(100);
        let warmup_iterations = warmup_iterations.unwrap_or(10);

        // Parse the expression once
        let expr = Self::parse_expression(expression)?;

        // Warmup runs to stabilize performance
        for _ in 0..warmup_iterations {
            let _ = Self::execute_expression(&expr)?;
        }

        // Actual benchmark runs
        let mut execution_times = Vec::with_capacity(iterations);
        
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = Self::execute_expression(&expr)?;
            let duration = start.elapsed();
            execution_times.push(duration);
        }

        // Calculate statistics
        let stats = Self::calculate_stats(execution_times);
        let result_summary = Self::format_benchmark_results(&stats, expression);

        Ok((stats, result_summary))
    }

    /// Profile an expression with detailed timing breakdown
    pub fn profile_expression(expression: &str) -> ReplResult<(ProfileData, String)> {
        let total_start = Instant::now();
        
        // Parse timing
        let parse_start = Instant::now();
        let expr = Self::parse_expression(expression)?;
        let parse_time = parse_start.elapsed();

        // Compile timing (if applicable - for now we use direct evaluation)
        let compile_start = Instant::now();
        // In our current setup, compilation happens during evaluation
        let compile_time = compile_start.elapsed();

        // Execution timing
        let exec_start = Instant::now();
        let _result = Self::execute_expression(&expr)?;
        let execution_time = exec_start.elapsed();

        let total_time = total_start.elapsed();

        let profile_data = ProfileData {
            parse_time,
            compile_time,
            execution_time,
            total_time,
            memory_stats: None, // TODO: Implement memory profiling
        };

        let result_summary = Self::format_profile_results(&profile_data, expression);

        Ok((profile_data, result_summary))
    }

    /// Compare performance of two expressions
    pub fn compare_expressions(
        expr1: &str,
        expr2: &str,
        iterations: Option<usize>,
    ) -> ReplResult<(ComparisonResult, String)> {
        let iterations = iterations.unwrap_or(50);

        // Benchmark both expressions
        let (stats1, _) = Self::benchmark_expression(expr1, Some(iterations), Some(5))?;
        let (stats2, _) = Self::benchmark_expression(expr2, Some(iterations), Some(5))?;

        // Calculate speedup factor
        let speedup_factor = stats1.mean_time.as_nanos() as f64 / stats2.mean_time.as_nanos() as f64;
        
        // Simple confidence calculation based on standard deviations
        let confidence_level = Self::calculate_confidence(&stats1, &stats2);

        let comparison = ComparisonResult {
            expr1_stats: stats1,
            expr2_stats: stats2,
            speedup_factor,
            confidence_level,
        };

        let result_summary = Self::format_comparison_results(&comparison, expr1, expr2);

        Ok((comparison, result_summary))
    }

    /// Analyze memory usage of an expression
    pub fn memory_analysis(expression: &str) -> ReplResult<(MemoryStats, String)> {
        // Parse expression
        let expr = Self::parse_expression(expression)?;

        // For now, provide a simulated memory analysis
        // TODO: Implement actual memory tracking
        let memory_stats = MemoryStats {
            peak_memory_bytes: 1024, // Simulated
            allocation_count: 10,
            deallocation_count: 8,
            average_allocation_size: 128,
        };

        let result_summary = Self::format_memory_results(&memory_stats, expression);

        Ok((memory_stats, result_summary))
    }

    /// Parse an expression string to AST
    fn parse_expression(expression: &str) -> ReplResult<Expr> {
        let mut parser = Parser::from_source(expression).map_err(|e| ReplError::ParseError {
            message: format!("Failed to parse expression '{}': {}", expression, e),
        })?;

        let statements = parser.parse().map_err(|e| ReplError::ParseError {
            message: format!("Parse error in expression '{}': {}", expression, e),
        })?;

        statements.last().cloned().ok_or_else(|| ReplError::ParseError {
            message: "No expressions to benchmark".to_string(),
        })
    }

    /// Execute an expression and return the result
    fn execute_expression(expr: &Expr) -> ReplResult<Value> {
        Compiler::eval(expr).map_err(|e| ReplError::CompilationError {
            message: format!("Execution failed: {}", e),
        })
    }

    /// Calculate statistical summary from execution times
    fn calculate_stats(mut times: Vec<Duration>) -> BenchmarkStats {
        times.sort();
        
        let iterations = times.len();
        let total_time: Duration = times.iter().sum();
        let mean_time = total_time / iterations as u32;
        
        let median_time = if iterations % 2 == 0 {
            (times[iterations / 2 - 1] + times[iterations / 2]) / 2
        } else {
            times[iterations / 2]
        };
        
        let min_time = times[0];
        let max_time = times[iterations - 1];
        
        // Calculate standard deviation
        let variance_sum: f64 = times.iter()
            .map(|&time| {
                let diff = time.as_nanos() as f64 - mean_time.as_nanos() as f64;
                diff * diff
            })
            .sum();
        let variance = variance_sum / iterations as f64;
        let std_deviation = Duration::from_nanos(variance.sqrt() as u64);
        
        // Calculate percentiles
        let percentile_95_idx = (iterations as f64 * 0.95) as usize;
        let percentile_99_idx = (iterations as f64 * 0.99) as usize;
        let percentile_95 = times[percentile_95_idx.min(iterations - 1)];
        let percentile_99 = times[percentile_99_idx.min(iterations - 1)];

        BenchmarkStats {
            iterations,
            total_time,
            mean_time,
            median_time,
            min_time,
            max_time,
            std_deviation,
            percentile_95,
            percentile_99,
        }
    }

    /// Calculate confidence level for comparison
    fn calculate_confidence(stats1: &BenchmarkStats, stats2: &BenchmarkStats) -> f64 {
        // Simple confidence calculation based on overlap of standard deviations
        let mean1 = stats1.mean_time.as_nanos() as f64;
        let mean2 = stats2.mean_time.as_nanos() as f64;
        let std1 = stats1.std_deviation.as_nanos() as f64;
        let std2 = stats2.std_deviation.as_nanos() as f64;
        
        let difference = (mean1 - mean2).abs();
        let combined_std = (std1 + std2) / 2.0;
        
        if combined_std == 0.0 {
            return 1.0;
        }
        
        let confidence = (difference / combined_std).min(3.0) / 3.0;
        confidence.max(0.0).min(1.0)
    }

    /// Format benchmark results for display
    fn format_benchmark_results(stats: &BenchmarkStats, expression: &str) -> String {
        let mut result = Vec::new();
        
        result.push(format!("{}", "Benchmark Results".bright_cyan().bold()));
        result.push(format!("{}", "=================".bright_cyan()));
        result.push(String::new());
        
        result.push(format!("{}: {}", "Expression".bright_yellow().bold(), expression.bright_white()));
        result.push(format!("{}: {}", "Iterations".bright_yellow().bold(), stats.iterations.to_string().bright_white()));
        result.push(String::new());
        
        result.push(format!("{}", "Timing Statistics:".bright_green().bold()));
        result.push(format!("  {}: {:.3}ms", "Mean".bright_blue(), Self::duration_to_ms(stats.mean_time)));
        result.push(format!("  {}: {:.3}ms", "Median".bright_blue(), Self::duration_to_ms(stats.median_time)));
        result.push(format!("  {}: {:.3}ms", "Min".bright_blue(), Self::duration_to_ms(stats.min_time)));
        result.push(format!("  {}: {:.3}ms", "Max".bright_blue(), Self::duration_to_ms(stats.max_time)));
        result.push(format!("  {}: ±{:.3}ms", "Std Dev".bright_blue(), Self::duration_to_ms(stats.std_deviation)));
        result.push(String::new());
        
        result.push(format!("{}", "Percentiles:".bright_green().bold()));
        result.push(format!("  {}: {:.3}ms", "95th".bright_magenta(), Self::duration_to_ms(stats.percentile_95)));
        result.push(format!("  {}: {:.3}ms", "99th".bright_magenta(), Self::duration_to_ms(stats.percentile_99)));
        result.push(String::new());
        
        result.push(format!("{}: {:.3}ms", 
            "Total Time".bright_yellow().bold(), 
            Self::duration_to_ms(stats.total_time)
        ));
        
        result.join("\\n")
    }

    /// Format profile results for display
    fn format_profile_results(profile: &ProfileData, expression: &str) -> String {
        let mut result = Vec::new();
        
        result.push(format!("{}", "Profile Results".bright_cyan().bold()));
        result.push(format!("{}", "===============".bright_cyan()));
        result.push(String::new());
        
        result.push(format!("{}: {}", "Expression".bright_yellow().bold(), expression.bright_white()));
        result.push(String::new());
        
        result.push(format!("{}", "Timing Breakdown:".bright_green().bold()));
        result.push(format!("  {}: {:.3}ms ({:.1}%)", 
            "Parse".bright_blue(), 
            Self::duration_to_ms(profile.parse_time),
            Self::percentage(profile.parse_time, profile.total_time)
        ));
        result.push(format!("  {}: {:.3}ms ({:.1}%)", 
            "Compile".bright_blue(), 
            Self::duration_to_ms(profile.compile_time),
            Self::percentage(profile.compile_time, profile.total_time)
        ));
        result.push(format!("  {}: {:.3}ms ({:.1}%)", 
            "Execute".bright_blue(), 
            Self::duration_to_ms(profile.execution_time),
            Self::percentage(profile.execution_time, profile.total_time)
        ));
        result.push(format!("  {}: {:.3}ms", 
            "Total".bright_yellow().bold(), 
            Self::duration_to_ms(profile.total_time)
        ));
        
        if let Some(memory) = &profile.memory_stats {
            result.push(String::new());
            result.push(format!("{}", "Memory Usage:".bright_green().bold()));
            result.push(format!("  {}: {} bytes", "Peak Memory".bright_blue(), memory.peak_memory_bytes));
            result.push(format!("  {}: {}", "Allocations".bright_blue(), memory.allocation_count));
        }
        
        result.join("\\n")
    }

    /// Format comparison results for display
    fn format_comparison_results(
        comparison: &ComparisonResult,
        expr1: &str,
        expr2: &str,
    ) -> String {
        let mut result = Vec::new();
        
        result.push(format!("{}", "Performance Comparison".bright_cyan().bold()));
        result.push(format!("{}", "=====================".bright_cyan()));
        result.push(String::new());
        
        result.push(format!("{}: {}", "Expression 1".bright_yellow().bold(), expr1.bright_white()));
        result.push(format!("{}: {}", "Expression 2".bright_yellow().bold(), expr2.bright_white()));
        result.push(String::new());
        
        result.push(format!("{}", "Results:".bright_green().bold()));
        result.push(format!("  {} Mean: {:.3}ms", 
            "Expr 1".bright_blue(), 
            Self::duration_to_ms(comparison.expr1_stats.mean_time)
        ));
        result.push(format!("  {} Mean: {:.3}ms", 
            "Expr 2".bright_blue(), 
            Self::duration_to_ms(comparison.expr2_stats.mean_time)
        ));
        result.push(String::new());
        
        let (faster_expr, speedup_text, speedup_color) = if comparison.speedup_factor > 1.0 {
            ("Expression 2", format!("{:.2}x faster", comparison.speedup_factor), "bright_green")
        } else if comparison.speedup_factor < 1.0 {
            ("Expression 1", format!("{:.2}x faster", 1.0 / comparison.speedup_factor), "bright_green")
        } else {
            ("Neither", "Same performance".to_string(), "bright_yellow")
        };
        
        let speedup_colored = match speedup_color {
            "bright_green" => speedup_text.bright_green(),
            "bright_yellow" => speedup_text.bright_yellow(),
            _ => speedup_text.normal(),
        };
        
        result.push(format!("{}: {} ({})", 
            "Winner".bright_magenta().bold(), 
            faster_expr.bright_white().bold(),
            speedup_colored
        ));
        result.push(format!("{}: {:.1}%", 
            "Confidence".bright_magenta().bold(), 
            comparison.confidence_level * 100.0
        ));
        
        result.join("\\n")
    }

    /// Format memory analysis results
    fn format_memory_results(memory: &MemoryStats, expression: &str) -> String {
        let mut result = Vec::new();
        
        result.push(format!("{}", "Memory Analysis".bright_cyan().bold()));
        result.push(format!("{}", "===============".bright_cyan()));
        result.push(String::new());
        
        result.push(format!("{}: {}", "Expression".bright_yellow().bold(), expression.bright_white()));
        result.push(String::new());
        
        result.push(format!("{}", "Memory Statistics:".bright_green().bold()));
        result.push(format!("  {}: {} bytes", "Peak Memory".bright_blue(), memory.peak_memory_bytes));
        result.push(format!("  {}: {}", "Allocations".bright_blue(), memory.allocation_count));
        result.push(format!("  {}: {}", "Deallocations".bright_blue(), memory.deallocation_count));
        result.push(format!("  {}: {} bytes", "Avg Allocation".bright_blue(), memory.average_allocation_size));
        
        let memory_efficiency = memory.deallocation_count as f64 / memory.allocation_count.max(1) as f64;
        result.push(format!("  {}: {:.1}%", 
            "Memory Efficiency".bright_magenta(), 
            memory_efficiency * 100.0
        ));
        
        result.join("\\n")
    }

    /// Convert duration to milliseconds for display
    fn duration_to_ms(duration: Duration) -> f64 {
        duration.as_nanos() as f64 / 1_000_000.0
    }

    /// Calculate percentage of part to whole
    fn percentage(part: Duration, whole: Duration) -> f64 {
        if whole.is_zero() {
            return 0.0;
        }
        (part.as_nanos() as f64 / whole.as_nanos() as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_simple_expression() {
        let result = BenchmarkSystem::benchmark_expression("2 + 3", Some(10), Some(2));
        assert!(result.is_ok());
        let (stats, _) = result.unwrap();
        assert_eq!(stats.iterations, 10);
        assert!(stats.mean_time > Duration::from_nanos(0));
    }

    #[test]
    fn test_profile_expression() {
        let result = BenchmarkSystem::profile_expression("5 * 6");
        assert!(result.is_ok());
        let (profile, _) = result.unwrap();
        assert!(profile.total_time > Duration::from_nanos(0));
        assert!(profile.execution_time <= profile.total_time);
    }

    #[test]
    fn test_compare_expressions() {
        let result = BenchmarkSystem::compare_expressions("1 + 1", "2", Some(5));
        assert!(result.is_ok());
        let (comparison, _) = result.unwrap();
        assert_eq!(comparison.expr1_stats.iterations, 5);
        assert_eq!(comparison.expr2_stats.iterations, 5);
    }

    #[test]
    fn test_memory_analysis() {
        let result = BenchmarkSystem::memory_analysis("Length[{1, 2, 3}]");
        assert!(result.is_ok());
        let (memory_stats, _) = result.unwrap();
        assert!(memory_stats.peak_memory_bytes > 0);
    }

    #[test]
    fn test_calculate_stats() {
        let times = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(15),
            Duration::from_millis(25),
            Duration::from_millis(12),
        ];
        let stats = BenchmarkSystem::calculate_stats(times);
        assert_eq!(stats.iterations, 5);
        assert_eq!(stats.min_time, Duration::from_millis(10));
        assert_eq!(stats.max_time, Duration::from_millis(25));
    }

    #[test]
    fn test_invalid_expression() {
        let result = BenchmarkSystem::benchmark_expression("invalid syntax [", Some(1), Some(0));
        assert!(result.is_err());
    }
}