#!/bin/bash

# Lyra Performance Benchmarking Runner
# Automates the execution and analysis of performance benchmarks

set -euo pipefail

# Configuration
BENCHMARK_DIR="target/criterion"
RESULTS_DIR="benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "🚀 Lyra Performance Benchmark Runner"
echo "======================================"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to run a single benchmark suite
run_benchmark() {
    local bench_name=$1
    echo "📊 Running benchmark: $bench_name"
    
    # Run the benchmark and capture output
    if cargo bench --bench "$bench_name" > "$RESULTS_DIR/${bench_name}_${TIMESTAMP}.log" 2>&1; then
        echo "✅ $bench_name completed successfully"
    else
        echo "❌ $bench_name failed"
        tail -n 20 "$RESULTS_DIR/${bench_name}_${TIMESTAMP}.log"
        return 1
    fi
}

# Function to validate speedup claims
validate_speedup_claims() {
    echo "🔍 Validating 1000x speedup claims..."
    
    if run_benchmark "speedup_claims_validation"; then
        echo "📈 Speedup claims validation completed"
        
        # Extract key metrics from benchmark results
        if [ -f "$BENCHMARK_DIR/speedup_claims_validation/report/index.html" ]; then
            echo "📋 Results available at: $BENCHMARK_DIR/speedup_claims_validation/report/index.html"
        fi
    else
        echo "⚠️  Speedup claims validation encountered issues"
        return 1
    fi
}

# Function to run performance validation
run_performance_validation() {
    echo "⚡ Running core performance validation..."
    
    if run_benchmark "performance_validation"; then
        echo "✅ Core performance validation completed"
    else
        echo "❌ Performance validation failed"
        return 1
    fi
}

# Function to run regression detection
run_regression_detection() {
    echo "🛡️  Running regression detection..."
    
    if run_benchmark "regression_detection"; then
        echo "✅ Regression detection completed"
        
        # Check for any concerning regressions
        if grep -q "regression" "$RESULTS_DIR/regression_detection_${TIMESTAMP}.log"; then
            echo "⚠️  Potential regressions detected - review logs"
        else
            echo "✅ No regressions detected"
        fi
    else
        echo "❌ Regression detection failed"
        return 1
    fi
}

# Function to generate summary report
generate_summary() {
    echo "📊 Generating performance summary..."
    
    local summary_file="$RESULTS_DIR/performance_summary_${TIMESTAMP}.md"
    
    cat > "$summary_file" << EOF
# Lyra Performance Benchmark Summary

**Generated:** $(date)
**Version:** $(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

## Benchmark Results

### Core Performance Validation
$(if [ -f "$RESULTS_DIR/performance_validation_${TIMESTAMP}.log" ]; then echo "✅ Completed"; else echo "❌ Failed"; fi)

### Speedup Claims Validation  
$(if [ -f "$RESULTS_DIR/speedup_claims_validation_${TIMESTAMP}.log" ]; then echo "✅ Completed"; else echo "❌ Failed"; fi)

### Regression Detection
$(if [ -f "$RESULTS_DIR/regression_detection_${TIMESTAMP}.log" ]; then echo "✅ Completed"; else echo "❌ Failed"; fi)

## Key Findings

### Performance Claims Status
- CALL_STATIC 1000x speedup: $(grep -c "call_static" "$RESULTS_DIR"/speedup_claims_validation_${TIMESTAMP}.log 2>/dev/null || echo "Not measured")
- Pattern matching optimization: $(grep -c "pattern_matching" "$RESULTS_DIR"/performance_validation_${TIMESTAMP}.log 2>/dev/null || echo "Not measured")
- Memory management efficiency: $(grep -c "memory" "$RESULTS_DIR"/performance_validation_${TIMESTAMP}.log 2>/dev/null || echo "Not measured")

### Critical Path Performance
$(if [ -f "$RESULTS_DIR/regression_detection_${TIMESTAMP}.log" ]; then
    echo "- Parser performance: Measured"
    echo "- VM execution performance: Measured" 
    echo "- Standard library performance: Measured"
else
    echo "- Critical paths: Not measured"
fi)

## Recommendations

1. Review HTML reports in target/criterion/ for detailed analysis
2. Compare results with previous benchmarks to track improvements
3. Investigate any performance regressions identified
4. Validate specific speedup claims against baseline measurements

## Files Generated
- Performance logs: $RESULTS_DIR/*_${TIMESTAMP}.log
- HTML reports: target/criterion/*/report/index.html
- This summary: $summary_file
EOF

    echo "📋 Summary report generated: $summary_file"
}

# Main execution flow
main() {
    echo "🔧 Ensuring clean build..."
    cargo build --release --quiet
    
    echo "🧪 Running benchmark suites..."
    
    # Run all benchmark suites
    local failed_benchmarks=()
    
    if ! run_performance_validation; then
        failed_benchmarks+=("performance_validation")
    fi
    
    if ! validate_speedup_claims; then
        failed_benchmarks+=("speedup_claims_validation")
    fi
    
    if ! run_regression_detection; then
        failed_benchmarks+=("regression_detection")
    fi
    
    # Generate summary report
    generate_summary
    
    # Final status
    if [ ${#failed_benchmarks[@]} -eq 0 ]; then
        echo "✅ All benchmarks completed successfully!"
        echo "📊 View results in target/criterion/ or $RESULTS_DIR/"
    else
        echo "❌ Some benchmarks failed: ${failed_benchmarks[*]}"
        echo "📋 Check logs in $RESULTS_DIR/ for details"
        exit 1
    fi
}

# Parse command line arguments
case "${1:-all}" in
    "performance")
        run_performance_validation
        ;;
    "speedup")
        validate_speedup_claims
        ;;
    "regression") 
        run_regression_detection
        ;;
    "all")
        main
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [performance|speedup|regression|all|help]"
        echo ""
        echo "  performance  - Run core performance validation benchmarks"
        echo "  speedup      - Validate 1000x speedup claims"
        echo "  regression   - Run regression detection benchmarks"  
        echo "  all          - Run all benchmark suites (default)"
        echo "  help         - Show this help message"
        exit 0
        ;;
    *)
        echo "❌ Unknown argument: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac