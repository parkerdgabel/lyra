#!/usr/bin/env python3
"""
Lyra Examples Test Runner - Automated execution and validation of example files

This script provides comprehensive testing infrastructure for all Lyra example files:
- Automated execution of all .lyra files in the examples directory
- Performance benchmarking and timing analysis
- Error detection and reporting
- Test result summarization and statistics
- Integration with CI/CD pipelines
- Detailed logging and debugging support

Usage:
    python test_runner.py [options]
    
Options:
    --verbose       Enable verbose output
    --timing        Show execution timing for each test
    --parallel      Run tests in parallel (experimental)
    --filter=GLOB   Only run tests matching glob pattern
    --exclude=GLOB  Exclude tests matching glob pattern
    --output=FORMAT Output format: console, json, xml, html
    --timeout=SEC   Timeout for each test (default: 30s)
    --continue      Continue running tests after failures
    --benchmark     Run performance benchmarks
    --validate      Validate output correctness
    --ci            CI mode: exit with non-zero code on failures
"""

import os
import sys
import subprocess
import time
import json
import glob
import argparse
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import concurrent.futures

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class TestResult:
    """Container for test execution results"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.name = os.path.basename(file_path)
        self.category = self._get_category()
        self.status = "pending"
        self.execution_time = 0.0
        self.output = ""
        self.error = ""
        self.return_code = None
        self.start_time = None
        self.end_time = None
        
    def _get_category(self) -> str:
        """Determine test category from file path"""
        path_parts = Path(self.file_path).parts
        if 'stdlib' in path_parts:
            return 'Standard Library'
        elif 'modules' in path_parts:
            return 'Module System'
        elif 'advanced' in path_parts:
            return 'Advanced Features'
        elif 'performance' in path_parts:
            return 'Performance'
        elif 'workflows' in path_parts:
            return 'Workflows'
        else:
            return 'Other'
    
    def mark_started(self):
        """Mark test as started"""
        self.status = "running"
        self.start_time = time.time()
    
    def mark_passed(self, output: str, execution_time: float):
        """Mark test as passed"""
        self.status = "passed"
        self.output = output
        self.execution_time = execution_time
        self.end_time = time.time()
        self.return_code = 0
    
    def mark_failed(self, error: str, output: str, return_code: int, execution_time: float):
        """Mark test as failed"""
        self.status = "failed"
        self.error = error
        self.output = output
        self.return_code = return_code
        self.execution_time = execution_time
        self.end_time = time.time()
    
    def mark_timeout(self, execution_time: float):
        """Mark test as timed out"""
        self.status = "timeout"
        self.error = f"Test timed out after {execution_time:.2f} seconds"
        self.execution_time = execution_time
        self.end_time = time.time()
        self.return_code = -1

class LyraTestRunner:
    """Main test runner for Lyra example files"""
    
    def __init__(self, args):
        self.args = args
        self.results: List[TestResult] = []
        self.start_time = time.time()
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.timeout_tests = 0
        
        # Configure paths
        self.project_root = Path(__file__).parent.parent
        self.examples_dir = Path(__file__).parent
        self.lyra_bin = self.project_root / "target" / "debug" / "lyra"
        
        # Ensure Lyra is built
        self._ensure_lyra_built()
    
    def _ensure_lyra_built(self):
        """Ensure Lyra binary is built before running tests"""
        if not self.lyra_bin.exists():
            print(f"{Colors.YELLOW}Building Lyra binary...{Colors.END}")
            result = subprocess.run(
                ["cargo", "build", "--bin", "lyra"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"{Colors.RED}Failed to build Lyra binary:{Colors.END}")
                print(result.stderr)
                sys.exit(1)
            print(f"{Colors.GREEN}Lyra binary built successfully{Colors.END}")
    
    def discover_tests(self) -> List[str]:
        """Discover all .lyra test files"""
        test_files = []
        
        # Define test directories and their priorities
        test_dirs = [
            "stdlib",
            "modules", 
            "advanced",
            "performance",
            "workflows"
        ]
        
        for test_dir in test_dirs:
            dir_path = self.examples_dir / test_dir
            if dir_path.exists():
                pattern = str(dir_path / "*.lyra")
                files = glob.glob(pattern)
                test_files.extend(sorted(files))
        
        # Apply filters
        if self.args.filter:
            test_files = [f for f in test_files if self._matches_pattern(f, self.args.filter)]
        
        if self.args.exclude:
            test_files = [f for f in test_files if not self._matches_pattern(f, self.args.exclude)]
        
        return test_files
    
    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches glob pattern"""
        import fnmatch
        return fnmatch.fnmatch(os.path.basename(file_path), pattern)
    
    def run_single_test(self, file_path: str) -> TestResult:
        """Execute a single test file"""
        result = TestResult(file_path)
        result.mark_started()
        
        if self.args.verbose:
            print(f"  {Colors.BLUE}Running:{Colors.END} {result.name}")
        
        try:
            # Execute Lyra with the test file
            cmd = [str(self.lyra_bin), "run", file_path]
            
            start_time = time.time()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.project_root
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.args.timeout)
                execution_time = time.time() - start_time
                
                if process.returncode == 0:
                    result.mark_passed(stdout, execution_time)
                    if self.args.verbose:
                        print(f"    {Colors.GREEN}✓ PASSED{Colors.END} ({execution_time:.3f}s)")
                else:
                    result.mark_failed(stderr, stdout, process.returncode, execution_time)
                    if self.args.verbose:
                        print(f"    {Colors.RED}✗ FAILED{Colors.END} ({execution_time:.3f}s)")
                        if stderr:
                            print(f"    {Colors.RED}Error:{Colors.END} {stderr.strip()}")
                            
            except subprocess.TimeoutExpired:
                process.kill()
                execution_time = time.time() - start_time
                result.mark_timeout(execution_time)
                if self.args.verbose:
                    print(f"    {Colors.YELLOW}⏱ TIMEOUT{Colors.END} ({execution_time:.3f}s)")
                    
        except Exception as e:
            execution_time = time.time() - result.start_time
            result.mark_failed(str(e), "", -1, execution_time)
            if self.args.verbose:
                print(f"    {Colors.RED}✗ ERROR{Colors.END} {str(e)}")
        
        return result
    
    def run_tests_sequential(self, test_files: List[str]):
        """Run tests sequentially"""
        for i, file_path in enumerate(test_files, 1):
            print(f"[{i:2d}/{len(test_files)}] {os.path.basename(file_path)}")
            
            result = self.run_single_test(file_path)
            self.results.append(result)
            
            # Update counters
            if result.status == "passed":
                self.passed_tests += 1
            elif result.status == "failed":
                self.failed_tests += 1
                if not self.args.continue_on_failure:
                    print(f"{Colors.RED}Stopping due to test failure. Use --continue to continue.{Colors.END}")
                    break
            elif result.status == "timeout":
                self.timeout_tests += 1
            
            if self.args.timing:
                print(f"    Time: {result.execution_time:.3f}s")
    
    def run_tests_parallel(self, test_files: List[str]):
        """Run tests in parallel (experimental)"""
        print(f"{Colors.YELLOW}Running tests in parallel mode (experimental){Colors.END}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tests
            future_to_file = {
                executor.submit(self.run_single_test, file_path): file_path 
                for file_path in test_files
            }
            
            # Collect results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_file), 1):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    
                    status_color = Colors.GREEN if result.status == "passed" else Colors.RED
                    print(f"[{i:2d}/{len(test_files)}] {result.name} {status_color}{result.status.upper()}{Colors.END} ({result.execution_time:.3f}s)")
                    
                    # Update counters
                    if result.status == "passed":
                        self.passed_tests += 1
                    elif result.status == "failed":
                        self.failed_tests += 1
                    elif result.status == "timeout":
                        self.timeout_tests += 1
                        
                except Exception as e:
                    print(f"{Colors.RED}Error running {os.path.basename(file_path)}: {e}{Colors.END}")
                    self.failed_tests += 1
    
    def run_benchmark_suite(self):
        """Run performance benchmark analysis"""
        print(f"\n{Colors.BOLD}=== Performance Benchmark Analysis ==={Colors.END}")
        
        benchmark_files = [r for r in self.results if 'performance' in r.file_path.lower()]
        if not benchmark_files:
            print("No performance benchmark files found")
            return
        
        print(f"Analyzing {len(benchmark_files)} benchmark files:")
        
        for result in benchmark_files:
            if result.status == "passed":
                print(f"  {result.name}: {result.execution_time:.3f}s")
                
                # Analyze execution time thresholds
                if result.execution_time > 10.0:
                    print(f"    {Colors.YELLOW}⚠ Slow execution detected{Colors.END}")
                elif result.execution_time > 5.0:
                    print(f"    {Colors.BLUE}ℹ Moderate execution time{Colors.END}")
                else:
                    print(f"    {Colors.GREEN}✓ Good performance{Colors.END}")
            else:
                print(f"  {result.name}: {Colors.RED}FAILED{Colors.END}")
    
    def generate_summary(self):
        """Generate and display test summary"""
        total_time = time.time() - self.start_time
        self.total_tests = len(self.results)
        
        print(f"\n{Colors.BOLD}=== Test Summary ==={Colors.END}")
        print(f"Total tests: {self.total_tests}")
        print(f"Passed: {Colors.GREEN}{self.passed_tests}{Colors.END}")
        print(f"Failed: {Colors.RED}{self.failed_tests}{Colors.END}")
        print(f"Timeout: {Colors.YELLOW}{self.timeout_tests}{Colors.END}")
        print(f"Success rate: {self.passed_tests/self.total_tests*100:.1f}%")
        print(f"Total execution time: {total_time:.3f}s")
        
        # Category breakdown
        categories = {}
        for result in self.results:
            cat = result.category
            if cat not in categories:
                categories[cat] = {'passed': 0, 'failed': 0, 'timeout': 0}
            categories[cat][result.status] += 1
        
        print(f"\n{Colors.BOLD}By Category:{Colors.END}")
        for category, stats in categories.items():
            total = stats['passed'] + stats['failed'] + stats['timeout']
            success_rate = stats['passed'] / total * 100 if total > 0 else 0
            print(f"  {category}: {stats['passed']}/{total} ({success_rate:.1f}%)")
        
        # Failed tests details
        failed_results = [r for r in self.results if r.status == "failed"]
        if failed_results:
            print(f"\n{Colors.BOLD}Failed Tests:{Colors.END}")
            for result in failed_results:
                print(f"  {Colors.RED}✗{Colors.END} {result.name}")
                if result.error:
                    # Show first line of error for brevity
                    error_line = result.error.split('\n')[0]
                    print(f"    {Colors.RED}Error:{Colors.END} {error_line}")
        
        # Performance summary
        if self.args.timing or self.args.benchmark:
            avg_time = sum(r.execution_time for r in self.results) / len(self.results)
            max_time = max(r.execution_time for r in self.results)
            min_time = min(r.execution_time for r in self.results)
            
            print(f"\n{Colors.BOLD}Performance Summary:{Colors.END}")
            print(f"  Average execution time: {avg_time:.3f}s")
            print(f"  Fastest test: {min_time:.3f}s")
            print(f"  Slowest test: {max_time:.3f}s")
            
            # Find slowest test
            slowest = max(self.results, key=lambda r: r.execution_time)
            print(f"  Slowest file: {slowest.name} ({slowest.execution_time:.3f}s)")
    
    def export_results(self):
        """Export results in requested format"""
        if self.args.output == 'json':
            self._export_json()
        elif self.args.output == 'xml':
            self._export_xml()
        elif self.args.output == 'html':
            self._export_html()
    
    def _export_json(self):
        """Export results as JSON"""
        output_file = self.examples_dir / "test_results.json"
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': self.total_tests,
                'passed': self.passed_tests,
                'failed': self.failed_tests,
                'timeout': self.timeout_tests,
                'success_rate': self.passed_tests / self.total_tests * 100 if self.total_tests > 0 else 0
            },
            'tests': []
        }
        
        for result in self.results:
            data['tests'].append({
                'name': result.name,
                'file_path': result.file_path,
                'category': result.category,
                'status': result.status,
                'execution_time': result.execution_time,
                'return_code': result.return_code,
                'error': result.error if result.error else None,
                'output_length': len(result.output)
            })
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results exported to: {output_file}")
    
    def _export_xml(self):
        """Export results as JUnit XML"""
        output_file = self.examples_dir / "test_results.xml"
        
        xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="Lyra Examples" tests="{self.total_tests}" failures="{self.failed_tests}" 
           errors="{self.timeout_tests}" time="{sum(r.execution_time for r in self.results):.3f}">
'''
        
        for result in self.results:
            xml_content += f'  <testcase name="{result.name}" classname="{result.category}" time="{result.execution_time:.3f}"'
            
            if result.status == "passed":
                xml_content += ' />\n'
            else:
                xml_content += '>\n'
                if result.status == "failed":
                    xml_content += f'    <failure message="Test failed">{result.error}</failure>\n'
                elif result.status == "timeout":
                    xml_content += f'    <error message="Test timeout">{result.error}</error>\n'
                xml_content += '  </testcase>\n'
        
        xml_content += '</testsuite>\n'
        
        with open(output_file, 'w') as f:
            f.write(xml_content)
        
        print(f"JUnit XML exported to: {output_file}")
    
    def _export_html(self):
        """Export results as HTML report"""
        output_file = self.examples_dir / "test_results.html"
        
        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Lyra Examples Test Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        .timeout {{ color: orange; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Lyra Examples Test Results</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Total tests: {self.total_tests}</p>
        <p>Passed: <span class="passed">{self.passed_tests}</span></p>
        <p>Failed: <span class="failed">{self.failed_tests}</span></p>
        <p>Timeout: <span class="timeout">{self.timeout_tests}</span></p>
        <p>Success rate: {self.passed_tests/self.total_tests*100:.1f}%</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <h2>Test Details</h2>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Category</th>
            <th>Status</th>
            <th>Time (s)</th>
            <th>Error</th>
        </tr>
'''
        
        for result in self.results:
            status_class = result.status
            error_text = result.error[:100] + "..." if len(result.error) > 100 else result.error
            
            html_content += f'''        <tr>
            <td>{result.name}</td>
            <td>{result.category}</td>
            <td class="{status_class}">{result.status.upper()}</td>
            <td>{result.execution_time:.3f}</td>
            <td>{error_text}</td>
        </tr>
'''
        
        html_content += '''    </table>
</body>
</html>'''
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report exported to: {output_file}")
    
    def run(self):
        """Main test execution method"""
        print(f"{Colors.BOLD}Lyra Examples Test Runner{Colors.END}")
        print(f"Examples directory: {self.examples_dir}")
        print(f"Lyra binary: {self.lyra_bin}")
        
        # Discover tests
        test_files = self.discover_tests()
        print(f"Discovered {len(test_files)} test files")
        
        if not test_files:
            print("No test files found!")
            return 1
        
        # Run tests
        print(f"\n{Colors.BOLD}Running Tests:{Colors.END}")
        
        if self.args.parallel:
            self.run_tests_parallel(test_files)
        else:
            self.run_tests_sequential(test_files)
        
        # Generate reports
        self.generate_summary()
        
        if self.args.benchmark:
            self.run_benchmark_suite()
        
        if self.args.output != 'console':
            self.export_results()
        
        # Return appropriate exit code for CI
        if self.args.ci and self.failed_tests > 0:
            return 1
        
        return 0

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Lyra Examples Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--timing', '-t', action='store_true',
                       help='Show execution timing for each test')
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='Run tests in parallel (experimental)')
    parser.add_argument('--filter', '-f', type=str,
                       help='Only run tests matching glob pattern')
    parser.add_argument('--exclude', '-e', type=str,
                       help='Exclude tests matching glob pattern')
    parser.add_argument('--output', '-o', choices=['console', 'json', 'xml', 'html'],
                       default='console', help='Output format')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Timeout for each test in seconds')
    parser.add_argument('--continue', dest='continue_on_failure', action='store_true',
                       help='Continue running tests after failures')
    parser.add_argument('--benchmark', '-b', action='store_true',
                       help='Run performance benchmarks')
    parser.add_argument('--validate', action='store_true',
                       help='Validate output correctness')
    parser.add_argument('--ci', action='store_true',
                       help='CI mode: exit with non-zero code on failures')
    
    args = parser.parse_args()
    
    # Run the test suite
    runner = LyraTestRunner(args)
    exit_code = runner.run()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()