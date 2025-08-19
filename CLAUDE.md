# Lyra Development Guidelines

## Test-Driven Development (TDD) Requirements

**CRITICAL: ALL changes to this codebase MUST follow strict Test-Driven Development practices.**

### TDD Process
1. **RED**: Write a failing test first
2. **GREEN**: Write minimal code to make the test pass
3. **REFACTOR**: Clean up code while keeping tests green

### Before Making Any Code Changes
- [ ] Write comprehensive tests that describe the expected behavior
- [ ] Ensure tests fail initially (RED phase)
- [ ] Implement only enough code to make tests pass (GREEN phase)
- [ ] Refactor if needed while maintaining green tests

### Test Requirements
- **Unit Tests**: Every function/method must have unit tests
- **Integration Tests**: Components must be tested together
- **Snapshot Tests**: Use `insta` for complex output verification
- **Property Tests**: Where applicable, test with random inputs

### Running Tests
```bash
# Run all tests
cargo test

# Run tests with coverage
cargo test --all-features

# Update snapshots
cargo insta review
```

### Development Commands
```bash
# Check formatting
cargo fmt --check

# Run clippy lints
cargo clippy -- -D warnings

# Build project
cargo build

# Run CLI
cargo run -- --help
```

### Architecture Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Source    │───▶│    Lexer    │───▶│   Parser    │
│    Code     │    │  (Tokens)   │    │   (AST)     │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│     VM      │◀───│  Compiler   │◀───│  Desugarer  │
│ (Execution) │    │ (Bytecode)  │    │ (Core AST)  │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Module Structure
- `lexer`: Tokenization of source code
- `parser`: AST generation from tokens
- `ast`: AST node definitions and utilities
- `compiler`: Bytecode generation from AST
- `vm`: Virtual machine for bytecode execution
- `runtime`: Built-in functions and standard library
- `error`: Error types and handling

### Language Syntax (Wolfram-Inspired)
- Function calls: `f[x, y]`
- Lists: `{1, 2, 3}`
- Patterns: `x_`, `x__`, `x_Integer`
- Rules: `x -> x^2`, `x :> RandomReal[]`
- Replacement: `expr /. rule`
- Definitions: `f[x_] = x^2`, `f[x_] := RandomReal[]`

### Test Organization
```
tests/
├── lexer/          # Lexer unit tests
├── parser/         # Parser unit tests
├── compiler/       # Compiler unit tests
├── vm/             # VM unit tests
├── integration/    # End-to-end tests
└── snapshots/      # Insta snapshot files
```

### Quality Gates
Before any commit:
1. All tests must pass (`cargo test`)
2. Code must be formatted (`cargo fmt`)
3. No clippy warnings (`cargo clippy`)
4. Documentation must be updated if APIs change

### Performance Guidelines
- Optimize only after correctness is established
- Use benchmarks to measure performance improvements
- Profile before optimizing
- Maintain test coverage during optimization

### Error Handling
- Use `Result<T, Error>` for fallible operations
- Provide clear error messages with context
- Test error cases thoroughly
- Use `thiserror` for error definitions

### VM Design Principles

**CRITICAL: The VM must remain simple and focused on symbolic computation.**

#### VM Simplicity Requirements
- **Minimal Core Types**: Keep VM Value enum as small as possible
- **No Feature Pollution**: Complex features must NOT be added to VM core
- **Symbolic First**: VM should handle symbolic expressions, not imperative logic
- **Foreign Object Pattern**: Use LyObj/Foreign trait for complex types outside VM

#### Prohibited in VM Core
- ❌ **Complex Data Structures**: Arrays, matrices, images, tables belong in stdlib as Foreign objects
- ❌ **Async/Concurrency Primitives**: Futures, channels, threads stay in stdlib  
- ❌ **I/O Operations**: File handling, network, database access via stdlib only
- ❌ **Domain-Specific Types**: ML models, signal processing, optimization belong in Foreign objects

#### Required Approach for New Features
1. **Evaluate if feature can be implemented as stdlib function**
2. **Use Foreign objects for complex state or external resources**
3. **Keep VM focused on symbolic expression evaluation**
4. **Maintain clear separation between VM core and stdlib functionality**

#### Examples of Correct Design
```rust
// ✅ GOOD: Complex functionality as Foreign object
pub struct AsyncFuture { value: Value }
impl Foreign for AsyncFuture { /* methods */ }
Value::LyObj(LyObj::new(Box::new(future)))

// ❌ BAD: Adding to VM core types
pub enum Value {
    Future(Box<Value>),  // NO - pollutes VM
}
```

Remember: **VM simplicity over convenience. Tests first, implementation second, always verify tests pass before proceeding.**

### Concurrency System Architecture

**COMPLETE: Production-ready async/concurrency system implemented as Foreign objects.**

#### Core Concurrency Components
- **ThreadPool**: Thread pool management for parallel task execution
- **Channel**: Thread-safe message passing with bounded/unbounded variants  
- **Future**: Async computation results with Promise/Await patterns
- **Advanced Patterns**: ParallelMap, ParallelReduce, Pipeline processing

#### Concurrency API Reference

**ThreadPool Operations:**
```rust
ThreadPool[]                    // Create with default 4 workers
ThreadPool[worker_count]        // Create with specified workers
pool.submit(function, args...)  // Submit task, returns task ID
pool.getResult(taskId)         // Get result (non-blocking)
pool.isCompleted(taskId)       // Check completion status
pool.workerCount()             // Get worker thread count
pool.pendingTasks()            // Get queued task count
```

**Channel Operations:**
```rust
Channel[]                      // Create unbounded channel
BoundedChannel[capacity]       // Create bounded channel
Send[channel, value]           // Send value (blocking)
Receive[channel]               // Receive value (blocking) 
TrySend[channel, value]        // Non-blocking send
TryReceive[channel]            // Non-blocking receive
ChannelClose[channel]          // Close channel
channel.capacity()             // Get capacity (Missing if unbounded)
channel.len()                  // Current message count
channel.isEmpty()              // Check if empty
channel.isClosed()             // Check if closed
```

**Parallel Execution Patterns:**
```rust
// Adaptive parallel execution
Parallel[{function, list}]                    // Auto-optimized chunking
Parallel[{function, list}, threadpool]        // Custom ThreadPool

// Optimized patterns  
ParallelMap[function, list]                   // Parallel map operation
ParallelReduce[function, list]               // Tree-like reduction
Pipeline[channels, functions]                // Multi-stage processing

// Legacy future support
Parallel[{future1, future2, ...}]           // Resolve futures
```

**Future/Promise Operations:**
```rust
Promise[value]                 // Create resolved Future
Await[future]                  // Extract Future value
AsyncFunction[function]        // Wrap function as async
All[{future1, future2, ...}]  // Wait for all futures
Any[{future1, future2, ...}]  // Return first completed
```

#### Performance Characteristics

**Adaptive Work Distribution:**
- Automatically calculates optimal chunk sizes based on worker count
- Target: 3 chunks per worker for ideal load balancing
- Switches between individual and chunked processing based on data size
- Minimal memory copying during work distribution

**Parallel Processing Performance:**
- **ParallelMap**: O(n/p) time complexity with p workers
- **ParallelReduce**: O(log n) depth, O(n) work, maximum parallelism
- **Pipeline**: Real-time processing with backpressure handling
- **ThreadPool**: Lock-free task queue with efficient work stealing

**Memory Efficiency:**
- Zero-copy operations where possible
- Minimal overhead for concurrent operations
- Proper resource cleanup on object destruction
- Thread-safe reference counting for shared objects

#### Concurrency Design Principles

**Foreign Object Architecture:**
- All concurrency primitives live outside VM as Foreign objects
- No VM type pollution - maintains clean symbolic computation focus
- Uses LyObj wrapper for seamless integration with VM
- Complete separation between VM core and concurrency logic

**Thread Safety:**
- Built on battle-tested crossbeam-channel library
- All concurrent operations are thread-safe by design
- Proper synchronization without performance penalties
- Race condition and deadlock prevention through design

**Error Handling:**
- Graceful degradation on worker thread failures
- Clean error propagation from concurrent tasks
- Timeout and cancellation support
- Resource cleanup on partial failures

#### Usage Examples

**Basic ThreadPool Usage:**
```wolfram
(* Create thread pool and submit tasks *)
pool = ThreadPool[4]
taskId = pool.submit(Add, 10, 20)
result = pool.getResult(taskId)  (* → 30 *)
```

**Producer-Consumer Pattern:**
```wolfram
(* Create channel and coordinate between tasks *)
ch = BoundedChannel[10]
Send[ch, data]
result = Receive[ch]
```

**Advanced Parallel Processing:**
```wolfram
(* Parallel map-reduce operations *)
data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
squared = ParallelMap[Square, data]
sum = ParallelReduce[Add, squared]
```

**Pipeline Processing:**
```wolfram
(* Multi-stage processing pipeline *)
channels = {inputCh, processCh, outputCh}
functions = {ProcessStage1, ProcessStage2}
pipeline = Pipeline[channels, functions]
```

#### Integration with Existing Systems

The concurrency system integrates seamlessly with:
- **Gradual Typing**: Full type safety for Future[T] and other concurrent types
- **Pattern Matching**: Concurrent operations work with existing pattern system  
- **Error Handling**: Consistent error propagation through VM error system
- **Memory Management**: Proper integration with VM memory management
- **Standard Library**: All concurrency functions registered as stdlib functions

#### Testing and Validation

**Comprehensive Test Coverage:**
- Unit tests for all Foreign object methods
- Integration tests for cross-component interaction
- Concurrency tests for race conditions and thread safety
- Performance tests demonstrating speedup over sequential execution
- Stress tests under high concurrent loads
- Edge case tests for error handling and resource cleanup

**Production Readiness:**
- All tests pass under concurrent conditions
- Demonstrable performance improvements over sequential execution
- Robust error handling and graceful failure recovery
- Memory usage optimization and leak prevention
- Stability under high concurrent workloads