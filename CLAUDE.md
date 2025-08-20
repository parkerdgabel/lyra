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

## Threading Model and Concurrency Architecture

**COMPLETE: Production-ready work-stealing thread pool with NUMA optimization.**

### Thread Safety Guarantees

**VM Core Thread Safety:**
- **Single-Threaded VM**: VM core remains single-threaded and simple
- **Immutable Values**: Core Value types are immutable and safe to share
- **Foreign Object Isolation**: All concurrent operations isolated in Foreign objects
- **No Shared Mutable State**: VM state never shared between threads

**Foreign Object Thread Safety:**
- **Send + Sync Required**: All Foreign objects must implement Send + Sync
- **Internal Synchronization**: Foreign objects handle their own thread safety
- **Lock-Free Where Possible**: Performance-critical paths use lock-free algorithms
- **Deadlock Prevention**: No circular locking dependencies in design

**Memory Safety Guarantees:**
- **Rust Ownership**: Compile-time prevention of data races
- **Reference Counting**: Thread-safe reference counting for shared objects
- **Arena Allocation**: Thread-local arenas for temporary computation
- **No Dangling Pointers**: Lifetime system prevents use-after-free

### Work-Stealing Thread Pool Architecture

**Core Components:**
```rust
WorkStealingScheduler {
    // Per-worker local queues (LIFO for cache locality)
    workers: Vec<Worker>,
    
    // Global queue for load balancing (FIFO)
    global_queue: Injector<Task>,
    
    // Inter-worker communication
    stealers: Vec<Stealer<Task>>,
    
    // NUMA-aware topology management
    numa_topology: NumaTopology,
}
```

**Performance Characteristics:**
- **Linear Scaling**: 2-5x speedup on multi-core systems
- **NUMA Optimization**: Memory bandwidth optimization on large systems
- **Adaptive Load Balancing**: Automatic work redistribution
- **Cache Efficiency**: LIFO local queues improve cache utilization

### Concurrency Patterns and Best Practices

**Producer-Consumer Pattern:**
```wolfram
(* Create bounded channel for backpressure *)
ch = BoundedChannel[100]
producer = ThreadPool[1]
consumer = ThreadPool[1]

(* Producer task *)
prodTask = producer.submit(Function[
    For[i = 1, i <= 1000, i++,
        Send[ch, ProcessData[i]]
    ]
])

(* Consumer task *)
consTask = consumer.submit(Function[
    results = {};
    While[True,
        data = TryReceive[ch];
        If[data === Missing, Break[]];
        AppendTo[results, ProcessResult[data]]
    ];
    results
])
```

**Parallel Map-Reduce Pattern:**
```wolfram
(* Adaptive chunking for optimal performance *)
data = Range[1, 1000000]
chunks = AdaptiveChunk[data, ThreadCount[]]

(* Parallel map phase *)
mapped = ParallelMap[Square, data]

(* Parallel reduce phase with tree reduction *)
result = ParallelReduce[Add, mapped]
```

**Pipeline Processing Pattern:**
```wolfram
(* Multi-stage processing pipeline *)
pipeline = Pipeline[{
    Function[data, FilterData[data]],      (* Stage 1 *)
    Function[data, TransformData[data]],   (* Stage 2 *)
    Function[data, AggregateData[data]]    (* Stage 3 *)
}]

ProcessPipeline[pipeline, inputData]
```

### Resource Management

**Automatic Resource Cleanup:**
- ThreadPool objects automatically shut down worker threads on drop
- Channels automatically close and notify all waiters
- Future objects clean up async resources on completion
- Memory pools return allocations when objects are dropped

**Resource Limits:**
```wolfram
(* Configure resource limits *)
SetThreadPoolLimits[maxWorkers -> 16, queueSize -> 10000]
SetChannelLimits[maxBufferSize -> 1000000]
SetMemoryLimits[maxAsyncMemory -> Gigabytes[2]]
```

**Monitoring and Metrics:**
```wolfram
(* Get performance statistics *)
stats = GetConcurrencyStats[]
Print["Work steal efficiency: ", stats["workStealEfficiency"]]
Print["Cache hit rate: ", stats["cacheHitRate"]]
Print["Memory usage: ", stats["memoryUsage"]]
```

### Error Handling in Concurrent Code

**Error Propagation:**
- Errors in worker threads captured and propagated to caller
- Panics in worker threads isolated and converted to errors
- Timeout handling for long-running async operations
- Graceful degradation when workers fail

**Error Recovery Patterns:**
```wolfram
(* Retry mechanism with exponential backoff *)
result = Retry[
    risky_operation,
    maxAttempts -> 3,
    backoff -> Exponential[baseDelay -> Milliseconds[100]]
]

(* Circuit breaker pattern *)
breaker = CircuitBreaker[
    failureThreshold -> 5,
    timeout -> Seconds[30]
]
safeResult = breaker.call(unstable_operation)
```

### Performance Tuning Guidelines

**Thread Pool Sizing:**
- Default: Number of CPU cores
- CPU-bound tasks: cores
- I/O-bound tasks: 2-4x cores  
- NUMA systems: Multiple pools per node

**Memory Configuration:**
```wolfram
(* NUMA-aware configuration *)
SetNumaPolicy[
    memoryBinding -> "local",
    workerPlacement -> "spread",
    cacheAlignment -> True
]

(* Memory pool tuning *)
ConfigureMemoryPools[
    arenaSize -> Megabytes[64],
    poolGrowthFactor -> 1.5,
    maxPoolSize -> Gigabytes[1]
]
```

**Monitoring Performance:**
```wolfram
(* Enable performance monitoring *)
EnableConcurrencyMonitoring[True]

(* Benchmark concurrent operations *)
benchmark = BenchmarkConcurrent[
    operation,
    threadCounts -> {1, 2, 4, 8, 16},
    iterations -> 1000
]
PlotScalability[benchmark]
```

### Advanced Concurrency Features

**Actor Model Support:**
```wolfram
(* Create actor system *)
actorSystem = ActorSystem[workerCount -> 8]

(* Define actor behavior *)
counterActor = Actor[
    state -> 0,
    receive -> Function[{message, state},
        Switch[message,
            "increment", state + 1,
            "get", state,
            "reset", 0
        ]
    ]
]

(* Spawn and interact with actors *)
counter = actorSystem.spawn(counterActor)
counter ! "increment"
result = Ask[counter, "get"]  (* → 1 *)
```

**Software Transactional Memory:**
```wolfram
(* Transactional updates *)
account1 = TransactionalRef[1000]
account2 = TransactionalRef[500]

transfer = Transaction[
    balance1 = ReadRef[account1];
    balance2 = ReadRef[account2];
    If[balance1 >= amount,
        WriteRef[account1, balance1 - amount];
        WriteRef[account2, balance2 + amount];
        True,
        False
    ]
]

success = CommitTransaction[transfer]
```

### Troubleshooting Async Operations

**Common Issues and Solutions:**

**Deadlock Detection:**
```wolfram
(* Enable deadlock detection *)
SetDebugMode["deadlockDetection" -> True]

(* Detect circular wait conditions *)
deadlocks = DetectDeadlocks[]
If[Length[deadlocks] > 0,
    Print["Deadlock detected: ", deadlocks]
]
```

**Performance Bottlenecks:**
```wolfram
(* Profile concurrent operations *)
profile = ProfileConcurrency[operation]
Print["Bottlenecks: ", profile["bottlenecks"]]
Print["Contention points: ", profile["contention"]]
```

**Memory Leaks:**
```wolfram
(* Monitor memory usage *)
baseline = GetMemoryUsage[]
RunConcurrentOperation[operation]
leak = GetMemoryUsage[] - baseline
If[leak > threshold,
    Print["Potential memory leak: ", leak]
]
```

### Migration from Sequential Code

**Step-by-Step Migration:**

1. **Identify Parallelizable Operations:**
   - Independent computations
   - Map operations on lists
   - Reduce operations with associative functions

2. **Replace with Parallel Equivalents:**
   ```wolfram
   (* Before: Sequential *)
   result = Map[expensiveFunction, largeList]
   
   (* After: Parallel *)
   result = ParallelMap[expensiveFunction, largeList]
   ```

3. **Add Resource Management:**
   ```wolfram
   (* Create thread pool for reuse *)
   pool = ThreadPool[8]
   
   (* Use pool for multiple operations *)
   result1 = pool.parallelMap(func1, data1)
   result2 = pool.parallelMap(func2, data2)
   ```

4. **Handle Errors Gracefully:**
   ```wolfram
   (* Add error handling *)
   result = TryCatch[
       ParallelMap[riskyFunction, data],
       error -> {
           Print["Parallel operation failed: ", error];
           Map[riskyFunction, data]  (* Fallback to sequential *)
       }
   ]
   ```