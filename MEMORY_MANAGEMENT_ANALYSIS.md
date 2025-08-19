# Lyra Memory Management Analysis & Design

## Executive Summary

This document provides a comprehensive analysis of the current memory usage patterns in the Lyra symbolic computation engine and proposes an advanced memory management system to achieve the target 30%+ memory reduction. The analysis covers ~27,424 lines of source code across 42 files with 1,033 instances of smart pointer usage (Arc, Box, Rc, RefCell).

## Current Memory Usage Analysis

### 1. Value Enum Memory Profile

The core `vm::Value` enum is the primary memory allocation hotspot:

```rust
#[derive(Debug, Clone)]
pub enum Value {
    Integer(i64),           // 8 bytes + discriminant
    Real(f64),              // 8 bytes + discriminant  
    String(String),         // 24 bytes (ptr + cap + len) + heap allocation
    Symbol(String),         // 24 bytes + heap allocation
    List(Vec<Value>),       // 24 bytes + heap allocation for Vec<T>
    Function(String),       // 24 bytes + heap allocation
    Boolean(bool),          // 1 byte + discriminant
    Missing,                // discriminant only
    LyObj(LyObj),          // Foreign object wrapper - significant heap usage
    Quote(Box<crate::ast::Expr>), // 8 bytes + heap allocation
    Pattern(crate::ast::Pattern), // Variable size
}
```

**Memory Hotspots Identified:**
1. **String types** (Symbol, String, Function): Heavy heap allocation for common symbols like "x", "Plus", "Times"
2. **List operations**: Frequent `Vec<Value>` allocations during symbolic computations
3. **Foreign objects**: `LyObj` wrapper with `Arc<dyn Foreign>` indirection
4. **AST expressions**: Recursive `Box<Expr>` allocations in `Quote` values

### 2. AST Expression Memory Patterns

```rust
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum Expr {
    Function {
        head: Box<Expr>,    // 8 bytes + recursive allocation
        args: Vec<Expr>,    // 24 bytes + Vec allocation
    },
    List(Vec<Expr>),        // Frequent allocations
    Rule { lhs: Box<Expr>, rhs: Box<Expr>, delayed: bool },
    // ... other variants with similar patterns
}
```

**Key Issues:**
- **Deep nesting**: Complex expressions create deep Box<Expr> chains
- **Repeated structures**: Common patterns like `Plus[x, y]` allocated separately each time
- **Pattern matching overhead**: Temporary Value allocations during pattern matching

### 3. Pattern Matching Memory Analysis

From `pattern_matcher.rs`:
- **Variable bindings**: `HashMap<String, Value>` allocated per match attempt
- **Match frames**: `Vec<MatchFrame>` for recursive matching state
- **String interning**: Basic attempt but not comprehensive

```rust
fn create_optimized_bindings() -> HashMap<String, Value> {
    HashMap::with_capacity(4)  // Pre-allocated, but still per-match overhead
}
```

### 4. Foreign Object Memory Usage

Current foreign objects (`ForeignSeries`, `ForeignTable`, `ForeignTensor`) use `Arc` for thread safety:

```rust
pub struct ForeignSeries {
    pub data: Arc<Vec<Value>>,    // Double indirection: Arc -> Vec -> Value
    pub dtype: SeriesType,
    pub length: usize,            // Redundant with data.len()
}
```

**Memory Inefficiencies:**
- Double indirection through Arc -> Vec -> Value
- Redundant metadata storage
- Full Value enum overhead for homogeneous data

### 5. VM Stack and Execution Memory

```rust
pub struct VirtualMachine {
    pub stack: Vec<Value>,          // Main execution stack
    pub constants: Vec<Value>,      // Constant pool
    pub symbols: HashMap<String, usize>, // Symbol table
    // ... other fields
}
```

**Analysis:**
- Stack grows dynamically with no memory pool reuse
- Constants pool duplicates common values
- Symbol table uses full String keys

## Benchmarking Current Memory Patterns

The existing `memory_benchmarks.rs` reveals allocation patterns:

1. **Pattern Matching**: 5-50 allocations per match operation
2. **Rule Application**: 10-100 allocations for complex rule chains  
3. **Expression Creation**: 3-15 allocations for nested expressions
4. **Foreign Objects**: 20-50 allocations for tensor operations

## Proposed Memory Management System Design

### 1. Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   lyra-memory   │───▶│  Memory Pools   │───▶│  Arena Alloc    │
│   (main crate)  │    │  (by type/size) │    │  (bump alloc)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ String Interner │    │ Value Recycling │    │   GC Manager    │
│ (static cache)  │    │  (type pools)   │    │ (cycle detect)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2. Core Components

#### A. String Interning System

```rust
pub struct StringInterner {
    common_symbols: &'static [&'static str],
    dynamic_cache: Arc<DashMap<String, &'static str>>,
    arena: Arena<String>,
}

impl StringInterner {
    const COMMON_SYMBOLS: &'static [&'static str] = &[
        "x", "y", "z", "a", "b", "c", "n", "i", "j", "k",
        "Plus", "Times", "Power", "Equal", "List", "Function",
        "Integer", "Real", "String", "Symbol", "True", "False"
    ];
    
    pub fn intern(&self, s: &str) -> InternedString {
        // Return static reference for common symbols
        // Use arena allocation for dynamic strings
    }
}
```

**Expected Savings**: 40-60% reduction in String allocations

#### B. Value Memory Pools

```rust
pub struct ValuePools {
    integers: Pool<i64>,
    reals: Pool<f64>, 
    lists: Pool<Vec<ManagedValue>>,
    symbols: Pool<InternedString>,
}

#[repr(C)]
pub struct ManagedValue {
    tag: ValueTag,      // 1 byte discriminant
    data: ValueData,    // 8 byte union
}

#[repr(C)]
union ValueData {
    integer: i64,
    real: f64,
    string_ref: InternedString,
    list_ref: ListRef,
    object_ref: ObjectRef,
}
```

**Expected Savings**: 25-35% reduction in Value allocations

#### C. Arena Allocation for Temporary Computations

```rust
pub struct ComputationArena {
    value_arena: Arena<ManagedValue>,
    expr_arena: Arena<ManagedExpr>, 
    bindings_arena: Arena<BindingMap>,
    current_scope: ScopeId,
}

impl ComputationArena {
    pub fn with_scope<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let scope = self.push_scope();
        let result = f(self);
        self.pop_scope(scope);
        result
    }
}
```

**Expected Savings**: 50-70% reduction in temporary allocations

#### D. Reference-Counted Expressions

```rust
pub struct RcExpr {
    inner: Rc<ExprNode>,
}

struct ExprNode {
    kind: ExprKind,
    cached_hash: Option<u64>,
    metadata: ExprMetadata,
}

enum ExprKind {
    Function { head: RcExpr, args: SmallVec<[RcExpr; 4]> },
    Symbol(InternedString),
    Number(NumberValue),
    List(SmallVec<[RcExpr; 8]>),
}
```

**Expected Savings**: 30-50% reduction in expression allocations through sharing

### 3. MemoryManaged Trait Interface

```rust
pub trait MemoryManaged: Send + Sync {
    /// Allocate a managed value in the appropriate pool
    fn alloc_value(value: Value) -> ManagedValue;
    
    /// Allocate a managed expression with potential sharing
    fn alloc_expr(expr: Expr) -> RcExpr;
    
    /// Create a temporary computation scope
    fn with_temp_scope<T>(f: impl FnOnce() -> T) -> T;
    
    /// Trigger garbage collection and return bytes freed
    fn collect_garbage() -> usize;
    
    /// Get memory usage statistics
    fn memory_stats() -> MemoryStats;
}

pub struct MemoryStats {
    pub total_allocated: usize,
    pub pool_usage: HashMap<String, PoolStats>,
    pub arena_usage: ArenaStats,
    pub gc_cycles: u64,
    pub last_gc_freed: usize,
}
```

### 4. Integration with Existing Systems

#### A. VM Integration

```rust
pub struct ManagedVirtualMachine {
    ip: usize,
    stack: PooledVec<ManagedValue>,           // Pool-allocated stack
    constants: PooledVec<ManagedValue>,       // Interned constants
    symbols: InternedSymbolTable,             // String-interned symbols
    memory_manager: MemoryManager,
    // ... other fields
}
```

#### B. Foreign Object Integration

```rust
pub struct ManagedLyObj {
    inner: Arc<dyn ManagedForeign>,
    metadata: ObjectMetadata,
}

pub trait ManagedForeign: Foreign {
    fn memory_usage(&self) -> usize;
    fn collect_unused(&mut self) -> usize;
    fn compress(&mut self) -> Result<(), ForeignError>;
}
```

### 5. Memory Pool Design

```rust
pub struct TypedPool<T> {
    available: Vec<Box<T>>,
    allocated: Vec<Weak<T>>,
    max_size: usize,
    allocation_stats: AllocationStats,
}

impl<T> TypedPool<T> {
    pub fn get(&mut self) -> PooledRef<T> {
        if let Some(item) = self.available.pop() {
            PooledRef::new(item, self)
        } else {
            PooledRef::new(Box::new(T::default()), self)
        }
    }
    
    pub fn recycle(&mut self, item: Box<T>) {
        if self.available.len() < self.max_size {
            // Reset item to default state
            item.reset();
            self.available.push(item);
        }
        // Otherwise let it drop
    }
}
```

### 6. Cycle Detection for Recursive Structures

```rust
pub struct CycleDetector {
    visit_stack: Vec<ObjectId>,
    visited: HashSet<ObjectId>,
    cycles_found: Vec<Vec<ObjectId>>,
}

impl CycleDetector {
    pub fn detect_cycles(&mut self, root: &ManagedValue) -> Vec<CycleInfo> {
        // Depth-first traversal with stack-based cycle detection
        // Mark objects for potential collection
    }
}
```

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Create `lyra-memory` crate structure
- [ ] Implement basic string interning system
- [ ] Design ManagedValue representation
- [ ] Create memory pool infrastructure

### Phase 2: Core Integration (Weeks 3-4)  
- [ ] Integrate string interning into VM and AST
- [ ] Replace Value enum with ManagedValue
- [ ] Implement arena allocation for temporary computations
- [ ] Add memory pools for common types

### Phase 3: Advanced Features (Weeks 5-6)
- [ ] Implement reference-counted expressions
- [ ] Add cycle detection for recursive structures
- [ ] Optimize foreign object memory usage
- [ ] Add comprehensive memory statistics

### Phase 4: Optimization & Validation (Weeks 7-8)
- [ ] Performance tuning and memory pool sizing
- [ ] Benchmark against target 30% reduction
- [ ] Integration testing with existing test suite
- [ ] Documentation and API stabilization

## Expected Memory Reduction Breakdown

| Component | Current Usage | Optimized Usage | Reduction |
|-----------|---------------|-----------------|-----------|
| String Allocations | 100% | 45% | 55% |
| Value Allocations | 100% | 70% | 30% |
| Temporary Computations | 100% | 35% | 65% |
| Expression Sharing | 100% | 60% | 40% |
| Foreign Objects | 100% | 75% | 25% |
| **Overall** | **100%** | **65%** | **35%** |

## Benchmark Strategy

### Memory Usage Benchmarks
1. **Baseline Measurement**: Current memory usage for standard workloads
2. **Component Testing**: Individual memory pool and arena performance
3. **Integration Testing**: End-to-end memory usage with real symbolic computations
4. **Stress Testing**: Large expression trees and tensor operations

### Performance Benchmarks
1. **Allocation Speed**: Time to allocate/deallocate managed values
2. **GC Performance**: Garbage collection overhead and effectiveness
3. **Cache Efficiency**: Memory access patterns and cache utilization
4. **Thread Safety**: Multi-threaded memory management performance

### Success Metrics
- **Primary**: ≥30% reduction in peak memory usage
- **Secondary**: ≤10% performance regression in computation speed
- **Tertiary**: Maintained thread safety and API compatibility

## Risk Analysis & Mitigation

### Technical Risks
1. **API Compatibility**: Extensive refactoring may break existing code
   - *Mitigation*: Gradual migration with compatibility layers
2. **Performance Regression**: Memory management overhead may slow computations
   - *Mitigation*: Extensive benchmarking and optimization
3. **Complexity**: Advanced memory management increases code complexity
   - *Mitigation*: Comprehensive testing and documentation

### Memory Safety Risks
1. **Use-After-Free**: Arena allocation with incorrect lifetimes
   - *Mitigation*: Rust's ownership system and careful scope management
2. **Memory Leaks**: Cycle detection may miss complex reference cycles  
   - *Mitigation*: Conservative GC approach and leak detection tools

## Conclusion

The proposed memory management system provides a comprehensive approach to achieving the target 30%+ memory reduction through:

1. **String interning** for common symbols and function names
2. **Memory pools** for frequent Value allocations  
3. **Arena allocation** for temporary computations
4. **Reference counting** for expression sharing
5. **Cycle detection** for memory safety

The design maintains compatibility with the existing foreign object architecture while providing significant memory efficiency improvements. The phased implementation approach allows for gradual integration and validation of each component.

The estimated 35% overall memory reduction exceeds the target requirement while maintaining the performance and safety characteristics of the current system.