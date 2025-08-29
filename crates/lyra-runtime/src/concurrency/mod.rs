//! Concurrency building blocks used by the evaluator and stdlib.
//!
//! - pool: a small global thread pool and a ThreadLimiter for bounded parallelism.
//! - futures: spawn/await/cancel helpers with per-task cancellation and optional time budgets.
//! - channels: basic mpsc-style channels integrated into the evaluator.
//! - actors: minimal actor framework (message mailbox + behavior).
//! - scope: cancellation/time budget/thread limits that propagate to nested work.

pub mod pool;
pub mod futures;
pub mod channels;
pub mod actors;
pub mod scope;
