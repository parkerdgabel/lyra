//! Evaluator orchestration.
//!
//! The evaluator coordinates expression evaluation and delegates specialized
//! behavior to focused modules:
//! - core::call: call-pipeline mechanics (Sequence splicing, Hold*, Flat,
//!   Orderless, Listable) used just before invoking builtins.
//! - core::rewrite: rule-driven semantics (Sub/Up/Down/OwnValues, Replace*,
//!   Thread) applied during evaluation.
//! - concurrency::*: futures, channels, actors, scopes, and thread pool.
//! - core::schema_explain: Explain/Schema utilities.
//! - core::introspection: builtins/docs introspection helpers.
//!
//! This file stays lean and focused on orchestration and dispatch.

use crate::attrs::Attributes;
use lyra_core::value::Value;
use lyra_rewrite::defs::{DefKind, DefinitionStore};
use lyra_rewrite::rule::RuleSet;
use std::collections::HashMap;
use std::sync::mpsc::Receiver;
use std::sync::OnceLock;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
//
use std::time::{Duration, Instant};

/// Signature for native builtin implementations invoked by the evaluator.
pub(crate) type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

//

use crate::concurrency::pool::{spawn_task, ThreadLimiter};

//

//

//

pub use crate::core::docs::DocEntry;

/// Stateful evaluator that orchestrates rewrite semantics, call pipelines,
/// concurrency, and environment during expression evaluation.
pub struct Evaluator {
    builtins: HashMap<String, (NativeFn, Attributes)>,
    // Documentation registry: function name -> summary, params, examples
    pub(crate) docs: HashMap<String, DocEntry>,
    pub(crate) env: HashMap<String, Value>,
    pub(crate) tasks: HashMap<i64, crate::concurrency::futures::TaskInfo>, // minimal future registry
    pub(crate) next_task_id: i64,
    pub(crate) cancel_token: Option<Arc<AtomicBool>>, // cooperative cancellation
    pub(crate) thread_limiter: Option<Arc<ThreadLimiter>>, // scope-wide thread budget
    pub(crate) deadline: Option<Instant>,             // scope-wide deadline
    pub(crate) trace_enabled: bool,
    pub(crate) trace_steps: Vec<Value>,
    defs: DefinitionStore,
    pub(crate) current_span: Option<(usize, usize)>,
}

static DEFAULT_REGISTRAR: OnceLock<fn(&mut Evaluator)> = OnceLock::new();

/// Set a function invoked by `Evaluator::new()` to register builtins/docs.
pub fn set_default_registrar(f: fn(&mut Evaluator)) {
    let _ = DEFAULT_REGISTRAR.set(f);
}

impl Evaluator {
    /// Create a new evaluator and invoke the default registrar if configured.
    pub fn new() -> Self {
        let mut ev = Self {
            builtins: HashMap::new(),
            docs: HashMap::new(),
            env: HashMap::new(),
            tasks: HashMap::new(),
            next_task_id: 1,
            cancel_token: None,
            thread_limiter: None,
            deadline: None,
            trace_enabled: false,
            trace_steps: Vec::new(),
            defs: DefinitionStore::new(),
            current_span: None,
        };
        if let Some(f) = DEFAULT_REGISTRAR.get().copied() {
            f(&mut ev);
        }
        ev
    }

    /// Create a new evaluator with a pre-populated environment map.
    pub fn with_env(env: HashMap<String, Value>) -> Self {
        let mut ev = Self::new();
        ev.env = env;
        ev
    }

    /// Create a new evaluator with environment and a cancellation token.
    pub fn with_env_and_token(env: HashMap<String, Value>, token: Arc<AtomicBool>) -> Self {
        let mut ev = Self::new();
        ev.env = env;
        ev.cancel_token = Some(token);
        ev
    }

    /// Register a builtin function by name with its implementation and attributes.
    pub fn register(&mut self, name: &str, f: NativeFn, attrs: Attributes) {
        self.builtins.insert(name.to_string(), (f, attrs));
    }

    // --- Documentation registry helpers ---
    /// Attach a short summary and parameter list to a builtin.
    pub fn set_doc<S: Into<String>>(&mut self, name: &str, summary: S, params: &[&str]) {
        crate::core::docs::set_doc(self, name, summary, params)
    }
    /// Attach example snippets to a builtin's documentation entry.
    pub fn set_doc_examples(&mut self, name: &str, examples: &[&str]) {
        crate::core::docs::set_doc_examples(self, name, examples)
    }
    /// Get the summary and parameter names for a documented builtin.
    pub fn get_doc(&self, name: &str) -> Option<(String, Vec<String>)> {
        crate::core::docs::get_doc(self, name)
    }
    /// Get the full documentation entry for a builtin, if present.
    pub fn get_doc_full(&self, name: &str) -> Option<DocEntry> {
        crate::core::docs::get_doc_full(self, name)
    }

    pub(crate) fn builtins_snapshot(&self) -> Vec<(String, Attributes)> {
        self.builtins
            .iter()
            .map(|(name, (_f, attrs))| (name.clone(), *attrs))
            .collect()
    }

    // DefinitionStore accessors for core modules
    pub(crate) fn rules_mut(&mut self, kind: DefKind, sym: &str) -> &mut RuleSet {
        self.defs.rules_mut(kind, sym)
    }
    pub(crate) fn rules(&self, kind: DefKind, sym: &str) -> Option<&RuleSet> {
        self.defs.rules(kind, sym)
    }

    /// List keys currently present in the evaluator environment.
    pub fn env_keys(&self) -> Vec<String> { crate::core::env::env_keys(self) }

    /// Set an environment value by key.
    pub fn set_env(&mut self, key: &str, v: lyra_core::value::Value) { crate::core::env::set_env(self, key, v) }

    /// Get an environment value by key.
    pub fn get_env(&self, key: &str) -> Option<lyra_core::value::Value> { crate::core::env::get_env(self, key) }

    /// Remove a key from the environment map, if present.
    pub fn unset_env(&mut self, key: &str) { crate::core::env::unset_env(self, key) }

    /// Set the current source span for error reporting and tracing.
    pub fn set_current_span(&mut self, span: Option<(usize, usize)>) { crate::core::env::set_current_span(self, span) }

    pub(crate) fn make_error(&self, message: &str, tag: &str) -> Value { crate::core::env::make_error(self, message, tag) }

    /// Evaluate a `Value` with rewrite semantics, call pipeline rules, and
    /// configured concurrency/cancellation settings.
    pub fn eval(&mut self, v: Value) -> Value {
        if let Some(tok) = &self.cancel_token {
            if tok.load(Ordering::Relaxed) {
                return cancelled_failure();
            }
        }
        if let Some(dl) = self.deadline {
            if Instant::now() > dl {
                return time_budget_failure();
            }
        }
        match v {
            Value::Expr { head, args } => {
                let head_eval = self.eval(*head);
                // Determine function name
                if let Value::PureFunction { params, body } = head_eval.clone() {
                    let eval_args: Vec<Value> = args.into_iter().map(|a| self.eval(a)).collect();
                    let applied = apply_pure_function(*body, params.as_ref(), &eval_args);
                    return self.eval(applied);
                }
                let fname = match &head_eval {
                    Value::Symbol(s) => s.clone(),
                    _ => {
                        if let Some(val) = crate::core::rewrite::try_subvalues(self, &head_eval, &args) { return val; }
                        return Value::Expr { head: Box::new(head_eval), args };
                    }
                };
                // UpValues rewrite (single step) prior to DownValues and builtin dispatch
                if let Some(val) = crate::core::rewrite::try_upvalues(self, &fname, &args) { return val; }
                // DownValues rewrite (single step) prior to builtin dispatch
                if let Some(val) = crate::core::rewrite::try_downvalues(self, &fname, &args) { return val; }
                // Listable threading and builtin dispatch
                let (fun, attrs) = match self.builtins.get(&fname) {
                    Some(t) => (t.0, t.1),
                    None => return Value::Expr { head: Box::new(Value::Symbol(fname)), args },
                };
                if attrs.contains(Attributes::LISTABLE) {
                    if args.iter().any(|a| matches!(a, Value::List(_))) {
                        if self.trace_enabled {
                            let count = args
                                .iter()
                                .filter_map(|a| {
                                    if let Value::List(v) = a {
                                        Some(v.len() as i64)
                                    } else {
                                        None
                                    }
                                })
                                .max()
                                .unwrap_or(0);
                            let arg_lens: Vec<Value> = args
                                .iter()
                                .map(|a| match a {
                                    Value::List(v) => Value::Integer(v.len() as i64),
                                    _ => Value::Integer(0),
                                })
                                .collect();
                            let data = Value::Assoc(
                                vec![
                                    ("count".to_string(), Value::Integer(count)),
                                    ("argLens".to_string(), Value::List(arg_lens)),
                                ]
                                .into_iter()
                                .collect(),
                            );
                            self.trace_steps.push(Value::Assoc(
                                vec![
                                    ("action".to_string(), Value::String("ListableThread".into())),
                                    ("head".to_string(), head_eval.clone()),
                                    ("data".to_string(), data),
                                ]
                                .into_iter()
                                .collect(),
                            ));
                        }
                        return crate::core::call::listable_thread(self, fun, args);
                    }
                }
                // Evaluate arguments respecting Hold* attributes
                if self.trace_enabled {
                    let held = crate::core::call::held_positions(attrs, args.len());
                    if !held.is_empty() {
                        let data = Value::Assoc(
                            vec![(
                                "held".to_string(),
                                Value::List(held.into_iter().map(Value::Integer).collect()),
                            )]
                            .into_iter()
                            .collect(),
                        );
                        self.trace_steps.push(Value::Assoc(
                            vec![
                                ("action".to_string(), Value::String("Hold".into())),
                                ("head".to_string(), head_eval.clone()),
                                ("data".to_string(), data),
                            ]
                            .into_iter()
                            .collect(),
                        ));
                    }
                }
                let mut eval_args: Vec<Value> = crate::core::call::eval_with_hold(self, args, attrs);
                // Sequence splicing at top-level arguments
                eval_args = crate::core::call::splice_sequences(eval_args);
                // Flat: flatten same-head nested calls in arguments
                if attrs.contains(Attributes::FLAT) {
                    if self.trace_enabled {
                        eval_args = crate::core::call::flat_flatten_traced(self, &head_eval, &fname, eval_args);
                    } else {
                        eval_args = crate::core::call::flat_flatten(&fname, eval_args);
                    }
                }
                // Orderless: canonical sort of args
                if attrs.contains(Attributes::ORDERLESS) {
                    if self.trace_enabled {
                        eval_args = crate::core::call::orderless_sort_traced(self, &head_eval, eval_args);
                    } else {
                        eval_args = crate::core::call::orderless_sort(eval_args);
                    }
                }
                // OneIdentity: f[x] -> x (after canonicalization and holds)
                if attrs.contains(Attributes::ONE_IDENTITY) && eval_args.len() == 1 {
                    return eval_args.into_iter().next().unwrap();
                }
                fun(self, eval_args)
            }
            Value::List(items) => Value::List(items.into_iter().map(|x| self.eval(x)).collect()),
            Value::Assoc(m) => {
                Value::Assoc(m.into_iter().map(|(k, v)| (k, self.eval(v))).collect())
            }
            Value::Symbol(s) => {
                if let Some(val) = crate::core::rewrite::try_ownvalues(self, &s) { return val; }
                self.env.get(&s).cloned().unwrap_or(Value::Symbol(s))
            }
            other => other,
        }
    }
}


/// Convenience helper: evaluate a single value using a fresh `Evaluator`.
pub fn evaluate(v: Value) -> Value {
    Evaluator::new().eval(v)
}

// Public registration helpers for stdlib wrappers
#[cfg(feature = "core")]
/// Register core rewrite/defs/assign primitives.
pub fn register_core(ev: &mut Evaluator) {
    crate::core::assign::register_assign(ev);
    crate::core::rewrite::register_rewrite(ev);
    crate::core::defs::register_defs(ev);
}

/// Register all concurrency primitives (futures, channels, actors, scopes)
/// and convenience parallel helpers.
pub fn register_concurrency(ev: &mut Evaluator) {
    crate::concurrency::futures::register_futures(ev);
    crate::concurrency::channels::register_channels(ev);
    crate::concurrency::actors::register_actors(ev);
    crate::concurrency::scope::register_scopes(ev);
    ev.register("ParallelMap", parallel_map as NativeFn, Attributes::empty());
    ev.register("ParallelTable", parallel_table as NativeFn, Attributes::HOLD_ALL);
    ev.register("MapAsync", map_async as NativeFn, Attributes::empty());
    ev.register("Gather", gather as NativeFn, Attributes::empty());
    ev.register("BusyWait", busy_wait as NativeFn, Attributes::empty());
    ev.register("Fail", fail_fn as NativeFn, Attributes::empty());
    
}

// Filtered registration variant for tree-shaken builds.
// Only registers symbols that satisfy the provided predicate.
/// Conditionally register selected concurrency primitives based on `pred`.
pub fn register_concurrency_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    // Register Futures first if any of them are requested, to avoid borrow issues
    if pred("Future") || pred("Await") || pred("Cancel") {
        crate::concurrency::futures::register_futures(ev);
    }
    if pred("BoundedChannel") || pred("Send") || pred("Receive") || pred("CloseChannel") || pred("TrySend") || pred("TryReceive") {
        crate::concurrency::channels::register_channels(ev);
    }
    let mut reg = |name: &str, f: NativeFn, attrs: Attributes| {
        if pred(name) {
            ev.register(name, f, attrs);
        }
    };
    // Scope/Actors registrars invoked above when requested
    reg("ParallelMap", parallel_map as NativeFn, Attributes::empty());
    reg("ParallelTable", parallel_table as NativeFn, Attributes::HOLD_ALL);
    reg("MapAsync", map_async as NativeFn, Attributes::empty());
    reg("Gather", gather as NativeFn, Attributes::empty());
    reg("BusyWait", busy_wait as NativeFn, Attributes::empty());
    reg("Fail", fail_fn as NativeFn, Attributes::empty());
    
}

/// Register `Explain` helpers for tracing and schema insights.
pub fn register_explain(ev: &mut Evaluator) { crate::core::schema_explain::register_explain(ev); }

/// Register `Schema` helpers for shape/type inspection.
pub fn register_schema(ev: &mut Evaluator) { crate::core::schema_explain::register_schema(ev); }

// Introspection: list current builtin functions with their attributes
/// Register introspection helpers for listing builtins and docs.
pub fn register_introspection(ev: &mut Evaluator) { crate::core::introspection::register_introspection(ev); }

// listable_thread moved to core::call

// splice_sequences moved to core::call

//
//

fn cancelled_failure() -> Value {
    // no access to self here; embed minimal shape
    Value::Assoc(
        vec![
            ("error".to_string(), Value::Boolean(true)),
            ("message".to_string(), Value::String("Computation cancelled".into())),
            ("tag".to_string(), Value::String("Cancel::abort".into())),
        ]
        .into_iter()
        .collect(),
    )
}

fn time_budget_failure() -> Value {
    Value::Assoc(
        vec![
            ("error".to_string(), Value::Boolean(true)),
            ("message".to_string(), Value::String("Time budget exceeded".into())),
            ("tag".to_string(), Value::String("TimeBudget::exceeded".into())),
        ]
        .into_iter()
        .collect(),
    )
}

fn parallel_map(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("ParallelMap".into())), args };
    }
    let f = args[0].clone();
    let target = args[1].clone();
    let mut opt_limiter: Option<Arc<ThreadLimiter>> = None;
    let mut opt_deadline: Option<Instant> = None;
    if args.len() >= 3 {
        if let Value::Assoc(m) = ev.eval(args[2].clone()) {
            if let Some(Value::Integer(n)) = m.get("MaxThreads").or_else(|| m.get("maxThreads")) {
                if *n > 0 {
                    opt_limiter = Some(Arc::new(ThreadLimiter::new(*n as usize)));
                }
            }
            if let Some(Value::Integer(ms)) = m.get("TimeBudgetMs").or_else(|| m.get("timeBudgetMs")) {
                if *ms > 0 {
                    opt_deadline = Some(Instant::now() + Duration::from_millis(*ms as u64));
                }
            }
        }
    }
    // Capture env so sub-evaluations inherit current lexical bindings
    let env_snapshot = ev.env.clone();
    match ev.eval(target) {
        Value::List(items) => {
            let token = ev.cancel_token.clone();
            let limiter = opt_limiter.or_else(|| ev.thread_limiter.clone());
            let deadline = opt_deadline.or(ev.deadline);
            if ev.trace_enabled {
                let data = Value::Assoc(
                    vec![
                        ("items".to_string(), Value::Integer(items.len() as i64)),
                        (
                            "maxThreads".to_string(),
                            Value::Integer(limiter.as_ref().map(|l| l.max_permits() as i64).unwrap_or(-1)),
                        ),
                        ("hasDeadline".to_string(), Value::Boolean(deadline.is_some())),
                    ]
                    .into_iter()
                    .collect(),
                );
                ev.trace_steps.push(Value::Assoc(
                    vec![
                        ("action".to_string(), Value::String("ParallelDispatch".into())),
                        ("head".to_string(), Value::Symbol("ParallelMap".into())),
                        ("data".to_string(), data),
                    ]
                    .into_iter()
                    .collect(),
                ));
            }
            let mut rxs: Vec<Receiver<Value>> = Vec::with_capacity(items.len());
            for it in items.into_iter() {
                let f_cl = f.clone();
                let token_cl = token.clone();
                let limiter_cl = limiter.clone();
                let deadline_cl = deadline;
                let env_cl = env_snapshot.clone();
                rxs.push(spawn_task(move || {
                    if let Some(l) = limiter_cl.as_ref() {
                        l.acquire();
                    }
                    let mut ev2 = if let Some(tok) = token_cl {
                        Evaluator::with_env_and_token(env_cl.clone(), tok)
                    } else {
                        Evaluator::with_env(env_cl.clone())
                    };
                    ev2.thread_limiter = limiter_cl;
                    ev2.deadline = deadline_cl;
                    let call = Value::Expr { head: Box::new(f_cl), args: vec![it] };
                    let out = ev2.eval(call);
                    if let Some(l) = ev2.thread_limiter.as_ref() {
                        l.release();
                    }
                    out
                }));
            }
            let mut out = Vec::with_capacity(rxs.len());
            for rx in rxs {
                out.push(rx.recv().unwrap_or(Value::Symbol("Null".into())));
            }
            Value::List(out)
        }
        other => Value::Expr {
            head: Box::new(Value::Symbol("ParallelMap".into())),
            args: vec![f, other],
        },
    }
}

// ParallelTable[expr, {i, imin, imax}] and ParallelTable[expr, {i, imin, imax, step}]
fn parallel_table(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("ParallelTable".into())), args };
    }
    let expr = args[0].clone();
    let spec = args[1].clone();
    let mut opt_limiter: Option<Arc<ThreadLimiter>> = None;
    let mut opt_deadline: Option<Instant> = None;
    if args.len() >= 3 {
        if let Value::Assoc(m) = ev.eval(args[2].clone()) {
            if let Some(Value::Integer(n)) = m.get("MaxThreads").or_else(|| m.get("maxThreads")) {
                if *n > 0 {
                    opt_limiter = Some(Arc::new(ThreadLimiter::new(*n as usize)));
                }
            }
            if let Some(Value::Integer(ms)) = m.get("TimeBudgetMs").or_else(|| m.get("timeBudgetMs")) {
                if *ms > 0 {
                    opt_deadline = Some(Instant::now() + Duration::from_millis(*ms as u64));
                }
            }
        }
    }
    // Expect spec to be a List: {Symbol(i), imin, imax[, step]}
    let (var_name, start, end, step) = match spec {
        Value::List(items) => {
            if items.len() < 3 {
                return Value::Expr {
                    head: Box::new(Value::Symbol("ParallelTable".into())),
                    args: vec![expr, Value::List(items)],
                };
            }
            let vname = match &items[0] {
                Value::Symbol(s) => s.clone(),
                other => {
                    let evd = ev.eval(other.clone());
                    if let Value::Symbol(s2) = evd {
                        s2
                    } else {
                        return Value::Expr {
                            head: Box::new(Value::Symbol("ParallelTable".into())),
                            args,
                        };
                    }
                }
            };
            let s = match ev.eval(items[1].clone()) {
                Value::Integer(i) => i,
                _ => {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("ParallelTable".into())),
                        args,
                    }
                }
            };
            let e = match ev.eval(items[2].clone()) {
                Value::Integer(i) => i,
                _ => {
                    return Value::Expr {
                        head: Box::new(Value::Symbol("ParallelTable".into())),
                        args,
                    }
                }
            };
            let st = if items.len() >= 4 {
                match ev.eval(items[3].clone()) {
                    Value::Integer(i) => {
                        if i == 0 {
                            1
                        } else {
                            i
                        }
                    }
                    _ => 1,
                }
            } else {
                1
            };
            (vname, s, e, st)
        }
        other => {
            return Value::Expr {
                head: Box::new(Value::Symbol("ParallelTable".into())),
                args: vec![expr, other],
            }
        }
    };

    // Build range according to step and bounds (inclusive)
    let mut values: Vec<i64> = Vec::new();
    if step > 0 {
        let mut i = start;
        while i <= end {
            values.push(i);
            i += step;
        }
    } else {
        let mut i = start;
        while i >= end {
            values.push(i);
            i += step;
        }
    }

    // Spawn per iteration (naive); each thread gets its own evaluator with env snapshot and bound variable
    let env_snapshot = ev.env.clone();
    let limiter = opt_limiter.or_else(|| ev.thread_limiter.clone());
    let token = ev.cancel_token.clone();
    let deadline = opt_deadline.or(ev.deadline);
    if ev.trace_enabled {
        let data = Value::Assoc(
            vec![
                ("items".to_string(), Value::Integer(values.len() as i64)),
                (
                    "maxThreads".to_string(),
                    Value::Integer(limiter.as_ref().map(|l| l.max_permits() as i64).unwrap_or(-1)),
                ),
                ("hasDeadline".to_string(), Value::Boolean(deadline.is_some())),
            ]
            .into_iter()
            .collect(),
        );
        ev.trace_steps.push(Value::Assoc(
            vec![
                ("action".to_string(), Value::String("ParallelDispatch".into())),
                ("head".to_string(), Value::Symbol("ParallelTable".into())),
                ("data".to_string(), data),
            ]
            .into_iter()
            .collect(),
        ));
    }
    let mut rxs: Vec<Receiver<Value>> = Vec::with_capacity(values.len());
    for iv in values.into_iter() {
        let expr_cl = expr.clone();
        let var = var_name.clone();
        let env_cl = env_snapshot.clone();
        let limiter_cl = limiter.clone();
        let token_cl = token.clone();
        let deadline_cl = deadline;
        rxs.push(spawn_task(move || {
            if let Some(l) = limiter_cl.as_ref() {
                l.acquire();
            }
            let mut ev2 = if let Some(tok) = token_cl {
                Evaluator::with_env_and_token(env_cl, tok)
            } else {
                Evaluator::with_env(env_cl)
            };
            ev2.thread_limiter = limiter_cl;
            ev2.deadline = deadline_cl;
            ev2.env.insert(var, Value::Integer(iv));
            let out = ev2.eval(expr_cl);
            if let Some(l) = ev2.thread_limiter.as_ref() {
                l.release();
            }
            out
        }));
    }
    let mut out: Vec<Value> = Vec::with_capacity(rxs.len());
    for rx in rxs {
        out.push(rx.recv().unwrap_or(Value::Symbol("Null".into())));
    }
    Value::List(out)
}

fn map_async(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("MapAsync".into())), args };
    }
    let f = args[0].clone();
    let v = ev.eval(args[1].clone());
    let opts = if args.len() >= 3 { Some(ev.eval(args[2].clone())) } else { None };
    // If options are provided, apply MaxThreads/TimeBudget to the current evaluator for the duration of this call
    let (old_lim, old_dead) = (ev.thread_limiter.clone(), ev.deadline);
    if let Some(Value::Assoc(m)) = &opts {
        if let Some(Value::Integer(n)) = m.get("MaxThreads").or_else(|| m.get("maxThreads")) {
            if *n > 0 {
                ev.thread_limiter = Some(Arc::new(ThreadLimiter::new(*n as usize)));
            }
        }
        if let Some(Value::Integer(ms)) = m.get("TimeBudgetMs").or_else(|| m.get("timeBudgetMs")) {
            if *ms > 0 {
                ev.deadline = Some(Instant::now() + Duration::from_millis(*ms as u64));
            }
        }
    }
    let out_list = Value::List(map_async_rec(ev, f, v, None));
    // restore
    ev.thread_limiter = old_lim;
    ev.deadline = old_dead;
    out_list
}

fn gather(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Gather".into())), args };
    }
    let v = ev.eval(args[0].clone());
    Value::List(gather_rec(ev, v))
}

fn busy_wait(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let cycles = match args.as_slice() {
        [Value::Integer(n)] => *n as usize,
        [n] => match ev.eval(n.clone()) {
            Value::Integer(i) => i as usize,
            _ => 10,
        },
        _ => 10,
    };
    for _ in 0..cycles {
        if let Some(tok) = &ev.cancel_token {
            if tok.load(Ordering::Relaxed) {
                return cancelled_failure();
            }
        }
        if let Some(dl) = ev.deadline {
            if Instant::now() > dl {
                return time_budget_failure();
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    Value::Symbol("Done".into())
}

fn fail_fn(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    let tag = match args.get(0) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Symbol(s)) => s.clone(),
        _ => "Fail".into(),
    };
    Value::Assoc(
        vec![
            ("message".to_string(), Value::String("Failure".into())),
            ("tag".to_string(), Value::String(tag)),
        ]
        .into_iter()
        .collect(),
    )
}

//
//

fn map_async_rec(ev: &mut Evaluator, f: Value, v: Value, opts: Option<Value>) -> Vec<Value> {
    match v {
        Value::List(items) => items
            .into_iter()
            .map(|it| {
                let mapped = map_async_rec(ev, f.clone(), it, opts.clone());
                if mapped.len() == 1 {
                    mapped.into_iter().next().unwrap()
                } else {
                    Value::List(mapped)
                }
            })
            .collect(),
        other => {
            let call = Value::Expr { head: Box::new(f), args: vec![other] };
            vec![crate::concurrency::futures::future_fn(ev, vec![call])]
        }
    }
}

fn gather_rec(ev: &mut Evaluator, v: Value) -> Vec<Value> {
    match v {
        Value::List(items) => items
            .into_iter()
            .map(|it| {
                let g = gather_rec(ev, it);
                if g.len() == 1 {
                    g.into_iter().next().unwrap()
                } else {
                    Value::List(g)
                }
            })
            .collect(),
        other => vec![crate::concurrency::futures::await_fn(ev, vec![other])],
    }
}

fn apply_pure_function(body: Value, params: Option<&Vec<String>>, args: &Vec<Value>) -> Value {
    if let Some(ps) = params {
        // Replace symbols matching params with args[i]
        fn subst(v: &Value, names: &Vec<String>, args: &Vec<Value>) -> Value {
            match v {
                Value::Symbol(s) => {
                    if let Some((i, _)) = names.iter().enumerate().find(|(_, n)| *n == s) {
                        args.get(i).cloned().unwrap_or(Value::Symbol("Null".into()))
                    } else {
                        v.clone()
                    }
                }
                Value::List(items) => Value::List(items.iter().map(|x| subst(x, names, args)).collect()),
                Value::Assoc(m) => Value::Assoc(m.iter().map(|(k, v)| (k.clone(), subst(v, names, args))).collect()),
                Value::Expr { head, args: a } => Value::Expr { head: Box::new(subst(head, names, args)), args: a.iter().map(|x| subst(x, names, args)).collect() },
                other => other.clone(),
            }
        }
        return subst(&body, ps, args);
    }
    // Slot-based substitution: # -> args[0], #n -> args[n-1]
    fn subst_slot(v: &Value, args: &Vec<Value>) -> Value {
        match v {
            Value::Slot(None) => args.get(0).cloned().unwrap_or(Value::Symbol("Null".into())),
            Value::Slot(Some(n)) => args.get(n.saturating_sub(1)).cloned().unwrap_or(Value::Symbol("Null".into())),
            Value::List(items) => Value::List(items.iter().map(|x| subst_slot(x, args)).collect()),
            Value::Assoc(m) => Value::Assoc(m.iter().map(|(k, v)| (k.clone(), subst_slot(v, args))).collect()),
            Value::Expr { head, args: a } => Value::Expr { head: Box::new(subst_slot(head, args)), args: a.iter().map(|x| subst_slot(x, args)).collect() },
            other => other.clone(),
        }
    }
    subst_slot(&body, args)
}

//
pub fn value_order_key(v: &Value) -> String { crate::core::rewrite::value_order_key(v) }

//
