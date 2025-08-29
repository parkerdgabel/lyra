use crate::attrs::Attributes;

use lyra_core::value::Value;
use lyra_rewrite::defs::{DefKind, DefinitionStore};
use lyra_rewrite::rule::{Rule, RuleSet};
use std::collections::HashMap;
use std::sync::mpsc::Receiver;
use std::sync::OnceLock;
use std::sync::{
    atomic::{AtomicBool, AtomicI64, Ordering},
    Arc, Mutex,
};
// thread usage moved to concurrency::pool
use std::time::{Duration, Instant};

pub(crate) type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

#[derive(Clone)]
struct ScopeCtx {
    cancel: Arc<AtomicBool>,
    limiter: Option<Arc<ThreadLimiter>>,
    deadline: Option<Instant>,
}

static SCOPE_REG: OnceLock<Mutex<HashMap<i64, ScopeCtx>>> = OnceLock::new();
static NEXT_SCOPE_ID: OnceLock<AtomicI64> = OnceLock::new();

fn scope_reg() -> &'static Mutex<HashMap<i64, ScopeCtx>> {
    SCOPE_REG.get_or_init(|| Mutex::new(HashMap::new()))
}
fn next_scope_id() -> i64 {
    let a = NEXT_SCOPE_ID.get_or_init(|| AtomicI64::new(1));
    a.fetch_add(1, Ordering::Relaxed)
}

use crate::concurrency::pool::{spawn_task, ThreadLimiter};

use crate::trace::{trace_drain_steps, trace_push_step};
// channels moved to concurrency::channels

// moved to concurrency::futures::TaskInfo

// ThreadLimiter moved to concurrency::pool

#[derive(Clone, Debug)]
pub struct DocEntry {
    summary: String,
    params: Vec<String>,
    examples: Vec<String>,
}

pub struct Evaluator {
    builtins: HashMap<String, (NativeFn, Attributes)>,
    // Documentation registry: function name -> summary, params, examples
    docs: HashMap<String, DocEntry>,
    pub(crate) env: HashMap<String, Value>,
    pub(crate) tasks: HashMap<i64, crate::concurrency::futures::TaskInfo>, // minimal future registry
    pub(crate) next_task_id: i64,
    pub(crate) cancel_token: Option<Arc<AtomicBool>>, // cooperative cancellation
    pub(crate) thread_limiter: Option<Arc<ThreadLimiter>>, // scope-wide thread budget
    pub(crate) deadline: Option<Instant>,             // scope-wide deadline
    pub(crate) trace_enabled: bool,
    pub(crate) trace_steps: Vec<Value>,
    defs: DefinitionStore,
    current_span: Option<(usize, usize)>,
}

static DEFAULT_REGISTRAR: OnceLock<fn(&mut Evaluator)> = OnceLock::new();

pub fn set_default_registrar(f: fn(&mut Evaluator)) {
    let _ = DEFAULT_REGISTRAR.set(f);
}

impl Evaluator {
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

    pub fn with_env(env: HashMap<String, Value>) -> Self {
        let mut ev = Self::new();
        ev.env = env;
        ev
    }

    pub fn with_env_and_token(env: HashMap<String, Value>, token: Arc<AtomicBool>) -> Self {
        let mut ev = Self::new();
        ev.env = env;
        ev.cancel_token = Some(token);
        ev
    }

    pub fn register(&mut self, name: &str, f: NativeFn, attrs: Attributes) {
        self.builtins.insert(name.to_string(), (f, attrs));
    }

    // --- Documentation registry helpers ---
    pub fn set_doc<S: Into<String>>(&mut self, name: &str, summary: S, params: &[&str]) {
        let entry = self.docs.entry(name.to_string()).or_insert(DocEntry {
            summary: String::new(),
            params: Vec::new(),
            examples: Vec::new(),
        });
        entry.summary = summary.into();
        entry.params = params.iter().map(|s| (*s).to_string()).collect();
    }

    pub fn set_doc_examples(&mut self, name: &str, examples: &[&str]) {
        let entry = self.docs.entry(name.to_string()).or_insert(DocEntry {
            summary: String::new(),
            params: Vec::new(),
            examples: Vec::new(),
        });
        entry.examples = examples.iter().map(|s| (*s).to_string()).collect();
    }

    pub fn get_doc(&self, name: &str) -> Option<(String, Vec<String>)> {
        self.docs
            .get(name)
            .map(|d| (d.summary.clone(), d.params.clone()))
    }

    pub fn get_doc_full(&self, name: &str) -> Option<DocEntry> {
        self.docs.get(name).cloned()
    }

    pub fn env_keys(&self) -> Vec<String> {
        self.env.keys().cloned().collect()
    }

    pub fn set_env(&mut self, key: &str, v: lyra_core::value::Value) {
        self.env.insert(key.to_string(), v);
    }

    pub fn get_env(&self, key: &str) -> Option<lyra_core::value::Value> {
        self.env.get(key).cloned()
    }

    pub fn unset_env(&mut self, key: &str) {
        self.env.remove(key);
    }

    pub fn set_current_span(&mut self, span: Option<(usize, usize)>) {
        self.current_span = span;
    }

    pub(crate) fn make_error(&self, message: &str, tag: &str) -> Value {
        let mut m = std::collections::HashMap::new();
        m.insert("error".to_string(), Value::Boolean(true));
        m.insert("message".to_string(), Value::String(message.into()));
        m.insert("tag".to_string(), Value::String(tag.into()));
        if let Some((s, e)) = self.current_span {
            let mut span = std::collections::HashMap::new();
            span.insert("start".to_string(), Value::Integer(s as i64));
            span.insert("end".to_string(), Value::Integer(e as i64));
            m.insert("span".to_string(), Value::Assoc(span));
        }
        Value::Assoc(m)
    }

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
                        // SubValues rewrite for compound head: (g[...])[args]
                        if let Value::Expr { head: inner_head, .. } = &head_eval {
                            if let Value::Symbol(sub_sym) = &**inner_head {
                                if let Some(rs) = self.defs.rules(DefKind::Sub, sub_sym) {
                                    if !rs.is_empty() {
                                        let rules: Vec<(Value, Value)> = rs
                                            .iter()
                                            .map(|r| (r.lhs.clone(), r.rhs.clone()))
                                            .collect();
                                        let expr0 = Value::Expr {
                                            head: Box::new(head_eval.clone()),
                                            args: args.clone(),
                                        };
                                        // Build closures for conditions/pattern tests
                                        let env_snapshot = self.env.clone();
                                        let pred = |pred: &Value, arg: &Value| {
                                            let mut ev2 = Evaluator::with_env(env_snapshot.clone());
                                            let call = Value::Expr {
                                                head: Box::new(pred.clone()),
                                                args: vec![arg.clone()],
                                            };
                                            let res =
                                                matches!(ev2.eval(call), Value::Boolean(true));
                                            let data = Value::Assoc(
                                                vec![
                                                    ("pred".to_string(), pred.clone()),
                                                    ("arg".to_string(), arg.clone()),
                                                    ("result".to_string(), Value::Boolean(res)),
                                                ]
                                                .into_iter()
                                                .collect(),
                                            );
                                            trace_push_step(Value::Assoc(
                                                vec![
                                                    (
                                                        "action".to_string(),
                                                        Value::String("ConditionEvaluated".into()),
                                                    ),
                                                    (
                                                        "head".to_string(),
                                                        Value::Symbol("PatternTest".into()),
                                                    ),
                                                    ("data".to_string(), data),
                                                ]
                                                .into_iter()
                                                .collect(),
                                            ));
                                            res
                                        };
                                        let cond = |cond: &Value, binds: &lyra_rewrite::matcher::Bindings| {
                                        let mut ev2 = Evaluator::with_env(env_snapshot.clone());
                                        let cond_sub = lyra_rewrite::matcher::substitute_named(cond, binds);
                                        let res = matches!(ev2.eval(cond_sub.clone()), Value::Boolean(true));
                                        let data = Value::Assoc(vec![
                                            ("expr".to_string(), cond_sub),
                                            ("bindsCount".to_string(), Value::Integer(binds.len() as i64)),
                                            ("result".to_string(), Value::Boolean(res)),
                                        ].into_iter().collect());
                                        trace_push_step(Value::Assoc(vec![
                                            ("action".to_string(), Value::String("ConditionEvaluated".into())),
                                            ("head".to_string(), Value::Symbol("Condition".into())),
                                            ("data".to_string(), data),
                                        ].into_iter().collect()));
                                        res
                                    };
                                        let ctx = lyra_rewrite::matcher::MatcherCtx {
                                            eval_pred: Some(&pred),
                                            eval_cond: Some(&cond),
                                        };
                                        let out =
                                            lyra_rewrite::engine::rewrite_once_indexed_with_ctx(
                                                &ctx,
                                                expr0.clone(),
                                                &rules,
                                            );
                                        if out != expr0 {
                                            if self.trace_enabled {
                                                for (lhs, rhs) in &rules {
                                                    if lyra_rewrite::matcher::match_rule_with(
                                                        &ctx, lhs, &expr0,
                                                    )
                                                    .is_some()
                                                    {
                                                        let data = Value::Assoc(
                                                            vec![
                                                                ("lhs".to_string(), lhs.clone()),
                                                                ("rhs".to_string(), rhs.clone()),
                                                            ]
                                                            .into_iter()
                                                            .collect(),
                                                        );
                                                        self.trace_steps.push(Value::Assoc(
                                                            vec![
                                                                (
                                                                    "action".to_string(),
                                                                    Value::String(
                                                                        "RuleMatch".into(),
                                                                    ),
                                                                ),
                                                                (
                                                                    "head".to_string(),
                                                                    Value::Symbol(
                                                                        "SubValues".into(),
                                                                    ),
                                                                ),
                                                                ("data".to_string(), data),
                                                            ]
                                                            .into_iter()
                                                            .collect(),
                                                        ));
                                                        break;
                                                    }
                                                }
                                                self.trace_steps.extend(trace_drain_steps());
                                            }
                                            return self.eval(out);
                                        }
                                    }
                                }
                            }
                        }
                        return Value::Expr { head: Box::new(head_eval), args };
                    }
                };
                // UpValues rewrite (single step) prior to DownValues and builtin dispatch
                {
                    let expr0 = Value::Expr {
                        head: Box::new(Value::Symbol(fname.clone())),
                        args: args.clone(),
                    };
                    let mut up_rules: Vec<(Value, Value)> = Vec::new();
                    // Collect UpValues for any symbol at top-level arguments (Symbol or Expr head Symbol)
                    for a in &args {
                        let sym_opt = match a {
                            Value::Symbol(s) => Some(s.clone()),
                            Value::Expr { head, .. } => {
                                if let Value::Symbol(s) = &**head {
                                    Some(s.clone())
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        };
                        if let Some(sym) = sym_opt {
                            if let Some(rs) = self.defs.rules(DefKind::Up, &sym) {
                                for r in rs.iter() {
                                    up_rules.push((r.lhs.clone(), r.rhs.clone()));
                                }
                            }
                        }
                    }
                    if !up_rules.is_empty() {
                        // Deterministic precedence: linear scan in collected order
                        // Build closures for Condition/PatternTest
                        let env_snapshot = self.env.clone();
                        let pred = |pred: &Value, arg: &Value| {
                            let mut ev2 = Evaluator::with_env(env_snapshot.clone());
                            let call = Value::Expr {
                                head: Box::new(pred.clone()),
                                args: vec![arg.clone()],
                            };
                            let res = matches!(ev2.eval(call), Value::Boolean(true));
                            let data = Value::Assoc(
                                vec![
                                    ("pred".to_string(), pred.clone()),
                                    ("arg".to_string(), arg.clone()),
                                    ("result".to_string(), Value::Boolean(res)),
                                ]
                                .into_iter()
                                .collect(),
                            );
                            trace_push_step(Value::Assoc(
                                vec![
                                    (
                                        "action".to_string(),
                                        Value::String("ConditionEvaluated".into()),
                                    ),
                                    ("head".to_string(), Value::Symbol("PatternTest".into())),
                                    ("data".to_string(), data),
                                ]
                                .into_iter()
                                .collect(),
                            ));
                            res
                        };
                        let cond = |cond: &Value, binds: &lyra_rewrite::matcher::Bindings| {
                            let mut ev2 = Evaluator::with_env(env_snapshot.clone());
                            let cond_sub = lyra_rewrite::matcher::substitute_named(cond, binds);
                            let res = matches!(ev2.eval(cond_sub.clone()), Value::Boolean(true));
                            let data = Value::Assoc(
                                vec![
                                    ("expr".to_string(), cond_sub),
                                    ("bindsCount".to_string(), Value::Integer(binds.len() as i64)),
                                    ("result".to_string(), Value::Boolean(res)),
                                ]
                                .into_iter()
                                .collect(),
                            );
                            trace_push_step(Value::Assoc(
                                vec![
                                    (
                                        "action".to_string(),
                                        Value::String("ConditionEvaluated".into()),
                                    ),
                                    ("head".to_string(), Value::Symbol("Condition".into())),
                                    ("data".to_string(), data),
                                ]
                                .into_iter()
                                .collect(),
                            ));
                            res
                        };
                        let ctx = lyra_rewrite::matcher::MatcherCtx {
                            eval_pred: Some(&pred),
                            eval_cond: Some(&cond),
                        };
                        for (lhs, rhs) in &up_rules {
                            if let Some(b) =
                                lyra_rewrite::matcher::match_rule_with(&ctx, lhs, &expr0)
                            {
                                if self.trace_enabled {
                                    let data = Value::Assoc(
                                        vec![
                                            ("lhs".to_string(), lhs.clone()),
                                            ("rhs".to_string(), rhs.clone()),
                                        ]
                                        .into_iter()
                                        .collect(),
                                    );
                                    self.trace_steps.push(Value::Assoc(
                                        vec![
                                            (
                                                "action".to_string(),
                                                Value::String("RuleMatch".into()),
                                            ),
                                            ("head".to_string(), Value::Symbol("UpValues".into())),
                                            ("data".to_string(), data),
                                        ]
                                        .into_iter()
                                        .collect(),
                                    ));
                                }
                                let out = lyra_rewrite::matcher::substitute_named(rhs, &b);
                                if self.trace_enabled {
                                    self.trace_steps.extend(trace_drain_steps());
                                }
                                return self.eval(out);
                            }
                        }
                    }
                }
                // DownValues rewrite (single step) prior to builtin dispatch
                if let Some(rs) = self.defs.rules(DefKind::Down, &fname) {
                    if !rs.is_empty() {
                        let rules: Vec<(Value, Value)> =
                            rs.iter().map(|r| (r.lhs.clone(), r.rhs.clone())).collect();
                        let expr0 = Value::Expr {
                            head: Box::new(Value::Symbol(fname.clone())),
                            args: args.clone(),
                        };
                        // Build closures to evaluate conditions/pattern tests and record Explain
                        let env_snapshot = self.env.clone();
                        let pred = |pred: &Value, arg: &Value| {
                            let mut ev2 = Evaluator::with_env(env_snapshot.clone());
                            let call = Value::Expr {
                                head: Box::new(pred.clone()),
                                args: vec![arg.clone()],
                            };
                            let res = matches!(ev2.eval(call), Value::Boolean(true));
                            let data = Value::Assoc(
                                vec![
                                    ("pred".to_string(), pred.clone()),
                                    ("arg".to_string(), arg.clone()),
                                    ("result".to_string(), Value::Boolean(res)),
                                ]
                                .into_iter()
                                .collect(),
                            );
                            trace_push_step(Value::Assoc(
                                vec![
                                    (
                                        "action".to_string(),
                                        Value::String("ConditionEvaluated".into()),
                                    ),
                                    ("head".to_string(), Value::Symbol("PatternTest".into())),
                                    ("data".to_string(), data),
                                ]
                                .into_iter()
                                .collect(),
                            ));
                            res
                        };
                        let cond = |cond: &Value, binds: &lyra_rewrite::matcher::Bindings| {
                            let mut ev2 = Evaluator::with_env(env_snapshot.clone());
                            let cond_sub = lyra_rewrite::matcher::substitute_named(cond, binds);
                            let res = matches!(ev2.eval(cond_sub.clone()), Value::Boolean(true));
                            let data = Value::Assoc(
                                vec![
                                    ("expr".to_string(), cond_sub),
                                    ("bindsCount".to_string(), Value::Integer(binds.len() as i64)),
                                    ("result".to_string(), Value::Boolean(res)),
                                ]
                                .into_iter()
                                .collect(),
                            );
                            trace_push_step(Value::Assoc(
                                vec![
                                    (
                                        "action".to_string(),
                                        Value::String("ConditionEvaluated".into()),
                                    ),
                                    ("head".to_string(), Value::Symbol("Condition".into())),
                                    ("data".to_string(), data),
                                ]
                                .into_iter()
                                .collect(),
                            ));
                            res
                        };
                        let ctx = lyra_rewrite::matcher::MatcherCtx {
                            eval_pred: Some(&pred),
                            eval_cond: Some(&cond),
                        };
                        let out = lyra_rewrite::engine::rewrite_once_indexed_with_ctx(
                            &ctx,
                            expr0.clone(),
                            &rules,
                        );
                        if out != expr0 {
                            if self.trace_enabled {
                                // Note: For simplicity, we re-scan rules to find the first matching lhs for trace
                                for (lhs, rhs) in &rules {
                                    if lyra_rewrite::matcher::match_rule_with(&ctx, lhs, &expr0)
                                        .is_some()
                                    {
                                        let data = Value::Assoc(
                                            vec![
                                                ("lhs".to_string(), lhs.clone()),
                                                ("rhs".to_string(), rhs.clone()),
                                            ]
                                            .into_iter()
                                            .collect(),
                                        );
                                        self.trace_steps.push(Value::Assoc(
                                            vec![
                                                (
                                                    "action".to_string(),
                                                    Value::String("RuleMatch".into()),
                                                ),
                                                ("head".to_string(), Value::Symbol(fname.clone())),
                                                ("data".to_string(), data),
                                            ]
                                            .into_iter()
                                            .collect(),
                                        ));
                                        break;
                                    }
                                }
                                self.trace_steps.extend(trace_drain_steps());
                            }
                            return self.eval(out);
                        }
                    }
                }
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
                        return crate::core::rewrite::listable_thread(self, fun, args);
                    }
                }
                // Evaluate arguments respecting Hold* attributes
                if self.trace_enabled {
                    let held: Vec<i64> = if attrs.contains(Attributes::HOLD_ALL) {
                        (1..=args.len()).map(|i| i as i64).collect()
                    } else if attrs.contains(Attributes::HOLD_FIRST) {
                        if args.is_empty() {
                            vec![]
                        } else {
                            vec![1]
                        }
                    } else if attrs.contains(Attributes::HOLD_REST) {
                        if args.len() <= 1 {
                            vec![]
                        } else {
                            (2..=args.len()).map(|i| i as i64).collect()
                        }
                    } else {
                        vec![]
                    };
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
                let mut eval_args: Vec<Value> = if attrs.contains(Attributes::HOLD_ALL) {
                    args
                } else if attrs.contains(Attributes::HOLD_FIRST) {
                    let mut out = Vec::with_capacity(args.len());
                    let mut it = args.into_iter();
                    if let Some(first) = it.next() {
                        out.push(first);
                    }
                    for a in it {
                        out.push(self.eval(a));
                    }
                    out
                } else if attrs.contains(Attributes::HOLD_REST) {
                    let mut out = Vec::with_capacity(args.len());
                    let mut it = args.into_iter();
                    if let Some(first) = it.next() {
                        out.push(self.eval(first));
                    }
                    for a in it {
                        out.push(a);
                    }
                    out
                } else {
                    args.into_iter().map(|a| self.eval(a)).collect()
                };
                // Sequence splicing at top-level arguments
                eval_args = crate::core::rewrite::splice_sequences(eval_args);
                // Flat: flatten same-head nested calls in arguments
                if attrs.contains(Attributes::FLAT) {
                    let mut flat: Vec<Value> = Vec::with_capacity(eval_args.len());
                    for a in eval_args.into_iter() {
                        if let Value::Expr { head: h2, args: a2 } = &a {
                            if matches!(&**h2, Value::Symbol(s) if s == &fname) {
                                if self.trace_enabled {
                                    let data = Value::Assoc(
                                        vec![(
                                            "added".to_string(),
                                            Value::Integer(a2.len() as i64),
                                        )]
                                        .into_iter()
                                        .collect(),
                                    );
                                    self.trace_steps.push(Value::Assoc(
                                        vec![
                                            (
                                                "action".to_string(),
                                                Value::String("FlatFlatten".into()),
                                            ),
                                            ("head".to_string(), head_eval.clone()),
                                            ("data".to_string(), data),
                                        ]
                                        .into_iter()
                                        .collect(),
                                    ));
                                }
                                flat.extend(a2.clone());
                                continue;
                            }
                        }
                        flat.push(a);
                    }
                    eval_args = flat;
                }
                // Orderless: canonical sort of args
                if attrs.contains(Attributes::ORDERLESS) {
                    eval_args.sort_by(|x, y| crate::core::rewrite::value_order(x).cmp(&crate::core::rewrite::value_order(y)));
                    if self.trace_enabled {
                        let data = Value::Assoc(
                            vec![("finalOrder".to_string(), Value::List(eval_args.clone()))]
                                .into_iter()
                                .collect(),
                        );
                        self.trace_steps.push(Value::Assoc(
                            vec![
                                ("action".to_string(), Value::String("OrderlessSort".into())),
                                ("head".to_string(), head_eval.clone()),
                                ("data".to_string(), data),
                            ]
                            .into_iter()
                            .collect(),
                        ));
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
                // OwnValues rewrite for symbol
                if let Some(rs) = self.defs.rules(DefKind::Own, &s) {
                    if !rs.is_empty() {
                        let rules: Vec<(Value, Value)> =
                            rs.iter().map(|r| (r.lhs.clone(), r.rhs.clone())).collect();
                        let expr0 = Value::Symbol(s.clone());
                        // Linear top-level match for simplicity
                        for (lhs, rhs) in &rules {
                            if lyra_rewrite::matcher::match_rule(lhs, &expr0).is_some() {
                                if self.trace_enabled {
                                    let data = Value::Assoc(
                                        vec![
                                            ("lhs".to_string(), lhs.clone()),
                                            ("rhs".to_string(), rhs.clone()),
                                        ]
                                        .into_iter()
                                        .collect(),
                                    );
                                    self.trace_steps.push(Value::Assoc(
                                        vec![
                                            (
                                                "action".to_string(),
                                                Value::String("RuleMatch".into()),
                                            ),
                                            ("head".to_string(), Value::Symbol("OwnValues".into())),
                                            ("data".to_string(), data),
                                        ]
                                        .into_iter()
                                        .collect(),
                                    ));
                                }
                                let out = lyra_rewrite::matcher::substitute_named(
                                    rhs,
                                    &lyra_rewrite::matcher::Bindings::new(),
                                );
                                return self.eval(out);
                            }
                        }
                    }
                }
                self.env.get(&s).cloned().unwrap_or(Value::Symbol(s))
            }
            other => other,
        }
    }
}

// (compat registrations removed)

pub fn evaluate(v: Value) -> Value {
    Evaluator::new().eval(v)
}

// Public registration helpers for stdlib wrappers
#[cfg(feature = "core")]
pub fn register_core(ev: &mut Evaluator) {
    ev.register("Set", set_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Unset", unset_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("SetDelayed", set_delayed_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("With", with_fn as NativeFn, Attributes::HOLD_ALL);
    // Rewrite-related builtins moved to core::rewrite
    crate::core::rewrite::register_rewrite(ev);
    ev.register("SetDownValues", set_downvalues_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("GetDownValues", get_downvalues_fn as NativeFn, Attributes::empty());
    ev.register("SetUpValues", set_upvalues_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("GetUpValues", get_upvalues_fn as NativeFn, Attributes::empty());
    ev.register("SetOwnValues", set_ownvalues_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("GetOwnValues", get_ownvalues_fn as NativeFn, Attributes::empty());
    ev.register("SetSubValues", set_subvalues_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("GetSubValues", get_subvalues_fn as NativeFn, Attributes::empty());
}

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
    // moved to module registrars
}

// Filtered registration variant for tree-shaken builds.
// Only registers symbols that satisfy the provided predicate.
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
    // moved to module registrars
}

pub fn register_explain(ev: &mut Evaluator) { crate::core::schema_explain::register_explain(ev); }

pub fn register_schema(ev: &mut Evaluator) { crate::core::schema_explain::register_schema(ev); }

// Introspection: list current builtin functions with their attributes
pub fn register_introspection(ev: &mut Evaluator) {
    fn describe_builtins_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
        if !args.is_empty() { return Value::Expr { head: Box::new(Value::Symbol("DescribeBuiltins".into())), args }; }
        let mut out: Vec<Value> = Vec::new();
        let snapshot: Vec<(String, Attributes)> = ev.builtins.iter().map(|(name, (_f, attrs))| (name.clone(), *attrs)).collect();
        for (name, attrs) in snapshot.into_iter() {
            let mut attr_list: Vec<Value> = Vec::new();
            if attrs.contains(Attributes::LISTABLE) { attr_list.push(Value::String("LISTABLE".into())); }
            if attrs.contains(Attributes::FLAT) { attr_list.push(Value::String("FLAT".into())); }
            if attrs.contains(Attributes::ORDERLESS) { attr_list.push(Value::String("ORDERLESS".into())); }
            if attrs.contains(Attributes::HOLD_ALL) { attr_list.push(Value::String("HOLD_ALL".into())); }
            if attrs.contains(Attributes::HOLD_FIRST) { attr_list.push(Value::String("HOLD_FIRST".into())); }
            if attrs.contains(Attributes::HOLD_REST) { attr_list.push(Value::String("HOLD_REST".into())); }
            if attrs.contains(Attributes::ONE_IDENTITY) { attr_list.push(Value::String("ONE_IDENTITY".into())); }
            let (summary, params): (String, Vec<String>) = ev.get_doc(&name).unwrap_or((String::new(), Vec::new()));
            let card = Value::Assoc(vec![
                ("id".to_string(), Value::String(name.clone())),
                ("name".to_string(), Value::String(name.clone())),
                ("summary".to_string(), Value::String(summary)),
                ("tags".to_string(), Value::List(vec![])),
                ("params".to_string(), Value::List(params.into_iter().map(Value::String).collect())),
                ("attributes".to_string(), Value::List(attr_list)),
                ("examples".to_string(), Value::List(match ev.get_doc_full(&name) { Some(ent) => ent.examples.into_iter().map(Value::String).collect::<Vec<_>>(), None => vec![] })),
            ].into_iter().collect());
            out.push(card);
        }
        Value::List(out)
    }
    ev.register("DescribeBuiltins", describe_builtins_fn as NativeFn, Attributes::empty());

    fn documentation_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
        if args.is_empty() { let head = Value::Symbol("DescribeBuiltins".to_string()); let expr = Value::Expr { head: Box::new(head), args: vec![] }; return ev.eval(expr); }
        if args.len() == 1 {
            let name = match &args[0] { Value::String(s) | Value::Symbol(s) => s.clone(), other => match ev.eval(other.clone()) { Value::String(s) | Value::Symbol(s) => s, _ => String::new(), }, };
            let head = Value::Symbol("DescribeBuiltins".to_string());
            let desc = ev.eval(Value::Expr { head: Box::new(head), args: vec![] });
            if let Value::List(items) = desc {
                for it in items {
                    if let Value::Assoc(mut m) = it {
                        if let Some(Value::String(n)) = m.get("name").cloned() { if n == name {
                            if let Some((sum, params)) = ev.get_doc(&n) {
                                m.insert("summary".into(), Value::String(sum));
                                m.insert("params".into(), Value::List(params.into_iter().map(Value::String).collect()));
                                if let Some(ent) = ev.get_doc_full(&n) { m.insert("examples".into(), Value::List(ent.examples.into_iter().map(Value::String).collect())); }
                            }
                            return Value::Assoc(m);
                        }}
                    }
                }
            }
            if let Some((sum, params)) = ev.get_doc(&name) {
                let examples = ev.get_doc_full(&name).map(|e| e.examples).unwrap_or_default();
                return Value::Assoc(vec![
                    ("id".to_string(), Value::String(name.clone())),
                    ("name".to_string(), Value::String(name.clone())),
                    ("summary".to_string(), Value::String(sum)),
                    ("params".to_string(), Value::List(params.into_iter().map(Value::String).collect())),
                    ("attributes".to_string(), Value::List(vec![])),
                    ("examples".to_string(), Value::List(examples.into_iter().map(Value::String).collect())),
                ].into_iter().collect());
            }
            return Value::Assoc(std::collections::HashMap::new());
        }
        Value::Expr { head: Box::new(Value::Symbol("Documentation".into())), args }
    }
    ev.register("Documentation", documentation_fn as NativeFn, Attributes::empty());
}

fn listable_thread(ev: &mut Evaluator, f: NativeFn, args: Vec<Value>) -> Value {
    // Option B: broadcast scalars and length-1 lists; require equal lengths otherwise.
    // Determine target length from list args with len > 1; mismatch yields Failure.
    let mut target_len: Option<usize> = None;
    let mut saw_list = false;
    let mut arg_lens: Vec<i64> = Vec::with_capacity(args.len());
    for a in &args {
        if let Value::List(v) = a {
            saw_list = true;
            let l = v.len();
            arg_lens.push(l as i64);
            if l > 1 {
                match target_len {
                    None => target_len = Some(l),
                    Some(t) if t == l => {}
                    Some(_) => {
                        // length mismatch → Failure
                        let mut m = std::collections::HashMap::new();
                        m.insert("message".into(), Value::String("Failure".into()));
                        m.insert("tag".into(), Value::String("Listable::lengthMismatch".into()));
                        m.insert("argLens".into(), Value::List(arg_lens.iter().map(|n| Value::Integer(*n)).collect()));
                        return Value::Assoc(m);
                    }
                }
            }
        } else {
            arg_lens.push(0);
        }
    }
    if !saw_list {
        // Should not happen (caller checks any list), but be safe: just call f normally.
        let evald: Vec<Value> = args.into_iter().map(|x| ev.eval(x)).collect();
        return f(ev, evald);
    }
    let len = target_len.unwrap_or(1);
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let mut elem_args = Vec::with_capacity(args.len());
        for a in &args {
            match a {
                Value::List(vs) => {
                    let l = vs.len();
                    if l == 0 {
                        // Empty list → propagate empty result consistently
                        elem_args.push(Value::List(vec![]));
                    } else if l == 1 {
                        elem_args.push(vs[0].clone());
                    } else {
                        // l > 1 implies len == l due to check above
                        elem_args.push(vs[i].clone());
                    }
                }
                other => elem_args.push(other.clone()),
            }
        }
        let evald: Vec<Value> = elem_args.into_iter().map(|x| ev.eval(x)).collect();
        out.push(f(ev, evald));
    }
    Value::List(out)
}

fn splice_sequences(args: Vec<Value>) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    for a in args.into_iter() {
        if let Value::Expr { head, args } = &a {
            if matches!(&**head, Value::Symbol(s) if s=="Sequence") {
                out.extend(args.clone());
                continue;
            }
        }
        out.push(a);
    }
    out
}

// removed unused compat helpers: plus, map

// moved to concurrency::futures::{future_fn, await_fn, cancel_fn}

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

// ParallelEvaluate[{expr1, expr2, ...}, opts?]
fn parallel_evaluate(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("ParallelEvaluate".into())), args };
    }
    let target = args[0].clone();
    let mut opt_limiter: Option<Arc<ThreadLimiter>> = None;
    let mut opt_deadline: Option<Instant> = None;
    if args.len() >= 2 {
        if let Value::Assoc(m) = ev.eval(args[1].clone()) {
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
                        ("head".to_string(), Value::Symbol("ParallelEvaluate".into())),
                        ("data".to_string(), data),
                    ]
                    .into_iter()
                    .collect(),
                ));
            }
            let mut rxs: Vec<Receiver<Value>> = Vec::with_capacity(items.len());
            for it in items.into_iter() {
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
                    let out = ev2.eval(it);
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
            head: Box::new(Value::Symbol("ParallelEvaluate".into())),
            args: vec![other],
        },
    }
}

// Channels moved to concurrency::channels

// Minimal actor
#[derive(Clone)]
struct ActorInfo {
    chan_id: i64,
}
static ACT_REG: OnceLock<Mutex<HashMap<i64, ActorInfo>>> = OnceLock::new();
static NEXT_ACT_ID: OnceLock<AtomicI64> = OnceLock::new();
fn act_reg() -> &'static Mutex<HashMap<i64, ActorInfo>> {
    ACT_REG.get_or_init(|| Mutex::new(HashMap::new()))
}
fn next_act_id() -> i64 {
    let a = NEXT_ACT_ID.get_or_init(|| AtomicI64::new(1));
    a.fetch_add(1, Ordering::Relaxed)
}

fn actor_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Actor".into())), args };
    }
    let handler = args[0].clone();
    // Create a channel and spawn worker reading messages and applying handler
    let cap = 64usize;
    let chan_id = crate::concurrency::channels::new_channel(cap);
    let chan = crate::concurrency::channels::get_channel(chan_id).unwrap();
    let token = ev.cancel_token.clone();
    let env_snapshot = ev.env.clone();
    let limiter = ev.thread_limiter.clone();
    let deadline = ev.deadline;
    // Spawn a worker loop in pool
    let _ = spawn_task(move || {
        loop {
            let msg_opt = chan.recv(token.clone(), deadline);
            match msg_opt {
                Some(msg) => {
                    if let Some(l) = limiter.as_ref() {
                        l.acquire();
                    }
                    // Evaluate handler[msg] in a fresh evaluator
                    let mut ev2 = if let Some(tok) = token.clone() {
                        Evaluator::with_env_and_token(env_snapshot.clone(), tok)
                    } else {
                        Evaluator::with_env(env_snapshot.clone())
                    };
                    ev2.thread_limiter = limiter.clone();
                    ev2.deadline = deadline;
                    let call = Value::Expr { head: Box::new(handler.clone()), args: vec![msg] };
                    let _ = ev2.eval(call);
                    if let Some(l) = limiter.as_ref() {
                        l.release();
                    }
                }
                None => break,
            }
        }
        Value::Symbol("Done".into())
    });
    let aid = next_act_id();
    act_reg().lock().unwrap().insert(aid, ActorInfo { chan_id });
    Value::Expr { head: Box::new(Value::Symbol("ActorId".into())), args: vec![Value::Integer(aid)] }
}

fn tell_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Tell".into())), args };
    }
    let actv = ev.eval(args[0].clone());
    let msg = args[1].clone();
    let aid = match actv {
        Value::Expr { head, args } if matches!(&*head, Value::Symbol(s) if s=="ActorId") => {
            args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None })
        }
        _ => None,
    };
    if let Some(id) = aid {
        if let Some(info) = act_reg().lock().unwrap().get(&id).cloned() {
            if let Some(ch) = crate::concurrency::channels::get_channel(info.chan_id) {
                let ok = ch.send(msg, ev.cancel_token.clone(), ev.deadline);
                return Value::Boolean(ok);
            }
        }
    }
    Value::Boolean(false)
}

fn stop_actor_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("StopActor".into())), args };
    }
    let actv = ev.eval(args[0].clone());
    let aid = match actv {
        Value::Expr { head, args } if matches!(&*head, Value::Symbol(s) if s=="ActorId") => {
            args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None })
        }
        _ => None,
    };
    if let Some(id) = aid {
        if let Some(info) = act_reg().lock().unwrap().remove(&id) {
            if let Some(ch) = crate::concurrency::channels::get_channel(info.chan_id) {
                ch.close();
            }
            return Value::Boolean(true);
        }
    }
    Value::Boolean(false)
}

fn ask_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() < 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Ask".into())), args };
    }
    // Extract actor id
    let actv = ev.eval(args[0].clone());
    let msg = args[1].clone();
    // Optional opts (e.g., <|TimeoutMs->t|>)
    let mut recv_opts: Option<Value> = None;
    if args.len() >= 3 {
        recv_opts = Some(ev.eval(args[2].clone()));
    }
    let aid = match actv {
        Value::Expr { head, args } if matches!(&*head, Value::Symbol(s) if s=="ActorId") => {
            args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None })
        }
        _ => None,
    };
    if let Some(id) = aid {
        // Create a reply channel (cap=1)
        let ch_id = crate::concurrency::channels::new_channel(1);
        // Build message: <|"msg"->msg, "replyTo"->ChannelId[ch_id]|>
        let reply = Value::Expr {
            head: Box::new(Value::Symbol("ChannelId".into())),
            args: vec![Value::Integer(ch_id)],
        };
        let m = Value::Assoc(
            vec![("msg".to_string(), msg), ("replyTo".to_string(), reply.clone())]
                .into_iter()
                .collect(),
        );
        // Send to actor
        let _ = tell_fn(
            ev,
            vec![
                Value::Expr {
                    head: Box::new(Value::Symbol("ActorId".into())),
                    args: vec![Value::Integer(id)],
                },
                m,
            ],
        );
        // Return a Future awaiting Receive[reply, opts?]
        let recv_expr = if let Some(o) = recv_opts {
            Value::Expr { head: Box::new(Value::Symbol("Receive".into())), args: vec![reply, o] }
        } else {
            Value::Expr { head: Box::new(Value::Symbol("Receive".into())), args: vec![reply] }
        };
                    return crate::concurrency::futures::future_fn(ev, vec![recv_expr]);
    }
    Value::Expr { head: Box::new(Value::Symbol("Ask".into())), args }
}

// legacy logic helpers removed (EvenQ/OddQ now in stdlib)

// legacy string helpers removed (StringTrim/StringContains now in stdlib)

// legacy math helpers removed (Abs/Min/Max now in stdlib)

// legacy list helpers removed (Length/Range/Join/Reverse now in stdlib)

// removed: test echo helper moved to stdlib/testing

// unused trace helper removed

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

// Pattern matching moved to lyra-rewrite

// legacy logic helpers removed (If/Equal/Comparisons/And/Or/Not now in stdlib)

// value_order moved to core::rewrite
pub fn value_order_key(v: &Value) -> String { crate::core::rewrite::value_order_key(v) }

// List/Assoc functions
// legacy list/assoc function removed: Apply

// removed: Total and helpers (now in stdlib)

// legacy list/assoc function removed: Fold

// legacy list/assoc function removed: Select

// legacy assoc functions removed (Keys/Values/Lookup/AssociationMap/KeySort/
// SortBy/GroupBy/Merge) — now in stdlib

// legacy list helpers removed: Flatten (+ helper)

// Thread moved to core::rewrite

// Partition[list, n], Partition[list, n, step]
// legacy list helpers removed: Partition

// Transpose[matrix]
// legacy list helper removed: Transpose

// collect_rules moved to core::rewrite

// Rule application moved to lyra-rewrite

// Replace/All/First/Repeated moved to core::rewrite

// ReplaceAll moved to core::rewrite

// ReplaceRepeated moved to core::rewrite

// ReplaceFirst moved to core::rewrite

fn set_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Set".into())), args };
    }
    match &args[0] {
        Value::Symbol(name) => {
            let v = ev.eval(args[1].clone());
            ev.env.insert(name.clone(), v.clone());
            v
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Set".into())), args },
    }
}

fn unset_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Unset".into())), args };
    }
    match &args[0] {
        Value::Symbol(name) => {
            ev.env.remove(name);
            Value::Symbol("Null".into())
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Unset".into())), args },
    }
}

fn set_downvalues_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // SetDownValues[symbol, {lhs->rhs, ...}]
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("SetDownValues".into())), args };
    }
    let sym = match &args[0] {
        Value::Symbol(s) => s.clone(),
        _ => return Value::Expr { head: Box::new(Value::Symbol("SetDownValues".into())), args },
    };
    let rules_pairs = crate::core::rewrite::collect_rules(ev, args[1].clone());
    let rs: &mut RuleSet = ev.defs.rules_mut(DefKind::Down, &sym);
    rs.0.clear();
    for (lhs, rhs) in rules_pairs {
        rs.push(Rule::immediate(lhs, rhs));
    }
    Value::Symbol("Null".into())
}

fn get_downvalues_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GetDownValues[symbol]
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("GetDownValues".into())), args };
    }
    let sym = match &args[0] {
        Value::Symbol(s) => s.clone(),
        _ => return Value::Expr { head: Box::new(Value::Symbol("GetDownValues".into())), args },
    };
    if let Some(rs) = ev.defs.rules(DefKind::Down, &sym) {
        let items: Vec<Value> = rs
            .iter()
            .map(|r| Value::Expr {
                head: Box::new(Value::Symbol("Rule".into())),
                args: vec![r.lhs.clone(), r.rhs.clone()],
            })
            .collect();
        Value::List(items)
    } else {
        Value::List(vec![])
    }
}

fn set_upvalues_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // SetUpValues[symbol, {lhs->rhs, ...}]
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("SetUpValues".into())), args };
    }
    let sym = match &args[0] {
        Value::Symbol(s) => s.clone(),
        _ => return Value::Expr { head: Box::new(Value::Symbol("SetUpValues".into())), args },
    };
    let rules_pairs = crate::core::rewrite::collect_rules(ev, args[1].clone());
    let rs: &mut RuleSet = ev.defs.rules_mut(DefKind::Up, &sym);
    rs.0.clear();
    for (lhs, rhs) in rules_pairs {
        rs.push(Rule::immediate(lhs, rhs));
    }
    Value::Symbol("Null".into())
}

fn get_upvalues_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GetUpValues[symbol]
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("GetUpValues".into())), args };
    }
    let sym = match &args[0] {
        Value::Symbol(s) => s.clone(),
        _ => return Value::Expr { head: Box::new(Value::Symbol("GetUpValues".into())), args },
    };
    if let Some(rs) = ev.defs.rules(DefKind::Up, &sym) {
        let items: Vec<Value> = rs
            .iter()
            .map(|r| Value::Expr {
                head: Box::new(Value::Symbol("Rule".into())),
                args: vec![r.lhs.clone(), r.rhs.clone()],
            })
            .collect();
        Value::List(items)
    } else {
        Value::List(vec![])
    }
}

fn set_ownvalues_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // SetOwnValues[symbol, {lhs->rhs, ...}]
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("SetOwnValues".into())), args };
    }
    let sym = match &args[0] {
        Value::Symbol(s) => s.clone(),
        _ => return Value::Expr { head: Box::new(Value::Symbol("SetOwnValues".into())), args },
    };
    let rules_pairs = crate::core::rewrite::collect_rules(ev, args[1].clone());
    let rs: &mut RuleSet = ev.defs.rules_mut(DefKind::Own, &sym);
    rs.0.clear();
    for (lhs, rhs) in rules_pairs {
        rs.push(Rule::immediate(lhs, rhs));
    }
    Value::Symbol("Null".into())
}

fn get_ownvalues_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GetOwnValues[symbol]
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("GetOwnValues".into())), args };
    }
    let sym = match &args[0] {
        Value::Symbol(s) => s.clone(),
        _ => return Value::Expr { head: Box::new(Value::Symbol("GetOwnValues".into())), args },
    };
    if let Some(rs) = ev.defs.rules(DefKind::Own, &sym) {
        let items: Vec<Value> = rs
            .iter()
            .map(|r| Value::Expr {
                head: Box::new(Value::Symbol("Rule".into())),
                args: vec![r.lhs.clone(), r.rhs.clone()],
            })
            .collect();
        Value::List(items)
    } else {
        Value::List(vec![])
    }
}

fn set_subvalues_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // SetSubValues[symbol, {lhs->rhs, ...}]
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("SetSubValues".into())), args };
    }
    let sym = match &args[0] {
        Value::Symbol(s) => s.clone(),
        _ => return Value::Expr { head: Box::new(Value::Symbol("SetSubValues".into())), args },
    };
    let rules_pairs = crate::core::rewrite::collect_rules(ev, args[1].clone());
    let rs: &mut RuleSet = ev.defs.rules_mut(DefKind::Sub, &sym);
    rs.0.clear();
    for (lhs, rhs) in rules_pairs {
        rs.push(Rule::immediate(lhs, rhs));
    }
    Value::Symbol("Null".into())
}

fn get_subvalues_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // GetSubValues[symbol]
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("GetSubValues".into())), args };
    }
    let sym = match &args[0] {
        Value::Symbol(s) => s.clone(),
        _ => return Value::Expr { head: Box::new(Value::Symbol("GetSubValues".into())), args },
    };
    if let Some(rs) = ev.defs.rules(DefKind::Sub, &sym) {
        let items: Vec<Value> = rs
            .iter()
            .map(|r| Value::Expr {
                head: Box::new(Value::Symbol("Rule".into())),
                args: vec![r.lhs.clone(), r.rhs.clone()],
            })
            .collect();
        Value::List(items)
    } else {
        Value::List(vec![])
    }
}

fn with_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("With".into())), args };
    }
    let assoc = ev.eval(args[0].clone());
    let body = args[1].clone();
    let mut saved: Vec<(String, Option<Value>)> = Vec::new();
    if let Value::Assoc(m) = assoc {
        for (k, v) in m {
            let old = ev.env.get(&k).cloned();
            let newv = ev.eval(v);
            saved.push((k.clone(), old));
            ev.env.insert(k, newv);
        }
        let result = ev.eval(body);
        // restore
        for (k, ov) in saved.into_iter() {
            if let Some(val) = ov {
                ev.env.insert(k, val);
            } else {
                ev.env.remove(&k);
            }
        }
        result
    } else {
        Value::Expr { head: Box::new(Value::Symbol("With".into())), args }
    }
}
fn set_delayed_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("SetDelayed".into())), args };
    }
    let lhs = args[0].clone();
    let rhs = args[1].clone();
    // Classify lhs: Own (Symbol), Down (Symbol[...] ), Sub ((Symbol[...])[...])
    match lhs.clone() {
        Value::Symbol(s) => {
            // OwnValues[s, { s -> rhs }]
            let rule = Value::Expr {
                head: Box::new(Value::Symbol("Rule".into())),
                args: vec![Value::Symbol(s.clone()), rhs],
            };
            return set_ownvalues_fn(ev, vec![Value::Symbol(s), Value::List(vec![rule])]);
        }
        Value::Expr { head, .. } => {
            match *head {
                Value::Symbol(ref s) => {
                    // DownValues[s, { lhs -> rhs }]
                    let rule = Value::Expr {
                        head: Box::new(Value::Symbol("Rule".into())),
                        args: vec![lhs, rhs],
                    };
                    return set_downvalues_fn(
                        ev,
                        vec![Value::Symbol(s.clone()), Value::List(vec![rule])],
                    );
                }
                Value::Expr { head: inner_head, .. } => {
                    if let Value::Symbol(s) = *inner_head {
                        // SubValues[s, { lhs -> rhs }]
                        let rule = Value::Expr {
                            head: Box::new(Value::Symbol("Rule".into())),
                            args: vec![lhs, rhs],
                        };
                        return set_subvalues_fn(
                            ev,
                            vec![Value::Symbol(s), Value::List(vec![rule])],
                        );
                    }
                    Value::Expr { head: Box::new(Value::Symbol("SetDelayed".into())), args }
                }
                _ => Value::Expr { head: Box::new(Value::Symbol("SetDelayed".into())), args },
            }
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("SetDelayed".into())), args },
    }
}
