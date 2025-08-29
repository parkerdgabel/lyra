use crate::attrs::Attributes;
use lyra_core::schema::schema_of;
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
                        return listable_thread(self, fun, args);
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
                eval_args = splice_sequences(eval_args);
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
                    eval_args.sort_by(|x, y| value_order(x).cmp(&value_order(y)));
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
    ev.register("Replace", replace as NativeFn, Attributes::HOLD_ALL);
    ev.register("ReplaceAll", replace_all_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("ReplaceRepeated", replace_repeated_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("ReplaceFirst", replace_first as NativeFn, Attributes::HOLD_ALL);
    ev.register("Thread", thread as NativeFn, Attributes::HOLD_ALL);
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
    ev.register("Scope", scope_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("StartScope", start_scope_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("InScope", in_scope_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("CancelScope", cancel_scope_fn as NativeFn, Attributes::empty());
    ev.register("EndScope", end_scope_fn as NativeFn, Attributes::empty());
    ev.register("ParallelEvaluate", parallel_evaluate as NativeFn, Attributes::HOLD_ALL);
    ev.register("Actor", actor_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Tell", tell_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Ask", ask_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("StopActor", stop_actor_fn as NativeFn, Attributes::empty());
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
    reg("Scope", scope_fn as NativeFn, Attributes::HOLD_ALL);
    reg("StartScope", start_scope_fn as NativeFn, Attributes::HOLD_ALL);
    reg("InScope", in_scope_fn as NativeFn, Attributes::HOLD_ALL);
    reg("CancelScope", cancel_scope_fn as NativeFn, Attributes::empty());
    reg("EndScope", end_scope_fn as NativeFn, Attributes::empty());
    reg("ParallelEvaluate", parallel_evaluate as NativeFn, Attributes::HOLD_ALL);
    // channel builtins registered above if requested
    reg("Actor", actor_fn as NativeFn, Attributes::HOLD_ALL);
    reg("Tell", tell_fn as NativeFn, Attributes::HOLD_ALL);
    reg("Ask", ask_fn as NativeFn, Attributes::HOLD_ALL);
    reg("StopActor", stop_actor_fn as NativeFn, Attributes::empty());
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

// removed: test echo helpers moved to stdlib/testing

fn schema_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Schema".into())), args };
    }
    let v = ev.eval(args[0].clone());
    schema_of(&v)
}

// Scope[<|MaxThreads->n, TimeBudgetMs->ms|>, body]
fn scope_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("Scope".into())), args };
    }
    let opts = ev.eval(args[0].clone());
    let body = args[1].clone();
    // New evaluator inheriting env; establish a scope token
    let mut ev2 = Evaluator::with_env(ev.env.clone());
    ev2.cancel_token = Some(Arc::new(AtomicBool::new(false)));
    if let Value::Assoc(m) = opts {
        if let Some(Value::Integer(n)) = m.get("MaxThreads").or_else(|| m.get("maxThreads")) {
            if *n > 0 {
                ev2.thread_limiter = Some(Arc::new(ThreadLimiter::new(*n as usize)));
            }
        }
        if let Some(Value::Integer(ms)) = m.get("TimeBudgetMs").or_else(|| m.get("timeBudgetMs")) {
            if *ms > 0 {
                ev2.deadline = Some(Instant::now() + Duration::from_millis(*ms as u64));
            }
        }
    }
    if ev.trace_enabled {
        let data = Value::Assoc(
            vec![
                (
                    "maxThreads".to_string(),
                    Value::Integer(ev2.thread_limiter.as_ref().map(|l| l.max_permits() as i64).unwrap_or(-1)),
                ),
                ("hasDeadline".to_string(), Value::Boolean(ev2.deadline.is_some())),
            ]
            .into_iter()
            .collect(),
        );
        ev.trace_steps.push(Value::Assoc(
            vec![
                ("action".to_string(), Value::String("ScopeApply".into())),
                ("head".to_string(), Value::Symbol("Scope".into())),
                ("data".to_string(), data),
            ]
            .into_iter()
            .collect(),
        ));
    }
    ev2.eval(body)
}

// StartScope[opts] -> ScopeId[id]
fn start_scope_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("StartScope".into())), args };
    }
    let opts = ev.eval(args[0].clone());
    let id = next_scope_id();
    let mut ctx =
        ScopeCtx { cancel: Arc::new(AtomicBool::new(false)), limiter: None, deadline: None };
    if let Value::Assoc(m) = opts {
        if let Some(Value::Integer(n)) = m.get("MaxThreads").or_else(|| m.get("maxThreads")) {
            if *n > 0 {
                ctx.limiter = Some(Arc::new(ThreadLimiter::new(*n as usize)));
            }
        }
        if let Some(Value::Integer(ms)) = m.get("TimeBudgetMs").or_else(|| m.get("timeBudgetMs")) {
            if *ms > 0 {
                ctx.deadline = Some(Instant::now() + Duration::from_millis(*ms as u64));
            }
        }
    }
    scope_reg().lock().unwrap().insert(id, ctx);
    Value::Expr { head: Box::new(Value::Symbol("ScopeId".into())), args: vec![Value::Integer(id)] }
}

// InScope[ScopeId[id], body] -- runs body in current evaluator under the scope budgets
fn in_scope_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("InScope".into())), args };
    }
    let sid_val = ev.eval(args[0].clone());
    let sid = match &sid_val {
        Value::Expr { head, args } if matches!(&**head, Value::Symbol(s) if s=="ScopeId") => {
            args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None })
        }
        _ => None,
    };
    if let Some(id) = sid {
        if let Some(ctx) = scope_reg().lock().unwrap().get(&id).cloned() {
            // Save and apply
            let old_tok = ev.cancel_token.clone();
            let old_lim = ev.thread_limiter.clone();
            let old_dead = ev.deadline;
            ev.cancel_token = Some(ctx.cancel.clone());
            ev.thread_limiter = ctx.limiter.clone();
            ev.deadline = ctx.deadline;
            let out = ev.eval(args[1].clone());
            // Restore
            ev.cancel_token = old_tok;
            ev.thread_limiter = old_lim;
            ev.deadline = old_dead;
            return out;
        }
    }
    Value::Assoc(
        vec![
            ("message".to_string(), Value::String("InScope: invalid scope id".into())),
            ("tag".to_string(), Value::String("InScope::invscope".into())),
        ]
        .into_iter()
        .collect(),
    )
}

// CancelScope[ScopeId[id]] -> True/False
fn cancel_scope_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("CancelScope".into())), args };
    }
    let sid_val = ev.eval(args[0].clone());
    let sid = match &sid_val {
        Value::Expr { head, args } if matches!(&**head, Value::Symbol(s) if s=="ScopeId") => {
            args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None })
        }
        _ => None,
    };
    if let Some(id) = sid {
        if let Some(ctx) = scope_reg().lock().unwrap().get(&id) {
            ctx.cancel.store(true, Ordering::Relaxed);
            return Value::Boolean(true);
        }
    }
    Value::Boolean(false)
}

// EndScope[ScopeId[id]] -> True/False (removes from registry)
fn end_scope_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("EndScope".into())), args };
    }
    let sid_val = ev.eval(args[0].clone());
    let sid = match &sid_val {
        Value::Expr { head, args } if matches!(&**head, Value::Symbol(s) if s=="ScopeId") => {
            args.get(0).and_then(|v| if let Value::Integer(i) = v { Some(*i) } else { None })
        }
        _ => None,
    };
    if let Some(id) = sid {
        return Value::Boolean(scope_reg().lock().unwrap().remove(&id).is_some());
    }
    Value::Boolean(false)
}

fn explain_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("Explain".into())), args };
    }
    let expr = args[0].clone();
    // Evaluate under tracing using same env
    let env_snapshot = ev.env.clone();
    let mut ev2 = Evaluator::with_env(env_snapshot);
    ev2.trace_enabled = true;
    let _ = ev2.eval(expr);
    let steps = Value::List(ev2.trace_steps);
    Value::Assoc(
        vec![
            ("steps".to_string(), steps),
            ("algorithm".to_string(), Value::String("stub".into())),
            ("provider".to_string(), Value::String("cpu".into())),
            ("estCost".to_string(), Value::Assoc(Default::default())),
        ]
        .into_iter()
        .collect(),
    )
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
                Value::List(items) => {
                    Value::List(items.iter().map(|x| subst(x, names, args)).collect())
                }
                Value::Assoc(m) => Value::Assoc(
                    m.iter().map(|(k, v)| (k.clone(), subst(v, names, args))).collect(),
                ),
                Value::Expr { head, args: a } => Value::Expr {
                    head: Box::new(subst(head, names, args)),
                    args: a.iter().map(|x| subst(x, names, args)).collect(),
                },
                other => other.clone(),
            }
        }
        return subst(&body, ps, args);
    }
    // Slot-based substitution: # -> args[0], #n -> args[n-1]
    fn subst_slot(v: &Value, args: &Vec<Value>) -> Value {
        match v {
            Value::Slot(None) => args.get(0).cloned().unwrap_or(Value::Symbol("Null".into())),
            Value::Slot(Some(n)) => {
                args.get(n.saturating_sub(1)).cloned().unwrap_or(Value::Symbol("Null".into()))
            }
            Value::List(items) => Value::List(items.iter().map(|x| subst_slot(x, args)).collect()),
            Value::Assoc(m) => {
                Value::Assoc(m.iter().map(|(k, v)| (k.clone(), subst_slot(v, args))).collect())
            }
            Value::Expr { head, args: a } => Value::Expr {
                head: Box::new(subst_slot(head, args)),
                args: a.iter().map(|x| subst_slot(x, args)).collect(),
            },
            other => other.clone(),
        }
    }
    subst_slot(&body, args)
}

// Pattern matching moved to lyra-rewrite

// legacy logic helpers removed (If/Equal/Comparisons/And/Or/Not now in stdlib)

pub fn value_order_key(v: &Value) -> String {
    value_order(v)
}

fn value_order(v: &Value) -> String {
    // Simple canonical string for ordering
    match v {
        Value::Integer(n) => format!("0:{n:020}"),
        Value::Real(f) => format!("1:{:.*}", 16, f),
        Value::BigReal(s) => format!("1b:{s}"),
        Value::Rational { num, den } => format!("1r:{}/{}", num, den),
        Value::Complex { re, im } => format!("1c:{}+{}i", value_order(re), value_order(im)),
        Value::PackedArray { shape, .. } => {
            format!("1p:[{}]", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x"))
        }
        Value::String(s) => format!("2:{s}"),
        Value::Symbol(s) => format!("3:{s}"),
        Value::Boolean(b) => format!("4:{}", if *b { 1 } else { 0 }),
        Value::List(items) => {
            format!("5:[{}]", items.iter().map(|x| value_order(x)).collect::<Vec<_>>().join(";"))
        }
        Value::Assoc(m) => {
            let mut keys: Vec<_> = m.keys().collect();
            keys.sort();
            let parts: Vec<_> = keys
                .into_iter()
                .map(|k| format!("{}=>{}", k, value_order(m.get(k).unwrap())))
                .collect();
            format!("6:<|{}|>", parts.join(","))
        }
        Value::Expr { head, args } => format!(
            "7:{}[{}]",
            value_order(head),
            args.iter().map(|x| value_order(x)).collect::<Vec<_>>().join(",")
        ),
        Value::Slot(n) => format!("8:#{}", n.unwrap_or(1)),
        Value::PureFunction { .. } => "9:PureFunction".into(),
    }
}

// List/Assoc functions
// legacy list/assoc function removed: Apply

// removed: Total and helpers (now in stdlib)

// legacy list/assoc function removed: Fold

// legacy list/assoc function removed: Select

// legacy assoc functions removed (Keys/Values/Lookup/AssociationMap/KeySort/
// SortBy/GroupBy/Merge) — now in stdlib

// legacy list helpers removed: Flatten (+ helper)

fn thread(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    // Forms: Thread[f, list1, ...] or Thread[expr]
    match args.as_slice() {
        [Value::Expr { head, args: a }] => {
            // thread over list arguments of expr
            let lists: Vec<_> = a.iter().map(|x| ev.eval(x.clone())).collect();
            let lens: Vec<usize> = lists
                .iter()
                .map(|v| if let Value::List(l) = v { l.len() } else { usize::MAX })
                .collect();
            let len = lens.into_iter().min().unwrap_or(0);
            let mut out = Vec::with_capacity(len);
            for i in 0..len {
                let mut call_args = Vec::with_capacity(lists.len());
                for lst in &lists {
                    match lst {
                        Value::List(v) => call_args.push(v[i].clone()),
                        other => call_args.push(other.clone()),
                    }
                }
                out.push(ev.eval(Value::Expr { head: head.clone(), args: call_args }));
            }
            Value::List(out)
        }
        [f, rest @ ..] if rest.len() >= 1 => {
            let lists: Vec<Value> = rest.iter().map(|a| ev.eval(a.clone())).collect();
            let lens: Vec<usize> =
                lists.iter().map(|v| if let Value::List(l) = v { l.len() } else { 0 }).collect();
            let len = lens.iter().copied().min().unwrap_or(0);
            let mut out = Vec::with_capacity(len);
            for i in 0..len {
                let mut call_args = Vec::with_capacity(lists.len());
                for lst in &lists {
                    if let Value::List(v) = lst {
                        call_args.push(v[i].clone());
                    }
                }
                let fh = ev.eval(f.clone());
                out.push(ev.eval(Value::Expr { head: Box::new(fh), args: call_args }));
            }
            Value::List(out)
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Thread".into())), args },
    }
}

// Partition[list, n], Partition[list, n, step]
// legacy list helpers removed: Partition

// Transpose[matrix]
// legacy list helper removed: Transpose

fn collect_rules(ev: &mut Evaluator, rules_v: Value) -> Vec<(Value, Value)> {
    fn extract(v: Value) -> Vec<(Value, Value)> {
        let mut out = Vec::new();
        match v {
            Value::List(rs) => {
                for r in rs {
                    if let Value::Expr { head, args } = r {
                        if matches!(*head, Value::Symbol(ref s) if s=="Rule") && args.len() == 2 {
                            out.push((args[0].clone(), args[1].clone()));
                        }
                    }
                }
            }
            Value::Expr { head, args }
                if matches!(*head, Value::Symbol(ref s) if s=="Rule") && args.len() == 2 =>
            {
                out.push((args[0].clone(), args[1].clone()))
            }
            _ => {}
        }
        out
    }
    // Prefer raw structure (to preserve patterns), then fall back to evaluated form
    let mut out = extract(rules_v.clone());
    if out.is_empty() {
        out = extract(ev.eval(rules_v));
    }
    out
}

// Rule application moved to lyra-rewrite

fn replace(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [expr, rules_v] => {
            let rules = collect_rules(ev, rules_v.clone());
            // HoldFirst semantics: do not evaluate target before replacement
            let target = expr.clone();
            // Emit first match for Explain if enabled
            if ev.trace_enabled {
                for (lhs, rhs) in &rules {
                    if lyra_rewrite::matcher::match_rule(lhs, &target).is_some() {
                        let data = Value::Assoc(
                            vec![
                                ("lhs".to_string(), lhs.clone()),
                                ("rhs".to_string(), rhs.clone()),
                            ]
                            .into_iter()
                            .collect(),
                        );
                        ev.trace_steps.push(Value::Assoc(
                            vec![
                                ("action".to_string(), Value::String("RuleMatch".into())),
                                ("head".to_string(), Value::Symbol("Replace".into())),
                                ("data".to_string(), data),
                            ]
                            .into_iter()
                            .collect(),
                        ));
                        break;
                    }
                }
            }
            // Build matcher context with closures that evaluate predicates/conditions and push Explain steps
            let env_snapshot = ev.env.clone();
            let pred = |pred: &Value, arg: &Value| {
                let mut ev2 = Evaluator::with_env(env_snapshot.clone());
                let call = Value::Expr { head: Box::new(pred.clone()), args: vec![arg.clone()] };
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
                        ("action".to_string(), Value::String("ConditionEvaluated".into())),
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
                        ("action".to_string(), Value::String("ConditionEvaluated".into())),
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
            let out = lyra_rewrite::engine::rewrite_once_indexed_with_ctx(&ctx, target, &rules);
            if ev.trace_enabled {
                ev.trace_steps.extend(trace_drain_steps());
            }
            return ev.eval(out);
        }
        [expr, rules_v, Value::Integer(n)] => {
            let limit = (*n as isize).max(0) as usize;
            let rules = collect_rules(ev, rules_v.clone());
            // HoldFirst semantics
            let target = expr.clone();
            let env_snapshot = ev.env.clone();
            let pred = |pred: &Value, arg: &Value| {
                let mut ev2 = Evaluator::with_env(env_snapshot.clone());
                let call = Value::Expr { head: Box::new(pred.clone()), args: vec![arg.clone()] };
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
                        ("action".to_string(), Value::String("ConditionEvaluated".into())),
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
                        ("action".to_string(), Value::String("ConditionEvaluated".into())),
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
            let out =
                lyra_rewrite::engine::rewrite_with_limit_with_ctx(&ctx, target, &rules, limit);
            if ev.trace_enabled {
                ev.trace_steps.extend(trace_drain_steps());
            }
            return ev.eval(out);
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Replace".into())), args },
    }
}

fn replace_all_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("ReplaceAll".into())), args };
    }
    let rules = collect_rules(ev, args[1].clone());
    // HoldFirst semantics
    let target = args[0].clone();
    let env_snapshot = ev.env.clone();
    let pred = |pred: &Value, arg: &Value| {
        let mut ev2 = Evaluator::with_env(env_snapshot.clone());
        let call = Value::Expr { head: Box::new(pred.clone()), args: vec![arg.clone()] };
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
                ("action".to_string(), Value::String("ConditionEvaluated".into())),
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
                ("action".to_string(), Value::String("ConditionEvaluated".into())),
                ("head".to_string(), Value::Symbol("Condition".into())),
                ("data".to_string(), data),
            ]
            .into_iter()
            .collect(),
        ));
        res
    };
    let ctx = lyra_rewrite::matcher::MatcherCtx { eval_pred: Some(&pred), eval_cond: Some(&cond) };
    // ReplaceAll: general recursive traversal using matcher
    use lyra_rewrite::matcher::{match_rule_with, substitute_named};
    fn replace_all_rec(
        ctx: &lyra_rewrite::matcher::MatcherCtx,
        v: Value,
        rules: &[(Value, Value)],
    ) -> Value {
        // Top-level match
        for (lhs, rhs) in rules {
            if let Some(b) = match_rule_with(ctx, lhs, &v) {
                return substitute_named(rhs, &b);
            }
        }
        match v {
            Value::List(items) => {
                Value::List(items.into_iter().map(|x| replace_all_rec(ctx, x, rules)).collect())
            }
            Value::Assoc(m) => Value::Assoc(
                m.into_iter().map(|(k, x)| (k, replace_all_rec(ctx, x, rules))).collect(),
            ),
            Value::Expr { head, args } => {
                let new_head = replace_all_rec(ctx, *head, rules);
                let new_args: Vec<Value> =
                    args.into_iter().map(|a| replace_all_rec(ctx, a, rules)).collect();
                Value::Expr { head: Box::new(new_head), args: new_args }
            }
            other => other,
        }
    }
    let out = replace_all_rec(&ctx, target, &rules);
    if ev.trace_enabled {
        ev.trace_steps.extend(trace_drain_steps());
    }
    return ev.eval(out);
}

fn replace_repeated_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("ReplaceRepeated".into())), args };
    }
    let rules = collect_rules(ev, args[1].clone());
    // HoldFirst semantics
    let target = args[0].clone();
    let env_snapshot = ev.env.clone();
    let pred = |pred: &Value, arg: &Value| {
        let mut ev2 = Evaluator::with_env(env_snapshot.clone());
        let call = Value::Expr { head: Box::new(pred.clone()), args: vec![arg.clone()] };
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
                ("action".to_string(), Value::String("ConditionEvaluated".into())),
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
                ("action".to_string(), Value::String("ConditionEvaluated".into())),
                ("head".to_string(), Value::Symbol("Condition".into())),
                ("data".to_string(), data),
            ]
            .into_iter()
            .collect(),
        ));
        res
    };
    let ctx = lyra_rewrite::matcher::MatcherCtx { eval_pred: Some(&pred), eval_cond: Some(&cond) };
    // ReplaceRepeated: iterate until fixed point
    let out = lyra_rewrite::engine::rewrite_all_with_ctx(&ctx, target, &rules);
    if ev.trace_enabled {
        ev.trace_steps.extend(trace_drain_steps());
    }
    return ev.eval(out);
}

fn replace_first(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("ReplaceFirst".into())), args };
    }
    let rules = collect_rules(ev, args[1].clone());
    // HoldFirst semantics
    let target = args[0].clone();
    let _limit = 1usize;
    let env_snapshot = ev.env.clone();
    let pred = |pred: &Value, arg: &Value| {
        let mut ev2 = Evaluator::with_env(env_snapshot.clone());
        let call = Value::Expr { head: Box::new(pred.clone()), args: vec![arg.clone()] };
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
                ("action".to_string(), Value::String("ConditionEvaluated".into())),
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
                ("action".to_string(), Value::String("ConditionEvaluated".into())),
                ("head".to_string(), Value::Symbol("Condition".into())),
                ("data".to_string(), data),
            ]
            .into_iter()
            .collect(),
        ));
        res
    };
    let ctx = lyra_rewrite::matcher::MatcherCtx { eval_pred: Some(&pred), eval_cond: Some(&cond) };
    // ReplaceFirst via recursive traversal with limit 1 (and elementwise fast path handled in ReplaceAll)
    use lyra_rewrite::matcher::{match_rule_with, substitute_named};
    fn replace_first_rec(
        ctx: &lyra_rewrite::matcher::MatcherCtx,
        v: Value,
        rules: &[(Value, Value)],
        left: &mut usize,
    ) -> Value {
        if *left == 0 {
            return v;
        }
        for (lhs, rhs) in rules {
            if let Some(b) = match_rule_with(ctx, lhs, &v) {
                *left = left.saturating_sub(1);
                return substitute_named(rhs, &b);
            }
        }
        match v {
            Value::List(items) => {
                let mut out = Vec::with_capacity(items.len());
                let mut it = items.into_iter();
                while let Some(x) = it.next() {
                    out.push(replace_first_rec(ctx, x, rules, left));
                    if *left == 0 {
                        out.extend(it);
                        break;
                    }
                }
                Value::List(out)
            }
            Value::Assoc(m) => {
                let mut out_map = std::collections::HashMap::new();
                let mut it = m.into_iter();
                while let Some((k, x)) = it.next() {
                    let val = replace_first_rec(ctx, x, rules, left);
                    out_map.insert(k, val);
                    if *left == 0 {
                        for (kk, vv) in it {
                            out_map.insert(kk, vv);
                        }
                        break;
                    }
                }
                Value::Assoc(out_map)
            }
            Value::Expr { head, args } => {
                let new_head = replace_first_rec(ctx, *head, rules, left);
                let mut new_args = Vec::with_capacity(args.len());
                let mut it = args.into_iter();
                while let Some(a) = it.next() {
                    new_args.push(replace_first_rec(ctx, a, rules, left));
                    if *left == 0 {
                        new_args.extend(it);
                        break;
                    }
                }
                Value::Expr { head: Box::new(new_head), args: new_args }
            }
            other => other,
        }
    }
    let mut left = 1usize;
    let out = replace_first_rec(&ctx, target, &rules, &mut left);
    if ev.trace_enabled {
        ev.trace_steps.extend(trace_drain_steps());
    }
    return ev.eval(out);
}

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
    let rules_pairs = collect_rules(ev, args[1].clone());
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
    let rules_pairs = collect_rules(ev, args[1].clone());
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
    let rules_pairs = collect_rules(ev, args[1].clone());
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
    let rules_pairs = collect_rules(ev, args[1].clone());
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
