use crate::attrs::Attributes;
use crate::eval::Evaluator;
use crate::trace::{trace_drain_steps, trace_push_step};
use lyra_core::value::Value;

// Canonical order key used for ORDERLESS sorting
pub(crate) fn value_order_key(v: &Value) -> String { value_order(v) }

pub(crate) fn value_order(v: &Value) -> String {
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

// Supports Orderless argument canonicalization in eval
pub(crate) fn splice_sequences(args: Vec<Value>) -> Vec<Value> {
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

// Listable helper used by eval when a builtin has LISTABLE
pub(crate) fn listable_thread(ev: &mut Evaluator, f: crate::eval::NativeFn, args: Vec<Value>) -> Value {
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
                        elem_args.push(Value::List(vec![]));
                    } else if l == 1 {
                        elem_args.push(vs[0].clone());
                    } else {
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

// Collect {Rule[...]...} without evaluating patterns
pub(crate) fn collect_rules(ev: &mut Evaluator, rules_v: Value) -> Vec<(Value, Value)> {
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
    let mut out = extract(rules_v.clone());
    if out.is_empty() {
        out = extract(ev.eval(rules_v));
    }
    out
}

// Thread[f, list1, ...] or Thread[expr]
fn thread(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Expr { head, args: head_args }, _rest @ ..] => {
            // If Thread[expr] form => elementwise map over sublists
            if matches!(&**head, Value::Symbol(s) if s!="Thread") {
                let mut lists: Vec<Vec<Value>> = Vec::new();
                for a in head_args {
                    if let Value::List(v) = a { lists.push(v.clone()); } else { return Value::Expr { head: Box::new(Value::Symbol("Thread".into())), args }; }
                }
                let len = lists.iter().map(|v| v.len()).max().unwrap_or(0);
                let mut out = Vec::with_capacity(len);
                for i in 0..len {
                    let mut call_args = Vec::with_capacity(lists.len());
                    for lst in &lists { call_args.push(lst[i].clone()); }
                    let fh = ev.eval((**head).clone());
                    out.push(ev.eval(Value::Expr { head: Box::new(fh), args: call_args }));
                }
                return Value::List(out);
            }
            Value::Expr { head: Box::new(Value::Symbol("Thread".into())), args }
        }
        [f, lists @ ..] => {
            let mut lists_v = Vec::new();
            for a in lists { if let Value::List(v) = a { lists_v.push(v.clone()); } else { return Value::Expr { head: Box::new(Value::Symbol("Thread".into())), args }; } }
            let len = lists_v.iter().map(|v| v.len()).max().unwrap_or(0);
            let mut out = Vec::with_capacity(len);
            for i in 0..len {
                let mut call_args = Vec::new();
                for lst in &lists_v { call_args.push(lst[i].clone()); }
                let fh = ev.eval(f.clone());
                out.push(ev.eval(Value::Expr { head: Box::new(fh), args: call_args }));
            }
            Value::List(out)
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Thread".into())), args },
    }
}

// Replace/ReplaceAll/ReplaceRepeated/ReplaceFirst wired into lyra_rewrite engine
fn replace(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [expr, rules_v] => {
            let rules = collect_rules(ev, rules_v.clone());
            let target = expr.clone();
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
            let out = lyra_rewrite::engine::rewrite_once_indexed_with_ctx(&ctx, target, &rules);
            if ev.trace_enabled { ev.trace_steps.extend(trace_drain_steps()); }
            ev.eval(out)
        }
        [expr, rules_v, Value::Integer(n)] => {
            let limit = (*n as isize).max(0) as usize;
            let rules = collect_rules(ev, rules_v.clone());
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
            let ctx = lyra_rewrite::matcher::MatcherCtx { eval_pred: Some(&pred), eval_cond: Some(&cond) };
            let out = lyra_rewrite::engine::rewrite_with_limit_with_ctx(&ctx, target, &rules, limit);
            if ev.trace_enabled { ev.trace_steps.extend(trace_drain_steps()); }
            ev.eval(out)
        }
        _ => Value::Expr { head: Box::new(Value::Symbol("Replace".into())), args },
    }
}

fn replace_all_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("ReplaceAll".into())), args }; }
    let rules = collect_rules(ev, args[1].clone());
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
            ].into_iter().collect(),
        );
        trace_push_step(Value::Assoc(vec![
            ("action".to_string(), Value::String("ConditionEvaluated".into())),
            ("head".to_string(), Value::Symbol("PatternTest".into())),
            ("data".to_string(), data),
        ].into_iter().collect()));
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
    let ctx = lyra_rewrite::matcher::MatcherCtx { eval_pred: Some(&pred), eval_cond: Some(&cond) };
    use lyra_rewrite::matcher::{match_rule_with, substitute_named};
    fn replace_all_rec(ctx: &lyra_rewrite::matcher::MatcherCtx, v: Value, rules: &[(Value, Value)]) -> Value {
        for (lhs, rhs) in rules {
            if let Some(b) = match_rule_with(ctx, lhs, &v) { return substitute_named(rhs, &b); }
        }
        match v {
            Value::List(items) => Value::List(items.into_iter().map(|x| replace_all_rec(ctx, x, rules)).collect()),
            Value::Assoc(m) => Value::Assoc(m.into_iter().map(|(k, x)| (k, replace_all_rec(ctx, x, rules))).collect()),
            Value::Expr { head, args } => {
                let new_head = replace_all_rec(ctx, *head, rules);
                let new_args: Vec<Value> = args.into_iter().map(|a| replace_all_rec(ctx, a, rules)).collect();
                Value::Expr { head: Box::new(new_head), args: new_args }
            }
            other => other,
        }
    }
    let out = replace_all_rec(&ctx, target, &rules);
    if ev.trace_enabled { ev.trace_steps.extend(trace_drain_steps()); }
    ev.eval(out)
}

fn replace_repeated_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("ReplaceRepeated".into())), args }; }
    let rules = collect_rules(ev, args[1].clone());
    let target = args[0].clone();
    let env_snapshot = ev.env.clone();
    let pred = |pred: &Value, arg: &Value| {
        let mut ev2 = Evaluator::with_env(env_snapshot.clone());
        let call = Value::Expr { head: Box::new(pred.clone()), args: vec![arg.clone()] };
        let res = matches!(ev2.eval(call), Value::Boolean(true));
        let data = Value::Assoc(vec![
            ("pred".to_string(), pred.clone()),
            ("arg".to_string(), arg.clone()),
            ("result".to_string(), Value::Boolean(res)),
        ].into_iter().collect());
        trace_push_step(Value::Assoc(vec![
            ("action".to_string(), Value::String("ConditionEvaluated".into())),
            ("head".to_string(), Value::Symbol("PatternTest".into())),
            ("data".to_string(), data),
        ].into_iter().collect()));
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
    let ctx = lyra_rewrite::matcher::MatcherCtx { eval_pred: Some(&pred), eval_cond: Some(&cond) };
    let out = lyra_rewrite::engine::rewrite_all_with_ctx(&ctx, target, &rules);
    if ev.trace_enabled { ev.trace_steps.extend(trace_drain_steps()); }
    ev.eval(out)
}

fn replace_first(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("ReplaceFirst".into())), args }; }
    let rules = collect_rules(ev, args[1].clone());
    let target = args[0].clone();
    let env_snapshot = ev.env.clone();
    let pred = |pred: &Value, arg: &Value| {
        let mut ev2 = Evaluator::with_env(env_snapshot.clone());
        let call = Value::Expr { head: Box::new(pred.clone()), args: vec![arg.clone()] };
        let res = matches!(ev2.eval(call), Value::Boolean(true));
        let data = Value::Assoc(vec![
            ("pred".to_string(), pred.clone()),
            ("arg".to_string(), arg.clone()),
            ("result".to_string(), Value::Boolean(res)),
        ].into_iter().collect());
        trace_push_step(Value::Assoc(vec![
            ("action".to_string(), Value::String("ConditionEvaluated".into())),
            ("head".to_string(), Value::Symbol("PatternTest".into())),
            ("data".to_string(), data),
        ].into_iter().collect()));
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
    let ctx = lyra_rewrite::matcher::MatcherCtx { eval_pred: Some(&pred), eval_cond: Some(&cond) };
    use lyra_rewrite::matcher::{match_rule_with, substitute_named};
    fn replace_first_rec(ctx: &lyra_rewrite::matcher::MatcherCtx, v: Value, rules: &[(Value, Value)], left: &mut usize) -> Value {
        if *left == 0 { return v; }
        for (lhs, rhs) in rules {
            if let Some(b) = match_rule_with(ctx, lhs, &v) { *left = left.saturating_sub(1); return substitute_named(rhs, &b); }
        }
        match v {
            Value::List(items) => {
                let mut out = Vec::with_capacity(items.len());
                let mut it = items.into_iter();
                while let Some(x) = it.next() {
                    out.push(replace_first_rec(ctx, x, rules, left));
                    if *left == 0 { out.extend(it); break; }
                }
                Value::List(out)
            }
            Value::Assoc(m) => {
                let mut out_map = std::collections::HashMap::new();
                let mut it = m.into_iter();
                while let Some((k, x)) = it.next() {
                    let val = replace_first_rec(ctx, x, rules, left);
                    out_map.insert(k, val);
                    if *left == 0 { for (kk, vv) in it { out_map.insert(kk, vv); } break; }
                }
                Value::Assoc(out_map)
            }
            Value::Expr { head, args } => {
                let new_head = replace_first_rec(ctx, *head, rules, left);
                let mut new_args = Vec::with_capacity(args.len());
                let mut it = args.into_iter();
                while let Some(a) = it.next() {
                    new_args.push(replace_first_rec(ctx, a, rules, left));
                    if *left == 0 { new_args.extend(it); break; }
                }
                Value::Expr { head: Box::new(new_head), args: new_args }
            }
            other => other,
        }
    }
    let mut left = 1usize;
    let out = replace_first_rec(&ctx, target, &rules, &mut left);
    if ev.trace_enabled { ev.trace_steps.extend(trace_drain_steps()); }
    ev.eval(out)
}

pub fn register_rewrite(ev: &mut Evaluator) {
    ev.register("Replace", replace as crate::eval::NativeFn, Attributes::HOLD_ALL);
    ev.register("ReplaceAll", replace_all_fn as crate::eval::NativeFn, Attributes::HOLD_ALL);
    ev.register("ReplaceRepeated", replace_repeated_fn as crate::eval::NativeFn, Attributes::HOLD_ALL);
    ev.register("ReplaceFirst", replace_first as crate::eval::NativeFn, Attributes::HOLD_ALL);
    ev.register("Thread", thread as crate::eval::NativeFn, Attributes::HOLD_ALL);
}
