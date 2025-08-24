use std::collections::HashMap;
use lyra_core::value::Value;
use crate::nets::PatternNet;

pub type Bindings = HashMap<String, Value>;

pub struct MatcherCtx<'a> {
    pub eval_pred: Option<&'a dyn Fn(&Value, &Value) -> bool>,
    pub eval_cond: Option<&'a dyn Fn(&Value, &Bindings) -> bool>,
}

impl<'a> Default for MatcherCtx<'a> {
    fn default() -> Self { Self { eval_pred: None, eval_cond: None } }
}

pub fn match_rules_with<'a>(ctx: &MatcherCtx, rules: impl IntoIterator<Item=&'a Value>, expr: &Value) -> Option<(&'a Value, Bindings)> {
    for r in rules.into_iter() {
        if let Some(b) = match_rule_with(ctx, r, expr) { return Some((r, b)); }
    }
    None
}

pub fn match_rules<'a>(rules: impl IntoIterator<Item=&'a Value>, expr: &Value) -> Option<(&'a Value, Bindings)> {
    match_rules_with(&MatcherCtx::default(), rules, expr)
}

pub fn match_rules_indexed_with<'a>(ctx: &MatcherCtx, rules: &'a [Value], net: &PatternNet, expr: &Value) -> Option<(&'a Value, Bindings)> {
    for idx in net.candidates(expr) {
        if let Some(lhs) = rules.get(idx) {
            if let Some(b) = match_rule_with(ctx, lhs, expr) { return Some((lhs, b)); }
        }
    }
    None
}

pub fn match_rules_indexed<'a>(rules: &'a [Value], net: &PatternNet, expr: &Value) -> Option<(&'a Value, Bindings)> {
    match_rules_indexed_with(&MatcherCtx::default(), rules, net, expr)
}

pub fn match_rule_with(ctx: &MatcherCtx, pat: &Value, expr: &Value) -> Option<Bindings> {
    let mut b = Bindings::new();
    if match_pat(ctx, pat, expr, &mut b) { Some(b) } else { None }
}

pub fn match_rule(pat: &Value, expr: &Value) -> Option<Bindings> {
    match_rule_with(&MatcherCtx::default(), pat, expr)
}

fn match_pat(ctx: &MatcherCtx, pat: &Value, expr: &Value, binds: &mut Bindings) -> bool {
    match pat {
        Value::Expr { head, args } => match &**head {
            Value::Symbol(s) if s=="Blank" => {
                if args.is_empty() { return true; }
                return type_matches(&args[0], expr);
            }
            Value::Symbol(s) if s=="NamedBlank" => {
                if args.is_empty() { return false; }
                let name = if let Value::Symbol(n) = &args[0] { n.clone() } else { return false };
                let ty_ok = if args.len()>1 { type_matches(&args[1], expr) } else { true };
                if !ty_ok { return false; }
                if let Some(prev) = binds.get(&name) { if prev != expr { return false; } }
                binds.insert(name, expr.clone());
                true
            }
            Value::Symbol(s) if s=="PatternTest" => {
                if args.len()!=2 { return false; }
                if !match_pat(ctx, &args[0], expr, binds) { return false; }
                if let Some(f) = ctx.eval_pred { return f(&args[1], expr); }
                match &args[1] {
                    Value::Symbol(sym) if sym=="EvenQ" => matches!(expr, Value::Integer(n) if n % 2 == 0),
                    Value::Symbol(sym) if sym=="OddQ" => matches!(expr, Value::Integer(n) if n % 2 != 0),
                    _ => true,
                }
            }
            Value::Symbol(s) if s=="Condition" => {
                if args.len()!=2 { return false; }
                let mut local = binds.clone();
                if !match_pat(ctx, &args[0], expr, &mut local) { return false; }
                if let Some(f) = ctx.eval_cond { if f(&args[1], &local) { *binds = local; return true; } else { return false; } }
                let cond = substitute_named(&args[1], &local);
                if matches!(cond, Value::Boolean(true)) { *binds = local; return true; }
                false
            }
            Value::Symbol(s) if s=="Alternative" => {
                for a in args { let mut local = binds.clone(); if match_pat(ctx, a, expr, &mut local) { *binds = local; return true; } }
                false
            }
            _ => {
                if let Value::Expr { head: h2, args: a2 } = expr {
                    if !match_pat(ctx, head, h2, binds) { return false; }
                    if !match_args(ctx, args, a2, binds) { return false; }
                    true
                } else { false }
            }
        },
        Value::Symbol(s) => match expr { Value::Symbol(s2) => s==s2, _=>false },
        _ => pat == expr,
    }
}

fn type_matches(ty: &Value, expr: &Value) -> bool {
    match ty {
        Value::Symbol(s) => match s.as_str() {
            "Integer" => matches!(expr, Value::Integer(_)),
            "Real" => matches!(expr, Value::Real(_)),
            "String" => matches!(expr, Value::String(_)),
            "Symbol" => matches!(expr, Value::Symbol(_)),
            "List" => matches!(expr, Value::List(_)),
            "Assoc" => matches!(expr, Value::Assoc(_)),
            _ => true,
        },
        _ => true,
    }
}

fn match_args(ctx: &MatcherCtx, pats: &Vec<Value>, exprs: &Vec<Value>, binds: &mut Bindings) -> bool {
    fn min_required(pats: &[Value]) -> usize {
        let mut c = 0usize;
        for p in pats {
            if let Value::Expr { head, .. } = p { if let Value::Symbol(s) = &**head { if s=="BlankNullSequence" || s=="NamedBlankNullSequence" { continue; } } }
            c += 1;
        }
        c
    }
    fn go(ctx: &MatcherCtx, pats: &[Value], exprs: &[Value], binds: &mut Bindings) -> bool {
        if pats.is_empty() { return exprs.is_empty(); }
        let p0 = &pats[0];
        if let Value::Expr { head, args } = p0 {
            if let Value::Symbol(hs) = &**head {
                let (min_take, named, ty_opt, _null_ok) = match hs.as_str() {
                    "BlankSequence" => (1usize, None, args.get(0), false),
                    "BlankNullSequence" => (0usize, None, args.get(0), true),
                    "NamedBlankSequence" => {
                        if args.is_empty() { (1usize, Some(String::new()), None, false) } else {
                            let name = if let Value::Symbol(n) = &args[0] { n.clone() } else { String::new() };
                            let ty = if args.len()>1 { Some(&args[1]) } else { None };
                            (1usize, Some(name), ty, false)
                        }
                    }
                    "NamedBlankNullSequence" => {
                        if args.is_empty() { (0usize, Some(String::new()), None, true) } else {
                            let name = if let Value::Symbol(n) = &args[0] { n.clone() } else { String::new() };
                            let ty = if args.len()>1 { Some(&args[1]) } else { None };
                            (0usize, Some(name), ty, true)
                        }
                    }
                    _ => (usize::MAX, None, None, false),
                };
                if min_take != usize::MAX {
                    let rem_min = min_required(&pats[1..]);
                    let max_take = if exprs.len()>=rem_min { exprs.len()-rem_min } else { 0 };
                    let start = min_take.min(exprs.len());
                    for k in start..=max_take {
                        let slice = &exprs[..k];
                        if let Some(ty) = ty_opt { if !slice.iter().all(|e| type_matches(ty, e)) { continue; } }
                        let mut local = binds.clone();
                        if let Some(n) = named.as_ref() {
                            if !n.is_empty() {
                                let seq = Value::Expr { head: Box::new(Value::Symbol("Sequence".into())), args: slice.to_vec() };
                                if let Some(prev) = local.get(n) { if prev != &seq { continue; } }
                                local.insert(n.clone(), seq);
                            }
                        }
                        if go(ctx, &pats[1..], &exprs[k..], &mut local) { *binds = local; return true; }
                    }
                    return false;
                }
                // Repeated/RepeatedNull: pattern element repeated 1+ or 0+ times
                if (hs == "Repeated" || hs == "RepeatedNull") && !args.is_empty() {
                    let min_take = if hs == "Repeated" { 1 } else { 0 };
                    let unit = &args[0];
                    let rem_min = min_required(&pats[1..]);
                    let max_take = if exprs.len()>=rem_min { exprs.len()-rem_min } else { 0 };
                    let start = min_take.min(exprs.len());
                    for k in start..=max_take {
                        let slice = &exprs[..k];
                        // every element must match unit
                        let mut ok = true;
                        let mut local = binds.clone();
                        for e in slice.iter() {
                            let mut inner = local.clone();
                            if !match_pat(ctx, unit, e, &mut inner) { ok = false; break; }
                            local = inner;
                        }
                        if !ok { continue; }
                        if go(ctx, &pats[1..], &exprs[k..], &mut local) { *binds = local; return true; }
                    }
                    return false;
                }
                // Optional[pat]: either match one element with pat or skip consuming
                if hs == "Optional" {
                    if args.len() >= 1 {
                        // Try consume one if available and matches
                        if !exprs.is_empty() {
                            let mut local = binds.clone();
                            if match_pat(ctx, &args[0], &exprs[0], &mut local) {
                                if go(ctx, &pats[1..], &exprs[1..], &mut local) { *binds = local; return true; }
                            }
                        }
                        // Try consume zero
                        let mut local = binds.clone();
                        if go(ctx, &pats[1..], exprs, &mut local) { *binds = local; return true; }
                        return false;
                    }
                }
            }
        }
        if exprs.is_empty() { return false; }
        let mut local = binds.clone();
        if match_pat(ctx, &pats[0], &exprs[0], &mut local) { if go(ctx, &pats[1..], &exprs[1..], &mut local) { *binds = local; return true; } }
        false
    }
    go(ctx, pats.as_slice(), exprs.as_slice(), binds)
}

pub fn substitute_named(v: &Value, binds: &Bindings) -> Value {
    match v {
        Value::Symbol(s) => binds.get(s).cloned().unwrap_or(v.clone()),
        Value::List(items) => Value::List(items.iter().map(|x| substitute_named(x, binds)).collect()),
        Value::Assoc(m) => Value::Assoc(m.iter().map(|(k,x)| (k.clone(), substitute_named(x, binds))).collect()),
        Value::Expr { head, args } => Value::Expr { head: Box::new(substitute_named(head, binds)), args: args.iter().map(|x| substitute_named(x, binds)).collect() },
        other => other.clone(),
    }
}
