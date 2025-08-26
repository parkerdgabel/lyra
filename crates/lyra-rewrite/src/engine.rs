use crate::matcher::{match_rule_with, substitute_named, MatcherCtx};
use crate::nets::{build_net_for_rules, PatternNet};
use lyra_core::value::Value;

pub fn rewrite_once_with_ctx(ctx: &MatcherCtx, v: Value, rules: &[(Value, Value)]) -> Value {
    for (lhs, rhs) in rules {
        if let Some(b) = match_rule_with(ctx, lhs, &v) {
            let replaced = substitute_named(rhs, &b);
            return replaced;
        }
    }
    match v {
        Value::List(items) => {
            let mut out = Vec::new();
            for it in items {
                let r = rewrite_once_with_ctx(ctx, it, rules);
                if let Value::Expr { head, args } = &r {
                    if matches!(**head, Value::Symbol(ref s) if s=="Sequence") {
                        out.extend(args.clone());
                        continue;
                    }
                }
                out.push(r);
            }
            Value::List(out)
        }
        Value::Assoc(m) => Value::Assoc(
            m.into_iter().map(|(k, x)| (k, rewrite_once_with_ctx(ctx, x, rules))).collect(),
        ),
        Value::Expr { head, args } => {
            let new_head = rewrite_once_with_ctx(ctx, *head, rules);
            let mut new_args: Vec<Value> = Vec::new();
            for a in args {
                let r = rewrite_once_with_ctx(ctx, a, rules);
                if let Value::Expr { head, args } = &r {
                    if matches!(**head, Value::Symbol(ref s) if s=="Sequence") {
                        new_args.extend(args.clone());
                        continue;
                    }
                }
                new_args.push(r);
            }
            if let Value::Symbol(ref s) = new_head {
                if s == "Sequence" {
                    return Value::List(new_args);
                }
            }
            Value::Expr { head: Box::new(new_head), args: new_args }
        }
        other => other,
    }
}

pub fn rewrite_once(v: Value, rules: &[(Value, Value)]) -> Value {
    rewrite_once_with_ctx(&MatcherCtx::default(), v, rules)
}

fn rewrite_once_with_net(
    ctx: &MatcherCtx,
    v: Value,
    rules: &[(Value, Value)],
    net: &PatternNet,
) -> Value {
    // Try top-level candidates via net.
    if let Some(replaced) = try_match_top(ctx, &v, rules, net) {
        return replaced;
    }
    // Otherwise, recurse.
    match v {
        Value::List(items) => {
            let mut out = Vec::new();
            for it in items {
                let r = rewrite_once_with_net(ctx, it, rules, net);
                if let Value::Expr { head, args } = &r {
                    if matches!(**head, Value::Symbol(ref s) if s=="Sequence") {
                        out.extend(args.clone());
                        continue;
                    }
                }
                out.push(r);
            }
            Value::List(out)
        }
        Value::Assoc(m) => Value::Assoc(
            m.into_iter().map(|(k, x)| (k, rewrite_once_with_net(ctx, x, rules, net))).collect(),
        ),
        Value::Expr { head, args } => {
            let new_head = rewrite_once_with_net(ctx, *head, rules, net);
            let mut new_args: Vec<Value> = Vec::new();
            for a in args {
                let r = rewrite_once_with_net(ctx, a, rules, net);
                if let Value::Expr { head, args } = &r {
                    if matches!(**head, Value::Symbol(ref s) if s=="Sequence") {
                        new_args.extend(args.clone());
                        continue;
                    }
                }
                new_args.push(r);
            }
            if let Value::Symbol(ref s) = new_head {
                if s == "Sequence" {
                    return Value::List(new_args);
                }
            }
            let expr = Value::Expr { head: Box::new(new_head), args: new_args };
            if let Some(replaced) = try_match_top(ctx, &expr, rules, net) {
                return replaced;
            }
            expr
        }
        other => other,
    }
}

fn try_match_top(
    ctx: &MatcherCtx,
    v: &Value,
    rules: &[(Value, Value)],
    net: &PatternNet,
) -> Option<Value> {
    for idx in net.candidates(v) {
        if let Some((lhs, rhs)) = rules.get(idx) {
            if let Some(b) = match_rule_with(ctx, lhs, v) {
                return Some(substitute_named(rhs, &b));
            }
        }
    }
    None
}

pub fn rewrite_once_indexed_with_ctx(
    ctx: &MatcherCtx,
    v: Value,
    rules: &[(Value, Value)],
) -> Value {
    let net = build_net_for_rules(rules);
    rewrite_once_with_net(ctx, v, rules, &net)
}

pub fn rewrite_all_with_ctx(ctx: &MatcherCtx, v: Value, rules: &[(Value, Value)]) -> Value {
    let mut curr = v;
    loop {
        let next = rewrite_once_indexed_with_ctx(ctx, curr.clone(), rules);
        if next == curr {
            return next;
        }
        curr = next;
    }
}

pub fn rewrite_all(v: Value, rules: &[(Value, Value)]) -> Value {
    rewrite_all_with_ctx(&MatcherCtx::default(), v, rules)
}

pub fn rewrite_with_limit_with_ctx(
    ctx: &MatcherCtx,
    v: Value,
    rules: &[(Value, Value)],
    mut limit: usize,
) -> Value {
    let net = build_net_for_rules(rules);
    fn go(
        ctx: &MatcherCtx,
        v: Value,
        rules: &[(Value, Value)],
        net: &PatternNet,
        limit: &mut usize,
    ) -> Value {
        if *limit == 0 {
            return v;
        }
        // Top-level match using net candidates
        for idx in net.candidates(&v) {
            if let Some((lhs, rhs)) = rules.get(idx) {
                if let Some(b) = match_rule_with(ctx, lhs, &v) {
                    *limit = limit.saturating_sub(1);
                    return substitute_named(rhs, &b);
                }
            }
        }
        match v {
            Value::List(items) => {
                let mut out = Vec::new();
                for it in items {
                    let r = go(ctx, it, rules, net, limit);
                    if let Value::Expr { head, args } = &r {
                        if matches!(**head, Value::Symbol(ref s) if s=="Sequence") {
                            out.extend(args.clone());
                            continue;
                        }
                    }
                    out.push(r);
                }
                Value::List(out)
            }
            Value::Assoc(m) => Value::Assoc(
                m.into_iter().map(|(k, x)| (k, go(ctx, x, rules, net, limit))).collect(),
            ),
            Value::Expr { head, args } => {
                let new_head = go(ctx, *head, rules, net, limit);
                let mut new_args: Vec<Value> = Vec::new();
                for a in args {
                    if *limit == 0 {
                        new_args.push(a);
                        continue;
                    }
                    let r = go(ctx, a, rules, net, limit);
                    if let Value::Expr { head, args } = &r {
                        if matches!(**head, Value::Symbol(ref s) if s=="Sequence") {
                            new_args.extend(args.clone());
                            continue;
                        }
                    }
                    new_args.push(r);
                }
                if let Value::Symbol(ref s) = new_head {
                    if s == "Sequence" {
                        return Value::List(new_args);
                    }
                }
                Value::Expr { head: Box::new(new_head), args: new_args }
            }
            other => other,
        }
    }
    go(ctx, v, rules, &net, &mut limit)
}

pub fn rewrite_with_limit(v: Value, rules: &[(Value, Value)], limit: usize) -> Value {
    rewrite_with_limit_with_ctx(&MatcherCtx::default(), v, rules, limit)
}
