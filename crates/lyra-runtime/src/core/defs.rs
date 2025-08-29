use crate::attrs::Attributes;
use crate::eval::Evaluator;
use lyra_core::value::Value;
use lyra_rewrite::defs::DefKind;
use lyra_rewrite::rule::RuleSet;
use lyra_rewrite::rule::Rule;

/// Register rule-definition helpers (`Set*Values`/`Get*Values`) for
/// DownValues, UpValues, OwnValues, and SubValues.
pub fn register_defs(ev: &mut Evaluator) {
    ev.register("SetDownValues", set_downvalues_fn as crate::eval::NativeFn, Attributes::HOLD_ALL);
    ev.register("GetDownValues", get_downvalues_fn as crate::eval::NativeFn, Attributes::empty());
    ev.register("SetUpValues", set_upvalues_fn as crate::eval::NativeFn, Attributes::HOLD_ALL);
    ev.register("GetUpValues", get_upvalues_fn as crate::eval::NativeFn, Attributes::empty());
    ev.register("SetOwnValues", set_ownvalues_fn as crate::eval::NativeFn, Attributes::HOLD_ALL);
    ev.register("GetOwnValues", get_ownvalues_fn as crate::eval::NativeFn, Attributes::empty());
    ev.register("SetSubValues", set_subvalues_fn as crate::eval::NativeFn, Attributes::HOLD_ALL);
    ev.register("GetSubValues", get_subvalues_fn as crate::eval::NativeFn, Attributes::empty());
}

fn set_downvalues_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("SetDownValues".into())), args }; }
    let sym = match &args[0] { Value::Symbol(s) => s.clone(), _ => return Value::Expr { head: Box::new(Value::Symbol("SetDownValues".into())), args } };
    let rules_pairs = crate::core::rewrite::collect_rules(ev, args[1].clone());
    let rs: &mut RuleSet = ev.rules_mut(DefKind::Down, &sym);
    rs.0.clear();
    for (lhs, rhs) in rules_pairs { rs.push(Rule::immediate(lhs, rhs)); }
    Value::Symbol("Null".into())
}

fn get_downvalues_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("GetDownValues".into())), args }; }
    let sym = match &args[0] { Value::Symbol(s) => s.clone(), _ => return Value::Expr { head: Box::new(Value::Symbol("GetDownValues".into())), args } };
    if let Some(rs) = ev.rules(DefKind::Down, &sym) {
        let items: Vec<Value> = rs.iter().map(|r| Value::Expr { head: Box::new(Value::Symbol("Rule".into())), args: vec![r.lhs.clone(), r.rhs.clone()] }).collect();
        Value::List(items)
    } else { Value::List(vec![]) }
}

fn set_upvalues_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("SetUpValues".into())), args }; }
    let sym = match &args[0] { Value::Symbol(s) => s.clone(), _ => return Value::Expr { head: Box::new(Value::Symbol("SetUpValues".into())), args } };
    let rules_pairs = crate::core::rewrite::collect_rules(ev, args[1].clone());
    let rs: &mut RuleSet = ev.rules_mut(DefKind::Up, &sym);
    rs.0.clear();
    for (lhs, rhs) in rules_pairs { rs.push(Rule::immediate(lhs, rhs)); }
    Value::Symbol("Null".into())
}

fn get_upvalues_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("GetUpValues".into())), args }; }
    let sym = match &args[0] { Value::Symbol(s) => s.clone(), _ => return Value::Expr { head: Box::new(Value::Symbol("GetUpValues".into())), args } };
    if let Some(rs) = ev.rules(DefKind::Up, &sym) {
        let items: Vec<Value> = rs.iter().map(|r| Value::Expr { head: Box::new(Value::Symbol("Rule".into())), args: vec![r.lhs.clone(), r.rhs.clone()] }).collect();
        Value::List(items)
    } else { Value::List(vec![]) }
}

fn set_ownvalues_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("SetOwnValues".into())), args }; }
    let sym = match &args[0] { Value::Symbol(s) => s.clone(), _ => return Value::Expr { head: Box::new(Value::Symbol("SetOwnValues".into())), args } };
    let rules_pairs = crate::core::rewrite::collect_rules(ev, args[1].clone());
    let rs: &mut RuleSet = ev.rules_mut(DefKind::Own, &sym);
    rs.0.clear();
    for (lhs, rhs) in rules_pairs { rs.push(Rule::immediate(lhs, rhs)); }
    Value::Symbol("Null".into())
}

fn get_ownvalues_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("GetOwnValues".into())), args }; }
    let sym = match &args[0] { Value::Symbol(s) => s.clone(), _ => return Value::Expr { head: Box::new(Value::Symbol("GetOwnValues".into())), args } };
    if let Some(rs) = ev.rules(DefKind::Own, &sym) {
        let items: Vec<Value> = rs.iter().map(|r| Value::Expr { head: Box::new(Value::Symbol("Rule".into())), args: vec![r.lhs.clone(), r.rhs.clone()] }).collect();
        Value::List(items)
    } else { Value::List(vec![]) }
}

fn set_subvalues_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("SetSubValues".into())), args }; }
    let sym = match &args[0] { Value::Symbol(s) => s.clone(), _ => return Value::Expr { head: Box::new(Value::Symbol("SetSubValues".into())), args } };
    let rules_pairs = crate::core::rewrite::collect_rules(ev, args[1].clone());
    let rs: &mut RuleSet = ev.rules_mut(DefKind::Sub, &sym);
    rs.0.clear();
    for (lhs, rhs) in rules_pairs { rs.push(Rule::immediate(lhs, rhs)); }
    Value::Symbol("Null".into())
}

fn get_subvalues_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 { return Value::Expr { head: Box::new(Value::Symbol("GetSubValues".into())), args }; }
    let sym = match &args[0] { Value::Symbol(s) => s.clone(), _ => return Value::Expr { head: Box::new(Value::Symbol("GetSubValues".into())), args } };
    if let Some(rs) = ev.rules(DefKind::Sub, &sym) {
        let items: Vec<Value> = rs.iter().map(|r| Value::Expr { head: Box::new(Value::Symbol("Rule".into())), args: vec![r.lhs.clone(), r.rhs.clone()] }).collect();
        Value::List(items)
    } else { Value::List(vec![]) }
}
