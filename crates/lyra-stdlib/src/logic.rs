use crate::register_if;
#[cfg(feature = "tools")]
use crate::tool_spec;
#[cfg(feature = "tools")]
use crate::tools::{add_specs, schema_object_value};
use lyra_core::value::Value;
use lyra_runtime::attrs::Attributes;
use lyra_runtime::Evaluator;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_logic(ev: &mut Evaluator) {
    ev.register("If", if_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("When", when_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Unless", unless_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Switch", switch_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("While", while_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Do", do_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("For", for_fn as NativeFn, Attributes::HOLD_ALL);
    ev.register("Equal", equal as NativeFn, Attributes::LISTABLE);
    ev.register("Less", less as NativeFn, Attributes::LISTABLE);
    ev.register("LessEqual", less_equal as NativeFn, Attributes::LISTABLE);
    ev.register("Greater", greater as NativeFn, Attributes::LISTABLE);
    ev.register("GreaterEqual", greater_equal as NativeFn, Attributes::LISTABLE);
    ev.register("And", and_fn as NativeFn, Attributes::empty());
    ev.register("Or", or_fn as NativeFn, Attributes::empty());
    ev.register("Not", not_fn as NativeFn, Attributes::empty());
    ev.register("EvenQ", even_q as NativeFn, Attributes::LISTABLE);
    ev.register("OddQ", odd_q as NativeFn, Attributes::LISTABLE);
    ev.register("MatchQ", match_q as NativeFn, Attributes::HOLD_ALL);
    ev.register("PatternQ", pattern_q as NativeFn, Attributes::HOLD_ALL);
    // General predicates
    ev.register("NumberQ", number_q as NativeFn, Attributes::empty());
    ev.register("IntegerQ", integer_q as NativeFn, Attributes::empty());
    ev.register("RealQ", real_q as NativeFn, Attributes::empty());
    ev.register("StringQ", string_q as NativeFn, Attributes::empty());
    ev.register("BooleanQ", boolean_q as NativeFn, Attributes::empty());
    ev.register("SymbolQ", symbol_q as NativeFn, Attributes::empty());
    ev.register("ListQ", list_q as NativeFn, Attributes::empty());
    ev.register("AssocQ", assoc_q as NativeFn, Attributes::empty());
    ev.register("EmptyQ", empty_q as NativeFn, Attributes::empty());
    ev.register("NonEmptyQ", nonempty_q as NativeFn, Attributes::empty());
    ev.register("PositiveQ", positive_q as NativeFn, Attributes::empty());
    ev.register("NegativeQ", negative_q as NativeFn, Attributes::empty());
    ev.register("NonPositiveQ", nonpositive_q as NativeFn, Attributes::empty());
    ev.register("NonNegativeQ", nonnegative_q as NativeFn, Attributes::empty());

    #[cfg(feature = "tools")]
    add_specs(vec![
        tool_spec!("Equal", summary: "Test equality across arguments", params: ["args"], tags: ["logic"], input_schema: schema_object_value(vec![ (String::from("args"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("array")))]))) ], vec![]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("Less", summary: "Strictly increasing sequence", params: ["args"], tags: ["logic"], input_schema: schema_object_value(vec![ (String::from("args"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("array")))]))) ], vec![]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("LessEqual", summary: "Non-decreasing sequence", params: ["args"], tags: ["logic"], input_schema: schema_object_value(vec![ (String::from("args"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("array")))]))) ], vec![]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("Greater", summary: "Strictly decreasing sequence", params: ["args"], tags: ["logic"], input_schema: schema_object_value(vec![ (String::from("args"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("array")))]))) ], vec![]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("GreaterEqual", summary: "Non-increasing sequence", params: ["args"], tags: ["logic"], input_schema: schema_object_value(vec![ (String::from("args"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("array")))]))) ], vec![]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("And", summary: "Logical AND (short-circuit)", params: ["args"], tags: ["logic"], input_schema: schema_object_value(vec![ (String::from("args"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("array")))]))) ], vec![]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("Or", summary: "Logical OR (short-circuit)", params: ["args"], tags: ["logic"], input_schema: schema_object_value(vec![ (String::from("args"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("array")))]))) ], vec![]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("Not", summary: "Logical NOT", params: ["x"], tags: ["logic"], input_schema: schema_object_value(vec![ (String::from("x"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))) ], vec![String::from("x")]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("EvenQ", summary: "Is integer even?", params: ["n"], tags: ["logic","math"], input_schema: schema_object_value(vec![ (String::from("n"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("integer")))]))) ], vec![String::from("n")]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("OddQ", summary: "Is integer odd?", params: ["n"], tags: ["logic","math"], input_schema: schema_object_value(vec![ (String::from("n"), Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("integer")))]))) ], vec![String::from("n")]), output_schema: Value::Assoc(std::collections::HashMap::from([(String::from("type"), Value::String(String::from("boolean")))]))),
        tool_spec!("While", summary: "Repeat body while test is True", params: ["test","body"], tags: ["logic","control"], examples: [Value::String("i:=0; While[i<3, i:=i+1]".into())]),
        tool_spec!("Do", summary: "Execute body n times", params: ["body","n"], tags: ["logic","control"], examples: [Value::String("i:=0; Do[i:=i+1, 3]".into())]),
        tool_spec!("For", summary: "C-style loop with init/test/step", params: ["init","test","step","body"], tags: ["logic","control"], examples: [Value::String("i:=0; For[i:=0, i<3, i:=i+1, Null]".into())]),
    ]);
}

pub fn register_logic_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    register_if(ev, pred, "If", if_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "When", when_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "Unless", unless_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "Switch", switch_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "While", while_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "Do", do_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "For", for_fn as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "Equal", equal as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Less", less as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "LessEqual", less_equal as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "Greater", greater as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "GreaterEqual", greater_equal as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "And", and_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Or", or_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "Not", not_fn as NativeFn, Attributes::empty());
    register_if(ev, pred, "EvenQ", even_q as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "OddQ", odd_q as NativeFn, Attributes::LISTABLE);
    register_if(ev, pred, "MatchQ", match_q as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "PatternQ", pattern_q as NativeFn, Attributes::HOLD_ALL);
    register_if(ev, pred, "NumberQ", number_q as NativeFn, Attributes::empty());
    register_if(ev, pred, "IntegerQ", integer_q as NativeFn, Attributes::empty());
    register_if(ev, pred, "RealQ", real_q as NativeFn, Attributes::empty());
    register_if(ev, pred, "StringQ", string_q as NativeFn, Attributes::empty());
    register_if(ev, pred, "BooleanQ", boolean_q as NativeFn, Attributes::empty());
    register_if(ev, pred, "SymbolQ", symbol_q as NativeFn, Attributes::empty());
    register_if(ev, pred, "ListQ", list_q as NativeFn, Attributes::empty());
    register_if(ev, pred, "AssocQ", assoc_q as NativeFn, Attributes::empty());
    register_if(ev, pred, "EmptyQ", empty_q as NativeFn, Attributes::empty());
    register_if(ev, pred, "NonEmptyQ", nonempty_q as NativeFn, Attributes::empty());
    register_if(ev, pred, "PositiveQ", positive_q as NativeFn, Attributes::empty());
    register_if(ev, pred, "NegativeQ", negative_q as NativeFn, Attributes::empty());
    register_if(ev, pred, "NonPositiveQ", nonpositive_q as NativeFn, Attributes::empty());
    register_if(ev, pred, "NonNegativeQ", nonnegative_q as NativeFn, Attributes::empty());
}

fn equal(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    Value::Boolean(args.windows(2).all(|w| w[0] == w[1]))
}
fn less(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    Value::Boolean(args.windows(2).all(|w| value_lt(&w[0], &w[1])))
}
fn less_equal(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    Value::Boolean(args.windows(2).all(|w| !value_gt(&w[0], &w[1])))
}
fn greater(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    Value::Boolean(args.windows(2).all(|w| value_gt(&w[0], &w[1])))
}
fn greater_equal(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    Value::Boolean(args.windows(2).all(|w| !value_lt(&w[0], &w[1])))
}

fn value_lt(a: &Value, b: &Value) -> bool {
    lyra_runtime::eval::value_order_key(a) < lyra_runtime::eval::value_order_key(b)
}
fn value_gt(a: &Value, b: &Value) -> bool {
    lyra_runtime::eval::value_order_key(a) > lyra_runtime::eval::value_order_key(b)
}

fn and_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    for a in args {
        if let Value::Boolean(false) = ev.eval(a) {
            return Value::Boolean(false);
        }
    }
    Value::Boolean(true)
}

// While[test, body]
fn while_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("While".into())), args }; }
    loop {
        let t = ev.eval(args[0].clone());
        if !matches!(t, Value::Boolean(true)) { break; }
        let _ = ev.eval(args[1].clone());
    }
    Value::Symbol("Null".into())
}

// Do[body, n]
fn do_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 { return Value::Expr { head: Box::new(Value::Symbol("Do".into())), args }; }
    let n = match ev.eval(args[1].clone()) { Value::Integer(k) if k >= 0 => k as usize, other => return Value::Expr { head: Box::new(Value::Symbol("Do".into())), args: vec![args[0].clone(), other] } };
    for _ in 0..n { let _ = ev.eval(args[0].clone()); }
    Value::Symbol("Null".into())
}

// For[init, test, step, body]
fn for_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 4 { return Value::Expr { head: Box::new(Value::Symbol("For".into())), args }; }
    let init = args[0].clone();
    let test = args[1].clone();
    let step = args[2].clone();
    let body = args[3].clone();
    let _ = ev.eval(init);
    loop {
        let t = ev.eval(test.clone());
        if !matches!(t, Value::Boolean(true)) { break; }
        let _ = ev.eval(body.clone());
        let _ = ev.eval(step.clone());
    }
    Value::Symbol("Null".into())
}
fn or_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    for a in args {
        if let Value::Boolean(true) = ev.eval(a) {
            return Value::Boolean(true);
        }
    }
    Value::Boolean(false)
}
fn not_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [a] => match ev.eval(a.clone()) {
            Value::Boolean(b) => Value::Boolean(!b),
            v => Value::Expr { head: Box::new(Value::Symbol("Not".into())), args: vec![v] },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("Not".into())), args },
    }
}

fn even_q(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Boolean(n % 2 == 0),
        [other] => Value::Boolean(matches!(other, Value::List(_)) == false && false),
        _ => Value::Expr { head: Box::new(Value::Symbol("EvenQ".into())), args },
    }
}

fn odd_q(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [Value::Integer(n)] => Value::Boolean(n % 2 != 0),
        [other] => Value::Boolean(matches!(other, Value::List(_)) == false && false),
        _ => Value::Expr { head: Box::new(Value::Symbol("OddQ".into())), args },
    }
}

fn match_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 2 {
        return Value::Expr { head: Box::new(Value::Symbol("MatchQ".into())), args };
    }
    let expr = ev.eval(args[0].clone());
    let pat = args[1].clone(); // pattern is held (not evaluated)
                               // Build matcher ctx that can evaluate PatternTest and Condition using current env
    let env_snapshot = ev
        .env_keys()
        .into_iter()
        .map(|k| (k.clone(), ev.eval(Value::Symbol(k))))
        .collect::<std::collections::HashMap<String, Value>>();
    let pred = |pred: &Value, arg: &Value| {
        let mut ev2 = Evaluator::with_env(env_snapshot.clone());
        let call = Value::Expr { head: Box::new(pred.clone()), args: vec![arg.clone()] };
        matches!(ev2.eval(call), Value::Boolean(true))
    };
    let condf = |cond: &Value, binds: &lyra_rewrite::matcher::Bindings| {
        let mut ev2 = Evaluator::with_env(env_snapshot.clone());
        let cond_sub = lyra_rewrite::matcher::substitute_named(cond, binds);
        matches!(ev2.eval(cond_sub), Value::Boolean(true))
    };
    let ctx = lyra_rewrite::matcher::MatcherCtx { eval_pred: Some(&pred), eval_cond: Some(&condf) };
    let ok = lyra_rewrite::matcher::match_rule_with(&ctx, &pat, &expr).is_some();
    Value::Boolean(ok)
}

fn pattern_q(_ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.len() != 1 {
        return Value::Expr { head: Box::new(Value::Symbol("PatternQ".into())), args };
    }
    let v = args[0].clone();
    Value::Boolean(contains_pattern(&v))
}

fn contains_pattern(v: &Value) -> bool {
    match v {
        Value::Symbol(s) => s.contains('_'),
        Value::Expr { head, args } => {
            if let Value::Symbol(hs) = &**head {
                match hs.as_str() {
                    "Blank"
                    | "NamedBlank"
                    | "BlankSequence"
                    | "BlankNullSequence"
                    | "NamedBlankSequence"
                    | "NamedBlankNullSequence"
                    | "PatternTest"
                    | "Condition"
                    | "Alternative"
                    | "Repeated"
                    | "RepeatedNull"
                    | "Optional" => return true,
                    _ => {}
                }
            }
            contains_pattern(head) || args.iter().any(contains_pattern)
        }
        Value::List(items) => items.iter().any(contains_pattern),
        Value::Assoc(m) => m.values().any(contains_pattern),
        _ => false,
    }
}

// General predicate helpers
fn number_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [x] => match ev.eval(x.clone()) {
            Value::Integer(_) | Value::Real(_) | Value::Rational { .. } | Value::BigReal(_) => {
                Value::Boolean(true)
            }
            _ => Value::Boolean(false),
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("NumberQ".into())), args },
    }
}
fn integer_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [x] => match ev.eval(x.clone()) {
            Value::Integer(_) => Value::Boolean(true),
            _ => Value::Boolean(false),
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("IntegerQ".into())), args },
    }
}
fn real_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [x] => match ev.eval(x.clone()) {
            Value::Real(_) | Value::Integer(_) | Value::Rational { .. } | Value::BigReal(_) => {
                Value::Boolean(true)
            }
            _ => Value::Boolean(false),
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("RealQ".into())), args },
    }
}
fn string_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [x] => match ev.eval(x.clone()) {
            Value::String(_) => Value::Boolean(true),
            _ => Value::Boolean(false),
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("StringQ".into())), args },
    }
}
fn boolean_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [x] => match ev.eval(x.clone()) {
            Value::Boolean(_) => Value::Boolean(true),
            _ => Value::Boolean(false),
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("BooleanQ".into())), args },
    }
}
fn symbol_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [x] => match ev.eval(x.clone()) {
            Value::Symbol(_) => Value::Boolean(true),
            _ => Value::Boolean(false),
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("SymbolQ".into())), args },
    }
}
fn list_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [x] => match ev.eval(x.clone()) {
            Value::List(_) => Value::Boolean(true),
            _ => Value::Boolean(false),
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("ListQ".into())), args },
    }
}
fn assoc_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [x] => match ev.eval(x.clone()) {
            Value::Assoc(_) => Value::Boolean(true),
            _ => Value::Boolean(false),
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("AssocQ".into())), args },
    }
}
fn empty_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [x] => match ev.eval(x.clone()) {
            Value::List(v) => Value::Boolean(v.is_empty()),
            Value::Assoc(m) => Value::Boolean(m.is_empty()),
            Value::String(s) => Value::Boolean(s.is_empty()),
            _ => Value::Boolean(false),
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("EmptyQ".into())), args },
    }
}
fn nonempty_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match empty_q(ev, args) {
        Value::Boolean(b) => Value::Boolean(!b),
        other => other,
    }
}
fn positive_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [x] => match ev.eval(x.clone()) {
            Value::Integer(n) => Value::Boolean(n > 0),
            Value::Real(f) => Value::Boolean(f > 0.0),
            Value::Rational { num, den } => Value::Boolean(num > 0 && den > 0),
            _ => Value::Boolean(false),
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("PositiveQ".into())), args },
    }
}
fn negative_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [x] => match ev.eval(x.clone()) {
            Value::Integer(n) => Value::Boolean(n < 0),
            Value::Real(f) => Value::Boolean(f < 0.0),
            Value::Rational { num, den } => Value::Boolean(num < 0 && den > 0),
            _ => Value::Boolean(false),
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("NegativeQ".into())), args },
    }
}
fn nonpositive_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match positive_q(ev, args) {
        Value::Boolean(b) => Value::Boolean(!b),
        other => other,
    }
}
fn nonnegative_q(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match negative_q(ev, args) {
        Value::Boolean(b) => Value::Boolean(!b),
        other => other,
    }
}

// -------- Control flow --------
fn if_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [cond, then_] => match ev.eval(cond.clone()) {
            Value::Boolean(true) => ev.eval(then_.clone()),
            Value::Boolean(false) => Value::Symbol("Null".into()),
            other => Value::Expr {
                head: Box::new(Value::Symbol("If".into())),
                args: vec![other, then_.clone()],
            },
        },
        [cond, then_, else_] => match ev.eval(cond.clone()) {
            Value::Boolean(true) => ev.eval(then_.clone()),
            Value::Boolean(false) => ev.eval(else_.clone()),
            other => Value::Expr {
                head: Box::new(Value::Symbol("If".into())),
                args: vec![other, then_.clone(), else_.clone()],
            },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("If".into())), args },
    }
}

fn when_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [cond, body] => match ev.eval(cond.clone()) {
            Value::Boolean(true) => ev.eval(body.clone()),
            Value::Boolean(false) => Value::Symbol("Null".into()),
            other => Value::Expr {
                head: Box::new(Value::Symbol("When".into())),
                args: vec![other, body.clone()],
            },
        },
        [cond, body, else_] => match ev.eval(cond.clone()) {
            Value::Boolean(true) => ev.eval(body.clone()),
            Value::Boolean(false) => ev.eval(else_.clone()),
            other => Value::Expr {
                head: Box::new(Value::Symbol("When".into())),
                args: vec![other, body.clone(), else_.clone()],
            },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("When".into())), args },
    }
}

fn unless_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    match args.as_slice() {
        [cond, body] => match ev.eval(cond.clone()) {
            Value::Boolean(false) => ev.eval(body.clone()),
            Value::Boolean(true) => Value::Symbol("Null".into()),
            other => Value::Expr {
                head: Box::new(Value::Symbol("Unless".into())),
                args: vec![other, body.clone()],
            },
        },
        [cond, body, else_] => match ev.eval(cond.clone()) {
            Value::Boolean(false) => ev.eval(body.clone()),
            Value::Boolean(true) => ev.eval(else_.clone()),
            other => Value::Expr {
                head: Box::new(Value::Symbol("Unless".into())),
                args: vec![other, body.clone(), else_.clone()],
            },
        },
        _ => Value::Expr { head: Box::new(Value::Symbol("Unless".into())), args },
    }
}

fn switch_fn(ev: &mut Evaluator, args: Vec<Value>) -> Value {
    if args.is_empty() {
        return Value::Expr { head: Box::new(Value::Symbol("Switch".into())), args };
    }
    let subj = ev.eval(args[0].clone());
    let (rules_v, default_v) = match args.as_slice() {
        [_, rules] => (ev.eval(rules.clone()), None),
        [_, rules, default_] => (ev.eval(rules.clone()), Some(default_.clone())),
        _ => (Value::Symbol("Null".into()), None),
    };
    // rules may be Assoc or List of pairs
    // Matching semantics:
    // - If rule key is a function/expr, apply to subj; if True, return evaluated value
    // - Else compare equality key == subj
    if let Value::Assoc(m) = rules_v {
        // try keys in sorted order for determinism
        let mut keys: Vec<String> = m.keys().cloned().collect();
        keys.sort();
        for k in keys {
            if k == "_" {
                continue;
            }
            let key_v = Value::String(k.clone());
            if value_matches(ev, &key_v, &subj) {
                return ev.eval(m.get(&k).cloned().unwrap());
            }
        }
        if let Some(v) = m.get("_") {
            return ev.eval(v.clone());
        }
        if let Some(d) = default_v {
            return ev.eval(d);
        }
        return Value::Symbol("Null".into());
    }
    if let Value::List(items) = rules_v {
        for it in items {
            if let Value::List(mut pair) = it {
                if pair.len() == 2 {
                    let rhs = pair.remove(1);
                    let lhs = pair.remove(0);
                    if value_matches(ev, &lhs, &subj) {
                        return ev.eval(rhs);
                    }
                }
            }
        }
        if let Some(d) = default_v {
            return ev.eval(d);
        }
        return Value::Symbol("Null".into());
    }
    Value::Expr { head: Box::new(Value::Symbol("Switch".into())), args: vec![subj, rules_v] }
}

fn value_matches(ev: &mut Evaluator, test: &Value, subj: &Value) -> bool {
    // Predicate form: test[subj] => True
    if matches!(test, Value::Expr { .. } | Value::PureFunction { .. } | Value::Symbol(_))
        && !matches!(test, Value::String(_))
    {
        let call = Value::Expr { head: Box::new(test.clone()), args: vec![subj.clone()] };
        return matches!(ev.eval(call), Value::Boolean(true));
    }
    // Literal equality
    test == subj
}
