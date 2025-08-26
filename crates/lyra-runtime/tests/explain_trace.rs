use lyra_core::value::Value;
use lyra_runtime::{eval::Evaluator, set_default_registrar};
use lyra_stdlib as stdlib;

fn sym(s: &str) -> Value {
    Value::Symbol(s.into())
}
fn int(n: i64) -> Value {
    Value::Integer(n)
}

fn has_condition_step(steps: &Vec<Value>, head_name: &str, expect: bool) -> bool {
    for st in steps {
        if let Value::Assoc(m) = st {
            if let (Some(Value::String(action)), Some(head), Some(Value::Assoc(data))) =
                (m.get("action").cloned(), m.get("head").cloned(), m.get("data").cloned())
            {
                if action == "ConditionEvaluated" {
                    if let Value::Symbol(hs) = head {
                        if hs == head_name {
                            if let Some(Value::Boolean(b)) = data.get("result") {
                                if *b == expect {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    false
}

#[test]
fn explain_records_pattern_test_results() {
    set_default_registrar(stdlib::register_all);
    let mut ev = Evaluator::new();
    // Explain[Replace[2, _?EvenQ -> 9]]
    let lhs = Value::Expr {
        head: Box::new(sym("PatternTest")),
        args: vec![Value::Expr { head: Box::new(sym("Blank")), args: vec![] }, sym("EvenQ")],
    };
    let rule = Value::Expr { head: Box::new(sym("Rule")), args: vec![lhs, int(9)] };
    let repl = Value::Expr { head: Box::new(sym("Replace")), args: vec![int(2), rule] };
    let ex = Value::Expr { head: Box::new(sym("Explain")), args: vec![repl] };
    let out = ev.eval(ex);
    let steps = match out {
        Value::Assoc(m) => match m.get("steps") {
            Some(Value::List(vs)) => vs.clone(),
            _ => vec![],
        },
        _ => vec![],
    };
    assert!(has_condition_step(&steps, "PatternTest", true));

    // Explain[Replace[1, _?EvenQ -> 9]]
    let lhs2 = Value::Expr {
        head: Box::new(sym("PatternTest")),
        args: vec![Value::Expr { head: Box::new(sym("Blank")), args: vec![] }, sym("EvenQ")],
    };
    let rule2 = Value::Expr { head: Box::new(sym("Rule")), args: vec![lhs2, int(9)] };
    let repl2 = Value::Expr { head: Box::new(sym("Replace")), args: vec![int(1), rule2] };
    let ex2 = Value::Expr { head: Box::new(sym("Explain")), args: vec![repl2] };
    let out2 = ev.eval(ex2);
    let steps2 = match out2 {
        Value::Assoc(m) => match m.get("steps") {
            Some(Value::List(vs)) => vs.clone(),
            _ => vec![],
        },
        _ => vec![],
    };
    assert!(has_condition_step(&steps2, "PatternTest", false));
}
