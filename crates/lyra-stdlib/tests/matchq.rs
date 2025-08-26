use lyra_runtime::Evaluator;
use lyra_runtime::eval::set_default_registrar;
use lyra_core::pretty::format_value;
use lyra_core::value::Value;
use lyra_stdlib as stdlib;
use std::sync::Once;

static INIT: Once = Once::new();
fn ensure_default() { INIT.call_once(|| set_default_registrar(stdlib::register_all)); }

fn eval_v(ev: &mut Evaluator, v: Value) -> String { format_value(&ev.eval(v)) }

#[test]
fn matchq_and_patternq() {
    ensure_default();
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    // MatchQ[1, _Integer]
    let pat = Value::expr(Value::Symbol("Blank".into()), vec![Value::Symbol("Integer".into())]);
    let r1 = eval_v(&mut ev, Value::expr(Value::Symbol("MatchQ".into()), vec![Value::Integer(1), pat.clone()]));
    assert_eq!(r1, "True");
    // PatternQ[_Integer]
    let r2 = eval_v(&mut ev, Value::expr(Value::Symbol("PatternQ".into()), vec![pat.clone()]));
    assert_eq!(r2, "True");
    // PatternQ[x_Integer] from symbol with underscore
    let r3 = eval_v(&mut ev, Value::expr(Value::Symbol("PatternQ".into()), vec![Value::Symbol("x_Integer".into())]));
    assert_eq!(r3, "True");
    // Non-pattern
    let r4 = eval_v(&mut ev, Value::expr(Value::Symbol("PatternQ".into()), vec![Value::expr(Value::Symbol("Plus".into()), vec![Value::Integer(1), Value::Integer(2)])]));
    assert_eq!(r4, "False");
    // MatchQ with PatternTest
    let pt = Value::expr(Value::Symbol("PatternTest".into()), vec![Value::expr(Value::Symbol("Blank".into()), vec![Value::Symbol("Integer".into())]), Value::Symbol("EvenQ".into())]);
    let r5 = eval_v(&mut ev, Value::expr(Value::Symbol("MatchQ".into()), vec![Value::Integer(4), pt]));
    assert_eq!(r5, "True");
}

