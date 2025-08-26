use lyra_core::pretty::format_value;
use lyra_core::value::Value;
use lyra_runtime::eval::set_default_registrar;
use lyra_runtime::Evaluator;
use lyra_stdlib as stdlib;
use std::sync::Once;

static INIT: Once = Once::new();
fn ensure_default() {
    INIT.call_once(|| set_default_registrar(stdlib::register_all));
}

fn eval_v(ev: &mut Evaluator, v: Value) -> String {
    format_value(&ev.eval(v))
}

#[test]
fn basic_predicates() {
    ensure_default();
    let mut ev = Evaluator::new();
    stdlib::register_all(&mut ev);
    assert_eq!(
        eval_v(&mut ev, Value::expr(Value::Symbol("NumberQ".into()), vec![Value::Real(3.2)])),
        "True"
    );
    assert_eq!(
        eval_v(&mut ev, Value::expr(Value::Symbol("IntegerQ".into()), vec![Value::Integer(3)])),
        "True"
    );
    assert_eq!(
        eval_v(&mut ev, Value::expr(Value::Symbol("RealQ".into()), vec![Value::Integer(3)])),
        "True"
    );
    assert_eq!(
        eval_v(
            &mut ev,
            Value::expr(Value::Symbol("StringQ".into()), vec![Value::String("x".into())])
        ),
        "True"
    );
    assert_eq!(
        eval_v(&mut ev, Value::expr(Value::Symbol("BooleanQ".into()), vec![Value::Boolean(true)])),
        "True"
    );
    assert_eq!(
        eval_v(
            &mut ev,
            Value::expr(Value::Symbol("SymbolQ".into()), vec![Value::Symbol("x".into())])
        ),
        "True"
    );
    assert_eq!(
        eval_v(
            &mut ev,
            Value::expr(
                Value::Symbol("ListQ".into()),
                vec![Value::list(vec![Value::Integer(1), Value::Integer(2)])]
            )
        ),
        "True"
    );
    assert_eq!(
        eval_v(
            &mut ev,
            Value::expr(
                Value::Symbol("AssocQ".into()),
                vec![Value::assoc(vec![("a", Value::Integer(1))])]
            )
        ),
        "True"
    );
    assert_eq!(
        eval_v(&mut ev, Value::expr(Value::Symbol("EmptyQ".into()), vec![Value::list(vec![])])),
        "True"
    );
    assert_eq!(
        eval_v(
            &mut ev,
            Value::expr(
                Value::Symbol("NonEmptyQ".into()),
                vec![Value::list(vec![Value::Integer(1)])]
            )
        ),
        "True"
    );
    assert_eq!(
        eval_v(&mut ev, Value::expr(Value::Symbol("PositiveQ".into()), vec![Value::Integer(2)])),
        "True"
    );
    assert_eq!(
        eval_v(&mut ev, Value::expr(Value::Symbol("NegativeQ".into()), vec![Value::Integer(-1)])),
        "True"
    );
    assert_eq!(
        eval_v(&mut ev, Value::expr(Value::Symbol("NonPositiveQ".into()), vec![Value::Integer(0)])),
        "True"
    );
    assert_eq!(
        eval_v(&mut ev, Value::expr(Value::Symbol("NonNegativeQ".into()), vec![Value::Integer(0)])),
        "True"
    );
}
