use lyra_core::value::Value;
use lyra_runtime::{Evaluator};
use lyra_runtime::attrs::Attributes;
use crate::register_if;

type NativeFn = fn(&mut Evaluator, Vec<Value>) -> Value;

pub fn register_testing(ev: &mut Evaluator) {
    ev.register("OrderlessEcho", orderless_echo as NativeFn, Attributes::ORDERLESS | Attributes::HOLD_ALL);
    ev.register("FlatEcho", flat_echo as NativeFn, Attributes::FLAT | Attributes::HOLD_ALL);
    ev.register("FlatOrderlessEcho", flat_orderless_echo as NativeFn, Attributes::FLAT | Attributes::ORDERLESS | Attributes::HOLD_ALL);
}

fn orderless_echo(_ev: &mut Evaluator, args: Vec<Value>) -> Value { Value::List(args) }
fn flat_echo(_ev: &mut Evaluator, args: Vec<Value>) -> Value { Value::List(args) }
fn flat_orderless_echo(_ev: &mut Evaluator, args: Vec<Value>) -> Value { Value::List(args) }



pub fn register_testing_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str)->bool) {
    register_if(ev, pred, "OrderlessEcho", orderless_echo as NativeFn, Attributes::ORDERLESS | Attributes::HOLD_ALL);
    register_if(ev, pred, "FlatEcho", flat_echo as NativeFn, Attributes::FLAT | Attributes::HOLD_ALL);
    register_if(ev, pred, "FlatOrderlessEcho", flat_orderless_echo as NativeFn, Attributes::FLAT | Attributes::ORDERLESS | Attributes::HOLD_ALL);
}
