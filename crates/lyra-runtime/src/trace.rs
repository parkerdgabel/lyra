use lyra_core::value::Value;
use std::cell::RefCell;

thread_local! {
    static TRACE_BUF: RefCell<Vec<Value>> = RefCell::new(Vec::new());
}

pub(crate) fn trace_push_step(step: Value) {
    TRACE_BUF.with(|b| b.borrow_mut().push(step));
}

pub(crate) fn trace_drain_steps() -> Vec<Value> {
    TRACE_BUF.with(|b| std::mem::take(&mut *b.borrow_mut()))
}
