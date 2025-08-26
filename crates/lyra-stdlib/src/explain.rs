use lyra_runtime::Evaluator;

pub fn register_explain_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    if pred("Explain") {
        lyra_runtime::eval::register_explain(ev);
    }
}

pub fn register_explain(ev: &mut Evaluator) {
    lyra_runtime::eval::register_explain(ev);
}
