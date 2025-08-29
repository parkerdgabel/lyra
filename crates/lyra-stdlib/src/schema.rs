use lyra_runtime::Evaluator;

/// Conditionally re-export runtime `Schema` helpers based on `pred`.
pub fn register_schema_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str) -> bool) {
    if pred("Schema") {
        lyra_runtime::eval::register_schema(ev);
    }
}

/// Re-export runtime `Schema` helpers for stdlib.
pub fn register_schema(ev: &mut Evaluator) {
    lyra_runtime::eval::register_schema(ev);
}
