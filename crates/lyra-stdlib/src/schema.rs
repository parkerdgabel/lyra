use lyra_runtime::Evaluator;

pub fn register_schema_filtered(ev: &mut Evaluator, pred: &dyn Fn(&str)->bool) {
    if pred("Schema") { lyra_runtime::eval::register_schema(ev); }
}

pub fn register_schema(ev: &mut Evaluator) {
    lyra_runtime::eval::register_schema(ev);
}
