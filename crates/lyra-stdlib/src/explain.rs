use lyra_runtime::Evaluator;

pub fn register_explain(ev: &mut Evaluator) {
    lyra_runtime::eval::register_explain(ev);
}

