use lyra_runtime::Evaluator;

pub fn register_schema(ev: &mut Evaluator) {
    lyra_runtime::eval::register_schema(ev);
}

