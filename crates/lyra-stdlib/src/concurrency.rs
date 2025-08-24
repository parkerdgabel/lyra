use lyra_runtime::Evaluator;

pub fn register_concurrency(ev: &mut Evaluator) {
    lyra_runtime::eval::register_concurrency(ev);
}

