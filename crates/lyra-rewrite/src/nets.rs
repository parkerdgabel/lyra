use std::collections::HashMap;
use lyra_core::value::Value;

// Minimal scaffold for discrimination nets to be implemented in M1.
// API intentionally small and internal for now.

#[derive(Default)]
pub struct PatternNet {
    size: usize,
    // Map (head_symbol, arity) -> rule IDs
    by_head_arity: HashMap<(String, usize), Vec<usize>>,
    // Map head_symbol -> rule IDs (any arity)
    by_head_any: HashMap<String, Vec<usize>>,
    // Fallback bucket (no head symbol match)
    general: Vec<usize>,
}

impl PatternNet {
    pub fn new() -> Self { Self { size: 0, by_head_arity: HashMap::new(), by_head_any: HashMap::new(), general: Vec::new() } }

    pub fn len(&self) -> usize { self.size }
    pub fn is_empty(&self) -> bool { self.size == 0 }

    pub fn insert_rule(&mut self, lhs: &Value, id: usize) {
        self.size += 1;
        match lhs {
            Value::Expr { head, args } => {
                if let Value::Symbol(h) = &**head {
                    let arity = args.len();
                    self.by_head_arity.entry((h.clone(), arity)).or_default().push(id);
                    self.by_head_any.entry(h.clone()).or_default().push(id);
                } else {
                    self.general.push(id);
                }
            }
            _ => self.general.push(id),
        }
    }

    pub fn remove_rule(&mut self, _id: usize) -> bool {
        // Simple lazy removal: decrement size; actual compaction not implemented yet.
        if self.size > 0 { self.size -= 1; }
        // Note: for now, we do not physically remove from buckets.
        true
    }

    pub fn candidates<'a>(&'a self, expr: &'a Value) -> impl Iterator<Item=usize> + 'a {
        match expr {
            Value::Expr { head, args } => {
                if let Value::Symbol(h) = &**head {
                    let mut v: Vec<usize> = Vec::new();
                    if let Some(xs) = self.by_head_arity.get(&(h.clone(), args.len())) { v.extend(xs.iter().copied()); }
                    if let Some(xs) = self.by_head_any.get(h) { v.extend(xs.iter().copied()); }
                    v.extend(self.general.iter().copied());
                    return v.into_iter();
                }
                self.general.clone().into_iter()
            }
            _ => self.general.clone().into_iter(),
        }
    }
}

pub fn build_net_for_rules(rules: &[(Value, Value)]) -> PatternNet {
    let mut net = PatternNet::new();
    for (i, (lhs, _)) in rules.iter().enumerate() {
        net.insert_rule(lhs, i);
    }
    net
}
