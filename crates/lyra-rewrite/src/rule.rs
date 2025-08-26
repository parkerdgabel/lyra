use lyra_core::value::Value;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Delayed {
    No,
    Yes,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Rule {
    pub lhs: Value,
    pub rhs: Value,
    pub delayed: Delayed,
}

impl Rule {
    pub fn immediate(lhs: Value, rhs: Value) -> Self {
        Self { lhs, rhs, delayed: Delayed::No }
    }
    pub fn delayed(lhs: Value, rhs: Value) -> Self {
        Self { lhs, rhs, delayed: Delayed::Yes }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RuleSet(pub Vec<Rule>);

impl RuleSet {
    pub fn new() -> Self {
        Self(Vec::new())
    }
    pub fn push(&mut self, r: Rule) {
        self.0.push(r);
    }
    pub fn iter(&self) -> impl Iterator<Item = &Rule> {
        self.0.iter()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
