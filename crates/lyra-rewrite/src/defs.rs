use std::collections::HashMap;
use crate::rule::RuleSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefKind { Own, Down, Up, Sub }

#[derive(Debug, Default)]
pub struct DefinitionStore {
    own: HashMap<String, RuleSet>,
    down: HashMap<String, RuleSet>,
    up: HashMap<String, RuleSet>,
    sub: HashMap<String, RuleSet>,
}

impl DefinitionStore {
    pub fn new() -> Self { Self::default() }

    pub fn rules(&self, kind: DefKind, sym: &str) -> Option<&RuleSet> {
        match kind {
            DefKind::Own => self.own.get(sym),
            DefKind::Down => self.down.get(sym),
            DefKind::Up => self.up.get(sym),
            DefKind::Sub => self.sub.get(sym),
        }
    }

    pub fn rules_mut(&mut self, kind: DefKind, sym: &str) -> &mut RuleSet {
        let map = match kind {
            DefKind::Own => &mut self.own,
            DefKind::Down => &mut self.down,
            DefKind::Up => &mut self.up,
            DefKind::Sub => &mut self.sub,
        };
        map.entry(sym.to_string()).or_insert_with(RuleSet::new)
    }
}

