use crate::eval::Evaluator;

/// Documentation entry for a builtin: summary, parameter names, examples.
#[derive(Clone, Debug)]
pub struct DocEntry {
    pub(crate) summary: String,
    pub(crate) params: Vec<String>,
    pub(crate) examples: Vec<String>,
}

impl DocEntry {
    pub(crate) fn examples(&self) -> Vec<String> { self.examples.clone() }
}

pub(crate) fn set_doc(ev: &mut Evaluator, name: &str, summary: impl Into<String>, params: &[&str]) {
    let entry = ev.docs.entry(name.to_string()).or_insert(DocEntry {
        summary: String::new(),
        params: Vec::new(),
        examples: Vec::new(),
    });
    entry.summary = summary.into();
    entry.params = params.iter().map(|s| (*s).to_string()).collect();
}

pub(crate) fn set_doc_examples(ev: &mut Evaluator, name: &str, examples: &[&str]) {
    let entry = ev.docs.entry(name.to_string()).or_insert(DocEntry {
        summary: String::new(),
        params: Vec::new(),
        examples: Vec::new(),
    });
    entry.examples = examples.iter().map(|s| (*s).to_string()).collect();
}

pub(crate) fn get_doc(ev: &Evaluator, name: &str) -> Option<(String, Vec<String>)> {
    ev.docs.get(name).map(|d| (d.summary.clone(), d.params.clone()))
}

pub(crate) fn get_doc_full(ev: &Evaluator, name: &str) -> Option<DocEntry> {
    ev.docs.get(name).cloned()
}
