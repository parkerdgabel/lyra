use std::{collections::HashSet, fs};
use anyhow::Result;
use lyra_core::Value;
use lyra_parser::Parser;

#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub heads: HashSet<String>,
}

pub struct Analyzer;

impl Analyzer {
    pub fn new() -> Self { Self }

    pub fn analyze_files(&self, files: &[std::path::PathBuf]) -> Result<AnalysisResult> {
        let mut heads: HashSet<String> = HashSet::new();
        for p in files {
            let src = fs::read_to_string(p)?;
            self.analyze_source(&src, &mut heads)?;
        }
        Ok(AnalysisResult { heads })
    }

    pub fn analyze_source(&self, src: &str, heads: &mut HashSet<String>) -> Result<()> {
        let mut parser = Parser::from_source(src);
        let exprs = parser.parse_all()?;
        for v in exprs { collect_heads(&v, heads); }
        Ok(())
    }
}

fn collect_heads(v: &Value, heads: &mut HashSet<String>) {
    match v {
        Value::Expr { head, args } => {
            if let Value::Symbol(s) = &**head { heads.insert(s.clone()); }
            collect_heads(head, heads);
            for a in args { collect_heads(a, heads); }
        }
        Value::List(items) => { for it in items { collect_heads(it, heads); } }
        Value::Assoc(m) => { for (_k, v) in m { collect_heads(v, heads); } }
        Value::Complex { re, im } => { collect_heads(re, heads); collect_heads(im, heads); }
        Value::PureFunction { body, .. } => { collect_heads(body, heads); }
        _ => {}
    }
}
