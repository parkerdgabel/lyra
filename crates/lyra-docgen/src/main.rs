use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;

use lyra_core::value::Value;
use lyra_runtime::Evaluator;

fn value_to_string(v: &Value) -> Option<String> {
    match v { Value::String(s) => Some(s.clone()), Value::Symbol(s) => Some(s.clone()), _ => None }
}

fn get_tags_for(ev: &mut Evaluator, name: &str) -> Vec<String> {
    let q = Value::Expr { head: Box::new(Value::Symbol("ToolsDescribe".into())), args: vec![Value::String(name.to_string())] };
    match ev.eval(q) {
        Value::Assoc(m) => match m.get("tags") { Some(Value::List(vs)) => vs.iter().filter_map(value_to_string).collect(), _ => Vec::new() },
        _ => Vec::new(),
    }
}

fn usage(sig_name: &str, params: &[String]) -> String {
    if params.is_empty() { format!("{}[]", sig_name) }
    else { format!("{}[{}]", sig_name, params.join(", ")) }
}

fn module_for(tags: &[String], name: &str) -> String {
    // Prefer known tags; else heuristic by name; default "misc"
    let lower: Vec<String> = tags.iter().map(|s| s.to_lowercase()).collect();
    let has = |t: &str| lower.iter().any(|x| x==t);
    if has("string") { return "string".into(); }
    if has("list") { return "list".into(); }
    if has("assoc") { return "assoc".into(); }
    if has("logic") { return "logic".into(); }
    if has("math") { return "math".into(); }
    if has("json") || has("yaml") || has("toml") || has("io") { return "io".into(); }
    if has("fs") || has("path") { return "fs".into(); }
    if has("http") || has("net") || has("server") { return "net".into(); }
    if has("dataset") || has("data") { return "dataset".into(); }
    if has("db") || has("sql") { return "db".into(); }
    if has("containers") || has("docker") { return "containers".into(); }
    if has("graph") || has("graphs") { return "graphs".into(); }
    if has("ndarray") || has("array") { return "ndarray".into(); }
    if has("time") || has("schedule") { return "time".into(); }
    if has("logging") || has("log") { return "logging".into(); }
    if has("process") || has("proc") { return "process".into(); }
    if has("crypto") { return "crypto".into(); }
    if has("image") { return "image".into(); }
    if has("audio") { return "audio".into(); }
    if has("media") { return "media".into(); }
    // Heuristic by name prefixes
    let n = name.to_lowercase();
    if ["string", "to", "json", "yaml", "toml"].iter().any(|p| n.starts_with(p)) { return "string".into(); }
    if ["key", "assoc", "association"].iter().any(|p| n.starts_with(p)) { return "assoc".into(); }
    if ["http", "download", "respond", "cors", "authjwt"].iter().any(|p| n.starts_with(p)) { return "net".into(); }
    if ["csv", "parsecsv", "writecsv", "rendercsv"].iter().any(|p| n.starts_with(p)) { return "io".into(); }
    if ["nd", "matrix"].iter().any(|p| n.starts_with(p)) { return "ndarray".into(); }
    if ["graph", "bfs", "dfs", "pagerank", "kcore"].iter().any(|p| n.starts_with(p)) { return "graphs".into(); }
    if ["dataset", "select", "filter", "groupby", "agg", "distinct", "union", "join", "concat", "explaindataset"].iter().any(|p| n.starts_with(p)) { return "dataset".into(); }
    if ["connect", "sql", "exec", "fet", "table", "insert", "upsert", "begin", "commit", "rollback", "listtables"].iter().any(|p| n.starts_with(p)) { return "db".into(); }
    if ["run", "which", "command", "popen", "pipe"].iter().any(|p| n.starts_with(p)) { return "process".into(); }
    if ["mkdir", "remove", "copy", "move", "touch", "glob", "read", "write", "temp", "watch"].iter().any(|p| n.starts_with(p)) { return "fs".into(); }
    if ["configurelogging", "log", "withlogger", "setloglevel", "getlogger"].iter().any(|p| n.starts_with(p)) { return "logging".into(); }
    "misc".into()
}

fn write_module_doc(module: &str, entries: &[(String, String, Vec<String>, Vec<String>, Vec<String>)], out_dir: &PathBuf) -> std::io::Result<()> {
    let mut md = String::new();
    md.push_str(&format!("# {}\n\n", module.to_uppercase()));
    md.push_str("| Function | Usage | Summary |\n|---|---|---|\n");
    for (name, summary, params, _tags, _examples) in entries.iter() {
        let use_str = usage(name, params);
        let safe_summary = summary.replace('|', "\\|");
        md.push_str(&format!("| `{}` | `{}` | {} |\n", name, use_str, safe_summary));
    }
    // Detailed sections with examples
    for (name, summary, params, tags, examples) in entries.iter() {
        if examples.is_empty() { continue; }
        md.push_str(&format!("\n## `{}`\n\n", name));
        md.push_str(&format!("- Usage: `{}`\n", usage(name, params)));
        if !summary.is_empty() { md.push_str(&format!("- Summary: {}\n", summary)); }
        if !tags.is_empty() { md.push_str(&format!("- Tags: {}\n", tags.join(", "))); }
        md.push_str("- Examples:\n");
        for ex in examples {
            md.push_str(&format!("  - `{}`\n", ex.replace('`', "\\`")));
        }
    }
    let mut path = out_dir.clone();
    path.push(format!("{}.md", module));
    fs::write(path, md)
}

fn write_tags_doc(tags_map: &std::collections::BTreeMap<String, Vec<(String, String, Vec<String>)>>, out_dir: &PathBuf) -> std::io::Result<()> {
    let mut md = String::new();
    md.push_str("# Tags Index\n\n");
    // Summary table
    md.push_str("| Tag | Functions |\n|---|---|\n");
    for (tag, entries) in tags_map.iter() {
        md.push_str(&format!("| `{}` | {} |\n", tag, entries.len()));
    }
    // Sections per tag
    for (tag, entries) in tags_map.iter() {
        md.push_str(&format!("\n## `{}`\n\n", tag));
        md.push_str("| Function | Usage | Summary |\n|---|---|---|\n");
        for (name, summary, params) in entries.iter() {
            let use_str = usage(name, params);
            let safe_summary = summary.replace('|', "\\|");
            md.push_str(&format!("| `{}` | `{}` | {} |\n", name, use_str, safe_summary));
        }
    }
    let mut path = out_dir.clone();
    path.push("tags.md");
    std::fs::write(path, md)
}

fn main() -> std::io::Result<()> {
    let mut ev = Evaluator::new();
    lyra_stdlib::register_all(&mut ev);

    // Gather docs via Documentation[]
    let docs = ev.eval(Value::Expr { head: Box::new(Value::Symbol("Documentation".into())), args: vec![] });
    let mut items: Vec<(String, String, Vec<String>, Vec<String>, Vec<String>)> = Vec::new(); // (name, summary, params, tags, examples)
    if let Value::List(vs) = docs {
        for v in vs {
            if let Value::Assoc(m) = v {
                let name = m.get("name").and_then(value_to_string).unwrap_or_default();
                // Hide internal symbols like __Dataset* and __DB* from public docs
                if name.starts_with("__") { continue; }
                let summary = m.get("summary").and_then(value_to_string).unwrap_or_default();
                let params: Vec<String> = match m.get("params") { Some(Value::List(ps)) => ps.iter().filter_map(value_to_string).collect(), _ => Vec::new() };
                let examples: Vec<String> = match m.get("examples") { Some(Value::List(xs)) => xs.iter().filter_map(value_to_string).collect(), _ => Vec::new() };
                let tags = get_tags_for(&mut ev, &name);
                if !name.is_empty() { items.push((name, summary, params, tags, examples)); }
            }
        }
    }
    // Group by module
    let mut by_mod: BTreeMap<String, Vec<(String, String, Vec<String>, Vec<String>, Vec<String>)>> = BTreeMap::new();
    // Group by tag
    let mut by_tag: BTreeMap<String, Vec<(String, String, Vec<String>)>> = BTreeMap::new();
    for (name, summary, params, tags, examples) in items.into_iter() {
        let module = module_for(&tags, &name);
        by_mod.entry(module).or_default().push((name.clone(), summary.clone(), params.clone(), tags.clone(), examples));
        for t in tags {
            by_tag.entry(t).or_default().push((name.clone(), summary.clone(), params.clone()));
        }
    }
    for v in by_mod.values_mut() { v.sort_by(|a,b| a.0.cmp(&b.0)); }
    for v in by_tag.values_mut() { v.sort_by(|a,b| a.0.cmp(&b.0)); }

    // Ensure output dir exists
    let mut out_dir = PathBuf::from("docs/stdlib/GENERATED");
    fs::create_dir_all(&out_dir)?;
    // Write per-module files
    for (module, entries) in by_mod.iter() {
        write_module_doc(module, entries, &out_dir)?;
    }
    // Write tags index
    write_tags_doc(&by_tag, &out_dir)?;

    // Write an index
    let mut index_md = String::from("# Generated Stdlib Index\n\n");
    for (module, entries) in by_mod.iter() {
        index_md.push_str(&format!("- [{}]({}.md) ({} functions)\n", module, module, entries.len()));
    }
    index_md.push_str(&format!("- [tags](tags.md) ({} tags)\n", by_tag.len()));
    let mut idx_path = out_dir.clone(); idx_path.push("README.md");
    fs::write(idx_path, index_md)?;

    // Coverage report: missing summaries
    let mut missing: Vec<String> = Vec::new();
    for (_module, entries) in by_mod.iter() {
        for (name, summary, _params, _tags, _examples) in entries.iter() {
            if summary.trim().is_empty() { missing.push(name.clone()); }
        }
    }
    missing.sort();
    let mut cov = String::new();
    cov.push_str("# Stdlib Documentation Coverage\n\n");
    cov.push_str(&format!("Total missing summaries: {}\n\n", missing.len()));
    if !missing.is_empty() {
        cov.push_str("## Missing\n\n");
        for n in missing { cov.push_str(&format!("- {}\n", n)); }
    }
    let mut cov_path = PathBuf::from("docs/stdlib/COVERAGE.md");
    fs::write(&cov_path, cov)?;
    Ok(())
}
