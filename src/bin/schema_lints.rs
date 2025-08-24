use lyra::vm::Value;
use std::{fs, path::Path, io};

// Simple Phase 1 schema lints runner
// - Disallow Value::Dict anywhere
// - In normalized modules, disallow snake_case keys in public Associations
// - In normalized modules, disallow returning Value::LyObj (Foreign) from stdlib functions

const NORMALIZED_DIRS: &[&str] = &[
    "src/stdlib/numerical",
    "src/stdlib/signal",
    "src/stdlib/analytics/timeseries.rs",
    "src/stdlib/number_theory",
    "src/stdlib/ai_ml/vector_store.rs",
    "src/stdlib/ai_ml/embeddings.rs",
];

fn is_normalized_file(path: &str) -> bool {
    NORMALIZED_DIRS.iter().any(|p| path.ends_with(p) || path.starts_with(p))
}

fn walk_and_collect<F: FnMut(&str, &str)>(root: &Path, cb: &mut F) -> io::Result<()> {
    if root.is_dir() {
        for entry in fs::read_dir(root)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                walk_and_collect(&path, cb)?;
            } else if let Some(ext) = path.extension() {
                if ext == "rs" {
                    if let Ok(content) = fs::read_to_string(&path) {
                        if let Some(s) = path.to_str() {
                            cb(s, &content);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn main() {
    let mut violations = 0usize;

    let project_root = Path::new(".");
    walk_and_collect(project_root, &mut |path, content| {
        // Global: disallow Value::Dict
        if content.contains("Value::Dict") {
            eprintln!("[DICT] {} contains Value::Dict", path);
            violations += 1;
        }

        // Normalized modules checks only
        if is_normalized_file(path) {
            // snake_case key literals like "foo_bar"
            let re = regex::Regex::new(r#"[a-z]+_[a-zA-Z0-9]+"#).unwrap();
            for m in re.find_iter(content) {
                eprintln!("[CASE] {} uses snake_case key literal {}", path, m.as_str());
                violations += 1;
            }

            // Returned Foreign objects heuristic: look for Ok(Value::LyObj( or return Value::LyObj(
            if content.contains("Ok(Value::LyObj(") || content.contains("return Ok(Value::LyObj(") {
                eprintln!("[FOREIGN] {} returns Foreign object in normalized module", path);
                violations += 1;
            }
        }
    }).expect("failed to scan files");

    if violations == 0 {
        println!("schema_lints: OK (no violations)");
        std::process::exit(0);
    } else {
        eprintln!("schema_lints: {} violation(s) found", violations);
        std::process::exit(1);
    }
}

