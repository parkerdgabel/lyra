use crate::schema::{Notebook, CURRENT_VERSION};
use anyhow::{anyhow, Result};
use serde_json::{json, Value as JsonValue};
use std::fs;
use std::path::Path;

pub struct WriteOpts {
    pub include_outputs: bool,
    pub pretty: bool,
}

impl Default for WriteOpts {
    fn default() -> Self {
        Self { include_outputs: false, pretty: true }
    }
}

pub fn read_notebook(path: impl AsRef<Path>) -> Result<Notebook> {
    let bytes = fs::read(path)?;
    let mut v: JsonValue = serde_json::from_slice(&bytes)?;
    // Migration hook
    let version = v
        .get("version")
        .and_then(|x| x.as_str())
        .unwrap_or(CURRENT_VERSION);
    if version != CURRENT_VERSION {
        v = migrate_to_current(v)?;
    }
    let nb: Notebook = serde_json::from_value(v)?;
    Ok(nb)
}

pub fn write_notebook(path: impl AsRef<Path>, nb: &Notebook, opts: WriteOpts) -> Result<()> {
    let mut v = serde_json::to_value(nb)?;
    if !opts.include_outputs {
        strip_outputs(&mut v);
    }
    let s = if opts.pretty {
        serde_json::to_string_pretty(&v)?
    } else {
        serde_json::to_string(&v)?
    };
    fs::write(path, s)?;
    Ok(())
}

fn strip_outputs(v: &mut JsonValue) {
    if let Some(cells) = v.get_mut("cells").and_then(|x| x.as_array_mut()) {
        for cell in cells.iter_mut() {
            if let Some(out) = cell.get_mut("output") {
                *out = JsonValue::Array(vec![]);
            }
        }
    }
}

fn migrate_to_current(v: JsonValue) -> Result<JsonValue> {
    // Placeholder: in future, transform older schemas to CURRENT_VERSION
    // For now, accept same shape and just set version
    let mut obj = v
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow!("Notebook root must be a JSON object"))?;
    obj.insert("version".into(), json!(CURRENT_VERSION));
    Ok(JsonValue::Object(obj))
}
