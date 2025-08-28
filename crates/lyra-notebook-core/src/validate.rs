use crate::schema::Notebook;

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationReport {
    pub valid: bool,
    pub errors: Vec<String>,
}

impl ValidationReport {
    pub fn ok() -> Self {
        Self { valid: true, errors: vec![] }
    }
}

pub fn validate_notebook(nb: &Notebook) -> ValidationReport {
    let mut errors = Vec::new();

    // Unique cell IDs
    {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for c in &nb.cells {
            if !seen.insert(c.id) {
                errors.push(format!("Duplicate cell id: {}", c.id));
            }
        }
    }

    // Basic cell checks
    for c in &nb.cells {
        if c.language.is_empty() {
            errors.push(format!("Cell {} has empty language", c.id));
        }
        if matches!(c.r#type, crate::schema::CellType::Code) && c.input.is_empty() {
            // Not a hard error; allow empty code cells. Leave as a warning in future.
        }
        // Output items must have mime and data
        for (j, d) in c.output.iter().enumerate() {
            if d.mime.trim().is_empty() { errors.push(format!("Cell {} output[{}] mime is empty", c.id, j)); }
        }
        // Reserved keys in meta are allowed; no strict check yet
    }

    ValidationReport { valid: errors.is_empty(), errors }
}
