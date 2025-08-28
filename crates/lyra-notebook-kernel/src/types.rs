use serde::{Deserialize, Serialize};
use std::time::Duration;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DisplayItem {
    pub mime: String,
    pub data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecResult {
    pub cell_id: Uuid,
    pub duration_ms: u128,
    pub outputs: Vec<DisplayItem>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExecEvent {
    Started { cell_id: Uuid },
    Output { cell_id: Uuid, item: DisplayItem },
    Finished { result: ExecResult },
    Error { cell_id: Uuid, message: String },
}

impl DisplayItem {
    pub fn text(s: impl Into<String>) -> Self {
        Self { mime: "text/plain".into(), data: s.into() }
    }
    pub fn lyra_value_json(v: &lyra_core::value::Value) -> Self {
        let json = serde_json::to_string(v).unwrap_or_else(|_| "{}".into());
        Self { mime: "application/lyra+value".into(), data: json }
    }
}
