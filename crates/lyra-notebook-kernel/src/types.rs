use serde::{Deserialize, Serialize};
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
        // Wrap value in a versioned envelope to allow protocol evolution
        #[derive(serde::Serialize)]
        struct Envelope<'a> {
            #[serde(rename = "__meta")] meta: Meta,
            value: &'a lyra_core::value::Value,
        }
        #[derive(serde::Serialize)]
        struct Meta { #[serde(rename = "x-lyra-version")] version: u32 }
        let env = Envelope { meta: Meta { version: 1 }, value: v };
        let json = serde_json::to_string(&env).unwrap_or_else(|_| "{}".into());
        Self { mime: "application/lyra+value".into(), data: json }
    }
}
