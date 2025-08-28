use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use uuid::Uuid;

pub const CURRENT_VERSION: &str = "0.1";

pub type Assoc = serde_json::Map<String, JsonValue>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum CellType {
    Code,
    Markdown,
    Text,
    Output,
    Graphics,
    Table,
    Raw,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct CellAttrs: u32 {
        const NONE           = 0;
        const COLLAPSED      = 0b0000_0001;
        const INITIALIZATION = 0b0000_0010;
        const HIDDEN         = 0b0000_0100;
        const LOCKED         = 0b0000_1000;
        const NO_OUTLINE     = 0b0001_0000;
        const NO_LINE_NUM    = 0b0010_0000;
    }
}

impl Default for CellAttrs {
    fn default() -> Self { CellAttrs::NONE }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DisplayData {
    pub mime: String,
    pub data: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<Assoc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<Assoc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Cell {
    #[serde(with = "uuid::serde::compact")]
    pub id: Uuid,
    #[serde(rename = "type")]
    pub r#type: CellType,
    pub language: String,
    pub attrs: CellAttrs,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub input: String,
    #[serde(default)]
    pub output: Vec<DisplayData>,
    #[serde(default)]
    pub meta: Assoc,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Notebook {
    #[serde(with = "uuid::serde::compact")]
    pub id: Uuid,
    pub version: String,
    #[serde(default)]
    pub metadata: Assoc,
    #[serde(default)]
    pub cells: Vec<Cell>,
    #[serde(default)]
    pub styles: Assoc,
    #[serde(default)]
    pub resources: Assoc,
}

impl Notebook {
    pub fn new(id: Uuid) -> Self {
        Self {
            id,
            version: CURRENT_VERSION.to_string(),
            metadata: Assoc::new(),
            cells: Vec::new(),
            styles: Assoc::new(),
            resources: Assoc::new(),
        }
    }
}
