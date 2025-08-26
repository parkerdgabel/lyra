use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Capability {
    Net,
    Fs,
    Db,
    Gpu,
    Process,
    Time,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub symbols: Vec<String>,
    pub features: Vec<String>,
    pub capabilities: Vec<Capability>,
}

impl Manifest {
    pub fn new() -> Self {
        Self { symbols: vec![], features: vec![], capabilities: vec![] }
    }
}
