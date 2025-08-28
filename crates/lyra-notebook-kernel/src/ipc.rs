use crate::types::{ExecEvent, ExecResult};
use lyra_notebook_core as nb;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRequest { pub path: String }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenResponse { pub session_id: Uuid, pub notebook: nb::schema::Notebook }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteCellRequest { pub session_id: Uuid, pub cell_id: Uuid }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteCellResponse { pub result: ExecResult }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteAllRequest { pub session_id: Uuid, pub ids: Option<Vec<Uuid>>, pub method: Option<String> }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteAllResponse { pub results: Vec<ExecResult> }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptRequest { pub session_id: Uuid }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptResponse { pub ok: bool }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveRequest { pub session_id: Uuid, pub path: String, pub include_outputs: Option<bool>, pub pretty: Option<bool> }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveResponse { pub ok: bool }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KernelEvent {
    Exec(ExecEvent),
}
