// removed unused State import

use crate::api;
use lyra_notebook_core as nbcore;
use serde_json::json;
// removed unused Manager import
use tauri::Emitter;
use anyhow::Result;
use lyra_notebook_kernel as kernel;
use uuid::Uuid;
use serde_json::Value as JsonValue;

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_open_notebook(path: String) -> Result<kernel::OpenResponse, String> {
    api::open_notebook(&path).map_err(|e| e.to_string())
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_execute_cell(session_id: String, cell_id: String) -> Result<kernel::ExecResult, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    let cid = Uuid::try_parse(&cell_id).map_err(|e| e.to_string())?;
    api::execute_cell(sid, cid).map_err(|e| e.to_string())
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_execute_cell_nocache(session_id: String, cell_id: String) -> Result<kernel::ExecResult, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    let cid = Uuid::try_parse(&cell_id).map_err(|e| e.to_string())?;
    api::execute_cell_nocache(sid, cid).map_err(|e| e.to_string())
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_new_notebook(title: Option<String>) -> Result<kernel::OpenResponse, String> {
    api::new_notebook(title.as_deref()).map_err(|e| e.to_string())
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_add_cell(session_id: String, cell_type: String) -> Result<nbcore::schema::Notebook, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    api::add_cell(sid, &cell_type).map_err(|e| e.to_string())
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_delete_cell(session_id: String, cell_id: String) -> Result<nbcore::schema::Notebook, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    let cid = Uuid::try_parse(&cell_id).map_err(|e| e.to_string())?;
    api::delete_cell(sid, cid).map_err(|e| e.to_string())
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_execute_cell_events(session_id: String, cell_id: String) -> Result<Vec<kernel::ExecEvent>, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    let cid = Uuid::try_parse(&cell_id).map_err(|e| e.to_string())?;
    api::execute_cell_events(sid, cid).map_err(|e| e.to_string())
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_execute_text(session_id: String, text: String) -> Result<crate::api::ExecTextResult, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    api::execute_text(sid, text).map_err(|e| e.to_string())
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_execute_all(session_id: String, ids: Option<Vec<String>>, method: Option<String>, ignore_cache: Option<bool>) -> Result<Vec<kernel::ExecResult>, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    let ids_parsed: Option<Vec<Uuid>> = ids.map(|v| v.into_iter().filter_map(|s| Uuid::try_parse(&s).ok()).collect());
    api::execute_all(sid, ids_parsed, method, ignore_cache.unwrap_or(false)).map_err(|e| e.to_string())
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_interrupt(session_id: String) -> Result<bool, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    Ok(api::interrupt(sid))
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_save_notebook(session_id: String, path: String, include_outputs: Option<bool>, pretty: Option<bool>) -> Result<bool, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    api::save_notebook(sid, &path, include_outputs.unwrap_or(false), pretty.unwrap_or(true)).map_err(|e| e.to_string())
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_update_session_notebook(session_id: String, notebook: nbcore::schema::Notebook) -> Result<bool, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    Ok(api::update_session_notebook(sid, notebook))
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_preview_value(session_id: String, value: String, limit: Option<u32>) -> Result<String, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    let lim = limit.unwrap_or(100);
    api::preview_value(sid, value, lim).map_err(|e| e.to_string())
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_execute_cell_stream(app: tauri::AppHandle, session_id: String, cell_id: JsonValue) -> Result<bool, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    let cid = parse_uuid_arg(cell_id)?;
    {
        let mut map = crate::api::SESSION_REG.lock();
        if let Some(sess) = map.get_mut(&sid) {
            let sid_s = sid.to_string();
            let _res = sess.execute_cell_with_cb(cid, kernel::ExecutionOpts::default(), |ev| {
                let payload = json!({ "sessionId": sid_s, "event": ev });
                let _ = app.emit("lyra://exec", payload.clone());
            });
            return Ok(true);
        }
    }
    Err("Unknown session".into())
}

fn parse_uuid_arg(val: JsonValue) -> Result<Uuid, String> {
    match val {
        JsonValue::String(s) => Uuid::try_parse(&s).map_err(|e| e.to_string()),
        JsonValue::Array(arr) => {
            // Expect 16 numbers (0..255)
            let bytes: Vec<u8> = arr.into_iter().filter_map(|v| v.as_u64().map(|n| n as u8)).collect();
            if bytes.len() == 16 {
                Ok(Uuid::from_bytes(bytes.try_into().unwrap()))
            } else {
                Err("invalid uuid bytes".into())
            }
        }
        other => Err(format!("invalid uuid arg: {}", other)),
    }
}

// --- Editor commands ---

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_editor_builtins() -> Result<Vec<String>, String> {
    Ok(api::editor_builtins())
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_editor_diagnostics(text: String) -> Result<Vec<api::EditorDiagnostic>, String> {
    Ok(api::editor_diagnostics(text))
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_editor_doc(name: String) -> Result<Option<api::EditorDoc>, String> {
    Ok(api::editor_doc(name))
}

// LSP basics: defs/refs/rename
#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_editor_defs(session_id: String, name: String) -> Result<Vec<api::EditorLocation>, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    Ok(api::editor_defs(sid, &name))
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_editor_refs(session_id: String, name: String) -> Result<Vec<api::EditorLocation>, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    Ok(api::editor_refs(sid, &name))
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_editor_rename(session_id: String, old: String, new_name: String) -> Result<nbcore::schema::Notebook, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    api::editor_rename(sid, &old, &new_name).map_err(|e| e.to_string())
}

// --- Health ---

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_ping(session_id: String) -> Result<bool, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    Ok(api::ping(sid))
}

// Cache UX
#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_cache_set_enabled(session_id: String, enabled: bool) -> Result<bool, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    Ok(api::set_cache_enabled(sid, enabled))
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_cache_clear(session_id: String) -> Result<bool, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    Ok(api::cache_clear(sid))
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_cache_gc(session_id: String, max_bytes: u64) -> Result<api::CacheGcResult, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    Ok(api::cache_gc(sid, max_bytes))
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_cache_info(session_id: String) -> Result<api::CacheInfo, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    Ok(api::cache_info(sid))
}

// Cache salt
#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_cache_set_salt(session_id: String, salt: Option<String>) -> Result<bool, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    Ok(api::cache_set_salt(sid, salt))
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_cache_get_salt(session_id: String) -> Result<Option<String>, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    Ok(api::cache_get_salt(sid))
}

// Language info + editor context
#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_lang_operators() -> Result<Vec<api::LangOperator>, String> {
    Ok(api::lang_operators())
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_editor_context(text: String, offset: u32) -> Result<api::EditorContext, String> {
    Ok(api::editor_context(text, offset))
}

// --- Data Table commands ---

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_table_open(session_id: String, value: String) -> Result<String, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    api::table_open(sid, value).map_err(|e| e.to_string())
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_table_close(session_id: String, handle: String) -> Result<bool, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    api::table_close(sid, handle).map_err(|e| e.to_string())
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_table_schema(session_id: String, handle: String, timeout_ms: Option<u32>) -> Result<api::TableSchema, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    api::table_schema(sid, handle, timeout_ms.map(|v| v as u64)).map_err(|e| e.to_string())
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_table_query(session_id: String, handle: String, query: api::TableQuery, timeout_ms: Option<u32>) -> Result<api::TableQueryResp, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    api::table_query(sid, handle, query, timeout_ms.map(|v| v as u64)).map_err(|e| e.to_string())
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_table_stats(session_id: String, handle: String, columns: Option<Vec<String>>, query: Option<api::TableQuery>, timeout_ms: Option<u32>) -> Result<serde_json::Value, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    api::table_stats(sid, handle, columns, query, timeout_ms.map(|v| v as u64)).map_err(|e| e.to_string())
}

#[cfg_attr(feature = "tauri_cmd", tauri::command)]
pub fn cmd_table_export(session_id: String, handle: String, format: String, query: Option<api::TableQuery>, columns: Option<Vec<String>>) -> Result<String, String> {
    let sid = Uuid::try_parse(&session_id).map_err(|e| e.to_string())?;
    api::table_export(sid, handle, format, query, columns).map_err(|e| e.to_string())
}
