#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use lyra_notebook_gui::tauri_cmd::*;

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            cmd_new_notebook,
            cmd_open_notebook,
            cmd_execute_cell,
            cmd_execute_cell_nocache,
            cmd_execute_cell_events,
            cmd_execute_cell_stream,
            cmd_execute_text,
            cmd_execute_all,
            cmd_interrupt,
            cmd_save_notebook,
            cmd_update_session_notebook,
            cmd_preview_value,
            cmd_table_open,
            cmd_table_close,
            cmd_table_schema,
            cmd_table_query,
            cmd_table_stats,
            cmd_table_export,
            cmd_add_cell,
            cmd_delete_cell,
            cmd_editor_builtins,
            cmd_editor_diagnostics,
            cmd_editor_doc,
            cmd_cache_info,
            cmd_editor_defs,
            cmd_editor_refs,
            cmd_editor_rename,
            cmd_ping,
            cmd_cache_set_enabled,
            cmd_cache_clear,
            cmd_cache_gc,
            cmd_cache_set_salt,
            cmd_cache_get_salt,
            cmd_lang_operators,
            cmd_editor_context,
        ])
        .run(tauri::generate_context!())
        .expect("error while running lyra-notebook-app");
}
