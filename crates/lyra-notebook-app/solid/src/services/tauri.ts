import { invoke } from '@tauri-apps/api/core';
import type { Notebook, ExecResult, ExecEvent } from '../state/types';

export interface OpenResponse { session_id: string; notebook: Notebook }

export async function openNotebook(path: string): Promise<OpenResponse> {
  return invoke('cmd_open_notebook', { path });
}

export async function newNotebook(title?: string): Promise<OpenResponse> {
  return invoke('cmd_new_notebook', { title });
}

export async function saveNotebook(sessionId: string, path: string, includeOutputs = true, pretty = true): Promise<boolean> {
  return invoke('cmd_save_notebook', { sessionId, path, includeOutputs, pretty });
}

export async function updateSessionNotebook(sessionId: string, notebook: Notebook): Promise<boolean> {
  return invoke('cmd_update_session_notebook', { sessionId, notebook }) as unknown as boolean;
}

export async function executeCell(sessionId: string, cellId: string): Promise<ExecResult> {
  return invoke('cmd_execute_cell', { sessionId, cellId });
}

export async function executeCellEvents(sessionId: string, cellId: string): Promise<ExecEvent[]> {
  return invoke('cmd_execute_cell_events', { sessionId, cellId });
}

export async function addCell(sessionId: string, cellType: 'Code' | 'Markdown' | 'Text'): Promise<Notebook> {
  return invoke('cmd_add_cell', { sessionId, cellType });
}

export async function deleteCell(sessionId: string, cellId: string): Promise<Notebook> {
  return invoke('cmd_delete_cell', { sessionId, cellId });
}

// Editor helpers — stubs; wire up as needed in CM6 extensions
export async function editorBuiltins(): Promise<string[]> {
  return invoke('cmd_editor_builtins');
}

export interface EditorDoc { name: string; summary: string; params: string[]; examples: string[] }
export async function editorDoc(name: string): Promise<EditorDoc | null> {
  return invoke('cmd_editor_doc', { name });
}

export interface EditorDiagnostic { message: string; start_line: number; start_col: number; end_line: number; end_col: number; severity: 'Error'|'Warning'|'Info' }
export async function editorDiagnostics(text: string): Promise<EditorDiagnostic[]> {
  return invoke('cmd_editor_diagnostics', { text });
}

// Preview typed Lyra values (Datasets/Frames) as row lists via kernel helper
export async function previewValue(sessionId: string, valueJson: string, limit = 50): Promise<string> {
  return invoke('cmd_preview_value', { sessionId, value: valueJson, limit });
}
