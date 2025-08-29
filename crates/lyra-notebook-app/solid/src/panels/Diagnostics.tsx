import { For, Show, createMemo } from 'solid-js';
import type { Notebook } from '../state/types';
import type { EditorDiagnostic } from '../services/tauri';

export function DiagnosticsPanel(props: { notebook: Notebook; diagMap: Record<string, EditorDiagnostic[]>; onJump?: (cellId: string, line: number, col: number) => void }) {
  const rows = createMemo(() => {
    const out: { cellId: string; diag: EditorDiagnostic }[] = [];
    for (const c of props.notebook.cells) {
      if (c.type !== 'Code' || c.language.toLowerCase() !== 'lyra') continue;
      const diags = props.diagMap[c.id] || [];
      for (const d of diags) out.push({ cellId: c.id, diag: d });
    }
    return out;
  });
  const color = (sev: string) => sev === 'Error' ? 'var(--bad)' : sev === 'Warning' ? 'var(--warn)' : 'var(--subtle)';
  return (
    <div style={{ padding: '8px 10px' }}>
      <Show when={rows().length > 0} fallback={<div class="badge">No issues</div>}>
        <For each={rows()}>
          {(r) => (
            <div style={{ padding: '6px 0', borderBottom: '1px solid #1a2030', cursor: 'pointer' }} onClick={() => props.onJump?.(r.cellId, r.diag.start_line, r.diag.start_col)}>
              <div style={{ display: 'flex', 'align-items': 'center', gap: '8px' }}>
                <span class="badge" style={{ background: 'transparent', color: color(r.diag.severity) }}>{r.diag.severity}</span>
                <span style={{ color: '#aab1c4' }}>Cell {r.cellId.slice(0, 8)}</span>
                <span style={{ color: '#aab1c4' }}>L{r.diag.start_line + 1}:{r.diag.start_col + 1}</span>
              </div>
              <div>{r.diag.message}</div>
            </div>
          )}
        </For>
      </Show>
    </div>
  );
}

