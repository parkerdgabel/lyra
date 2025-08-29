import { For, Show, createEffect, createResource, createSignal } from 'solid-js';
import { previewValue } from '../services/tauri';

export function TablePreview(props: { sessionId: string; envelope: string; limit?: number }) {
  const [rows, setRows] = createSignal<any[] | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const limit = () => props.limit ?? 50;

  async function load() {
    try {
      setError(null);
      const json = await previewValue(props.sessionId, props.envelope, limit());
      const parsed = JSON.parse(json);
      // Expect a list of assoc (objects)
      if (Array.isArray(parsed)) setRows(parsed);
      else setRows([]);
    } catch (e: any) {
      setRows(null);
      setError(String(e?.message || e));
    }
  }

  createEffect(() => { load(); });

  const columns = () => {
    const r = rows(); if (!r || r.length === 0) return [] as string[];
    const cols = new Set<string>();
    for (const row of r) { if (row && typeof row === 'object') for (const k of Object.keys(row)) cols.add(k); }
    return Array.from(cols);
  };

  return (
    <div class="output-block" style={{ 'background-color': 'var(--panel)' }}>
      <div class="mime">Preview rows (first {limit()})</div>
      <Show when={!error()} fallback={<div class="badge bad">{error()}</div>}>
        <Show when={rows() !== null} fallback={<div class="badge">Loading…</div>}>
          <div style={{ overflow: 'auto', maxHeight: '300px', border: '1px solid #1a2030' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <For each={columns()}>{(c) => <th style={{ position: 'sticky', top: 0, background: '#141a27', textAlign: 'left', padding: '4px 6px', borderBottom: '1px solid #1a2030' }}>{c}</th>}</For>
                </tr>
              </thead>
              <tbody>
                <For each={rows() || []}>
                  {(row) => (
                    <tr>
                      <For each={columns()}>
                        {(c) => <td style={{ padding: '4px 6px', borderBottom: '1px solid #1a2030' }}>{formatCell(row?.[c])}</td>}
                      </For>
                    </tr>
                  )}
                </For>
              </tbody>
            </table>
          </div>
        </Show>
      </Show>
    </div>
  );
}

function formatCell(v: any): string {
  if (v == null) return '';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

