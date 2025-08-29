import { For, createMemo } from 'solid-js';
import type { Notebook } from '../state/types';

export function Outline(props: { notebook: Notebook; onJump?: (cellId: string, line?: number, col?: number) => void }) {
  const entries = createMemo(() => {
    const out: { cellId: string; defs: { name: string; line: number }[] }[] = [];
    for (const c of props.notebook.cells) {
      if (c.type !== 'Code' || c.language.toLowerCase() !== 'lyra') continue;
      const defs: { name: string; line: number }[] = [];
      const lines = c.input.split(/\n/);
      for (let i = 0; i < lines.length; i++) {
        const l = lines[i];
        const m = /\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[.*?\])?\s*:=/.exec(l);
        if (m) defs.push({ name: m[1], line: i });
      }
      if (defs.length) out.push({ cellId: c.id, defs });
    }
    return out;
  });
  return (
    <div style={{ padding: '8px 10px' }}>
      <For each={entries()}>
        {(e) => (
          <div style={{ 'margin-bottom': '8px' }}>
            <div class="badge" style={{ 'margin-bottom': '4px' }}>Cell {e.cellId.slice(0, 8)}</div>
            <For each={e.defs}>
              {(d) => (
                <div style={{ cursor: 'pointer' }} onClick={() => props.onJump?.(e.cellId, d.line, 0)}>
                  {d.name}
                </div>
              )}
            </For>
          </div>
        )}
      </For>
    </div>
  );
}

