import type { Extension } from '@codemirror/state';
import { hoverTooltip } from '@codemirror/view';
import { editorDoc } from '../services/tauri';

export function lyraHover(): Extension {
  return hoverTooltip(async (view, pos) => {
    // Find word at position
    const line = view.state.doc.lineAt(pos);
    const rel = pos - line.from;
    const text = line.text;
    const m = /[A-Za-z_][A-Za-z0-9_]*/g;
    let found: { from: number; to: number; word: string } | null = null;
    for (;;) {
      const r = m.exec(text);
      if (!r) break;
      const s = r.index, e = s + r[0].length;
      if (s <= rel && rel <= e) { found = { from: line.from + s, to: line.from + e, word: r[0] }; break; }
    }
    if (!found) return null;
    const doc = await editorDoc(found.word);
    if (!doc) return null;
    const el = document.createElement('div');
    el.style.maxWidth = '420px';
    el.style.padding = '8px 10px';
    el.style.background = 'var(--panel)';
    el.style.border = '1px solid #1a2030';
    el.style.borderRadius = '6px';
    el.innerHTML = `<div style="font-weight:600;margin-bottom:4px">${doc.name}</div>
      <div style="opacity:.85;margin-bottom:6px">${doc.summary || ''}</div>
      ${doc.params?.length ? `<div style="font-family:var(--mono);font-size:12px;color:#aab1c4">[${doc.params.join(', ')}]</div>` : ''}`;
    return { pos: found.from, end: found.to, create() { return { dom: el }; } };
  }, { hoverTime: 200 });
}

