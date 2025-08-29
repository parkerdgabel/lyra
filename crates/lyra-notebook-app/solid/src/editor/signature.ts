import type { Extension } from '@codemirror/state';
import { showTooltip, Tooltip } from '@codemirror/view';
import { editorDoc } from '../services/tauri';

export function lyraSignature(): Extension {
  return showTooltip.compute([showTooltip.baseTheme], [], (state) => [])
}

// A minimal transactional signature helper that triggers when '[' is typed.
export function lyraSignatureOnBracket(): Extension {
  return EditorBracketSignature as unknown as Extension;
}

const EditorBracketSignature = {
  provide: (f: any) => f,
  // define an extension via plugin (works in minimal form without full view plugin types)
  extension: {
    update(update: any) {
      if (!update.docChanged) return;
      const tr = update.transactions[0];
      const inserted = tr && tr.changes && tr.changes.iter ? collectInserted(tr) : '';
      if (inserted !== '[') return;
      const pos = update.state.selection.main.head;
      const line = update.state.doc.lineAt(pos);
      const before = update.state.sliceDoc(line.from, pos);
      const m = /([A-Za-z_][A-Za-z0-9_]*)\s*$/m.exec(before);
      if (!m) return;
      const name = m[1];
      // fetch docs
      editorDoc(name).then((doc) => {
        if (!doc) return;
        const content = document.createElement('div');
        content.style.maxWidth = '420px';
        content.style.padding = '8px 10px';
        content.style.background = 'var(--panel)';
        content.style.border = '1px solid #1a2030';
        content.style.borderRadius = '6px';
        content.innerHTML = `<div style="font-weight:600;margin-bottom:4px">${doc.name}</div>
          <div style="font-family:var(--mono);font-size:12px;color:#aab1c4">[${(doc.params||[]).join(', ')}]</div>`;
        const tooltip: Tooltip = { pos, create: () => ({ dom: content }) };
        update.view.dispatch({ effects: showTooltip.of(tooltip) });
        // auto-hide after 2.5s
        setTimeout(() => update.view.dispatch({ effects: showTooltip.of(null) }), 2500);
      });
    }
  }
} as any;

function collectInserted(tr: any): string {
  let out = '';
  tr.changes.iterChanges((_f: number, _t: number, _f2: number, _t2: number, ins: any) => {
    out += ins.toString();
  });
  return out;
}

