import type { Extension } from '@codemirror/state';
import { linter, Diagnostic, lintGutter } from '@codemirror/lint';
import { editorDiagnostics } from '../services/tauri';

export function lyraDiagnostics(): Extension {
  let timer: number | null = null;
  let lastText = '';
  const run = async (text: string): Promise<Diagnostic[]> => {
    try {
      const list = await editorDiagnostics(text);
      const out: Diagnostic[] = [];
      const lines = text.split(/\n/);
      const toPos = (ln: number, col: number) => {
        const lineIdx = Math.max(0, Math.min(ln, lines.length - 1));
        let off = 0;
        for (let i = 0; i < lineIdx; i++) off += lines[i].length + 1;
        return off + col;
      };
      for (const d of list) {
        out.push({
          from: toPos(d.start_line, d.start_col),
          to: toPos(d.end_line, d.end_col),
          severity: d.severity.toLowerCase() as any,
          message: d.message,
          source: 'Lyra'
        });
      }
      return out;
    } catch {
      return [];
    }
  };

  const debounced = linter((view) => {
    const text = view.state.doc.toString();
    return new Promise<Diagnostic[]>((resolve) => {
      if (timer) window.clearTimeout(timer);
      if (text === lastText) { resolve([]); return; }
      lastText = text;
      timer = window.setTimeout(async () => {
        const diags = await run(text);
        resolve(diags);
      }, 250);
    });
  });

  return [debounced, lintGutter()];
}

