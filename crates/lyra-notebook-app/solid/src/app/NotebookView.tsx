import { For, Setter, createSignal, onMount } from 'solid-js';
import type { Notebook, Cell } from '../state/types';
import { CodeCell } from '../cells/CodeCell';
import { OutputsView } from '../outputs/OutputsView';
import { addCell, executeCell, executeCellEvents, updateSessionNotebook } from '../services/tauri';
import { Outline } from '../panels/Outline';
import { DiagnosticsPanel } from '../panels/Diagnostics';
import { ValuesPanel } from '../panels/Values';
import type { ValueEntry } from '../panels/Values';
import type { EditorView } from '@codemirror/view';
import { editorDiagnostics, type EditorDiagnostic } from '../services/tauri';

export function NotebookView(props: { notebook: Notebook; sessionId: string; onChange: Setter<Notebook | null> }) {
  const [nb, setNb] = createSignal(props.notebook);
  const [showOutline, setShowOutline] = createSignal(true);
  const [showDiagnostics, setShowDiagnostics] = createSignal(false);
  const [diagMap, setDiagMap] = createSignal<Record<string, EditorDiagnostic[]>>({});
  const [showValues, setShowValues] = createSignal(false);
  const [values, setValues] = createSignal<ValueEntry[]>([]);
  const diagTimers: Record<string, number> = {};
  const editors = new Map<string, EditorView>();

  onMount(() => {
    // Kick initial diagnostics for existing code cells
    for (const c of nb().cells) {
      if (c.type === 'Code' && c.language.toLowerCase() === 'lyra') scheduleDiagnostics(c.id, c.input);
    }
  });

  async function pushCodeCell() {
    const updated = await addCell(props.sessionId, 'Code');
    setNb(updated);
    props.onChange(updated);
  }

  async function runCell(cell: Cell) {
    // Use events to simulate streaming updates
    const events = await executeCellEvents(props.sessionId, cell.id);
    let draft: Notebook = { ...nb() };
    let outputs = [] as Cell['output'];
    let timing = 0; let error: string | undefined = undefined;
    for (const ev of events) {
      if ('Started' in ev) {
        outputs = [];
        draft = { ...draft, cells: draft.cells.map(c => c.id === cell.id ? { ...c, output: [], meta: { ...c.meta, error: undefined } } : c) };
        setNb(draft); props.onChange(draft);
      } else if ('Output' in ev) {
        outputs = outputs.concat(ev.Output.item);
        draft = { ...draft, cells: draft.cells.map(c => c.id === cell.id ? { ...c, output: outputs } : c) };
        setNb(draft); props.onChange(draft);
      } else if ('Error' in ev) {
        error = ev.Error.message;
      } else if ('Finished' in ev) {
        timing = Number(ev.Finished.result.duration_ms) || 0;
        error = ev.Finished.result.error ?? error;
        outputs = ev.Finished.result.outputs?.length ? ev.Finished.result.outputs : outputs;
      }
    }
    const final: Notebook = { ...draft, cells: draft.cells.map(c => c.id === cell.id ? { ...c, output: outputs, meta: { ...c.meta, timingMs: timing, error } } : c) };
    setNb(final);
    await updateSessionNotebook(props.sessionId, final);
    props.onChange(final);
    updateLatestValue(cell.id, outputs);
  }

  async function updateCellInput(cellId: string, input: string) {
    const next: Notebook = { ...nb(), cells: nb().cells.map(c => c.id === cellId ? { ...c, input } : c) };
    setNb(next);
    props.onChange(next);
    // Persist into session so the kernel sees updated input
    try { await updateSessionNotebook(props.sessionId, next); } catch {}
    // Recompute diagnostics (debounced)
    scheduleDiagnostics(cellId, input);
  }

  function scheduleDiagnostics(cellId: string, text: string) {
    if (diagTimers[cellId]) window.clearTimeout(diagTimers[cellId]);
    diagTimers[cellId] = window.setTimeout(async () => {
      try {
        const diags = await editorDiagnostics(text);
        setDiagMap((prev) => ({ ...prev, [cellId]: diags }));
      } catch {}
    }, 300);
  }

  function jumpTo(cellId: string, line = 0, col = 0) {
    const v = editors.get(cellId);
    if (v) {
      const docLine = v.state.doc.line(line + 1);
      const pos = Math.min(docLine.from + col, docLine.to);
      v.dispatch({ selection: { anchor: pos }, scrollIntoView: true });
      v.focus();
    } else {
      // Fallback: scroll to cell block
      const el = document.querySelector(`[data-cell-id="${cellId}"]`) as HTMLElement | null;
      el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  function updateLatestValue(cellId: string, outputs: Cell['output']) {
    const last = [...outputs].reverse().find((o) => o.mime === 'application/lyra+value');
    if (!last) return;
    const entry: ValueEntry = { cellId, envelope: last.data, at: Date.now() };
    setValues((prev) => {
      const filtered = prev.filter((e) => e.cellId !== cellId);
      return [entry, ...filtered].slice(0, 20);
    });
  }

  return (
    <div class="nb-shell">
      <div>
        <button class="badge" onClick={pushCodeCell}>+ Code Cell</button>
        <button class="badge" style={{ 'margin-left': '8px' }} onClick={() => setShowOutline(!showOutline())}>Outline</button>
        <button class="badge" style={{ 'margin-left': '8px' }} onClick={() => setShowDiagnostics(!showDiagnostics())}>Diagnostics</button>
        <button class="badge" style={{ 'margin-left': '8px' }} onClick={() => setShowValues(!showValues())}>Values</button>
      </div>
      <For each={nb().cells}>
        {(cell) => (
          <div class="cell" data-cell-id={cell.id}>
            <div class="cell-head">
              <div class="title">{cell.type} · {cell.language}</div>
              {cell.meta?.timingMs !== undefined && <div class="badge">{cell.meta.timingMs} ms</div>}
              {cell.meta?.cached && <div class="badge good">cached</div>}
              {cell.meta?.error && <div class="badge bad">error</div>}
            </div>
            <div class="cell-body">
              {cell.type === 'Code' ? (
                <div class="code"><CodeCell cell={cell} onRun={() => runCell(cell)} onChange={(t) => updateCellInput(cell.id, t)} onReady={(v) => { if (v) editors.set(cell.id, v); else editors.delete(cell.id); }} /></div>
              ) : (
                <div class="code"/>
              )}
              <div class="outputs"><OutputsView outputs={cell.output} sessionId={props.sessionId} /></div>
            </div>
          </div>
        )}
      </For>
      {showOutline() && (
        <div class="cell" style={{ position: 'fixed', right: '12px', top: '12px', width: '280px', maxHeight: '80%', overflow: 'auto' }}>
          <div class="cell-head"><div class="title">Outline</div></div>
          <Outline notebook={nb()} onJump={(id, ln, col) => jumpTo(id, ln, col)} />
        </div>
      )}
      {showDiagnostics() && (
        <div class="cell" style={{ position: 'fixed', right: '12px', top: '240px', width: '360px', maxHeight: '60%', overflow: 'auto' }}>
          <div class="cell-head"><div class="title">Diagnostics</div></div>
          <DiagnosticsPanel notebook={nb()} diagMap={diagMap()} onJump={(id, ln, col) => jumpTo(id, ln, col)} />
        </div>
      )}
      {showValues() && (
        <div class="cell" style={{ position: 'fixed', right: '12px', top: '520px', width: '420px', maxHeight: '40%', overflow: 'auto' }}>
          <div class="cell-head"><div class="title">Values</div></div>
          <ValuesPanel sessionId={props.sessionId} values={values()} />
        </div>
      )}
    </div>
  );
}
