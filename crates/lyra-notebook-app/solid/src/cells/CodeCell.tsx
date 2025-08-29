import { createEffect, onCleanup, onMount } from 'solid-js';
import type { Cell } from '../state/types';
// CodeMirror 6 minimal inline setup; full Lyra language will be wired later.
import { EditorView, keymap, placeholder } from '@codemirror/view';
import { EditorState, Compartment } from '@codemirror/state';
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import { bracketMatching, indentOnInput } from '@codemirror/language';
import { closeBrackets } from '@codemirror/autocomplete';
import { lyraLanguage } from '../editor/lyra-language';
import { lyraCompletion } from '../editor/completion';
import { lyraHover } from '../editor/hover';
import { lyraDiagnostics } from '../editor/diagnostics';
import { lyraSignatureOnBracket } from '../editor/signature';
import { lyraOperatorScaffolds } from '../editor/operators';

const langConf = new Compartment();
const readOnlyConf = new Compartment();

export function CodeCell(props: { cell: Cell; onRun: () => void; onChange?: (text: string) => void; onReady?: (view: EditorView | null) => void }) {
  let host!: HTMLDivElement;
  let view: EditorView | null = null;

  onMount(() => {
    const state = EditorState.create({
      doc: props.cell.input,
      extensions: [
        history(),
        keymap.of([
          ...defaultKeymap,
          ...historyKeymap,
          indentWithTab,
          {
            key: 'Mod-Enter',
            preventDefault: true,
            run: () => { props.onRun(); return true; }
          }
        ]),
        placeholder('Write Lyra here…'),
        closeBrackets(),
        bracketMatching(),
        indentOnInput(/^\s*[\]\)\}]/),
        lyraCompletion(),
        lyraHover(),
        lyraDiagnostics(),
        lyraSignatureOnBracket(),
        lyraOperatorScaffolds(),
        EditorView.updateListener.of((u) => {
          if (u.docChanged) {
            const text = u.state.doc.toString();
            props.onChange?.(text);
          }
        }),
        langConf.of([lyraLanguage()]),
        readOnlyConf.of(EditorState.readOnly.of(false)),
        EditorView.theme({
          '&': { height: '100%' },
          '.cm-content': { fontFamily: 'var(--mono)', fontSize: '14px', color: 'var(--code)' },
          '.cm-scroller': { fontFamily: 'var(--mono)' },
          '.cm-gutters': { background: 'var(--panel)', borderRight: '1px solid #1a2030' },
        }, {dark:true})
      ]
    });
    view = new EditorView({ state, parent: host });
    props.onReady?.(view);
  });

  createEffect(() => {
    // If external input changes, sync editor (simple replace for now)
    if (view && props.cell.input !== view.state.doc.toString()) {
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: props.cell.input } });
    }
  });

  onCleanup(() => { props.onReady?.(null); view?.destroy(); view = null; });

  return <div ref={host} style={{ height: '100%' }} />;
}
