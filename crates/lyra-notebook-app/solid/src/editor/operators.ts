import type { Extension } from '@codemirror/state';
import { EditorView } from '@codemirror/view';

// Basic operator snippets/scaffolds
export function lyraOperatorScaffolds(): Extension {
  return EditorView.inputHandler.of((view, from, to, text) => {
    // Detect insertion of '/.' to create a block scaffold "{  ->  }"
    if (text === '.' && from >= 1) {
      const prev = view.state.sliceDoc(from - 1, from);
      if (prev === '/') {
        // Replace the "/." with " /. {  ->  } " placing cursor before arrow
        const insert = ' /. {  ->  } ';
        view.dispatch({
          changes: [{ from: from - 1, to, insert }],
          selection: { anchor: from + 6 } // position before arrow
        });
        return true;
      }
    }
    // Neutral scaffolds for '//' and '/@' – just allow typing; completion will open via our source
    return false;
  });
}

