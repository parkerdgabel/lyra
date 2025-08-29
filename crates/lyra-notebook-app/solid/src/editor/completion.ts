import type { Extension } from '@codemirror/state';
import { autocompletion, Completion, CompletionContext } from '@codemirror/autocomplete';
import { editorBuiltins } from '../services/tauri';

let builtinsCache: string[] | null = null;
async function ensureBuiltins(): Promise<string[]> {
  if (builtinsCache) return builtinsCache;
  try { builtinsCache = await editorBuiltins(); } catch { builtinsCache = []; }
  return builtinsCache!;
}

function lyraCompletionSource() {
  return async (ctx: CompletionContext) => {
    const word = ctx.matchBefore(/[A-Za-z_][A-Za-z0-9_]*/);
    const pos = ctx.pos;
    const prev = ctx.state.sliceDoc(Math.max(0, pos - 2), pos);
    const triggerOps = ['//', '/@', '/.'];
    const shouldOpen = ctx.explicit || !!word || triggerOps.includes(prev);
    if (!shouldOpen) return null;
    const builtins = await ensureBuiltins();
    const from = word ? word.from : pos;
    const options: Completion[] = builtins.map((name) => ({ label: name, type: 'function' }));
    return { from, options, validFor: /[A-Za-z0-9_]+/ };
  };
}

export function lyraCompletion(): Extension {
  return autocompletion({ override: [lyraCompletionSource()], defaultKeymap: true, activateOnTyping: true });
}

