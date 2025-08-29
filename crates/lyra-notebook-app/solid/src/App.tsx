import { createSignal, onMount } from 'solid-js';
import { NotebookView } from './app/NotebookView';
import type { Notebook } from './state/types';
import { openNotebook, newNotebook } from './services/tauri';

export function App() {
  const [notebook, setNotebook] = createSignal<Notebook | null>(null);
  const [sessionId, setSessionId] = createSignal<string | null>(null);

  onMount(async () => {
    // For now open a new in-memory notebook to bootstrap UI
    try {
      const res = await newNotebook('Untitled');
      setNotebook(res.notebook);
      setSessionId(res.session_id);
    } catch (e) {
      console.error('Failed to init session', e);
    }
  });

  return (
    <div class="app-shell">
      {notebook() && sessionId() ? (
        <NotebookView notebook={notebook()!} sessionId={sessionId()!} onChange={setNotebook} />
      ) : (
        <div class="boot">Initializing…</div>
      )}
    </div>
  );
}

