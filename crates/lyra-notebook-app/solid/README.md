Lyra Notebook (SolidJS + CodeMirror) — Prototype

This is a fresh Solid + Vite frontend targeting the existing Tauri app.

Status
- Minimal scaffold: Solid app shell, TS schema mirrors, Tauri `invoke` client, basic CodeMirror editor, simple output renderer.
- It does not replace the current `ui/` yet. Build artifacts go to `../ui-solid-dist/`.

Develop
1. Install deps: `npm i` (or `pnpm i`) in this directory.
2. Dev: `npm run dev` and point Tauri dev server to `http://localhost:5173` if desired.
3. Build: `npm run build` to produce `ui-solid-dist/`. You can change `tauri.conf.json` to use this path as `distDir` when ready.

Next
- Implement Lyra CM6 language extension (tokenizer, pairs, indent, completion, hover, diagnostics).
- Wire state updates from editor to notebook store, and persist via `cmd_update_session_notebook`.
- Add panels (Outline, Diagnostics, Cache) and richer output renderers (tables/frames).

