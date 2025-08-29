import { For, Show } from 'solid-js';
import type { Assoc } from '../state/types';
import { LyraValueTree } from '../outputs/LyraValueTree';

export type ValueEntry = { cellId: string; envelope: string; at: number };

export function ValuesPanel(props: { sessionId: string; values: ValueEntry[] }) {
  return (
    <div style={{ padding: '8px 10px' }}>
      <Show when={props.values.length > 0} fallback={<div class="badge">No values yet</div>}>
        <For each={props.values}>
          {(v) => (
            <div style={{ margin: '6px 0', border: '1px solid #1a2030', borderRadius: '6px' }}>
              <div class="cell-head" style={{ borderBottom: '1px solid #1a2030' }}>
                <div class="title">Cell {v.cellId.slice(0, 8)}</div>
                <div class="badge">{new Date(v.at).toLocaleTimeString()}</div>
              </div>
              <div style={{ padding: '6px 8px' }}>
                <LyraValueTree data={v.envelope} sessionId={props.sessionId} />
              </div>
            </div>
          )}
        </For>
      </Show>
    </div>
  );
}

