import { For, Show } from 'solid-js';
import type { DisplayData } from '../state/types';
import { LyraValueTree } from './LyraValueTree';

export function OutputsView(props: { outputs: DisplayData[]; sessionId?: string }) {
  return (
    <div>
      <For each={props.outputs}>
        {(it) => (
          <div class="output-block">
            <div class="mime">{it.mime}</div>
            <Show when={it.mime.startsWith('text/plain')}>
              <div>{it.data}</div>
            </Show>
            <Show when={it.mime === 'application/lyra+value'}>
              <LyraValueTree data={it.data} sessionId={props.sessionId} />
            </Show>
            <Show when={it.mime.startsWith('image/') || it.mime === 'image/svg+xml'}>
              <img src={it.data.startsWith('data:') ? it.data : `data:${it.mime},${it.data}`} alt={it.mime} />
            </Show>
            <Show when={it.mime === 'text/html'}>
              <iframe src={it.data} style={{width:'100%',height:'300px',border:'0'}} />
            </Show>
          </div>
        )}
      </For>
    </div>
  );
}

// moved to LyraValueTree
