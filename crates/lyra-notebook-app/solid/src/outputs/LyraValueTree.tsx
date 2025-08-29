import { For, Show, createSignal } from 'solid-js';

type Node = any;

export function LyraValueTree(props: { data: string; sessionId?: string }) {
  // Parse envelope and extract value
  let env: any = null;
  try { env = JSON.parse(props.data); } catch {}
  const value: Node = env && typeof env === 'object' && 'value' in env ? env.value : env ?? props.data;
  const [showRaw, setShowRaw] = createSignal(false);
  const type = () => (value && typeof value === 'object' && (value as any).__type) || null;
  const isTabular = () => type() === 'Dataset' || type() === 'Frame';
  return (
    <div>
      <div style={{ display: 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '6px' }}>
        <div class="mime">application/lyra+value</div>
        <label class="badge" style={{ cursor: 'pointer' }}>
          <input type="checkbox" checked={showRaw()} onInput={(e) => setShowRaw((e.target as HTMLInputElement).checked)} /> raw
        </label>
      </div>
      <Show when={!showRaw()} fallback={<pre style={{ margin: 0 }}>{props.data}</pre>}>
        <Show when={isTabular() && props.sessionId}>
          <TablePreview sessionId={props.sessionId!} envelope={props.data} />
        </Show>
        <Tree node={value} depth={0} />
      </Show>
    </div>
  );
}

function Tree(props: { node: Node; depth: number }) {
  const pad = (props.depth * 12) + 'px';
  const isObj = (v: any) => v && typeof v === 'object' && !Array.isArray(v);
  const isArr = Array.isArray(props.node);
  const isAssoc = isObj(props.node) && '__type' in props.node ? props.node.__type : null;
  const [open, setOpen] = createSignal(true);
  const entries = () => isArr ? props.node.map((v: any, i: number) => [String(i), v]) : isObj(props.node) ? Object.entries(props.node) : [];
  if (!isObj(props.node) && !isArr) {
    return <div style={{ 'padding-left': pad }}><Leaf value={props.node} /></div>;
  }
  return (
    <div>
      <div style={{ 'padding-left': pad }}>
        <span style={{ cursor: 'pointer', color: 'var(--accent)' }} onClick={() => setOpen(!open())}>{open() ? '▾' : '▸'}</span>
        <span style={{ opacity: .8, 'margin-left': '6px' }}>{isArr ? `[${props.node.length}]` : isAssoc ? `{${isAssoc}}` : '{ }'}</span>
      </div>
      <Show when={open()}>
        <For each={entries()}>
          {([k, v]) => (
            <div>
              <div style={{ 'padding-left': `calc(${pad} + 12px)` }}>
                <span style={{ color: '#aab1c4' }}>{k}</span>
              </div>
              <Tree node={v} depth={props.depth + 2} />
            </div>
          )}
        </For>
      </Show>
    </div>
  );
}

function Leaf(props: { value: any }) {
  const v = props.value;
  if (typeof v === 'string') return <span style={{ color: '#b4d3ff' }}>&quot;{v}&quot;</span>;
  if (typeof v === 'number') return <span style={{ color: '#ffd29a' }}>{String(v)}</span>;
  if (typeof v === 'boolean') return <span style={{ color: '#9fe7c0' }}>{String(v)}</span>;
  if (v === null) return <span style={{ color: '#aab1c4' }}>null</span>;
  return <span style={{ color: '#e6e8ee' }}>{String(v)}</span>;
}

import { TablePreview } from './TablePreview';
