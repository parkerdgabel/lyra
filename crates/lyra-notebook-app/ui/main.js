(() => {
  console.log('[Lyra Notebook] Frontend loaded');
  const { invoke } = window.__TAURI__.invoke ? window.__TAURI__ : { invoke: () => Promise.reject(new Error("Tauri not available")) };
  const byId = (id) => document.getElementById(id);
  const KEYWORDS = ['let','fn','type','if','then','else','match','with','for','in','while','return','true','false','null'];
  const bytesToUuid = (arr) => {
    if (!Array.isArray(arr) || arr.length !== 16) return null;
    const hex = arr.map(b => b.toString(16).padStart(2,'0')).join('');
    return [
      hex.slice(0,8),
      hex.slice(8,12),
      hex.slice(12,16),
      hex.slice(16,20),
      hex.slice(20)
    ].join('-');
  };
  const normId = (v) => Array.isArray(v) ? bytesToUuid(v) : v;

  let sessionId = null;
  let notebook = null; // JS object mirroring schema
  const editors = new Map(); // cellId -> editor API
  let activeEditor = null; // { cellId, textarea, focus(), insert(text) }
  let BUILTINS = new Set();
  let runningCount = 0;
  const runQueue = [];
  const problems = new Map(); // cellId -> { message, severity }
  let activeTab = 'Outline';
  const dirtyCells = new Set(); // cid set
  const lastRunHash = new Map(); // cid -> hash
  // Per-cell dependency map
  let cellDeps = new Map(); // cid -> { defines:Set, uses:Set, deps:Set }
  function isIdent(s){ return /^[A-Za-z_][A-Za-z0-9_]*$/.test(s||''); }
  function analyzeCell(text){
    const defines = new Set();
    const usesRaw = new Set();
    try{
      // let definitions
      const reLet = /\blet\s+([A-Za-z_][A-Za-z0-9_]*)/g; let m;
      while((m = reLet.exec(text))){ defines.add(m[1]); }
      // function style calls Name[ ... ] also indicate a symbol usage
      const reCall = /\b([A-Za-z_][A-Za-z0-9_]*)\s*\[/g; while((m = reCall.exec(text))){ usesRaw.add(m[1]); }
      // general identifiers
      const reId = /\b([A-Za-z_][A-Za-z0-9_]*)\b/g; while((m = reId.exec(text))){ const w=m[1]; if (!defines.has(w) && !KEYWORDS.includes(w)) usesRaw.add(w); }
    }catch(_){ }
    return { defines, usesRaw };
  }
  function analyzeNotebookDeps(){
    cellDeps.clear();
    if (!notebook || !Array.isArray(notebook.cells)) return;
    const symbolToCells = new Map();
    // First pass: defs/usesRaw per cell
    const temp = new Map();
    for (const c of notebook.cells){
      const cid = normId(c.id);
      if (c.type === 'Code' && c.language === 'Lyra'){ const { defines, usesRaw } = analyzeCell(c.input||''); temp.set(cid, { defines, usesRaw }); defines.forEach(s => { const arr = symbolToCells.get(s) || []; arr.push(cid); symbolToCells.set(s, arr); }); }
      else { temp.set(cid, { defines: new Set(), usesRaw: new Set() }); }
    }
    // Second pass: resolve uses to only those that match other cells' defines (exclude builtins)
    for (const c of notebook.cells){
      const cid = normId(c.id); const t = temp.get(cid); if (!t) continue;
      const uses = new Set();
      t.usesRaw.forEach(s => { if (symbolToCells.has(s) && !BUILTINS.has(s)) uses.add(s); });
      // deps: all cells that define any used symbol (excluding self)
      const deps = new Set();
      uses.forEach(s => { (symbolToCells.get(s)||[]).forEach(owner => { if (owner !== cid) deps.add(owner); }); });
      cellDeps.set(cid, { defines: t.defines, uses, deps });
    }
  }
  function computeDependentsMap(){
    const rev = new Map();
    if (!notebook || !Array.isArray(notebook.cells)) return rev;
    notebook.cells.forEach(c => { rev.set(normId(c.id), new Set()); });
    cellDeps.forEach((v, cid) => { v.deps.forEach(depCid => { const s = rev.get(depCid) || new Set(); s.add(cid); rev.set(depCid, s); }); });
    return rev;
  }
  function computeImpacted(){
    const rev = computeDependentsMap();
    const impacted = new Set();
    const q = [];
    dirtyCells.forEach(cid => { impacted.add(cid); q.push(cid); });
    while(q.length){ const cur = q.shift(); const depSet = rev.get(cur) || new Set(); depSet.forEach(nxt => { if (!impacted.has(nxt)){ impacted.add(nxt); q.push(nxt); } }); }
    // return as ordered list by document order
    const order = (notebook?.cells||[]).map(c=>normId(c.id));
    return order.filter(cid => impacted.has(cid));
  }
  function hashText(s){ let h=5381; for (let i=0;i<s.length;i++){ h=((h<<5)+h) ^ s.charCodeAt(i); } return (h>>>0); }
  function initHashes(){ if (!notebook) return; (notebook.cells||[]).forEach(c => { if (c.type==='Code' && c.language==='Lyra'){ lastRunHash.set(normId(c.id), hashText(c.input||'')); } }); }
  function updateImpactUI(){ const btn = byId('runImpBtn'); if (!btn) return; const ids = computeImpacted(); btn.textContent = ids.length ? `Run Impacted (${ids.length})` : 'Run Impacted'; btn.disabled = ids.length === 0; }

  // Persist run state (hashes + dirties) across reloads
  function runStateKey(){
    const p = (byId('path') && byId('path').value.trim()) || '';
    const base = p || (sessionId || '');
    return `lyra_runstate:${base}`;
  }
  function saveRunState(){
    try{
      const obj = { hashes: {}, dirty: [] };
      lastRunHash.forEach((v,k)=>{ obj.hashes[k] = v; });
      dirtyCells.forEach(k => obj.dirty.push(k));
      localStorage.setItem(runStateKey(), JSON.stringify(obj));
    }catch(_){/* ignore */}
  }
  function loadRunState(){
    try{
      const raw = localStorage.getItem(runStateKey());
      if (!raw){ initHashes(); return; }
      const obj = JSON.parse(raw);
      lastRunHash.clear();
      if (obj && obj.hashes){ for (const k in obj.hashes){ lastRunHash.set(k, obj.hashes[k]|0); } }
      dirtyCells.clear();
      if (obj && Array.isArray(obj.dirty)){ obj.dirty.forEach(k => dirtyCells.add(k)); }
      updateImpactUI();
    }catch(_){ initHashes(); }
  }
  let docsCache = new Map();
  let currentDocSymbol = null;
  const outputPrefs = new Map(); // cellId -> { expanded: Set(index), views: Map(index -> 'table'|'raw'), table: Map(index -> state), handles: Map(index -> handle) }
  // Settings-backed preview sizes
  function loadPreviewSizes(){
    try {
      const raw = localStorage.getItem('lyra_preview_sizes');
      if (!raw) return [50,200,1000];
      const arr = JSON.parse(raw);
      const nums = Array.isArray(arr) ? arr.map(x => parseInt(x,10)).filter(n => Number.isFinite(n) && n>0) : [];
      return nums.length ? nums.slice(0,8) : [50,200,1000];
    } catch(_) { return [50,200,1000]; }
  }
  function loadPreviewDefault(){
    const sizes = loadPreviewSizes();
    try {
      const raw = localStorage.getItem('lyra_preview_default');
      const n = raw ? parseInt(raw,10) : NaN;
      return (Number.isFinite(n) && sizes.includes(n)) ? n : (sizes[1] || sizes[0] || 200);
    } catch(_) { return sizes[1] || sizes[0] || 200; }
  }
  function savePreviewSettings(sizes, def){
    try { localStorage.setItem('lyra_preview_sizes', JSON.stringify(sizes)); localStorage.setItem('lyra_preview_default', String(def)); } catch(_){}
  }
  // Persist output view prefs across reloads using localStorage
  function loadAllSavedOutputPrefs(){
    try { const raw = localStorage.getItem('lyra_output_prefs'); return raw ? JSON.parse(raw) : {}; } catch(_) { return {}; }
  }
  function saveAllOutputPrefs(obj){ try { localStorage.setItem('lyra_output_prefs', JSON.stringify(obj)); } catch(_) { /* ignore */ } }
  function persistOutputPrefs(cellId){
    const p = outputPrefs.get(cellId); if (!p) return;
    const saved = loadAllSavedOutputPrefs();
    const entry = saved[cellId] || {};
    // serialize maps
    const views = {}; p.views.forEach((v,k)=>{ views[String(k)] = v; });
    const table = {}; p.table.forEach((v,k)=>{ table[String(k)] = v; });
    saved[cellId] = Object.assign({}, entry, { views, table });
    saveAllOutputPrefs(saved);
  }
  function hydrateOutputPrefs(cellId, prefs){
    const saved = loadAllSavedOutputPrefs();
    const entry = saved[cellId]; if (!entry) return prefs;
    if (entry.views && typeof entry.views === 'object'){
      for (const k in entry.views){ prefs.views.set(Number(k), entry.views[k]); }
    }
    if (entry.table && typeof entry.table === 'object'){
      for (const k in entry.table){ prefs.table.set(Number(k), entry.table[k]); }
    }
    return prefs;
  }
  let lastFocusCellId = null; // for keyboard actions like delete
  const codePrefs = new Map(); // cellId -> { collapsed: boolean }
  function getCodePrefs(cellId){ let p = codePrefs.get(cellId); if (!p){ p={ collapsed:false }; codePrefs.set(cellId,p);} return p; }
  let dragCid = null; // for drag-and-drop reordering
  function getOutputPrefs(cellId){
    let p = outputPrefs.get(cellId);
    if (!p){ p = hydrateOutputPrefs(cellId, { expanded: new Set(), views: new Map(), table: new Map(), handles: new Map() }); outputPrefs.set(cellId, p); }
    if (!p.expanded) p.expanded = new Set();
    if (!p.views) p.views = new Map();
    if (!p.table) p.table = new Map();
    if (!p.handles) p.handles = new Map();
    return p;
  }
  function truncateText(text, opts={}){
    const maxChars = opts.maxChars ?? 1000;
    const maxLines = opts.maxLines ?? 12;
    const lines = text.split('\n');
    const overLines = lines.length > maxLines;
    const overChars = text.length > maxChars;
    const truncated = overLines || overChars;
    if (!truncated) return { visible: text, truncated: false };
    const sliceByLines = lines.slice(0, maxLines).join('\n');
    const visible = (overLines ? sliceByLines : text).slice(0, maxChars);
    return { visible, truncated: true };
  }

  function formatOutput(item){
    const mime = item.mime || 'text/plain';
    let text = '' + (item.data ?? '');
    if (mime === 'application/json'){
      try { const obj = JSON.parse(text); text = JSON.stringify(obj, null, 2); } catch(_){ /* leave as is */ }
    } else if (mime === 'application/lyra+value'){
      // Heuristic: try to parse JSON
      try { const obj = JSON.parse(text); text = JSON.stringify(obj, null, 2); } catch(_){ }
    } else if (mime === 'text/plain'){
      // Try prettifying if looks like JSON
      const t = text.trim(); if ((t.startsWith('{') && t.endsWith('}')) || (t.startsWith('[') && t.endsWith(']'))) {
        try { const obj = JSON.parse(t); text = JSON.stringify(obj, null, 2); } catch(_){ }
      }
    }
    return text;
  }

  function base64urlToBase64(s){
    let out = (s||'').replace(/-/g,'+').replace(/_/g,'/');
    const pad = out.length % 4; if (pad) out += '='.repeat(4-pad);
    return out;
  }
  function isDataUrl(s){ return typeof s === 'string' && s.startsWith('data:'); }
  function asDataUrl(mime, data){
    if (isDataUrl(data)) return data;
    const b64 = base64urlToBase64(data||'');
    return `data:${mime};base64,${b64}`;
  }
  function decodeB64ToText(b64){ try{ return atob(b64); }catch(_){ return ''; } }
  function dataUrlToText(url){
    if (!isDataUrl(url)) return '';
    const i = url.indexOf(','); if (i<0) return '';
    const head = url.slice(5, i); // after 'data:'
    const payload = url.slice(i+1);
    // If not base64, try decodeURIComponent
    if (/;base64/i.test(head)) return decodeB64ToText(payload);
    try { return decodeURIComponent(payload); } catch(_) { return payload; }
  }
  function sanitizeHtml(src){
    // Very small sanitizer: strips scripts/styles and on* attributes.
    let s = String(src || '');
    s = s.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, '')
         .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, '')
         .replace(/on[a-z]+\s*=\s*"[^"]*"/gi, '')
         .replace(/on[a-z]+\s*=\s*'[^']*'/gi, '')
         .replace(/on[a-z]+\s*=\s*[^\s>]+/gi, '');
    return s;
  }
  function tryParseJSON(text){ try { return JSON.parse(text); } catch(_) { return null; } }
  function lyraValueType(parsed){
    // Detect Lyra value type from either plain JSON ({__type:"Frame"})
    // or typed JSON ({Assoc:{"__type": {String:"Frame"}, ...}})
    if (!parsed || typeof parsed !== 'object') return null;
    if (typeof parsed.__type === 'string') return parsed.__type;
    if (parsed.Assoc && typeof parsed.Assoc === 'object'){
      const t = parsed.Assoc['__type'];
      if (t && typeof t === 'object' && typeof t.String === 'string') return t.String;
    }
    return null;
  }
  function lyraDecode(node, depth=0){
    if (depth > 64) return null;
    if (node == null) return null;
    if (Array.isArray(node)) return node.map(v => lyraDecode(v, depth+1));
    if (typeof node !== 'object') return node;
    // Typed forms
    if (node.Assoc && typeof node.Assoc === 'object'){
      const out = {};
      for (const k in node.Assoc){ out[k] = lyraDecode(node.Assoc[k], depth+1); }
      return out;
    }
    if (node.List && Array.isArray(node.List)) return node.List.map(v => lyraDecode(v, depth+1));
    if (node.Integer != null) return Number(node.Integer);
    if (node.Real != null) return Number(node.Real);
    if (Object.prototype.hasOwnProperty.call(node, 'String')) return String(node.String);
    if (Object.prototype.hasOwnProperty.call(node, 'Boolean')) return !!node.Boolean;
    if (node.Symbol === 'Null') return null;
    // Plain object path
    const out = {}; let found=false;
    for (const k in node){ found=true; out[k] = lyraDecode(node[k], depth+1); }
    return found ? out : null;
  }
  function invokeWithTimeout(cmd, args, ms=4000){
    let to; const timeout = new Promise((_, rej) => { to = setTimeout(()=>rej(new Error('timeout')), ms); });
    return Promise.race([invoke(cmd, args), timeout]).finally(()=>clearTimeout(to));
  }
  function isPrimitive(v){ return v == null || ['string','number','boolean'].includes(typeof v); }
  function toCell(v){ if (isPrimitive(v)) return String(v); try { return JSON.stringify(v); } catch(_) { return String(v); } }
  function unionKeys(rows){ const set = new Set(); rows.forEach(r=>{ Object.keys(r||{}).forEach(k=>set.add(k)); }); return Array.from(set); }
  function toCSV(rows, headers){
    const esc = s => '"' + String(s).replace(/"/g,'""') + '"';
    const lines = [];
    lines.push(headers.map(esc).join(','));
    for (const r of rows){ lines.push(headers.map(h=>esc(toCell(r[h]))).join(',')); }
    return lines.join('\n');
  }
  function renderJsonTable(obj){
    // returns { el, actions?: [{label,onclick}] } or null
    if (!Array.isArray(obj) || obj.length === 0) return null;
    const maxRows = 200; // safety cap
    if (obj.every(row => Array.isArray(row))){
      const rows = obj.slice(0, maxRows);
      const cols = rows.reduce((m, r)=>Math.max(m, r.length||0), 0);
      const table = document.createElement('table'); table.className='mini-table';
      const thead = document.createElement('thead'); const thr = document.createElement('tr');
      for (let c=0; c<cols; c++){ const th=document.createElement('th'); th.textContent = `c${c+1}`; thr.appendChild(th); }
      thead.appendChild(thr); table.appendChild(thead);
      const tbody = document.createElement('tbody');
      rows.forEach(r => { const tr=document.createElement('tr'); for(let c=0;c<cols;c++){ const td=document.createElement('td'); td.textContent = toCell(r[c]); tr.appendChild(td);} tbody.appendChild(tr); });
      table.appendChild(tbody);
      const wrap = document.createElement('div'); wrap.className='table-wrap'; wrap.style.overflow='auto'; wrap.appendChild(table);
      return { el: wrap };
    }
    if (obj.every(row => row && typeof row === 'object' && !Array.isArray(row))){
      const rows = obj.slice(0, maxRows);
      const headers = unionKeys(rows);
      const table = document.createElement('table'); table.className='mini-table';
      const thead = document.createElement('thead'); const thr = document.createElement('tr');
      headers.forEach(h => { const th=document.createElement('th'); th.textContent = h; thr.appendChild(th); });
      thead.appendChild(thr); table.appendChild(thead);
      const tbody = document.createElement('tbody');
      rows.forEach(r => { const tr=document.createElement('tr'); headers.forEach(h => { const td=document.createElement('td'); td.textContent = toCell(r[h]); tr.appendChild(td); }); tbody.appendChild(tr); });
      table.appendChild(tbody);
      const wrap = document.createElement('div'); wrap.className='table-wrap'; wrap.style.overflow='auto'; wrap.appendChild(table);
      const csv = toCSV(rows, headers);
      const actions = [{ label: 'Download CSV', onclick: () => { const blob=new Blob([csv], {type:'text/csv'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download='table.csv'; a.click(); setTimeout(()=>URL.revokeObjectURL(url), 1000); } }];
      return { el: wrap, actions };
    }
    return null;
  }
  function canRenderJsonTable(obj){
    if (!Array.isArray(obj) || obj.length === 0) return false;
    if (obj.every(row => Array.isArray(row))) return true;
    if (obj.every(row => row && typeof row === 'object' && !Array.isArray(row))) return true;
    return false;
  }

  // --- Viewer Registry (Phase 1 scaffold) ---
  const VIEWERS = [];
  function registerViewer(id, match, render){ VIEWERS.push({ id, match, render }); }
  function tryRenderWithViewers(ctx){
    // ctx: { item, parsed, decoded, list, prefs, idx, cid }
    for (const v of VIEWERS){
      try { if (v.match(ctx)) { if (v.render(ctx)) return true; } } catch(e){ console.error('viewer error', v.id, e); }
    }
    return false;
  }

  // Helpers to render previews (shared by registry + fallback)
  function renderLyraFrameDatasetPreview(ctx){
    const { list, prefs, idx, item, cid } = ctx;
    const mime = item.mime; const data = item.data;
    if (!sessionId || mime !== 'application/lyra+value') return false;
    const ph = document.createElement('div'); ph.className='pre-wrap'; ph.textContent = 'Loading preview…'; list.appendChild(ph);
    const loadingCtl = document.createElement('div'); loadingCtl.className='controls';
    const cancelChip = document.createElement('button'); cancelChip.className='chip'; cancelChip.textContent='Cancel';
    cancelChip.onclick = async () => { try { await invoke('cmd_interrupt', { sessionId: sessionId }); ph.textContent = 'Preview canceled'; } catch(e){ console.error(e); } };
    loadingCtl.appendChild(cancelChip);
    list.appendChild(loadingCtl);
    const saved = prefs.table.get(idx) || {};
    const sizes = loadPreviewSizes();
    let limit = (typeof saved.previewLimit === 'number' ? saved.previewLimit : loadPreviewDefault());
    const runPreview = () => {
      ph.textContent = 'Loading preview…';
      invokeWithTimeout('cmd_preview_value', { sessionId: sessionId, value: data, limit }, 4000).then((rowsJson) => {
        const rowsTyped = tryParseJSON(rowsJson);
        const rows = lyraDecode(rowsTyped);
        const table = Array.isArray(rows) ? renderJsonTable(rows) : null;
        if (table){
          if (loadingCtl && loadingCtl.parentNode === list) list.removeChild(loadingCtl);
          const header = document.createElement('div'); header.className = 'controls';
          const cols = Array.isArray(rows) && rows.length>0 && rows[0] && typeof rows[0]==='object' && !Array.isArray(rows[0]) ? Object.keys(rows[0]) : [];
          const colText = cols.length ? `Columns (${cols.length}): ${cols.slice(0,6).join(', ')}${cols.length>6?'…':''}` : '';
          if (colText){ const span=document.createElement('span'); span.style.color='var(--sub)'; span.textContent = colText; header.appendChild(span); }
          const sampleLab = document.createElement('span'); sampleLab.style.marginLeft='8px'; sampleLab.style.color='var(--sub)'; sampleLab.textContent = `Preview: `;
          const sel = document.createElement('select'); sel.className='select xs'; sizes.forEach(v=>{ const o=document.createElement('option'); o.value=String(v); o.textContent=String(v); if (limit===v) o.selected=true; sel.appendChild(o); });
          sel.onchange = () => { limit = parseInt(sel.value, 10); prefs.table.set(idx, Object.assign({}, saved, { previewLimit: limit })); persistOutputPrefs(cid); runPreview(); };
          sampleLab.appendChild(sel); header.appendChild(sampleLab);
          const statsChip = document.createElement('button'); statsChip.className='chip'; statsChip.style.marginLeft = '8px'; statsChip.textContent = 'View stats';
          statsChip.onclick = () => {
            invoke('cmd_table_open', { sessionId: sessionId, value: data }).then((handle) => {
              invoke('cmd_table_schema', { sessionId: sessionId, handle }).then(schema => {
                const savedState = prefs.table.get(idx) || null;
                openDataViewer(sessionId, handle, schema, savedState);
              }).catch(console.error);
            }).catch(console.error);
          };
          header.appendChild(statsChip);
          list.replaceChild(header, ph);
          list.appendChild(table.el);
          const line = document.createElement('div'); line.className='controls';
          const rawBtn = document.createElement('button'); rawBtn.className='toggle-link'; rawBtn.textContent='Raw JSON'; rawBtn.onclick = () => { prefs.views.set(idx, 'raw'); persistOutputPrefs(cid); renderCells(); };
          line.appendChild(rawBtn);
          if (table.actions){ for (const act of table.actions){ const b=document.createElement('button'); b.className='toggle-link'; b.textContent=act.label; b.onclick=act.onclick; line.appendChild(b);} }
          const openBtn = document.createElement('button'); openBtn.className='toggle-link'; openBtn.textContent='Open in Data Viewer'; openBtn.onclick = () => {
            invoke('cmd_table_open', { sessionId: sessionId, value: data }).then((handle) => {
              invoke('cmd_table_schema', { sessionId: sessionId, handle }).then(schema => {
                const savedState = prefs.table.get(idx) || null;
                openDataViewer(sessionId, handle, schema, savedState);
              }).catch(console.error);
            }).catch(console.error);
          };
          line.appendChild(openBtn);
          list.appendChild(line);
        } else { ph.textContent = 'No preview'; }
      }).catch((e)=>{
        const timedOut = (''+e).includes('timeout');
        ph.textContent = timedOut ? 'Preview timed out' : 'Preview failed';
        const line = document.createElement('div'); line.className='controls';
        const retryBtn = document.createElement('button'); retryBtn.className='toggle-link'; retryBtn.textContent='Retry'; retryBtn.onclick = () => { runPreview(); };
        line.appendChild(retryBtn);
        const cancelBtn = document.createElement('button'); cancelBtn.className='chip'; cancelBtn.textContent='Cancel';
        cancelBtn.onclick = async () => { try { await invoke('cmd_interrupt', { sessionId: sessionId }); setStatus('Stopped'); } catch(err){ console.error(err); } };
        line.appendChild(cancelBtn);
        const sampleLab = document.createElement('span'); sampleLab.style.marginLeft='8px'; sampleLab.style.color='var(--sub)'; sampleLab.textContent = `Preview: `;
        const sel = document.createElement('select'); sel.className='select xs'; sizes.forEach(v=>{ const o=document.createElement('option'); o.value=String(v); o.textContent=String(v); if (limit===v) o.selected=true; sel.appendChild(o); });
        sel.onchange = () => { limit = parseInt(sel.value, 10); prefs.table.set(idx, Object.assign({}, saved, { previewLimit: limit })); persistOutputPrefs(cid); runPreview(); };
        sampleLab.appendChild(sel); line.appendChild(sampleLab);
        list.appendChild(line);
      });
    };
    runPreview();
    return true;
  }

  function renderPlainJsonTablePreview(ctx){
    const { list, prefs, idx, decoded, item, cid } = ctx;
    const data = item.data;
    const table = decoded && canRenderJsonTable(decoded) ? renderJsonTable(decoded) : null;
    if (!table) return false;
    const header = document.createElement('div'); header.className = 'controls';
    const rows = Array.isArray(decoded) ? decoded : [];
    let cols = [];
    if (rows.length > 0){ if (rows[0] && typeof rows[0] === 'object' && !Array.isArray(rows[0])) cols = Object.keys(rows[0]); else if (Array.isArray(rows[0])) cols = rows[0].map((_,i)=>`c${i+1}`); }
    const colText = cols.length ? `Columns (${cols.length}): ${cols.slice(0,6).join(', ')}${cols.length>6?'…':''}` : '';
    if (colText){ const span=document.createElement('span'); span.style.color='var(--sub)'; span.textContent = colText; header.appendChild(span); }
    const statsChip = document.createElement('button'); statsChip.className='chip'; statsChip.style.marginLeft = '8px'; statsChip.textContent = 'View stats';
    statsChip.onclick = () => {
      invoke('cmd_table_open', { sessionId: sessionId, value: data }).then((handle) => {
        invoke('cmd_table_schema', { sessionId: sessionId, handle }).then(schema => {
          const savedState = prefs.table.get(idx) || null;
          openDataViewer(sessionId, handle, schema, savedState);
        }).catch(console.error);
      }).catch(console.error);
    };
    header.appendChild(statsChip);
    list.appendChild(header);
    list.appendChild(table.el);
    const line = document.createElement('div'); line.className='controls';
    const copyBtn = document.createElement('button'); copyBtn.className='copy-link'; copyBtn.textContent='Copy JSON'; copyBtn.onclick = async () => { await copyToClipboard(data); };
    line.appendChild(copyBtn);
    if (table.actions){ for (const act of table.actions){ const b=document.createElement('button'); b.className='toggle-link'; b.textContent=act.label; b.onclick=act.onclick; line.appendChild(b);} }
    const rawBtn = document.createElement('button'); rawBtn.className='toggle-link'; rawBtn.textContent='Raw JSON'; rawBtn.onclick = () => { prefs.views.set(idx, 'raw'); persistOutputPrefs(cid); renderCells(); };
    line.appendChild(rawBtn);
    list.appendChild(line);
    return true;
  }

  // Register built-in viewers (feature-flag ready)
  registerViewer('lyra.frame', (ctx) => {
    const { item, prefs, idx, parsed } = ctx;
    if (!item || item.mime !== 'application/lyra+value') return false;
    const t = lyraValueType(parsed); const isDF = t === 'Dataset' || t === 'Frame';
    const viewMode = prefs.views.get(idx) || 'table';
    return isDF && viewMode === 'table';
  }, (ctx) => renderLyraFrameDatasetPreview(ctx));

  registerViewer('json.table', (ctx) => {
    const { item, decoded, prefs, idx } = ctx;
    if (!item || item.mime !== 'application/json') return false;
    const viewMode = prefs.views.get(idx) || 'table';
    return decoded && Array.isArray(decoded) && viewMode === 'table' && canRenderJsonTable(decoded);
  }, (ctx) => renderPlainJsonTablePreview(ctx));

  // --- Virtualized Table ---
  function createVirtualTable(handle, schema, sessionId, state, onState, onQueryChange){
    const rowHeight = 28; const headHeight = 32; const filterHeight = 28; const overscan = 8;
    const cols = schema.columns || [];
    let total = Math.max(0, schema.row_count_approx || 0);
    const colWidths = Array.isArray(state?.widths) && state.widths.length===cols.length ? state.widths.slice() : cols.map(() => 160);
    let order = Array.isArray(state?.order) && state.order.length===cols.length ? state.order.slice() : cols.map((_,i)=>i);
    let frozenCount = typeof state?.frozenCount === 'number' ? Math.max(0, Math.min(cols.length, state.frozenCount)) : 0;
    let sortCol = (typeof state?.sortCol === 'number') ? state.sortCol : null;
    let sortDir = state?.sortDir || null; // 'asc'|'desc'|null
    const filters = (state?.filters && typeof state.filters === 'object') ? Object.assign({}, state.filters) : {}; // colIdx -> string
    let searchText = state?.searchText || '';
    const selected = new Set(); let lastSel = null;
    const wrap = document.createElement('div'); wrap.className='vt-wrap scrollshadow'; wrap.style.position='relative'; wrap.style.height='360px';

    const head = document.createElement('div'); head.className='vt-head'; head.style.top = '0px';
    const headRow = document.createElement('div'); headRow.className='vt-row'; headRow.style.height = headHeight+'px';
    const filterRow = document.createElement('div'); filterRow.className='vt-row'; filterRow.style.height = filterHeight+'px'; filterRow.style.top = headHeight+'px';
    let stats = {}; let statsReady = false; let statsFetchInFlight = false; const statsCache = new Map(); // key -> stats map
    head.style.left = '0'; head.style.right = '0';
    function renderHead(){
      head.innerHTML=''; head.appendChild(headRow); head.appendChild(filterRow);
      const leftOffsets = [];
      let acc = 0; for (let i=0;i<order.length;i++){ leftOffsets[i]=acc; acc += colWidths[order[i]]; }
      headRow.innerHTML=''; filterRow.innerHTML='';
      order.forEach((colIdx, i)=>{
        const c = cols[colIdx];
        const cell=document.createElement('div'); cell.className='vt-cell'; cell.style.width = colWidths[colIdx]+'px';
        // header label with sort indicator
        const label = document.createElement('span');
        label.textContent = c.name + (sortCol===colIdx ? (sortDir==='asc'?' ▲':' ▼') : '');
        cell.appendChild(label);
        // mini histogram if available
        const s = stats[c.name];
        if (s && s.histogram && Array.isArray(s.histogram.bins) && s.histogram.bins.length){
          const maxc = Math.max(...s.histogram.bins);
          const spark = document.createElement('div'); spark.className='vt-spark';
          s.histogram.bins.forEach(count => { const bar = document.createElement('div'); bar.style.height = (maxc>0 ? Math.max(1, Math.round((count/maxc)*12)) : 1)+'px'; spark.appendChild(bar); });
          cell.appendChild(spark);
        }
        // null ratio bar
        if (s && typeof s.nulls === 'number' && total > 0){
          const pct = Math.max(0, Math.min(100, Math.round((s.nulls/total)*100)));
          const nb = document.createElement('div'); nb.className='vt-nullbar';
          const fill = document.createElement('div'); fill.style.width = pct + '%'; nb.appendChild(fill);
          cell.appendChild(nb);
        }
        // tooltip with stats
        const showStats = (ev) => {
          const st = stats[c.name]; if (!st) return;
          const fmt = (x, d=4) => {
            if (x == null) return '—';
            const n = Number(x); if (Number.isFinite(n)) return n.toFixed(d).replace(/\.0+$/,'').replace(/(\.[0-9]*?)0+$/,'$1');
            return String(x);
          };
          const nulls = st.nulls || 0; const approxPct = total > 0 ? ((nulls/total)*100).toFixed(1) : '0.0';
          let html = `<div class=\"usage\">${escapeHtml(c.name)}</div>`;
          if (st.histogram){
            html += `<div>min: <b>${fmt(st.min)}</b>, max: <b>${fmt(st.max)}</b></div>`;
            if (st.mean != null) html += `<div>mean: <b>${fmt(st.mean)}</b>, stddev: <b>${fmt(st.stddev)}</b></div>`;
          }
          if (Array.isArray(st.topk)){
            const top = st.topk.slice(0,5).map(pair => `${escapeHtml(String(pair[0]))} <span style=\"color:var(--sub)\">(${pair[1]})</span>`).join(', ');
            if (top) html += `<div>top: ${top}</div>`;
          }
          html += `<div>nulls: <b>${nulls}</b> <span style=\"color:var(--sub)\">(${approxPct}%)</span></div>`;
          showDocTipAt(ev.clientX, ev.clientY, html);
        };
        cell.addEventListener('mousemove', showStats);
        cell.addEventListener('mouseleave', hideDocTip);
        // sort on click
        cell.onclick = (ev)=>{ if (ev.target !== cell) return; toggleSort(colIdx); };
        cell.style.position = '';
        if (i < frozenCount){ cell.classList.add('sticky'); cell.style.left = leftOffsets[i]+ 'px'; cell.style.zIndex = 3; }
        // Resize handle
        const rz=document.createElement('div'); rz.className='vt-resizer'; rz.onmousedown=(ev)=>startResize(ev, colIdx); cell.appendChild(rz);
        // Reorder via drag
        cell.setAttribute('draggable','true');
        cell.ondragstart = (e)=>{ e.dataTransfer.setData('text/plain', String(i)); };
        cell.ondragover = (e)=>{ e.preventDefault(); };
        cell.ondrop = (e)=>{ e.preventDefault(); const from = parseInt(e.dataTransfer.getData('text/plain')||'-1',10); if (!isNaN(from) && from>=0){ reorder(from, i); } };
        headRow.appendChild(cell);
        // filter cell
        const fcell = document.createElement('div'); fcell.className='vt-cell'; fcell.style.width = colWidths[colIdx]+'px';
        const fin = document.createElement('input'); fin.type='text'; fin.placeholder='Filter'; fin.value = filters[colIdx] || '';
        fin.oninput = debounce(()=>{ filters[colIdx] = fin.value; refreshQuery(true); }, 250);
        fin.style.width='100%'; fin.style.boxSizing='border-box'; fin.style.height='20px'; fin.style.margin='3px 0';
        fcell.appendChild(fin);
        if (i < frozenCount){ fcell.classList.add('sticky'); fcell.style.left = leftOffsets[i] + 'px'; fcell.style.zIndex = 3; }
        filterRow.appendChild(fcell);
      });
    }
    wrap.appendChild(head);

    const inner = document.createElement('div'); inner.style.position='relative'; inner.style.height = (headHeight + filterHeight + total*rowHeight) + 'px'; wrap.appendChild(inner);
    const rowsHost = document.createElement('div'); rowsHost.style.position='absolute'; rowsHost.style.left='0'; rowsHost.style.right='0'; inner.appendChild(rowsHost);

    attachScrollShadow(wrap, wrap);

    // simple cache by page offset
    const pageSize = 200;
    const pages = new Map(); // offset -> rows (array)
    let fetching = new Set();

    function ensurePage(offset){
      const key = Math.floor(offset / pageSize) * pageSize;
      if (pages.has(key) || fetching.has(key)) return;
      fetching.add(key);
      invoke('cmd_table_query', { sessionId, handle, query: { offset: key, limit: pageSize } })
        .then(resp => {
          const rows = resp && resp.rows ? resp.rows : [];
          pages.set(key, Array.isArray(rows) ? rows : []);
          fetching.delete(key);
          renderViewport();
        })
        .catch(e => { console.error('table query error', e); fetching.delete(key); });
    }

    function getRow(idx){
      const key = Math.floor(idx / pageSize) * pageSize;
      const arr = pages.get(key); if (!arr) return null;
      return arr[idx - key] || null;
    }

    function renderViewport(){
      const scrollTop = wrap.scrollTop;
      const h = wrap.clientHeight;
      const firstRow = Math.max(0, Math.floor(Math.max(0, scrollTop - headHeight - filterHeight) / rowHeight) - overscan);
      const visible = Math.ceil((h - headHeight - filterHeight) / rowHeight) + overscan*2;
      const startY = headHeight + filterHeight + firstRow * rowHeight;
      // ensure needed pages
      ensurePage(firstRow);
      ensurePage(firstRow + visible);
      // draw rows
      rowsHost.innerHTML='';
      // compute sticky lefts for frozen
      const leftOffsets = [];
      let acc = 0; for (let i=0;i<order.length;i++){ leftOffsets[i]=acc; acc += colWidths[order[i]]; }
      let y = startY;
      for(let r = firstRow; r < Math.min(total, firstRow + visible); r++){
        const row = getRow(r);
        const rowEl = document.createElement('div'); rowEl.className='vt-row'; rowEl.style.top = y+'px'; rowEl.style.height = rowHeight+'px';
        if (selected.has(r)) rowEl.classList.add('selected');
        rowEl.onclick = (ev) => {
          if (ev.shiftKey && lastSel != null){
            const [a,b] = [lastSel, r].sort((x,y)=>x-y);
            for (let k=a; k<=b; k++) selected.add(k);
          } else if (ev.metaKey || ev.ctrlKey){
            if (selected.has(r)) selected.delete(r); else selected.add(r);
            lastSel = r;
          } else {
            selected.clear(); selected.add(r); lastSel = r;
          }
          renderViewport();
        };
        // cells
        order.forEach((colOrderIdx, i) => {
          const c = cols[colOrderIdx];
          const cell = document.createElement('div'); cell.className='vt-cell'; cell.style.width = colWidths[colOrderIdx]+'px';
          let val = '';
          if (row && typeof row === 'object'){
            if (Array.isArray(row)) { val = toCell(row[colOrderIdx]); }
            else { val = toCell(row[c.name]); }
          }
          cell.textContent = val;
          if (i < frozenCount){ cell.classList.add('sticky'); cell.style.left = leftOffsets[i] + 'px'; cell.style.zIndex = 2; }
          rowEl.appendChild(cell);
        });
        rowsHost.appendChild(rowEl);
        y += rowHeight;
      }
    }

    wrap.addEventListener('scroll', renderViewport);
    function buildQuery(offset, limit){
      const q = { offset, limit, sort: [], filters: [], search: (searchText||'').trim() || undefined };
      if (sortCol != null && sortDir){ q.sort.push({ col: columnKey(sortCol), dir: sortDir }); }
      for (const k in filters){ const val = (filters[k]||'').trim(); if (val){ q.filters.push({ col: columnKey(parseInt(k,10)), op: 'contains', value: val }); } }
      return q;
    }
    function columnKey(colIdx){
      const col = cols[colIdx];
      // For arrays-of-arrays we expect cN keys on backend; send c{index+1}
      return col && col.name ? col.name : 'c'+(colIdx+1);
    }
    function refreshQuery(resetScroll){
      pages.clear(); fetching.clear();
      if (resetScroll){ wrap.scrollTop = 0; }
      ensurePage(0);
      fetchStatsDebounced();
      if (typeof onQueryChange === 'function') { try { onQueryChange(buildQuery(undefined, undefined)); } catch(_){} }
    }
    // initial fetch
    invoke('cmd_table_query', { sessionId, handle, query: buildQuery(0, pageSize) })
      .then(resp => { const rows = resp && resp.rows ? resp.rows : []; pages.set(0, Array.isArray(rows) ? rows : []); total = (resp && resp.page && typeof resp.page.total_approx==='number') ? resp.page.total_approx : total; inner.style.height = (headHeight + filterHeight + total*rowHeight) + 'px'; renderViewport(); })
      .catch(e => console.error(e));
    // initial stats fetch
    fetchStatsDebounced();
    if (typeof onQueryChange === 'function') { try { onQueryChange(buildQuery(undefined, undefined)); } catch(_){} }

    function persist(){ if (typeof onState === 'function') onState({ widths: colWidths.slice(), order: order.slice(), frozenCount, sortCol, sortDir, filters: Object.assign({}, filters), searchText }); }
    function startResize(ev, colIdx){
      ev.preventDefault(); ev.stopPropagation();
      const startX = ev.clientX; const startW = colWidths[colIdx];
      const onMove = (e)=>{ const dx = e.clientX - startX; colWidths[colIdx] = Math.max(60, startW + dx); renderHead(); renderViewport(); };
      const onUp = ()=>{ window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); persist(); };
      window.addEventListener('mousemove', onMove); window.addEventListener('mouseup', onUp);
    }
    function reorder(fromIdx, toIdx){
      if (fromIdx === toIdx) return;
      const it = order.splice(fromIdx, 1)[0]; order.splice(toIdx, 0, it);
      renderHead(); renderViewport(); persist();
    }
    function toggleSort(colIdx){
      if (sortCol !== colIdx){ sortCol = colIdx; sortDir = 'asc'; }
      else if (sortDir === 'asc'){ sortDir = 'desc'; }
      else { sortCol = null; sortDir = null; }
      renderHead(); persist(); refreshQuery(true);
    }
    // Small toolbar for freeze toggle and reset
    const tools = document.createElement('div'); tools.className='controls';
    const freezeBtn = document.createElement('button'); freezeBtn.className='toggle-link';
    const updateFreezeLabel = ()=>{ freezeBtn.textContent = frozenCount>0 ? `Unfreeze (${frozenCount})` : 'Freeze first col'; };
    freezeBtn.onclick = ()=>{ frozenCount = frozenCount>0 ? 0 : 1; renderHead(); renderViewport(); persist(); };
    updateFreezeLabel();
    const resetBtn = document.createElement('button'); resetBtn.className='toggle-link'; resetBtn.textContent='Reset layout';
    resetBtn.onclick = ()=>{ for (let i=0;i<colWidths.length;i++) colWidths[i]=160; order = cols.map((_,i)=>i); frozenCount=0; renderHead(); renderViewport(); persist(); };
    const searchIn = document.createElement('input'); searchIn.type='text'; searchIn.placeholder='Search'; searchIn.value = searchText || ''; searchIn.oninput = debounce(()=>{ searchText = searchIn.value; persist(); refreshQuery(true); }, 250);
    tools.appendChild(searchIn);
    // Copy selection
    const copySelBtn = document.createElement('button'); copySelBtn.className='toggle-link'; copySelBtn.textContent='Copy selection';
    copySelBtn.onclick = async () => {
      const rows = await getRowsForIndices(Array.from(selected.values()).sort((a,b)=>a-b));
      const csv = rowsToCsv(rows);
      await copyToClipboard(csv);
    };
    // Export CSV (all results)
    const exportBtn = document.createElement('button'); exportBtn.className='toggle-link'; exportBtn.textContent='Export CSV';
    exportBtn.onclick = async () => {
      const csv = await exportAllCsv();
      const blob=new Blob([csv], {type:'text/csv'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download='table.csv'; a.click(); setTimeout(()=>URL.revokeObjectURL(url), 1000);
    };
    tools.appendChild(copySelBtn);
    tools.appendChild(exportBtn);
    tools.appendChild(freezeBtn); tools.appendChild(resetBtn);

    renderHead();
    return { el: wrap, tools };

    function headers(){ return order.map(colIdx => cols[colIdx]?.name || ('c'+(colIdx+1))); }
    function rowsToCsv(rows){
      const hdr = headers();
      const esc = s => '"' + String(s??'').replace(/"/g,'""') + '"';
      const lines = [];
      lines.push(hdr.map(esc).join(','));
      for (const r of rows){
        if (Array.isArray(r)) { lines.push(order.map(i => esc(toCell(r[i]))).join(',')); }
        else if (r && typeof r === 'object') { lines.push(order.map(i => esc(toCell(r[cols[i].name]))).join(',')); }
        else { lines.push(esc(toCell(r))); }
      }
      return lines.join('\n');
    }
    async function getRowsForIndices(indices){
      // Ensure pages for any missing indices
      const needed = new Set(indices.map(i => Math.floor(i / pageSize) * pageSize));
      const fetches = [];
      needed.forEach(off => { if (!pages.has(off)) fetches.push(invoke('cmd_table_query', { sessionId, handle, query: buildQuery(off, pageSize) }).then(resp => { const rows = resp && resp.rows ? resp.rows : []; pages.set(off, Array.isArray(rows) ? rows : []); })); });
      if (fetches.length) await Promise.all(fetches);
      // Collect rows in displayed shape (array or object)
      const out = [];
      for (const i of indices){ const key = Math.floor(i / pageSize) * pageSize; const arr = pages.get(key)||[]; const row = arr[i - key]; if (row != null) out.push(row); }
      return out;
    }
    async function exportAllCsv(){
      const hdr = headers();
      const esc = s => '"' + String(s??'').replace(/"/g,'""') + '"';
      const chunks = [hdr.map(esc).join(',')];
      const limit = pageSize;
      let offset = 0; let totalRows = total;
      while (offset < totalRows){
        // fetch page
        /* eslint-disable no-await-in-loop */
        const resp = await invoke('cmd_table_query', { sessionId, handle, query: buildQuery(offset, limit) });
        const rows = (resp && resp.rows && Array.isArray(resp.rows)) ? resp.rows : [];
        totalRows = (resp && resp.page && typeof resp.page.total_approx==='number') ? resp.page.total_approx : totalRows;
        for (const r of rows){
          if (Array.isArray(r)) { chunks.push(order.map(i => esc(toCell(r[i]))).join(',')); }
          else if (r && typeof r === 'object') { chunks.push(order.map(i => esc(toCell(r[cols[i].name]))).join(',')); }
          else { chunks.push(esc(toCell(r))); }
        }
        offset += limit;
      }
      return chunks.join('\n');
    }

    const fetchStatsDebounced = debounce(fetchStats, 200);
    function fetchStats(){
      const key = statsKey();
      if (statsCache.has(key)) { stats = statsCache.get(key) || {}; statsReady = true; renderHead(); return; }
      if (statsFetchInFlight) return;
      statsFetchInFlight = true;
      const colsReq = headers();
      const q = buildQuery(undefined, undefined);
      // exclude sort from stats signature/use to maximize reuse
      delete q.sort;
      invoke('cmd_table_stats', { sessionId, handle, columns: colsReq, query: q })
        .then(map => { stats = map || {}; statsCache.set(key, stats); statsReady = true; renderHead(); })
        .catch(e => console.error('stats error', e))
        .finally(()=>{ statsFetchInFlight = false; });
    }
    function statsKey(){
      // Keyed by search + filters only (sort/order/frozen/widths don’t affect stats)
      const keyFilters = {};
      for (const k in filters){ const v = (filters[k]||'').trim(); if (v) keyFilters[k] = v; }
      return JSON.stringify({ s: (searchText||'').trim(), f: keyFilters });
    }
  }
  function mimeExt(mime){
    if (!mime || typeof mime !== 'string') return 'bin';
    const m = mime.toLowerCase();
    if (m === 'image/png') return 'png';
    if (m === 'image/jpeg') return 'jpg';
    if (m === 'image/svg+xml') return 'svg';
    if (m === 'audio/wav' || m==='audio/x-wav') return 'wav';
    if (m === 'audio/mpeg') return 'mp3';
    if (m === 'audio/ogg') return 'ogg';
    const parts = m.split('/'); return parts[1] || 'bin';
  }

  function highlightJson(src){
    const esc = (s) => s.replace(/&/g,'&amp;').replace(/</g,'&lt;');
    return esc(src)
      .replace(/(".*?")(?=\s*:)/g, '<span class="out-key">$1</span>')
      .replace(/"(.*?)"/g, '<span class="out-string">"$1"<\/span>')
      .replace(/\b(true|false)\b/g, '<span class="out-bool">$1<\/span>')
      .replace(/\b(null)\b/g, '<span class="out-null">$1<\/span>')
      .replace(/-?\b(\d+(?:\.\d+)?)\b/g, '<span class="out-number">$1<\/span>')
      .replace(/[\{\}\[\]\:,]/g, '<span class="out-punct">$&<\/span>');
  }

  function setStatus(msg, opts={}) {
    const el = byId('status'); if (!el) return;
    el.textContent = msg;
    const isError = opts.level === 'error' || /^\s*error\b/i.test(msg);
    el.setAttribute('aria-live', isError ? 'assertive' : 'polite');
  }

  function renderCells() {
    const cont = byId('cells');
    cont.innerHTML = '';
    if (!notebook || !Array.isArray(notebook.cells)) return;
    notebook.cells.forEach((cell, idx) => {
      const cid = normId(cell.id);
      const root = document.createElement('div');
      root.className = 'cell';
      root.id = `cell-${cid}`;
      root.addEventListener('mousedown', () => { lastFocusCellId = cid; });
      // DnD drop targets
      root.ondragover = (e) => {
        if (!dragCid || dragCid === cid) return;
        e.preventDefault();
        const rect = root.getBoundingClientRect();
        const before = (e.clientY - rect.top) < rect.height/2;
        root.classList.toggle('drop-before', before);
        root.classList.toggle('drop-after', !before);
      };
      root.ondragleave = () => { root.classList.remove('drop-before','drop-after'); };
      root.ondrop = async (e) => {
        if (!dragCid || dragCid === cid) return;
        e.preventDefault(); root.classList.remove('drop-before','drop-after');
        const fromIdx = notebook.cells.findIndex(c => normId(c.id)===dragCid);
        const toIdxBase = notebook.cells.findIndex(c => normId(c.id)===cid);
        if (fromIdx < 0 || toIdxBase < 0) { dragCid = null; return; }
        const rect = root.getBoundingClientRect();
        const before = (e.clientY - rect.top) < rect.height/2;
        let toIdx = before ? toIdxBase : toIdxBase + 1;
        const [moved] = notebook.cells.splice(fromIdx, 1);
        if (fromIdx < toIdx) toIdx -= 1;
        toIdx = Math.max(0, Math.min(notebook.cells.length, toIdx));
        notebook.cells.splice(toIdx, 0, moved);
        dragCid = null;
        renderCells(); renderSidebar();
        if (sessionId) invoke('cmd_update_session_notebook', { sessionId: sessionId, notebook }).catch(console.error);
      };
      // state: idle | running | ok | error
      const state = (() => {
        const m = cell.meta || {};
        if (m.running) return 'running';
        if (m.error) return 'error';
        const outs = Array.isArray(cell.output) ? cell.output : [];
        if (outs.length > 0) return 'ok';
        return 'idle';
      })();
      root.dataset.state = state;

      const gutter = document.createElement('div');
      gutter.className = 'gutter';
      const inLab = document.createElement('div'); inLab.className='label'; inLab.textContent = `In [${idx+1}]`;
      gutter.appendChild(inLab);
      root.appendChild(gutter);

      const body = document.createElement('div'); body.className = 'body';

      const head = document.createElement('div');
      head.className = 'cell-head';
      const left = document.createElement('div');
      const execCount = (cell.meta && typeof cell.meta.execCount === 'number') ? cell.meta.execCount : 0;
      const timing = (cell.meta && typeof cell.meta.timingMs === 'number') ? `${cell.meta.timingMs} ms` : '';
      const err = (cell.meta && typeof cell.meta.error === 'string') ? 'error' : '';
      left.innerHTML = `<div class="badges"><span class="badge">${cell.type}</span><span class="badge">${cell.language}</span>${timing?`<span class="badge">${timing}</span>`:''}${execCount?`<span class="badge">#${execCount}</span>`:''}${err?`<span class="badge" style="border-color:#ef4444;color:#ef4444">Error</span>`:''}</div>`;
      const actions = document.createElement('div');
      actions.className = 'cell-actions';
      const drag = document.createElement('div'); drag.className='drag-handle'; drag.title='Drag to reorder'; drag.setAttribute('draggable','true');
      drag.ondragstart = (e) => { dragCid = cid; e.dataTransfer.effectAllowed='move'; try{ e.dataTransfer.setData('text/plain', cid);}catch(_){} };
      drag.ondragend = () => { dragCid = null; document.querySelectorAll('.cell.drop-before,.cell.drop-after').forEach(el => el.classList.remove('drop-before','drop-after')); };
      actions.appendChild(drag);
      const runBtn = document.createElement('button'); runBtn.className='btn'; runBtn.textContent='Run';
      runBtn.disabled = !sessionId || cell.type !== 'Code' || cell.language !== 'Lyra' || state === 'running';
      runBtn.onclick = () => runCell(cid);
      // per-cell spinner when running
      if (state === 'running') {
        const sp = document.createElement('span'); sp.className = 'spinner'; sp.title = 'Running';
        actions.appendChild(sp);
      }
      const delBtn = document.createElement('button'); delBtn.className='btn ghost'; delBtn.textContent='Delete';
      delBtn.onclick = () => deleteCell(cid);
      actions.appendChild(runBtn);
      actions.appendChild(delBtn);
      head.appendChild(left);
      head.appendChild(actions);
      body.appendChild(head);

      if (cell.type === 'Code' && typeof cell.input === 'string') {
        const shell = document.createElement('div'); shell.className = 'editor-shell';
        const wrap = document.createElement('div'); wrap.className = 'ed-wrap';
        const highlight = document.createElement('div'); highlight.className = 'ed-highlight';
        const ta = document.createElement('textarea'); ta.className = 'code'; ta.spellcheck = false; ta.setAttribute('wrap','off');
        ta.value = cell.input || '';
        wrap.appendChild(highlight);
        wrap.appendChild(ta);
        shell.appendChild(wrap);
        body.appendChild(shell);
        const diag = document.createElement('div'); diag.className = 'diag-msg';
        shell.appendChild(diag);
        const api = createLyraEditor(cid, ta, highlight, diag);
        editors.set(cid, api);
        api.refresh();
        ta.addEventListener('focus', () => { activeEditor = { cellId: cid, textarea: ta, focus: () => ta.focus(), insert: (txt) => insertTextAtCaret(ta, txt) }; lastFocusCellId = cid; });
        // Auto-resize with collapse/expand toggle
        const cp = getCodePrefs(cid);
        const collapsedMax = 220; const minH = 80;
        function autoResize(){
          // Compute content height (prefer textarea scrollHeight)
          ta.style.height = 'auto';
          let h = Math.max(minH, ta.scrollHeight);
          if (cp.collapsed) { h = Math.min(h, collapsedMax); ta.style.overflow='auto'; }
          else { ta.style.overflow='hidden'; }
          ta.style.height = h + 'px';
          // match wrapper height so highlight overlay fits
          if (wrap) wrap.style.height = h + 'px';
        }
        ta.addEventListener('input', autoResize);
        // Ensure initial size after syntax highlight refresh
        setTimeout(autoResize, 0);
        const edTools = document.createElement('div'); edTools.className='editor-tools';
        const toggle = document.createElement('button'); toggle.className='btn xs ghost';
        function updateToggle(){ toggle.textContent = cp.collapsed ? 'Expand editor ▾' : 'Collapse editor ▴'; }
        toggle.onclick = () => { cp.collapsed = !cp.collapsed; updateToggle(); autoResize(); };
        updateToggle(); edTools.appendChild(toggle); shell.appendChild(edTools);
      } else if (cell.type === 'Markdown' || cell.type === 'Text') {
        const ta = document.createElement('textarea'); ta.className='textcell'; ta.value = cell.input || '';
        ta.oninput = debounce(() => {
          const c = notebook.cells.find(c=>normId(c.id)===cid); if (c) c.input = ta.value; if (sessionId) invoke('cmd_update_session_notebook', { sessionId, notebook }).catch(console.error);
        }, 200);
        ta.addEventListener('focus', () => { lastFocusCellId = cid; });
        body.appendChild(ta);
      }

      // Error panel (if any)
      if (cell.meta && typeof cell.meta.error === 'string' && cell.meta.error.length > 0) {
        const ep = document.createElement('div'); ep.className = 'error-panel';
        const eh = document.createElement('div'); eh.className = 'error-head'; eh.innerHTML = '<span class="error-dot"></span> Error';
        const pre = document.createElement('pre'); pre.className = 'error-pre pre-wrap scrollshadow'; pre.textContent = cell.meta.error;
        ep.appendChild(eh); ep.appendChild(pre);
        attachScrollShadow(pre, pre);
        // actions: copy
        const line = document.createElement('div'); line.className = 'controls';
        const copyBtn = document.createElement('button'); copyBtn.className = 'copy-link'; copyBtn.textContent = 'Copy error';
        copyBtn.onclick = async () => { await copyToClipboard(cell.meta.error); };
        line.appendChild(copyBtn);
        ep.appendChild(line);
        body.appendChild(ep);
      }

      // Outputs
      const out = document.createElement('div'); out.className='outputs';
      const outLab = document.createElement('div'); outLab.className='out-label'; outLab.textContent = `Out[${idx+1}]`;
      out.appendChild(outLab);
      const list = document.createElement('div');
      const prefs = getOutputPrefs(cid);
      const outs = cell.output || [];
      outs.forEach((item, idx) => {
        const mime = item.mime || 'text/plain';
        const data = item.data;
        if (typeof mime === 'string' && mime.startsWith('image/')){
          const img = document.createElement('img');
          img.src = asDataUrl(mime, data);
          img.alt = 'image output';
          img.style.maxWidth = '100%'; img.style.display='block';
          list.appendChild(img);
          const line = document.createElement('div'); line.className = 'controls';
          const copyBtn = document.createElement('button'); copyBtn.className='copy-link'; copyBtn.textContent='Copy data URL';
          copyBtn.onclick = async () => { await copyToClipboard(img.src); };
          const dlBtn = document.createElement('button'); dlBtn.className='toggle-link'; dlBtn.textContent='Download';
          dlBtn.onclick = () => { const a=document.createElement('a'); a.href=img.src; a.download=`output.${mimeExt(mime)}`; a.click(); };
          line.appendChild(copyBtn); line.appendChild(dlBtn); list.appendChild(line);
        } else if (typeof mime === 'string' && mime.startsWith('audio/')){
          const au = document.createElement('audio'); au.controls = true; au.preload = 'metadata'; au.src = asDataUrl(mime, data);
          list.appendChild(au);
          const line = document.createElement('div'); line.className = 'controls';
          const copyBtn = document.createElement('button'); copyBtn.className='copy-link'; copyBtn.textContent='Copy data URL';
          copyBtn.onclick = async () => { await copyToClipboard(au.src); };
          const dlBtn = document.createElement('button'); dlBtn.className='toggle-link'; dlBtn.textContent='Download';
          dlBtn.onclick = () => { const a=document.createElement('a'); a.href=au.src; a.download=`output.${mimeExt(mime)}`; a.click(); };
          line.appendChild(copyBtn); line.appendChild(dlBtn); list.appendChild(line);
        } else if (typeof mime === 'string' && mime.startsWith('video/')){
          const vd = document.createElement('video'); vd.controls = true; vd.preload = 'metadata'; vd.playsInline = true; vd.style.maxWidth='100%';
          vd.src = asDataUrl(mime, data);
          list.appendChild(vd);
          const line = document.createElement('div'); line.className = 'controls';
          const copyBtn = document.createElement('button'); copyBtn.className='copy-link'; copyBtn.textContent='Copy data URL';
          copyBtn.onclick = async () => { await copyToClipboard(vd.src); };
          const dlBtn = document.createElement('button'); dlBtn.className='toggle-link'; dlBtn.textContent='Download';
          dlBtn.onclick = () => { const a=document.createElement('a'); a.href=vd.src; a.download=`output.${mimeExt(mime)}`; a.click(); };
          line.appendChild(copyBtn); line.appendChild(dlBtn); list.appendChild(line);
        } else if (mime === 'text/html'){
          const html = isDataUrl(data) ? dataUrlToText(data) : decodeB64ToText(base64urlToBase64(data));
          const iframe = document.createElement('iframe');
          iframe.setAttribute('sandbox',''); // fully sandboxed; no scripts
          iframe.style.width = '100%'; iframe.style.border = '1px solid var(--border)'; iframe.style.borderRadius = '8px';
          iframe.srcdoc = sanitizeHtml(html);
          list.appendChild(iframe);
          const line = document.createElement('div'); line.className = 'controls';
          const copyBtn = document.createElement('button'); copyBtn.className='copy-link'; copyBtn.textContent='Copy HTML';
          copyBtn.onclick = async () => { await copyToClipboard(iframe.srcdoc || ''); };
          const dlBtn = document.createElement('button'); dlBtn.className='toggle-link'; dlBtn.textContent='Download';
          const url = asDataUrl('text/html', data);
          dlBtn.onclick = () => { const a=document.createElement('a'); a.href=url; a.download='output.html'; a.click(); };
          line.appendChild(copyBtn); line.appendChild(dlBtn); list.appendChild(line);
        } else if (mime === 'application/json' || mime === 'application/lyra+value'){
          const parsed = tryParseJSON(typeof data === 'string' ? data : '');
          // Phase 1: Viewer registry hook
          const decodedForRegistry = lyraDecode(parsed);
          const ctxForRegistry = { item, parsed, decoded: decodedForRegistry, list, prefs, idx, cid };
          if (tryRenderWithViewers(ctxForRegistry)) return;
          const viewMode = prefs.views.get(idx) || 'table';
          const lyraT = mime === 'application/lyra+value' ? lyraValueType(parsed) : null;
          const isLyraDF = !!sessionId && (lyraT === 'Dataset' || lyraT === 'Frame');
          // Special-casing common Lyra types for preview (only when table view selected)
          if (viewMode === 'table' && isLyraDF){
            const ph = document.createElement('div'); ph.className='pre-wrap'; ph.textContent = 'Loading preview…'; list.appendChild(ph);
            // Loading controls with a tiny Cancel chip to interrupt kernel if stuck
            const loadingCtl = document.createElement('div'); loadingCtl.className='controls';
            const cancelChip = document.createElement('button'); cancelChip.className='chip'; cancelChip.textContent='Cancel';
            cancelChip.onclick = async () => {
              try { await invoke('cmd_interrupt', { sessionId: sessionId }); ph.textContent = 'Preview canceled'; } catch(e){ console.error(e); }
            };
            loadingCtl.appendChild(cancelChip);
            list.appendChild(loadingCtl);
            const saved = prefs.table.get(idx) || {};
            const sizes = loadPreviewSizes();
            let limit = (typeof saved.previewLimit === 'number' ? saved.previewLimit : loadPreviewDefault());
            const runPreview = () => {
              ph.textContent = 'Loading preview…';
              invokeWithTimeout('cmd_preview_value', { sessionId: sessionId, value: data, limit }, 4000).then((rowsJson) => {
                const rowsTyped = tryParseJSON(rowsJson);
                const rows = lyraDecode(rowsTyped);
                const table = Array.isArray(rows) ? renderJsonTable(rows) : null;
                if (table){
                  // Remove loading controls
                  if (loadingCtl && loadingCtl.parentNode === list) list.removeChild(loadingCtl);
                  // Build a mini header showing columns and sample size
                  const header = document.createElement('div'); header.className = 'controls';
                  const cols = Array.isArray(rows) && rows.length>0 && rows[0] && typeof rows[0]==='object' && !Array.isArray(rows[0]) ? Object.keys(rows[0]) : [];
                  const colText = cols.length ? `Columns (${cols.length}): ${cols.slice(0,6).join(', ')}${cols.length>6?'…':''}` : '';
                  if (colText){ const span=document.createElement('span'); span.style.color='var(--sub)'; span.textContent = colText; header.appendChild(span); }
                  // Sample size selector (settings-backed)
                  const sampleLab = document.createElement('span'); sampleLab.style.marginLeft='8px'; sampleLab.style.color='var(--sub)'; sampleLab.textContent = `Preview: `;
                  const sel = document.createElement('select'); sel.className='select xs'; sizes.forEach(v=>{ const o=document.createElement('option'); o.value=String(v); o.textContent=String(v); if (limit===v) o.selected=true; sel.appendChild(o); });
                  sel.onchange = () => { limit = parseInt(sel.value, 10); prefs.table.set(idx, Object.assign({}, saved, { previewLimit: limit })); persistOutputPrefs(cid); runPreview(); };
                  sampleLab.appendChild(sel); header.appendChild(sampleLab);
                  // Inline stats chip in header
                  const statsChip = document.createElement('button'); statsChip.className='chip'; statsChip.style.marginLeft = '8px'; statsChip.textContent = 'View stats';
                  statsChip.onclick = () => {
                    invoke('cmd_table_open', { sessionId: sessionId, value: data }).then((handle) => {
                      invoke('cmd_table_schema', { sessionId: sessionId, handle }).then(schema => {
                        const savedState = prefs.table.get(idx) || null;
                        openDataViewer(sessionId, handle, schema, savedState);
                      }).catch(console.error);
                    }).catch(console.error);
                  };
                  header.appendChild(statsChip);
                  list.replaceChild(header, ph);
                  list.appendChild(table.el);
                  const line = document.createElement('div'); line.className='controls';
                  const rawBtn = document.createElement('button'); rawBtn.className='toggle-link'; rawBtn.textContent='Raw JSON'; rawBtn.onclick = () => { prefs.views.set(idx, 'raw'); persistOutputPrefs(cid); renderCells(); };
                  line.appendChild(rawBtn);
                  if (table.actions){ for (const act of table.actions){ const b=document.createElement('button'); b.className='toggle-link'; b.textContent=act.label; b.onclick=act.onclick; line.appendChild(b);} }
                  const openBtn = document.createElement('button'); openBtn.className='toggle-link'; openBtn.textContent='Open in Data Viewer'; openBtn.onclick = () => {
                    // Open full virtualized viewer using table APIs
                    invoke('cmd_table_open', { sessionId: sessionId, value: data }).then((handle) => {
                      invoke('cmd_table_schema', { sessionId: sessionId, handle }).then(schema => {
                        const savedState = prefs.table.get(idx) || null;
                        openDataViewer(sessionId, handle, schema, savedState);
                      }).catch(console.error);
                    }).catch(console.error);
                  };
                  line.appendChild(openBtn);
                  list.appendChild(line);
                } else { ph.textContent = 'No preview'; }
              }).catch((e)=>{
                const timedOut = (''+e).includes('timeout');
                ph.textContent = timedOut ? 'Preview timed out' : 'Preview failed';
                const line = document.createElement('div'); line.className='controls';
                const retryBtn = document.createElement('button'); retryBtn.className='toggle-link'; retryBtn.textContent='Retry'; retryBtn.onclick = () => { runPreview(); };
                line.appendChild(retryBtn);
                const cancelBtn = document.createElement('button'); cancelBtn.className='chip'; cancelBtn.textContent='Cancel';
                cancelBtn.onclick = async () => { try { await invoke('cmd_interrupt', { sessionId: sessionId }); setStatus('Stopped'); } catch(err){ console.error(err); } };
                line.appendChild(cancelBtn);
                const openBtn = document.createElement('button'); openBtn.className='toggle-link'; openBtn.textContent='Open in Data Viewer'; openBtn.onclick = () => {
                  invoke('cmd_table_open', { sessionId: sessionId, value: data }).then((handle) => {
                    invoke('cmd_table_schema', { sessionId: sessionId, handle }).then(schema => {
                      const savedState = prefs.table.get(idx) || null;
                      openDataViewer(sessionId, handle, schema, savedState);
                    }).catch(console.error);
                  }).catch(console.error);
                };
                line.appendChild(openBtn);
                // sample selector available even on error (settings-backed)
                const sampleLab = document.createElement('span'); sampleLab.style.marginLeft='8px'; sampleLab.style.color='var(--sub)'; sampleLab.textContent = `Preview: `;
                const sel = document.createElement('select'); sel.className='select xs'; sizes.forEach(v=>{ const o=document.createElement('option'); o.value=String(v); o.textContent=String(v); if (limit===v) o.selected=true; sel.appendChild(o); });
                sel.onchange = () => { limit = parseInt(sel.value, 10); prefs.table.set(idx, Object.assign({}, saved, { previewLimit: limit })); persistOutputPrefs(cid); runPreview(); };
                sampleLab.appendChild(sel); line.appendChild(sampleLab);
                list.appendChild(line);
              });
            };
            runPreview();
            return;
          }
          const decoded = lyraDecode(parsed);
          const table = (viewMode === 'table' && decoded && canRenderJsonTable(decoded)) ? renderJsonTable(decoded) : null;
          if (viewMode === 'table' && table){
            // Header with columns summary and inline stats chip
            const header = document.createElement('div'); header.className = 'controls';
            const rows = Array.isArray(decoded) ? decoded : [];
            let cols = [];
            if (rows.length > 0){
              if (rows[0] && typeof rows[0] === 'object' && !Array.isArray(rows[0])) cols = Object.keys(rows[0]);
              else if (Array.isArray(rows[0])) cols = rows[0].map((_,i)=>`c${i+1}`);
            }
            const colText = cols.length ? `Columns (${cols.length}): ${cols.slice(0,6).join(', ')}${cols.length>6?'…':''}` : '';
            if (colText){ const span=document.createElement('span'); span.style.color='var(--sub)'; span.textContent = colText; header.appendChild(span); }
            const statsChip = document.createElement('button'); statsChip.className='chip'; statsChip.style.marginLeft = '8px'; statsChip.textContent = 'View stats';
            statsChip.onclick = () => {
              // Attempt to open JSON data in Data Viewer
              invoke('cmd_table_open', { sessionId: sessionId, value: data }).then((handle) => {
                invoke('cmd_table_schema', { sessionId: sessionId, handle }).then(schema => {
                  const savedState = prefs.table.get(idx) || null;
                  openDataViewer(sessionId, handle, schema, savedState);
                }).catch(console.error);
              }).catch(console.error);
            };
            header.appendChild(statsChip);
            list.appendChild(header);
            list.appendChild(table.el);
            const line = document.createElement('div'); line.className='controls';
            const copyBtn = document.createElement('button'); copyBtn.className='copy-link'; copyBtn.textContent='Copy JSON';
            copyBtn.onclick = async () => { await copyToClipboard(data); };
            line.appendChild(copyBtn);
            if (table.actions){ for (const act of table.actions){ const b=document.createElement('button'); b.className='toggle-link'; b.textContent=act.label; b.onclick=act.onclick; line.appendChild(b);} }
            const rawBtn = document.createElement('button'); rawBtn.className='toggle-link'; rawBtn.textContent='Raw JSON'; rawBtn.onclick = () => { prefs.views.set(idx, 'raw'); persistOutputPrefs(cid); renderCells(); };
            line.appendChild(rawBtn);
            list.appendChild(line);
          } else {
            // Raw JSON view
            const pre = document.createElement('pre');
            const formatted = formatOutput(item);
            const expanded = prefs.expanded.has(idx);
            const { visible, truncated } = expanded ? { visible: formatted, truncated: false } : truncateText(formatted);
            pre.innerHTML = highlightJson(visible) + (truncated ? '<br>…' : '');
            pre.className = 'pre-wrap scrollshadow';
            list.appendChild(pre);
            attachScrollShadow(pre, pre);
            const line = document.createElement('div'); line.className = 'controls';
            const copyBtn = document.createElement('button'); copyBtn.className = 'copy-link'; copyBtn.textContent = 'Copy';
            copyBtn.onclick = async () => { await copyToClipboard(item.data); };
            line.appendChild(copyBtn);
            const tableCapable = parsed && canRenderJsonTable(parsed);
            if (tableCapable || isLyraDF) {
              const tableBtn = document.createElement('button'); tableBtn.className='toggle-link'; tableBtn.textContent='Table view'; tableBtn.onclick = () => { prefs.views.set(idx, 'table'); persistOutputPrefs(cid); renderCells(); };
              line.appendChild(tableBtn);
            }
            if (truncated || expanded) {
              const btn = document.createElement('button');
              btn.className = 'toggle-link';
              btn.textContent = expanded ? 'Collapse' : 'Show more';
              btn.onclick = () => { if (expanded) prefs.expanded.delete(idx); else prefs.expanded.add(idx); renderCells(); };
              line.appendChild(btn);
            }
            list.appendChild(line);
          }
        } else {
          const pre = document.createElement('pre');
          const formatted = formatOutput(item);
          const expanded = prefs.expanded.has(idx);
          const { visible, truncated } = expanded ? { visible: formatted, truncated: false } : truncateText(formatted);
          pre.innerHTML = highlightJson(visible) + (truncated ? '<br>…' : '');
          pre.className = 'pre-wrap scrollshadow';
          list.appendChild(pre);
          attachScrollShadow(pre, pre);
          const line = document.createElement('div'); line.className = 'controls';
          const copyBtn = document.createElement('button'); copyBtn.className = 'copy-link'; copyBtn.textContent = 'Copy';
          copyBtn.onclick = async () => { await copyToClipboard(item.data); };
          line.appendChild(copyBtn);
          if (truncated || expanded) {
            const btn = document.createElement('button');
            btn.className = 'toggle-link';
            btn.textContent = expanded ? 'Collapse' : 'Show more';
            btn.onclick = () => { if (expanded) prefs.expanded.delete(idx); else prefs.expanded.add(idx); renderCells(); };
            line.appendChild(btn);
          }
          list.appendChild(line);
        }
      });
      out.appendChild(list);
      body.appendChild(out);

      root.appendChild(body);
      cont.appendChild(root);
    });
    renderSidebar();
    updateImpactUI();
  }

  // --- Data Viewer Overlay ---
  function openDataViewer(sessionId, handle, schema, state){
    const root = byId('dataViewer'); if (!root) return;
    const colsEl = byId('viewerCols'); const tableEl = byId('viewerTable'); const statsEl = byId('viewerStats');
    root.hidden = false;
    const close = () => { root.hidden = true; tableEl.innerHTML=''; colsEl.innerHTML=''; statsEl.innerHTML=''; };
    const closeBtn = byId('viewerClose'); if (closeBtn) closeBtn.onclick = close;
    root.querySelector('.viewer-backdrop').onclick = close;

    // Build columns list
    const colNames = (schema.columns||[]).map(c=>c.name || '');
    let selectedCol = colNames[0] || null;
    function renderCols(){
      colsEl.innerHTML='';
      const title = document.createElement('div'); title.style.fontWeight='600'; title.style.margin='6px 0'; title.textContent = 'Columns'; colsEl.appendChild(title);
      colNames.forEach(name => {
        const d = document.createElement('div'); d.className = 'col' + (name===selectedCol?' active':''); d.textContent = name || '—'; d.onclick = ()=>{ selectedCol = name; renderCols(); fetchAndRenderStats(lastQuery); };
        colsEl.appendChild(d);
      });
    }
    renderCols();

    // Create virtual table inside viewer
    let lastState = state || null;
    let lastQuery = null;
    const vt = createVirtualTable(handle, schema, sessionId, state, (st)=>{ lastState = st; }, (q)=>{ lastQuery = q; fetchAndRenderStats(q); });
    tableEl.appendChild(vt.el);
    if (vt.tools){ const tools = document.createElement('div'); tools.style.marginTop='8px'; tools.appendChild(vt.tools); tableEl.appendChild(tools); }
    // Fetch initial stats for the first column
    fetchAndRenderStats(null);

    async function fetchAndRenderStats(q){
      try{
        const res = await invoke('cmd_table_stats', { sessionId, handle, columns: selectedCol?[selectedCol]:[], query: Object.assign({}, q, { sort: undefined }) });
        renderStats(res);
      }catch(e){ console.error(e); }
    }

    function renderStats(map){
      statsEl.innerHTML = '';
      const name = selectedCol;
      if (!name || !map || !map[name]){ statsEl.textContent = 'No stats'; return; }
      const st = map[name];
      const title = document.createElement('div'); title.style.fontWeight='600'; title.textContent = name; statsEl.appendChild(title);
      // numbers
      const list = document.createElement('div'); list.style.display='grid'; list.style.gridTemplateColumns='1fr 1fr'; list.style.gap='6px'; list.style.marginTop='8px';
      const addKV = (k,v) => { const kdiv=document.createElement('div'); kdiv.style.color='var(--sub)'; kdiv.textContent=k; const vdiv=document.createElement('div'); vdiv.style.textAlign='right'; vdiv.textContent=v; list.appendChild(kdiv); list.appendChild(vdiv); };
      const fmt = (x, d=4) => { if (x == null) return '—'; const n = Number(x); if (Number.isFinite(n)) return n.toFixed(d).replace(/\.0+$/,'').replace(/(\.[0-9]*?)0+$/,'$1'); return String(x); };
      addKV('Nulls', String(st.nulls ?? 0));
      if (st.min != null) addKV('Min', fmt(st.min));
      if (st.max != null) addKV('Max', fmt(st.max));
      if (st.mean != null) addKV('Mean', fmt(st.mean));
      if (st.stddev != null) addKV('Stddev', fmt(st.stddev));
      statsEl.appendChild(list);

      if (st.histogram && Array.isArray(st.histogram.bins) && st.histogram.bins.length){
        const maxc = Math.max(...st.histogram.bins);
        const container = document.createElement('div'); container.style.height='120px'; container.style.display='flex'; container.style.alignItems='flex-end'; container.style.gap='2px'; container.style.marginTop='12px';
        st.histogram.bins.forEach(count => { const bar=document.createElement('div'); bar.style.flex='1'; bar.style.background='var(--accent)'; bar.style.opacity='.8'; bar.style.height = (maxc>0 ? Math.max(2, Math.round((count/maxc)*120)) : 2)+'px'; container.appendChild(bar); });
        statsEl.appendChild(container);
      }
      if (Array.isArray(st.topk) && st.topk.length){
        const chart = document.createElement('div'); chart.className='chart-bars';
        const maxv = Math.max(...st.topk.map(x=>x[1]));
        st.topk.slice(0,10).forEach(([label,count])=>{
          const row = document.createElement('div'); row.className='chart-row';
          const lab = document.createElement('div'); lab.className='label'; lab.textContent = String(label);
          const bar = document.createElement('div'); bar.className='bar';
          const fill = document.createElement('div'); fill.style.width = (maxv>0? Math.round((count/maxv)*100):0) + '%'; bar.appendChild(fill);
          const val = document.createElement('div'); val.style.minWidth='3ch'; val.textContent = String(count);
          row.appendChild(lab); row.appendChild(bar); row.appendChild(val); chart.appendChild(row);
        });
        statsEl.appendChild(chart);
      }
    }
  }

  function insertTextAtCaret(textarea, text){
    const s = textarea.selectionStart, e = textarea.selectionEnd, v = textarea.value;
    textarea.value = v.slice(0,s) + text + v.slice(e);
    const p = s + text.length; textarea.selectionStart = textarea.selectionEnd = p;
    textarea.dispatchEvent(new Event('input'));
  }

  // Compute caret coordinates inside a textarea
  function caretCoords(textarea){
    const rect = textarea.getBoundingClientRect();
    const style = window.getComputedStyle(textarea);
    // mirror div
    const div = document.createElement('div');
    div.style.position = 'absolute';
    div.style.whiteSpace = 'pre-wrap';
    div.style.wordWrap = 'break-word';
    div.style.visibility = 'hidden';
    div.style.zIndex = '-1';
    const props = ['fontFamily','fontSize','lineHeight','letterSpacing','tabSize','padding','border','boxSizing','width'];
    props.forEach(p => { div.style[p.replace(/[A-Z]/g, m=>'-'+m.toLowerCase())] = style[p]; });
    div.style.width = rect.width + 'px';
    const before = textarea.value.substring(0, textarea.selectionStart);
    const span = document.createElement('span'); span.textContent = '\u200b';
    const escape = (s) => s.replace(/&/g,'&amp;').replace(/</g,'&lt;');
    div.innerHTML = escape(before).replace(/\n/g, '<br>');
    div.appendChild(span);
    document.body.appendChild(div);
    const srect = span.getBoundingClientRect();
    const x = srect.left - div.getBoundingClientRect().left - textarea.scrollLeft +  parseFloat(style.paddingLeft || '0');
    const y = srect.top - div.getBoundingClientRect().top  - textarea.scrollTop  +  parseFloat(style.paddingTop || '0');
    const lh = parseFloat(style.lineHeight) || 16;
    document.body.removeChild(div);
    return { x, y, line: lh };
  }

  // Caret coordinates relative to viewport (for positioning popups)
  function caretClient(textarea){
    const rect = textarea.getBoundingClientRect();
    const style = window.getComputedStyle(textarea);
    const div = document.createElement('div');
    div.style.position = 'fixed';
    div.style.left = rect.left + 'px';
    div.style.top = rect.top + 'px';
    div.style.width = rect.width + 'px';
    div.style.whiteSpace = 'pre-wrap';
    div.style.wordWrap = 'break-word';
    div.style.visibility = 'hidden';
    const props = ['fontFamily','fontSize','lineHeight','letterSpacing','tabSize','paddingTop','paddingBottom','paddingLeft','paddingRight','boxSizing'];
    props.forEach(p => { div.style[p.replace(/[A-Z]/g, m=>'-'+m.toLowerCase())] = style[p]; });
    const before = textarea.value.substring(0, textarea.selectionStart);
    const span = document.createElement('span'); span.textContent = '\u200b';
    const escape = (s) => s.replace(/&/g,'&amp;').replace(/</g,'&lt;');
    div.innerHTML = escape(before).replace(/  /g,' &nbsp;').replace(/\n/g, '<br>');
    div.appendChild(span);
    document.body.appendChild(div);
    const srect = span.getBoundingClientRect();
    const lh = parseFloat(style.lineHeight) || 16;
    document.body.removeChild(div);
    return { clientX: srect.left - textarea.scrollLeft, clientY: srect.top - textarea.scrollTop, line: lh };
  }

  function showProgress(on) {
    const bar = byId('topProgress');
    if (!bar) return;
    if (on) { bar.hidden = false; }
    else { bar.hidden = true; }
  }
  // Track running cells and bulk ops for top progress
  const runningCells = new Set();
  let bulkOps = 0;
  function updateProgress(){ showProgress(runningCells.size > 0 || bulkOps > 0); }
  function addRunning(cid){ runningCells.add(cid); updateProgress(); }
  function removeRunning(cid){ runningCells.delete(cid); updateProgress(); }

  // Docs sidebar rendering
  async function setCurrentDocSymbol(name){
    if (!name || name === currentDocSymbol) return renderDocs();
    currentDocSymbol = name;
    if (!docsCache.has(name)){
      try { const doc = await invoke('cmd_editor_doc', { name }); docsCache.set(name, doc || null); } catch(_) { docsCache.set(name, null); }
    }
    renderDocs();
  }
  function renderDocs(){
    const cont = byId('panelDocs'); if (!cont) return;
    cont.innerHTML = '';
    if (!currentDocSymbol){ cont.textContent = 'Place the caret on a symbol or use Cmd/Ctrl+I'; return; }
    const d = docsCache.get(currentDocSymbol);
    if (!d){ cont.textContent = `No documentation for ${currentDocSymbol}`; return; }
    const title = document.createElement('div'); title.className='doc-name'; title.textContent = d.name || currentDocSymbol; cont.appendChild(title);
    const usage = document.createElement('div'); usage.className='doc-usage'; usage.textContent = (d.params && d.params.length) ? `${currentDocSymbol}[${d.params.join(', ')}]` : `${currentDocSymbol}[]`; cont.appendChild(usage);
    const summary = document.createElement('div'); summary.className='doc-summary'; summary.textContent = d.summary || '';
    cont.appendChild(summary);
    // actions
    const actions = document.createElement('div'); actions.style.margin='8px 0'; actions.style.display='flex'; actions.style.gap='8px';
    const insertUsage = document.createElement('button'); insertUsage.className='btn xs'; insertUsage.textContent='Insert usage'; insertUsage.onclick = () => { if (activeEditor && activeEditor.insert) activeEditor.insert(usage.textContent || ''); };
    actions.appendChild(insertUsage);
    cont.appendChild(actions);
    if (Array.isArray(d.examples) && d.examples.length){
      const exH = document.createElement('div'); exH.className='doc-examples'; exH.textContent='Examples'; cont.appendChild(exH);
      d.examples.slice(0,3).forEach(ex => { const pre=document.createElement('pre'); pre.textContent = ex; cont.appendChild(pre); const btn=document.createElement('button'); btn.className='btn xs link'; btn.textContent='Insert example'; btn.onclick=()=>{ if (activeEditor && activeEditor.insert) activeEditor.insert(ex); }; cont.appendChild(btn); });
    }
  }
  function incRunning(){ runningCount++; showProgress(true); }
  function decRunning(){ runningCount = Math.max(0, runningCount-1); if (runningCount===0) showProgress(false); }

  async function copyToClipboard(text){
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        setStatus('Copied output');
        return true;
      }
    } catch (_){}
    // Fallback
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.position='fixed'; ta.style.opacity='0'; ta.style.pointerEvents='none';
    document.body.appendChild(ta); ta.focus(); ta.select();
    try { document.execCommand('copy'); setStatus('Copied output'); return true; } catch(_) { setStatus('Copy failed'); return false; }
    finally { document.body.removeChild(ta); }
  }

  async function runCell(cellId) {
    if (!sessionId) return;
    setStatus('Running...');
    // clear outputs immediately and render
    const cell = notebook.cells.find(c => normId(c.id) === cellId);
    if (cell) { cell.output = []; cell.meta = Object.assign({}, cell.meta, { running: true, error: null }); renderCells(); }
    try {
      await invoke('cmd_execute_cell_stream', { sessionId: sessionId, cellId: cellId });
      setStatus('Done');
    } catch (e) {
      console.error(e);
      setStatus('Error: ' + e);
    } finally {
      // Fallback: if events didn't clear running, stop it now
      const c2 = notebook.cells.find(c => normId(c.id) === cellId);
      if (c2 && c2.meta && c2.meta.running) { c2.meta.running = false; renderCells(); }
      // Also drop top-progress for this cell if still tracked
      removeRunning(cellId);
    }
  }

  async function runAndAdvance(cellId){
    await runCell(cellId);
    if (!notebook) return;
    const idx = notebook.cells.findIndex(c => normId(c.id)===cellId);
    const next = notebook.cells[idx+1];
    if (next){ const cid = normId(next.id); const el = byId('cell-'+cid); if (el) el.scrollIntoView({behavior:'smooth', block:'center'}); const ed = editors.get(cid); if (ed && ed.focus) ed.focus(); else { const ta = el?.querySelector('textarea'); if (ta) ta.focus(); } lastFocusCellId = cid; }
  }

  async function addCodeCellBelow(cellId){
    if (!sessionId || !notebook) return;
    try {
      const beforeIds = new Set(notebook.cells.map(c=>normId(c.id)));
      const nb = await invoke('cmd_add_cell', { sessionId: sessionId, cellType: 'Code' });
      notebook = nb;
      // find the new cell id appended at end
      const newCell = notebook.cells.find(c => !beforeIds.has(normId(c.id)));
      if (!newCell) { renderCells(); return; }
      const fromIdx = notebook.cells.findIndex(c => normId(c.id)===normId(newCell.id));
      const idx = notebook.cells.findIndex(c => normId(c.id)===cellId);
      const insertAt = Math.min(notebook.cells.length-1, idx+1);
      if (fromIdx >= 0 && insertAt !== fromIdx){
        const [moved] = notebook.cells.splice(fromIdx, 1);
        notebook.cells.splice(insertAt, 0, moved);
        if (sessionId) await invoke('cmd_update_session_notebook', { sessionId: sessionId, notebook });
      }
      renderCells();
      const cid = normId(newCell.id); const el = byId('cell-'+cid); if (el) el.scrollIntoView({behavior:'smooth', block:'center'}); const ed = editors.get(cid); if (ed && ed.focus) ed.focus();
      lastFocusCellId = cid;
    } catch (e) { console.error(e); setStatus('Error: '+e, { level:'error' }); }
  }

async function duplicateCell(cellId){
    if (!sessionId || !notebook) return;
    const src = notebook.cells.find(c => normId(c.id)===cellId);
    if (!src) return;
    try {
      // create a new cell of same type, then copy content and insert below
      const typ = src.type === 'Text' ? 'Text' : (src.type === 'Markdown' ? 'Markdown' : 'Code');
      const beforeIds = new Set(notebook.cells.map(c=>normId(c.id)));
      const nb = await invoke('cmd_add_cell', { sessionId: sessionId, cellType: typ });
      notebook = nb;
      const newCell = notebook.cells.find(c => !beforeIds.has(normId(c.id)));
      if (!newCell) { renderCells(); return; }
      newCell.input = src.input || '';
      newCell.language = src.language || newCell.language;
      // insert below source
      const fromIdx = notebook.cells.findIndex(c => normId(c.id)===normId(newCell.id));
      const idx = notebook.cells.findIndex(c => normId(c.id)===cellId);
      const insertAt = Math.min(notebook.cells.length-1, idx+1);
      if (fromIdx >= 0 && insertAt !== fromIdx){
        const [moved] = notebook.cells.splice(fromIdx, 1);
        notebook.cells.splice(insertAt, 0, moved);
}
  function jumpToProblem(delta){
  if (!notebook) return;
  const list = notebook.cells.map(c => normId(c.id)).filter(cid => problems.has(cid));
  if (list.length === 0){ setStatus('No problems'); return; }
  let curIdx = list.indexOf(lastFocusCellId || '');
  if (curIdx < 0) curIdx = -1;
  let nextIdx = (curIdx + delta) % list.length; if (nextIdx < 0) nextIdx = list.length - 1;
  const cid = list[nextIdx];
  const cell = notebook.cells.find(c => normId(c.id)===cid);
  if (!cell){ setStatus('No problems'); return; }
  const el = byId('cell-'+cid); if (el){ el.scrollIntoView({behavior:'smooth', block:'center'}); el.classList.add('flash'); setTimeout(()=>el.classList.remove('flash'), 800); }
  const prob = problems.get(cid);
  const ed = editors.get(cid);
  if (ed && ed.focus){ ed.focus(); }
  if (prob && prob.range && ed && typeof ed.setSelectionFromLineCol === 'function'){
    ed.setSelectionFromLineCol(prob.range.start_line|0, prob.range.start_col|0);
  }
  lastFocusCellId = cid;
}
      await invoke('cmd_update_session_notebook', { sessionId: sessionId, notebook });
      renderCells();
      const cid = normId(newCell.id); const el = byId('cell-'+cid); if (el) el.scrollIntoView({behavior:'smooth', block:'center'}); const ed = editors.get(cid); if (ed && ed.focus) ed.focus();
      lastFocusCellId = cid;
    } catch (e) { console.error(e); setStatus('Error: '+e, { level:'error' }); }
  }
  function moveCell(cellId, delta){
    if (!notebook || !Array.isArray(notebook.cells)) return;
    const idx = notebook.cells.findIndex(c => normId(c.id)===cellId);
    if (idx < 0) return;
    const to = Math.max(0, Math.min(notebook.cells.length-1, idx + delta));
    if (to === idx) return;
    const [moved] = notebook.cells.splice(idx, 1);
    notebook.cells.splice(to, 0, moved);
    if (sessionId) invoke('cmd_update_session_notebook', { sessionId: sessionId, notebook }).catch(console.error);
    renderCells(); renderSidebar();
    const cid = normId(moved.id);
    const el = byId('cell-'+cid); if (el) el.scrollIntoView({behavior:'smooth', block:'center'});
    const ed = editors.get(cid); if (ed && ed.focus) ed.focus(); else { const ta = el?.querySelector('textarea'); if (ta) ta.focus(); }
    lastFocusCellId = cid;
  }

  async function runAll() {
    if (!sessionId || !notebook) return;
    setStatus('Running all...');
    bulkOps++; updateProgress();
    try {
      const ids = notebook.cells.filter(c => c.type === 'Code' && c.language === 'Lyra').map(c => normId(c.id));
      const results = await invoke('cmd_execute_all', { sessionId: sessionId, ids, method: 'Linear' });
      // Merge outputs back into local model
      const map = new Map();
      results.forEach(r => {
        const k = normId(r.cell_id || r.cellId);
        if (k) map.set(k, r.outputs || r.output || []);
      });
      notebook.cells.forEach(c => { const outs = map.get(normId(c.id)); if (outs) { c.output = outs.map(it => ({ mime: it.mime, data: it.data })); } });
      renderCells();
      // snapshot hashes and clear dirties
      (notebook.cells||[]).forEach(c => { if (c.type==='Code' && c.language==='Lyra'){ lastRunHash.set(normId(c.id), hashText(c.input||'')); dirtyCells.delete(normId(c.id)); } });
      saveRunState(); updateImpactUI();
      setStatus('Done');
    } catch (e) {
      console.error(e); setStatus('Error: ' + e);
    } finally { bulkOps = Math.max(0, bulkOps-1); updateProgress(); }
  }
  function showRunConfirm(impCount, allCount){
    const barId = 'runConfirmBar';
    let bar = document.getElementById(barId);
    if (!bar){
      bar = document.createElement('div'); bar.id = barId; bar.style.position='sticky'; bar.style.top='52px'; bar.style.zIndex='45'; bar.style.background='var(--surface-2)'; bar.style.borderBottom='1px solid var(--border)'; bar.style.padding='8px 16px'; bar.style.display='flex'; bar.style.alignItems='center'; bar.style.gap='8px';
      const main = document.querySelector('main > .toolbar');
      if (main && main.parentElement) main.parentElement.insertBefore(bar, main.nextSibling);
      else document.body.appendChild(bar);
    }
    bar.innerHTML='';
    const msg = document.createElement('span'); msg.textContent = `Detected ${impCount} impacted cell${impCount!==1?'s':''}.`;
    const runImp = document.createElement('button'); runImp.className='btn primary'; runImp.textContent = `Run Impacted (${impCount})`; runImp.onclick = async ()=>{ bar.remove(); await runImpacted(); };
    const runAllBtn = document.createElement('button'); runAllBtn.className='btn'; runAllBtn.textContent = `Run All (${allCount})`; runAllBtn.onclick = async ()=>{ bar.remove(); await runAll(); };
    const cancel = document.createElement('button'); cancel.className='btn ghost'; cancel.textContent='Cancel'; cancel.onclick = ()=>{ bar.remove(); };
    bar.appendChild(msg); bar.appendChild(runImp); bar.appendChild(runAllBtn); bar.appendChild(cancel);
    setTimeout(()=>{ if (bar && bar.parentElement) bar.parentElement.removeChild(bar); }, 10000);
  }
  async function confirmAndRunAll(){
    if (!notebook) { await runAll(); return; }
    const ids = computeImpacted();
    const codeCount = (notebook.cells||[]).filter(c=>c.type==='Code' && c.language==='Lyra').length;
    if (ids.length > 0 && ids.length < codeCount){ showRunConfirm(ids.length, codeCount); }
    else { await runAll(); }
  }

  async function runImpacted(){
    if (!sessionId || !notebook) return;
    const ids = computeImpacted();
    if (ids.length === 0){ setStatus('Nothing impacted'); return; }
    setStatus(`Running ${ids.length} impacted cells...`);
    bulkOps++; updateProgress();
    try {
      const ordered = topoOrder(ids);
      const results = await invoke('cmd_execute_all', { sessionId: sessionId, ids: ordered, method: 'Linear' });
      const map = new Map();
      results.forEach(r => { const k = normId(r.cell_id || r.cellId); if (k) map.set(k, r.outputs || r.output || []); });
      notebook.cells.forEach(c => { const k = normId(c.id); if (map.has(k)) { const outs = map.get(k); c.output = outs.map(it => ({ mime: it.mime, data: it.data })); lastRunHash.set(k, hashText(c.input||'')); dirtyCells.delete(k); } });
      saveRunState(); renderCells(); updateImpactUI();
      setStatus('Done');
    } catch (e) {
      console.error(e); setStatus('Error: ' + e);
    } finally { bulkOps = Math.max(0, bulkOps-1); updateProgress(); }
  }
  function topoOrder(ids){
    const idSet = new Set(ids);
    const inDeg = new Map(); const adj = new Map();
    ids.forEach(cid => { inDeg.set(cid, 0); adj.set(cid, new Set()); });
    ids.forEach(cid => { const info = cellDeps.get(cid); if (!info) return; info.deps.forEach(depCid => { if (idSet.has(depCid)) { inDeg.set(cid, (inDeg.get(cid)||0)+1); const s = adj.get(depCid)||new Set(); s.add(cid); adj.set(depCid, s); } }); });
    const q = []; inDeg.forEach((d,cid)=>{ if (d===0) q.push(cid); });
    const out = [];
    while(q.length){ const cur=q.shift(); out.push(cur); (adj.get(cur)||new Set()).forEach(nxt=>{ const d=(inDeg.get(nxt)||0)-1; inDeg.set(nxt,d); if (d===0) q.push(nxt); }); }
    if (out.length === ids.length) return out;
    const order = (notebook?.cells||[]).map(c=>normId(c.id));
    return order.filter(cid => idSet.has(cid));
  }

  async function stop() {
    if (!sessionId) return;
    try { const ok = await invoke('cmd_interrupt', { sessionId: sessionId }); setStatus(ok? 'Stopped' : 'Stop failed'); } catch(e){ console.error(e); setStatus('Error: '+e); }
  }

  async function addCodeCell() {
    if (!sessionId) return;
    try {
      const nb = await invoke('cmd_add_cell', { sessionId: sessionId, cellType: 'Code' });
      notebook = nb; renderCells();
    } catch (e) { console.error(e); setStatus('Error: '+e); }
  }

  async function addTextCell() {
    if (!sessionId) return;
    try { const nb = await invoke('cmd_add_cell', { sessionId: sessionId, cellType: 'Text' }); notebook = nb; renderCells(); }
    catch(e){ console.error(e); setStatus('Error: '+e); }
  }

  async function deleteCell(cellId) {
    if (!sessionId) return;
    try { const nb = await invoke('cmd_delete_cell', { sessionId: sessionId, cellId: cellId }); notebook = nb; renderCells(); }
    catch(e){ console.error(e); setStatus('Error: '+e); }
  }

  async function openNotebook() {
    const path = byId('path').value.trim();
    if (!path) return;
    setStatus('Opening...');
    try {
      const res = await invoke('cmd_open_notebook', { path });
      // res: { session_id, notebook }
      sessionId = res.session_id || res.sessionId;
      notebook = res.notebook;
      byId('saveBtn').disabled = false;
      // load persisted run state for this path
      loadRunState();
      renderCells();
      setStatus('Opened');
    } catch (e) {
      console.error(e);
      setStatus('Error: ' + e);
    }
  }

  async function saveNotebook() {
    if (!sessionId) return;
    const path = byId('path').value.trim();
    const includeOutputs = byId('includeOutputs').checked;
    setStatus('Saving...');
    try {
      // ensure session state reflects latest local edits
      await invoke('cmd_update_session_notebook', { sessionId: sessionId, notebook });
      await invoke('cmd_save_notebook', { sessionId: sessionId, path, includeOutputs: includeOutputs, pretty: true });
      setStatus('Saved');
    } catch (e) {
      console.error(e);
      setStatus('Error: ' + e);
    }
  }

  let subscribed = false;
  async function ensureSubscribed() {
    if (subscribed || !window.__TAURI__ || !window.__TAURI__.event) return;
    subscribed = true;
    await window.__TAURI__.event.listen('lyra://exec', (ev) => {
      try {
        const p = ev.payload || {};
        const sid = p.sessionId || p.session_id;
        if (!sid || sid !== sessionId) return;
        const evt = p.event || p;
        // handle shapes {Started:{cell_id}}, {Output:{cell_id,item}}, {Finished:{result}}
        const started = evt.Started || evt.started;
        const output = evt.Output || evt.output;
        const finished = evt.Finished || evt.finished;
        const errorEvt = evt.Error || evt.error;
        if (started && (started.cell_id || started.cellId)) {
          const raw = started.cell_id || started.cellId; const cid = normId(raw);
          const cell = notebook.cells.find(c => normId(c.id) === cid);
          if (cell) { cell.output = []; cell.meta = Object.assign({}, cell.meta, { running: true, error: null }); renderCells(); }
          addRunning(cid);
          // add to runQueue
          const idx = notebook.cells.findIndex(c => normId(c.id) === cid);
          runQueue.push({ cid, idx, startedAt: Date.now() });
          renderSidebar();
        } else if (output && (output.cell_id || output.cellId)) {
          const raw = output.cell_id || output.cellId; const cid = normId(raw);
          const cell = notebook.cells.find(c => normId(c.id) === cid);
          const item = output.item || { mime: output.mime, data: output.data };
          if (cell && item) { cell.output.push({ mime: item.mime, data: item.data }); renderCells(); }
        } else if (finished && (finished.cell_id || finished.cellId || finished.result)) {
          // Try to infer which cell finished; update running=false and error/meta
          let raw = finished.cell_id || finished.cellId;
          if (!raw && finished.result) raw = finished.result.cell_id || finished.result.cellId;
          if (raw) {
            const cid = normId(raw);
            const cell = notebook.cells.find(c => normId(c.id) === cid);
            if (cell) {
              const meta = Object.assign({}, cell.meta, { running: false });
              if (finished.result && (finished.result.error || finished.result.duration_ms != null)) {
                if (finished.result.error) meta.error = finished.result.error;
                if (finished.result.duration_ms != null) meta.timingMs = finished.result.duration_ms;
              }
              cell.meta = meta;
            }
            // remove from runQueue for this cell
            const i = runQueue.findIndex(x=>x.cid===cid); if (i>=0) runQueue.splice(i,1);
          } else if (Array.isArray(notebook?.cells)) {
            // Fallback: clear running on all (linear exec)
            notebook.cells.forEach(c => { if (c.meta && c.meta.running) c.meta.running = false; });
            runQueue.length = 0;
          }
          if (raw) removeRunning(normId(raw)); else runningCells.clear();
          renderCells();
          renderSidebar();
        } else if (errorEvt && (errorEvt.cell_id || errorEvt.cellId)) {
          // Error event: stop spinner and surface error
          const raw = errorEvt.cell_id || errorEvt.cellId; const cid = normId(raw);
          const cell = notebook.cells.find(c => normId(c.id) === cid);
          if (cell) {
            const meta = Object.assign({}, cell.meta, { running: false, error: errorEvt.message || 'Error' });
            cell.meta = meta;
            // ensure runQueue cleanup
            const i = runQueue.findIndex(x=>x.cid===cid); if (i>=0) runQueue.splice(i,1);
          }
          removeRunning(cid);
          renderCells();
          renderSidebar();
        }
      } catch (e) { console.error('event err', e); }
    });
  }

  byId('openBtn').onclick = async () => { await ensureSubscribed(); await openNotebook(); };
  byId('newBtn').onclick = async () => {
    await ensureSubscribed();
    setStatus('Creating new notebook...');
    try {
      const title = byId('newTitle').value.trim();
      const res = await invoke('cmd_new_notebook', { title: title.length ? title : null });
      sessionId = res.session_id || res.sessionId;
      notebook = res.notebook;
      byId('saveBtn').disabled = false;
      // initialize run state for new notebook (session-scoped)
      initHashes(); saveRunState();
      renderCells();
      setStatus('New notebook ready');
    } catch (e) {
      console.error(e);
      setStatus('Error: ' + e);
    }
  };
  byId('saveBtn').onclick = saveNotebook;
  byId('addCodeBtn').onclick = addCodeCell;
  const addTextBtn = byId('addTextBtn'); if (addTextBtn) addTextBtn.onclick = addTextCell;
  byId('runAllBtn').onclick = confirmAndRunAll;
  byId('stopBtn').onclick = stop;
  const runImpBtn = byId('runImpBtn'); if (runImpBtn) runImpBtn.onclick = runImpacted;
  // Settings overlay
  const settingsBtn = byId('settingsBtn');
  if (settingsBtn){
    settingsBtn.onclick = () => {
      const root = byId('settings'); if (!root) return;
      const close = () => { root.hidden = true; };
      root.hidden = false;
      const sizesIn = byId('previewSizesInput');
      const defSel = byId('previewDefaultSelect');
      const render = () => {
        const sizes = loadPreviewSizes();
        sizesIn.value = sizes.join(',');
        defSel.innerHTML = '';
        sizes.forEach(v => { const o=document.createElement('option'); o.value=String(v); o.textContent=String(v); defSel.appendChild(o); });
        const def = loadPreviewDefault();
        defSel.value = String(def);
      };
      render();
      const saveBtn = byId('settingsSave');
      const closeBtn = byId('settingsClose');
      if (closeBtn) closeBtn.onclick = close;
      const backdrop = root.querySelector('.viewer-backdrop'); if (backdrop) backdrop.onclick = close;
      if (saveBtn) saveBtn.onclick = () => {
        const raw = (sizesIn.value || '').split(',').map(s=>parseInt(s.trim(),10)).filter(n=>Number.isFinite(n) && n>0);
        const uniq = Array.from(new Set(raw)).slice(0,8).sort((a,b)=>a-b);
        const def = parseInt(defSel.value, 10);
        const defFixed = uniq.includes(def) ? def : (uniq[0] || 200);
        if (uniq.length === 0) { savePreviewSettings([50,200,1000], 200); }
        else { savePreviewSettings(uniq, defFixed); }
        setStatus('Settings saved');
        close();
      };
    };
  }

  // Theme toggle
  const themeT = byId('themeToggle');
  const storedTheme = localStorage.getItem('lyra_theme');
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  const initialMode = storedTheme ? storedTheme : (prefersDark ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme', initialMode);
  themeT.checked = (initialMode === 'dark');
  themeT.onchange = () => {
    const mode = themeT.checked ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', mode);
    localStorage.setItem('lyra_theme', mode);
  };

  // Density toggle
  const densitySel = byId('densitySelect');
  const storedDensity = localStorage.getItem('lyra_density');
  const initialDensity = storedDensity || 'comfortable';
  document.documentElement.setAttribute('data-density', initialDensity);
  if (densitySel) densitySel.value = initialDensity;
  if (densitySel) densitySel.onchange = () => {
    const val = densitySel.value === 'compact' ? 'compact' : 'comfortable';
    document.documentElement.setAttribute('data-density', val);
    localStorage.setItem('lyra_density', val);
    updateToolbarCompact();
  };

  // Compact toolbar mode: apply on small screens or compact density
  function updateToolbarCompact(){
    const compact = document.documentElement.getAttribute('data-density') === 'compact' || window.innerWidth < 900;
    const headerTb = document.querySelector('header .toolbar');
    const mainTb = document.querySelector('main > .toolbar');
    [headerTb, mainTb].forEach(el => { if (!el) return; el.classList.toggle('compact', compact); el.classList.add('scrollable'); updateToolbarShadows(el); });
  }
  function updateToolbarShadows(el){
    if (!el) return;
    const sc = el.scrollLeft;
    const max = el.scrollWidth - el.clientWidth - 1;
    el.classList.toggle('has-left-shadow', sc > 0);
    el.classList.toggle('has-right-shadow', sc < max);
  }
  function attachToolbarScroll(){
    const tb = document.querySelector('main > .toolbar');
    if (!tb) return;
    tb.addEventListener('scroll', () => updateToolbarShadows(tb));
    const ro = new ResizeObserver(() => updateToolbarShadows(tb));
    ro.observe(tb);
    updateToolbarShadows(tb);
  }
  window.addEventListener('resize', updateToolbarCompact);
  updateToolbarCompact();
  attachToolbarScroll();

  // Generic scrollshadow helpers
  function updateScrollShadow(hostEl, scroller){
    if (!hostEl || !scroller) return;
    const sc = scroller.scrollLeft;
    const max = scroller.scrollWidth - scroller.clientWidth - 1;
    hostEl.classList.toggle('has-left-shadow', sc > 0);
    hostEl.classList.toggle('has-right-shadow', sc < max);
  }
  function attachScrollShadow(scroller, hostEl){
    if (!scroller || !hostEl) return;
    hostEl.classList.add('scrollshadow');
    const onScroll = () => updateScrollShadow(hostEl, scroller);
    scroller.addEventListener('scroll', onScroll);
    const ro = new ResizeObserver(onScroll);
    ro.observe(scroller);
    onScroll();
  }
  function debounce(fn, ms) {
    let t = null; return function(...args){ clearTimeout(t); t=setTimeout(()=>fn.apply(this,args), ms); };
  }

  // Hover docs tooltip for editor tokens
  let docTip = null; let docHoverTimer = null;
  function ensureDocTip(){ if (docTip) return docTip; const d = document.createElement('div'); d.className='doc-tip'; d.style.display='none'; document.body.appendChild(d); d.addEventListener('click', (ev) => { const t = ev.target; if (t && t.getAttribute && t.getAttribute('data-doc-symbol')){ const sym = t.getAttribute('data-doc-symbol'); if (sym){ setCurrentDocSymbol(sym); if (typeof setRightOpen === 'function') setRightOpen(true); hideDocTip(); } } }); docTip = d; return d; }
  function showDocTipAt(x,y,html){ const d=ensureDocTip(); d.innerHTML = html; d.style.display='block'; d.style.left = (x + 12) + 'px'; d.style.top = (y + 12) + 'px'; }
  function hideDocTip(){ if (docTip) docTip.style.display='none'; }

  // Auto-fix trivial issues in the active editor (palette command)
  async function autoFixTrivialActiveEditor(){
    if (!activeEditor || !activeEditor.textarea){ setStatus('No active editor'); return; }
    const ta = activeEditor.textarea;
    const offsetFromLineColLocal = (text, line, col) => { let l=0, off=0; for (let i=0;i<text.length;i++){ if (l===line){ off = i; break; } if (text[i]==='\n') l++; } if (l < line) off = text.length; return Math.min(text.length, off + col); };
    const computeFixes = (msg) => {
      const lower = (msg||'').toLowerCase(); const out = [];
      if ((lower.includes('expect') || lower.includes('expected')) && lower.includes("]")) out.push({ kind:'insert', text:']' });
      if ((lower.includes('expect') || lower.includes('expected')) && lower.includes(")")) out.push({ kind:'insert', text:')' });
      if ((lower.includes('expect') || lower.includes('expected')) && lower.includes(",")) out.push({ kind:'insert', text:',' });
      if (lower.includes('string') && (lower.includes('unclosed') || lower.includes('unterminated'))) out.push({ kind:'quote' });
      return out;
    };
    let applied = 0; let rounds = 0;
    while (rounds < 10){
      let diags = [];
      try { diags = await invoke('cmd_editor_diagnostics', { text: ta.value }); } catch(_) { break; }
      if (!Array.isArray(diags) || diags.length === 0) break;
      const d = diags[0]; const fixes = computeFixes(d?.message||''); if (!fixes.length) break;
      const fx = fixes[0]; const v = ta.value; const off = offsetFromLineColLocal(v, d.start_line|0, d.start_col|0);
      if (fx.kind === 'insert'){
        ta.value = v.slice(0, off) + fx.text + v.slice(off);
        ta.selectionStart = ta.selectionEnd = off + fx.text.length;
      } else if (fx.kind === 'quote'){
        const s = ta.selectionStart, e = ta.selectionEnd;
        if (e > s){ ta.value = v.slice(0,s) + '"' + v.slice(s,e) + '"' + v.slice(e); ta.selectionStart = s; ta.selectionEnd = e + 2; }
        else { ta.value = v.slice(0, off) + '""' + v.slice(off); ta.selectionStart = ta.selectionEnd = off + 1; }
      }
      ta.dispatchEvent(new Event('input'));
      applied++; rounds++;
    }
    setStatus(applied ? `Applied ${applied} quick fix${applied>1?'es':''}` : 'No trivial fixes found');
  }

  // Quick Doc hotkey at caret (Cmd/Ctrl+I)
  window.addEventListener('keydown', async (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase()==='i'){
      const ed = activeEditor; if (!ed || !ed.textarea) return;
      e.preventDefault();
      const v = ed.textarea.value; const pos = ed.textarea.selectionStart;
      let i=pos-1; while(i>=0 && /[A-Za-z0-9_]/.test(v[i])) i--; let j=pos; while(j<v.length && /[A-Za-z0-9_]/.test(v[j])) j++;
      const name = v.slice(i+1, j);
      if (!name) return;
      try { const doc = await invoke('cmd_editor_doc', { name }); docsCache.set(name, doc || null); const usage = (doc && doc.params && doc.params.length)? `${name}[${doc.params.join(', ')}]`:`${name}[]`; const summary = doc && doc.summary ? doc.summary : '—'; const c = caretClient(ed.textarea); showDocTipAt(c.clientX, c.clientY, `<div class=\"usage\">${usage}</div><div>${escapeHtml(summary)}</div>`); await setCurrentDocSymbol(name); } catch(_){ }
    }
  });

  // --- Lyra Editor ---
  function escapeHtml(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
  function tokenizeLyra(src){
    let i=0, out=''; const n=src.length;
    const emit=(cls,txt)=>{ out += cls?`<span class="${cls}">${escapeHtml(txt)}</span>`:escapeHtml(txt); };
    while(i<n){
      const ch = src[i];
      if (ch==='(' && i+1<n && src[i+1]==='*'){
        let depth=1; let j=i+2; while(j<n && depth>0){ if (src[j]==='(' && j+1<n && src[j+1]==='*'){ depth++; j+=2; continue;} if (src[j]==='*' && j+1<n && src[j+1]===')'){ depth--; j+=2; continue;} j++; }
        emit('tok-comment', src.slice(i,j)); i=j; continue;
      }
      if (ch==='"'){
        let j=i+1; while(j<n){ if (src[j]==='"' && src[j-1]!=="\\"){ j++; break;} j++; }
        emit('tok-string', src.slice(i,j)); i=j; continue;
      }
      if (/[0-9]/.test(ch)){
        let j=i+1; while(j<n && /[0-9\.]/.test(src[j])) j++;
        emit('tok-number', src.slice(i,j)); i=j; continue;
      }
      if ('[]{}()'.includes(ch)) { emit('tok-bracket', ch); i++; continue; }
      if (/[A-Za-z_]/.test(ch)){
        let j=i+1; while(j<n && /[A-Za-z0-9_]/.test(src[j])) j++;
        const word = src.slice(i,j);
        if (BUILTINS.has(word)) emit('tok-builtin', word); else emit('tok-symbol', word);
        i=j; continue;
      }
      emit('', ch); i++;
    }
    return out;
  }
  function createLyraEditor(cellId, textarea, highlightDiv, diagDiv){
    // attach scroll shadows on editor wrapper
    const wrapEl = highlightDiv && highlightDiv.parentElement ? highlightDiv.parentElement : null;
    if (wrapEl) attachScrollShadow(textarea, wrapEl);
    // autocomplete UI
    let acOpen = false; let acIdx = 0; let acItems = []; let acDocs = new Map();
    let acWrap = null; let acBox = null; let acDocEl = null;
    if (wrapEl){
      acWrap = document.createElement('div'); acWrap.className='ac-wrap'; acWrap.style.display='none';
      acBox = document.createElement('div'); acBox.className='ac-box';
      acDocEl = document.createElement('div'); acDocEl.className='ac-doc';
      acWrap.appendChild(acBox); acWrap.appendChild(acDocEl); wrapEl.appendChild(acWrap);
    }
    // diagnostics overlay layer
    let diagLayer = null;
    if (wrapEl){ diagLayer = document.createElement('div'); diagLayer.className='diag-layer'; wrapEl.appendChild(diagLayer); }
    // signature help UI
    let sigWrap = null; let sigBox = null;
    if (wrapEl){ sigWrap = document.createElement('div'); sigWrap.className='sig-wrap'; sigWrap.style.display='none'; sigBox = document.createElement('div'); sigBox.className='sig-box'; sigWrap.appendChild(sigBox); wrapEl.appendChild(sigWrap); }

    function getWordBeforeCaret(){
      const pos = textarea.selectionStart;
      const v = textarea.value;
      let i = pos-1; while(i>=0 && /[A-Za-z0-9_]/.test(v[i])) i--;
      const start = i+1; const word = v.slice(start, pos);
      return { start, end: pos, word };
    }
    function positionAC(){ if (!acWrap) return; const c = caretClient(textarea); const wrect = wrapEl.getBoundingClientRect(); acWrap.style.left = (c.clientX - wrect.left) + 'px'; acWrap.style.top = (c.clientY - wrect.top + c.line + 6) + 'px'; acWrap.style.right = 'auto'; acWrap.style.bottom = 'auto'; }
    function openAC(list){ if (!acWrap) return; acOpen = true; acIdx = 0; acItems = list.slice(0,100); renderAC(); positionAC(); acWrap.style.display='block'; updateDoc(); }
    function closeAC(){ acOpen = false; acItems = []; if (acWrap) acWrap.style.display='none'; }
    function renderAC(){ if (!acBox) return; acBox.innerHTML=''; acItems.forEach((name, i)=>{ const d=document.createElement('div'); d.className='ac-item'+(i===acIdx?' active':''); d.textContent=name; d.onclick=()=>{ applyAC(name); }; acBox.appendChild(d); }); }
    async function updateDoc(){ if (!acDocEl) return; const name = acItems[acIdx]; if (!name){ acDocEl.textContent=''; return; } try { if (!acDocs.has(name)) { const doc = await invoke('cmd_editor_doc', { name }); acDocs.set(name, doc || null); } const d = acDocs.get(name); if (d) { const usage = d.params && d.params.length ? `${name}[${d.params.join(', ')}]` : `${name}[]`; const summary = d.summary || '—'; acDocEl.innerHTML = `<div class=\"usage\">${usage}</div><div class=\"summary\">${escapeHtml(summary)}</div><div style=\"margin-top:6px\"><button class=\"btn xs link\" data-doc-symbol=\"${name}\">Open docs</button></div>`; } else { acDocEl.textContent='No documentation found'; } } catch(_){ acDocEl.textContent=''; }
    }
    async function applyAC(name){
      const { start, end } = getWordBeforeCaret();
      const v = textarea.value;
      let insertText = name; let selStartOffset = 0; let selEndOffset = 0;
      try { const doc = await invoke('cmd_editor_doc', { name }); if (doc && Array.isArray(doc.params)) { const params = doc.params; const inside = params.join(', '); insertText = `${name}[${inside}]`; if (params.length > 0){ selStartOffset = name.length + 1; selEndOffset = selStartOffset + String(params[0]).length; } else { selStartOffset = selEndOffset = name.length + 1; } } } catch(_){}
      const before = v.slice(0,start), after = v.slice(end);
      textarea.value = before + insertText + after;
      const selStart = start + (selStartOffset || insertText.length);
      const selEnd = start + (selEndOffset || insertText.length);
      textarea.selectionStart = selStart; textarea.selectionEnd = selEnd;
      refresh(); updateModel(); closeAC();
    }
    function unique(list){ const s=new Set(); const out=[]; for(const x of list){ if(!s.has(x)){ s.add(x); out.push(x);} } return out; }
    function score(name, q){ if (name.startsWith(q)) return 100 - (name.length - q.length); const i = name.indexOf(q); return i>=0 ? 50 - i : -1; }
    function collectLocalSymbols(text){ const out = new Set(); try { const reFn = /\b([A-Za-z_][A-Za-z0-9_]*)\s*\[/g; let m; while((m = reFn.exec(text))){ out.add(m[1]); if (out.size > 200) break; } } catch(_){} try { const reLet = /\blet\s+([A-Za-z_][A-Za-z0-9_]*)/g; let m; while((m = reLet.exec(text))){ out.add(m[1]); if (out.size > 400) break; } } catch(_){} return Array.from(out); }
    function triggerAC(){ const { word } = getWordBeforeCaret(); if (!word || word.length===0){ closeAC(); return; } const q=word.toLowerCase(); const built = Array.from(BUILTINS); const local = collectLocalSymbols(textarea.value); const pool = unique([...KEYWORDS, ...local, ...built]); const ranked = pool.map(n=>[n, score(n.toLowerCase(), q)]).filter(([,s])=>s>=0).sort((a,b)=>b[1]-a[1]).map(([n])=>n); if (ranked.length>0) openAC(ranked); else closeAC(); }
    async function updateSignature(){
      const pos = textarea.selectionStart; const v = textarea.value;
      const lbr = v.lastIndexOf('[', pos-1); if (lbr < 0){ if (sigWrap) sigWrap.style.display='none'; return; }
      const rbr = v.indexOf(']', lbr+1); if (rbr >= 0 && rbr < pos){ if (sigWrap) sigWrap.style.display='none'; return; }
      let i=lbr-1; while(i>=0 && /\s/.test(v[i])) i--; let j=i; while(j>=0 && /[A-Za-z0-9_]/.test(v[j])) j--; const name = v.slice(j+1, i+1);
      if (!name){ if (sigWrap) sigWrap.style.display='none'; return; }
      let params=[]; try { const doc = await invoke('cmd_editor_doc', { name }); if (doc && Array.isArray(doc.params)) params = doc.params; } catch(_){ }
      const inner = v.slice(lbr+1, pos); const commaCount = (inner.match(/,/g)||[]).length; const idx = Math.min(commaCount, Math.max(0, params.length-1));
      const usage = params.length? `${name}[${params.map((p,k)=>`<span class=\"param${k===idx?' active':''}\">${p}</span>`).join(', ')}]` : `${name}[]`;
      if (sigBox){ sigBox.innerHTML = `<div class="usage">${usage}</div>`; if (sigWrap) sigWrap.style.display='block'; }
    }
    function offsetFromLineCol(text, line, col){
      let l=0, off=0; for (let i=0;i<text.length;i++){ if (l===line){ off = i; break; } if (text[i]==='\n') l++; }
      // if line exceeds length, clamp to end
      if (l < line) off = text.length;
      return Math.min(text.length, off + col);
    }
    function clientForOffset(textarea, offset){
      const rect = textarea.getBoundingClientRect();
      const style = window.getComputedStyle(textarea);
      const div = document.createElement('div');
      div.style.position = 'fixed';
      div.style.left = rect.left + 'px';
      div.style.top = rect.top + 'px';
      div.style.width = rect.width + 'px';
      div.style.whiteSpace = 'pre-wrap';
      div.style.wordWrap = 'break-word';
      div.style.visibility = 'hidden';
      const props = ['fontFamily','fontSize','lineHeight','letterSpacing','tabSize','paddingTop','paddingBottom','paddingLeft','paddingRight','boxSizing'];
      props.forEach(p => { div.style[p.replace(/[A-Z]/g, m=>'-'+m.toLowerCase())] = style[p]; });
      const before = textarea.value.substring(0, offset);
      const span = document.createElement('span'); span.textContent = '\u200b';
      const escape = (s) => s.replace(/&/g,'&amp;').replace(/</g,'&lt;');
      div.innerHTML = escape(before).replace(/  /g,' &nbsp;').replace(/\n/g, '<br>');
      div.appendChild(span);
      document.body.appendChild(div);
      const srect = span.getBoundingClientRect();
      const lh = parseFloat(style.lineHeight) || 16;
      document.body.removeChild(div);
      return { clientX: srect.left - textarea.scrollLeft, clientY: srect.top - textarea.scrollTop, line: lh };
    }
    function computeQuickFixes(msg){
      const lower = (msg||'').toLowerCase();
      const fixes = [];
      if ((lower.includes('expect') || lower.includes('expected')) && lower.includes("]")) fixes.push({ kind: 'insert', text: ']' });
      if ((lower.includes('expect') || lower.includes('expected')) && lower.includes(")")) fixes.push({ kind: 'insert', text: ')' });
      if ((lower.includes('expect') || lower.includes('expected')) && lower.includes(",")) fixes.push({ kind: 'insert', text: ',' });
      if (lower.includes('string') && (lower.includes('unclosed') || lower.includes('unterminated'))) fixes.push({ kind: 'quote' });
      return fixes;
    }
    function drawDiagUnderline(diag){
      if (!diagLayer){ return; }
      diagLayer.innerHTML = '';
      if (!diag || typeof diag.start_line!=='number') return;
      const startLine = diag.start_line|0, endLine = (diag.end_line!=null ? diag.end_line|0 : startLine);
      const startCol = diag.start_col|0, endCol = (diag.end_col!=null ? diag.end_col|0 : startCol+1);
      const wrect = wrapEl.getBoundingClientRect();
      for (let ln = startLine; ln <= endLine; ln++){
        const fromCol = (ln === startLine) ? startCol : 0;
        // compute end col: if middle line, underline to end of line; approximate by large col
        const toCol = (ln === endLine) ? endCol : 9999;
        const startOff = offsetFromLineCol(textarea.value, ln, fromCol);
        const endOff = offsetFromLineCol(textarea.value, ln, toCol);
        const sPos = clientForOffset(textarea, startOff);
        const ePos = clientForOffset(textarea, Math.max(startOff+1, endOff));
        const x1 = Math.max(0, sPos.clientX - wrect.left);
        const x2 = Math.max(0, ePos.clientX - wrect.left);
        const y = Math.max(0, (sPos.clientY - wrect.top) + sPos.line - 2);
        const u = document.createElement('div'); u.className='diag-uline';
        const sevRaw = (diag.severity||'Error')+''; const sev = sevRaw.toLowerCase();
        if (sev.startsWith('warn')) u.classList.add('warn'); else if (sev.startsWith('info')) u.classList.add('info');
        u.style.left = Math.min(x1,x2)+'px'; u.style.width = Math.max(2, Math.abs(x2-x1))+'px'; u.style.top = y+'px';
        const msg = (diag && diag.message) ? diag.message : sevRaw;
        u.setAttribute('title', msg);
        const tagCls = sev.startsWith('warn') ? 'warn' : (sev.startsWith('info') ? 'info' : 'error');
        const loc = (ln===startLine && ln===endLine)
          ? `Ln ${ln+1}, Col ${startCol+1}–${endCol+1}`
          : (ln===startLine ? `Ln ${ln+1}, Col ${startCol+1}–`
            : (ln===endLine ? `Ln ${ln+1}, Col –${endCol+1}` : `Ln ${ln+1}`));
        u.addEventListener('mouseenter', (e) => { showDocTipAt(e.clientX, e.clientY, `<div><span class=\"tag ${tagCls}\">${sevRaw}</span> <span class=\"sub\">${loc}</span></div><div>${escapeHtml(msg)}</div>`); });
        u.addEventListener('mouseleave', () => { hideDocTip(); });
        u.addEventListener('click', (e) => {
          e.stopPropagation();
          const fixes = computeQuickFixes(msg);
          if (!fixes.length) return;
          const tip = ensureDocTip();
          // Build content with buttons
          tip.innerHTML = '';
          const head = document.createElement('div'); head.innerHTML = `<span class=\"tag ${tagCls}\">${sevRaw}</span> <span class=\"sub\">${loc}</span>`; tip.appendChild(head);
          const title = document.createElement('div'); title.style.margin='6px 0'; title.textContent = 'Quick fixes'; tip.appendChild(title);
          fixes.slice(0,3).forEach((fxSpec, idx) => {
            const b = document.createElement('button'); b.className='btn xs'; b.style.marginRight='6px'; b.textContent = (fxSpec.kind==='insert'?`Insert \"${fxSpec.text}\"`:(fxSpec.kind==='quote'?'Wrap in quotes':'Fix'));
            b.onclick = () => {
              const v = textarea.value;
              const off = offsetFromLineCol(v, startLine, startCol);
              if (fxSpec.kind === 'insert'){
                const ins = fxSpec.text;
                textarea.value = v.slice(0, off) + ins + v.slice(off);
                textarea.selectionStart = textarea.selectionEnd = off + ins.length;
              } else if (fxSpec.kind === 'quote'){
                const s = textarea.selectionStart, e = textarea.selectionEnd;
                if (e > s){ textarea.value = v.slice(0,s) + '"' + v.slice(s,e) + '"' + v.slice(e); textarea.selectionStart = s; textarea.selectionEnd = e + 2; }
                else { textarea.value = v.slice(0, off) + '""' + v.slice(off); textarea.selectionStart = textarea.selectionEnd = off + 1; }
              }
              textarea.dispatchEvent(new Event('input'));
              hideDocTip();
            };
            tip.appendChild(b);
          });
          tip.style.display='block'; tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY + 12) + 'px';
        });
        diagLayer.appendChild(u);
        if (ln === endLine) break;
      }
    }
    const updateModel = debounce(() => {
      const c = notebook.cells.find(c=>normId(c.id)===cellId);
      if (c) c.input = textarea.value;
      // mark dirty if content hash changed
      const hNew = hashText(textarea.value||''); const prev = lastRunHash.get(cellId);
      if (prev !== hNew) dirtyCells.add(cellId); else dirtyCells.delete(cellId);
      saveRunState(); updateImpactUI();
      if (sessionId) invoke('cmd_update_session_notebook', { sessionId: sessionId, notebook }).catch(console.error);
      invoke('cmd_editor_diagnostics', { text: textarea.value }).then(diags => {
        const d = Array.isArray(diags) && diags.length ? diags[0] : null;
        const msg = d ? d.message : '';
        const sevRaw = (d?.severity || 'Error') + '';
        const sev = sevRaw.toLowerCase(); // error | warning | info (normalize)
        if (diagDiv){
          diagDiv.classList.remove('info','warn','error');
          diagDiv.innerHTML = '';
          if (msg){
            const txt = document.createElement('span'); txt.textContent = `${sevRaw}: ${msg}`; diagDiv.appendChild(txt);
            // Quick fix chip(s) for common parse issues
            const fixes = computeQuickFixes(msg);
            if (fixes.length){
              fixes.slice(0,2).forEach((fxSpec, idx) => {
                const fx = document.createElement('button'); fx.className='btn xs link'; fx.style.marginLeft='8px'; fx.textContent = idx===0 ? 'Quick fix' : 'Fix 2';
                fx.onclick = () => {
                  const v = textarea.value;
                  const startLine = (d && typeof d.start_line==='number') ? d.start_line|0 : 0;
                  const startCol  = (d && typeof d.start_col==='number')  ? d.start_col|0  : 0;
                  const off = offsetFromLineCol(v, startLine, startCol);
                  if (fxSpec.kind === 'insert'){
                    const ins = fxSpec.text;
                    textarea.value = v.slice(0, off) + ins + v.slice(off);
                    textarea.selectionStart = textarea.selectionEnd = off + ins.length;
                  } else if (fxSpec.kind === 'quote'){
                    const s = textarea.selectionStart, e = textarea.selectionEnd;
                    if (e > s){ textarea.value = v.slice(0,s) + '"' + v.slice(s,e) + '"' + v.slice(e); textarea.selectionStart = s; textarea.selectionEnd = e + 2; }
                    else { textarea.value = v.slice(0, off) + '""' + v.slice(off); textarea.selectionStart = textarea.selectionEnd = off + 1; }
                  }
                  textarea.dispatchEvent(new Event('input'));
                };
                diagDiv.appendChild(fx);
              });
            }
            diagDiv.classList.add(sev.startsWith('warn')?'warn':(sev.startsWith('info')?'info':'error'));
          }
        }
        if (msg) problems.set(cellId, { message: msg, severity: (sev.startsWith('warn')?'warn':(sev.startsWith('info')?'info':'error')) }); else problems.delete(cellId);
        renderSidebar();
        updateTabs();
        drawDiagUnderline(d);
      }).catch(()=>{});
      // refresh dependency map
      analyzeNotebookDeps(); if (activeTab==='Deps') renderSidebar();
    }, 200);
    const refresh = () => { 
      highlightDiv.innerHTML = tokenizeLyra(textarea.value);
      // wire hover docs on tokens
      highlightDiv.onmouseover = (e) => {
        const t = e.target; if (!(t && t.classList)) return; if (!(t.classList.contains('tok-builtin') || t.classList.contains('tok-symbol'))) return;
        const name = t.textContent.trim(); if (!name) return;
        if (docHoverTimer) clearTimeout(docHoverTimer);
        const { clientX, clientY } = e;
        docHoverTimer = setTimeout(async () => {
          try { const doc = await invoke('cmd_editor_doc', { name });
            if (doc) {
              const usage = doc.params && doc.params.length ? `${name}[${doc.params.join(', ')}]` : `${name}[]`;
              const summary = doc.summary || '—';
              showDocTipAt(clientX, clientY, `<div class=\"usage\">${usage}</div><div>${escapeHtml(summary)}</div><div style=\"margin-top:6px\"><button class=\"btn xs link\" data-doc-symbol=\"${name}\">Open docs</button></div>`);
            }
          } catch(_){}
        }, 200);
      };
      highlightDiv.onmouseout = () => { if (docHoverTimer) clearTimeout(docHoverTimer); hideDocTip(); };
    };
    const updateCaretSymbol = debounce(() => {
      const pos = textarea.selectionStart; const v = textarea.value; let i=pos-1; while(i>=0 && /[A-Za-z0-9_]/.test(v[i])) i--; let j=pos; while(j<v.length && /[A-Za-z0-9_]/.test(v[j])) j++; const name = v.slice(i+1, j).trim(); if (name) setCurrentDocSymbol(name);
    }, 150);
    textarea.addEventListener('scroll', () => { highlightDiv.style.transform = `translate3d(${-textarea.scrollLeft}px, ${-textarea.scrollTop}px, 0)`; if (wrapEl) updateScrollShadow(wrapEl, textarea); if (acOpen) positionAC(); positionSig(); });
    textarea.addEventListener('input', () => { refresh(); updateModel(); triggerAC(); updateSignature(); if (acOpen) positionAC(); positionSig(); updateCaretSymbol(); });
    textarea.addEventListener('scroll', () => { /* keep underline positioned */ invoke('cmd_editor_diagnostics', { text: textarea.value }).then(diags => { const d = Array.isArray(diags) && diags.length ? diags[0] : null; drawDiagUnderline(d); }).catch(()=>{}); });
    textarea.addEventListener('keyup', (e) => { if (!e.ctrlKey && !e.metaKey) updateCaretSymbol(); });
    textarea.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); runCell(cellId); return; }
      if (e.shiftKey && e.key === 'Enter') { e.preventDefault(); runAndAdvance(cellId); return; }
      if (e.altKey && e.key === 'Enter') { e.preventDefault(); addCodeCellBelow(cellId); return; }
      // autocomplete
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase()===' ') { e.preventDefault(); triggerAC(); return; }
      if (acOpen){
        if (e.key === 'ArrowDown' || (e.ctrlKey && e.key.toLowerCase()==='n')){ e.preventDefault(); acIdx = Math.min(acItems.length-1, acIdx+1); renderAC(); updateDoc(); return; }
        if (e.key === 'ArrowUp' || (e.ctrlKey && e.key.toLowerCase()==='p')){ e.preventDefault(); acIdx = Math.max(0, acIdx-1); renderAC(); updateDoc(); return; }
        if (e.key === 'Enter' || e.key === 'Tab'){ e.preventDefault(); if (acItems[acIdx]) applyAC(acItems[acIdx]); return; }
        if (e.key === 'Escape'){ e.preventDefault(); closeAC(); return; }
      }
      if (e.key === 'Tab') { e.preventDefault(); const s=textarea.selectionStart; const v=textarea.value; textarea.value = v.slice(0,s)+"  "+v.slice(textarea.selectionEnd); textarea.selectionStart = textarea.selectionEnd = s+2; refresh(); updateModel(); return; }
      const pairs = { '(':')', '[':']', '{':'}', '"':'"' };
      if (pairs[e.key]) { e.preventDefault(); const s=textarea.selectionStart; const v=textarea.value; const close=pairs[e.key]; textarea.value = v.slice(0,s)+e.key+close+v.slice(textarea.selectionEnd); textarea.selectionStart = textarea.selectionEnd = s+1; refresh(); updateModel(); return; }
    });
    textarea.addEventListener('blur', () => { closeAC(); }, true);
    // Signature positioner
    function positionSig(){ if (!sigWrap || sigWrap.style.display==='none') return; const c = caretClient(textarea); const wrect = wrapEl.getBoundingClientRect(); sigWrap.style.left = (c.clientX - wrect.left) + 'px'; sigWrap.style.top = Math.max(0, (c.clientY - wrect.top) - 8 - sigWrap.offsetHeight) + 'px'; }
    return { refresh, focus: () => textarea.focus(), setSelectionFromLineCol: (line, col) => { const off = offsetFromLineCol(textarea.value, line|0, col|0); textarea.selectionStart = textarea.selectionEnd = off; textarea.focus(); } };
  }

  // Prefetch builtins for highlighting
  (async () => { try { const names = await invoke('cmd_editor_builtins'); BUILTINS = new Set(names || []); } catch(_){} })();
  // Initialize hashes
  initHashes();

  // Command palette
  let palOpen = false; let palWrap = null; let palInput = null; let palList = null; let palIdx = 0; let palItems = [];
  const COMMANDS = [
    { name: 'Run All', type: 'command', action: () => runAll() },
    { name: 'Stop', type: 'command', action: () => stop() },
    { name: 'Auto-fix trivial issues', type: 'command', action: () => autoFixTrivialActiveEditor() },
    { name: 'Run Selection', type: 'command', action: () => runSelection() },
    { name: 'Run To Cursor', type: 'command', action: () => runToCursor() },
    { name: 'Add Code Cell', type: 'command', action: () => addCodeCell() },
    { name: 'Add Text Cell', type: 'command', action: () => addTextCell() },
    { name: 'Save Notebook', type: 'command', action: () => saveNotebook() },
    { name: 'Toggle Sidebar', type: 'command', action: () => { const open = document.documentElement.getAttribute('data-left-open')==='true'; const btn = byId('toggleSidebarBtn'); if (btn) btn.click(); else document.documentElement.setAttribute('data-left-open', open? 'false':'true'); } },
  ];
  let palDoc = null;
  function buildPalette(){
    if (palWrap) return;
    const back = document.createElement('div'); back.className='pal-backdrop'; back.style.display='none'; document.body.appendChild(back);
    const pal = document.createElement('div'); pal.className='palette'; pal.style.display='none'; pal.setAttribute('role','dialog'); pal.setAttribute('aria-modal','true');
    const head = document.createElement('div'); head.className='pal-head';
    const inp = document.createElement('input'); inp.className='pal-input'; inp.placeholder='Search symbols, commands…'; head.appendChild(inp);
    const list = document.createElement('div'); list.className='pal-list'; list.setAttribute('role','listbox'); list.setAttribute('aria-label','Command Palette Results');
    const doc = document.createElement('div'); doc.className='pal-doc'; doc.textContent = 'Select a symbol to see docs';
    pal.appendChild(head); pal.appendChild(list); pal.appendChild(doc); document.body.appendChild(pal);
    palWrap = { back, pal, head, inp, list };
    palInput = inp; palList = list;
    palDoc = doc;
    back.onclick = closePalette;
    inp.addEventListener('keydown', onPaletteKey);
    inp.addEventListener('input', () => refreshPalette(inp.value.trim()));
  }
  function openPalette(){ buildPalette(); palOpen = true; palIdx=0; palItems=[]; palWrap.back.style.display='block'; palWrap.pal.style.display='block'; palInput.value=''; refreshPalette(''); palInput.focus(); }
  function closePalette(){ if (!palOpen) return; palOpen=false; palWrap.back.style.display='none'; palWrap.pal.style.display='none'; }
  function onPaletteKey(e){
    if (e.key==='Escape'){ e.preventDefault(); closePalette(); return; }
    if (e.key==='ArrowDown'){ e.preventDefault(); palIdx=Math.min(palItems.length-1, palIdx+1); renderPalette(); return; }
    if (e.key==='ArrowUp'){ e.preventDefault(); palIdx=Math.max(0, palIdx-1); renderPalette(); return; }
    if (e.key==='Home'){ e.preventDefault(); palIdx=0; renderPalette(); return; }
    if (e.key==='End'){ e.preventDefault(); palIdx=Math.max(0, palItems.length-1); renderPalette(); return; }
    if (e.key==='PageDown'){ e.preventDefault(); palIdx=Math.min(palItems.length-1, palIdx+10); renderPalette(); return; }
    if (e.key==='PageUp'){ e.preventDefault(); palIdx=Math.max(0, palIdx-10); renderPalette(); return; }
    if (e.key==='Enter'){ e.preventDefault(); const it = palItems[palIdx]; if (it) activatePaletteItem(it); return; }
    if (e.key==='Tab'){ e.stopPropagation(); }
  }
  async function refreshPalette(q){
    const built = Array.from(BUILTINS);
    const poolSyms = Array.from(new Set([...KEYWORDS, ...built]));
    const symsRanked = poolSyms.map(n=>({ name:n, type:'symbol' })).filter(it=>fuzzyScore(it.name,q)>=0).sort((a,b)=>fuzzyScore(b.name,q)-fuzzyScore(a.name,q)).slice(0,80);
    const cmds = COMMANDS.filter(c=>fuzzyScore(c.name,q)>=0).sort((a,b)=>fuzzyScore(b.name,q)-fuzzyScore(a.name,q));
    palItems = [...cmds, ...symsRanked];
    renderPalette();
  }
  function fuzzyScore(name, q){ if (!q) return 1; const n=name.toLowerCase(), s=q.toLowerCase(); if (n.startsWith(s)) return 100 - (n.length - s.length); const i=n.indexOf(s); return i>=0? 50 - i : -1; }
  async function updatePaletteDoc(){
    if (!palDoc) return;
    const it = palItems[palIdx];
    if (!it){ palDoc.textContent = ''; return; }
    if (it.type === 'command'){
      palDoc.innerHTML = `<span class="tag command">command</span> <span class="usage">${it.name}</span>`;
      palDoc.scrollTop = 0;
      return;
    }
    const name = it.name;
    try {
      if (!docsCache.has(name)){
        const d = await invoke('cmd_editor_doc', { name }); docsCache.set(name, d || null);
      }
      const d = docsCache.get(name);
      if (d){
        const usage = (d.params && d.params.length) ? `${name}[${d.params.join(', ')}]` : `${name}[]`;
        const summary = d.summary || '';
        palDoc.innerHTML = `<span class="tag symbol">symbol</span> <div class="usage">${usage}</div><div>${escapeHtml(summary)}</div>`;
      } else {
        palDoc.innerHTML = `<span class="tag symbol">symbol</span> <div class="usage">${name}[]</div><div>No documentation.</div>`;
      }
      palDoc.scrollTop = 0;
    } catch(_){ palDoc.textContent = ''; }
  }
  function renderPalette(){
    if (!palList) return;
    palList.innerHTML='';
    palItems.forEach((it, i) => {
      const row = document.createElement('div');
      row.className = 'pal-item' + (i===palIdx ? ' active' : '');
      row.setAttribute('role','option');
      row.setAttribute('aria-selected', i===palIdx ? 'true' : 'false');
      row.tabIndex = 0;
      const left = document.createElement('div'); left.className='name'; left.textContent = it.name;
      const right = document.createElement('div'); right.className='meta'; const tag=document.createElement('span'); tag.className='tag '+(it.type==='command'?'command':'symbol'); tag.textContent = it.type; right.appendChild(tag);
      row.appendChild(left); row.appendChild(right);
      row.onclick = () => activatePaletteItem(it);
      row.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activatePaletteItem(it); } };
      palList.appendChild(row);
    });
    // ensure active item is visible
    const active = palList.children[palIdx]; if (active && active.scrollIntoView) active.scrollIntoView({ block: 'nearest' });
    updatePaletteDoc();
  }
  async function activatePaletteItem(it){ if (it.type==='command'){ it.action(); closePalette(); return; } // symbol
    // Insert usage into active editor
    const name = it.name; let usage = `${name}[]`;
    try { const doc = await invoke('cmd_editor_doc', { name }); if (doc && Array.isArray(doc.params)) { usage = doc.params.length? `${name}[${doc.params.join(', ')}]` : `${name}[]`; }} catch(_){}
    if (activeEditor && activeEditor.insert){ activeEditor.insert(usage); closePalette(); } else { setStatus('No active editor'); }
  }
  window.addEventListener('keydown', (e)=>{ if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase()==='k'){ e.preventDefault(); openPalette(); } });

  // Run Selection / Run To Cursor helpers
  async function runSelection(){
    if (!activeEditor || !activeEditor.textarea) { setStatus('No active editor'); return; }
    const ta = activeEditor.textarea; const sel = ta.value.slice(ta.selectionStart, ta.selectionEnd);
    if (!sel || !sel.trim()) { setStatus('No selection'); return; }
    await runTextInScope(sel);
  }
  async function runToCursor(){
    if (!activeEditor || !activeEditor.textarea) { setStatus('No active editor'); return; }
    const ta = activeEditor.textarea; const text = ta.value.slice(0, ta.selectionStart);
    if (!text || !text.trim()) { setStatus('Nothing before cursor'); return; }
    await runTextInScope(text);
  }
  async function runTextInScope(text){
    if (!sessionId) { setStatus('Open a notebook first'); return; }
    setStatus('Running selection...');
    try {
      const res = await invoke('cmd_execute_text', { sessionId, text });
      const outs = res && res.outputs ? res.outputs : [];
      // Append outputs to the active cell
      if (lastFocusCellId){
        const cell = notebook.cells.find(c => normId(c.id)===lastFocusCellId);
        if (cell){
          const existing = Array.isArray(cell.output) ? cell.output : [];
          cell.output = existing.concat(outs.map(it => ({ mime: it.mime, data: it.data })));
          renderCells();
        }
      }
      setStatus(res && res.error ? ('Error: ' + res.error) : 'Done');
    } catch (e) { console.error(e); setStatus('Error: ' + e); }
  }

  // Sidebar logic
  function setActiveTab(name){ activeTab = name; localStorage.setItem('lyra_sidebar_tab', name); updateTabs(); renderSidebar(); }
  function updateTabs(){
    const tabs = ['Outline','Queue','Problems','Deps'];
    const fmtCount = (n) => (n>99? '99+' : (n>0? (''+n) : ''));
    tabs.forEach(t => {
      const btn = byId('tab'+t); const panel = byId('panel'+t);
      if (!btn || !panel) return;
      const on = (activeTab === t);
      btn.classList.toggle('active', on);
      btn.setAttribute('aria-selected', on ? 'true' : 'false');
      btn.tabIndex = on ? 0 : -1;
      panel.hidden = !on;
      if (t === 'Problems'){
        const count = problems.size;
        const lab = fmtCount(count);
        btn.innerHTML = count > 0 ? `Problems <span class="count">${lab}</span>` : 'Problems';
      } else if (t === 'Outline'){
        btn.textContent = 'Outline';
      } else if (t === 'Queue'){
        btn.textContent = 'Queue';
      } else if (t === 'Problems'){
        // already handled above for count
      } else if (t === 'Deps'){
        btn.textContent = 'Deps';
      }
    });
    // Header problems indicator
    const pb = byId('problemsBtn');
    if (pb){
      const n = problems.size; const lab = fmtCount(n);
      pb.innerHTML = n>0 ? `Problems <span class="count">${lab}</span>` : 'Problems';
    }
    // Header status dot
    const sd = byId('statusDot');
    if (sd){
      const has = problems.size > 0;
      const known = !!notebook; // only show green when a notebook is loaded
      const busy = runningCells.size > 0;
      sd.classList.toggle('error', has);
      sd.classList.toggle('busy', !has && busy);
      sd.classList.toggle('ok', known && !has && !busy);
      const label = has ? 'Problems present' : (busy ? 'Kernel busy' : (known ? 'All clear' : 'No problems'));
      sd.setAttribute('aria-label', label);
      sd.setAttribute('title', label);
    }
  }
  // Tabs keyboard model: ArrowLeft/Right/Home/End
  function onTabKey(e){
    const order = ['Outline','Queue','Problems'];
    const current = activeTab;
    let idx = order.indexOf(current);
    if (idx < 0) idx = 0;
    if (e.key === 'ArrowRight') { e.preventDefault(); idx = (idx+1) % order.length; setActiveTab(order[idx]); byId('tab'+order[idx])?.focus(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); idx = (idx-1+order.length) % order.length; setActiveTab(order[idx]); byId('tab'+order[idx])?.focus(); }
    else if (e.key === 'Home') { e.preventDefault(); setActiveTab(order[0]); byId('tab'+order[0])?.focus(); }
    else if (e.key === 'End') { e.preventDefault(); setActiveTab(order[order.length-1]); byId('tab'+order[order.length-1])?.focus(); }
  }
  function renderSidebar(){
    // Outline
    const pO = byId('panelOutline');
    if (pO){
      pO.innerHTML='';
      if (notebook && Array.isArray(notebook.cells)){
        const rev = computeDependentsMap();
        notebook.cells.forEach((c, i) => {
          const cid = normId(c.id);
          const row = document.createElement('div');
          row.className = 'row'; row.tabIndex = 0; row.setAttribute('role','button'); row.setAttribute('aria-label', `Focus cell ${i+1}`);
          const state = (c.meta?.running) ? 'running' : ((c.meta?.error) ? 'error' : (Array.isArray(c.output) && c.output.length>0 ? 'ok' : 'idle'));
          const isCode = (c.type==='Code' && c.language==='Lyra');
          const hCur = isCode ? hashText(c.input||'') : 0;
          const hPrev = isCode ? (lastRunHash.get(cid)||null) : null;
          const statusChip = isCode ? (hPrev===null ? '' : (hCur===hPrev ? '<span class="tag info">cached</span>' : '<span class="tag warn">dirty</span>')) : '';
          const depCount = (rev.get(cid) || new Set()).size;
          const impChip = depCount > 0 ? `<span class="tag">${depCount}↓</span>` : '';
          row.innerHTML = `<span class="label">${c.type === 'Code' ? 'Code' : c.type} #${i+1}</span> <span class="sub">${c.language||''}</span> ${statusChip} ${impChip} ${state==='running'?'<span class="spinner" style="vertical-align:middle"></span>':''}`;
          row.style.cursor='pointer';
          const act = () => {
            const el = byId('cell-'+cid); if (el) { el.scrollIntoView({behavior:'smooth', block:'center'}); }
            const ed = editors.get(cid); if (ed && ed.focus) ed.focus();
          };
          row.onclick = act;
          row.onkeydown = (e) => {
            if (e.key==='Enter' || e.key===' ') { e.preventDefault(); act(); return; }
            if (!e.metaKey && !e.ctrlKey && e.altKey && e.shiftKey && (e.key==='ArrowUp' || e.key==='ArrowDown')){
              e.preventDefault(); moveCell(cid, e.key==='ArrowUp'?-1:1); return;
            }
          };
          pO.appendChild(row);
        });
      }
    }
    // Queue
    const pQ = byId('panelQueue');
    if (pQ){
      pQ.innerHTML='';
      if (runQueue.length===0){ pQ.textContent = 'No running cells'; }
      runQueue.forEach(item => {
        const row = document.createElement('div'); row.className='row'; row.tabIndex = 0; row.setAttribute('role','listitem');
        const dur = Math.max(0, Date.now()-item.startedAt);
        row.innerHTML = `<span class="label">Cell #${item.idx+1}</span> <span class="sub">${(dur/1000).toFixed(1)}s</span>`;
        pQ.appendChild(row);
      });
    }
    // Problems
    const pP = byId('panelProblems');
    if (pP){
      pP.innerHTML='';
      if (problems.size===0){ pP.textContent='No problems'; }
      problems.forEach((val, keyCid) => {
        const idx = notebook?.cells?.findIndex(c => normId(c.id)===keyCid) ?? -1;
        const row = document.createElement('div'); row.className='row'; row.tabIndex = 0; row.setAttribute('role','button');
        const sev = (val.severity||'error'); row.classList.add('sev-'+sev);
        row.innerHTML = `<span class=\"label\">Cell #${idx+1}</span> <span class=\"sub\">${val.message}</span>`;
        const act = () => { const el = byId('cell-'+keyCid); if (el) el.scrollIntoView({behavior:'smooth', block:'center'}); };
        row.onclick = act;
        row.onkeydown = (e) => {
          if (e.key==='Enter' || e.key===' ') { e.preventDefault(); act(); return; }
          if (!e.metaKey && !e.ctrlKey && e.altKey && e.shiftKey && (e.key==='ArrowUp' || e.key==='ArrowDown')){
            e.preventDefault(); moveCell(keyCid, e.key==='ArrowUp'?-1:1); return;
          }
        };
        pP.appendChild(row);
      });
    }
    // Deps
    const pD = byId('panelDeps');
    if (pD){
      pD.innerHTML='';
      analyzeNotebookDeps();
      if (!notebook || !Array.isArray(notebook.cells)) { pD.textContent='No notebook'; return; }
      notebook.cells.forEach((c, i) => {
        const cid = normId(c.id);
        const dep = cellDeps.get(cid) || { defines:new Set(), uses:new Set(), deps:new Set() };
        const row = document.createElement('div'); row.className='row'; row.tabIndex = 0; row.setAttribute('role','button');
        const defs = Array.from(dep.defines).slice(0,6).join(', ');
        const uses = Array.from(dep.uses).slice(0,6).join(', ');
        const deps = Array.from(dep.deps).map(id => notebook.cells.findIndex(x=>normId(x.id)===id)+1).filter(n=>n>0).slice(0,6).join(', ');
        row.innerHTML = `<span class="label">Cell #${i+1}</span> <span class="sub">defines: ${defs||'—'} | uses: ${uses||'—'} | deps: ${deps||'—'}</span>`;
        row.onclick = () => { const el = byId('cell-'+cid); if (el) el.scrollIntoView({behavior:'smooth', block:'center'}); };
        pD.appendChild(row);
      });
    }
  }

  // Sidebar init + events
  const tabOutline = byId('tabOutline'); const tabQueue = byId('tabQueue'); const tabProblems = byId('tabProblems'); const tabDeps = byId('tabDeps');
  if (tabOutline){ tabOutline.onclick = () => setActiveTab('Outline'); tabOutline.addEventListener('keydown', onTabKey); }
  if (tabQueue){ tabQueue.onclick = () => setActiveTab('Queue'); tabQueue.addEventListener('keydown', onTabKey); }
  if (tabProblems){ tabProblems.onclick = () => setActiveTab('Problems'); tabProblems.addEventListener('keydown', onTabKey); }
  if (tabDeps){ tabDeps.onclick = () => setActiveTab('Deps'); tabDeps.addEventListener('keydown', onTabKey); }
  activeTab = localStorage.getItem('lyra_sidebar_tab') || 'Outline';
  updateTabs();
  renderSidebar();

  // Resizer + persist width
  const sidebar = byId('leftSidebar');
  const resizer = byId('sidebarResizer');
  const backdrop = byId('leftBackdrop');
  const toggleSidebarBtn = byId('toggleSidebarBtn');
  const rightSidebar = byId('rightSidebar');
  const rightResizer = byId('rightResizer');
  const rightBackdrop = byId('rightBackdrop');
  const toggleDocsBtn = byId('toggleDocsBtn');
  const problemsBtn = byId('problemsBtn');
  function isMobile(){ return window.innerWidth <= 1000; }
  function setSidebarOpen(open){
    document.documentElement.setAttribute('data-left-open', open ? 'true' : 'false');
    localStorage.setItem('lyra_sidebar_open', open ? 'true' : 'false');
    updateSidebarBackdrop();
  }
  function updateSidebarBackdrop(){
    if (!backdrop) return;
    const open = document.documentElement.getAttribute('data-left-open')==='true';
    backdrop.hidden = !(open && isMobile());
  }
  function setRightOpen(open){
    document.documentElement.setAttribute('data-right-open', open ? 'true' : 'false');
    localStorage.setItem('lyra_right_open', open ? 'true' : 'false');
    if (rightBackdrop) rightBackdrop.hidden = !(open && isMobile());
  }
  function initRightFromStorage(){ const s = localStorage.getItem('lyra_right_open'); if (s !== null) document.documentElement.setAttribute('data-right-open', s==='true'?'true':'false'); if (rightBackdrop) rightBackdrop.hidden = !(document.documentElement.getAttribute('data-right-open')==='true' && isMobile()); }
  // initialize from storage
  const storedOpen = localStorage.getItem('lyra_sidebar_open');
  if (storedOpen !== null) document.documentElement.setAttribute('data-left-open', storedOpen==='true' ? 'true' : 'false');
  updateSidebarBackdrop();
  initRightFromStorage();
  let startX=0, startW=0, resizing=false;
  const storedW = parseInt(localStorage.getItem('lyra_sidebar_w')||'0',10);
  if (sidebar && storedW>0) sidebar.style.width = storedW+'px';
  if (resizer){
    resizer.addEventListener('mousedown', (e)=>{
      resizing=true; startX=e.clientX; startW=sidebar.getBoundingClientRect().width; document.body.style.userSelect='none';
    });
    window.addEventListener('mousemove', (e)=>{
      if(!resizing) return; const dx = e.clientX - startX; const w = Math.min(400, Math.max(200, startW + dx)); sidebar.style.width = w+'px';
    });
    window.addEventListener('mouseup', ()=>{ if(resizing){ resizing=false; document.body.style.userSelect=''; localStorage.setItem('lyra_sidebar_w', parseInt(sidebar.getBoundingClientRect().width)+''); }});
  }
  // Right resizer
  if (rightResizer){
    let rStartX=0, rStartW=0, rResizing=false;
    const storedW = parseInt(localStorage.getItem('lyra_right_w')||'0',10);
    if (rightSidebar && storedW>0) rightSidebar.style.width = storedW+'px';
    rightResizer.addEventListener('mousedown', (e)=>{ rResizing=true; rStartX=e.clientX; rStartW=rightSidebar.getBoundingClientRect().width; document.body.style.userSelect='none'; });
    window.addEventListener('mousemove', (e)=>{ if(!rResizing) return; const dx = rStartX - e.clientX; const w = Math.min(420, Math.max(220, rStartW + dx)); rightSidebar.style.width = w+'px'; });
    window.addEventListener('mouseup', ()=>{ if(rResizing){ rResizing=false; document.body.style.userSelect=''; localStorage.setItem('lyra_right_w', parseInt(rightSidebar.getBoundingClientRect().width)+''); }});
  }
  // Keyboard bindings (global)
  window.addEventListener('keydown', (e)=>{
    // Toggle left sidebar: Ctrl/Cmd+B
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase()==='b'){
      e.preventDefault(); const open = document.documentElement.getAttribute('data-left-open')==='true'; setSidebarOpen(!open); return;
    }
    // Toggle right docs: Ctrl/Cmd+J
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase()==='j'){
      e.preventDefault(); const open = document.documentElement.getAttribute('data-right-open')==='true'; setRightOpen(!open); return;
    }
    // Command palette: Ctrl/Cmd+K (already bound above)
    // Save: Ctrl/Cmd+S
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase()==='s'){
      e.preventDefault(); saveNotebook(); return;
    }
    // Run All: Ctrl/Cmd+Shift+Enter
    if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'Enter'){
      e.preventDefault(); runAll(); return;
    }
    // Stop: Ctrl/Cmd+.
    if ((e.metaKey || e.ctrlKey) && e.key === '.'){
      e.preventDefault(); stop(); return;
    }
    // Add Code Cell: Ctrl/Cmd+Alt+C
    if ((e.metaKey || e.ctrlKey) && e.altKey && e.key.toLowerCase()==='c'){
      e.preventDefault(); addCodeCell(); return;
    }
    // Add Text Cell: Ctrl/Cmd+Alt+T
    if ((e.metaKey || e.ctrlKey) && e.altKey && e.key.toLowerCase()==='t'){
      e.preventDefault(); addTextCell(); return;
    }
    // Delete focused cell: Ctrl/Cmd+Backspace/Delete
    if ((e.metaKey || e.ctrlKey) && (e.key === 'Backspace' || e.key === 'Delete')){
      if (!lastFocusCellId) return; e.preventDefault(); deleteCell(lastFocusCellId); return;
    }
    // Select previous/next cell: Alt+ArrowUp/Alt+ArrowDown
    if (e.altKey && !e.metaKey && !e.ctrlKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')){
      e.preventDefault();
      if (!notebook || !Array.isArray(notebook.cells) || notebook.cells.length===0) return;
      const idx = notebook.cells.findIndex(c => normId(c.id)===lastFocusCellId);
      const nextIdx = e.key==='ArrowDown' ? Math.min(notebook.cells.length-1, (idx<0?0:idx+1)) : Math.max(0, (idx<0?0:idx-1));
      const next = notebook.cells[nextIdx]; if (!next) return;
      const cid = normId(next.id);
      const el = byId('cell-'+cid); if (el) el.scrollIntoView({behavior:'smooth', block:'center'});
      const ed = editors.get(cid); if (ed && ed.focus) ed.focus(); else { const ta = el?.querySelector('textarea'); if (ta) ta.focus(); }
      lastFocusCellId = cid; return;
    }
    // Duplicate cell: Ctrl/Cmd+Shift+D
    if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase()==='d'){
      if (!lastFocusCellId) return; e.preventDefault(); duplicateCell(lastFocusCellId); return;
    }
    // Problems navigation: F8 (next), Shift+F8 (prev)
    if (e.key === 'F8'){
      e.preventDefault();
      jumpToProblem(e.shiftKey ? -1 : 1);
      return;
    }
    // Keyboard reorder: Alt+Shift+ArrowUp/ArrowDown
    if (!e.metaKey && !e.ctrlKey && e.altKey && e.shiftKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')){
      if (!lastFocusCellId) return; e.preventDefault();
      moveCell(lastFocusCellId, e.key === 'ArrowUp' ? -1 : 1);
      return;
    }
  });
  if (toggleSidebarBtn){ toggleSidebarBtn.onclick = () => { const open = document.documentElement.getAttribute('data-left-open')==='true'; setSidebarOpen(!open); }; }
  if (toggleDocsBtn){ toggleDocsBtn.onclick = () => { const open = document.documentElement.getAttribute('data-right-open')==='true'; setRightOpen(!open); }; }
  if (problemsBtn){ problemsBtn.onclick = () => { setSidebarOpen(true); setActiveTab('Problems'); }; }
  if (backdrop) backdrop.addEventListener('click', ()=>{ setSidebarOpen(false); });
  if (rightBackdrop) rightBackdrop.addEventListener('click', ()=>{ setRightOpen(false); });
  window.addEventListener('resize', () => { updateSidebarBackdrop(); if (rightBackdrop) rightBackdrop.hidden = !(document.documentElement.getAttribute('data-right-open')==='true' && isMobile()); });
})();
