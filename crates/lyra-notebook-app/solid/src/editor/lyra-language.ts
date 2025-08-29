// Placeholder for Lyra CM6 language support. We will implement
// tokenizer, indentation, smart pairs, completion, hover, and diagnostics.
import type { Extension } from '@codemirror/state';
import { EditorState } from '@codemirror/state';
import { indentUnit, StreamLanguage, bracketMatching, indentOnInput, HighlightStyle, syntaxHighlighting, indentService } from '@codemirror/language';
import { closeBrackets } from '@codemirror/autocomplete';
import { tags as t } from '@lezer/highlight';

// Simple stream tokenizer for Lyra-like syntax
function lyraTokenizer() {
  const isWord = (ch: string) => /[A-Za-z_]/.test(ch);
  const isWordCont = (ch: string) => /[A-Za-z0-9_]/.test(ch);
  return {
    startState() {
      return { inString: false as false | '"', inComment: 0 };
    },
    token(stream: any, state: any) {
      if (state.inString) {
        while (!stream.eol()) {
          const ch = stream.next();
          if (ch === '\\') { stream.next(); continue; }
          if (ch === state.inString) { state.inString = false; break; }
        }
        return 'string';
      }
      if (state.inComment > 0) {
        while (!stream.eol()) {
          const ch = stream.next();
          if (ch === '*' && stream.peek() === ')') { stream.next(); state.inComment--; if (state.inComment < 0) state.inComment = 0; break; }
          if (ch === '(' && stream.peek() === '*') { stream.next(); state.inComment++; }
        }
        return 'comment';
      }
      if (stream.eatSpace()) return null;
      const ch = stream.next();
      // Comments (* ... *)
      if (ch === '(' && stream.peek() === '*') { stream.next(); state.inComment++; return 'comment'; }
      // Strings
      if (ch === '"') { state.inString = '"'; return 'string'; }
      // Numbers
      if (/[0-9]/.test(ch)) { stream.eatWhile(/[0-9_]/); if (stream.peek() === '.') { stream.next(); stream.eatWhile(/[0-9_]/); } return 'number'; }
      // Assignment :=
      if (ch === ':' && stream.peek() === '=') { stream.next(); return 'operator assign'; }
      // Guard /;  and other operators starting with /
      if (ch === '/' && (stream.peek() === ';' || stream.peek() === '/' || stream.peek() === '@' || stream.peek() === '.')) { stream.next(); return 'operator'; }
      // Brackets
      if ('[](){}'.includes(ch)) return 'bracket';
      // Names and function heads
      if (isWord(ch)) {
        stream.eatWhile(isWordCont);
        // Lookahead: if next non-space is '[' then treat as function head token
        const rest = stream.string.slice(stream.pos).trimStart();
        if (rest.startsWith('[')) return 'def';
        return 'variableName';
      }
      // Fallback: operators/punct
      return 'operator';
    }
  };
}

// Basic highlight theme for the tokens above
const lyraHighlight = HighlightStyle.define([
  { tag: t.definition(t.variableName), class: 'tok-def', color: '#c5d5ff' },
  { tag: t.variableName, color: '#e6e8ee' },
  { tag: t.string, color: '#9fe7c0' },
  { tag: t.number, color: '#ffd29a' },
  { tag: t.comment, color: '#8691a7' },
  { tag: t.operator, color: '#a7b1cc' },
  { tag: t.bracket, color: '#a7b1cc' },
]);

// Map simple style strings to tags for StreamLanguage
const lyra = StreamLanguage.define({
  token: lyraTokenizer().token,
  startState: lyraTokenizer().startState,
  languageData: {
    commentTokens: { block: { open: '(*', close: '*)' } },
    closeBrackets: { brackets: ['(', '[', '{', '"'] }
  }
});

// Indentation: 2 spaces per bracket depth; simple heuristic
function computeIndent(state: EditorState, pos: number) {
  const text = state.sliceDoc(0, pos);
  let depth = 0;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"') {
      i++;
      while (i < text.length) { if (text[i] === '\\') { i += 2; continue; } if (text[i] === '"') { break; } i++; }
    } else if (ch === '(' && text[i+1] === '*') {
      i += 2; while (i + 1 < text.length) { if (text[i] === '*' && text[i+1] === ')') { i += 1; break; } i++; }
    } else if (ch === '[' || ch === '(' || ch === '{') depth++;
    else if (ch === ']' || ch === ')' || ch === '}') depth = Math.max(0, depth - 1);
  }
  const unit = state.facet(indentUnit);
  return depth * unit.length;
}

export function lyraLanguage(): Extension {
  return [
    indentUnit.of('  '),
    lyra,
    bracketMatching(),
    indentOnInput(/^[\]\)\}]/),
    closeBrackets(),
    syntaxHighlighting(lyraHighlight),
    indentService.of((ctx) => computeIndent(ctx.state, ctx.pos)),
  ];
}
