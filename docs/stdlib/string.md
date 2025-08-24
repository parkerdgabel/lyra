# String Standard Library

A tour of string functions, interpolation, templating, and filters.

## Interpolation

- `StringInterpolate["sum={Total[{1,2,3}]}" ]` → `"sum=6"`
- `StringInterpolateWith["Hello \{name\}!", <|name->"Lyra"|>]` → `"Hello Lyra!"`
- `StringFormat["\{0\}-\{1\}", {"a", 123}]` → `"a-123"`
- `StringFormatMap["\{a\}:\{b\}", <|"a"->1, "b"->"x"|>]` → `"1:x"`

Notes:
- Lyra parses interpolation inside string literals at parse time for `{...}`. When passing templates at runtime, escape braces in the source: use `\{` and `\}` so the braces reach the function unchanged.

## Template Rendering (Mustache-like)

`TemplateRender[template, data, opts?]`

Supported features:
- Variables: `{{name}}` (HTML-escaped), unescaped: `{{{name}}}` or `{{& name}}`
- Sections: `{{#items}}...{{/items}}`
  - Lists iterate with each item as the current context (`{{.}}`)
  - Associations enter nested scope once if truthy
  - Truthy scalars render block once; falsy skip
- Inverted sections: `{{^empty}}...{{/empty}}`
- Partials: `{{>partial}}` via `opts = <|"Partials"-><|"partial"->"..."|>|>`
- Filters: `{{name|ToUpper}}` applies stdlib function to the resolved value
- Escaping: Variables are HTML-escaped by default. Disable globally with `opts = <|"EscapeHtml"->False|>` or use triple mustache.

Examples:
```lyra
TemplateRender["Hello, \{\{name\}\}!", <|"name"->"<Lyra>"|>]  // => "Hello, &lt;Lyra&gt;!"
TemplateRender["Hello, \{\{\{name\}\}\}!", <|"name"->"<Lyra>"|>] // => "Hello, <Lyra>!"
TemplateRender["\{\{#items\}\}\{\{.\}\}|\{\{/items\}\}", <|"items"->{"a","b"}|>] // => "a|b|"
TemplateRender["\{\{user.name\}\}", <|"user"-><|"name"->"Ann"|>|>] // => "Ann"
TemplateRender["X\{\{>p\}\}Y", <|"n"->"Z"|>, <|"Partials"-><|"p"->"\{\{n\}\}"|>|>] // => "XZY"
TemplateRender["\{\{name|ToUpper\}\}", <|"name"->"Lyra"|>] // => "LYRA"
```

## Core String Functions (highlights)

- Case: `ToUpper`, `ToLower`, `Capitalize`, `TitleCase`, `EqualsIgnoreCase`
- Length: `StringLength`
- Trim: `StringTrim`, `StringTrimLeft`, `StringTrimRight`, `StringTrimPrefix`, `StringTrimSuffix`, `StringTrimChars`
- Search: `StringContains`, `StartsWith`, `EndsWith`, `IndexOf`, `LastIndexOf`
- Split/Join: `StringSplit`, `StringJoin`, `StringJoinWith`, `SplitLines`, `JoinLines`
- Edit: `StringReplace`, `StringReplaceFirst`, `StringReverse`
- Padding: `StringPadLeft`, `StringPadRight`
- Chars: `StringChars`, `StringFromChars`
- Slices: `StringSlice[s, start, len?]` (UTF‑8 safe; negative start is from end)
- Repetition: `StringRepeat[s, n]`
- Blank: `IsBlank[s]`

## Filters & Utilities

- HTML:
  - `HtmlEscape[s]` → escape `& < > " '`  
  - `HtmlUnescape[s]`
- URL (RFC 3986):
  - `UrlEncode[s]`, `UrlDecode[s]`
- URL Form (application/x-www-form-urlencoded):
  - `UrlFormEncode[s]` (space → `+`), `UrlFormDecode[s]`
- JSON:
  - `JsonEscape[s]` (quotes, backslashes, control chars), `JsonUnescape[s]`
- Slug/case/formatting:
  - `Slugify[s]` → `"hello-world"`
  - `CamelCase[s]` → `"helloWorldTest"`
  - `SnakeCase[s]` → `"hello_world_test"`
  - `KebabCase[s]` → `"hello-world-test"`
  - `StringTruncate[s, max, suffix?]` (default suffix is `…`)

These filters can be called directly or used from `TemplateRender` via `{{value|Filter}}`.
