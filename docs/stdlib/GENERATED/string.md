# STRING

| Function | Usage | Summary |
|---|---|---|
| `EndsWith` | `EndsWith[s, suffix]` | True if string ends with suffix |
| `HtmlEscape` | `HtmlEscape[s]` | Escape string for HTML |
| `HtmlUnescape` | `HtmlUnescape[s]` | Unescape HTML-escaped string |
| `JsonEscape` | `JsonEscape[s]` | Escape string for JSON |
| `JsonUnescape` | `JsonUnescape[s]` | Unescape JSON-escaped string |
| `RegexFind` | `RegexFind[s, pattern]` | Find first regex capture groups |
| `RegexFindAll` | `RegexFindAll[s, pattern]` | Find all regex capture groups |
| `RegexMatch` | `RegexMatch[s, pattern]` | Return first regex match |
| `RegexMatchQ` | `RegexMatchQ[pattern, s]` | Alias: regex predicate (Boolean) |
| `RegexReplace` | `RegexReplace[s, pattern, repl]` | Replace matches using regex |
| `Slugify` | `Slugify[s]` | Slugify for URLs |
| `StartsWith` | `StartsWith[s, prefix]` | True if string starts with prefix |
| `StringChars` | `StringChars[s]` | Split string into list of characters |
| `StringContains` | `StringContains[s, substr]` | Does string contain substring? |
| `StringFormat` | `StringFormat[fmt, args]` | Format using placeholders: {0}, {name} |
| `StringFormatMap` | `StringFormatMap[fmt, map]` | Format using map placeholders |
| `StringFromChars` | `StringFromChars[chars]` | Join list of characters into string |
| `StringInterpolate` | `StringInterpolate[fmt, map?]` | Interpolate ${var} from env or map |
| `StringInterpolateWith` | `StringInterpolateWith[fmt, resolver]` | Interpolate with custom resolver |
| `StringJoin` | `StringJoin[parts]` | Concatenate list of parts. |
| `StringJoinWith` | `StringJoinWith[parts, sep]` | Join strings with a separator |
| `StringLength` | `StringLength[s]` | Length of string (Unicode scalar count). |
| `StringPadLeft` | `StringPadLeft[s, width, pad?]` | Pad left to width with char |
| `StringPadRight` | `StringPadRight[s, width, pad?]` | Pad right to width with char |
| `StringQ` | `StringQ[x]` | Is value a string? |
| `StringRepeat` | `StringRepeat[s, n]` | Repeat string n times |
| `StringReplace` | `StringReplace[s, from, to]` | Replace all substring matches |
| `StringReplaceFirst` | `StringReplaceFirst[s, from, to]` | Replace first substring match |
| `StringReverse` | `StringReverse[s]` | Reverse characters in a string |
| `StringSlice` | `StringSlice[s, start, len?]` | Slice by start and optional length |
| `StringSplit` | `StringSplit[s, sep]` | Split string by separator |
| `StringTrim` | `StringTrim[s]` | Trim whitespace from both ends |
| `StringTrimChars` | `StringTrimChars[s, chars]` | Trim characters from ends |
| `StringTrimLeft` | `StringTrimLeft[s]` | Trim from left |
| `StringTrimPrefix` | `StringTrimPrefix[s, prefix]` | Remove prefix if present |
| `StringTrimRight` | `StringTrimRight[s]` | Trim from right |
| `StringTrimSuffix` | `StringTrimSuffix[s, suffix]` | Remove suffix if present |
| `StringTruncate` | `StringTruncate[s, len, ellipsis?]` | Truncate string to length |
| `ToDegrees` | `ToDegrees[x]` | Convert radians to degrees (Listable) |
| `ToLower` | `ToLower[s]` | Lowercase string. |
| `ToRadians` | `ToRadians[x]` | Convert degrees to radians (Listable) |
| `ToUpper` | `ToUpper[s]` | Uppercase string. |
| `ToolsCacheClear` | `ToolsCacheClear[]` | Clear tool registry caches. |
| `ToolsCards` | `ToolsCards[cursor?, limit?]` | Paginate tool cards for external UIs. |
| `ToolsDescribe` | `ToolsDescribe[id|name]` | Describe a tool by id or name. |
| `ToolsDryRun` | `ToolsDryRun[id|name, args]` | Validate a tool call and return normalized args and estimates. |
| `ToolsExportBundle` | `ToolsExportBundle[]` | Export all registered tool specs. |
| `ToolsExportOpenAI` | `ToolsExportOpenAI[]` | Export tools as OpenAI functions format. |
| `ToolsGetCapabilities` | `ToolsGetCapabilities[]` | Get current capabilities list. |
| `ToolsInvoke` | `ToolsInvoke[id|name, args?]` | Invoke a tool with an args assoc. |
| `ToolsList` | `ToolsList[]` | List available tools as cards. |
| `ToolsRegister` | `ToolsRegister[spec|list]` | Register one or more tool specs. |
| `ToolsResolve` | `ToolsResolve[pattern, topK?]` | Resolve tools matching a pattern. |
| `ToolsSearch` | `ToolsSearch[query, topK?]` | Search tools by name/summary. |
| `ToolsSetCapabilities` | `ToolsSetCapabilities[caps]` | Set allowed capabilities (e.g., net, fs). |
| `ToolsUnregister` | `ToolsUnregister[id|name]` | Unregister a tool by id or name. |
| `Top` | `Top[list, k, opts?]` | Take top-k items (optionally by key). |
| `TopologicalSort` | `TopologicalSort[graph]` | Topologically sort DAG nodes |
| `UrlDecode` | `UrlDecode[s]` | Decode percent-encoded string |
| `UrlEncode` | `UrlEncode[s]` | Percent-encode string for URLs |

## `EndsWith`

- Usage: `EndsWith[s, suffix]`
- Summary: True if string ends with suffix
- Tags: string, predicate
- Examples:
  - `EndsWith["foobar", "bar"]  ==> True`

## `RegexFindAll`

- Usage: `RegexFindAll[s, pattern]`
- Summary: Find all regex capture groups
- Tags: string, regex
- Examples:
  - `RegexFindAll[\"a1 b22\", \"\\d+\"]  ==> {\"1\",\"22\"}`

## `Slugify`

- Usage: `Slugify[s]`
- Summary: Slugify for URLs
- Tags: string, url
- Examples:
  - `Slugify["Hello, World!"]  ==> "hello-world"`

## `StartsWith`

- Usage: `StartsWith[s, prefix]`
- Summary: True if string starts with prefix
- Tags: string, predicate
- Examples:
  - `StartsWith["foobar", "foo"]  ==> True`

## `StringChars`

- Usage: `StringChars[s]`
- Summary: Split string into list of characters
- Examples:
  - `StringChars["abc"]  ==> {"a","b","c"}`

## `StringContains`

- Usage: `StringContains[s, substr]`
- Summary: Does string contain substring?
- Tags: string, predicate
- Examples:
  - `StringContains["hello", "ell"]  ==> True`

## `StringFromChars`

- Usage: `StringFromChars[chars]`
- Summary: Join list of characters into string
- Examples:
  - `StringFromChars[{"a","b"}]  ==> "ab"`

## `StringJoin`

- Usage: `StringJoin[parts]`
- Summary: Concatenate list of parts.
- Tags: string, text
- Examples:
  - `StringJoin[{"a","b","c"}]  ==> "abc"`

## `StringJoinWith`

- Usage: `StringJoinWith[parts, sep]`
- Summary: Join strings with a separator
- Examples:
  - `StringJoinWith[{"a","b"}, "-"]  ==> "a-b"`

## `StringLength`

- Usage: `StringLength[s]`
- Summary: Length of string (Unicode scalar count).
- Tags: string, text
- Examples:
  - `StringLength["hello"]  ==> 5`

## `StringPadLeft`

- Usage: `StringPadLeft[s, width, pad?]`
- Summary: Pad left to width with char
- Examples:
  - `StringPadLeft["7", 3, "0"]  ==> "007"`

## `StringReplace`

- Usage: `StringReplace[s, from, to]`
- Summary: Replace all substring matches
- Tags: string, replace
- Examples:
  - `StringReplace["foo bar", "o", "0"]  ==> "f00 bar"`

## `StringReverse`

- Usage: `StringReverse[s]`
- Summary: Reverse characters in a string
- Examples:
  - `StringReverse["abc"]  ==> "cba"`

## `StringSplit`

- Usage: `StringSplit[s, sep]`
- Summary: Split string by separator
- Tags: string, text
- Examples:
  - `StringSplit["a,b,c", ","]  ==> {"a","b","c"}`

## `StringTrim`

- Usage: `StringTrim[s]`
- Summary: Trim whitespace from both ends
- Tags: string, text
- Examples:
  - `StringTrim["  hi  "]  ==> "hi"`

## `ToDegrees`

- Usage: `ToDegrees[x]`
- Summary: Convert radians to degrees (Listable)
- Examples:
  - `ToDegrees[Pi]  ==> 180`

## `ToLower`

- Usage: `ToLower[s]`
- Summary: Lowercase string.
- Tags: string, text
- Examples:
  - `ToLower["Hello"]  ==> "hello"`

## `ToRadians`

- Usage: `ToRadians[x]`
- Summary: Convert degrees to radians (Listable)
- Examples:
  - `ToRadians[180]  ==> 3.14159...`

## `ToUpper`

- Usage: `ToUpper[s]`
- Summary: Uppercase string.
- Tags: string, text
- Examples:
  - `ToUpper["hi"]  ==> "HI"`

## `ToolsInvoke`

- Usage: `ToolsInvoke[id|name, args?]`
- Summary: Invoke a tool with an args assoc.
- Examples:
  - `ToolsInvoke["Hello", <|"name"->"Lyra"|>]  ==> "Hello, Lyra"`

## `ToolsRegister`

- Usage: `ToolsRegister[spec|list]`
- Summary: Register one or more tool specs.
- Examples:
  - `ToolsRegister[<|"id"->"Hello", "summary"->"Say hi", "params"->{"name"}|>]  ==> <|...|>`
  - `ToolsList[]  ==> {...}`

## `UrlEncode`

- Usage: `UrlEncode[s]`
- Summary: Percent-encode string for URLs
- Tags: string, url
- Examples:
  - `UrlEncode["a b"]  ==> "a%20b"`
