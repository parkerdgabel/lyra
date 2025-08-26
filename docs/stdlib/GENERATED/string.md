# STRING

| Function | Usage | Summary |
|---|---|---|
| `JsonEscape` | `JsonEscape[s]` | Escape string for JSON |
| `JsonUnescape` | `JsonUnescape[s]` | Unescape JSON-escaped string |
| `Length` | `Length[x]` | Length of a list or string. |
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
| `ToDegrees` | `ToDegrees[]` |  |
| `ToLower` | `ToLower[s]` | Lowercase string. |
| `ToRadians` | `ToRadians[]` |  |
| `ToUpper` | `ToUpper[s]` | Uppercase string. |
| `ToolsCacheClear` | `ToolsCacheClear[]` |  |
| `ToolsCards` | `ToolsCards[]` |  |
| `ToolsDescribe` | `ToolsDescribe[]` |  |
| `ToolsDryRun` | `ToolsDryRun[]` |  |
| `ToolsExportBundle` | `ToolsExportBundle[]` |  |
| `ToolsExportOpenAI` | `ToolsExportOpenAI[]` |  |
| `ToolsGetCapabilities` | `ToolsGetCapabilities[]` |  |
| `ToolsInvoke` | `ToolsInvoke[]` |  |
| `ToolsList` | `ToolsList[]` |  |
| `ToolsRegister` | `ToolsRegister[]` |  |
| `ToolsResolve` | `ToolsResolve[]` |  |
| `ToolsSearch` | `ToolsSearch[]` |  |
| `ToolsSetCapabilities` | `ToolsSetCapabilities[]` |  |
| `ToolsUnregister` | `ToolsUnregister[]` |  |
| `Top` | `Top[]` |  |
| `TopologicalSort` | `TopologicalSort[graph]` | Topologically sort DAG nodes |
| `Touch` | `Touch[path]` | Create file if missing (update mtime) |

## `Length`

- Usage: `Length[x]`
- Summary: Length of a list or string.
- Examples:
  - `Length[{1,2,3}]  ==> 3`
  - `Length["ok"]  ==> 2`

## `StringChars`

- Usage: `StringChars[s]`
- Summary: Split string into list of characters
- Examples:
  - `StringChars["abc"]  ==> {"a","b","c"}`

## `StringContains`

- Usage: `StringContains[s, substr]`
- Summary: Does string contain substring?
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
- Examples:
  - `StringSplit["a,b,c", ","]  ==> {"a","b","c"}`

## `StringTrim`

- Usage: `StringTrim[s]`
- Summary: Trim whitespace from both ends
- Examples:
  - `StringTrim["  hi  "]  ==> "hi"`

## `ToLower`

- Usage: `ToLower[s]`
- Summary: Lowercase string.
- Examples:
  - `ToLower["Hello"]  ==> "hello"`

## `ToUpper`

- Usage: `ToUpper[s]`
- Summary: Uppercase string.
- Examples:
  - `ToUpper["hi"]  ==> "HI"`
