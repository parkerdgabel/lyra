# IO

| Function | Usage | Summary |
|---|---|---|
| `AudioSave` | `AudioSave[input, output, encoding]` | Encode and write audio to path (WAV) |
| `Base64Decode` | `Base64Decode[text, opts]` | Decode base64 to bytes |
| `Base64Encode` | `Base64Encode[bytes, opts]` | Encode bytes to base64 |
| `Basename` | `Basename[path]` | Filename without directories |
| `BytesConcat` | `BytesConcat[chunks]` | Concatenate byte arrays |
| `BytesLength` | `BytesLength[bytes]` | Length of byte array |
| `BytesSlice` | `BytesSlice[bytes, start, end]` | Slice a byte array |
| `CanonicalPath` | `CanonicalPath[path]` | Resolve symlinks and normalize |
| `CsvRead` | `CsvRead[csv, opts]` | Alias of ParseCSV |
| `CsvWrite` | `CsvWrite[rows, opts]` | Alias of RenderCSV |
| `CurrentDirectory` | `CurrentDirectory[]` | Get current working directory |
| `Dirname` | `Dirname[path]` | Parent directory path |
| `ExpandPath` | `ExpandPath[path]` | Expand ~ and env vars |
| `FileExistsQ` | `FileExistsQ[path]` | Does path exist? |
| `FileExtension` | `FileExtension[path]` | File extension (no dot) |
| `FileStem` | `FileStem[path]` | Filename without extension |
| `FromJson` | `FromJson[json]` | Parse JSON string to value |
| `GetEnv` | `GetEnv[name]` | Read environment variable |
| `HexDecode` | `HexDecode[text]` | Decode hex to bytes |
| `HexEncode` | `HexEncode[bytes]` | Encode bytes to hex |
| `ImageSave` | `ImageSave[input, output, encoding]` | Encode and write image to path |
| `JsonParse` | `JsonParse[json]` | Alias of FromJson |
| `JsonStringify` | `JsonStringify[value, opts]` | Alias of ToJson |
| `ListDirectory` | `ListDirectory[path]` | List names in directory |
| `ParseCSV` | `ParseCSV[csv, opts]` | Parse CSV string to rows |
| `PathJoin` | `PathJoin[parts]` | Join path segments |
| `PathSplit` | `PathSplit[path]` | Split path into parts |
| `ReadCSV` | `ReadCSV[path, opts]` | Read and parse CSV file |
| `ReadFile` | `ReadFile[path]` | Read entire file as string |
| `ReadLines` | `ReadLines[path]` | Read file into list of lines |
| `ReadStdin` | `ReadStdin[]` | Read all text from stdin |
| `RenderCSV` | `RenderCSV[rows, opts]` | Render rows to CSV string |
| `RespondJson` | `RespondJson[value, opts]` | Build a JSON response for HttpServe |
| `SetDirectory` | `SetDirectory[path]` | Change current working directory |
| `SetEnv` | `SetEnv[name, value]` | Set environment variable |
| `Stat` | `Stat[path]` | Basic file metadata as assoc |
| `TextDecode` | `TextDecode[bytes, opts]` | Decode bytes to text (utf-8) |
| `TextEncode` | `TextEncode[text, opts]` | Encode text to bytes (utf-8) |
| `ToJson` | `ToJson[value, opts]` | Serialize value to JSON string |
| `TomlParse` | `TomlParse[toml]` | Parse TOML string to value |
| `TomlStringify` | `TomlStringify[value]` | Render value as TOML |
| `WriteCSV` | `WriteCSV[path, rows, opts]` | Write rows to CSV file |
| `WriteFile` | `WriteFile[path, content]` | Write stringified content to file |
| `YamlParse` | `YamlParse[yaml]` | Parse YAML string to value |
| `YamlStringify` | `YamlStringify[value, opts]` | Render value as YAML |

## `Basename`

- Usage: `Basename[path]`
- Summary: Filename without directories
- Examples:
  - `Basename["/a/b/c.txt"]  ==> "c.txt"`

## `FileExistsQ`

- Usage: `FileExistsQ[path]`
- Summary: Does path exist?
- Examples:
  - `FileExistsQ["/etc/passwd"]  ==> True`

## `FromJson`

- Usage: `FromJson[json]`
- Summary: Parse JSON string to value
- Examples:
  - `FromJson["{\"a\":1}"]  ==> <|"a"->1|>`

## `ParseCSV`

- Usage: `ParseCSV[csv, opts]`
- Summary: Parse CSV string to rows
- Examples:
  - `ParseCSV["a,b\n1,2\n"]  ==> {{"a","b"},{"1","2"}}`

## `PathJoin`

- Usage: `PathJoin[parts]`
- Summary: Join path segments
- Examples:
  - `PathJoin[{"/tmp", "dir", "file.txt"}]  ==> "/tmp/dir/file.txt"`

## `ReadFile`

- Usage: `ReadFile[path]`
- Summary: Read entire file as string
- Examples:
  - `WriteFile["/tmp/x.txt", "hi"]; ReadFile["/tmp/x.txt"]  ==> "hi"`

## `ReadLines`

- Usage: `ReadLines[path]`
- Summary: Read file into list of lines
- Examples:
  - `ReadLines["/etc/hosts"]  ==> {"127.0.0.1 localhost", ...}`

## `RenderCSV`

- Usage: `RenderCSV[rows, opts]`
- Summary: Render rows to CSV string
- Examples:
  - `RenderCSV[{{"a","b"},{1,2}}]  ==> "a,b\n1,2\n"`

## `ToJson`

- Usage: `ToJson[value, opts]`
- Summary: Serialize value to JSON string
- Examples:
  - `ToJson[<|"a"->1|>]  ==> "{\"a\":1}"`

## `WriteFile`

- Usage: `WriteFile[path, content]`
- Summary: Write stringified content to file
- Examples:
  - `WriteFile["/tmp/y.txt", <|"a"->1|>]  ==> True`
