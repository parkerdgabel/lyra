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
| `Collect` | `Collect[ds, limit?, opts?]` | Materialize dataset rows as a list |
| `Copy` | `Copy[src, dst, opts?]` | Copy file or directory (Recursive option) |
| `CsvRead` | `CsvRead[csv, opts]` | Alias of ParseCSV |
| `CsvWrite` | `CsvWrite[rows, opts]` | Alias of RenderCSV |
| `CurrentDirectory` | `CurrentDirectory[]` | Get current working directory |
| `Dirname` | `Dirname[path]` | Parent directory path |
| `ExpandPath` | `ExpandPath[path]` | Expand ~ and env vars |
| `Export` | `Export[symbols]` | Mark symbol(s) as public |
| `FileExistsQ` | `FileExistsQ[path]` | Does path exist? |
| `FileExtension` | `FileExtension[path]` | File extension (no dot) |
| `FileStem` | `FileStem[path]` | Filename without extension |
| `FrameCollect` | `FrameCollect[frame]` | Materialize Frame to list of rows |
| `FrameWriteCSV` | `FrameWriteCSV[path, frame, opts?]` | Write Frame to CSV file |
| `FrameWriteJSONLines` | `FrameWriteJSONLines[path, frame, opts?]` | Write Frame rows as JSON Lines |
| `FromJson` | `FromJson[json]` | Parse JSON string to value |
| `GetEnv` | `GetEnv[name]` | Read environment variable |
| `HexDecode` | `HexDecode[text]` | Decode hex to bytes |
| `HexEncode` | `HexEncode[bytes]` | Encode bytes to hex |
| `ImageSave` | `ImageSave[input, output, encoding]` | Encode and write image to path |
| `Import` | `Import[source, opts?]` | Import data from path/URL into Frame (default), Dataset (Target->"Dataset"), or Value. Automatically sniffs Type/Delimiter/Header. |
| `ImportBytes` | `ImportBytes[bytes, opts?]` | Parse byte buffer using Type (text/json/etc.) |
| `ImportString` | `ImportString[content, opts?]` | Parse in-memory strings into Frame/Dataset/Value. Automatically sniffs Type if missing. |
| `JsonParse` | `JsonParse[json]` | Alias of FromJson |
| `JsonStringify` | `JsonStringify[value, opts]` | Alias of ToJson |
| `ListDirectory` | `ListDirectory[path]` | List names in directory |
| `MakeDirectory` | `MakeDirectory[path, opts?]` | Create a directory (Parents option) |
| `Move` | `Move[src, dst]` | Move or rename a file/directory |
| `ParseCSV` | `ParseCSV[csv, opts]` | Parse CSV string to rows |
| `PathJoin` | `PathJoin[parts]` | Join path segments |
| `PathRemove` | `PathRemove[path, opts?]` | Alias: remove file or directory |
| `PathSplit` | `PathSplit[path]` | Split path into parts |
| `ReadBytes` | `ReadBytes[path]` | Read entire file as bytes |
| `ReadCSV` | `ReadCSV[path, opts]` | Read and parse CSV file |
| `ReadCSVDataset` | `ReadCSVDataset[path, opts?]` | Read a CSV file into a dataset |
| `ReadFile` | `ReadFile[path]` | Read entire file as string |
| `ReadJsonLinesDataset` | `ReadJsonLinesDataset[path, opts?]` | Read a JSONL file into a dataset |
| `ReadLines` | `ReadLines[path]` | Read file into list of lines |
| `ReadStdin` | `ReadStdin[]` | Read all text from stdin |
| `RenderCSV` | `RenderCSV[rows, opts]` | Render rows to CSV string |
| `RespondJson` | `RespondJson[value, opts]` | Build a JSON response for HttpServe |
| `SetDirectory` | `SetDirectory[path]` | Change current working directory |
| `SetEnv` | `SetEnv[name, value]` | Set environment variable |
| `Sniff` | `Sniff[source]` | Suggest Type and options for a source (file/url/string/bytes). |
| `Stat` | `Stat[path]` | Basic file metadata as assoc |
| `Symlink` | `Symlink[src, dst]` | Create a symbolic link |
| `TempDir` | `TempDir[]` | Create a unique temporary directory |
| `TempFile` | `TempFile[]` | Create a unique temporary file |
| `TextDecode` | `TextDecode[bytes, opts]` | Decode bytes to text (utf-8) |
| `TextEncode` | `TextEncode[text, opts]` | Encode text to bytes (utf-8) |
| `ToJson` | `ToJson[value, opts]` | Serialize value to JSON string |
| `TomlParse` | `TomlParse[toml]` | Parse TOML string to value |
| `TomlStringify` | `TomlStringify[value]` | Render value as TOML |
| `Touch` | `Touch[path]` | Create file if missing (update mtime) |
| `WriteBytes` | `WriteBytes[path, bytes]` | Write bytes to file |
| `WriteCSV` | `WriteCSV[path, rows, opts]` | Write rows to CSV file |
| `WriteFile` | `WriteFile[path, content]` | Write stringified content to file |
| `YamlParse` | `YamlParse[yaml]` | Parse YAML string to value |
| `YamlStringify` | `YamlStringify[value, opts]` | Render value as YAML |

## `AudioSave`

- Usage: `AudioSave[input, output, encoding]`
- Summary: Encode and write audio to path (WAV)
- Tags: audio, io
- Examples:
  - `AudioSave[<|path->"in.wav"|>, "out.wav", <|format->"wav"|>]`

## `Basename`

- Usage: `Basename[path]`
- Summary: Filename without directories
- Tags: io, path
- Examples:
  - `Basename["/a/b/c.txt"]  ==> "c.txt"`

## `Export`

- Usage: `Export[symbols]`
- Summary: Mark symbol(s) as public
- Tags: generic, export, io
- Examples:
  - `Export[{"Foo", "Bar"}]`

## `FileExistsQ`

- Usage: `FileExistsQ[path]`
- Summary: Does path exist?
- Tags: io, fs
- Examples:
  - `FileExistsQ["/etc/passwd"]  ==> True`

## `FrameCollect`

- Usage: `FrameCollect[frame]`
- Summary: Materialize Frame to list of rows
- Tags: frame, io
- Examples:
  - `FrameCollect[f]  ==> {<|...|>,...}`

## `FrameWriteCSV`

- Usage: `FrameWriteCSV[path, frame, opts?]`
- Summary: Write Frame to CSV file
- Tags: frame, io, csv
- Examples:
  - `FrameWriteCSV["out.csv", f]`

## `FrameWriteJSONLines`

- Usage: `FrameWriteJSONLines[path, frame, opts?]`
- Summary: Write Frame rows as JSON Lines
- Tags: frame, io, json
- Examples:
  - `FrameWriteJSONLines["out.jsonl", f]`

## `FromJson`

- Usage: `FromJson[json]`
- Summary: Parse JSON string to value
- Tags: io, json
- Examples:
  - `FromJson["{\"a\":1}"]  ==> <|"a"->1|>`

## `ImageSave`

- Usage: `ImageSave[input, output, encoding]`
- Summary: Encode and write image to path
- Tags: image, io
- Examples:
  - `ImageSave[<|Path->"in.png"|>, "out.jpg", <|Format->"jpeg"|>]`

## `Import`

- Usage: `Import[source, opts?]`
- Summary: Import data from path/URL into Frame (default), Dataset (Target->"Dataset"), or Value. Automatically sniffs Type/Delimiter/Header.
- Tags: generic, import, io, frame, dataset
- Examples:
  - `Import["data.csv"]  ==> Frame[...]`
  - `Import["data.csv", <|Target->"Dataset"|>]  ==> Dataset[...]`
  - `Import["logs/*.jsonl", <|Type->"JSONL"|>]  ==> Frame[...]`

## `ImportBytes`

- Usage: `ImportBytes[bytes, opts?]`
- Summary: Parse byte buffer using Type (text/json/etc.)
- Tags: generic, import, io
- Examples:
  - `ImportBytes[bytes, <|Type->"Text"|>]`

## `ImportString`

- Usage: `ImportString[content, opts?]`
- Summary: Parse in-memory strings into Frame/Dataset/Value. Automatically sniffs Type if missing.
- Tags: generic, import, io
- Examples:
  - `ImportString["a,b\n1,2"]  ==> Frame[...]`
  - `ImportString["[{\"a\":1}]", <|Target->"Dataset"|>]  ==> Dataset[...]`

## `ParseCSV`

- Usage: `ParseCSV[csv, opts]`
- Summary: Parse CSV string to rows
- Tags: io, csv
- Examples:
  - `ParseCSV["a,b\n1,2\n"]  ==> {{"a","b"},{"1","2"}}`

## `PathJoin`

- Usage: `PathJoin[parts]`
- Summary: Join path segments
- Tags: io, path
- Examples:
  - `PathJoin[{"/tmp", "dir", "file.txt"}]  ==> "/tmp/dir/file.txt"`

## `ReadBytes`

- Usage: `ReadBytes[path]`
- Summary: Read entire file as bytes
- Tags: fs, io, bytes
- Examples:
  - `ReadBytes["/etc/hosts"]  ==> <byte list>`

## `ReadCSVDataset`

- Usage: `ReadCSVDataset[path, opts?]`
- Summary: Read a CSV file into a dataset
- Tags: dataset, io, csv
- Examples:
  - `ds := ReadCSVDataset["people.csv"]`
  - `Head[ds, 3]  ==> {{...},{...},{...}}`

## `ReadFile`

- Usage: `ReadFile[path]`
- Summary: Read entire file as string
- Tags: io, fs
- Examples:
  - `WriteFile["/tmp/x.txt", "hi"]; ReadFile["/tmp/x.txt"]  ==> "hi"`

## `ReadLines`

- Usage: `ReadLines[path]`
- Summary: Read file into list of lines
- Tags: io, fs
- Examples:
  - `ReadLines["/etc/hosts"]  ==> {"127.0.0.1 localhost", ...}`

## `RenderCSV`

- Usage: `RenderCSV[rows, opts]`
- Summary: Render rows to CSV string
- Tags: io, csv
- Examples:
  - `RenderCSV[{{"a","b"},{1,2}}]  ==> "a,b\n1,2\n"`

## `Sniff`

- Usage: `Sniff[source]`
- Summary: Suggest Type and options for a source (file/url/string/bytes).
- Tags: generic, import, io
- Examples:
  - `Sniff["data.csv"]  ==> <|Type->"CSV", Delimiter->",", Header->True|>`
  - `Sniff["https://ex.com/data.jsonl"]  ==> <|Type->"JSONL"|>`

## `ToJson`

- Usage: `ToJson[value, opts]`
- Summary: Serialize value to JSON string
- Tags: io, json
- Examples:
  - `ToJson[<|"a"->1|>]  ==> "{\"a\":1}"`

## `WriteBytes`

- Usage: `WriteBytes[path, bytes]`
- Summary: Write bytes to file
- Tags: fs, io, bytes
- Examples:
  - `WriteBytes["/tmp/x.bin", {0,255}]  ==> "/tmp/x.bin"`

## `WriteFile`

- Usage: `WriteFile[path, content]`
- Summary: Write stringified content to file
- Tags: io, fs
- Examples:
  - `WriteFile["/tmp/y.txt", <|"a"->1|>]  ==> True`
