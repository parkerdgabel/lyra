Data Formats and Encoding

This page covers JSON, YAML, TOML, CSV, bytes/encoding, and UUID helpers. All functions are part of the `io` and `crypto` stdlib groups and are available by default.

JSON

- JsonParse(text) -> value: Parses JSON into Lyra values.
- JsonStringify(value, opts?) -> string: Renders a value as JSON.
- Options (JsonStringify):
  - Pretty: Bool (default false)
  - SortKeys: Bool (default false)
  - EnsureAscii: Bool (default false)
  - TrailingNewline: Bool (default false)
  - Indent: Int|String (spaces or custom indent when Pretty)
- Aliases: FromJson ≡ JsonParse, ToJson ≡ JsonStringify.
- Example:
  - `JsonParse("{\"a\":1}")` → `{a: 1}`
  - `JsonStringify({a: 1}, {Pretty: true})` → "{\n  \"a\": 1\n}"

YAML

- YamlParse(text) -> value: Parses YAML (1 doc) to Lyra values.
- YamlStringify(value) -> string: Renders a value as YAML.
- Notes: YAML scalars/collections map to Lyra’s Null/Bool/Integer/Real/String/List/Assoc.
- Example:
  - `YamlParse("a: 1\nb: [2, 3]\n")` → `{a: 1, b: [2, 3]}`
  - `YamlStringify({a: 1, b: [2, 3]})` → "a: 1\nb:\n  - 2\n  - 3\n"

TOML

- TomlParse(text) -> value: Parses TOML to Lyra values.
- TomlStringify(value) -> string: Renders a value as pretty TOML.
- Notes: Datetime values parse to strings; rendering routes via JSON mapping for broad coverage.
- Example:
  - `TomlParse("a = 1\n[b]\nx = 2\n")` → `{a: 1, b: {x: 2}}`
  - `TomlStringify({a: 1, b: {x: 2}})` → "a = 1\n[b]\nx = 2\n"

CSV

- ParseCSV(text, opts?) -> rows | objects: In-memory CSV parse.
- ReadCSV(path, opts?) -> rows | objects: Reads a file then parses.
- RenderCSV(rows, opts?) -> string: In-memory render.
- WriteCSV(path, rows, opts?) -> true|Failure: Renders then writes.
- Aliases: CsvRead ≡ ParseCSV, CsvWrite ≡ RenderCSV.
- Options (CSV):
  - Delimiter: String (default ",")
  - Quote: String (default ")
  - Header: Bool (default true) — treat first row as headers
  - Eol: String ("\n" or "\r\n")
  - Columns: [String] — select/rename columns when rendering
  - Headers: [String] — override header row on parse
- Returns:
  - When Header=true or Headers provided: List of Assoc rows
  - Else: List of List rows (no header)
- Example:
  - `ParseCSV("a,b\n1,2\n", {Header: true})` → `[{a: "1", b: "2"}]`
  - `RenderCSV([{a: "1", b: "2"}], {Columns: ["b", "a"]})` → "b,a\n2,1\n"

Bytes and Text Encoding

- TextEncode(text) -> bytes: Encodes UTF‑8 text to bytes.
- TextDecode(bytes) -> string: Decodes UTF‑8 bytes to text.
- Base64Encode(bytes) -> string: Encodes bytes to Base64.
- Base64Decode(text) -> bytes: Decodes Base64 to bytes.
- HexEncode(bytes) -> string: Encodes bytes as lowercase hex.
- HexDecode(text) -> bytes: Decodes hex string to bytes.
- BytesConcat([bytes]) -> bytes: Concatenates sequences of bytes.
- BytesSlice(bytes, start, end?) -> bytes: Slices byte array.
- BytesLength(bytes) -> int: Length in bytes.
- Bytes representation: list of integers 0..255. Some APIs also accept `Binary[base64url]`.
- Examples:
  - `TextEncode("hi")` → `[104, 105]`; `TextDecode([104,105])` → "hi"
  - `Base64Encode([104,101,108,108,111])` → "aGVsbG8="
  - `HexEncode([222,173,190,239])` → "deadbeef"
  - `BytesConcat([[1,2],[3]])` → `[1,2,3]`; `BytesSlice([1,2,3,4], 1, 3)` → `[2,3]`

UUIDs

- UuidV4() -> string: Random UUID v4.
- UuidV7() -> string: Time-ordered UUID v7.
- Examples:
  - `UuidV4()` → "d1b1f1a3-..."
  - `UuidV7()` → "0192f7...-..."

Failure Tags

- Common failures: InvalidInput, ParseError, IoError.
- CSV read/write: may also fail with Fs read/write errors when using file helpers.

Notes

- All parsers normalize data into Lyra’s core value types: Null, Bool, Integer, Real, String, List, Assoc.
- JSON stringify supports pretty-print, key sorting, and ASCII-escaping options for reproducible outputs.
