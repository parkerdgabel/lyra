# FS

| Function | Usage | Summary |
|---|---|---|
| `Copy` | `Copy[src, dst, opts?]` | Copy file or directory (Recursive option) |
| `CopyFromContainer` | `CopyFromContainer[id, src, dst]` | Copy file/dir from container |
| `CopyToContainer` | `CopyToContainer[id, src, dst]` | Copy file/dir into container |
| `Download` | `Download[url, path, opts]` | Download URL to file (http/https) |
| `DownloadStream` | `DownloadStream[url, path, opts]` | Stream download URL directly to file |
| `Glob` | `Glob[pattern]` | Expand glob pattern to matching paths |
| `GlobalClustering` | `GlobalClustering[graph]` | Global clustering coefficient |
| `Move` | `Move[src, dst]` | Move or rename a file/directory |
| `ReadBytes` | `ReadBytes[path]` | Read entire file as bytes |
| `ReadCSVDataset` | `ReadCSVDataset[path, opts?]` | Read a CSV file into a dataset |
| `ReadJsonLinesDataset` | `ReadJsonLinesDataset[path, opts?]` | Read a JSONL file into a dataset |
| `ReadProcess` | `ReadProcess[proc, opts?]` | Read from process stdout/stderr |
| `Remove` | `Remove[path, opts?]` | Remove a file or directory (Recursive option) |
| `RemoveContainer` | `RemoveContainer[id, opts?]` | Remove a container |
| `RemoveEdges` | `RemoveEdges[graph, edges]` | Remove edges by id or (src,dst,key) |
| `RemoveImage` | `RemoveImage[ref]` | Remove local image |
| `RemoveNetwork` | `RemoveNetwork[name]` | Remove network |
| `RemoveNodes` | `RemoveNodes[graph, ids]` | Remove nodes by id |
| `RemovePackage` | `RemovePackage[name, opts?]` | Remove a package (requires lyra-pm) |
| `RemoveVolume` | `RemoveVolume[name]` | Remove volume |
| `TempDir` | `TempDir[]` | Create a unique temporary directory |
| `TempFile` | `TempFile[]` | Create a unique temporary file |
| `TemplateRender` | `TemplateRender[template, data, opts?]` | Render Mustache-like template with assoc data. |
| `WatchDirectory` | `WatchDirectory[path, handler, opts?]` | Watch directory and stream events (held) |
| `WriteBytes` | `WriteBytes[path, bytes]` | Write bytes to file |
| `WriteDataset` | `WriteDataset[conn, table, dataset, opts?]` | Write a Dataset into a table |
| `WriteProcess` | `WriteProcess[proc, data]` | Write to process stdin |

## `Download`

- Usage: `Download[url, path, opts]`
- Summary: Download URL to file (http/https)
- Examples:
  - `Download["https://example.com", "/tmp/index.html"]  ==> "/tmp/index.html"`

## `Glob`

- Usage: `Glob[pattern]`
- Summary: Expand glob pattern to matching paths
- Examples:
  - `Glob["**/*.lyra"]  ==> {"examples/app.lyra", ...}`

## `ReadBytes`

- Usage: `ReadBytes[path]`
- Summary: Read entire file as bytes
- Examples:
  - `ReadBytes["/etc/hosts"]  ==> <byte list>`

## `ReadCSVDataset`

- Usage: `ReadCSVDataset[path, opts?]`
- Summary: Read a CSV file into a dataset
- Examples:
  - `ds := ReadCSVDataset["people.csv"]`
  - `Head[ds, 3]  ==> {{...},{...},{...}}`

## `TemplateRender`

- Usage: `TemplateRender[template, data, opts?]`
- Summary: Render Mustache-like template with assoc data.
- Examples:
  - `TemplateRender["Hello {{name}}!", <|"name"->"Lyra"|>]  ==> "Hello Lyra!"`

## `WriteBytes`

- Usage: `WriteBytes[path, bytes]`
- Summary: Write bytes to file
- Examples:
  - `WriteBytes["/tmp/x.bin", {0,255}]  ==> "/tmp/x.bin"`
