# FS

| Function | Usage | Summary |
|---|---|---|
| `CancelWatch` | `CancelWatch[token]` | Cancel a directory watch |
| `CopyFromContainer` | `CopyFromContainer[id, src, dst]` | Copy file/dir from container |
| `CopyToContainer` | `CopyToContainer[id, src, dst]` | Copy file/dir into container |
| `Download` | `Download[url, path, opts]` | Download URL to file (http/https) |
| `DownloadStream` | `DownloadStream[url, path, opts]` | Stream download URL directly to file |
| `Glob` | `Glob[pattern]` | Expand glob pattern to matching paths |
| `GlobalClustering` | `GlobalClustering[graph]` | Global clustering coefficient |
| `Gunzip` | `Gunzip[dataOrPath, opts?]` | Gunzip-decompress a string or a .gz file; optionally write to path. |
| `Gzip` | `Gzip[dataOrPath, opts?]` | Gzip-compress a string or a file; optionally write to path. |
| `Remove` | `Remove[path, opts?]` | Remove a file or directory (Recursive option) |
| `RemoveContainer` | `RemoveContainer[id, opts?]` | Remove a container |
| `RemoveImage` | `RemoveImage[ref]` | Remove local image |
| `RemoveNetwork` | `RemoveNetwork[name]` | Remove network |
| `RemovePackage` | `RemovePackage[name, opts?]` | Remove a package (requires lyra-pm) |
| `RemoveVolume` | `RemoveVolume[name]` | Remove volume |
| `Tar` | `Tar[dest, inputs, opts?]` | Create a .tar (optionally .tar.gz) archive from inputs. |
| `TarExtract` | `TarExtract[src, dest]` | Extract a .tar or .tar.gz archive into a directory. |
| `TemplateRender` | `TemplateRender[template, data, opts?]` | Render Mustache-like template with assoc data. |
| `WatchDirectory` | `WatchDirectory[path, handler, opts?]` | Watch directory and stream events (held) |
| `Zip` | `Zip[dest, inputs]` | Create a .zip archive from files/directories. |
| `ZipExtract` | `ZipExtract[src, dest]` | Extract a .zip archive into a directory. |

## `Download`

- Usage: `Download[url, path, opts]`
- Summary: Download URL to file (http/https)
- Tags: net, http, fs
- Examples:
  - `Download["https://example.com/image.png", "/tmp/image.png"]`

## `Glob`

- Usage: `Glob[pattern]`
- Summary: Expand glob pattern to matching paths
- Tags: fs, path, glob
- Examples:
  - `Glob["**/*.lyra"]  ==> {"examples/app.lyra", ...}`

## `Gunzip`

- Usage: `Gunzip[dataOrPath, opts?]`
- Summary: Gunzip-decompress a string or a .gz file; optionally write to path.
- Tags: fs, archive, compress
- Examples:
  - `Gunzip[Gzip["hello"]]  ==> "hello"`
  - `Gunzip["/tmp/a.txt.gz", <|"Out"->"/tmp/a.txt"|>]  ==> <|"path"->"/tmp/a.txt", "bytes_written"->...|>`

## `Gzip`

- Usage: `Gzip[dataOrPath, opts?]`
- Summary: Gzip-compress a string or a file; optionally write to path.
- Tags: fs, archive, compress
- Examples:
  - `Gzip["hello"]  ==> <compressed bytes as string>`
  - `Gzip["/tmp/a.txt", <|"Out"->"/tmp/a.txt.gz"|>]  ==> <|"path"->"/tmp/a.txt.gz", "bytes_written"->...|>`

## `Remove`

- Usage: `Remove[path, opts?]`
- Summary: Remove a file or directory (Recursive option)
- Tags: generic, collection
- Examples:
  - `Remove[{1,2,3}, 2]  ==> {1,3}`
  - `Remove[Queue[]]  ==> Null (dequeue)`
  - `Remove[Stack[]]  ==> Null (pop)`
  - `Remove[PriorityQueue[]]  ==> Null (pop)`
  - `Remove[g, {"a"}]`
  - `Remove[g, {<|Src->"a",Dst->"b"|>}]`

## `Tar`

- Usage: `Tar[dest, inputs, opts?]`
- Summary: Create a .tar (optionally .tar.gz) archive from inputs.
- Tags: fs, archive
- Examples:
  - `Tar["/tmp/bundle.tar", {"/tmp/data"}]  ==> <|"path"->"/tmp/bundle.tar"|>`
  - `Tar["/tmp/bundle.tar.gz", {"/tmp/data"}, <|"Gzip"->True|>]  ==> <|"path"->...|>`

## `TarExtract`

- Usage: `TarExtract[src, dest]`
- Summary: Extract a .tar or .tar.gz archive into a directory.
- Tags: fs, archive
- Examples:
  - `TarExtract["/tmp/bundle.tar", "/tmp/untar"]  ==> <|"path"->"/tmp/untar"|>`

## `TemplateRender`

- Usage: `TemplateRender[template, data, opts?]`
- Summary: Render Mustache-like template with assoc data.
- Examples:
  - `TemplateRender["Hello {{name}}!", <|"name"->"Lyra"|>]  ==> "Hello Lyra!"`

## `Zip`

- Usage: `Zip[dest, inputs]`
- Summary: Create a .zip archive from files/directories.
- Tags: fs, archive
- Examples:
  - `Zip["/tmp/bundle.zip", {"/tmp/a.txt", "/tmp/dir"}]  ==> <|"path"->"/tmp/bundle.zip", ...|>`

## `ZipExtract`

- Usage: `ZipExtract[src, dest]`
- Summary: Extract a .zip archive into a directory.
- Tags: fs, archive
- Examples:
  - `ZipExtract["/tmp/bundle.zip", "/tmp/unzipped"]  ==> <|"path"->"/tmp/unzipped", "files"->...|>`
