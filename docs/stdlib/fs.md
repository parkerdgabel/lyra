Filesystem

Overview
- Create, remove, copy, move files/directories; symlinks; temp files/dirs; globbing; directory watching; archives/compression.

Functions
- MakeDirectory[path, <|Parents->True|False|>] -> True|Failure
- Remove[path, <|Recursive->True|False|>] -> True|Failure
- Copy[src, dst, <|Recursive->True|False|>] -> True|Failure
- Move[src, dst] -> True|Failure
- Touch[path] -> True|Failure
- Symlink[target, linkPath] -> True|Failure
- ReadBytes[path] -> String|Failure (returns text in current build)
- WriteBytes[path, bytes] -> True|Failure (accepts String for bytes)
- TempFile[] -> path
- TempDir[] -> path
- Glob[pattern, opts?] -> List[path]
  - patterns: String or List[String]; supports *, ?, and ** for recursive directory components.
  - opts: <|Cwd, Recursive, Include, Exclude, Dotfiles, FollowSymlinks?|>
- WatchDirectory[path, handler, <|Recursive->True, DebounceMs->50|>] -> <|__type:"FSWatch", id|> | Failure
  - Calls handler[<|Event->"create"|"modify"|"delete"|"rename", Path->string|>] on changes.
- CancelWatch[handle] -> True|False

Archives & Compression (feature fs_archive, enabled by default)
- ZipCreate[dest, inputs, <|BaseDir, Include, Exclude, StripComponents, Overwrite, CreateDirs|>] -> <|path, files, bytes|>
- ZipExtract[src, dest, <|Include, Exclude, StripComponents, Overwrite, CreateDirs|>] -> <|path, files|>
- TarCreate[dest, inputs, <|Gzip->True|False, BaseDir, Include, Exclude, StripComponents, Overwrite, CreateDirs|>] -> <|path, files, bytes|>
- TarExtract[src, dest, <|Include, Exclude, StripComponents, Overwrite, CreateDirs|>] -> <|path, files|>
- Gzip[data|path, <|Out, Level, Overwrite, CreateDirs|>] -> Bytes | <|path, bytes_written|>
- Gunzip[data|path, <|Out, Overwrite, CreateDirs|>] -> Bytes | <|path, bytes_written|>

Failures
- FS::mkdir, ::rm, ::cp, ::mv, ::read, ::write, ::temp, ::watch.

Examples
- p = TempFile[]; WriteBytes[p, "abc"]; ReadBytes[p] -> "abc"
- MakeDirectory["tmp/x", <|Parents->True|>]
- Glob["src/**/*.rs", <|Recursive->True, Exclude->{"**/target/**"}|>]
- w = WatchDirectory["./", Function[Log["info", #1]]]; CancelWatch[w]
- ZipCreate["out.zip", {"src/", "README.md"}]
- ZipExtract["out.zip", "./out"]

Notes
- WatchDirectory requires feature fs_watch (enabled by default). Handler runs in a new evaluator with stdlib.
- ReadBytes/WriteBytes currently use String for payloads. A binary Bytes type is planned.
