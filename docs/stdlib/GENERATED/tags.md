# Tags Index

| Tag | Functions |
|---|---|
| `actor` | 4 |
| `admin` | 1 |
| `aead` | 3 |
| `aggregate` | 4 |
| `apply` | 1 |
| `archive` | 6 |
| `assoc` | 13 |
| `async` | 4 |
| `audio` | 12 |
| `binary` | 1 |
| `branch` | 5 |
| `bytes` | 11 |
| `calendar` | 2 |
| `centrality` | 1 |
| `channel` | 6 |
| `chart` | 3 |
| `classification` | 1 |
| `clock` | 2 |
| `clustering` | 1 |
| `collection` | 3 |
| `commit` | 2 |
| `compose` | 2 |
| `compress` | 2 |
| `concurrency` | 21 |
| `conn` | 4 |
| `cookies` | 2 |
| `copy` | 1 |
| `create` | 2 |
| `cron` | 1 |
| `crypto` | 16 |
| `csv` | 8 |
| `cursor` | 3 |
| `cv` | 1 |
| `dataset` | 28 |
| `datetime` | 3 |
| `db` | 18 |
| `decode` | 3 |
| `decomposition` | 1 |
| `delete` | 2 |
| `diff` | 1 |
| `distinct` | 3 |
| `duration` | 4 |
| `edit` | 6 |
| `encode` | 3 |
| `encoding` | 6 |
| `env` | 2 |
| `explain` | 2 |
| `export` | 1 |
| `expr` | 1 |
| `ffmpeg` | 7 |
| `filter` | 3 |
| `fixedpoint` | 2 |
| `flow` | 1 |
| `fold` | 3 |
| `frame` | 20 |
| `fs` | 27 |
| `functional` | 11 |
| `generic` | 31 |
| `git` | 22 |
| `glob` | 1 |
| `graph` | 16 |
| `graphs` | 1 |
| `groupby` | 1 |
| `hash` | 1 |
| `html` | 3 |
| `http` | 25 |
| `image` | 12 |
| `import` | 4 |
| `index` | 1 |
| `inference` | 2 |
| `info` | 1 |
| `inspect` | 4 |
| `introspect` | 7 |
| `introspection` | 2 |
| `io` | 65 |
| `iteration` | 2 |
| `join` | 1 |
| `json` | 9 |
| `jwt` | 2 |
| `kdf` | 3 |
| `layer` | 15 |
| `layout` | 1 |
| `lifecycle` | 2 |
| `link` | 1 |
| `list` | 28 |
| `log` | 1 |
| `logic` | 12 |
| `mac` | 2 |
| `map` | 1 |
| `math` | 9 |
| `media` | 7 |
| `metrics` | 2 |
| `ml` | 11 |
| `move` | 1 |
| `mst` | 1 |
| `mutate` | 2 |
| `net` | 17 |
| `nn` | 23 |
| `optimize` | 1 |
| `os` | 9 |
| `parallel` | 2 |
| `patch` | 1 |
| `path` | 17 |
| `paths` | 1 |
| `pipeline` | 2 |
| `plot` | 2 |
| `predicate` | 11 |
| `preprocess` | 2 |
| `proc` | 10 |
| `process` | 10 |
| `product` | 1 |
| `query` | 3 |
| `random` | 2 |
| `redirect` | 1 |
| `regex` | 5 |
| `regression` | 1 |
| `remote` | 4 |
| `remove` | 2 |
| `replace` | 2 |
| `repo` | 2 |
| `routing` | 1 |
| `schedule` | 3 |
| `schema` | 7 |
| `scope` | 5 |
| `search` | 4 |
| `select` | 3 |
| `server` | 12 |
| `set` | 5 |
| `sign` | 3 |
| `sleep` | 1 |
| `slice` | 5 |
| `sort` | 2 |
| `sql` | 17 |
| `stats` | 2 |
| `status` | 2 |
| `stdio` | 1 |
| `store` | 1 |
| `string` | 23 |
| `sum` | 1 |
| `sync` | 1 |
| `table` | 1 |
| `temp` | 2 |
| `testing` | 1 |
| `text` | 6 |
| `time` | 16 |
| `tls` | 1 |
| `toml` | 2 |
| `train` | 1 |
| `transform` | 11 |
| `traversal` | 2 |
| `tune` | 1 |
| `tx` | 3 |
| `types` | 2 |
| `tz` | 1 |
| `upsert` | 1 |
| `url` | 3 |
| `vcs` | 2 |
| `vector` | 6 |
| `viz` | 6 |
| `watch` | 2 |
| `write` | 3 |
| `yaml` | 2 |
| `zip` | 1 |

## `actor`

| Function | Usage | Summary |
|---|---|---|
| `Actor` | `Actor[handler]` | Create actor with handler (held) |
| `Ask` | `Ask[actor, msg]` | Request/response with actor (held) |
| `StopActor` | `StopActor[actor]` | Stop actor |
| `Tell` | `Tell[actor, msg]` | Send message to actor (held) |

## `admin`

| Function | Usage | Summary |
|---|---|---|
| `VectorReset` | `VectorReset[store]` | Clear all items in store |

## `aead`

| Function | Usage | Summary |
|---|---|---|
| `AeadDecrypt` | `AeadDecrypt[envelope, key, opts]` | Decrypt AEAD envelope |
| `AeadEncrypt` | `AeadEncrypt[plaintext, key, opts]` | Encrypt with AEAD (ChaCha20-Poly1305) |
| `AeadKeyGen` | `AeadKeyGen[opts]` | Generate AEAD key (ChaCha20-Poly1305) |

## `aggregate`

| Function | Usage | Summary |
|---|---|---|
| `Agg` | `Agg[ds, aggs]` | Aggregate groups to single rows |
| `Count` | `Count[list, value|pred]` | Count elements equal to value or matching predicate |
| `CountBy` | `CountBy[f, list]` | Counts by key function (assoc) |
| `Tally` | `Tally[list]` | Counts by value (assoc) |

## `apply`

| Function | Usage | Summary |
|---|---|---|
| `Apply` | `Apply[f, list]` | Apply head to list elements: Apply[f, {…}] |

## `archive`

| Function | Usage | Summary |
|---|---|---|
| `Gunzip` | `Gunzip[dataOrPath, opts?]` | Gunzip-decompress a string or a .gz file; optionally write to path. |
| `Gzip` | `Gzip[dataOrPath, opts?]` | Gzip-compress a string or a file; optionally write to path. |
| `Tar` | `Tar[dest, inputs, opts?]` | Create a .tar (optionally .tar.gz) archive from inputs. |
| `TarExtract` | `TarExtract[src, dest]` | Extract a .tar or .tar.gz archive into a directory. |
| `Zip` | `Zip[dest, inputs]` | Create a .zip archive from files/directories. |
| `ZipExtract` | `ZipExtract[src, dest]` | Extract a .zip archive into a directory. |

## `assoc`

| Function | Usage | Summary |
|---|---|---|
| `AssociationMap` | `AssociationMap[f, assoc]` | Map values with f[v] |
| `AssociationMapKV` | `AssociationMapKV[fn, assoc]` | Map key/value pairs in an association |
| `AssociationMapKeys` | `AssociationMapKeys[f, assoc]` | Map keys with f[k] |
| `AssociationMapPairs` | `AssociationMapPairs[f, assoc]` | Map over (k,v) pairs |
| `Columns` | `Columns[ds]` | List column names for a dataset |
| `KeySort` | `KeySort[assoc]` | Sort association by key |
| `Keys` | `Keys[subject]` | Keys/columns for assoc, rows, Dataset, or Frame |
| `Lookup` | `Lookup[assoc, key, default]` | Lookup value from association |
| `Merge` | `Merge[args]` | Merge associations with optional combiner |
| `Part` | `Part[subject, index]` | Index into list/assoc |
| `Select` | `Select[ds, cols]` | Select/compute columns |
| `SortBy` | `SortBy[keyFn, assoc]` | Sort association by derived key |
| `Values` | `Values[assoc]` | List values of an association |

## `async`

| Function | Usage | Summary |
|---|---|---|
| `Await` | `Await[future]` | Wait for Future and return value |
| `Future` | `Future[expr]` | Create a Future from an expression (held) |
| `Gather` | `Gather[futures]` | Await Futures in same structure |
| `MapAsync` | `MapAsync[f, list]` | Map to Futures over list |

## `audio`

| Function | Usage | Summary |
|---|---|---|
| `AudioChannelMix` | `AudioChannelMix[input, opts]` | Convert channel count (mono/stereo) |
| `AudioConcat` | `AudioConcat[inputs, opts]` | Concatenate multiple inputs |
| `AudioConvert` | `AudioConvert[input, format, opts]` | Convert audio to WAV |
| `AudioDecode` | `AudioDecode[input, opts]` | Decode audio to raw (s16le) or WAV |
| `AudioEncode` | `AudioEncode[raw, opts]` | Encode raw PCM to WAV |
| `AudioFade` | `AudioFade[input, opts]` | Fade in/out |
| `AudioGain` | `AudioGain[input, opts]` | Apply gain in dB or linear |
| `AudioInfo` | `AudioInfo[input]` | Probe audio metadata |
| `AudioResample` | `AudioResample[input, opts]` | Resample to new sample rate |
| `AudioSave` | `AudioSave[input, output, encoding]` | Encode and write audio to path (WAV) |
| `AudioTrim` | `AudioTrim[input, opts]` | Trim audio by time range |
| `MediaExtractAudio` | `MediaExtractAudio[input, opts]` | Extract audio track to format |

## `binary`

| Function | Usage | Summary |
|---|---|---|
| `RespondBytes` | `RespondBytes[bytes, opts]` | Build a binary response for HttpServe |

## `branch`

| Function | Usage | Summary |
|---|---|---|
| `GitBranch` | `GitBranch[name, opts?]` | Create a new branch |
| `GitBranchList` | `GitBranchList[]` | List local branches |
| `GitCurrentBranch` | `GitCurrentBranch[]` | Current branch name |
| `GitFeatureBranch` | `GitFeatureBranch[opts?]` | Create and switch to a feature branch |
| `GitSwitch` | `GitSwitch[name, opts?]` | Switch to branch (optionally create) |

## `bytes`

| Function | Usage | Summary |
|---|---|---|
| `Base64Decode` | `Base64Decode[text, opts]` | Decode base64 to bytes |
| `Base64Encode` | `Base64Encode[bytes, opts]` | Encode bytes to base64 |
| `BytesConcat` | `BytesConcat[chunks]` | Concatenate byte arrays |
| `BytesLength` | `BytesLength[bytes]` | Length of byte array |
| `BytesSlice` | `BytesSlice[bytes, start, end]` | Slice a byte array |
| `HexDecode` | `HexDecode[text]` | Decode hex to bytes |
| `HexEncode` | `HexEncode[bytes]` | Encode bytes to hex |
| `ReadBytes` | `ReadBytes[path]` | Read entire file as bytes |
| `TextDecode` | `TextDecode[bytes, opts]` | Decode bytes to text (utf-8) |
| `TextEncode` | `TextEncode[text, opts]` | Encode text to bytes (utf-8) |
| `WriteBytes` | `WriteBytes[path, bytes]` | Write bytes to file |

## `calendar`

| Function | Usage | Summary |
|---|---|---|
| `EndOf` | `EndOf[dt, unit]` | End of unit (day/week/month) |
| `StartOf` | `StartOf[dt, unit]` | Start of unit (day/week/month) |

## `centrality`

| Function | Usage | Summary |
|---|---|---|
| `PageRank` | `PageRank[graph, opts?]` | PageRank centrality scores |

## `channel`

| Function | Usage | Summary |
|---|---|---|
| `BoundedChannel` | `BoundedChannel[n]` | Create bounded channel |
| `CloseChannel` | `CloseChannel[ch]` | Close channel |
| `Receive` | `Receive[ch, opts?]` | Receive value from channel |
| `Send` | `Send[ch, value]` | Send value to channel (held) |
| `TryReceive` | `TryReceive[ch]` | Non-blocking receive |
| `TrySend` | `TrySend[ch, value]` | Non-blocking send (held) |

## `chart`

| Function | Usage | Summary |
|---|---|---|
| `BarChart` | `BarChart[data, opts]` | Render a bar chart |
| `Chart` | `Chart[spec, opts]` | Render a chart from a spec |
| `Histogram` | `Histogram[data, opts]` | Render a histogram |

## `classification`

| Function | Usage | Summary |
|---|---|---|
| `Classify` | `Classify[data, opts]` | Train a classifier (baseline/logistic) |

## `clock`

| Function | Usage | Summary |
|---|---|---|
| `MonotonicNow` | `MonotonicNow[]` | Monotonic clock milliseconds since start |
| `NowMs` | `NowMs[]` | Current UNIX time in milliseconds |

## `clustering`

| Function | Usage | Summary |
|---|---|---|
| `Cluster` | `Cluster[data, opts]` | Cluster points (prototype) |

## `collection`

| Function | Usage | Summary |
|---|---|---|
| `Add` | `Add[target, value]` | Add value to a collection (alias of Insert for some types) |
| `Insert` | `Insert[target, value]` | Insert into collection or structure (dispatched) |
| `Remove` | `Remove[path, opts?]` | Remove a file or directory (Recursive option) |

## `commit`

| Function | Usage | Summary |
|---|---|---|
| `GitCommit` | `GitCommit[message, opts?]` | Create a commit with message |
| `GitSmartCommit` | `GitSmartCommit[opts?]` | Stage + conventional commit (auto msg option) |

## `compose`

| Function | Usage | Summary |
|---|---|---|
| `Compose` | `Compose[f, g, …]` | Compose functions left-to-right |
| `RightCompose` | `RightCompose[f, g, …]` | Compose functions right-to-left |

## `compress`

| Function | Usage | Summary |
|---|---|---|
| `Gunzip` | `Gunzip[dataOrPath, opts?]` | Gunzip-decompress a string or a .gz file; optionally write to path. |
| `Gzip` | `Gzip[dataOrPath, opts?]` | Gzip-compress a string or a file; optionally write to path. |

## `concurrency`

| Function | Usage | Summary |
|---|---|---|
| `Actor` | `Actor[handler]` | Create actor with handler (held) |
| `Ask` | `Ask[actor, msg]` | Request/response with actor (held) |
| `Await` | `Await[future]` | Wait for Future and return value |
| `BoundedChannel` | `BoundedChannel[n]` | Create bounded channel |
| `CancelScope` | `CancelScope[scope]` | Cancel running scope |
| `CloseChannel` | `CloseChannel[ch]` | Close channel |
| `EndScope` | `EndScope[scope]` | End scope and release resources |
| `Future` | `Future[expr]` | Create a Future from an expression (held) |
| `Gather` | `Gather[futures]` | Await Futures in same structure |
| `InScope` | `InScope[scope, body]` | Run body inside a scope (held) |
| `MapAsync` | `MapAsync[f, list]` | Map to Futures over list |
| `ParallelMap` | `ParallelMap[f, list]` | Map in parallel over list |
| `ParallelTable` | `ParallelTable[exprs]` | Evaluate list of expressions in parallel (held) |
| `Receive` | `Receive[ch, opts?]` | Receive value from channel |
| `Scope` | `Scope[opts, body]` | Run body with resource limits (held) |
| `Send` | `Send[ch, value]` | Send value to channel (held) |
| `StartScope` | `StartScope[opts, body]` | Start a managed scope (held) |
| `StopActor` | `StopActor[actor]` | Stop actor |
| `Tell` | `Tell[actor, msg]` | Send message to actor (held) |
| `TryReceive` | `TryReceive[ch]` | Non-blocking receive |
| `TrySend` | `TrySend[ch, value]` | Non-blocking send (held) |

## `conn`

| Function | Usage | Summary |
|---|---|---|
| `Connect` | `Connect[dsn|opts]` | Open database connection (mock/sqlite/duckdb) |
| `ConnectionInfo` | `ConnectionInfo[conn]` | Inspect connection details (dsn, kind, tx) |
| `Disconnect` | `Disconnect[conn]` | Close a database connection |
| `Ping` | `Ping[conn]` | Check connectivity for a database connection |

## `cookies`

| Function | Usage | Summary |
|---|---|---|
| `CookiesHeader` | `CookiesHeader[cookies]` | Build a Cookie header string from an assoc |
| `GetResponseCookies` | `GetResponseCookies[response]` | Parse Set-Cookie headers from a response map |

## `copy`

| Function | Usage | Summary |
|---|---|---|
| `Copy` | `Copy[src, dst, opts?]` | Copy file or directory (Recursive option) |

## `create`

| Function | Usage | Summary |
|---|---|---|
| `DatasetFromRows` | `DatasetFromRows[rows]` | Create dataset from list of row assocs |
| `FrameFromRows` | `FrameFromRows[rows]` | Create a Frame from assoc rows |

## `cron`

| Function | Usage | Summary |
|---|---|---|
| `Cron` | `Cron[expr, body]` | Schedule with cron expression (held) |

## `crypto`

| Function | Usage | Summary |
|---|---|---|
| `AeadDecrypt` | `AeadDecrypt[envelope, key, opts]` | Decrypt AEAD envelope |
| `AeadEncrypt` | `AeadEncrypt[plaintext, key, opts]` | Encrypt with AEAD (ChaCha20-Poly1305) |
| `AeadKeyGen` | `AeadKeyGen[opts]` | Generate AEAD key (ChaCha20-Poly1305) |
| `Hash` | `Hash[input, opts]` | Compute digest (BLAKE3/SHA-256) |
| `Hkdf` | `Hkdf[ikm, opts]` | HKDF (SHA-256 or SHA-512) |
| `Hmac` | `Hmac[message, key, opts]` | HMAC (SHA-256 or SHA-512) |
| `HmacVerify` | `HmacVerify[message, key, signature, opts]` | Verify HMAC signature |
| `JwtSign` | `JwtSign[claims, key, opts]` | Sign JWT (HS256 or EdDSA) |
| `JwtVerify` | `JwtVerify[jwt, keys, opts]` | Verify JWT and return claims |
| `KeypairGenerate` | `KeypairGenerate[opts]` | Generate signing keypair (Ed25519) |
| `PasswordHash` | `PasswordHash[password, opts]` | Password hash with Argon2id (PHC string) |
| `PasswordVerify` | `PasswordVerify[password, hash]` | Verify Argon2id password hash |
| `RandomBytes` | `RandomBytes[len, opts]` | Generate cryptographically secure random bytes |
| `RandomHex` | `RandomHex[len]` | Generate random hex string of n bytes |
| `Sign` | `Sign[message, secretKey, opts]` | Sign message (Ed25519) |
| `Verify` | `Verify[message, signature, publicKey, opts]` | Verify signature (Ed25519) |

## `csv`

| Function | Usage | Summary |
|---|---|---|
| `CsvRead` | `CsvRead[csv, opts]` | Alias of ParseCSV |
| `CsvWrite` | `CsvWrite[rows, opts]` | Alias of RenderCSV |
| `FrameWriteCSV` | `FrameWriteCSV[path, frame, opts?]` | Write Frame to CSV file |
| `ParseCSV` | `ParseCSV[csv, opts]` | Parse CSV string to rows |
| `ReadCSV` | `ReadCSV[path, opts]` | Read and parse CSV file |
| `ReadCSVDataset` | `ReadCSVDataset[path, opts?]` | Read a CSV file into a dataset |
| `RenderCSV` | `RenderCSV[rows, opts]` | Render rows to CSV string |
| `WriteCSV` | `WriteCSV[path, rows, opts]` | Write rows to CSV file |

## `cursor`

| Function | Usage | Summary |
|---|---|---|
| `CursorInfo` | `CursorInfo[cursor]` | Inspect cursor details (dsn, kind, sql, offset) |
| `Fetch` | `Fetch[cursor, limit?]` | Fetch next batch of rows from a cursor |
| `SQLCursor` | `SQLCursor[conn, sql, params?]` | Run a query and return a cursor handle |

## `cv`

| Function | Usage | Summary |
|---|---|---|
| `MLCrossValidate` | `MLCrossValidate[data, opts]` | Cross-validate with simple split |

## `dataset`

| Function | Usage | Summary |
|---|---|---|
| `Agg` | `Agg[ds, aggs]` | Aggregate groups to single rows |
| `Cast` | `Cast[value, type]` | Cast a value to a target type (string, integer, real, boolean). |
| `Coalesce` | `Coalesce[values…]` | First non-null value |
| `Collect` | `Collect[ds, limit?, opts?]` | Materialize dataset rows as a list |
| `Columns` | `Columns[ds]` | List column names for a dataset |
| `Concat` | `Concat[inputs]` | Concatenate datasets by rows (schema-union) |
| `DatasetFromRows` | `DatasetFromRows[rows]` | Create dataset from list of row assocs |
| `DatasetSchema` | `DatasetSchema[ds]` | Describe schema for a dataset |
| `DistinctOn` | `DistinctOn[ds, keys, orderBy?, keepLast?]` | Keep one row per key with order policy |
| `ExplainDataset` | `ExplainDataset[ds]` | Inspect logical plan for a dataset |
| `ExplainSQL` | `ExplainSQL[ds]` | Render SQL for pushdown-capable parts |
| `Filter` | `Filter[pred, list]` | Keep elements where pred[x] is True |
| `FilterRows` | `FilterRows[ds, pred]` | Filter rows by predicate (held) |
| `GroupBy` | `GroupBy[ds, keys]` | Group rows by key(s) |
| `Import` | `Import[source, opts?]` | Import data from path/URL into Frame (default), Dataset (Target->"Dataset"), or Value. Automatically sniffs Type/Delimiter/Header. |
| `Join` | `Join[left, right, on, how?]` | Join two datasets on keys |
| `Keys` | `Keys[subject]` | Keys/columns for assoc, rows, Dataset, or Frame |
| `ReadCSVDataset` | `ReadCSVDataset[path, opts?]` | Read a CSV file into a dataset |
| `ReadJsonLinesDataset` | `ReadJsonLinesDataset[path, opts?]` | Read a JSONL file into a dataset |
| `RenameCols` | `RenameCols[ds, mapping]` | Rename columns via mapping |
| `Select` | `Select[ds, cols]` | Select/compute columns |
| `SelectCols` | `SelectCols[ds, cols]` | Select subset of columns by name |
| `Table` | `Table[conn, name]` | Reference a table as a Dataset |
| `Union` | `Union[inputs, byColumns?]` | Union multiple datasets (by columns) |
| `UnionByPosition` | `UnionByPosition[ds1, ds2, …]` | Union datasets by column position. |
| `WithColumns` | `WithColumns[ds, defs]` | Add/compute new columns (held) |
| `WriteDataset` | `WriteDataset[conn, table, dataset, opts?]` | Write a Dataset into a table |
| `col` | `col[name]` | Column accessor helper for Dataset expressions. |

## `datetime`

| Function | Usage | Summary |
|---|---|---|
| `DateFormat` | `DateFormat[dt, fmt?]` | Format DateTime or epochMs to string |
| `DateParse` | `DateParse[s]` | Parse date/time string to epochMs |
| `DateTime` | `DateTime[spec]` | Build/parse DateTime assoc (UTC) |

## `db`

| Function | Usage | Summary |
|---|---|---|
| `Begin` | `Begin[conn]` | Begin a transaction |
| `Commit` | `Commit[conn]` | Commit the current transaction |
| `Connect` | `Connect[dsn|opts]` | Open database connection (mock/sqlite/duckdb) |
| `ConnectionInfo` | `ConnectionInfo[conn]` | Inspect connection details (dsn, kind, tx) |
| `CursorInfo` | `CursorInfo[cursor]` | Inspect cursor details (dsn, kind, sql, offset) |
| `Disconnect` | `Disconnect[conn]` | Close a database connection |
| `Exec` | `Exec[conn, sql, params?]` | Execute DDL/DML (non-SELECT) |
| `Fetch` | `Fetch[cursor, limit?]` | Fetch next batch of rows from a cursor |
| `InsertRows` | `InsertRows[conn, table, rows]` | Insert multiple rows (assoc list) into a table |
| `ListTables` | `ListTables[conn]` | List tables on a connection |
| `Ping` | `Ping[conn]` | Check connectivity for a database connection |
| `RegisterTable` | `RegisterTable[conn, name, rows]` | Register in-memory rows as a table (mock) |
| `Rollback` | `Rollback[conn]` | Rollback the current transaction |
| `SQL` | `SQL[conn, sql, params?]` | Run a SELECT query and return rows |
| `SQLCursor` | `SQLCursor[conn, sql, params?]` | Run a query and return a cursor handle |
| `Table` | `Table[conn, name]` | Reference a table as a Dataset |
| `UpsertRows` | `UpsertRows[conn, table, rows, keys?]` | Upsert rows (assoc list) into a table |
| `WriteDataset` | `WriteDataset[conn, table, dataset, opts?]` | Write a Dataset into a table |

## `decode`

| Function | Usage | Summary |
|---|---|---|
| `AudioDecode` | `AudioDecode[input, opts]` | Decode audio to raw (s16le) or WAV |
| `ImageDecode` | `ImageDecode[input, opts]` | Decode image to raw or reencoded bytes |
| `NetDecoder` | `NetDecoder[net]` | Get network output decoder |

## `decomposition`

| Function | Usage | Summary |
|---|---|---|
| `KCore` | `KCore[graph, k]` | Induced subgraph with k-core |

## `delete`

| Function | Usage | Summary |
|---|---|---|
| `PathRemove` | `PathRemove[path, opts?]` | Alias: remove file or directory |
| `VectorDelete` | `VectorDelete[store, ids]` | Delete items by ids |

## `diff`

| Function | Usage | Summary |
|---|---|---|
| `GitDiff` | `GitDiff[opts?]` | Diff against base and optional paths |

## `distinct`

| Function | Usage | Summary |
|---|---|---|
| `Distinct` | `Distinct[ds, cols?]` | Drop duplicate rows (optionally by columns) |
| `DistinctOn` | `DistinctOn[ds, keys, orderBy?, keepLast?]` | Keep one row per key with order policy |
| `FrameDistinct` | `FrameDistinct[frame, cols?]` | Distinct rows in Frame (optional columns) |

## `duration`

| Function | Usage | Summary |
|---|---|---|
| `AddDuration` | `AddDuration[dt, dur]` | Add duration to DateTime/epochMs |
| `DiffDuration` | `DiffDuration[a, b]` | Difference between DateTimes |
| `Duration` | `Duration[spec]` | Build Duration assoc from ms or fields |
| `DurationParse` | `DurationParse[s]` | Parse human duration (e.g., 1h30m) |

## `edit`

| Function | Usage | Summary |
|---|---|---|
| `AudioChannelMix` | `AudioChannelMix[input, opts]` | Convert channel count (mono/stereo) |
| `AudioConcat` | `AudioConcat[inputs, opts]` | Concatenate multiple inputs |
| `AudioFade` | `AudioFade[input, opts]` | Fade in/out |
| `AudioGain` | `AudioGain[input, opts]` | Apply gain in dB or linear |
| `AudioResample` | `AudioResample[input, opts]` | Resample to new sample rate |
| `AudioTrim` | `AudioTrim[input, opts]` | Trim audio by time range |

## `encode`

| Function | Usage | Summary |
|---|---|---|
| `AudioEncode` | `AudioEncode[raw, opts]` | Encode raw PCM to WAV |
| `ImageEncode` | `ImageEncode[input, encoding]` | Encode raw pixels or reencode bytes |
| `NetEncoder` | `NetEncoder[net]` | Get network input encoder |

## `encoding`

| Function | Usage | Summary |
|---|---|---|
| `Base64Decode` | `Base64Decode[text, opts]` | Decode base64 to bytes |
| `Base64Encode` | `Base64Encode[bytes, opts]` | Encode bytes to base64 |
| `HexDecode` | `HexDecode[text]` | Decode hex to bytes |
| `HexEncode` | `HexEncode[bytes]` | Encode bytes to hex |
| `TextDecode` | `TextDecode[bytes, opts]` | Decode bytes to text (utf-8) |
| `TextEncode` | `TextEncode[text, opts]` | Encode text to bytes (utf-8) |

## `env`

| Function | Usage | Summary |
|---|---|---|
| `GetEnv` | `GetEnv[name]` | Read environment variable |
| `SetEnv` | `SetEnv[name, value]` | Set environment variable |

## `explain`

| Function | Usage | Summary |
|---|---|---|
| `ExplainDataset` | `ExplainDataset[ds]` | Inspect logical plan for a dataset |
| `ExplainSQL` | `ExplainSQL[ds]` | Render SQL for pushdown-capable parts |

## `export`

| Function | Usage | Summary |
|---|---|---|
| `Export` | `Export[symbols]` | Mark symbol(s) as public |

## `expr`

| Function | Usage | Summary |
|---|---|---|
| `col` | `col[name]` | Column accessor helper for Dataset expressions. |

## `ffmpeg`

| Function | Usage | Summary |
|---|---|---|
| `MediaConcat` | `MediaConcat[inputs, opts]` | Concatenate media files |
| `MediaExtractAudio` | `MediaExtractAudio[input, opts]` | Extract audio track to format |
| `MediaMux` | `MediaMux[video, audio, opts]` | Mux separate video+audio into container |
| `MediaPipeline` | `MediaPipeline[desc]` | Run arbitrary ffmpeg pipeline (builder) |
| `MediaProbe` | `MediaProbe[input]` | Probe media via ffprobe |
| `MediaThumbnail` | `MediaThumbnail[input, opts]` | Extract video frame as image |
| `MediaTranscode` | `MediaTranscode[input, opts]` | Transcode media with options |

## `filter`

| Function | Usage | Summary |
|---|---|---|
| `FilterRows` | `FilterRows[ds, pred]` | Filter rows by predicate (held) |
| `FrameFilter` | `FrameFilter[frame, pred]` | Filter rows in a Frame |
| `Reject` | `Reject[pred, list]` | Drop elements where pred[x] is True |

## `fixedpoint`

| Function | Usage | Summary |
|---|---|---|
| `FixedPoint` | `FixedPoint[f, x]` | Iterate f until convergence |
| `FixedPointList` | `FixedPointList[f, x]` | List of iterates until convergence |

## `flow`

| Function | Usage | Summary |
|---|---|---|
| `MaxFlow` | `MaxFlow[graph, src, dst]` | Maximum flow value and cut |

## `fold`

| Function | Usage | Summary |
|---|---|---|
| `FoldList` | `FoldList[f, init, list]` | Cumulative fold producing intermediates |
| `Reduce` | `Reduce[f, init?, list]` | Fold list with function |
| `Scan` | `Scan[f, init?, list]` | Prefix scan with function |

## `frame`

| Function | Usage | Summary |
|---|---|---|
| `Columns` | `Columns[ds]` | List column names for a dataset |
| `Filter` | `Filter[pred, list]` | Keep elements where pred[x] is True |
| `FrameCollect` | `FrameCollect[frame]` | Materialize Frame to list of rows |
| `FrameColumns` | `FrameColumns[frame]` | List column names for a Frame |
| `FrameDescribe` | `FrameDescribe[frame, opts?]` | Quick stats by columns |
| `FrameDistinct` | `FrameDistinct[frame, cols?]` | Distinct rows in Frame (optional columns) |
| `FrameFilter` | `FrameFilter[frame, pred]` | Filter rows in a Frame |
| `FrameFromRows` | `FrameFromRows[rows]` | Create a Frame from assoc rows |
| `FrameHead` | `FrameHead[frame, n?]` | Take first n rows from Frame |
| `FrameJoin` | `FrameJoin[left, right, on?, opts?]` | Join two Frames by keys |
| `FrameOffset` | `FrameOffset[frame, n]` | Skip first n rows of Frame |
| `FrameSelect` | `FrameSelect[frame, spec]` | Select/compute columns in Frame |
| `FrameSort` | `FrameSort[frame, by]` | Sort Frame by columns |
| `FrameTail` | `FrameTail[frame, n?]` | Take last n rows from Frame |
| `FrameUnion` | `FrameUnion[frames…]` | Union Frames by columns (schema union) |
| `FrameWriteCSV` | `FrameWriteCSV[path, frame, opts?]` | Write Frame to CSV file |
| `FrameWriteJSONLines` | `FrameWriteJSONLines[path, frame, opts?]` | Write Frame rows as JSON Lines |
| `Import` | `Import[source, opts?]` | Import data from path/URL into Frame (default), Dataset (Target->"Dataset"), or Value. Automatically sniffs Type/Delimiter/Header. |
| `Keys` | `Keys[subject]` | Keys/columns for assoc, rows, Dataset, or Frame |
| `Select` | `Select[ds, cols]` | Select/compute columns |

## `fs`

| Function | Usage | Summary |
|---|---|---|
| `CancelWatch` | `CancelWatch[token]` | Cancel a directory watch |
| `Copy` | `Copy[src, dst, opts?]` | Copy file or directory (Recursive option) |
| `Download` | `Download[url, path, opts]` | Download URL to file (http/https) |
| `DownloadStream` | `DownloadStream[url, path, opts]` | Stream download URL directly to file |
| `FileExistsQ` | `FileExistsQ[path]` | Does path exist? |
| `Glob` | `Glob[pattern]` | Expand glob pattern to matching paths |
| `Gunzip` | `Gunzip[dataOrPath, opts?]` | Gunzip-decompress a string or a .gz file; optionally write to path. |
| `Gzip` | `Gzip[dataOrPath, opts?]` | Gzip-compress a string or a file; optionally write to path. |
| `ListDirectory` | `ListDirectory[path]` | List names in directory |
| `MakeDirectory` | `MakeDirectory[path, opts?]` | Create a directory (Parents option) |
| `Move` | `Move[src, dst]` | Move or rename a file/directory |
| `PathRemove` | `PathRemove[path, opts?]` | Alias: remove file or directory |
| `ReadBytes` | `ReadBytes[path]` | Read entire file as bytes |
| `ReadFile` | `ReadFile[path]` | Read entire file as string |
| `ReadLines` | `ReadLines[path]` | Read file into list of lines |
| `Stat` | `Stat[path]` | Basic file metadata as assoc |
| `Symlink` | `Symlink[src, dst]` | Create a symbolic link |
| `Tar` | `Tar[dest, inputs, opts?]` | Create a .tar (optionally .tar.gz) archive from inputs. |
| `TarExtract` | `TarExtract[src, dest]` | Extract a .tar or .tar.gz archive into a directory. |
| `TempDir` | `TempDir[]` | Create a unique temporary directory |
| `TempFile` | `TempFile[]` | Create a unique temporary file |
| `Touch` | `Touch[path]` | Create file if missing (update mtime) |
| `WatchDirectory` | `WatchDirectory[path, handler, opts?]` | Watch directory and stream events (held) |
| `WriteBytes` | `WriteBytes[path, bytes]` | Write bytes to file |
| `WriteFile` | `WriteFile[path, content]` | Write stringified content to file |
| `Zip` | `Zip[dest, inputs]` | Create a .zip archive from files/directories. |
| `ZipExtract` | `ZipExtract[src, dest]` | Extract a .zip archive into a directory. |

## `functional`

| Function | Usage | Summary |
|---|---|---|
| `Apply` | `Apply[f, list]` | Apply head to list elements: Apply[f, {…}] |
| `Compose` | `Compose[f, g, …]` | Compose functions left-to-right |
| `ConstantFunction` | `ConstantFunction[c]` | Constant function returning c |
| `FixedPoint` | `FixedPoint[f, x]` | Iterate f until convergence |
| `FixedPointList` | `FixedPointList[f, x]` | List of iterates until convergence |
| `FoldList` | `FoldList[f, init, list]` | Cumulative fold producing intermediates |
| `Identity` | `Identity[x]` | Identity function: returns its argument |
| `Nest` | `Nest[f, x, n]` | Nest function n times: Nest[f, x, n] |
| `NestList` | `NestList[f, x, n]` | Nest and collect intermediate values |
| `RightCompose` | `RightCompose[f, g, …]` | Compose functions right-to-left |
| `Through` | `Through[fs, x]` | Through[{f,g}, x] applies each to x |

## `generic`

| Function | Usage | Summary |
|---|---|---|
| `Add` | `Add[target, value]` | Add value to a collection (alias of Insert for some types) |
| `Close` | `Close[handle]` | Close an open handle (cursor, channel) |
| `Columns` | `Columns[ds]` | List column names for a dataset |
| `Contains` | `Contains[container, item]` | Membership test for strings/lists/sets/assocs |
| `ContainsKeyQ` | `ContainsKeyQ[subject, key]` | Key membership for assoc/rows/Dataset/Frame |
| `ContainsQ` | `ContainsQ[container, item]` | Alias: membership predicate |
| `Count` | `Count[list, value|pred]` | Count elements equal to value or matching predicate |
| `Describe` | `Describe[name, items, opts?]` | Define a test suite (held). |
| `Distinct` | `Distinct[ds, cols?]` | Drop duplicate rows (optionally by columns) |
| `EmptyQ` | `EmptyQ[x]` | Is list/string/assoc empty? |
| `Export` | `Export[symbols]` | Mark symbol(s) as public |
| `Filter` | `Filter[pred, list]` | Keep elements where pred[x] is True |
| `HasKeyQ` | `HasKeyQ[subject, key]` | Alias: key membership predicate |
| `Head` | `Head[ds, n]` | Take first n rows |
| `Import` | `Import[source, opts?]` | Import data from path/URL into Frame (default), Dataset (Target->"Dataset"), or Value. Automatically sniffs Type/Delimiter/Header. |
| `ImportBytes` | `ImportBytes[bytes, opts?]` | Parse byte buffer using Type (text/json/etc.) |
| `ImportString` | `ImportString[content, opts?]` | Parse in-memory strings into Frame/Dataset/Value. Automatically sniffs Type if missing. |
| `Info` | `Info[target]` | Information about a handle (Graph, etc.) |
| `Insert` | `Insert[target, value]` | Insert into collection or structure (dispatched) |
| `Join` | `Join[left, right, on, how?]` | Join two datasets on keys |
| `Keys` | `Keys[subject]` | Keys/columns for assoc, rows, Dataset, or Frame |
| `Length` | `Length[x]` | Length of a list or string. |
| `MemberQ` | `MemberQ[container, item]` | Alias: membership predicate (Contains) |
| `Offset` | `Offset[ds, n]` | Skip first n rows |
| `Remove` | `Remove[path, opts?]` | Remove a file or directory (Recursive option) |
| `Search` | `Search[target, query, opts?]` | Search within a store or index (VectorStore, Index) |
| `Select` | `Select[ds, cols]` | Select/compute columns |
| `Sniff` | `Sniff[source]` | Suggest Type and options for a source (file/url/string/bytes). |
| `Sort` | `Sort[ds, by, opts?]` | Sort rows by columns |
| `Tail` | `Tail[ds, n]` | Take last n rows |
| `Union` | `Union[inputs, byColumns?]` | Union multiple datasets (by columns) |

## `git`

| Function | Usage | Summary |
|---|---|---|
| `GitAdd` | `GitAdd[paths, opts?]` | Stage files for commit |
| `GitApply` | `GitApply[patch, opts?]` | Apply a patch (or check only) |
| `GitBranch` | `GitBranch[name, opts?]` | Create a new branch |
| `GitBranchList` | `GitBranchList[]` | List local branches |
| `GitCommit` | `GitCommit[message, opts?]` | Create a commit with message |
| `GitCurrentBranch` | `GitCurrentBranch[]` | Current branch name |
| `GitDiff` | `GitDiff[opts?]` | Diff against base and optional paths |
| `GitEnsureRepo` | `GitEnsureRepo[opts?]` | Ensure Cwd is a git repo (init if needed) |
| `GitFeatureBranch` | `GitFeatureBranch[opts?]` | Create and switch to a feature branch |
| `GitFetch` | `GitFetch[remote?]` | Fetch from remote |
| `GitInit` | `GitInit[opts?]` | Initialize a new git repository |
| `GitLog` | `GitLog[opts?]` | List commits with formatting options |
| `GitPull` | `GitPull[remote?, opts?]` | Pull from remote |
| `GitPush` | `GitPush[opts?]` | Push to remote |
| `GitRemoteList` | `GitRemoteList[]` | List remotes |
| `GitRoot` | `GitRoot[]` | Path to repository root (Null if absent) |
| `GitSmartCommit` | `GitSmartCommit[opts?]` | Stage + conventional commit (auto msg option) |
| `GitStatus` | `GitStatus[opts?]` | Status (porcelain) with branch/ahead/behind/changes |
| `GitStatusSummary` | `GitStatusSummary[opts?]` | Summarize status counts and branch |
| `GitSwitch` | `GitSwitch[name, opts?]` | Switch to branch (optionally create) |
| `GitSyncUpstream` | `GitSyncUpstream[opts?]` | Fetch, rebase (or merge), and push upstream |
| `GitVersion` | `GitVersion[]` | Get git client version string |

## `glob`

| Function | Usage | Summary |
|---|---|---|
| `Glob` | `Glob[pattern]` | Expand glob pattern to matching paths |

## `graph`

| Function | Usage | Summary |
|---|---|---|
| `AddEdges` | `AddEdges[graph, edges]` | Add edges with optional keys and weights |
| `AddNodes` | `AddNodes[graph, nodes]` | Add nodes by id and attributes |
| `BFS` | `BFS[graph, start, opts?]` | Breadth-first search order |
| `DFS` | `DFS[graph, start, opts?]` | Depth-first search order |
| `Graph` | `Graph[opts?]` | Create a graph handle |
| `GraphInfo` | `GraphInfo[graph]` | Summary and counts for graph |
| `KCore` | `KCore[graph, k]` | Induced subgraph with k-core |
| `ListEdges` | `ListEdges[graph, opts?]` | List edges |
| `ListNodes` | `ListNodes[graph, opts?]` | List nodes |
| `MaxFlow` | `MaxFlow[graph, src, dst]` | Maximum flow value and cut |
| `MinimumSpanningTree` | `MinimumSpanningTree[graph]` | Edges in a minimum spanning tree |
| `Neighbors` | `Neighbors[graph, id, opts?]` | Neighbor node ids for a node |
| `PageRank` | `PageRank[graph, opts?]` | PageRank centrality scores |
| `RemoveEdges` | `RemoveEdges[graph, edges]` | Remove edges by id or (src,dst,key) |
| `RemoveNodes` | `RemoveNodes[graph, ids]` | Remove nodes by id |
| `ShortestPaths` | `ShortestPaths[graph, start, opts?]` | Shortest path distances from start |

## `graphs`

| Function | Usage | Summary |
|---|---|---|
| `Graph` | `Graph[opts?]` | Create a graph handle |

## `groupby`

| Function | Usage | Summary |
|---|---|---|
| `GroupBy` | `GroupBy[ds, keys]` | Group rows by key(s) |

## `hash`

| Function | Usage | Summary |
|---|---|---|
| `Hash` | `Hash[input, opts]` | Compute digest (BLAKE3/SHA-256) |

## `html`

| Function | Usage | Summary |
|---|---|---|
| `HtmlEscape` | `HtmlEscape[s]` | Escape string for HTML |
| `HtmlUnescape` | `HtmlUnescape[s]` | Unescape HTML-escaped string |
| `RespondHtml` | `RespondHtml[html, opts]` | Build an HTML response for HttpServe |

## `http`

| Function | Usage | Summary |
|---|---|---|
| `CookiesHeader` | `CookiesHeader[cookies]` | Build a Cookie header string from an assoc |
| `Download` | `Download[url, path, opts]` | Download URL to file (http/https) |
| `DownloadStream` | `DownloadStream[url, path, opts]` | Stream download URL directly to file |
| `GetResponseCookies` | `GetResponseCookies[response]` | Parse Set-Cookie headers from a response map |
| `HttpDelete` | `HttpDelete[url, opts]` | HTTP DELETE request (http/https) |
| `HttpGet` | `HttpGet[url, opts]` | HTTP GET request (http/https) |
| `HttpHead` | `HttpHead[url, opts]` | HTTP HEAD request (http/https) |
| `HttpOptions` | `HttpOptions[url, opts]` | HTTP OPTIONS request (http/https) |
| `HttpPatch` | `HttpPatch[url, body, opts]` | HTTP PATCH request (http/https) |
| `HttpPost` | `HttpPost[url, body, opts]` | HTTP POST request (http/https) |
| `HttpPut` | `HttpPut[url, body, opts]` | HTTP PUT request (http/https) |
| `HttpRequest` | `HttpRequest[options]` | Generic HTTP request via options object |
| `HttpServe` | `HttpServe[handler, opts]` | Start an HTTP server and handle requests with a function |
| `HttpServeRoutes` | `HttpServeRoutes[routes, opts]` | Start an HTTP server with a routes table |
| `HttpServeTls` | `HttpServeTls[handler, opts]` | Start an HTTPS server with TLS cert/key |
| `HttpServerAddr` | `HttpServerAddr[server]` | Get bound address for a server id |
| `HttpServerStop` | `HttpServerStop[server]` | Stop a running HTTP server by id |
| `PathMatch` | `PathMatch[pattern, path]` | Match a path pattern like /users/:id against a path |
| `RespondBytes` | `RespondBytes[bytes, opts]` | Build a binary response for HttpServe |
| `RespondFile` | `RespondFile[path, opts]` | Build a file response for HttpServe |
| `RespondHtml` | `RespondHtml[html, opts]` | Build an HTML response for HttpServe |
| `RespondJson` | `RespondJson[value, opts]` | Build a JSON response for HttpServe |
| `RespondNoContent` | `RespondNoContent[opts]` | Build an empty 204/205/304 response |
| `RespondRedirect` | `RespondRedirect[location, opts]` | Build a redirect response (Location header) |
| `RespondText` | `RespondText[text, opts]` | Build a text response for HttpServe |

## `image`

| Function | Usage | Summary |
|---|---|---|
| `ImageCanvas` | `ImageCanvas[opts]` | Create a blank canvas (PNG) |
| `ImageConvert` | `ImageConvert[input, format, opts]` | Convert image format |
| `ImageCrop` | `ImageCrop[input, opts]` | Crop image by rect or gravity |
| `ImageDecode` | `ImageDecode[input, opts]` | Decode image to raw or reencoded bytes |
| `ImageEncode` | `ImageEncode[input, encoding]` | Encode raw pixels or reencode bytes |
| `ImageInfo` | `ImageInfo[input, opts]` | Read basic image info |
| `ImagePad` | `ImagePad[input, opts]` | Pad image to target size |
| `ImageResize` | `ImageResize[input, opts]` | Resize image (contain/cover) |
| `ImageSave` | `ImageSave[input, output, encoding]` | Encode and write image to path |
| `ImageThumbnail` | `ImageThumbnail[input, opts]` | Create thumbnail (cover) |
| `ImageTransform` | `ImageTransform[input, pipeline]` | Apply pipeline of operations |
| `MediaThumbnail` | `MediaThumbnail[input, opts]` | Extract video frame as image |

## `import`

| Function | Usage | Summary |
|---|---|---|
| `Import` | `Import[source, opts?]` | Import data from path/URL into Frame (default), Dataset (Target->"Dataset"), or Value. Automatically sniffs Type/Delimiter/Header. |
| `ImportBytes` | `ImportBytes[bytes, opts?]` | Parse byte buffer using Type (text/json/etc.) |
| `ImportString` | `ImportString[content, opts?]` | Parse in-memory strings into Frame/Dataset/Value. Automatically sniffs Type if missing. |
| `Sniff` | `Sniff[source]` | Suggest Type and options for a source (file/url/string/bytes). |

## `index`

| Function | Usage | Summary |
|---|---|---|
| `GitAdd` | `GitAdd[paths, opts?]` | Stage files for commit |

## `inference`

| Function | Usage | Summary |
|---|---|---|
| `MLApply` | `MLApply[model, x, opts]` | Apply a trained model to input |
| `NetApply` | `NetApply[net, x, opts]` | Apply network to input |

## `info`

| Function | Usage | Summary |
|---|---|---|
| `VectorCount` | `VectorCount[store]` | Count items in store |

## `inspect`

| Function | Usage | Summary |
|---|---|---|
| `FrameHead` | `FrameHead[frame, n?]` | Take first n rows from Frame |
| `FrameTail` | `FrameTail[frame, n?]` | Take last n rows from Frame |
| `Head` | `Head[ds, n]` | Take first n rows |
| `Tail` | `Tail[ds, n]` | Take last n rows |

## `introspect`

| Function | Usage | Summary |
|---|---|---|
| `ConnectionInfo` | `ConnectionInfo[conn]` | Inspect connection details (dsn, kind, tx) |
| `CursorInfo` | `CursorInfo[cursor]` | Inspect cursor details (dsn, kind, sql, offset) |
| `GraphInfo` | `GraphInfo[graph]` | Summary and counts for graph |
| `MLProperty` | `MLProperty[model, prop]` | Inspect trained model properties |
| `NetProperty` | `NetProperty[net, prop]` | Inspect network properties |
| `NetSummary` | `NetSummary[net]` | Summarize network structure |
| `ProcessInfo` | `ProcessInfo[proc]` | Inspect process handle (pid, running, exit) |

## `introspection`

| Function | Usage | Summary |
|---|---|---|
| `Describe` | `Describe[name, items, opts?]` | Define a test suite (held). |
| `Info` | `Info[target]` | Information about a handle (Graph, etc.) |

## `io`

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

## `iteration`

| Function | Usage | Summary |
|---|---|---|
| `Nest` | `Nest[f, x, n]` | Nest function n times: Nest[f, x, n] |
| `NestList` | `NestList[f, x, n]` | Nest and collect intermediate values |

## `join`

| Function | Usage | Summary |
|---|---|---|
| `FrameJoin` | `FrameJoin[left, right, on?, opts?]` | Join two Frames by keys |

## `json`

| Function | Usage | Summary |
|---|---|---|
| `FrameWriteJSONLines` | `FrameWriteJSONLines[path, frame, opts?]` | Write Frame rows as JSON Lines |
| `FromJson` | `FromJson[json]` | Parse JSON string to value |
| `JsonEscape` | `JsonEscape[s]` | Escape string for JSON |
| `JsonParse` | `JsonParse[json]` | Alias of FromJson |
| `JsonStringify` | `JsonStringify[value, opts]` | Alias of ToJson |
| `JsonUnescape` | `JsonUnescape[s]` | Unescape JSON-escaped string |
| `ReadJsonLinesDataset` | `ReadJsonLinesDataset[path, opts?]` | Read a JSONL file into a dataset |
| `RespondJson` | `RespondJson[value, opts]` | Build a JSON response for HttpServe |
| `ToJson` | `ToJson[value, opts]` | Serialize value to JSON string |

## `jwt`

| Function | Usage | Summary |
|---|---|---|
| `JwtSign` | `JwtSign[claims, key, opts]` | Sign JWT (HS256 or EdDSA) |
| `JwtVerify` | `JwtVerify[jwt, keys, opts]` | Verify JWT and return claims |

## `kdf`

| Function | Usage | Summary |
|---|---|---|
| `Hkdf` | `Hkdf[ikm, opts]` | HKDF (SHA-256 or SHA-512) |
| `PasswordHash` | `PasswordHash[password, opts]` | Password hash with Argon2id (PHC string) |
| `PasswordVerify` | `PasswordVerify[password, hash]` | Verify Argon2id password hash |

## `layer`

| Function | Usage | Summary |
|---|---|---|
| `ActivationLayer` | `ActivationLayer[kind, opts]` | Activation layer (Relu/Tanh/Sigmoid) |
| `AddLayer` | `AddLayer[opts]` | Elementwise add layer |
| `BatchNormLayer` | `BatchNormLayer[opts]` | Batch normalization layer |
| `ConcatLayer` | `ConcatLayer[axis]` | Concatenate along axis |
| `ConvolutionLayer` | `ConvolutionLayer[opts]` | 2D convolution layer |
| `DropoutLayer` | `DropoutLayer[p]` | Dropout probability p |
| `EmbeddingLayer` | `EmbeddingLayer[opts]` | Embedding lookup layer |
| `FlattenLayer` | `FlattenLayer[opts]` | Flatten to 1D |
| `LayerNormLayer` | `LayerNormLayer[opts]` | Layer normalization layer |
| `LinearLayer` | `LinearLayer[opts]` | Linear (fully-connected) layer |
| `MulLayer` | `MulLayer[opts]` | Elementwise multiply layer |
| `PoolingLayer` | `PoolingLayer[opts]` | Pooling layer (Max/Avg) |
| `ReshapeLayer` | `ReshapeLayer[shape]` | Reshape to given shape |
| `SoftmaxLayer` | `SoftmaxLayer[opts]` | Softmax over last dimension |
| `TransposeLayer` | `TransposeLayer[perm]` | Transpose dimensions |

## `layout`

| Function | Usage | Summary |
|---|---|---|
| `Figure` | `Figure[items, opts]` | Compose multiple charts in a grid |

## `lifecycle`

| Function | Usage | Summary |
|---|---|---|
| `Close` | `Close[handle]` | Close an open handle (cursor, channel) |
| `Graph` | `Graph[opts?]` | Create a graph handle |

## `link`

| Function | Usage | Summary |
|---|---|---|
| `Symlink` | `Symlink[src, dst]` | Create a symbolic link |

## `list`

| Function | Usage | Summary |
|---|---|---|
| `All` | `All[list, pred?]` | True if all match (optionally with pred) |
| `Any` | `Any[list, pred?]` | True if any matches (optionally with pred) |
| `CountBy` | `CountBy[f, list]` | Counts by key function (assoc) |
| `Drop` | `Drop[list, n]` | Drop first n (last if negative) |
| `DropWhile` | `DropWhile[pred, list]` | Drop while pred[x] holds |
| `Filter` | `Filter[pred, list]` | Keep elements where pred[x] is True |
| `Find` | `Find[pred, list]` | First element where pred[x] |
| `Flatten` | `Flatten[list, levels?]` | Flatten by levels (default 1) |
| `Join` | `Join[left, right, on, how?]` | Join two datasets on keys |
| `ListEdges` | `ListEdges[graph, opts?]` | List edges |
| `ListNodes` | `ListNodes[graph, opts?]` | List nodes |
| `MapIndexed` | `MapIndexed[f, list]` | Map with index (1-based) |
| `Part` | `Part[subject, index]` | Index into list/assoc |
| `Partition` | `Partition[list, n, step?]` | Partition into fixed-size chunks |
| `Position` | `Position[pred, list]` | 1-based index of first match |
| `Range` | `Range[a, b, step?]` | Create numeric range |
| `Reduce` | `Reduce[f, init?, list]` | Fold list with function |
| `Reject` | `Reject[pred, list]` | Drop elements where pred[x] is True |
| `Scan` | `Scan[f, init?, list]` | Prefix scan with function |
| `Slice` | `Slice[list, start, len?]` | Slice list by start and length |
| `Take` | `Take[list, n]` | Take first n (last if negative) |
| `TakeWhile` | `TakeWhile[pred, list]` | Take while pred[x] holds |
| `Tally` | `Tally[list]` | Counts by value (assoc) |
| `Total` | `Total[list]` | Sum elements in a list |
| `Transpose` | `Transpose[rows]` | Transpose list of lists |
| `Union` | `Union[inputs, byColumns?]` | Union multiple datasets (by columns) |
| `Unique` | `Unique[list]` | Stable deduplicate list |
| `Unzip` | `Unzip[pairs]` | Unzip pairs into two lists |

## `log`

| Function | Usage | Summary |
|---|---|---|
| `GitLog` | `GitLog[opts?]` | List commits with formatting options |

## `logic`

| Function | Usage | Summary |
|---|---|---|
| `All` | `All[list, pred?]` | True if all match (optionally with pred) |
| `And` | `And[args…]` | Logical AND (short-circuit) |
| `Any` | `Any[list, pred?]` | True if any matches (optionally with pred) |
| `Equal` | `Equal[args…]` | Test equality across arguments |
| `EvenQ` | `EvenQ[n]` | Is integer even? |
| `Greater` | `Greater[args…]` | Strictly decreasing sequence |
| `GreaterEqual` | `GreaterEqual[args…]` | Non-increasing sequence |
| `Less` | `Less[args…]` | Strictly increasing sequence |
| `LessEqual` | `LessEqual[args…]` | Non-decreasing sequence |
| `Not` | `Not[x]` | Logical NOT |
| `OddQ` | `OddQ[n]` | Is integer odd? |
| `Or` | `Or[args…]` | Logical OR (short-circuit) |

## `mac`

| Function | Usage | Summary |
|---|---|---|
| `Hmac` | `Hmac[message, key, opts]` | HMAC (SHA-256 or SHA-512) |
| `HmacVerify` | `HmacVerify[message, key, signature, opts]` | Verify HMAC signature |

## `map`

| Function | Usage | Summary |
|---|---|---|
| `MapIndexed` | `MapIndexed[f, list]` | Map with index (1-based) |

## `math`

| Function | Usage | Summary |
|---|---|---|
| `Abs` | `Abs[x]` | Absolute value |
| `EvenQ` | `EvenQ[n]` | Is integer even? |
| `Max` | `Max[args]` | Maximum of values or list |
| `Min` | `Min[args]` | Minimum of values or list |
| `OddQ` | `OddQ[n]` | Is integer odd? |
| `Plus` | `Plus[a, b, …]` | Add numbers; Listable, Flat, Orderless. |
| `Range` | `Range[a, b, step?]` | Create numeric range |
| `Times` | `Times[a, b, …]` | Multiply numbers; Listable, Flat, Orderless. |
| `Total` | `Total[list]` | Sum elements in a list |

## `media`

| Function | Usage | Summary |
|---|---|---|
| `MediaConcat` | `MediaConcat[inputs, opts]` | Concatenate media files |
| `MediaExtractAudio` | `MediaExtractAudio[input, opts]` | Extract audio track to format |
| `MediaMux` | `MediaMux[video, audio, opts]` | Mux separate video+audio into container |
| `MediaPipeline` | `MediaPipeline[desc]` | Run arbitrary ffmpeg pipeline (builder) |
| `MediaProbe` | `MediaProbe[input]` | Probe media via ffprobe |
| `MediaThumbnail` | `MediaThumbnail[input, opts]` | Extract video frame as image |
| `MediaTranscode` | `MediaTranscode[input, opts]` | Transcode media with options |

## `metrics`

| Function | Usage | Summary |
|---|---|---|
| `ClassifyMeasurements` | `ClassifyMeasurements[model, data, opts]` | Evaluate classifier metrics |
| `PredictMeasurements` | `PredictMeasurements[model, data, opts]` | Evaluate regressor metrics |

## `ml`

| Function | Usage | Summary |
|---|---|---|
| `Classify` | `Classify[data, opts]` | Train a classifier (baseline/logistic) |
| `ClassifyMeasurements` | `ClassifyMeasurements[model, data, opts]` | Evaluate classifier metrics |
| `Cluster` | `Cluster[data, opts]` | Cluster points (prototype) |
| `DimensionReduce` | `DimensionReduce[data, opts]` | Reduce dimensionality (PCA-like) |
| `FeatureExtract` | `FeatureExtract[data, opts]` | Learn preprocessing (impute/encode/standardize) |
| `MLApply` | `MLApply[model, x, opts]` | Apply a trained model to input |
| `MLCrossValidate` | `MLCrossValidate[data, opts]` | Cross-validate with simple split |
| `MLProperty` | `MLProperty[model, prop]` | Inspect trained model properties |
| `MLTune` | `MLTune[data, opts]` | Parameter sweep with basic scoring |
| `Predict` | `Predict[data, opts]` | Train a regressor (baseline/linear) |
| `PredictMeasurements` | `PredictMeasurements[model, data, opts]` | Evaluate regressor metrics |

## `move`

| Function | Usage | Summary |
|---|---|---|
| `Move` | `Move[src, dst]` | Move or rename a file/directory |

## `mst`

| Function | Usage | Summary |
|---|---|---|
| `MinimumSpanningTree` | `MinimumSpanningTree[graph]` | Edges in a minimum spanning tree |

## `mutate`

| Function | Usage | Summary |
|---|---|---|
| `AddEdges` | `AddEdges[graph, edges]` | Add edges with optional keys and weights |
| `AddNodes` | `AddNodes[graph, nodes]` | Add nodes by id and attributes |

## `net`

| Function | Usage | Summary |
|---|---|---|
| `Download` | `Download[url, path, opts]` | Download URL to file (http/https) |
| `DownloadStream` | `DownloadStream[url, path, opts]` | Stream download URL directly to file |
| `HttpDelete` | `HttpDelete[url, opts]` | HTTP DELETE request (http/https) |
| `HttpGet` | `HttpGet[url, opts]` | HTTP GET request (http/https) |
| `HttpHead` | `HttpHead[url, opts]` | HTTP HEAD request (http/https) |
| `HttpOptions` | `HttpOptions[url, opts]` | HTTP OPTIONS request (http/https) |
| `HttpPatch` | `HttpPatch[url, body, opts]` | HTTP PATCH request (http/https) |
| `HttpPost` | `HttpPost[url, body, opts]` | HTTP POST request (http/https) |
| `HttpPut` | `HttpPut[url, body, opts]` | HTTP PUT request (http/https) |
| `HttpRequest` | `HttpRequest[options]` | Generic HTTP request via options object |
| `HttpServe` | `HttpServe[handler, opts]` | Start an HTTP server and handle requests with a function |
| `HttpServeRoutes` | `HttpServeRoutes[routes, opts]` | Start an HTTP server with a routes table |
| `HttpServeTls` | `HttpServeTls[handler, opts]` | Start an HTTPS server with TLS cert/key |
| `HttpServerAddr` | `HttpServerAddr[server]` | Get bound address for a server id |
| `HttpServerStop` | `HttpServerStop[server]` | Stop a running HTTP server by id |
| `NetChain` | `NetChain[layers, opts]` | Construct a sequential network from layers |
| `NetInitialize` | `NetInitialize[net, opts]` | Initialize network parameters |

## `nn`

| Function | Usage | Summary |
|---|---|---|
| `ActivationLayer` | `ActivationLayer[kind, opts]` | Activation layer (Relu/Tanh/Sigmoid) |
| `AddLayer` | `AddLayer[opts]` | Elementwise add layer |
| `BatchNormLayer` | `BatchNormLayer[opts]` | Batch normalization layer |
| `ConcatLayer` | `ConcatLayer[axis]` | Concatenate along axis |
| `ConvolutionLayer` | `ConvolutionLayer[opts]` | 2D convolution layer |
| `DropoutLayer` | `DropoutLayer[p]` | Dropout probability p |
| `EmbeddingLayer` | `EmbeddingLayer[opts]` | Embedding lookup layer |
| `FlattenLayer` | `FlattenLayer[opts]` | Flatten to 1D |
| `LayerNormLayer` | `LayerNormLayer[opts]` | Layer normalization layer |
| `LinearLayer` | `LinearLayer[opts]` | Linear (fully-connected) layer |
| `MulLayer` | `MulLayer[opts]` | Elementwise multiply layer |
| `NetApply` | `NetApply[net, x, opts]` | Apply network to input |
| `NetChain` | `NetChain[layers, opts]` | Construct a sequential network from layers |
| `NetDecoder` | `NetDecoder[net]` | Get network output decoder |
| `NetEncoder` | `NetEncoder[net]` | Get network input encoder |
| `NetInitialize` | `NetInitialize[net, opts]` | Initialize network parameters |
| `NetProperty` | `NetProperty[net, prop]` | Inspect network properties |
| `NetSummary` | `NetSummary[net]` | Summarize network structure |
| `NetTrain` | `NetTrain[net, data, opts]` | Train network on data |
| `PoolingLayer` | `PoolingLayer[opts]` | Pooling layer (Max/Avg) |
| `ReshapeLayer` | `ReshapeLayer[shape]` | Reshape to given shape |
| `SoftmaxLayer` | `SoftmaxLayer[opts]` | Softmax over last dimension |
| `TransposeLayer` | `TransposeLayer[perm]` | Transpose dimensions |

## `optimize`

| Function | Usage | Summary |
|---|---|---|
| `ImageThumbnail` | `ImageThumbnail[input, opts]` | Create thumbnail (cover) |

## `os`

| Function | Usage | Summary |
|---|---|---|
| `CommandExistsQ` | `CommandExistsQ[cmd]` | Does a command exist in PATH? |
| `KillProcess` | `KillProcess[proc, signal?]` | Send signal to process |
| `Pipe` | `Pipe[cmds]` | Compose processes via pipes |
| `Popen` | `Popen[cmd, args?, opts?]` | Spawn process and return handle |
| `ReadProcess` | `ReadProcess[proc, opts?]` | Read from process stdout/stderr |
| `Run` | `Run[cmd, args?, opts?]` | Run a process and capture output |
| `WaitProcess` | `WaitProcess[proc]` | Wait for process to exit |
| `Which` | `Which[cmd]` | Resolve command path from PATH |
| `WriteProcess` | `WriteProcess[proc, data]` | Write to process stdin |

## `parallel`

| Function | Usage | Summary |
|---|---|---|
| `ParallelMap` | `ParallelMap[f, list]` | Map in parallel over list |
| `ParallelTable` | `ParallelTable[exprs]` | Evaluate list of expressions in parallel (held) |

## `patch`

| Function | Usage | Summary |
|---|---|---|
| `GitApply` | `GitApply[patch, opts?]` | Apply a patch (or check only) |

## `path`

| Function | Usage | Summary |
|---|---|---|
| `Basename` | `Basename[path]` | Filename without directories |
| `CanonicalPath` | `CanonicalPath[path]` | Resolve symlinks and normalize |
| `Copy` | `Copy[src, dst, opts?]` | Copy file or directory (Recursive option) |
| `CurrentDirectory` | `CurrentDirectory[]` | Get current working directory |
| `Dirname` | `Dirname[path]` | Parent directory path |
| `ExpandPath` | `ExpandPath[path]` | Expand ~ and env vars |
| `FileExtension` | `FileExtension[path]` | File extension (no dot) |
| `FileStem` | `FileStem[path]` | Filename without extension |
| `Glob` | `Glob[pattern]` | Expand glob pattern to matching paths |
| `MakeDirectory` | `MakeDirectory[path, opts?]` | Create a directory (Parents option) |
| `Move` | `Move[src, dst]` | Move or rename a file/directory |
| `PathJoin` | `PathJoin[parts]` | Join path segments |
| `PathRemove` | `PathRemove[path, opts?]` | Alias: remove file or directory |
| `PathSplit` | `PathSplit[path]` | Split path into parts |
| `SetDirectory` | `SetDirectory[path]` | Change current working directory |
| `Symlink` | `Symlink[src, dst]` | Create a symbolic link |
| `Touch` | `Touch[path]` | Create file if missing (update mtime) |

## `paths`

| Function | Usage | Summary |
|---|---|---|
| `ShortestPaths` | `ShortestPaths[graph, start, opts?]` | Shortest path distances from start |

## `pipeline`

| Function | Usage | Summary |
|---|---|---|
| `ImageTransform` | `ImageTransform[input, pipeline]` | Apply pipeline of operations |
| `MediaPipeline` | `MediaPipeline[desc]` | Run arbitrary ffmpeg pipeline (builder) |

## `plot`

| Function | Usage | Summary |
|---|---|---|
| `LinePlot` | `LinePlot[data, opts]` | Render a line plot |
| `ScatterPlot` | `ScatterPlot[data, opts]` | Render a scatter plot |

## `predicate`

| Function | Usage | Summary |
|---|---|---|
| `Contains` | `Contains[container, item]` | Membership test for strings/lists/sets/assocs |
| `ContainsKeyQ` | `ContainsKeyQ[subject, key]` | Key membership for assoc/rows/Dataset/Frame |
| `ContainsQ` | `ContainsQ[container, item]` | Alias: membership predicate |
| `EmptyQ` | `EmptyQ[x]` | Is list/string/assoc empty? |
| `EndsWith` | `EndsWith[s, suffix]` | True if string ends with suffix |
| `HasKeyQ` | `HasKeyQ[subject, key]` | Alias: key membership predicate |
| `MemberQ` | `MemberQ[container, item]` | Alias: membership predicate (Contains) |
| `RegexMatch` | `RegexMatch[s, pattern]` | Return first regex match |
| `RegexMatchQ` | `RegexMatchQ[pattern, s]` | Alias: regex predicate (Boolean) |
| `StartsWith` | `StartsWith[s, prefix]` | True if string starts with prefix |
| `StringContains` | `StringContains[s, substr]` | Does string contain substring? |

## `preprocess`

| Function | Usage | Summary |
|---|---|---|
| `DimensionReduce` | `DimensionReduce[data, opts]` | Reduce dimensionality (PCA-like) |
| `FeatureExtract` | `FeatureExtract[data, opts]` | Learn preprocessing (impute/encode/standardize) |

## `proc`

| Function | Usage | Summary |
|---|---|---|
| `CommandExistsQ` | `CommandExistsQ[cmd]` | Does a command exist in PATH? |
| `KillProcess` | `KillProcess[proc, signal?]` | Send signal to process |
| `Pipe` | `Pipe[cmds]` | Compose processes via pipes |
| `Popen` | `Popen[cmd, args?, opts?]` | Spawn process and return handle |
| `ProcessInfo` | `ProcessInfo[proc]` | Inspect process handle (pid, running, exit) |
| `ReadProcess` | `ReadProcess[proc, opts?]` | Read from process stdout/stderr |
| `Run` | `Run[cmd, args?, opts?]` | Run a process and capture output |
| `WaitProcess` | `WaitProcess[proc]` | Wait for process to exit |
| `Which` | `Which[cmd]` | Resolve command path from PATH |
| `WriteProcess` | `WriteProcess[proc, data]` | Write to process stdin |

## `process`

| Function | Usage | Summary |
|---|---|---|
| `CommandExistsQ` | `CommandExistsQ[cmd]` | Does a command exist in PATH? |
| `KillProcess` | `KillProcess[proc, signal?]` | Send signal to process |
| `Pipe` | `Pipe[cmds]` | Compose processes via pipes |
| `Popen` | `Popen[cmd, args?, opts?]` | Spawn process and return handle |
| `ProcessInfo` | `ProcessInfo[proc]` | Inspect process handle (pid, running, exit) |
| `ReadProcess` | `ReadProcess[proc, opts?]` | Read from process stdout/stderr |
| `Run` | `Run[cmd, args?, opts?]` | Run a process and capture output |
| `WaitProcess` | `WaitProcess[proc]` | Wait for process to exit |
| `Which` | `Which[cmd]` | Resolve command path from PATH |
| `WriteProcess` | `WriteProcess[proc, data]` | Write to process stdin |

## `product`

| Function | Usage | Summary |
|---|---|---|
| `Times` | `Times[a, b, …]` | Multiply numbers; Listable, Flat, Orderless. |

## `query`

| Function | Usage | Summary |
|---|---|---|
| `Exec` | `Exec[conn, sql, params?]` | Execute DDL/DML (non-SELECT) |
| `Neighbors` | `Neighbors[graph, id, opts?]` | Neighbor node ids for a node |
| `SQL` | `SQL[conn, sql, params?]` | Run a SELECT query and return rows |

## `random`

| Function | Usage | Summary |
|---|---|---|
| `RandomBytes` | `RandomBytes[len, opts]` | Generate cryptographically secure random bytes |
| `RandomHex` | `RandomHex[len]` | Generate random hex string of n bytes |

## `redirect`

| Function | Usage | Summary |
|---|---|---|
| `RespondRedirect` | `RespondRedirect[location, opts]` | Build a redirect response (Location header) |

## `regex`

| Function | Usage | Summary |
|---|---|---|
| `RegexFind` | `RegexFind[s, pattern]` | Find first regex capture groups |
| `RegexFindAll` | `RegexFindAll[s, pattern]` | Find all regex capture groups |
| `RegexMatch` | `RegexMatch[s, pattern]` | Return first regex match |
| `RegexMatchQ` | `RegexMatchQ[pattern, s]` | Alias: regex predicate (Boolean) |
| `RegexReplace` | `RegexReplace[s, pattern, repl]` | Replace matches using regex |

## `regression`

| Function | Usage | Summary |
|---|---|---|
| `Predict` | `Predict[data, opts]` | Train a regressor (baseline/linear) |

## `remote`

| Function | Usage | Summary |
|---|---|---|
| `GitFetch` | `GitFetch[remote?]` | Fetch from remote |
| `GitPull` | `GitPull[remote?, opts?]` | Pull from remote |
| `GitPush` | `GitPush[opts?]` | Push to remote |
| `GitRemoteList` | `GitRemoteList[]` | List remotes |

## `remove`

| Function | Usage | Summary |
|---|---|---|
| `RemoveEdges` | `RemoveEdges[graph, edges]` | Remove edges by id or (src,dst,key) |
| `RemoveNodes` | `RemoveNodes[graph, ids]` | Remove nodes by id |

## `replace`

| Function | Usage | Summary |
|---|---|---|
| `StringReplace` | `StringReplace[s, from, to]` | Replace all substring matches |
| `StringReplaceFirst` | `StringReplaceFirst[s, from, to]` | Replace first substring match |

## `repo`

| Function | Usage | Summary |
|---|---|---|
| `GitEnsureRepo` | `GitEnsureRepo[opts?]` | Ensure Cwd is a git repo (init if needed) |
| `GitInit` | `GitInit[opts?]` | Initialize a new git repository |

## `routing`

| Function | Usage | Summary |
|---|---|---|
| `PathMatch` | `PathMatch[pattern, path]` | Match a path pattern like /users/:id against a path |

## `schedule`

| Function | Usage | Summary |
|---|---|---|
| `CancelSchedule` | `CancelSchedule[token]` | Cancel scheduled task |
| `Cron` | `Cron[expr, body]` | Schedule with cron expression (held) |
| `ScheduleEvery` | `ScheduleEvery[ms, body]` | Schedule recurring task (held) |

## `schema`

| Function | Usage | Summary |
|---|---|---|
| `Columns` | `Columns[ds]` | List column names for a dataset |
| `ContainsKeyQ` | `ContainsKeyQ[subject, key]` | Key membership for assoc/rows/Dataset/Frame |
| `DatasetSchema` | `DatasetSchema[ds]` | Describe schema for a dataset |
| `FrameColumns` | `FrameColumns[frame]` | List column names for a Frame |
| `HasKeyQ` | `HasKeyQ[subject, key]` | Alias: key membership predicate |
| `Keys` | `Keys[subject]` | Keys/columns for assoc, rows, Dataset, or Frame |
| `ListTables` | `ListTables[conn]` | List tables on a connection |

## `scope`

| Function | Usage | Summary |
|---|---|---|
| `CancelScope` | `CancelScope[scope]` | Cancel running scope |
| `EndScope` | `EndScope[scope]` | End scope and release resources |
| `InScope` | `InScope[scope, body]` | Run body inside a scope (held) |
| `Scope` | `Scope[opts, body]` | Run body with resource limits (held) |
| `StartScope` | `StartScope[opts, body]` | Start a managed scope (held) |

## `search`

| Function | Usage | Summary |
|---|---|---|
| `Find` | `Find[pred, list]` | First element where pred[x] |
| `Position` | `Position[pred, list]` | 1-based index of first match |
| `Search` | `Search[target, query, opts?]` | Search within a store or index (VectorStore, Index) |
| `VectorSearch` | `VectorSearch[store, query, opts]` | Search by vector or text (hybrid supported) |

## `select`

| Function | Usage | Summary |
|---|---|---|
| `FrameSelect` | `FrameSelect[frame, spec]` | Select/compute columns in Frame |
| `RenameCols` | `RenameCols[ds, mapping]` | Rename columns via mapping |
| `SelectCols` | `SelectCols[ds, cols]` | Select subset of columns by name |

## `server`

| Function | Usage | Summary |
|---|---|---|
| `HttpServe` | `HttpServe[handler, opts]` | Start an HTTP server and handle requests with a function |
| `HttpServeRoutes` | `HttpServeRoutes[routes, opts]` | Start an HTTP server with a routes table |
| `HttpServeTls` | `HttpServeTls[handler, opts]` | Start an HTTPS server with TLS cert/key |
| `HttpServerAddr` | `HttpServerAddr[server]` | Get bound address for a server id |
| `HttpServerStop` | `HttpServerStop[server]` | Stop a running HTTP server by id |
| `RespondBytes` | `RespondBytes[bytes, opts]` | Build a binary response for HttpServe |
| `RespondFile` | `RespondFile[path, opts]` | Build a file response for HttpServe |
| `RespondHtml` | `RespondHtml[html, opts]` | Build an HTML response for HttpServe |
| `RespondJson` | `RespondJson[value, opts]` | Build a JSON response for HttpServe |
| `RespondNoContent` | `RespondNoContent[opts]` | Build an empty 204/205/304 response |
| `RespondRedirect` | `RespondRedirect[location, opts]` | Build a redirect response (Location header) |
| `RespondText` | `RespondText[text, opts]` | Build a text response for HttpServe |

## `set`

| Function | Usage | Summary |
|---|---|---|
| `Concat` | `Concat[inputs]` | Concatenate datasets by rows (schema-union) |
| `FrameUnion` | `FrameUnion[frames…]` | Union Frames by columns (schema union) |
| `Union` | `Union[inputs, byColumns?]` | Union multiple datasets (by columns) |
| `UnionByPosition` | `UnionByPosition[ds1, ds2, …]` | Union datasets by column position. |
| `Unique` | `Unique[list]` | Stable deduplicate list |

## `sign`

| Function | Usage | Summary |
|---|---|---|
| `KeypairGenerate` | `KeypairGenerate[opts]` | Generate signing keypair (Ed25519) |
| `Sign` | `Sign[message, secretKey, opts]` | Sign message (Ed25519) |
| `Verify` | `Verify[message, signature, publicKey, opts]` | Verify signature (Ed25519) |

## `sleep`

| Function | Usage | Summary |
|---|---|---|
| `Sleep` | `Sleep[ms]` | Sleep for N milliseconds |

## `slice`

| Function | Usage | Summary |
|---|---|---|
| `Drop` | `Drop[list, n]` | Drop first n (last if negative) |
| `DropWhile` | `DropWhile[pred, list]` | Drop while pred[x] holds |
| `Slice` | `Slice[list, start, len?]` | Slice list by start and length |
| `Take` | `Take[list, n]` | Take first n (last if negative) |
| `TakeWhile` | `TakeWhile[pred, list]` | Take while pred[x] holds |

## `sort`

| Function | Usage | Summary |
|---|---|---|
| `FrameSort` | `FrameSort[frame, by]` | Sort Frame by columns |
| `Sort` | `Sort[ds, by, opts?]` | Sort rows by columns |

## `sql`

| Function | Usage | Summary |
|---|---|---|
| `Begin` | `Begin[conn]` | Begin a transaction |
| `Commit` | `Commit[conn]` | Commit the current transaction |
| `Connect` | `Connect[dsn|opts]` | Open database connection (mock/sqlite/duckdb) |
| `Disconnect` | `Disconnect[conn]` | Close a database connection |
| `Exec` | `Exec[conn, sql, params?]` | Execute DDL/DML (non-SELECT) |
| `ExplainSQL` | `ExplainSQL[ds]` | Render SQL for pushdown-capable parts |
| `Fetch` | `Fetch[cursor, limit?]` | Fetch next batch of rows from a cursor |
| `InsertRows` | `InsertRows[conn, table, rows]` | Insert multiple rows (assoc list) into a table |
| `ListTables` | `ListTables[conn]` | List tables on a connection |
| `Ping` | `Ping[conn]` | Check connectivity for a database connection |
| `RegisterTable` | `RegisterTable[conn, name, rows]` | Register in-memory rows as a table (mock) |
| `Rollback` | `Rollback[conn]` | Rollback the current transaction |
| `SQL` | `SQL[conn, sql, params?]` | Run a SELECT query and return rows |
| `SQLCursor` | `SQLCursor[conn, sql, params?]` | Run a query and return a cursor handle |
| `Table` | `Table[conn, name]` | Reference a table as a Dataset |
| `UpsertRows` | `UpsertRows[conn, table, rows, keys?]` | Upsert rows (assoc list) into a table |
| `WriteDataset` | `WriteDataset[conn, table, dataset, opts?]` | Write a Dataset into a table |

## `stats`

| Function | Usage | Summary |
|---|---|---|
| `Describe` | `Describe[name, items, opts?]` | Define a test suite (held). |
| `FrameDescribe` | `FrameDescribe[frame, opts?]` | Quick stats by columns |

## `status`

| Function | Usage | Summary |
|---|---|---|
| `GitStatus` | `GitStatus[opts?]` | Status (porcelain) with branch/ahead/behind/changes |
| `GitStatusSummary` | `GitStatusSummary[opts?]` | Summarize status counts and branch |

## `stdio`

| Function | Usage | Summary |
|---|---|---|
| `ReadStdin` | `ReadStdin[]` | Read all text from stdin |

## `store`

| Function | Usage | Summary |
|---|---|---|
| `VectorStore` | `VectorStore[optsOrDsn]` | Create/open a vector store (memory or DSN) |

## `string`

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
| `StringContains` | `StringContains[s, substr]` | Does string contain substring? |
| `StringJoin` | `StringJoin[parts]` | Concatenate list of parts. |
| `StringLength` | `StringLength[s]` | Length of string (Unicode scalar count). |
| `StringReplace` | `StringReplace[s, from, to]` | Replace all substring matches |
| `StringReplaceFirst` | `StringReplaceFirst[s, from, to]` | Replace first substring match |
| `StringSplit` | `StringSplit[s, sep]` | Split string by separator |
| `StringTrim` | `StringTrim[s]` | Trim whitespace from both ends |
| `ToLower` | `ToLower[s]` | Lowercase string. |
| `ToUpper` | `ToUpper[s]` | Uppercase string. |
| `UrlDecode` | `UrlDecode[s]` | Decode percent-encoded string |
| `UrlEncode` | `UrlEncode[s]` | Percent-encode string for URLs |

## `sum`

| Function | Usage | Summary |
|---|---|---|
| `Plus` | `Plus[a, b, …]` | Add numbers; Listable, Flat, Orderless. |

## `sync`

| Function | Usage | Summary |
|---|---|---|
| `GitSyncUpstream` | `GitSyncUpstream[opts?]` | Fetch, rebase (or merge), and push upstream |

## `table`

| Function | Usage | Summary |
|---|---|---|
| `RegisterTable` | `RegisterTable[conn, name, rows]` | Register in-memory rows as a table (mock) |

## `temp`

| Function | Usage | Summary |
|---|---|---|
| `TempDir` | `TempDir[]` | Create a unique temporary directory |
| `TempFile` | `TempFile[]` | Create a unique temporary file |

## `testing`

| Function | Usage | Summary |
|---|---|---|
| `Describe` | `Describe[name, items, opts?]` | Define a test suite (held). |

## `text`

| Function | Usage | Summary |
|---|---|---|
| `StringJoin` | `StringJoin[parts]` | Concatenate list of parts. |
| `StringLength` | `StringLength[s]` | Length of string (Unicode scalar count). |
| `StringSplit` | `StringSplit[s, sep]` | Split string by separator |
| `StringTrim` | `StringTrim[s]` | Trim whitespace from both ends |
| `ToLower` | `ToLower[s]` | Lowercase string. |
| `ToUpper` | `ToUpper[s]` | Uppercase string. |

## `time`

| Function | Usage | Summary |
|---|---|---|
| `AddDuration` | `AddDuration[dt, dur]` | Add duration to DateTime/epochMs |
| `CancelSchedule` | `CancelSchedule[token]` | Cancel scheduled task |
| `Cron` | `Cron[expr, body]` | Schedule with cron expression (held) |
| `DateFormat` | `DateFormat[dt, fmt?]` | Format DateTime or epochMs to string |
| `DateParse` | `DateParse[s]` | Parse date/time string to epochMs |
| `DateTime` | `DateTime[spec]` | Build/parse DateTime assoc (UTC) |
| `DiffDuration` | `DiffDuration[a, b]` | Difference between DateTimes |
| `Duration` | `Duration[spec]` | Build Duration assoc from ms or fields |
| `DurationParse` | `DurationParse[s]` | Parse human duration (e.g., 1h30m) |
| `EndOf` | `EndOf[dt, unit]` | End of unit (day/week/month) |
| `MonotonicNow` | `MonotonicNow[]` | Monotonic clock milliseconds since start |
| `NowMs` | `NowMs[]` | Current UNIX time in milliseconds |
| `ScheduleEvery` | `ScheduleEvery[ms, body]` | Schedule recurring task (held) |
| `Sleep` | `Sleep[ms]` | Sleep for N milliseconds |
| `StartOf` | `StartOf[dt, unit]` | Start of unit (day/week/month) |
| `TimeZoneConvert` | `TimeZoneConvert[dt, tz]` | Convert DateTime to another timezone |

## `tls`

| Function | Usage | Summary |
|---|---|---|
| `HttpServeTls` | `HttpServeTls[handler, opts]` | Start an HTTPS server with TLS cert/key |

## `toml`

| Function | Usage | Summary |
|---|---|---|
| `TomlParse` | `TomlParse[toml]` | Parse TOML string to value |
| `TomlStringify` | `TomlStringify[value]` | Render value as TOML |

## `train`

| Function | Usage | Summary |
|---|---|---|
| `NetTrain` | `NetTrain[net, data, opts]` | Train network on data |

## `transform`

| Function | Usage | Summary |
|---|---|---|
| `FilterRows` | `FilterRows[ds, pred]` | Filter rows by predicate (held) |
| `FrameFilter` | `FrameFilter[frame, pred]` | Filter rows in a Frame |
| `FrameOffset` | `FrameOffset[frame, n]` | Skip first n rows of Frame |
| `FrameSelect` | `FrameSelect[frame, spec]` | Select/compute columns in Frame |
| `ImageCrop` | `ImageCrop[input, opts]` | Crop image by rect or gravity |
| `ImagePad` | `ImagePad[input, opts]` | Pad image to target size |
| `ImageResize` | `ImageResize[input, opts]` | Resize image (contain/cover) |
| `Offset` | `Offset[ds, n]` | Skip first n rows |
| `RenameCols` | `RenameCols[ds, mapping]` | Rename columns via mapping |
| `SelectCols` | `SelectCols[ds, cols]` | Select subset of columns by name |
| `WithColumns` | `WithColumns[ds, defs]` | Add/compute new columns (held) |

## `traversal`

| Function | Usage | Summary |
|---|---|---|
| `BFS` | `BFS[graph, start, opts?]` | Breadth-first search order |
| `DFS` | `DFS[graph, start, opts?]` | Depth-first search order |

## `tune`

| Function | Usage | Summary |
|---|---|---|
| `MLTune` | `MLTune[data, opts]` | Parameter sweep with basic scoring |

## `tx`

| Function | Usage | Summary |
|---|---|---|
| `Begin` | `Begin[conn]` | Begin a transaction |
| `Commit` | `Commit[conn]` | Commit the current transaction |
| `Rollback` | `Rollback[conn]` | Rollback the current transaction |

## `types`

| Function | Usage | Summary |
|---|---|---|
| `Cast` | `Cast[value, type]` | Cast a value to a target type (string, integer, real, boolean). |
| `Coalesce` | `Coalesce[values…]` | First non-null value |

## `tz`

| Function | Usage | Summary |
|---|---|---|
| `TimeZoneConvert` | `TimeZoneConvert[dt, tz]` | Convert DateTime to another timezone |

## `upsert`

| Function | Usage | Summary |
|---|---|---|
| `VectorUpsert` | `VectorUpsert[store, rows]` | Insert or update vectors with metadata |

## `url`

| Function | Usage | Summary |
|---|---|---|
| `Slugify` | `Slugify[s]` | Slugify for URLs |
| `UrlDecode` | `UrlDecode[s]` | Decode percent-encoded string |
| `UrlEncode` | `UrlEncode[s]` | Percent-encode string for URLs |

## `vcs`

| Function | Usage | Summary |
|---|---|---|
| `GitRoot` | `GitRoot[]` | Path to repository root (Null if absent) |
| `GitVersion` | `GitVersion[]` | Get git client version string |

## `vector`

| Function | Usage | Summary |
|---|---|---|
| `VectorCount` | `VectorCount[store]` | Count items in store |
| `VectorDelete` | `VectorDelete[store, ids]` | Delete items by ids |
| `VectorReset` | `VectorReset[store]` | Clear all items in store |
| `VectorSearch` | `VectorSearch[store, query, opts]` | Search by vector or text (hybrid supported) |
| `VectorStore` | `VectorStore[optsOrDsn]` | Create/open a vector store (memory or DSN) |
| `VectorUpsert` | `VectorUpsert[store, rows]` | Insert or update vectors with metadata |

## `viz`

| Function | Usage | Summary |
|---|---|---|
| `BarChart` | `BarChart[data, opts]` | Render a bar chart |
| `Chart` | `Chart[spec, opts]` | Render a chart from a spec |
| `Figure` | `Figure[items, opts]` | Compose multiple charts in a grid |
| `Histogram` | `Histogram[data, opts]` | Render a histogram |
| `LinePlot` | `LinePlot[data, opts]` | Render a line plot |
| `ScatterPlot` | `ScatterPlot[data, opts]` | Render a scatter plot |

## `watch`

| Function | Usage | Summary |
|---|---|---|
| `CancelWatch` | `CancelWatch[token]` | Cancel a directory watch |
| `WatchDirectory` | `WatchDirectory[path, handler, opts?]` | Watch directory and stream events (held) |

## `write`

| Function | Usage | Summary |
|---|---|---|
| `InsertRows` | `InsertRows[conn, table, rows]` | Insert multiple rows (assoc list) into a table |
| `UpsertRows` | `UpsertRows[conn, table, rows, keys?]` | Upsert rows (assoc list) into a table |
| `WriteDataset` | `WriteDataset[conn, table, dataset, opts?]` | Write a Dataset into a table |

## `yaml`

| Function | Usage | Summary |
|---|---|---|
| `YamlParse` | `YamlParse[yaml]` | Parse YAML string to value |
| `YamlStringify` | `YamlStringify[value, opts]` | Render value as YAML |

## `zip`

| Function | Usage | Summary |
|---|---|---|
| `Unzip` | `Unzip[pairs]` | Unzip pairs into two lists |
