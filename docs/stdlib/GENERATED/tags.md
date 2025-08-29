# Tags Index

| Tag | Functions |
|---|---|
| `activation` | 4 |
| `actor` | 4 |
| `admin` | 1 |
| `aead` | 3 |
| `aggregate` | 5 |
| `apply` | 1 |
| `archive` | 8 |
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
| `classification` | 2 |
| `clock` | 2 |
| `clustering` | 2 |
| `collection` | 3 |
| `commit` | 2 |
| `compose` | 2 |
| `compress` | 2 |
| `concurrency` | 21 |
| `conn` | 4 |
| `control` | 3 |
| `cookies` | 2 |
| `copy` | 1 |
| `create` | 2 |
| `cron` | 1 |
| `crypto` | 16 |
| `csv` | 8 |
| `cursor` | 3 |
| `cv` | 2 |
| `dataset` | 29 |
| `datetime` | 3 |
| `db` | 18 |
| `decode` | 2 |
| `decomposition` | 1 |
| `delete` | 2 |
| `diff` | 1 |
| `dist` | 8 |
| `distinct` | 4 |
| `duration` | 4 |
| `edit` | 6 |
| `encode` | 2 |
| `encoding` | 6 |
| `env` | 2 |
| `estimator` | 1 |
| `explain` | 2 |
| `export` | 1 |
| `expr` | 1 |
| `ffmpeg` | 7 |
| `filter` | 3 |
| `fixedpoint` | 2 |
| `flow` | 1 |
| `fold` | 3 |
| `frame` | 22 |
| `fs` | 30 |
| `functional` | 11 |
| `generic` | 35 |
| `git` | 22 |
| `glob` | 1 |
| `graph` | 16 |
| `graphs` | 1 |
| `group` | 2 |
| `hash` | 1 |
| `html` | 8 |
| `http` | 25 |
| `image` | 12 |
| `import` | 4 |
| `index` | 1 |
| `inference` | 2 |
| `info` | 1 |
| `init` | 1 |
| `inspect` | 5 |
| `introspect` | 7 |
| `introspection` | 2 |
| `io` | 65 |
| `iteration` | 2 |
| `join` | 1 |
| `json` | 9 |
| `jwt` | 2 |
| `kdf` | 3 |
| `key` | 1 |
| `layer` | 28 |
| `layout` | 1 |
| `lifecycle` | 2 |
| `link` | 1 |
| `list` | 41 |
| `log` | 1 |
| `logic` | 15 |
| `mac` | 2 |
| `map` | 2 |
| `math` | 16 |
| `media` | 7 |
| `metrics` | 3 |
| `ml` | 20 |
| `move` | 1 |
| `mst` | 1 |
| `mutate` | 2 |
| `net` | 16 |
| `network` | 6 |
| `nn` | 45 |
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
| `random` | 9 |
| `redirect` | 1 |
| `regex` | 8 |
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
| `set` | 8 |
| `sign` | 3 |
| `sleep` | 1 |
| `slice` | 5 |
| `sort` | 7 |
| `sql` | 17 |
| `stats` | 18 |
| `status` | 2 |
| `stdio` | 1 |
| `store` | 1 |
| `string` | 35 |
| `sum` | 1 |
| `sync` | 1 |
| `table` | 1 |
| `temp` | 2 |
| `template` | 4 |
| `tensor` | 6 |
| `testing` | 1 |
| `text` | 5 |
| `time` | 16 |
| `tls` | 1 |
| `toml` | 2 |
| `train` | 2 |
| `transform` | 11 |
| `transformer` | 13 |
| `traversal` | 2 |
| `tune` | 2 |
| `tx` | 3 |
| `types` | 2 |
| `tz` | 1 |
| `unicode` | 4 |
| `update` | 2 |
| `upsert` | 1 |
| `url` | 3 |
| `vcs` | 2 |
| `vector` | 6 |
| `vision` | 1 |
| `viz` | 6 |
| `watch` | 3 |
| `write` | 3 |
| `xml` | 1 |
| `yaml` | 2 |

## `activation`

| Function | Usage | Summary |
|---|---|---|
| `Gelu` | `Gelu[x?]` | GELU activation (tanh approx): tensor op or zero-arg layer |
| `Relu` | `Relu[x]` | Rectified Linear Unit: max(0, x). Tensor-aware: elementwise on tensors. |
| `Sigmoid` | `Sigmoid[x?]` | Sigmoid activation: tensor op or zero-arg layer |
| `Softmax` | `Softmax[x?, opts?]` | Softmax activation: zero-arg layer (tensor variant TBD) |

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
| `Aggregate` | `Aggregate[group, spec]` | Aggregate grouped data (stub) |
| `Count` | `Count[x]` | Count items/elements (lists, assocs, Bag/VectorStore) |
| `CountBy` | `CountBy[f, list]` | Counts by key function (assoc) |
| `Tally` | `Tally[list]` | Counts by value (assoc) |

## `apply`

| Function | Usage | Summary |
|---|---|---|
| `Apply` | `Apply[f, list, level?]` | Apply head to list elements: Apply[f, {…}] |

## `archive`

| Function | Usage | Summary |
|---|---|---|
| `Gunzip` | `Gunzip[dataOrPath, opts?]` | Gunzip-decompress a string or a .gz file; optionally write to path. |
| `Gzip` | `Gzip[dataOrPath, opts?]` | Gzip-compress a string or a file; optionally write to path. |
| `Tar` | `Tar[dest, inputs, opts?]` | Create a .tar (optionally .tar.gz) archive from inputs. |
| `TarExtract` | `TarExtract[src, dest]` | Extract a .tar or .tar.gz archive into a directory. |
| `Untar` | `Untar[src, dest]` | Extract a .tar or .tar.gz archive into a directory. |
| `Unzip` | `Unzip[pairs|src, dest?]` | Unzip list of pairs or extract a .zip (dispatched). Overloads: Unzip[pairs]; Unzip[src, dest] |
| `Zip` | `Zip[a|dest, b|inputs]` | Zip lists into pairs or create a .zip archive (dispatched). Overloads: Zip[a, b]; Zip[dest, inputs] |
| `ZipExtract` | `ZipExtract[src, dest]` | Extract a .zip archive into a directory. |

## `assoc`

| Function | Usage | Summary |
|---|---|---|
| `AssociationMap` | `AssociationMap[fn, assoc]` | Map values in an association |
| `AssociationMapKV` | `AssociationMapKV[fn, assoc]` | Map key/value pairs in an association |
| `AssociationMapKeys` | `AssociationMapKeys[fn, assoc]` | Map keys in an association |
| `AssociationMapPairs` | `AssociationMapPairs[fn, assoc]` | Map to key/value pairs or assoc |
| `Columns` | `Columns[ds]` | List column names for a dataset |
| `KeySort` | `KeySort[assoc]` | Sort association by key |
| `Keys` | `Keys[subject]` | Keys/columns for assoc, rows, Dataset, or Frame |
| `MapAt` | `MapAt[f, subject, indexOrKey]` | Apply function at 1-based index or key. |
| `Merge` | `Merge[args]` | Merge associations with optional combiner |
| `Part` | `Part[subject, index]` | Index into list/assoc |
| `ReplacePart` | `ReplacePart[subject, indexOrKey, value]` | Replace element at 1-based index or key. |
| `Select` | `Select[assoc|ds, pred|keys|cols]` | Select keys/columns or compute columns (dispatched). Overloads: Select[assoc, pred\|keys]; Select[ds, cols] |
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
| `Classifier` | `Classifier[opts?]` | Create classifier spec (Logistic/Baseline) |
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
| `Clusterer` | `Clusterer[opts?]` | Create clusterer spec |

## `collection`

| Function | Usage | Summary |
|---|---|---|
| `Add` | `Add[target, value]` | Add value to a collection (alias of Insert for some types) |
| `Insert` | `Insert[target, value]` | Insert into collection or structure (dispatched) |
| `Remove` | `Remove[target|path, value?|opts?]` | Remove from a collection/structure or remove a file/directory (dispatched). Overloads: Remove[target, value?]; Remove[path, opts?] |

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

## `control`

| Function | Usage | Summary |
|---|---|---|
| `Do` | `Do[body, n]` | Execute body n times. |
| `For` | `For[init, test, step, body]` | C-style loop with init/test/step. |
| `While` | `While[test, body]` | Repeat body while test evaluates to True. |

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
| `CrossValidate` | `CrossValidate[obj, data, opts?]` | Cross-validate estimator + data (dispatched) |
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
| `DistinctBy` | `DistinctBy[dataset, keys, opts?]` | Alias for DistinctOn |
| `DistinctOn` | `DistinctOn[ds, keys, orderBy?, keepLast?]` | Keep one row per key with order policy |
| `ExplainDataset` | `ExplainDataset[ds]` | Inspect logical plan for a dataset |
| `ExplainSQL` | `ExplainSQL[ds]` | Render SQL for pushdown-capable parts |
| `Filter` | `Filter[pred, list]` | Keep elements where pred[x] is True |
| `FilterRows` | `FilterRows[ds, pred]` | Filter rows by predicate (held) |
| `Import` | `Import[source, opts?]` | Import data from path/URL into Frame (default), Dataset (Target->"Dataset"), or Value. Automatically sniffs Type/Delimiter/Header. |
| `Join` | `Join[a|left, b|right, on?, how?]` | Join lists or datasets (dispatched). Overloads: Join[list1, list2]; Join[left, right, on, how?] |
| `Keys` | `Keys[subject]` | Keys/columns for assoc, rows, Dataset, or Frame |
| `Limit` | `Limit[dataset, n]` | Alias for Head on Dataset |
| `ReadCSVDataset` | `ReadCSVDataset[path, opts?]` | Read a CSV file into a dataset |
| `ReadJsonLinesDataset` | `ReadJsonLinesDataset[path, opts?]` | Read a JSONL file into a dataset |
| `RenameCols` | `RenameCols[ds, mapping]` | Rename columns via mapping |
| `Select` | `Select[assoc|ds, pred|keys|cols]` | Select keys/columns or compute columns (dispatched). Overloads: Select[assoc, pred\|keys]; Select[ds, cols] |
| `SelectCols` | `SelectCols[ds, cols]` | Select subset of columns by name |
| `Table` | `Table[conn, name]` | Reference a table as a Dataset |
| `Union` | `Union[args]` | Union for lists (stable) or sets (dispatched) |
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
| `Execute` | `Execute[conn, sql, opts?]` | Execute DDL/DML (non-SELECT) |
| `Fetch` | `Fetch[cursor, limit?]` | Fetch next batch of rows from a cursor |
| `InsertRows` | `InsertRows[conn, table, rows]` | Insert multiple rows (assoc list) into a table |
| `ListTables` | `ListTables[conn]` | List tables on a connection |
| `Ping` | `Ping[conn]` | Check connectivity for a database connection |
| `Query` | `Query[conn, sql, opts?]` | Run a SELECT query and return rows |
| `RegisterTable` | `RegisterTable[conn, name, rows]` | Register in-memory rows as a table (mock) |
| `Rollback` | `Rollback[conn]` | Rollback the current transaction |
| `SQLCursor` | `SQLCursor[conn, sql, params?]` | Run a query and return a cursor handle |
| `Table` | `Table[conn, name]` | Reference a table as a Dataset |
| `UpsertRows` | `UpsertRows[conn, table, rows, keys?]` | Upsert rows (assoc list) into a table |
| `WriteDataset` | `WriteDataset[conn, table, dataset, opts?]` | Write a Dataset into a table |

## `decode`

| Function | Usage | Summary |
|---|---|---|
| `AudioDecode` | `AudioDecode[input, opts]` | Decode audio to raw (s16le) or WAV |
| `ImageDecode` | `ImageDecode[input, opts]` | Decode image to raw or reencoded bytes |

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

## `dist`

| Function | Usage | Summary |
|---|---|---|
| `Bernoulli` | `Bernoulli[p]` | Bernoulli distribution head (probability p). |
| `BinomialDistribution` | `BinomialDistribution[n, p]` | Binomial distribution head (trials n, prob p). |
| `CDF` | `CDF[dist, x]` | Cumulative distribution for a distribution at x. |
| `Exponential` | `Exponential[lambda]` | Exponential distribution head (rate λ). |
| `Gamma` | `Gamma[k, theta]` | Gamma distribution head (shape k, scale θ). |
| `Normal` | `Normal[mu, sigma]` | Normal distribution head (mean μ, stddev σ). |
| `PDF` | `PDF[dist, x]` | Probability density/mass for a distribution at x. |
| `Poisson` | `Poisson[lambda]` | Poisson distribution head (rate λ). |

## `distinct`

| Function | Usage | Summary |
|---|---|---|
| `Distinct` | `Distinct[ds, cols?]` | Drop duplicate rows (optionally by columns) |
| `DistinctBy` | `DistinctBy[dataset, keys, opts?]` | Alias for DistinctOn |
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

## `estimator`

| Function | Usage | Summary |
|---|---|---|
| `Estimator` | `Estimator[opts]` | Create ML estimator spec (Task/Method) |

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
| `Aggregate` | `Aggregate[group, spec]` | Aggregate grouped data (stub) |
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
| `GroupBy` | `GroupBy[ds, keys]` | Group rows by key(s) |
| `Import` | `Import[source, opts?]` | Import data from path/URL into Frame (default), Dataset (Target->"Dataset"), or Value. Automatically sniffs Type/Delimiter/Header. |
| `Keys` | `Keys[subject]` | Keys/columns for assoc, rows, Dataset, or Frame |
| `Select` | `Select[assoc|ds, pred|keys|cols]` | Select keys/columns or compute columns (dispatched). Overloads: Select[assoc, pred\|keys]; Select[ds, cols] |

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
| `Untar` | `Untar[src, dest]` | Extract a .tar or .tar.gz archive into a directory. |
| `Unzip` | `Unzip[pairs|src, dest?]` | Unzip list of pairs or extract a .zip (dispatched). Overloads: Unzip[pairs]; Unzip[src, dest] |
| `Watch` | `Watch[path, handler, opts?]` | Watch directory and stream events (held) |
| `WatchDirectory` | `WatchDirectory[path, handler, opts?]` | Watch directory and stream events (held) |
| `WriteBytes` | `WriteBytes[path, bytes]` | Write bytes to file |
| `WriteFile` | `WriteFile[path, content]` | Write stringified content to file |
| `Zip` | `Zip[a|dest, b|inputs]` | Zip lists into pairs or create a .zip archive (dispatched). Overloads: Zip[a, b]; Zip[dest, inputs] |
| `ZipExtract` | `ZipExtract[src, dest]` | Extract a .zip archive into a directory. |

## `functional`

| Function | Usage | Summary |
|---|---|---|
| `Apply` | `Apply[f, list, level?]` | Apply head to list elements: Apply[f, {…}] |
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
| `Count` | `Count[x]` | Count items/elements (lists, assocs, Bag/VectorStore) |
| `Describe` | `Describe[name, items, opts?]` | Define a test suite (held). |
| `Distinct` | `Distinct[ds, cols?]` | Drop duplicate rows (optionally by columns) |
| `EmptyQ` | `EmptyQ[x]` | Is the subject empty? (lists, strings, assocs, handles) |
| `EqualQ` | `EqualQ[a, b]` | Structural equality for sets and handles |
| `Export` | `Export[symbols]` | Mark symbol(s) as public |
| `Filter` | `Filter[pred, list]` | Keep elements where pred[x] is True |
| `HasKeyQ` | `HasKeyQ[subject, key]` | Alias: key membership predicate |
| `Head` | `Head[ds, n]` | Take first n rows |
| `Import` | `Import[source, opts?]` | Import data from path/URL into Frame (default), Dataset (Target->"Dataset"), or Value. Automatically sniffs Type/Delimiter/Header. |
| `ImportBytes` | `ImportBytes[bytes, opts?]` | Parse byte buffer using Type (text/json/etc.) |
| `ImportString` | `ImportString[content, opts?]` | Parse in-memory strings into Frame/Dataset/Value. Automatically sniffs Type if missing. |
| `Info` | `Info[target]` | Information about a handle (Logger, Graph, etc.) |
| `Insert` | `Insert[target, value]` | Insert into collection or structure (dispatched) |
| `Join` | `Join[a|left, b|right, on?, how?]` | Join lists or datasets (dispatched). Overloads: Join[list1, list2]; Join[left, right, on, how?] |
| `Keys` | `Keys[subject]` | Keys/columns for assoc, rows, Dataset, or Frame |
| `Length` | `Length[x]` | Length of a list or string. |
| `MemberQ` | `MemberQ[container, item]` | Alias: membership predicate (Contains) |
| `Offset` | `Offset[ds, n]` | Skip first n rows |
| `Remove` | `Remove[target|path, value?|opts?]` | Remove from a collection/structure or remove a file/directory (dispatched). Overloads: Remove[target, value?]; Remove[path, opts?] |
| `Search` | `Search[target, query, opts?]` | Search within a store or index (VectorStore, Index) |
| `Select` | `Select[assoc|ds, pred|keys|cols]` | Select keys/columns or compute columns (dispatched). Overloads: Select[assoc, pred\|keys]; Select[ds, cols] |
| `Sniff` | `Sniff[source]` | Suggest Type and options for a source (file/url/string/bytes). |
| `Sort` | `Sort[list|ds, by?, opts?]` | Sort a list or dataset (dispatched). Overloads: Sort[list]; Sort[ds, by, opts?] |
| `SortBy` | `SortBy[f, subject]` | Sort list by key or association by derived key. |
| `StableKey` | `StableKey[x]` | Canonical stable key string for ordering/dedup. |
| `SubsetQ` | `SubsetQ[a, b]` | Is a subset of b? (sets, lists) |
| `Tail` | `Tail[ds, n]` | Take last n rows |
| `Union` | `Union[args]` | Union for lists (stable) or sets (dispatched) |

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

## `group`

| Function | Usage | Summary |
|---|---|---|
| `Aggregate` | `Aggregate[group, spec]` | Aggregate grouped data (stub) |
| `GroupBy` | `GroupBy[ds, keys]` | Group rows by key(s) |

## `hash`

| Function | Usage | Summary |
|---|---|---|
| `Hash` | `Hash[input, opts]` | Compute digest (BLAKE3/SHA-256) |

## `html`

| Function | Usage | Summary |
|---|---|---|
| `HtmlAttr` | `HtmlAttr[s]` | Escape string for HTML attribute context |
| `HtmlEscape` | `HtmlEscape[s]` | Escape string for HTML |
| `HtmlTemplate` | `HtmlTemplate[templateOrPath, data, opts?]` | Render HTML/XML templates with Mustache semantics (sections, inverted, partials, comments, indented partials, standalone trimming; unescaped via {{{...}}} or {{& name}}). Options: Mode(html\|xml), Strict, Whitespace(preserve\|trim-tags\|smart), Partials, Components, Layout, Loader. |
| `HtmlTemplateCompile` | `HtmlTemplateCompile[templateOrPath, opts?]` | Precompile HTML template (returns handle) |
| `HtmlTemplateRender` | `HtmlTemplateRender[handle, data, opts?]` | Render compiled HTML template with data |
| `HtmlUnescape` | `HtmlUnescape[s]` | Unescape HTML-escaped string |
| `RespondHtml` | `RespondHtml[html, opts]` | Build an HTML response for HttpServe |
| `SafeHtml` | `SafeHtml[s]` | Mark string as safe HTML (no escaping) |

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
| `Predict` | `Predict[obj, x, opts?]` | Predict using a network or estimator (dispatched) |

## `info`

| Function | Usage | Summary |
|---|---|---|
| `VectorCount` | `VectorCount[store]` | Count items in store |

## `init`

| Function | Usage | Summary |
|---|---|---|
| `Initializer` | `Initializer[opts?]` | Initializer spec for layer parameters |

## `inspect`

| Function | Usage | Summary |
|---|---|---|
| `FrameHead` | `FrameHead[frame, n?]` | Take first n rows from Frame |
| `FrameTail` | `FrameTail[frame, n?]` | Take last n rows from Frame |
| `Head` | `Head[ds, n]` | Take first n rows |
| `Limit` | `Limit[dataset, n]` | Alias for Head on Dataset |
| `Tail` | `Tail[ds, n]` | Take last n rows |

## `introspect`

| Function | Usage | Summary |
|---|---|---|
| `ConnectionInfo` | `ConnectionInfo[conn]` | Inspect connection details (dsn, kind, tx) |
| `CursorInfo` | `CursorInfo[cursor]` | Inspect cursor details (dsn, kind, sql, offset) |
| `GraphInfo` | `GraphInfo[graph]` | Summary and counts for graph |
| `MLProperty` | `MLProperty[model, prop]` | Inspect trained model properties |
| `ProcessInfo` | `ProcessInfo[proc]` | Inspect process handle (pid, running, exit) |
| `Property` | `Property[obj, key]` | Property of a network or ML model (dispatch) |
| `Summary` | `Summary[obj]` | Summary of a network (dispatch to NetSummary) |

## `introspection`

| Function | Usage | Summary |
|---|---|---|
| `Describe` | `Describe[name, items, opts?]` | Define a test suite (held). |
| `Info` | `Info[target]` | Information about a handle (Logger, Graph, etc.) |

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

## `key`

| Function | Usage | Summary |
|---|---|---|
| `StableKey` | `StableKey[x]` | Canonical stable key string for ordering/dedup. |

## `layer`

| Function | Usage | Summary |
|---|---|---|
| `BatchNorm` | `BatchNorm[opts?]` | Batch normalization layer |
| `CausalSelfAttention` | `CausalSelfAttention[opts?]` | Self-attention with causal mask |
| `ConvTranspose2D` | `ConvTranspose2D[opts?]` | Transposed 2D convolution (deconv) |
| `Convolution1D` | `Convolution1D[opts?]` | 1D convolution layer |
| `Convolution2D` | `Convolution2D[opts?]` | 2D convolution layer (uses InputChannels/Height/Width for forward) |
| `CrossAttention` | `CrossAttention[opts?]` | Cross-attention over Memory (seq x dim) |
| `Dense` | `Dense[opts?]` | Linear (fully-connected) layer |
| `DepthwiseConv2D` | `DepthwiseConv2D[opts?]` | Depthwise 2D convolution (per-channel) |
| `Embedding` | `Embedding[vocab, dim]` | Embedding lookup layer |
| `FFN` | `FFN[opts?]` | Position-wise feed-forward (supports SwiGLU/GEGLU) |
| `Flatten` | `Flatten[list, levels?]` | Flatten by levels (default 1) |
| `GlobalAvgPool2D` | `GlobalAvgPool2D[opts?]` | Global average pooling per channel over HxW |
| `GroupNorm` | `GroupNorm[opts?]` | Group normalization over channels (NumGroups) |
| `LayerNorm` | `LayerNorm[opts?]` | Layer normalization layer |
| `MultiHeadAttention` | `MultiHeadAttention[opts?]` | Self-attention with NumHeads (single-batch) |
| `PatchEmbedding2D` | `PatchEmbedding2D[opts?]` | 2D to tokens via patch conv |
| `Pooling` | `Pooling[kind, size, opts?]` | Pooling layer (Max/Avg) |
| `Pooling2D` | `Pooling2D[kind, size, opts?]` | 2D pooling layer (Max/Avg; requires InputChannels/Height/Width) |
| `PositionalEmbedding` | `PositionalEmbedding[opts?]` | Learnable positional embeddings (adds to input) |
| `PositionalEncoding` | `PositionalEncoding[opts?]` | Sinusoidal positional encoding (adds to input) |
| `RMSNorm` | `RMSNorm[opts?]` | RMS normalization (seq x dim) |
| `Residual` | `Residual[layers]` | Residual wrapper with inner layers (adds skip) |
| `ResidualBlock` | `ResidualBlock[opts?]` | Two convs + skip (MVP no norm) |
| `SeparableConv2D` | `SeparableConv2D[opts?]` | Depthwise + 1x1 pointwise convolution |
| `TransformerDecoder` | `TransformerDecoder[opts?]` | Decoder block: self-attn + cross-attn + FFN (single-batch) |
| `TransformerEncoder` | `TransformerEncoder[opts?]` | Encoder block: MHA + FFN with residuals and layer norms (single-batch) |
| `Transpose` | `Transpose[A, perm?]` | Transpose of a matrix (or NDTranspose for permutations). |
| `Upsample2D` | `Upsample2D[opts?]` | Upsample HxW (Nearest/Bilinear) |

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
| `ArgMax` | `ArgMax[f, list]` | 1-based index of maximal key. |
| `ArgMin` | `ArgMin[f, list]` | 1-based index of minimal key. |
| `CountBy` | `CountBy[f, list]` | Counts by key function (assoc) |
| `Drop` | `Drop[list, n]` | Drop first n (last if negative) |
| `DropWhile` | `DropWhile[pred, list]` | Drop while pred[x] holds |
| `Filter` | `Filter[pred, list]` | Keep elements where pred[x] is True |
| `Find` | `Find[pred, list]` | First element where pred[x] |
| `First` | `First[list]` | First element of a list (or Null). |
| `Init` | `Init[list]` | All but the last element. |
| `Join` | `Join[a|left, b|right, on?, how?]` | Join lists or datasets (dispatched). Overloads: Join[list1, list2]; Join[left, right, on, how?] |
| `Last` | `Last[list]` | Last element of a list (or Null). |
| `ListEdges` | `ListEdges[graph, opts?]` | List edges |
| `ListNodes` | `ListNodes[graph, opts?]` | List nodes |
| `MapAt` | `MapAt[f, subject, indexOrKey]` | Apply function at 1-based index or key. |
| `MapIndexed` | `MapIndexed[f, list]` | Map with index (1-based) |
| `MapThread` | `MapThread[f, lists]` | Map function over zipped lists (zip-with). |
| `MaxBy` | `MaxBy[f, list]` | Element with maximal derived key. |
| `MinBy` | `MinBy[f, list]` | Element with minimal derived key. |
| `Part` | `Part[subject, index]` | Index into list/assoc |
| `Partition` | `Partition[list, n, step?]` | Partition into fixed-size chunks |
| `Position` | `Position[pred, list]` | 1-based index of first match |
| `RandomChoice` | `RandomChoice[list]` | Random element from a list. |
| `Range` | `Range[a, b, step?]` | Create numeric range |
| `Reduce` | `Reduce[f, init?, list]` | Fold list with function |
| `Reject` | `Reject[pred, list]` | Drop elements where pred[x] is True |
| `ReplacePart` | `ReplacePart[subject, indexOrKey, value]` | Replace element at 1-based index or key. |
| `Rest` | `Rest[list]` | All but the first element. |
| `Sample` | `Sample[list, k]` | Sample k distinct elements from a list. |
| `Scan` | `Scan[f, init?, list]` | Prefix scan with function |
| `Shuffle` | `Shuffle[list]` | Shuffle list uniformly. |
| `Slice` | `Slice[list, start, len?]` | Slice list by start and length |
| `SubsetQ` | `SubsetQ[a, b]` | Is a subset of b? (sets, lists) |
| `Take` | `Take[list, n]` | Take first n (last if negative) |
| `TakeWhile` | `TakeWhile[pred, list]` | Take while pred[x] holds |
| `Tally` | `Tally[list]` | Counts by value (assoc) |
| `Total` | `Total[list]` | Sum elements in a list |
| `Union` | `Union[args]` | Union for lists (stable) or sets (dispatched) |
| `Unique` | `Unique[list]` | Stable deduplicate list |
| `UniqueBy` | `UniqueBy[f, list]` | Stable dedupe by derived key. |

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
| `Do` | `Do[body, n]` | Execute body n times. |
| `Equal` | `Equal[args…]` | Test equality across arguments |
| `EvenQ` | `EvenQ[n]` | Is integer even? |
| `For` | `For[init, test, step, body]` | C-style loop with init/test/step. |
| `Greater` | `Greater[args…]` | Strictly decreasing sequence |
| `GreaterEqual` | `GreaterEqual[args…]` | Non-increasing sequence |
| `Less` | `Less[args…]` | Strictly increasing sequence |
| `LessEqual` | `LessEqual[args…]` | Non-decreasing sequence |
| `Not` | `Not[x]` | Logical NOT |
| `OddQ` | `OddQ[n]` | Is integer odd? |
| `Or` | `Or[args…]` | Logical OR (short-circuit) |
| `While` | `While[test, body]` | Repeat body while test evaluates to True. |

## `mac`

| Function | Usage | Summary |
|---|---|---|
| `Hmac` | `Hmac[message, key, opts]` | HMAC (SHA-256 or SHA-512) |
| `HmacVerify` | `HmacVerify[message, key, signature, opts]` | Verify HMAC signature |

## `map`

| Function | Usage | Summary |
|---|---|---|
| `MapIndexed` | `MapIndexed[f, list]` | Map with index (1-based) |
| `MapThread` | `MapThread[f, lists]` | Map function over zipped lists (zip-with). |

## `math`

| Function | Usage | Summary |
|---|---|---|
| `Abs` | `Abs[x]` | Absolute value |
| `Correlation` | `Correlation[a, b]` | Pearson correlation of two numeric lists (population moments). |
| `Covariance` | `Covariance[a, b]` | Covariance of two numeric lists (population). |
| `EvenQ` | `EvenQ[n]` | Is integer even? |
| `Kurtosis` | `Kurtosis[data]` | Kurtosis (fourth standardized moment). |
| `Max` | `Max[args]` | Maximum of values or list |
| `Min` | `Min[args]` | Minimum of values or list |
| `Mode` | `Mode[data]` | Most frequent element (ties broken by first appearance). |
| `OddQ` | `OddQ[n]` | Is integer odd? |
| `Percentile` | `Percentile[data, p|list]` | Percentile(s) of numeric data using R-7 interpolation. |
| `Plus` | `Plus[a, b, …]` | Add numbers; Listable, Flat, Orderless. |
| `Quantile` | `Quantile[data, q|list]` | Quantile(s) of numeric data using R-7 interpolation. |
| `Range` | `Range[a, b, step?]` | Create numeric range |
| `Skewness` | `Skewness[data]` | Skewness (third standardized moment). |
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
| `Evaluate` | `Evaluate[model, data, opts?]` | Evaluate an ML model on data (dispatched) |
| `PredictMeasurements` | `PredictMeasurements[model, data, opts]` | Evaluate regressor metrics |

## `ml`

| Function | Usage | Summary |
|---|---|---|
| `Classifier` | `Classifier[opts?]` | Create classifier spec (Logistic/Baseline) |
| `Classify` | `Classify[data, opts]` | Train a classifier (baseline/logistic) |
| `ClassifyMeasurements` | `ClassifyMeasurements[model, data, opts]` | Evaluate classifier metrics |
| `Cluster` | `Cluster[data, opts]` | Cluster points (prototype) |
| `Clusterer` | `Clusterer[opts?]` | Create clusterer spec |
| `CrossValidate` | `CrossValidate[obj, data, opts?]` | Cross-validate estimator + data (dispatched) |
| `DimensionReduce` | `DimensionReduce[data, opts]` | Reduce dimensionality (PCA-like) |
| `Estimator` | `Estimator[opts]` | Create ML estimator spec (Task/Method) |
| `Evaluate` | `Evaluate[model, data, opts?]` | Evaluate an ML model on data (dispatched) |
| `FeatureExtract` | `FeatureExtract[data, opts]` | Learn preprocessing (impute/encode/standardize) |
| `MLApply` | `MLApply[model, x, opts]` | Apply a trained model to input |
| `MLCrossValidate` | `MLCrossValidate[data, opts]` | Cross-validate with simple split |
| `MLProperty` | `MLProperty[model, prop]` | Inspect trained model properties |
| `MLTune` | `MLTune[data, opts]` | Parameter sweep with basic scoring |
| `Predict` | `Predict[obj, x, opts?]` | Predict using a network or estimator (dispatched) |
| `PredictMeasurements` | `PredictMeasurements[model, data, opts]` | Evaluate regressor metrics |
| `Property` | `Property[obj, key]` | Property of a network or ML model (dispatch) |
| `Regressor` | `Regressor[opts?]` | Create regressor spec (Linear/Baseline) |
| `Train` | `Train[net, data, opts?]` | Train a network (dispatch to NetTrain) |
| `Tune` | `Tune[obj, data, opts?]` | Hyperparameter search for estimator (dispatched) |

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
| `Initialize` | `Initialize[net, opts?]` | Initialize a network (dispatch to NetInitialize) |

## `network`

| Function | Usage | Summary |
|---|---|---|
| `GraphNetwork` | `GraphNetwork[nodes, edges, opts?]` | Construct a graph network from nodes/edges |
| `Network` | `Network[opts?]` | Create network |
| `Sequential` | `Sequential[layers, opts?]` | Construct a sequential network from layers |
| `TransformerDecoderStack` | `TransformerDecoderStack[opts?]` | Stack N decoder blocks (returns Sequential network) |
| `TransformerEncoderDecoder` | `TransformerEncoderDecoder[opts?]` | Convenience: builds Encoder/Decoder stacks |
| `TransformerEncoderStack` | `TransformerEncoderStack[opts?]` | Stack N encoder blocks (returns Sequential network) |

## `nn`

| Function | Usage | Summary |
|---|---|---|
| `BatchNorm` | `BatchNorm[opts?]` | Batch normalization layer |
| `CausalSelfAttention` | `CausalSelfAttention[opts?]` | Self-attention with causal mask |
| `ConvTranspose2D` | `ConvTranspose2D[opts?]` | Transposed 2D convolution (deconv) |
| `Convolution1D` | `Convolution1D[opts?]` | 1D convolution layer |
| `Convolution2D` | `Convolution2D[opts?]` | 2D convolution layer (uses InputChannels/Height/Width for forward) |
| `CrossAttention` | `CrossAttention[opts?]` | Cross-attention over Memory (seq x dim) |
| `Dense` | `Dense[opts?]` | Linear (fully-connected) layer |
| `DepthwiseConv2D` | `DepthwiseConv2D[opts?]` | Depthwise 2D convolution (per-channel) |
| `Embedding` | `Embedding[vocab, dim]` | Embedding lookup layer |
| `FFN` | `FFN[opts?]` | Position-wise feed-forward (supports SwiGLU/GEGLU) |
| `Fit` | `Fit[net, data, opts?]` | Train a network (alias of NetTrain) |
| `Flatten` | `Flatten[list, levels?]` | Flatten by levels (default 1) |
| `Gelu` | `Gelu[x?]` | GELU activation (tanh approx): tensor op or zero-arg layer |
| `GlobalAvgPool2D` | `GlobalAvgPool2D[opts?]` | Global average pooling per channel over HxW |
| `GraphNetwork` | `GraphNetwork[nodes, edges, opts?]` | Construct a graph network from nodes/edges |
| `GroupNorm` | `GroupNorm[opts?]` | Group normalization over channels (NumGroups) |
| `Initialize` | `Initialize[net, opts?]` | Initialize a network (dispatch to NetInitialize) |
| `Initializer` | `Initializer[opts?]` | Initializer spec for layer parameters |
| `LayerNorm` | `LayerNorm[opts?]` | Layer normalization layer |
| `MultiHeadAttention` | `MultiHeadAttention[opts?]` | Self-attention with NumHeads (single-batch) |
| `Network` | `Network[opts?]` | Create network |
| `PatchEmbedding2D` | `PatchEmbedding2D[opts?]` | 2D to tokens via patch conv |
| `Pooling` | `Pooling[kind, size, opts?]` | Pooling layer (Max/Avg) |
| `Pooling2D` | `Pooling2D[kind, size, opts?]` | 2D pooling layer (Max/Avg; requires InputChannels/Height/Width) |
| `PositionalEmbedding` | `PositionalEmbedding[opts?]` | Learnable positional embeddings (adds to input) |
| `PositionalEncoding` | `PositionalEncoding[opts?]` | Sinusoidal positional encoding (adds to input) |
| `Predict` | `Predict[obj, x, opts?]` | Predict using a network or estimator (dispatched) |
| `Property` | `Property[obj, key]` | Property of a network or ML model (dispatch) |
| `RMSNorm` | `RMSNorm[opts?]` | RMS normalization (seq x dim) |
| `Relu` | `Relu[x]` | Rectified Linear Unit: max(0, x). Tensor-aware: elementwise on tensors. |
| `Residual` | `Residual[layers]` | Residual wrapper with inner layers (adds skip) |
| `ResidualBlock` | `ResidualBlock[opts?]` | Two convs + skip (MVP no norm) |
| `SeparableConv2D` | `SeparableConv2D[opts?]` | Depthwise + 1x1 pointwise convolution |
| `Sequential` | `Sequential[layers, opts?]` | Construct a sequential network from layers |
| `Sigmoid` | `Sigmoid[x?]` | Sigmoid activation: tensor op or zero-arg layer |
| `Softmax` | `Softmax[x?, opts?]` | Softmax activation: zero-arg layer (tensor variant TBD) |
| `Summary` | `Summary[obj]` | Summary of a network (dispatch to NetSummary) |
| `Train` | `Train[net, data, opts?]` | Train a network (dispatch to NetTrain) |
| `TransformerDecoder` | `TransformerDecoder[opts?]` | Decoder block: self-attn + cross-attn + FFN (single-batch) |
| `TransformerDecoderStack` | `TransformerDecoderStack[opts?]` | Stack N decoder blocks (returns Sequential network) |
| `TransformerEncoder` | `TransformerEncoder[opts?]` | Encoder block: MHA + FFN with residuals and layer norms (single-batch) |
| `TransformerEncoderDecoder` | `TransformerEncoderDecoder[opts?]` | Convenience: builds Encoder/Decoder stacks |
| `TransformerEncoderStack` | `TransformerEncoderStack[opts?]` | Stack N encoder blocks (returns Sequential network) |
| `Transpose` | `Transpose[A, perm?]` | Transpose of a matrix (or NDTranspose for permutations). |
| `Upsample2D` | `Upsample2D[opts?]` | Upsample HxW (Nearest/Bilinear) |

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
| `EmptyQ` | `EmptyQ[x]` | Is the subject empty? (lists, strings, assocs, handles) |
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
| `Execute` | `Execute[conn, sql, opts?]` | Execute DDL/DML (non-SELECT) |
| `Neighbors` | `Neighbors[graph, id, opts?]` | Neighbor node ids for a node |
| `Query` | `Query[conn, sql, opts?]` | Run a SELECT query and return rows |

## `random`

| Function | Usage | Summary |
|---|---|---|
| `RandomBytes` | `RandomBytes[len, opts]` | Generate cryptographically secure random bytes |
| `RandomChoice` | `RandomChoice[list]` | Random element from a list. |
| `RandomHex` | `RandomHex[len]` | Generate random hex string of n bytes |
| `RandomInteger` | `RandomInteger[spec?]` | Random integer; supports {min,max}. |
| `RandomReal` | `RandomReal[spec?]` | Random real; supports {min,max}. |
| `RandomVariate` | `RandomVariate[dist, n?]` | Sample from a distribution (optionally n samples). |
| `Sample` | `Sample[list, k]` | Sample k distinct elements from a list. |
| `SeedRandom` | `SeedRandom[seed?]` | Seed deterministic RNG scoped to this evaluator. |
| `Shuffle` | `Shuffle[list]` | Shuffle list uniformly. |

## `redirect`

| Function | Usage | Summary |
|---|---|---|
| `RespondRedirect` | `RespondRedirect[location, opts]` | Build a redirect response (Location header) |

## `regex`

| Function | Usage | Summary |
|---|---|---|
| `RegexCaptureNames` | `RegexCaptureNames[pattern]` | Ordered list of named capture groups. |
| `RegexFind` | `RegexFind[s, pattern]` | Find first regex capture groups |
| `RegexFindAll` | `RegexFindAll[s, pattern]` | Find all regex capture groups |
| `RegexGroups` | `RegexGroups[pattern, s]` | Capture groups of first match. |
| `RegexMatch` | `RegexMatch[s, pattern]` | Return first regex match |
| `RegexMatchQ` | `RegexMatchQ[pattern, s]` | Alias: regex predicate (Boolean) |
| `RegexReplace` | `RegexReplace[s, pattern, repl]` | Replace matches using regex |
| `RegexSplit` | `RegexSplit[pattern, s]` | Split string by regex pattern. |

## `regression`

| Function | Usage | Summary |
|---|---|---|
| `Regressor` | `Regressor[opts?]` | Create regressor spec (Linear/Baseline) |

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
| `EqualQ` | `EqualQ[a, b]` | Structural equality for sets and handles |
| `FrameUnion` | `FrameUnion[frames…]` | Union Frames by columns (schema union) |
| `SubsetQ` | `SubsetQ[a, b]` | Is a subset of b? (sets, lists) |
| `Union` | `Union[args]` | Union for lists (stable) or sets (dispatched) |
| `UnionByPosition` | `UnionByPosition[ds1, ds2, …]` | Union datasets by column position. |
| `Unique` | `Unique[list]` | Stable deduplicate list |
| `UniqueBy` | `UniqueBy[f, list]` | Stable dedupe by derived key. |

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
| `ArgMax` | `ArgMax[f, list]` | 1-based index of maximal key. |
| `ArgMin` | `ArgMin[f, list]` | 1-based index of minimal key. |
| `FrameSort` | `FrameSort[frame, by]` | Sort Frame by columns |
| `MaxBy` | `MaxBy[f, list]` | Element with maximal derived key. |
| `MinBy` | `MinBy[f, list]` | Element with minimal derived key. |
| `Sort` | `Sort[list|ds, by?, opts?]` | Sort a list or dataset (dispatched). Overloads: Sort[list]; Sort[ds, by, opts?] |
| `SortBy` | `SortBy[f, subject]` | Sort list by key or association by derived key. |

## `sql`

| Function | Usage | Summary |
|---|---|---|
| `Begin` | `Begin[conn]` | Begin a transaction |
| `Commit` | `Commit[conn]` | Commit the current transaction |
| `Connect` | `Connect[dsn|opts]` | Open database connection (mock/sqlite/duckdb) |
| `Disconnect` | `Disconnect[conn]` | Close a database connection |
| `Execute` | `Execute[conn, sql, opts?]` | Execute DDL/DML (non-SELECT) |
| `ExplainSQL` | `ExplainSQL[ds]` | Render SQL for pushdown-capable parts |
| `Fetch` | `Fetch[cursor, limit?]` | Fetch next batch of rows from a cursor |
| `InsertRows` | `InsertRows[conn, table, rows]` | Insert multiple rows (assoc list) into a table |
| `ListTables` | `ListTables[conn]` | List tables on a connection |
| `Ping` | `Ping[conn]` | Check connectivity for a database connection |
| `Query` | `Query[conn, sql, opts?]` | Run a SELECT query and return rows |
| `RegisterTable` | `RegisterTable[conn, name, rows]` | Register in-memory rows as a table (mock) |
| `Rollback` | `Rollback[conn]` | Rollback the current transaction |
| `SQLCursor` | `SQLCursor[conn, sql, params?]` | Run a query and return a cursor handle |
| `Table` | `Table[conn, name]` | Reference a table as a Dataset |
| `UpsertRows` | `UpsertRows[conn, table, rows, keys?]` | Upsert rows (assoc list) into a table |
| `WriteDataset` | `WriteDataset[conn, table, dataset, opts?]` | Write a Dataset into a table |

## `stats`

| Function | Usage | Summary |
|---|---|---|
| `Bernoulli` | `Bernoulli[p]` | Bernoulli distribution head (probability p). |
| `BinomialDistribution` | `BinomialDistribution[n, p]` | Binomial distribution head (trials n, prob p). |
| `CDF` | `CDF[dist, x]` | Cumulative distribution for a distribution at x. |
| `Correlation` | `Correlation[a, b]` | Pearson correlation of two numeric lists (population moments). |
| `Covariance` | `Covariance[a, b]` | Covariance of two numeric lists (population). |
| `Describe` | `Describe[name, items, opts?]` | Define a test suite (held). |
| `Exponential` | `Exponential[lambda]` | Exponential distribution head (rate λ). |
| `FrameDescribe` | `FrameDescribe[frame, opts?]` | Quick stats by columns |
| `Gamma` | `Gamma[k, theta]` | Gamma distribution head (shape k, scale θ). |
| `Kurtosis` | `Kurtosis[data]` | Kurtosis (fourth standardized moment). |
| `Mode` | `Mode[data]` | Most frequent element (ties broken by first appearance). |
| `Normal` | `Normal[mu, sigma]` | Normal distribution head (mean μ, stddev σ). |
| `PDF` | `PDF[dist, x]` | Probability density/mass for a distribution at x. |
| `Percentile` | `Percentile[data, p|list]` | Percentile(s) of numeric data using R-7 interpolation. |
| `Poisson` | `Poisson[lambda]` | Poisson distribution head (rate λ). |
| `Quantile` | `Quantile[data, q|list]` | Quantile(s) of numeric data using R-7 interpolation. |
| `RandomVariate` | `RandomVariate[dist, n?]` | Sample from a distribution (optionally n samples). |
| `Skewness` | `Skewness[data]` | Skewness (third standardized moment). |

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
| `CaseFold` | `CaseFold[s]` | Unicode case-fold |
| `EndsWith` | `EndsWith[s, suffix]` | True if string ends with suffix |
| `HtmlAttr` | `HtmlAttr[s]` | Escape string for HTML attribute context |
| `HtmlEscape` | `HtmlEscape[s]` | Escape string for HTML |
| `HtmlTemplate` | `HtmlTemplate[templateOrPath, data, opts?]` | Render HTML/XML templates with Mustache semantics (sections, inverted, partials, comments, indented partials, standalone trimming; unescaped via {{{...}}} or {{& name}}). Options: Mode(html\|xml), Strict, Whitespace(preserve\|trim-tags\|smart), Partials, Components, Layout, Loader. |
| `HtmlTemplateCompile` | `HtmlTemplateCompile[templateOrPath, opts?]` | Precompile HTML template (returns handle) |
| `HtmlTemplateRender` | `HtmlTemplateRender[handle, data, opts?]` | Render compiled HTML template with data |
| `HtmlUnescape` | `HtmlUnescape[s]` | Unescape HTML-escaped string |
| `JsonEscape` | `JsonEscape[s]` | Escape string for JSON |
| `JsonUnescape` | `JsonUnescape[s]` | Unescape JSON-escaped string |
| `NormalizeUnicode` | `NormalizeUnicode[s, form?]` | Normalize to NFC/NFD/NFKC/NFKD |
| `RegexCaptureNames` | `RegexCaptureNames[pattern]` | Ordered list of named capture groups. |
| `RegexFind` | `RegexFind[s, pattern]` | Find first regex capture groups |
| `RegexFindAll` | `RegexFindAll[s, pattern]` | Find all regex capture groups |
| `RegexGroups` | `RegexGroups[pattern, s]` | Capture groups of first match. |
| `RegexMatch` | `RegexMatch[s, pattern]` | Return first regex match |
| `RegexMatchQ` | `RegexMatchQ[pattern, s]` | Alias: regex predicate (Boolean) |
| `RegexReplace` | `RegexReplace[s, pattern, repl]` | Replace matches using regex |
| `RegexSplit` | `RegexSplit[pattern, s]` | Split string by regex pattern. |
| `RemoveDiacritics` | `RemoveDiacritics[s]` | Strip diacritics (stub) |
| `SafeHtml` | `SafeHtml[s]` | Mark string as safe HTML (no escaping) |
| `Slugify` | `Slugify[s]` | Slugify for URLs |
| `Split` | `Split[s, sep]` | Split string by separator |
| `StartsWith` | `StartsWith[s, prefix]` | True if string starts with prefix |
| `StringContains` | `StringContains[s, substr]` | Does string contain substring? |
| `StringJoin` | `StringJoin[parts]` | Concatenate list of parts. |
| `StringReplace` | `StringReplace[s, from, to]` | Replace all substring matches |
| `StringReplaceFirst` | `StringReplaceFirst[s, from, to]` | Replace first substring match |
| `StringTrim` | `StringTrim[s]` | Trim whitespace from both ends |
| `TemplateRender` | `TemplateRender[template, data, opts?]` | Render Mustache-like template with assoc data. |
| `ToLower` | `ToLower[s]` | Lowercase string. |
| `ToUpper` | `ToUpper[s]` | Uppercase string. |
| `Transliterate` | `Transliterate[s]` | Transliterate to ASCII (stub) |
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

## `template`

| Function | Usage | Summary |
|---|---|---|
| `HtmlTemplate` | `HtmlTemplate[templateOrPath, data, opts?]` | Render HTML/XML templates with Mustache semantics (sections, inverted, partials, comments, indented partials, standalone trimming; unescaped via {{{...}}} or {{& name}}). Options: Mode(html\|xml), Strict, Whitespace(preserve\|trim-tags\|smart), Partials, Components, Layout, Loader. |
| `HtmlTemplateCompile` | `HtmlTemplateCompile[templateOrPath, opts?]` | Precompile HTML template (returns handle) |
| `HtmlTemplateRender` | `HtmlTemplateRender[handle, data, opts?]` | Render compiled HTML template with data |
| `TemplateRender` | `TemplateRender[template, data, opts?]` | Render Mustache-like template with assoc data. |

## `tensor`

| Function | Usage | Summary |
|---|---|---|
| `Gelu` | `Gelu[x?]` | GELU activation (tanh approx): tensor op or zero-arg layer |
| `Relu` | `Relu[x]` | Rectified Linear Unit: max(0, x). Tensor-aware: elementwise on tensors. |
| `Reshape` | `Reshape[tensor, dims]` | Reshape a tensor to new dims (supports -1) |
| `Shape` | `Shape[tensor]` | Shape of a tensor |
| `Sigmoid` | `Sigmoid[x?]` | Sigmoid activation: tensor op or zero-arg layer |
| `Transpose` | `Transpose[A, perm?]` | Transpose of a matrix (or NDTranspose for permutations). |

## `testing`

| Function | Usage | Summary |
|---|---|---|
| `Describe` | `Describe[name, items, opts?]` | Define a test suite (held). |

## `text`

| Function | Usage | Summary |
|---|---|---|
| `Split` | `Split[s, sep]` | Split string by separator |
| `StringJoin` | `StringJoin[parts]` | Concatenate list of parts. |
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
| `Fit` | `Fit[net, data, opts?]` | Train a network (alias of NetTrain) |
| `Train` | `Train[net, data, opts?]` | Train a network (dispatch to NetTrain) |

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

## `transformer`

| Function | Usage | Summary |
|---|---|---|
| `CausalSelfAttention` | `CausalSelfAttention[opts?]` | Self-attention with causal mask |
| `CrossAttention` | `CrossAttention[opts?]` | Cross-attention over Memory (seq x dim) |
| `FFN` | `FFN[opts?]` | Position-wise feed-forward (supports SwiGLU/GEGLU) |
| `MultiHeadAttention` | `MultiHeadAttention[opts?]` | Self-attention with NumHeads (single-batch) |
| `PatchEmbedding2D` | `PatchEmbedding2D[opts?]` | 2D to tokens via patch conv |
| `PositionalEmbedding` | `PositionalEmbedding[opts?]` | Learnable positional embeddings (adds to input) |
| `PositionalEncoding` | `PositionalEncoding[opts?]` | Sinusoidal positional encoding (adds to input) |
| `RMSNorm` | `RMSNorm[opts?]` | RMS normalization (seq x dim) |
| `TransformerDecoder` | `TransformerDecoder[opts?]` | Decoder block: self-attn + cross-attn + FFN (single-batch) |
| `TransformerDecoderStack` | `TransformerDecoderStack[opts?]` | Stack N decoder blocks (returns Sequential network) |
| `TransformerEncoder` | `TransformerEncoder[opts?]` | Encoder block: MHA + FFN with residuals and layer norms (single-batch) |
| `TransformerEncoderDecoder` | `TransformerEncoderDecoder[opts?]` | Convenience: builds Encoder/Decoder stacks |
| `TransformerEncoderStack` | `TransformerEncoderStack[opts?]` | Stack N encoder blocks (returns Sequential network) |

## `traversal`

| Function | Usage | Summary |
|---|---|---|
| `BFS` | `BFS[graph, start, opts?]` | Breadth-first search order |
| `DFS` | `DFS[graph, start, opts?]` | Depth-first search order |

## `tune`

| Function | Usage | Summary |
|---|---|---|
| `MLTune` | `MLTune[data, opts]` | Parameter sweep with basic scoring |
| `Tune` | `Tune[obj, data, opts?]` | Hyperparameter search for estimator (dispatched) |

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

## `unicode`

| Function | Usage | Summary |
|---|---|---|
| `CaseFold` | `CaseFold[s]` | Unicode case-fold |
| `NormalizeUnicode` | `NormalizeUnicode[s, form?]` | Normalize to NFC/NFD/NFKC/NFKD |
| `RemoveDiacritics` | `RemoveDiacritics[s]` | Strip diacritics (stub) |
| `Transliterate` | `Transliterate[s]` | Transliterate to ASCII (stub) |

## `update`

| Function | Usage | Summary |
|---|---|---|
| `MapAt` | `MapAt[f, subject, indexOrKey]` | Apply function at 1-based index or key. |
| `ReplacePart` | `ReplacePart[subject, indexOrKey, value]` | Replace element at 1-based index or key. |

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

## `vision`

| Function | Usage | Summary |
|---|---|---|
| `PatchEmbedding2D` | `PatchEmbedding2D[opts?]` | 2D to tokens via patch conv |

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
| `Watch` | `Watch[path, handler, opts?]` | Watch directory and stream events (held) |
| `WatchDirectory` | `WatchDirectory[path, handler, opts?]` | Watch directory and stream events (held) |

## `write`

| Function | Usage | Summary |
|---|---|---|
| `InsertRows` | `InsertRows[conn, table, rows]` | Insert multiple rows (assoc list) into a table |
| `UpsertRows` | `UpsertRows[conn, table, rows, keys?]` | Upsert rows (assoc list) into a table |
| `WriteDataset` | `WriteDataset[conn, table, dataset, opts?]` | Write a Dataset into a table |

## `xml`

| Function | Usage | Summary |
|---|---|---|
| `HtmlTemplate` | `HtmlTemplate[templateOrPath, data, opts?]` | Render HTML/XML templates with Mustache semantics (sections, inverted, partials, comments, indented partials, standalone trimming; unescaped via {{{...}}} or {{& name}}). Options: Mode(html\|xml), Strict, Whitespace(preserve\|trim-tags\|smart), Partials, Components, Layout, Loader. |

## `yaml`

| Function | Usage | Summary |
|---|---|---|
| `YamlParse` | `YamlParse[yaml]` | Parse YAML string to value |
| `YamlStringify` | `YamlStringify[value, opts]` | Render value as YAML |
