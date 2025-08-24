# Lyra Stdlib API Proposals (Draft)

Status: Draft for review
Scope: Public Lyra function APIs (Wolfram-style), options, return types, and foreign object types for the next stdlib expansion. Implementation to follow.

## Conventions
- Function form: `Name[arg1, arg2, ..., opts]` with `opts` as `Association` (`<|key -> value|>`), optional unless stated.
- Options: All options are case-sensitive strings; unknown options cause a validation error.
- Returns: Values, lists, or foreign objects (documented per API). Errors raised via standard error channels.
- Streams and handles: Long-lived resources are foreign objects with method access and `Close[handle]` when applicable.
- Timeouts: Milliseconds unless stated. Sizes: bytes unless stated.

---

## 1) Serialization & Formats

### YAML & TOML
- `YAMLParse[string | file | bytes, opts] -> Any`
  - opts: `<|schema -> schemaSpec?, allowAnchors -> True|False (default True)|>`
  - errors: invalid YAML, schema mismatch
- `YAMLStringify[expr, opts] -> String`
  - opts: `<|indent -> 2|4, flowStyle -> "auto"|"block"|"flow"|>`
- `TOMLParse[string | file | bytes] -> Association`
- `TOMLStringify[assoc, opts] -> String`

### Binary Formats
- `CBORRead[input, opts] -> Any`
- `CBORWrite[expr, opts] -> Bytes`
- `MessagePackRead[input, opts] -> Any`
- `MessagePackWrite[expr, opts] -> Bytes`
- `BincodeRead[input] -> Any`
- `BincodeWrite[expr] -> Bytes`

### JSON/YAML/TOML Query
- `JSONPath[json, pathString] -> Any | Missing`
- `YAMLPath[yaml, pathString] -> Any | Missing`

### Compression
- `Compress[data | file, algo:"gzip"|"zstd"|"lz4"|"snappy", opts] -> Bytes | File`
- `Decompress[data | file, opts] -> Bytes | File`
  - opts: `<|level -> 0..11 (algo-dependent), stream -> True|False|>`

---

## 2) Config & Secrets

### Config
- `ConfigLoad[path | string, opts] -> Association`
  - opts: `<|format -> "auto"|"json"|"yaml"|"toml"|>`
- `ConfigMerge[assoc1, assoc2, opts] -> Association`
  - opts: `<|deep -> True (default) | False|>`
- `ConfigOverlayEnv[assoc, opts] -> Association`
  - opts: `<|prefix -> "LYRA_", separator -> "__"|>`
- `ConfigGet[assoc, keyPath:"a.b.c", default?] -> Any`
- `ConfigSet[assoc, keyPath, value] -> Association`
- `ConfigValidate[assoc, schema] -> Ok|Error`

### Secrets
- `SecretGet[name, opts] -> String | Bytes`
- `SecretSet[name, value, opts] -> Ok`
  - opts: `<|provider -> "env"|"dotenv"|"keychain"|"vault"|"aws-kms"|"gcp-kms", namespace -> "app", version -> "latest"|>`

---

## 3) Security & Identity

### JWT
- `JWTEncode[claims_Association, secret|keyObj, opts] -> String`
  - opts: `<|alg -> "HS256"|"RS256"|"ES256", expiresIn -> 3600, issuer -> "", audience -> ""|>`
- `JWTDecode[token_String, opts] -> Association`
- `JWTVerify[token, secret|keyObj, opts] -> True | Error`

### OAuth2 / OIDC
- `OAuth2ClientCredentials[tokenUrl, clientId, clientSecret, opts] -> Association(*token*)`
  - opts: `<|scopes -> {"scope1", ...}, audience -> ""|>`
- `OAuth2AuthCodePKCE[authUrl, tokenUrl, clientId, redirectUri, opts] -> Association`
- `OpenIDDiscover[issuerUrl] -> Association(*discovery doc*)`
- `TokenIntrospect[introspectUrl, token, opts] -> Association`

### TLS / Certificates
- `CertParse[input] -> CertObject`
- `CertChainValidate[{certs...}, opts] -> Result`
- `KeyLoad[input] -> KeyObject`
- `PKCS12Load[input, password] -> {CertObject, KeyObject}`
- `MTLSClient[httpClient, cert, key, opts] -> HttpClient`
- `MTLSServer[server, cert, key, opts] -> Server`

---

## 4) HTTP Enhancements & GraphQL

### HTTP Enhancements
- `HTTPRetry[request|func, opts] -> Response | Value`
  - opts: `<|retries -> 3, backoff -> "exp"|"jitter"|"fixed", baseMs -> 100, circuitBreaker -> None|>`
- `RateLimit[func, qps] -> WrappedFunc`
- `CircuitBreaker[func, opts] -> WrappedFunc`
- `CookieJar[client, opts] -> Client` (exposes cookie store)

### GraphQL
- `GraphQLClient[endpoint, opts] -> GraphQLClientObject`
- `GraphQLQuery[client|endpoint, queryString, variables_Association?] -> Association`
- `GraphQLIntrospect[client|endpoint] -> Association(*schema*)`
- `GraphQLValidate[schema, resolvers] -> Result`
- `GraphQLServer[schema, resolvers, opts] -> Server`
  - opts: `<|port -> 4000, playground -> True|>`

---

## 5) Streaming & Realtime

### Event Streams & Queues
- `TopicCreate[name, opts] -> Ok`
- `Publish[topicOrClient, message, opts] -> Ok`
- `Consume[topicOrClient, opts] -> Stream`
- `Ack[message] -> Ok`
- `Offsets[topicOrClient] -> Association`
  - opts common: `<|backend -> "kafka"|"nats"|"redis"|"memory", group -> "", autoCommit -> True, batchSize -> 1000|>`

### Realtime APIs
- `WebSocketServer[url, opts] -> Server`
- `WebSocketClient[url, opts] -> Client`
- `SSEServer[url, opts] -> Server`
- `SSEClient[url, opts] -> Client`
- `gRPCServer[serviceDef, opts] -> Server`
- `gRPCClient[endpoint, opts] -> Client`

### Stream Processing
- `Window[stream, type:"tumbling"|"sliding"|"session", size, opts] -> Stream`
- `Watermark[stream, tsExtractor, latenessMs] -> Stream`
- `JoinStream[left, right, keyFn, opts] -> Stream`
- `AggregateStream[stream, aggFn, opts] -> Stream`
- `MapAsync[stream, fn, opts] -> Stream`
- `Sink[stream, target, opts] -> Result`

---

## 6) Columnar Data & DataFrames

### Arrow & Parquet
- `ArrowTable[data|schema] -> ArrowTableObject`
- `ArrowRead[path|bytes, opts] -> ArrowTableObject`
- `ArrowWrite[arrowTable, path, opts] -> Ok`
- `ParquetRead[path|paths, opts] -> ArrowTableObject`
- `ParquetWrite[arrowTable, path, opts] -> Ok`
  - opts: `<|projection -> {cols...}, predicate -> expr, batchSize -> 65536, compression -> "zstd"|>`

### DataFrame API (on Arrow backend)
- `DataFrame[table|rows|schema] -> DataFrameObject`
- `Select[df, {cols...}] -> DataFrameObject`
- `Where[df, predicate] -> DataFrameObject`
- `Join[df1, df2, key|{keys...}, opts] -> DataFrameObject`
- `GroupBy[df, key|{keys...}] -> GroupedDataFrame`
- `Aggregate[grouped, aggSpec_Association] -> DataFrameObject`
- `Window[df, spec] -> DataFrameObject`
- `WithColumn[df, name, expr] -> DataFrameObject`
- `Collect[df] -> Table | List`

---

## 7) GPU / Device‑Aware Tensors

- `TensorDevice[tensor] -> "CPU"|"GPU[n]"`
- `ToDevice[tensor, device:"CPU"|"GPU[n]"] -> tensor`
- `TensorAllocate[shape, opts:<|device->..., dtype->...|>] -> tensor`
- GPU‑accelerated ops use same names; dispatch determined by device.

---

## 8) Vector Search & IR

### ANN Indexes
- `HNSWIndex[dim, opts] -> IndexObject`
- `IVFPQIndex[dim, opts] -> IndexObject`
- `IndexAdd[index, vectors, ids?] -> Ok`
- `IndexTrain[index, vectors] -> Ok`
- `IndexSearch[index, query|queries, k, opts] -> {ids, scores}`
- `IndexPersist[index, path] -> Ok`
- `IndexLoad[path] -> IndexObject`
- `IndexStats[index] -> Association`

### Text IR
- `TextIndex[docs, opts] -> TextIndexObject`
- `TextSearchBM25[index, query, k] -> {ids, scores}`
- `HybridSearch[annIndex, textIndex, query, k, opts] -> {ids, score}`

---

## 9) Units & Quantities

- `Quantity[value, unit_String] -> Quantity`
- `UnitConvert[q, targetUnit] -> Quantity`
- `UnitDimensions[q] -> Association`
- `UnitSimplify[q] -> Quantity`
- `UnitEquivalents[unit] -> {units...}`

---

## 10) Numerical & CAS

### ODE/PDE / Sparse
- `NDSolve[eqns, vars, {t, t0, t1}, opts] -> SolutionObject`
- `PDEModel[eqns, domain, bcs, opts] -> Model`
- `SolvePDE[model, opts] -> Solution`
- `LinearSolve[A, b, opts:<|method->"cg"|"gmres", preconditioner->...|>] -> x`

### Algebra & Polynomial
- `Factor[expr] -> expr`
- `Expand[expr] -> expr`
- `GroebnerBasis[{polys...}, vars, opts] -> {polys...}`
- `Solve[eqns, vars, opts] -> Rules`
- `Reduce[eqns, vars, opts] -> Conditions`
- `PolynomialMod[poly, m] -> poly`

### Interval
- `Interval[{lo, hi}] -> Interval`
- `IntervalArithmetic[op, args...] -> Interval`

---

## 11) Geospatial

### Types & IO
- `GeoJSONRead[path|string] -> GeoObject`
- `GeoJSONWrite[obj, path] -> Ok`
- `WKTRead[string] -> GeoObject`
- `WKTWrite[obj] -> String`
- `ShapefileRead[path] -> {GeoObject...}`

### Ops
- `Reproject[obj, targetEPSG] -> obj`
- `Within[a, b] -> True|False`
- `Contains[a, b] -> True|False`
- `Intersects[a, b] -> True|False`
- `Buffer[obj, distance, opts] -> obj`
- `Simplify[obj, tolerance] -> obj`
- `Union[objs...] -> obj`
- `Voronoi[points] -> cells`
- `Delaunay[points] -> triangles`
- `KNN[points, query, k, opts:<|metric->"haversine"|"euclidean"|>] -> {idxs, dists}`

---

## 12) Media (Audio/Video)

### Audio
- `AudioRead[path|bytes, opts] -> Audio`
- `AudioWrite[audio, path, opts] -> Ok`
- `Spectrogram[audio, opts] -> Tensor`
- `Resample[audio, rate] -> Audio`
- `AudioFilter[audio, type, opts] -> Audio`

### Video
- `VideoRead[path, opts] -> Video`
- `VideoWrite[video, path, opts] -> Ok`
- `FrameExtract[video, spec] -> {frames}`
- `Transcode[video, opts] -> Video`
- `VideoInfo[path] -> Association`

---

## 13) Caching & Memoization

### Caches
- `CacheCreate[type:"LRU"|"LFU", capacity, opts:<|ttlMs->..., shard->...|>] -> Cache`
- `CacheGet[cache, key] -> Any | Missing`
- `CachePut[cache, key, value, opts:<|ttlMs->...|>] -> Ok`
- `CacheInvalidate[cache, key|All] -> Ok`
- `CacheStats[cache] -> Association`

### Function
- `Memoize[f, opts:<|maxEntries->..., ttlMs->..., keyFn->Function|>] -> f'`
- `Unmemoize[f'] -> f`

---

## 14) Resilience & Concurrency

### Structured Concurrency
- `TaskScope[fn] -> Result`
- `TaskSpawn[fn] -> Task`
- `TaskJoinAll[{tasks...}, opts:<|timeoutMs->...|>] -> {results|errors}`
- `TaskCancel[task] -> Ok`
- `Timeout[fn, ms] -> Result`
- `Deadline[fn, time] -> Result`

### Resilience
- `Retry[fn|thunk, opts:<|retries->3, backoff->..., jitter->...|>] -> Result`
- `Backoff[policy, opts] -> BackoffPolicy`
- `RateLimit[fn, qps] -> fn'`
- `CircuitBreaker[fn, opts:<|failureThreshold->..., resetMs->...|>] -> fn'`

---

## 15) Observability

### OpenTelemetry
- `TraceStart[name, opts] -> Span`
- `TraceSpan[span, name, opts] -> Span`
- `TraceEnd[span] -> Ok`
- `TraceInject[carrier_Association] -> Association`
- `TraceExtract[carrier_Association] -> Context`
- `MetricCounter[name, opts] -> Counter`
- `MetricGauge[name, opts] -> Gauge`
- `MetricHistogram[name, opts] -> Histogram`
- `OTelExport[endpoint, protocol:"http"|"grpc", opts] -> Ok`

### Logging
- `LogStructured[level, message, fields_Association] -> Ok`
- `LogSink["stdout"|"file"|"http", opts] -> Ok`

---

## 16) Cloud & Storage Connectors

### Object Storage – Generic & Specific

Design: Provide both generic, pluggable `ObjectStore*` APIs and provider‑specific shorthands (S3/GCS/Azure). The generic functions accept a `uri` or an opened `ObjectStore` handle and route to the right backend based on scheme (`s3://`, `gs://`, `az://`, `file://`, `memory://`). Specific functions are thin wrappers that set defaults and validate provider‑specific options.

Common return types: `Bytes` for small reads, `Stream` or `ObjectReader` for large reads when `stream -> True`. Writes accept `Bytes`, `List[Bytes]`, `Stream`, or `File` path. All list calls return a list of Associations with at least `Key`, `Size`, `ETag?`, `LastModified?`, `StorageClass?`.

Credentials & config resolution (precedence): explicit opts > env vars > shared config/metadata > default runtime identity. All functions share options and error types.

Generic APIs
- `ObjectStoreOpen[uri|opts] -> ObjectStore`
  - Example: `ObjectStoreOpen[<|provider->"s3", region->"us-east-1", bucket->"b"|>]`
- `ObjectStoreRead[uri|{store, key}, opts] -> Bytes | Stream | ObjectReader`
- `ObjectStoreWrite[uri|{store, key}, data|Stream, opts] -> Ok`
- `ObjectStoreDelete[uri|{store, key}, opts] -> Ok`
- `ObjectStoreList[uri|{store, prefix}, opts] -> {objects}`
- `ObjectStoreHead[uri|{store, key}] -> Association`
- `ObjectStoreCopy[srcUri, dstUri, opts] -> Ok`
- `ObjectStorePresign[uri, method:"GET"|"PUT"|"HEAD", opts] -> Url`

Generic options (subset depends on method)
  - `<| provider->"s3"|"gcs"|"azure"|"file"|"memory",
        region->"us-east-1",
        endpoint->"https://..." (S3‑compatible),
        bucket->"name", key->"path", 
        accessKey->..., secretKey->..., sessionToken->..., profile->..., roleArn->..., audience->...,
        requestPayer->"requester",
        sse->"SSE-S3"|"SSE-KMS", kmsKeyId->..., 
        range->{start, end}|All,
        recursive->True|False,
        retries->3, backoff->"exp"|"jitter"|"fixed", baseMs->100,
        parallelism->4, partSizeMB->8, stream->False, bufferSize->8_388_608,
        contentType->"application/octet-stream",
        metadata-><|...|>, 
        expiresSec->900 (for presign)
     |>`

Provider‑Specific APIs (wrappers)
- `S3Read["s3://bucket/key"|{bucket, key}, opts] -> Bytes|Stream`
- `S3Write["s3://bucket/key"|{bucket, key}, data|Stream, opts] -> Ok`
- `S3List["s3://bucket/prefix/"|{bucket, prefix}, opts] -> {objects}`
- `S3Delete[uri, opts] -> Ok`, `S3Head[uri] -> Association`, `S3Presign[uri, method, opts] -> Url`
  - S3 opts: `<|region, endpoint, sse, kmsKeyId, requestPayer, acl, storageClass|>`

- `GCSRead/Write/List/Delete/Head/Presign` with opts like `<|projectId, location, serviceAccount, uniformAccess->True|>` and URIs `gs://bucket/key`.

- `AzureBlobRead/Write/List/Delete/Head/Presign` with opts like `<|accountName, container, sasToken, credential, endpoint|>` and URIs `az://container/blob`.

Notes
- If a `uri` is given to a provider‑specific function with a mismatched scheme, raise a clear error.
- For large writes, `ObjectStoreWrite` auto‑selects multipart/chunked uploads based on `partSizeMB` and size.
- Range reads via `range->{start,end}`; `All` reads entire object.
- `ObjectReader` foreign object exposes methods: `Read[bytes?]`, `Seek[pos]`, `Close[]`, `BytesRead[]`.

Examples
```
(* Generic *)
ObjectStoreRead["s3://my-bucket/data.parquet", <|range -> {0, 1048575}, retries -> 5|>]
ObjectStoreWrite["gs://ml-bucket/model.bin", bytes, <|contentType -> "application/octet-stream"|>]
ObjectStoreList["az://logs/2025/", <|recursive -> True|>]

(* Specific *)
S3Read["s3://b/k", <|region -> "us-west-2", sse -> "SSE-KMS", kmsKeyId -> "arn:aws:kms:..."|>]
GCSWrite["gs://b/ckpt", stream, <|location -> "us-central1"|>]
AzureBlobPresign["az://container/blob", "GET", <|expiresSec -> 600|>]
```

### Embedded Stores
- `RocksDBOpen[path, opts] -> DB`
- `RocksDBGet[db, key] -> Bytes|Missing`
- `RocksDBPut[db, key, val] -> Ok`
- `RocksDBScan[db, prefix, opts] -> {{k,v}...}`
- `SQLite[dbPath, query, params_List?] -> Table`

---

## 17) Ergonomics

### Prelude
- Define `Prelude[]` importing: math, list, string, table, data IO, HTTP, config, basic plotting/visualization when enabled.

### Options Pattern
- Standardize common options keys across IO/network/compute: `Timeout`, `Retries`, `Backoff`, `MemoryLimit`, `Device`, `Parallelism`, `Seed`.

---

## Foreign Object Types (abbreviated)
- `ArrowTableObject`, `DataFrameObject`, `IndexObject(HNSW|IVFPQ)`, `GraphQLClientObject`, `GraphQLServerObject`, `WebSocketClient/Server`, `Stream`, `Audio`, `Video`, `Quantity`, `Cache`, `Task`, `Span`, `Counter/Gauge/Histogram`, `DB`, `GeoObject`, `CertObject`, `KeyObject`, `HttpClient`, `Server`.

## Error Model
- Type: input validation (wrong arity/type), option validation, IO/network errors, auth/crypto errors, backend errors (GPU/DB), schema validation.
- Return `Result` semantics at VM level: raise with informative messages and recovery hints.

## Versioning & Features
- Feature flags for heavy deps: `gpu`, `arrow_parquet`, `graphQL`, `grpc`, `cloud`, `audio_video`, `geospatial`, `vector_search`, `otel`.
- Backends chosen via options; graceful no-op or clear error when feature disabled.

---

This document specifies function names, argument patterns, high-level return types, and options to wire into `src/stdlib/mod.rs` by module. After review, we will scaffold stubs and documentation examples, then implement incrementally behind feature flags.
