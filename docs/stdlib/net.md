HTTP / Net

Overview
- Client requests (GET/POST/etc.), generic request builder, streaming download, cached download, retry wrapper.

Core
- HttpGet[url, opts?] -> <|status, url, headers, body|json|bytes|>
- HttpRequest[<|Method, Url, Headers?, Query?, Body?, ContentType?, TimeoutMs?, ConnectTimeoutMs?, FollowRedirects?, MaxRedirects?, VerifyTls?, Proxy?|>] -> response
  - Body: String | JSON value | Bytes (list of 0..255)
  - decode heuristic: JSON if Content-Type contains application/json; can override via opts in net module.

Streaming
- HttpStreamRequest[method, url, opts?] -> <|status, url, headers, stream|>
- HttpStreamRead[stream, max_bytes?] -> <|chunk: bytes, done: Bool|>
- HttpStreamClose[stream] -> Unit

Caching & Retry
- HttpDownloadCached[url, dest, <|CacheDir, TtlMs, Overwrite, CreateDirs|> & HttpOptions] -> <|path, bytes_written, from_cache, etag?|>
- HttpRetry[requestMap, <|Attempts, BackoffMs, JitterPct?, RetryOn?|>] -> response
  - Current backoff is linear by BackoffMs; extendable to min/max/factor.

Options (HttpOptions)
- Headers: Assoc
- Query: Assoc
- TimeoutMs, ConnectTimeoutMs
- FollowRedirects (Bool), MaxRedirects (Int)
- VerifyTls (Bool)
- Proxy: <|http, https, no_proxy|>

Examples
- `HttpGet("https://example.com", {TimeoutMs: 5000})`
- `HttpRequest(<|Method:"POST", Url:url, Headers:<|content-type:"application/json"|>, Body: JsonStringify(obj)|>)`
- `r = HttpStreamRequest("GET", url, {}); loop { ch = HttpStreamRead[r.stream, 65536]; If[ ch.done, Break[] ]; WriteFile["out.bin", ch.chunk] }; HttpStreamClose[r.stream]`
- `HttpDownloadCached(url, "./file.bin", {CacheDir:"./.cache", TtlMs:86400000})`
- `HttpRetry(<|Method:"GET", Url:url|>, {Attempts:3, BackoffMs:200})`

Failures
- HTTP::error, HTTP::status, HTTP::download, HTTP::cache, HTTP::stream

