# NET

| Function | Usage | Summary |
|---|---|---|
| `AuthJwt` | `AuthJwt[]` |  |
| `AuthJwtApply` | `AuthJwtApply[]` |  |
| `CookiesHeader` | `CookiesHeader[cookies]` | Build a Cookie header string from an assoc |
| `Cors` | `Cors[opts]` | CORS middleware builder |
| `CorsApply` | `CorsApply[cors, req]` | Apply CORS to request/response |
| `GetResponseCookies` | `GetResponseCookies[response]` | Parse Set-Cookie headers from a response map |
| `HttpDelete` | `HttpDelete[url, opts]` | HTTP DELETE request (http/https) |
| `HttpDownloadCached` | `HttpDownloadCached[url, path, opts?]` | Download a URL to a file with ETag/TTL caching; returns path and bytes written. |
| `HttpGet` | `HttpGet[url, opts]` | HTTP GET request (http/https) |
| `HttpHead` | `HttpHead[url, opts]` | HTTP HEAD request (http/https) |
| `HttpOptions` | `HttpOptions[url, opts]` | HTTP OPTIONS request (http/https) |
| `HttpPatch` | `HttpPatch[url, body, opts]` | HTTP PATCH request (http/https) |
| `HttpPost` | `HttpPost[url, body, opts]` | HTTP POST request (http/https) |
| `HttpPut` | `HttpPut[url, body, opts]` | HTTP PUT request (http/https) |
| `HttpRequest` | `HttpRequest[options]` | Generic HTTP request via options object |
| `HttpRetry` | `HttpRetry[request, opts?]` | Retry a generic HTTP request map until success or attempts exhausted. |
| `HttpServe` | `HttpServe[handler, opts]` | Start an HTTP server and handle requests with a function |
| `HttpServeRoutes` | `HttpServeRoutes[routes, opts]` | Start an HTTP server with a routes table |
| `HttpServeTls` | `HttpServeTls[handler, opts]` | Start an HTTPS server with TLS cert/key |
| `HttpServerAddr` | `HttpServerAddr[server]` | Get bound address for a server id |
| `HttpServerStop` | `HttpServerStop[server]` | Stop a running HTTP server by id |
| `HttpStreamClose` | `HttpStreamClose[streamId]` | Close a streaming HTTP handle. |
| `HttpStreamRead` | `HttpStreamRead[streamId, maxBytes?]` | Read a chunk from a streaming HTTP handle; returns <\|"chunk"->bytes, "done"->bool\|>. |
| `HttpStreamRequest` | `HttpStreamRequest[method, url, opts?]` | Start a streaming HTTP request; returns headers, status, and a stream handle. |
| `NetChain` | `NetChain[layers, opts]` | Construct a sequential network from layers |
| `NetInitialize` | `NetInitialize[net, opts]` | Initialize network parameters |
| `PathMatch` | `PathMatch[pattern, path]` | Match a path pattern like /users/:id against a path |
| `RespondBytes` | `RespondBytes[bytes, opts]` | Build a binary response for HttpServe |
| `RespondFile` | `RespondFile[path, opts]` | Build a file response for HttpServe |
| `RespondHtml` | `RespondHtml[html, opts]` | Build an HTML response for HttpServe |
| `RespondNoContent` | `RespondNoContent[opts]` | Build an empty 204/205/304 response |
| `RespondRedirect` | `RespondRedirect[location, opts]` | Build a redirect response (Location header) |
| `RespondText` | `RespondText[text, opts]` | Build a text response for HttpServe |

## `HttpDownloadCached`

- Usage: `HttpDownloadCached[url, path, opts?]`
- Summary: Download a URL to a file with ETag/TTL caching; returns path and bytes written.
- Examples:
  - `HttpDownloadCached["https://httpbin.org/get", "/tmp/get.json", <|"TtlMs"->60000|>]  ==> <|"path"->..., "from_cache"->True|>`

## `HttpGet`

- Usage: `HttpGet[url, opts]`
- Summary: HTTP GET request (http/https)
- Examples:
  - `HttpGet["https://example.com"]  ==> <|"Status"->200, ...|>`

## `HttpRetry`

- Usage: `HttpRetry[request, opts?]`
- Summary: Retry a generic HTTP request map until success or attempts exhausted.
- Examples:
  - `HttpRetry[<|"Method"->"GET", "Url"->"https://example.com"|>, <|"Attempts"->3, "BackoffMs"->200|>]  ==> <|"status"->200,...|>`

## `HttpServe`

- Usage: `HttpServe[handler, opts]`
- Summary: Start an HTTP server and handle requests with a function
- Examples:
  - `srv := HttpServe[(req) => RespondText["ok"], <|"Port"->0|>]`
  - `HttpServerAddr[srv]  ==> "127.0.0.1:PORT"`
  - `HttpServerStop[srv]`

## `HttpStreamClose`

- Usage: `HttpStreamClose[streamId]`
- Summary: Close a streaming HTTP handle.
- Examples:
  - `HttpStreamClose[handle]  ==> True`

## `HttpStreamRead`

- Usage: `HttpStreamRead[streamId, maxBytes?]`
- Summary: Read a chunk from a streaming HTTP handle; returns <|"chunk"->bytes, "done"->bool|>.
- Examples:
  - `HttpStreamRead[handle, 16384]  ==> <|"chunk"->{...}, "done"->False|>`

## `HttpStreamRequest`

- Usage: `HttpStreamRequest[method, url, opts?]`
- Summary: Start a streaming HTTP request; returns headers, status, and a stream handle.
- Examples:
  - `r := HttpStreamRequest["GET", "https://example.com/large.bin"]`
  - `While[! Part[r2, "done"], r2 := HttpStreamRead[Part[r, "stream"], 8192]]`
  - `HttpStreamClose[Part[r, "stream"]]`
