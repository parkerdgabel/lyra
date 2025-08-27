# NET

| Function | Usage | Summary |
|---|---|---|
| `AuthJwt` | `AuthJwt[opts, handler]` | JWT auth middleware; verifies Bearer token and injects claims. |
| `AuthJwtApply` | `AuthJwtApply[opts, handler, req]` | Verify JWT on request and call handler or return 401. |
| `CookiesHeader` | `CookiesHeader[cookies]` | Build a Cookie header string from an assoc |
| `Cors` | `Cors[opts, handler]` | Build CORS middleware (wraps handler). |
| `CorsApply` | `CorsApply[opts, handler, req]` | Apply CORS preflight/headers using options and handler. |
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

## `AuthJwtApply`

- Usage: `AuthJwtApply[opts, handler, req]`
- Summary: Verify JWT on request and call handler or return 401.
- Examples:
  - `AuthJwtApply[<|"Secret"->"s"|>, (r)=>RespondText["ok"], <||>]  ==> <|"status"->401, ...|>`
  - `tok := JwtSign[<|"sub"->"u1"|>, "s", <|"Alg"->"HS256"|>]`
  - `req := <|"headers"-><|"Authorization"->StringJoin[{"Bearer ", tok}]|>|>`
  - `AuthJwtApply[<|"Secret"->"s"|>, (r)=>RespondText["ok"], req]  ==> <|"status"->200, ...|>`

## `Cors`

- Usage: `Cors[opts, handler]`
- Summary: Build CORS middleware (wraps handler).
- Examples:
  - `srv := HttpServe[Cors[<|"AllowOrigin"->"*"|>, (req)=>RespondText["ok"]], <|"Port"->0|>]`
  - `HttpServerStop[srv]`

## `CorsApply`

- Usage: `CorsApply[opts, handler, req]`
- Summary: Apply CORS preflight/headers using options and handler.
- Examples:
  - `CorsApply[<|"AllowOrigin"->"*", "AllowMethods"->"GET"|>, (r)=>RespondText["ok"], <|"method"->"OPTIONS", "headers"-><||>|>]  ==> <|"status"->204, ...|>`

## `HttpDownloadCached`

- Usage: `HttpDownloadCached[url, path, opts?]`
- Summary: Download a URL to a file with ETag/TTL caching; returns path and bytes written.
- Examples:
  - `HttpDownloadCached["https://httpbin.org/get", "/tmp/get.json", <|"TtlMs"->60000|>]  ==> <|"path"->..., "from_cache"->True|>`

## `HttpGet`

- Usage: `HttpGet[url, opts]`
- Summary: HTTP GET request (http/https)
- Tags: net, http
- Examples:
  - `HttpGet["https://example.com"]`

## `HttpPost`

- Usage: `HttpPost[url, body, opts]`
- Summary: HTTP POST request (http/https)
- Tags: net, http
- Examples:
  - `HttpPost["https://example.com/api", <|"json"-><|"x"->1|>|>]`

## `HttpRequest`

- Usage: `HttpRequest[options]`
- Summary: Generic HTTP request via options object
- Tags: net, http
- Examples:
  - `HttpRequest[<|"Method"->"GET", "Url"->"https://example.com"|>]`

## `HttpRetry`

- Usage: `HttpRetry[request, opts?]`
- Summary: Retry a generic HTTP request map until success or attempts exhausted.
- Examples:
  - `HttpRetry[<|"Method"->"GET", "Url"->"https://example.com"|>, <|"Attempts"->3, "BackoffMs"->200|>]  ==> <|"status"->200,...|>`

## `HttpServe`

- Usage: `HttpServe[handler, opts]`
- Summary: Start an HTTP server and handle requests with a function
- Tags: net, http, server
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

## `PathMatch`

- Usage: `PathMatch[pattern, path]`
- Summary: Match a path pattern like /users/:id against a path
- Tags: http, routing
- Examples:
  - `PathMatch["/users/:id", "/users/42"]  ==> <|id->"42"|>`

## `RespondText`

- Usage: `RespondText[text, opts]`
- Summary: Build a text response for HttpServe
- Tags: http, server
- Examples:
  - `RespondText["ok", <|"Status"->200|>]`
