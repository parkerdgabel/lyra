Lyra Stdlib: Cloud Object Storage (Feature: cloud)

Enable with cargo feature:
- Run with `--features cloud` to compile S3/GCS/Azure support using the `object_store` crate.

URIs
- S3: `s3://bucket/key`
- GCS: `gs://bucket/key`
- Azure: `az://container/blob`

Basic operations
- Read: `ObjectStoreRead["s3://bucket/key"]`
- Write: `ObjectStoreWrite["gs://bucket/key", bytes]`
- Delete: `ObjectStoreDelete["az://container/blob"]`
- List: `ObjectStoreList["s3://bucket/prefix/"]`
- Head: `ObjectStoreHead["gs://bucket/key"]`
- Presign: `ObjectStorePresign["s3://bucket/key", <|method -> "GET", expiresSec -> 600|>]`

Options precedence for providers
- opts.providerOpts > environment > shared config/metadata
- S3 supported in `providerOpts`: `region`, `endpoint`, `accessKey`, `secretKey`, `sessionToken`

Examples
- `ObjectStoreRead["s3://my-bucket/data.bin", <|providerOpts -> <|region -> "us-west-2"|>|>]`
- `ObjectStoreWrite["s3://my-bucket/out", "hello", <|providerOpts -> <|endpoint -> "http://localhost:4566"|>|>]` (for localstack)
- `ObjectStorePresign["gs://my-bucket/file.txt", <|method -> "PUT", expiresSec -> 1200|>]`
- `ObjectStoreExists["file:///tmp/data.bin"]` -> True|False
- `ObjectStoreMove["file:///tmp/a", "file:///tmp/b"]`

HTTP retry examples
- `HTTPRetry["https://api.example.com", <|retries -> 3, retryOn -> {500,503}, backoff -> <|type -> "fixed", baseMs -> 50|>|>]`
- `HTTPRetry[NetworkRequest["https://api.example.com", "POST", <|"Content-Type" -> "application/json"|>, body], <|retries -> 2|>]`

Notes
- Credentials are resolved by `object_store` builders (env, config files, or explicit opts for S3).
- GCS/Azure current build uses environment/default credentials; explicit option overrides will be expanded in future iterations.
- Network calls require valid credentials and network access; tests are feature-gated and ignored by default.
