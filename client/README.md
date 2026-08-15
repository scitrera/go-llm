# Scitrera LLM Client for Go

A provider-neutral Go client for OpenAI Chat Completions, OpenAI Responses,
Anthropic Messages, Gemini GenerateContent, and Amazon Bedrock Converse
backends, plus OpenAI, Gemini, Amazon Titan, and Cohere embedding backends. The
client consumes and returns types from
[`github.com/scitrera/go-llm/protocol`](../protocol).

The package has no gateway or agent-runtime dependency. Callers provide backend
configuration, authentication, and an `http.Client`; gateway concerns such as
tenant routing, circuit breakers, shared retry budgets, persistence, and model
lifecycle remain outside this module.

Retries are disabled by default. Streaming uses a capacity-one event channel to
preserve downstream backpressure and supports separate first-event and idle
liveness deadlines. Configured retries apply to buffered calls; streams are
never replayed by the library after an attempt begins.

Default Gemini endpoints select `generateContent` or
`streamGenerateContent?alt=sse`; embedding calls select `embedContent` or
`batchEmbedContents` from input cardinality. Default Bedrock endpoints select `converse` or
`converse-stream`, safely escaping the model as one URI label. Bedrock streams
use a bounded AWS event-stream decoder with prelude/message CRC validation.
`SigV4Authenticator` signs the finalized URL, headers, and body using
standard-library cryptography. Its `AWSCredentialsProvider` remains
caller-owned, so applications can plug in static material, an AWS SDK chain, or
workload identity without this module owning discovery, refresh, or storage.
`Backend.Path`, `Backend.StreamPath`, and `Backend.BatchPath` may override
defaults and use a `{model}` placeholder. Titan and Cohere embeddings use the
Bedrock InvokeModel endpoint and the same caller-owned SigV4 authenticator.

`Client.Embed` returns the rank-aware embedding contract from
the `protocol` package; one response item can contain a validated rank-1 vector or
rank-2 matrix without conflating that shape with request batch cardinality.

`Backend.BodyDefaults` supplies bounded, top-level provider request defaults
(the analogue of an SDK `extra_body`). A caller-provided field always wins;
defaults are validated before any network request and are limited to 64 fields,
1 MiB per value, and 4 MiB total. Protocol-owned fields such as `model`,
`messages`, `input`, `tools`, and `stream` are rejected. Gateways which retain
ownership of raw proxying can call `MergeBodyDefaults` to reuse the exact same
bounded, request-wins semantics without adopting this client's retry policy.

Licensed under the Apache License, Version 2.0.

Run correctness tests and the request-default microbenchmark with:

```sh
go test ./...
go test -run '^$' -bench . -benchmem ./...
```
