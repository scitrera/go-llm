# Scitrera LLM Protocol for Go

Provider-neutral LLM request, response, streaming, and protocol translation
contracts for Go applications.

The package deliberately has no gateway, server, provider SDK, telemetry, or
agent-runtime dependency. It is suitable for gateways, agents, evaluation
tools, and direct provider clients.

Generation codecs cover OpenAI Chat Completions, OpenAI Responses, Anthropic
Messages, Gemini GenerateContent, and Amazon Bedrock Converse. A separate
embedding-codec registry covers OpenAI Embeddings, Gemini EmbedContent, and the
Amazon Titan and Cohere Bedrock InvokeModel families. Gemini SSE and
decoded Bedrock event-stream events use the same typed streaming contract as
the other formats. Codecs reject known lossy conversions by default and return
bounded, content-free diagnostics when permissive conversion is explicitly
requested. Normalized input-token usage always includes cached-input and
cache-creation components even when a provider reports those separately.

Embedding results use an explicit dense tensor: row-major `Values` plus a
`Shape` whose rank is exactly one or two. Batch cardinality lives separately in
`EmbeddingResponse.Data`, so a rank-2 embedding cannot be mistaken for two
rank-1 results. Codecs validate shape products and fail closed when a target
wire format can represent only rank 1. OpenAI float and base64 embeddings,
Gemini shaped embeddings, and float Titan/Cohere responses are typed.

Structured outputs and strict tools are typed across OpenAI, Anthropic, and
Bedrock. Gemini request controls include penalties, seed, candidate count,
modalities, logprobs, and thinking configuration. OpenAI Chat and Gemini token
log probabilities share buffered and streaming types.

The Bedrock codec does not own AWS credentials, SigV4, regional endpoint
selection, or binary framing. Its streaming API consumes and produces decoded
event type plus JSON payload pairs. The `client` package supplies a bounded,
CRC-validating binary event-stream reader and standard-library SigV4 signer
over a caller-owned credential provider; gateways may retain their own
transport for policy and raw-byte capture.

Licensed under the Apache License, Version 2.0.

Run correctness tests and the allocation/throughput microbenchmarks with:

```sh
go test ./...
go test -run '^$' -bench . -benchmem ./codec
```
