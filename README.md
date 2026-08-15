# Scitrera LLM for Go

Provider-neutral LLM contracts and an HTTP client for Go, in one module:

| Package | Import path | What it is |
|---|---|---|
| [`protocol`](protocol) | `github.com/scitrera/go-llm/protocol` | Request, response, streaming and usage types, translation codecs for five provider wire formats, and an embedding-codec registry |
| [`protocol/codec`](protocol/codec) | `github.com/scitrera/go-llm/protocol/codec` | The codecs themselves: OpenAI Chat Completions and Responses, Anthropic Messages, Gemini GenerateContent, Amazon Bedrock Converse |
| [`client`](client) | `github.com/scitrera/go-llm/client` | Buffered and streaming execution over those types: backend configuration, retries, liveness deadlines, SigV4, Bedrock event-stream framing |

```bash
go get github.com/scitrera/go-llm@latest
```

## Versioning

`versions.yaml` is authoritative; `sync-versions` propagates the number to
`version.go`, and the release workflow refuses a tag that disagrees with it.
While below `v1`, a breaking exported-contract change increments the minor
version.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
