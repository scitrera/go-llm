// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package llmclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"path"
	"strings"
	"unicode"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

const (
	maxBodyDefaultFields     = 64
	maxBodyDefaultNameBytes  = 128
	maxBodyDefaultValueBytes = 1 << 20
	maxBodyDefaultsBytes     = 4 << 20
)

var protocolOwnedBodyFields = map[string]struct{}{
	"autotruncate":       {},
	"contents":           {},
	"content":            {},
	"dimensions":         {},
	"embedcontentconfig": {},
	"embedding_types":    {},
	"embeddingtypes":     {},
	"extra_body":         {},
	"input":              {},
	"input_type":         {},
	"inputtext":          {},
	"messages":           {},
	"model":              {},
	"normalize":          {},
	"output_dimension":   {},
	"prompt":             {},
	"requests":           {},
	"stream":             {},
	"stream_options":     {},
	"system":             {},
	"systeminstruction":  {},
	"tool_choice":        {},
	"toolconfig":         {},
	"tools":              {},
	"texts":              {},
}

type Authenticator interface {
	Apply(context.Context, *http.Request) error
}

type AuthFunc func(context.Context, *http.Request) error

func (f AuthFunc) Apply(ctx context.Context, request *http.Request) error {
	return f(ctx, request)
}

func BearerToken(token string) Authenticator {
	return AuthFunc(func(_ context.Context, request *http.Request) error {
		if token != "" {
			request.Header.Set("Authorization", "Bearer "+token)
		}
		return nil
	})
}

func APIKeyHeader(name, value string) Authenticator {
	return AuthFunc(func(_ context.Context, request *http.Request) error {
		if err := validateHeader(name, value); err != nil {
			return err
		}
		if value != "" {
			request.Header.Set(name, value)
		}
		return nil
	})
}

type Backend struct {
	Name         string
	Format       llmprotocol.Format
	BaseURL      string
	Path         string
	StreamPath   string
	BatchPath    string
	Model        string
	Headers      map[string]string
	BodyDefaults map[string]json.RawMessage
	Auth         Authenticator
}

func (b Backend) Validate() error {
	if b.Format == "" {
		return fmt.Errorf("llm client backend format is required")
	}
	parsed, err := url.ParseRequestURI(b.BaseURL)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return fmt.Errorf("llm client backend base URL must be an absolute HTTP(S) URL")
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return fmt.Errorf("llm client backend base URL must not contain credentials, query, or fragment")
	}
	if b.Path != "" && !strings.HasPrefix(b.Path, "/") {
		return fmt.Errorf("llm client backend path must be absolute")
	}
	if b.StreamPath != "" && !strings.HasPrefix(b.StreamPath, "/") {
		return fmt.Errorf("llm client backend stream path must be absolute")
	}
	if b.BatchPath != "" && !strings.HasPrefix(b.BatchPath, "/") {
		return fmt.Errorf("llm client backend batch path must be absolute")
	}
	for name, value := range b.Headers {
		if isReservedHeader(name) {
			return fmt.Errorf("llm client backend header %q is transport-owned", name)
		}
		if err := validateHeader(name, value); err != nil {
			return err
		}
	}
	if err := validateBodyDefaults(b.BodyDefaults); err != nil {
		return err
	}
	return nil
}

func (b Backend) endpoint(model string, stream bool) (*url.URL, error) {
	return b.endpointMode(model, stream, false)
}

func (b Backend) embeddingEndpoint(model string, batch bool) (*url.URL, error) {
	return b.endpointMode(model, false, batch)
}

func (b Backend) endpointMode(model string, stream, batch bool) (*url.URL, error) {
	if err := b.Validate(); err != nil {
		return nil, err
	}
	base, _ := url.Parse(b.BaseURL)
	endpointPath := b.Path
	if stream && b.StreamPath != "" {
		endpointPath = b.StreamPath
	}
	if batch && b.BatchPath != "" {
		endpointPath = b.BatchPath
	}
	if endpointPath == "" {
		endpointPath = defaultEndpointPath(b.Format, base.Path, stream, batch)
	}
	if strings.Contains(endpointPath, "{model}") && strings.TrimSpace(model) == "" {
		return nil, fmt.Errorf("llm client backend model is required by endpoint path")
	}
	if strings.ContainsAny(model, "\x00\r\n?#") {
		return nil, fmt.Errorf("llm client backend model is invalid")
	}
	escapedEndpointPath := strings.ReplaceAll(endpointPath, "{model}", awsURIEncode(model, false))
	decodedEndpointPath := strings.ReplaceAll(endpointPath, "{model}", model)
	basePath := base.Path
	baseEscapedPath := base.EscapedPath()
	base.Path = joinURLPath(basePath, decodedEndpointPath)
	base.RawPath = joinURLPath(baseEscapedPath, escapedEndpointPath)
	if !strings.Contains(base.RawPath, "%") {
		base.RawPath = ""
	}
	if stream && b.Format == llmprotocol.FormatGemini {
		query := base.Query()
		query.Set("alt", "sse")
		base.RawQuery = query.Encode()
	}
	return base, nil
}

func defaultEndpointPath(format llmprotocol.Format, basePath string, stream, batch bool) string {
	leaf := ""
	switch format {
	case llmprotocol.FormatOpenAIChat:
		leaf = "chat/completions"
	case llmprotocol.FormatOpenAIResponses:
		leaf = "responses"
	case llmprotocol.FormatAnthropic:
		leaf = "messages"
	case llmprotocol.FormatGemini:
		if stream {
			return "/models/{model}:streamGenerateContent"
		}
		return "/models/{model}:generateContent"
	case llmprotocol.FormatBedrock:
		if stream {
			return "/model/{model}/converse-stream"
		}
		return "/model/{model}/converse"
	case llmprotocol.FormatOpenAIEmbeddings:
		leaf = "embeddings"
	case llmprotocol.FormatGeminiEmbeddings:
		if batch {
			return "/models/{model}:batchEmbedContents"
		}
		return "/models/{model}:embedContent"
	case llmprotocol.FormatBedrockTitanEmbeddings, llmprotocol.FormatBedrockCohereEmbeddings:
		return "/model/{model}/invoke"
	default:
		leaf = string(format)
	}
	if strings.HasSuffix(strings.TrimRight(basePath, "/"), "/v1") {
		return "/" + leaf
	}
	return "/v1/" + leaf
}

func joinURLPath(basePath, endpointPath string) string {
	if basePath == "" || basePath == "/" {
		return path.Clean("/" + strings.TrimLeft(endpointPath, "/"))
	}
	return path.Clean(strings.TrimRight(basePath, "/") + "/" + strings.TrimLeft(endpointPath, "/"))
}

// MergeBodyDefaults shallow-merges bounded provider-specific defaults into an
// encoded request object. Fields already present in body always win. This
// helper is exported so gateways that retain ownership of raw HTTP proxying can
// share the same validation and merge semantics as Client.
func MergeBodyDefaults(body json.RawMessage, defaults map[string]json.RawMessage) (json.RawMessage, error) {
	if len(defaults) == 0 {
		return body, nil
	}
	if err := validateBodyDefaults(defaults); err != nil {
		return nil, err
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil || object == nil {
		return nil, fmt.Errorf("llm client encoded request is not a JSON object")
	}
	for name, value := range defaults {
		if _, exists := object[name]; !exists {
			object[name] = append(json.RawMessage(nil), value...)
		}
	}
	return json.Marshal(object)
}

// validateBodyDefaults keeps operator-supplied extension defaults bounded.
// Defaults remain deliberately protocol-neutral: callers may use provider
// fields such as service_tier or chat_template_kwargs, while the request-wins
// merge in MergeBodyDefaults prevents them from replacing caller fields.
func validateBodyDefaults(defaults map[string]json.RawMessage) error {
	if len(defaults) > maxBodyDefaultFields {
		return fmt.Errorf("llm client backend body defaults have %d fields; maximum is %d", len(defaults), maxBodyDefaultFields)
	}
	total := 0
	for name, value := range defaults {
		if strings.TrimSpace(name) == "" || name != strings.TrimSpace(name) || len(name) > maxBodyDefaultNameBytes {
			return fmt.Errorf("llm client backend body default %q has an invalid name", name)
		}
		for _, character := range name {
			if unicode.IsControl(character) {
				return fmt.Errorf("llm client backend body default %q has an invalid name", name)
			}
		}
		if _, protected := protocolOwnedBodyFields[strings.ToLower(name)]; protected {
			return fmt.Errorf("llm client backend body default %q is protocol-owned", name)
		}
		if len(value) > maxBodyDefaultValueBytes || !json.Valid(value) {
			return fmt.Errorf("llm client backend body default %q is invalid or exceeds %d bytes", name, maxBodyDefaultValueBytes)
		}
		total += len(name) + len(value)
		if total > maxBodyDefaultsBytes {
			return fmt.Errorf("llm client backend body defaults exceed %d bytes", maxBodyDefaultsBytes)
		}
	}
	return nil
}

func validateHeader(name, value string) error {
	if strings.TrimSpace(name) == "" || strings.ContainsAny(name, "\r\n:") || strings.ContainsAny(value, "\r\n") {
		return fmt.Errorf("llm client header is invalid")
	}
	return nil
}

func isReservedHeader(name string) bool {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "host", "content-length", "connection", "transfer-encoding", "content-type", "accept", "authorization", "x-api-key":
		return true
	default:
		return false
	}
}

func cloneBody(body []byte) *bytes.Reader { return bytes.NewReader(body) }
