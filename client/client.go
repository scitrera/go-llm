// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

// Package llmclient provides provider-neutral buffered and streaming HTTP
// execution without importing gateway or agent-runtime concerns.
package llmclient

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"time"

	gollm "github.com/scitrera/go-llm"
	llmprotocol "github.com/scitrera/go-llm/protocol"
	"github.com/scitrera/go-llm/protocol/codec"
)

const (
	defaultMaxResponseBytes = 64 << 20
	// Built from the module version rather than a literal, which had already
	// drifted to 0.1 while the module moved on. Still a constant: gollm.Version
	// is one too, and the root package exists precisely to hold it.
	defaultUserAgent = "scitrera-go-llm-client/" + gollm.Version
)

type RetryPolicy struct {
	MaxRetries    int
	InitialDelay  time.Duration
	MaxDelay      time.Duration
	RetryAfterCap time.Duration
	Sleep         func(context.Context, time.Duration) error
	Jitter        func(time.Duration) time.Duration
}

func (p RetryPolicy) effective() RetryPolicy {
	if p.InitialDelay <= 0 {
		p.InitialDelay = 100 * time.Millisecond
	}
	if p.MaxDelay <= 0 {
		p.MaxDelay = 2 * time.Second
	}
	if p.RetryAfterCap <= 0 {
		p.RetryAfterCap = 10 * time.Second
	}
	if p.Sleep == nil {
		p.Sleep = sleepContext
	}
	if p.Jitter == nil {
		p.Jitter = func(max time.Duration) time.Duration {
			if max <= 0 {
				return 0
			}
			return time.Duration(rand.Int64N(int64(max) + 1))
		}
	}
	return p
}

type Options struct {
	HTTPClient         *http.Client
	Registry           *codec.Registry
	EmbeddingRegistry  *codec.EmbeddingRegistry
	Policy             llmprotocol.Policy
	MaxResponseBytes   int64
	FirstEventTimeout  time.Duration
	StreamIdleTimeout  time.Duration
	MaxStreamLineBytes int
	IncludeStreamUsage *bool
	UserAgent          string
	Retry              RetryPolicy
}

type Client struct {
	httpClient         *http.Client
	streamClient       *http.Client
	registry           *codec.Registry
	embeddingRegistry  *codec.EmbeddingRegistry
	policy             llmprotocol.Policy
	maxResponseBytes   int64
	firstEventTimeout  time.Duration
	streamIdleTimeout  time.Duration
	maxStreamLineBytes int
	includeStreamUsage bool
	userAgent          string
	retry              RetryPolicy
}

func New(options Options) *Client {
	httpClient := options.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 60 * time.Second}
	}
	clone := *httpClient
	clone.CheckRedirect = func(_ *http.Request, _ []*http.Request) error { return http.ErrUseLastResponse }
	streamClone := clone
	streamClone.Timeout = 0
	registry := options.Registry
	if registry == nil {
		registry = codec.NewDefaultRegistry()
	}
	embeddingRegistry := options.EmbeddingRegistry
	if embeddingRegistry == nil {
		embeddingRegistry = codec.NewDefaultEmbeddingRegistry()
	}
	maxResponseBytes := options.MaxResponseBytes
	if maxResponseBytes <= 0 {
		maxResponseBytes = defaultMaxResponseBytes
	}
	maxStreamLineBytes := options.MaxStreamLineBytes
	if maxStreamLineBytes <= 0 {
		maxStreamLineBytes = 8 << 20
	}
	userAgent := options.UserAgent
	if userAgent == "" {
		userAgent = defaultUserAgent
	}
	includeStreamUsage := true
	if options.IncludeStreamUsage != nil {
		includeStreamUsage = *options.IncludeStreamUsage
	}
	return &Client{
		httpClient: &clone, streamClient: &streamClone, registry: registry, embeddingRegistry: embeddingRegistry, policy: options.Policy.Effective(),
		maxResponseBytes: maxResponseBytes, firstEventTimeout: options.FirstEventTimeout,
		streamIdleTimeout: options.StreamIdleTimeout, maxStreamLineBytes: maxStreamLineBytes,
		includeStreamUsage: includeStreamUsage, userAgent: userAgent, retry: options.Retry.effective(),
	}
}

type CallMetadata struct {
	BackendName         string
	Format              llmprotocol.Format
	Endpoint            string
	StatusCode          int
	Attempts            int
	StartedAt           time.Time
	Duration            time.Duration
	ResponseHeaders     http.Header
	RequestDiagnostics  []llmprotocol.Diagnostic
	ResponseDiagnostics []llmprotocol.Diagnostic
}

type CallResult struct {
	Response llmprotocol.Response
	Metadata CallMetadata
}

type EmbeddingCallResult struct {
	Response llmprotocol.EmbeddingResponse
	Metadata CallMetadata
}

// Embed executes one buffered embeddings request. Batch cardinality is derived
// from request.Inputs; each response Data item may independently contain a
// rank-one or rank-two tensor.
func (c *Client) Embed(ctx context.Context, backend Backend, request llmprotocol.EmbeddingRequest) (EmbeddingCallResult, error) {
	started := time.Now()
	metadata := CallMetadata{BackendName: backend.Name, Format: backend.Format, StartedAt: started}
	effectiveModel := request.Model
	if backend.Model != "" {
		effectiveModel = backend.Model
	}
	endpoint, err := backend.embeddingEndpoint(effectiveModel, len(request.Inputs) > 1)
	if err != nil {
		return EmbeddingCallResult{Metadata: metadata}, &Error{Kind: ErrorInvalidConfig, Message: err.Error(), Cause: err}
	}
	metadata.Endpoint = endpoint.Redacted()
	formatCodec, err := c.embeddingRegistry.Codec(backend.Format)
	if err != nil {
		return EmbeddingCallResult{Metadata: metadata}, &Error{Kind: ErrorInvalidConfig, Message: err.Error(), Cause: err}
	}
	if backend.Model != "" && backend.Model != request.Model {
		request.Model = backend.Model
		request.ClearPreservation()
	}
	encoded, err := formatCodec.EncodeEmbeddingRequest(request, c.policy)
	if err != nil {
		return EmbeddingCallResult{Metadata: metadata}, &Error{Kind: ErrorEncode, Message: err.Error(), Cause: err}
	}
	metadata.RequestDiagnostics = encoded.Diagnostics
	body, err := MergeBodyDefaults(encoded.Body, backend.BodyDefaults)
	if err != nil {
		return EmbeddingCallResult{Metadata: metadata}, &Error{Kind: ErrorEncode, Message: err.Error(), Cause: err}
	}
	maxAttempts := 1 + c.retry.MaxRetries
	if maxAttempts < 1 {
		maxAttempts = 1
	}
	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		metadata.Attempts = attempt
		response, doErr := c.do(ctx, backend, endpoint.String(), body, false)
		if doErr != nil {
			lastErr = doErr
			if attempt == maxAttempts || !retryableError(ctx, doErr) {
				metadata.Duration = time.Since(started)
				return EmbeddingCallResult{Metadata: metadata}, doErr
			}
			if err := c.waitRetry(ctx, attempt, ""); err != nil {
				metadata.Duration = time.Since(started)
				return EmbeddingCallResult{Metadata: metadata}, classifyContextError(err)
			}
			continue
		}
		metadata.StatusCode = response.StatusCode
		metadata.ResponseHeaders = response.Header.Clone()
		responseBody, readErr := readBounded(response.Body, c.maxResponseBytes)
		_ = response.Body.Close()
		if readErr != nil {
			metadata.Duration = time.Since(started)
			return EmbeddingCallResult{Metadata: metadata}, readErr
		}
		if response.StatusCode < 200 || response.StatusCode >= 300 {
			httpErr := decodeHTTPError(response.StatusCode, responseBody)
			lastErr = httpErr
			if attempt == maxAttempts || !httpErr.Retryable {
				metadata.Duration = time.Since(started)
				return EmbeddingCallResult{Metadata: metadata}, httpErr
			}
			if err := c.waitRetry(ctx, attempt, response.Header.Get("Retry-After")); err != nil {
				metadata.Duration = time.Since(started)
				return EmbeddingCallResult{Metadata: metadata}, classifyContextError(err)
			}
			continue
		}
		decoded, decodeErr := formatCodec.DecodeEmbeddingResponse(responseBody, c.policy)
		metadata.Duration = time.Since(started)
		if decodeErr != nil {
			return EmbeddingCallResult{Metadata: metadata}, &Error{Kind: ErrorDecode, Message: decodeErr.Error(), Cause: decodeErr}
		}
		metadata.ResponseDiagnostics = decoded.Diagnostics
		return EmbeddingCallResult{Response: decoded.Response, Metadata: metadata}, nil
	}
	metadata.Duration = time.Since(started)
	return EmbeddingCallResult{Metadata: metadata}, lastErr
}

func (c *Client) Call(ctx context.Context, backend Backend, request llmprotocol.Request) (CallResult, error) {
	started := time.Now()
	metadata := CallMetadata{BackendName: backend.Name, Format: backend.Format, StartedAt: started}
	effectiveModel := request.Model
	if backend.Model != "" {
		effectiveModel = backend.Model
	}
	endpoint, err := backend.endpoint(effectiveModel, false)
	if err != nil {
		return CallResult{Metadata: metadata}, &Error{Kind: ErrorInvalidConfig, Message: err.Error(), Cause: err}
	}
	metadata.Endpoint = endpoint.Redacted()
	formatCodec, err := c.registry.Codec(backend.Format)
	if err != nil {
		return CallResult{Metadata: metadata}, &Error{Kind: ErrorInvalidConfig, Message: err.Error(), Cause: err}
	}
	request.Stream = false
	if backend.Model != "" && backend.Model != request.Model {
		request.Model = backend.Model
		request.ClearPreservation()
	}
	encoded, err := formatCodec.EncodeRequest(request, c.policy)
	if err != nil {
		return CallResult{Metadata: metadata}, &Error{Kind: ErrorEncode, Message: err.Error(), Cause: err}
	}
	metadata.RequestDiagnostics = encoded.Diagnostics
	body, err := MergeBodyDefaults(encoded.Body, backend.BodyDefaults)
	if err != nil {
		return CallResult{Metadata: metadata}, &Error{Kind: ErrorEncode, Message: err.Error(), Cause: err}
	}
	maxAttempts := 1 + c.retry.MaxRetries
	if maxAttempts < 1 {
		maxAttempts = 1
	}
	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		metadata.Attempts = attempt
		response, doErr := c.do(ctx, backend, endpoint.String(), body, false)
		if doErr != nil {
			lastErr = doErr
			if attempt == maxAttempts || !retryableError(ctx, doErr) {
				metadata.Duration = time.Since(started)
				return CallResult{Metadata: metadata}, doErr
			}
			if err := c.waitRetry(ctx, attempt, ""); err != nil {
				metadata.Duration = time.Since(started)
				return CallResult{Metadata: metadata}, classifyContextError(err)
			}
			continue
		}
		metadata.StatusCode = response.StatusCode
		metadata.ResponseHeaders = response.Header.Clone()
		responseBody, readErr := readBounded(response.Body, c.maxResponseBytes)
		_ = response.Body.Close()
		if readErr != nil {
			metadata.Duration = time.Since(started)
			return CallResult{Metadata: metadata}, readErr
		}
		if response.StatusCode < 200 || response.StatusCode >= 300 {
			httpErr := decodeHTTPError(response.StatusCode, responseBody)
			lastErr = httpErr
			if attempt == maxAttempts || !httpErr.Retryable {
				metadata.Duration = time.Since(started)
				return CallResult{Metadata: metadata}, httpErr
			}
			if err := c.waitRetry(ctx, attempt, response.Header.Get("Retry-After")); err != nil {
				metadata.Duration = time.Since(started)
				return CallResult{Metadata: metadata}, classifyContextError(err)
			}
			continue
		}
		decoded, decodeErr := formatCodec.DecodeResponse(responseBody, c.policy)
		metadata.Duration = time.Since(started)
		if decodeErr != nil {
			return CallResult{Metadata: metadata}, &Error{Kind: ErrorDecode, Message: decodeErr.Error(), Cause: decodeErr}
		}
		metadata.ResponseDiagnostics = decoded.Diagnostics
		return CallResult{Response: decoded.Response, Metadata: metadata}, nil
	}
	metadata.Duration = time.Since(started)
	return CallResult{Metadata: metadata}, lastErr
}

func (c *Client) do(ctx context.Context, backend Backend, endpoint string, body []byte, stream bool) (*http.Response, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, cloneBody(body))
	if err != nil {
		return nil, &Error{Kind: ErrorInvalidConfig, Message: err.Error(), Cause: err}
	}
	request.Header.Set("Content-Type", "application/json")
	if stream {
		if backend.Format == llmprotocol.FormatBedrock {
			request.Header.Set("Accept", "application/vnd.amazon.eventstream")
		} else {
			request.Header.Set("Accept", "text/event-stream")
		}
	} else {
		request.Header.Set("Accept", "application/json")
	}
	request.Header.Set("User-Agent", c.userAgent)
	if backend.Format == llmprotocol.FormatAnthropic {
		request.Header.Set("Anthropic-Version", "2023-06-01")
	}
	for name, value := range backend.Headers {
		request.Header.Set(name, value)
	}
	if backend.Auth != nil {
		if err := backend.Auth.Apply(ctx, request); err != nil {
			return nil, &Error{Kind: ErrorInvalidConfig, Message: "backend authentication failed", Cause: err}
		}
	}
	client := c.httpClient
	if stream {
		client = c.streamClient
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, classifyTransport(ctx, err)
	}
	return response, nil
}

func (c *Client) waitRetry(ctx context.Context, attempt int, retryAfter string) error {
	delay := retryAfterDelay(retryAfter, c.retry.RetryAfterCap)
	if delay <= 0 {
		max := c.retry.InitialDelay
		for i := 1; i < attempt && max < c.retry.MaxDelay; i++ {
			max *= 2
			if max > c.retry.MaxDelay {
				max = c.retry.MaxDelay
			}
		}
		delay = c.retry.Jitter(max)
	}
	return c.retry.Sleep(ctx, delay)
}

func retryAfterDelay(value string, capDuration time.Duration) time.Duration {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0
	}
	if seconds, err := strconv.ParseFloat(value, 64); err == nil && seconds >= 0 {
		delay := time.Duration(seconds * float64(time.Second))
		if delay > capDuration {
			return capDuration
		}
		return delay
	}
	if at, err := http.ParseTime(value); err == nil {
		delay := time.Until(at)
		if delay < 0 {
			return 0
		}
		if delay > capDuration {
			return capDuration
		}
		return delay
	}
	return 0
}

func readBounded(reader io.Reader, limit int64) (json.RawMessage, error) {
	body, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil {
		return nil, &Error{Kind: ErrorTransport, Message: "read upstream response", Cause: err, Retryable: true}
	}
	if int64(len(body)) > limit {
		return nil, &Error{Kind: ErrorResponseLimit, Message: "upstream response exceeded configured limit"}
	}
	return body, nil
}

func retryableError(ctx context.Context, err error) bool {
	if ctx.Err() != nil {
		return false
	}
	var clientErr *Error
	return errors.As(err, &clientErr) && clientErr.Retryable
}

func classifyTransport(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return classifyContextError(ctx.Err())
	}
	return &Error{Kind: ErrorTransport, Message: "upstream transport failed", Cause: err, Retryable: true}
}

func classifyContextError(err error) error {
	if errors.Is(err, context.DeadlineExceeded) {
		return &Error{Kind: ErrorTimeout, Message: "request deadline exceeded", Cause: err, Retryable: true}
	}
	return &Error{Kind: ErrorCanceled, Message: "request canceled", Cause: err}
}

func sleepContext(ctx context.Context, duration time.Duration) error {
	if duration <= 0 {
		return nil
	}
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func protocolStreamError(value *llmprotocol.ProtocolError) error {
	if value == nil {
		return &Error{Kind: ErrorStream, Message: "upstream stream failed"}
	}
	return &Error{Kind: ErrorStream, Type: value.Type, Code: value.Code, Message: value.Message, Retryable: value.Retryable}
}
