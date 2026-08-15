// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package llmclient

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

func TestCallAppliesAuthModelAndBodyDefaults(t *testing.T) {
	t.Parallel()
	var received map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q", request.URL.Path)
		}
		if request.Header.Get("Authorization") != "Bearer secret" {
			t.Errorf("authorization header = %q", request.Header.Get("Authorization"))
		}
		if request.Header.Get("X-Profile") != "local" {
			t.Errorf("profile header = %q", request.Header.Get("X-Profile"))
		}
		if err := json.NewDecoder(request.Body).Decode(&received); err != nil {
			t.Error(err)
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"chat_1","model":"served","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	}))
	defer server.Close()
	maxTokens := int64(16)
	client := New(Options{})
	result, err := client.Call(context.Background(), Backend{
		Name: "test", Format: llmprotocol.FormatOpenAIChat, BaseURL: server.URL,
		Model: "upstream-model", Auth: BearerToken("secret"), Headers: map[string]string{"X-Profile": "local"},
		BodyDefaults: map[string]json.RawMessage{"seed": json.RawMessage(`7`), "temperature": json.RawMessage(`1`)},
	}, llmprotocol.Request{
		Model: "virtual", Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.ContentBlock{llmprotocol.Text("hi")}}},
		Output: llmprotocol.Output{MaxTokens: &maxTokens}, Sampling: llmprotocol.Sampling{Temperature: floatPointer(0)},
	})
	if err != nil {
		t.Fatalf("Call() error = %v", err)
	}
	if received["model"] != "upstream-model" || received["seed"] != float64(7) || received["temperature"] != float64(0) {
		t.Fatalf("request body = %#v", received)
	}
	if result.Response.Model != "served" || len(result.Response.Outputs) != 1 || result.Response.Outputs[0].Content[0].Text != "hello" {
		t.Fatalf("response = %#v", result.Response)
	}
}

func TestCallRetriesOnlyWhenExplicitlyConfigured(t *testing.T) {
	t.Parallel()
	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			writer.WriteHeader(http.StatusBadGateway)
			_, _ = io.WriteString(writer, `{"error":{"message":"temporary","type":"server_error"}}`)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"chat_1","model":"served","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()
	client := New(Options{Retry: RetryPolicy{MaxRetries: 1, Sleep: func(context.Context, time.Duration) error { return nil }, Jitter: func(time.Duration) time.Duration { return 0 }}})
	_, err := client.Call(context.Background(), Backend{Format: llmprotocol.FormatOpenAIChat, BaseURL: server.URL}, textRequest("model", "hello"))
	if err != nil {
		t.Fatalf("Call() error = %v", err)
	}
	if calls.Load() != 2 {
		t.Fatalf("calls = %d, want 2", calls.Load())
	}
}

func TestStreamCollectsUsageAfterFinishReason(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		options, _ := body["stream_options"].(map[string]any)
		if options["include_usage"] != true {
			t.Errorf("stream_options = %#v", options)
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		flusher := writer.(http.Flusher)
		for _, frame := range []string{
			`data: {"id":"chat_1","model":"served","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}` + "\n\n",
			`data: {"id":"chat_1","model":"served","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}` + "\n\n",
			`data: {"id":"chat_1","model":"served","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n",
			`data: {"id":"chat_1","model":"served","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}` + "\n\n",
			"data: [DONE]\n\n",
		} {
			_, _ = io.WriteString(writer, frame)
			flusher.Flush()
		}
	}))
	defer server.Close()
	client := New(Options{FirstEventTimeout: time.Second, StreamIdleTimeout: time.Second})
	stream, err := client.Stream(context.Background(), Backend{Format: llmprotocol.FormatOpenAIChat, BaseURL: server.URL}, textRequest("model", "hello"))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	response, err := stream.Collect(context.Background())
	if err != nil {
		t.Fatalf("Collect() error = %v", err)
	}
	if len(response.Outputs) != 1 || response.Outputs[0].Content[0].Text != "hello" || response.Usage.TotalTokens == nil || *response.Usage.TotalTokens != 3 {
		t.Fatalf("response = %#v", response)
	}
}

func TestStreamWithoutUsageStopsAtChatFinishReason(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		flusher := writer.(http.Flusher)
		_, _ = io.WriteString(writer, `data: {"id":"chat_1","model":"served","choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}`+"\n\n")
		flusher.Flush()
		<-request.Context().Done()
	}))
	defer server.Close()
	includeUsage := false
	client := New(Options{
		FirstEventTimeout: time.Second, StreamIdleTimeout: time.Second,
		IncludeStreamUsage: &includeUsage,
	})
	stream, err := client.Stream(context.Background(), Backend{Format: llmprotocol.FormatOpenAIChat, BaseURL: server.URL}, textRequest("model", "hello"))
	if err != nil {
		t.Fatal(err)
	}
	response, err := stream.Collect(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Outputs) != 1 || response.Outputs[0].Content[0].Text != "done" || response.Usage.TotalTokens != nil {
		t.Fatalf("response = %#v", response)
	}
}

func TestStreamFirstEventTimeout(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		writer.(http.Flusher).Flush()
		<-request.Context().Done()
	}))
	defer server.Close()
	client := New(Options{FirstEventTimeout: 25 * time.Millisecond, StreamIdleTimeout: time.Second})
	stream, err := client.Stream(context.Background(), Backend{Format: llmprotocol.FormatOpenAIChat, BaseURL: server.URL}, textRequest("model", "hello"))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	_, err = stream.Next(context.Background())
	var clientErr *Error
	if !errors.As(err, &clientErr) || clientErr.Kind != ErrorTimeout || !errors.Is(clientErr, ErrFirstEventTimeout) {
		t.Fatalf("Next() error = %#v", err)
	}
}

func TestResponseLimitIsEnforced(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(writer, `{"id":"far-too-large"}`)
	}))
	defer server.Close()
	client := New(Options{MaxResponseBytes: 4})
	_, err := client.Call(context.Background(), Backend{Format: llmprotocol.FormatOpenAIChat, BaseURL: server.URL}, textRequest("model", "hello"))
	var clientErr *Error
	if !errors.As(err, &clientErr) || clientErr.Kind != ErrorResponseLimit {
		t.Fatalf("Call() error = %#v", err)
	}
}

func textRequest(model, text string) llmprotocol.Request {
	return llmprotocol.Request{Model: model, Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.ContentBlock{llmprotocol.Text(text)}}}}
}

func floatPointer(value float64) *float64 { return &value }
