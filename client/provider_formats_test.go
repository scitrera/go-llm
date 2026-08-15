// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package llmclient

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

func TestGeminiSSEClient(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.EscapedPath() != "/v1beta/models/gemini-3:streamGenerateContent" || r.URL.Query().Get("alt") != "sse" || r.Header.Get("X-Goog-Api-Key") != "secret" {
			t.Errorf("request URL=%s key=%q", r.URL.String(), r.Header.Get("X-Goog-Api-Key"))
		}
		body, _ := io.ReadAll(r.Body)
		if !bytes.Contains(body, []byte(`"contents"`)) || bytes.Contains(body, []byte(`"model"`)) {
			t.Errorf("request body = %s", body)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"responseId\":\"resp_g\",\"modelVersion\":\"gemini-3\",\"candidates\":[{\"index\":0,\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"hello\"}]}}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"candidates\":[{\"index\":0,\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":4,\"candidatesTokenCount\":1,\"totalTokenCount\":5}}\n\n")
	}))
	defer server.Close()

	client := New(Options{})
	stream, err := client.Stream(context.Background(), Backend{
		Name: "gemini", Format: llmprotocol.FormatGemini, BaseURL: server.URL + "/v1beta", Model: "gemini-3",
		Auth: APIKeyHeader("X-Goog-Api-Key", "secret"),
	}, llmprotocol.Request{Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.ContentBlock{llmprotocol.Text("hello")}}}})
	if err != nil {
		t.Fatal(err)
	}
	response, err := stream.Collect(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if response.ID != "resp_g" || response.Model != "gemini-3" || len(response.Outputs) != 1 || response.Outputs[0].Content[0].Text != "hello" || response.Usage.TotalTokens == nil || *response.Usage.TotalTokens != 5 {
		t.Fatalf("response = %#v", response)
	}
}
