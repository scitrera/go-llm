// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package llmclient

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

func TestEmbedOpenAI(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/embeddings" || request.Header.Get("Authorization") != "Bearer secret" {
			t.Errorf("path=%q authorization=%q", request.URL.Path, request.Header.Get("Authorization"))
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["model"] != "upstream-embedding" || body["input"] != "hello" {
			t.Errorf("body = %#v", body)
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"object":"list","model":"upstream-embedding","data":[{"object":"embedding","index":0,"embedding":[1,2,3]}],"usage":{"prompt_tokens":2,"total_tokens":2}}`)
	}))
	defer server.Close()
	client := New(Options{})
	result, err := client.Embed(context.Background(), Backend{
		Name: "openai", Format: llmprotocol.FormatOpenAIEmbeddings, BaseURL: server.URL,
		Model: "upstream-embedding", Auth: BearerToken("secret"),
	}, llmprotocol.EmbeddingRequest{
		Model:  "virtual-embedding",
		Inputs: []llmprotocol.EmbeddingInput{{Type: llmprotocol.EmbeddingInputText, Text: "hello"}},
	})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(result.Response.Data) != 1 || result.Response.Data[0].Embedding.Rank() != 1 || result.Response.Data[0].Embedding.Values[2] != 3 {
		t.Fatalf("response = %#v", result.Response)
	}
}

func TestEmbedGeminiBatchRetainsRankTwoTensor(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1beta/models/gemini-embedding:batchEmbedContents" {
			t.Errorf("path = %q", request.URL.Path)
		}
		var body struct {
			Requests []json.RawMessage `json:"requests"`
		}
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil || len(body.Requests) != 2 {
			t.Errorf("body = %#v, error = %v", body, err)
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"embeddings":[{"values":[1,2,3,4],"shape":[2,2]},{"values":[5,6]}],"usageMetadata":{"promptTokenCount":4}}`)
	}))
	defer server.Close()
	client := New(Options{})
	result, err := client.Embed(context.Background(), Backend{
		Format: llmprotocol.FormatGeminiEmbeddings, BaseURL: server.URL + "/v1beta", Model: "gemini-embedding",
	}, llmprotocol.EmbeddingRequest{Inputs: []llmprotocol.EmbeddingInput{
		{Type: llmprotocol.EmbeddingInputText, Text: "one"},
		{Type: llmprotocol.EmbeddingInputText, Text: "two"},
	}})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(result.Response.Data) != 2 || result.Response.Data[0].Embedding.Rank() != 2 || result.Response.Data[1].Embedding.Rank() != 1 {
		t.Fatalf("response = %#v", result.Response)
	}
}
