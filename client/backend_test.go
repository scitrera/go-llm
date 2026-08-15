// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package llmclient

import (
	"encoding/json"
	"strings"
	"testing"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

func TestBackendEndpointComposition(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name, base string
		format     llmprotocol.Format
		want       string
	}{
		{"chat base v1", "https://api.example.test/v1", llmprotocol.FormatOpenAIChat, "https://api.example.test/v1/chat/completions"},
		{"responses host", "https://api.example.test", llmprotocol.FormatOpenAIResponses, "https://api.example.test/v1/responses"},
		{"anthropic host", "https://api.example.test", llmprotocol.FormatAnthropic, "https://api.example.test/v1/messages"},
		{"gemini v1beta", "https://generativelanguage.googleapis.com/v1beta", llmprotocol.FormatGemini, "https://generativelanguage.googleapis.com/v1beta/models/model:generateContent"},
		{"bedrock encoded model", "https://bedrock-runtime.us-east-1.amazonaws.com", llmprotocol.FormatBedrock, "https://bedrock-runtime.us-east-1.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123%3Ainference-profile%2Fprofile/converse"},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			model := "model"
			if test.name == "bedrock encoded model" {
				model = "arn:aws:bedrock:us-east-1:123:inference-profile/profile"
			}
			endpoint, err := (Backend{BaseURL: test.base, Format: test.format}).endpoint(model, false)
			if err != nil || endpoint.String() != test.want {
				t.Fatalf("endpoint = %v, %v; want %s", endpoint, err, test.want)
			}
		})
	}
}

func TestStreamingEndpointComposition(t *testing.T) {
	t.Parallel()
	gemini, err := (Backend{BaseURL: "https://generativelanguage.googleapis.com/v1beta", Format: llmprotocol.FormatGemini}).endpoint("gemini-3", true)
	if err != nil || gemini.String() != "https://generativelanguage.googleapis.com/v1beta/models/gemini-3:streamGenerateContent?alt=sse" {
		t.Fatalf("Gemini stream endpoint = %v, %v", gemini, err)
	}
	bedrock, err := (Backend{BaseURL: "https://bedrock-runtime.us-east-1.amazonaws.com", Format: llmprotocol.FormatBedrock}).endpoint("model-id", true)
	if err != nil || bedrock.String() != "https://bedrock-runtime.us-east-1.amazonaws.com/model/model-id/converse-stream" {
		t.Fatalf("Bedrock stream endpoint = %v, %v", bedrock, err)
	}
}

func TestEmbeddingEndpointComposition(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name, base, model, want string
		format                  llmprotocol.Format
		batch                   bool
	}{
		{"OpenAI", "https://api.example.test/v1", "embed", "https://api.example.test/v1/embeddings", llmprotocol.FormatOpenAIEmbeddings, false},
		{"Gemini single", "https://generativelanguage.googleapis.com/v1beta", "gemini-embedding", "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding:embedContent", llmprotocol.FormatGeminiEmbeddings, false},
		{"Gemini batch", "https://generativelanguage.googleapis.com/v1beta", "gemini-embedding", "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding:batchEmbedContents", llmprotocol.FormatGeminiEmbeddings, true},
		{"Titan", "https://bedrock-runtime.us-east-1.amazonaws.com", "amazon.titan-embed-text-v2:0", "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-embed-text-v2%3A0/invoke", llmprotocol.FormatBedrockTitanEmbeddings, false},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			endpoint, err := (Backend{BaseURL: test.base, Format: test.format}).embeddingEndpoint(test.model, test.batch)
			if err != nil || endpoint.String() != test.want {
				t.Fatalf("endpoint = %v, %v; want %s", endpoint, err, test.want)
			}
		})
	}
}

func TestBodyDefaultsAreBoundedAndValidated(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		defaults map[string]json.RawMessage
	}{
		{"blank name", map[string]json.RawMessage{" ": json.RawMessage(`1`)}},
		{"control name", map[string]json.RawMessage{"bad\nname": json.RawMessage(`1`)}},
		{"invalid value", map[string]json.RawMessage{"seed": json.RawMessage(`nope`)}},
		{"protocol-owned value", map[string]json.RawMessage{"messages": json.RawMessage(`[]`)}},
		{"Gemini protocol-owned value", map[string]json.RawMessage{"systemInstruction": json.RawMessage(`{}`)}},
		{"Bedrock protocol-owned value", map[string]json.RawMessage{"system": json.RawMessage(`[]`)}},
		{"oversized value", map[string]json.RawMessage{"vendor": json.RawMessage(`"` + strings.Repeat("x", maxBodyDefaultValueBytes) + `"`)}},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if err := validateBodyDefaults(test.defaults); err == nil {
				t.Fatal("validateBodyDefaults() error = nil")
			}
		})
	}
}

func TestBodyDefaultsNeverOverrideRequest(t *testing.T) {
	t.Parallel()
	body, err := MergeBodyDefaults(json.RawMessage(`{"temperature":0}`), map[string]json.RawMessage{"temperature": json.RawMessage(`1`), "seed": json.RawMessage(`7`)})
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != `{"seed":7,"temperature":0}` {
		t.Fatalf("body = %s", body)
	}
}
