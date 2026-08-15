// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package codec_test

import (
	"encoding/json"
	"errors"
	"testing"

	llmprotocol "github.com/scitrera/go-llm/protocol"
	"github.com/scitrera/go-llm/protocol/codec"
)

func TestRecognizedProviderBlockMetadataCannotCrossFormatsSilently(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	tests := []struct {
		name   string
		source llmprotocol.Format
		target llmprotocol.Format
		body   json.RawMessage
	}{
		{
			name: "anthropic citation", source: llmprotocol.FormatAnthropic, target: llmprotocol.FormatOpenAIChat,
			body: json.RawMessage(`{"id":"m","model":"claude","content":[{"type":"text","text":"answer","citations":[{"type":"char_location","start_char_index":0}]}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`),
		},
		{
			name: "responses annotation", source: llmprotocol.FormatOpenAIResponses, target: llmprotocol.FormatAnthropic,
			body: json.RawMessage(`{"id":"r","model":"gpt","output":[{"id":"i","type":"message","role":"assistant","content":[{"type":"output_text","text":"answer","annotations":[{"type":"url_citation","url":"https://example.com"}]}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`),
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			_, err := registry.TranslateResponse(test.source, test.target, test.body, llmprotocol.StrictPolicy())
			var translationErr *llmprotocol.TranslationError
			if !errors.As(err, &translationErr) || translationErr.Code != "provider_extensions_not_portable" {
				t.Fatalf("strict metadata error = %v", err)
			}
		})
	}
}

func TestNestedProviderResourceCannotEscapeToolResultGuard(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	body := json.RawMessage(`{
	  "model":"claude","max_tokens":32,
	  "messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":[{"type":"document","source":{"type":"file","file_id":"file_private"}}]}]}]
	}`)
	_, err := registry.TranslateRequest(llmprotocol.FormatAnthropic, llmprotocol.FormatOpenAIChat, body, llmprotocol.StrictPolicy())
	var translationErr *llmprotocol.TranslationError
	if !errors.As(err, &translationErr) || translationErr.Code != "provider_resource_not_portable" {
		t.Fatalf("nested provider resource error = %v", err)
	}
}

func TestStreamingProviderSignatureCannotCrossFormats(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	state := &codec.StreamState{}
	_, _, err := registry.TranslateStreamEvent(
		state,
		llmprotocol.FormatAnthropic,
		llmprotocol.FormatOpenAIChat,
		llmprotocol.WireEvent{Event: "content_block_delta", Data: json.RawMessage(`{"index":0,"delta":{"type":"signature_delta","signature":"signed"}}`)},
		llmprotocol.StrictPolicy(),
	)
	var translationErr *llmprotocol.TranslationError
	if !errors.As(err, &translationErr) || translationErr.Code != "reasoning_signature_not_supported" {
		t.Fatalf("stream signature error = %v", err)
	}
}
