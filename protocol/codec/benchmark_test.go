// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package codec_test

import (
	"encoding/json"
	"testing"

	llmprotocol "github.com/scitrera/go-llm/protocol"
	"github.com/scitrera/go-llm/protocol/codec"
)

var benchmarkChatRequest = json.RawMessage(`{
  "model":"logical-model",
  "messages":[
    {"role":"system","content":"Be precise."},
    {"role":"user","content":"Use the weather tool for Chicago."}
  ],
  "tools":[{"type":"function","function":{"name":"weather","description":"Look up weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
	"max_completion_tokens":1024,
  "stream":true
}`)

func BenchmarkTranslateRequest(b *testing.B) {
	registry := codec.NewDefaultRegistry()
	benchmarks := []struct {
		name           string
		source, target llmprotocol.Format
	}{
		{"chat_same_format", llmprotocol.FormatOpenAIChat, llmprotocol.FormatOpenAIChat},
		{"chat_to_responses", llmprotocol.FormatOpenAIChat, llmprotocol.FormatOpenAIResponses},
		{"chat_to_anthropic", llmprotocol.FormatOpenAIChat, llmprotocol.FormatAnthropic},
		{"chat_to_gemini", llmprotocol.FormatOpenAIChat, llmprotocol.FormatGemini},
		{"chat_to_bedrock", llmprotocol.FormatOpenAIChat, llmprotocol.FormatBedrock},
	}
	for _, benchmark := range benchmarks {
		benchmark := benchmark
		b.Run(benchmark.name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(benchmarkChatRequest)))
			for i := 0; i < b.N; i++ {
				result, err := registry.TranslateRequest(benchmark.source, benchmark.target, benchmarkChatRequest, llmprotocol.StrictPolicy())
				if err != nil || len(result.Body) == 0 {
					b.Fatalf("TranslateRequest() body=%d error=%v", len(result.Body), err)
				}
			}
		})
	}
}

func BenchmarkTranslateChatStreamToResponses(b *testing.B) {
	registry := codec.NewDefaultRegistry()
	frames := []llmprotocol.WireEvent{
		{Data: json.RawMessage(`{"id":"chat_1","model":"served","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`)},
		{Data: json.RawMessage(`{"id":"chat_1","model":"served","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}`)},
		{Data: json.RawMessage(`{"id":"chat_1","model":"served","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)},
		{Data: json.RawMessage(`{"id":"chat_1","model":"served","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}`)},
		{Data: json.RawMessage(`[DONE]`)},
	}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		state := &codec.StreamState{}
		for _, frame := range frames {
			if _, _, err := registry.TranslateStreamEvent(state, llmprotocol.FormatOpenAIChat, llmprotocol.FormatOpenAIResponses, frame, llmprotocol.StrictPolicy()); err != nil {
				b.Fatal(err)
			}
		}
	}
}
