// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package codec_test

import (
	"bytes"
	"encoding/json"
	"testing"

	llmprotocol "github.com/scitrera/go-llm/protocol"
	"github.com/scitrera/go-llm/protocol/codec"
)

func TestAnthropicStructuredOutputAndStrictTools(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	chat := json.RawMessage(`{
	  "model":"logical","messages":[{"role":"user","content":"extract"}],"max_completion_tokens":128,
	  "response_format":{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"]}}},
	  "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"},"strict":true}}]
	}`)
	translated, err := registry.TranslateRequest(llmprotocol.FormatOpenAIChat, llmprotocol.FormatAnthropic, chat, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("Chat -> Anthropic error = %v", err)
	}
	var body struct {
		OutputConfig struct {
			Format struct {
				Type   string         `json:"type"`
				Schema map[string]any `json:"schema"`
			} `json:"format"`
		} `json:"output_config"`
		Tools []struct {
			Strict bool `json:"strict"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(translated.Body, &body); err != nil {
		t.Fatal(err)
	}
	if body.OutputConfig.Format.Type != "json_schema" || body.OutputConfig.Format.Schema["type"] != "object" || len(body.Tools) != 1 || !body.Tools[0].Strict {
		t.Fatalf("Anthropic body = %s", translated.Body)
	}
}

func TestBedrockStructuredOutputAndStrictTools(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	chat := json.RawMessage(`{
	  "model":"logical","messages":[{"role":"user","content":"extract"}],
	  "response_format":{"type":"json_schema","json_schema":{"name":"answer","description":"answer schema","strict":true,"schema":{"type":"object","additionalProperties":false}}},
	  "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"},"strict":true}}]
	}`)
	translated, err := registry.TranslateRequest(llmprotocol.FormatOpenAIChat, llmprotocol.FormatBedrock, chat, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("Chat -> Bedrock error = %v", err)
	}
	var body struct {
		OutputConfig struct {
			TextFormat struct {
				Type      string `json:"type"`
				Structure struct {
					JSONSchema struct {
						Name   string `json:"name"`
						Schema string `json:"schema"`
					} `json:"jsonSchema"`
				} `json:"structure"`
			} `json:"textFormat"`
		} `json:"outputConfig"`
		ToolConfig struct {
			Tools []struct {
				ToolSpec struct {
					Strict bool `json:"strict"`
				} `json:"toolSpec"`
			} `json:"tools"`
		} `json:"toolConfig"`
	}
	if err := json.Unmarshal(translated.Body, &body); err != nil {
		t.Fatal(err)
	}
	if body.OutputConfig.TextFormat.Type != "json_schema" || body.OutputConfig.TextFormat.Structure.JSONSchema.Name != "answer" || !json.Valid([]byte(body.OutputConfig.TextFormat.Structure.JSONSchema.Schema)) || !body.ToolConfig.Tools[0].ToolSpec.Strict {
		t.Fatalf("Bedrock body = %s", translated.Body)
	}
}

func TestGeminiNativeGenerationControls(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	chat := json.RawMessage(`{
	  "model":"logical","messages":[{"role":"user","content":"draw"}],
	  "temperature":0.4,"top_p":0.8,"frequency_penalty":0.3,"presence_penalty":0.2,"seed":42,
	  "n":2,"logprobs":true,"top_logprobs":5,"modalities":["text","image"],"reasoning_effort":"high"
	}`)
	translated, err := registry.TranslateRequest(llmprotocol.FormatOpenAIChat, llmprotocol.FormatGemini, chat, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("Chat -> Gemini error = %v", err)
	}
	var body struct {
		Generation struct {
			FrequencyPenalty float64  `json:"frequencyPenalty"`
			PresencePenalty  float64  `json:"presencePenalty"`
			Seed             int64    `json:"seed"`
			CandidateCount   int64    `json:"candidateCount"`
			ResponseLogprobs bool     `json:"responseLogprobs"`
			Logprobs         int64    `json:"logprobs"`
			Modalities       []string `json:"responseModalities"`
			Thinking         struct {
				Level string `json:"thinkingLevel"`
			} `json:"thinkingConfig"`
		} `json:"generationConfig"`
	}
	if err := json.Unmarshal(translated.Body, &body); err != nil {
		t.Fatal(err)
	}
	config := body.Generation
	if config.FrequencyPenalty != 0.3 || config.PresencePenalty != 0.2 || config.Seed != 42 || config.CandidateCount != 2 || !config.ResponseLogprobs || config.Logprobs != 5 || len(config.Modalities) != 2 || config.Modalities[1] != "IMAGE" || config.Thinking.Level != "HIGH" {
		t.Fatalf("Gemini body = %s", translated.Body)
	}
}

func TestGeminiThinkingBudgetDecodeEncode(t *testing.T) {
	t.Parallel()
	value := codec.Gemini{}
	decoded, err := value.DecodeRequest(json.RawMessage(`{
	  "contents":[{"role":"user","parts":[{"text":"think"}]}],
	  "generationConfig":{"thinkingConfig":{"thinkingBudget":1024,"includeThoughts":true},"candidateCount":1}
	}`), llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if decoded.Request.Reasoning.BudgetTokens == nil || *decoded.Request.Reasoning.BudgetTokens != 1024 || decoded.Request.Reasoning.Include == nil || !*decoded.Request.Reasoning.Include {
		t.Fatalf("reasoning = %#v", decoded.Request.Reasoning)
	}
	decoded.Request.ClearPreservation()
	encoded, err := value.EncodeRequest(decoded.Request, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	if len(encoded.Body) == 0 {
		t.Fatal("EncodeRequest() returned an empty body")
	}
}

func TestLogprobsNormalizeBetweenOpenAIChatAndGemini(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	chat := json.RawMessage(`{
	  "id":"c","model":"served",
	  "choices":[{"index":0,"message":{"role":"assistant","content":"A"},"finish_reason":"stop","logprobs":{"content":[
	    {"token":"A","bytes":[65],"logprob":-0.1,"top_logprobs":[{"token":"A","bytes":[65],"logprob":-0.1},{"token":"B","bytes":[66],"logprob":-2.0}]}
	  ]}}]
	}`)
	gemini, err := registry.TranslateResponse(llmprotocol.FormatOpenAIChat, llmprotocol.FormatGemini, chat, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("Chat -> Gemini error = %v", err)
	}
	if !bytes.Contains(gemini.Body, []byte(`"logprobsResult"`)) || !bytes.Contains(gemini.Body, []byte(`"logProbability":-0.1`)) {
		t.Fatalf("Gemini response = %s", gemini.Body)
	}
	chatAgain, err := registry.TranslateResponse(llmprotocol.FormatGemini, llmprotocol.FormatOpenAIChat, gemini.Body, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("Gemini -> Chat error = %v", err)
	}
	if !bytes.Contains(chatAgain.Body, []byte(`"top_logprobs"`)) || !bytes.Contains(chatAgain.Body, []byte(`"token":"B"`)) {
		t.Fatalf("round-trip Chat response = %s", chatAgain.Body)
	}
}

func TestStreamingLogprobsNormalizeBetweenGeminiAndChat(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	state := &codec.StreamState{}
	frame := llmprotocol.WireEvent{Data: json.RawMessage(`{
	  "candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"A"}]},"logprobsResult":{
	    "chosenCandidates":[{"token":"A","tokenId":1,"logProbability":-0.1}],
	    "topCandidates":[{"candidates":[{"token":"A","tokenId":1,"logProbability":-0.1}]}]
	  }}]
	}`)}
	events, _, err := registry.TranslateStreamEvent(state, llmprotocol.FormatGemini, llmprotocol.FormatOpenAIChat, frame, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("TranslateStreamEvent() error = %v", err)
	}
	var joined bytes.Buffer
	for _, event := range events {
		joined.Write(event.Data)
	}
	if !bytes.Contains(joined.Bytes(), []byte(`"logprobs"`)) || !bytes.Contains(joined.Bytes(), []byte(`"token":"A"`)) {
		t.Fatalf("translated stream = %s", joined.Bytes())
	}
}

// The canonical Message.ID must never reach an OpenAI chat request: that
// dialect has no message-level id, and strict upstreams reject the entire call
// ("Extra inputs are not permitted, field: 'messages[1].id'") rather than
// ignoring the unknown field — which fails every turn before a token is
// produced. Decoding still reads `id` back, so the canonical field stays usable
// for correlation; only the OpenAI lowering drops it.
func TestOpenAIChatRequestOmitsCanonicalMessageID(t *testing.T) {
	t.Parallel()
	request := llmprotocol.Request{
		Model: "served-model",
		Messages: []llmprotocol.Message{
			{ID: "msg_system", Role: llmprotocol.RoleSystem, Content: []llmprotocol.ContentBlock{llmprotocol.Text("be brief")}},
			{ID: "msg_11nr8xfja", Role: llmprotocol.RoleUser, Content: []llmprotocol.ContentBlock{llmprotocol.Text("hello")}},
			{ID: "chatcmpl-abf8816", Role: llmprotocol.RoleAssistant, Content: []llmprotocol.ContentBlock{llmprotocol.Text("hi")}},
		},
	}

	encoded, err := codec.OpenAIChat{}.EncodeRequest(request, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	var body struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(encoded.Body, &body); err != nil {
		t.Fatalf("unmarshal encoded body: %v", err)
	}
	if len(body.Messages) != len(request.Messages) {
		t.Fatalf("messages = %d, want %d", len(body.Messages), len(request.Messages))
	}
	for i, message := range body.Messages {
		if _, present := message["id"]; present {
			t.Fatalf("messages[%d] carries an id: %s", i, encoded.Body)
		}
	}
	// The ids must not survive anywhere in the request body.
	for _, id := range []string{"msg_system", "msg_11nr8xfja", "chatcmpl-abf8816"} {
		if bytes.Contains(encoded.Body, []byte(id)) {
			t.Fatalf("request body leaked message id %q: %s", id, encoded.Body)
		}
	}
}
