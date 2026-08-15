// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package codec_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"testing"

	llmprotocol "github.com/scitrera/go-llm/protocol"
	"github.com/scitrera/go-llm/protocol/codec"
)

func TestSameFormatPreservationIsExact(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	fixtures := map[llmprotocol.Format]json.RawMessage{
		llmprotocol.FormatOpenAIChat:      json.RawMessage(`{"model":"chat-model","messages":[{"role":"user","content":"hello"}],"metadata":{"keep":"exact"},"temperature":0}`),
		llmprotocol.FormatOpenAIResponses: json.RawMessage(`{"model":"responses-model","input":"hello","store":false,"metadata":{"keep":"exact"}}`),
		llmprotocol.FormatAnthropic:       json.RawMessage(`{"model":"claude-model","max_tokens":32,"messages":[{"role":"user","content":"hello"}],"metadata":{"keep":"exact"}}`),
		llmprotocol.FormatGemini:          json.RawMessage(`{"contents":[{"role":"user","parts":[{"text":"hello"}]}],"labels":{"keep":"exact"}}`),
		llmprotocol.FormatBedrock:         json.RawMessage(`{"messages":[{"role":"user","content":[{"text":"hello"}]}],"requestMetadata":{"keep":"exact"}}`),
	}
	for format, original := range fixtures {
		format, original := format, original
		t.Run(string(format), func(t *testing.T) {
			t.Parallel()
			translated, err := registry.TranslateRequest(format, format, original, llmprotocol.StrictPolicy())
			if err != nil {
				t.Fatalf("TranslateRequest() error = %v", err)
			}
			if string(translated.Body) != string(original) {
				t.Fatalf("same-format body changed\n got: %s\nwant: %s", translated.Body, original)
			}
		})
	}
}

func TestChatTranslatesToGeminiAndBedrock(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	chat := json.RawMessage(`{
  "model":"logical-model",
  "messages":[
    {"role":"system","content":"be concise"},
    {"role":"user","content":[{"type":"text","text":"weather?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]},
    {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Austin\"}"}}]},
    {"role":"tool","tool_call_id":"call_1","content":"{\"forecast\":\"sunny\"}"}
  ],
  "tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"},"strict":false}}],
  "tool_choice":{"type":"function","function":{"name":"weather"}},
  "max_completion_tokens":64,
  "temperature":0.4,
  "top_p":0.8,
  "stop":["END"]
}`)

	gemini, err := registry.TranslateRequest(llmprotocol.FormatOpenAIChat, llmprotocol.FormatGemini, chat, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("Chat -> Gemini error = %v", err)
	}
	var geminiBody map[string]any
	if err := json.Unmarshal(gemini.Body, &geminiBody); err != nil {
		t.Fatal(err)
	}
	if geminiBody["model"] != nil || geminiBody["contents"] == nil || geminiBody["systemInstruction"] == nil || geminiBody["tools"] == nil || geminiBody["toolConfig"] == nil {
		t.Fatalf("Gemini request = %s", gemini.Body)
	}
	generation := geminiBody["generationConfig"].(map[string]any)
	if generation["maxOutputTokens"] != float64(64) || generation["stopSequences"].([]any)[0] != "END" {
		t.Fatalf("Gemini generationConfig = %#v", generation)
	}

	bedrock, err := registry.TranslateRequest(llmprotocol.FormatOpenAIChat, llmprotocol.FormatBedrock, chat, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("Chat -> Bedrock error = %v", err)
	}
	var bedrockBody map[string]any
	if err := json.Unmarshal(bedrock.Body, &bedrockBody); err != nil {
		t.Fatal(err)
	}
	if bedrockBody["model"] != nil || bedrockBody["messages"] == nil || bedrockBody["system"] == nil || bedrockBody["toolConfig"] == nil {
		t.Fatalf("Bedrock request = %s", bedrock.Body)
	}
	inference := bedrockBody["inferenceConfig"].(map[string]any)
	if inference["maxTokens"] != float64(64) || inference["stopSequences"].([]any)[0] != "END" {
		t.Fatalf("Bedrock inferenceConfig = %#v", inference)
	}
}

func TestGeminiAndBedrockResponsesNormalizeToChat(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	tests := []struct {
		name   string
		format llmprotocol.Format
		body   json.RawMessage
	}{
		{
			name: "gemini", format: llmprotocol.FormatGemini,
			body: json.RawMessage(`{"responseId":"resp_g","modelVersion":"gemini-upstream","candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"checking"},{"functionCall":{"id":"call_2","name":"weather","args":{"city":"Austin"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"cachedContentTokenCount":2,"candidatesTokenCount":5,"thoughtsTokenCount":1,"totalTokenCount":16}}`),
		},
		{
			name: "bedrock", format: llmprotocol.FormatBedrock,
			body: json.RawMessage(`{"output":{"message":{"role":"assistant","content":[{"text":"checking"},{"toolUse":{"toolUseId":"call_2","name":"weather","input":{"city":"Austin"}}}]}},"stopReason":"tool_use","usage":{"inputTokens":21,"outputTokens":6,"totalTokens":27,"cacheReadInputTokens":4}}`),
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			translated, err := registry.TranslateResponse(test.format, llmprotocol.FormatOpenAIChat, test.body, llmprotocol.StrictPolicy())
			if err != nil {
				t.Fatalf("TranslateResponse() error = %v", err)
			}
			var body map[string]any
			if err := json.Unmarshal(translated.Body, &body); err != nil {
				t.Fatal(err)
			}
			choices := body["choices"].([]any)
			choice := choices[0].(map[string]any)
			message := choice["message"].(map[string]any)
			if choice["finish_reason"] != "tool_calls" || message["content"] != "checking" || len(message["tool_calls"].([]any)) != 1 {
				t.Fatalf("translated Chat response = %s", translated.Body)
			}
		})
	}
}

func TestProviderOwnedGeminiAndBedrockContentFailsClosed(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	gemini := json.RawMessage(`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"call_1","name":"weather","args":{}},"thoughtSignature":"signed"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`)
	_, err := registry.TranslateResponse(llmprotocol.FormatGemini, llmprotocol.FormatOpenAIChat, gemini, llmprotocol.StrictPolicy())
	var translationErr *llmprotocol.TranslationError
	if !errors.As(err, &translationErr) || translationErr.Code != "reasoning_signature_not_supported" {
		t.Fatalf("Gemini signature error = %v", err)
	}

	bedrock := json.RawMessage(`{"output":{"message":{"role":"assistant","content":[{"image":{"format":"png","source":{"s3Location":{"uri":"s3://private/image.png"}}}}]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)
	_, err = registry.TranslateResponse(llmprotocol.FormatBedrock, llmprotocol.FormatOpenAIChat, bedrock, llmprotocol.StrictPolicy())
	if !errors.As(err, &translationErr) || translationErr.Code != "provider_resource_not_portable" {
		t.Fatalf("Bedrock provider resource error = %v", err)
	}
}

func TestNewNeutralControlsAreNeverSilentlyDropped(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	chatWithStop := json.RawMessage(`{"model":"m","messages":[{"role":"user","content":"hello"}],"stop":["END"]}`)
	_, err := registry.TranslateRequest(llmprotocol.FormatOpenAIChat, llmprotocol.FormatOpenAIResponses, chatWithStop, llmprotocol.StrictPolicy())
	var translationErr *llmprotocol.TranslationError
	if !errors.As(err, &translationErr) || translationErr.Code != "stop_sequences_not_supported" {
		t.Fatalf("Chat stop -> Responses error = %v", err)
	}
	chatWithPenalty := json.RawMessage(`{"model":"m","messages":[{"role":"user","content":"hello"}],"frequency_penalty":0.5}`)
	_, err = registry.TranslateRequest(llmprotocol.FormatOpenAIChat, llmprotocol.FormatBedrock, chatWithPenalty, llmprotocol.StrictPolicy())
	if !errors.As(err, &translationErr) || translationErr.Code != "sampling_control_not_supported" {
		t.Fatalf("Chat penalty -> Bedrock error = %v", err)
	}
}

func TestGeminiAndBedrockStreamsTranslateToChat(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	tests := []struct {
		name   string
		format llmprotocol.Format
		frames []llmprotocol.WireEvent
	}{
		{
			name: "gemini", format: llmprotocol.FormatGemini,
			frames: []llmprotocol.WireEvent{
				{Data: json.RawMessage(`{"responseId":"resp_g","modelVersion":"gemini-upstream","candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"hello"}]}}]}`)},
				{Data: json.RawMessage(`{"candidates":[{"index":0,"content":{"role":"model","parts":[{"functionCall":{"id":"call_1","name":"weather","args":{"city":"Austin"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2,"totalTokenCount":6}}`)},
			},
		},
		{
			name: "bedrock", format: llmprotocol.FormatBedrock,
			frames: []llmprotocol.WireEvent{
				{Event: "messageStart", Data: json.RawMessage(`{"role":"assistant"}`)},
				{Event: "contentBlockStart", Data: json.RawMessage(`{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"call_1","name":"weather"}}}`)},
				{Event: "contentBlockDelta", Data: json.RawMessage(`{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"city\":\"Austin\"}"}}}`)},
				{Event: "messageStop", Data: json.RawMessage(`{"stopReason":"tool_use"}`)},
				{Event: "metadata", Data: json.RawMessage(`{"usage":{"inputTokens":4,"outputTokens":2,"totalTokens":6}}`)},
			},
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			state := &codec.StreamState{}
			var output []llmprotocol.WireEvent
			for _, frame := range test.frames {
				translated, _, err := registry.TranslateStreamEvent(state, test.format, llmprotocol.FormatOpenAIChat, frame, llmprotocol.StrictPolicy())
				if err != nil {
					t.Fatalf("TranslateStreamEvent() error = %v", err)
				}
				output = append(output, translated...)
			}
			finished, _, err := registry.FinishStream(state, llmprotocol.FormatOpenAIChat, llmprotocol.StrictPolicy())
			if err != nil {
				t.Fatal(err)
			}
			output = append(output, finished...)
			var encoded bytes.Buffer
			for _, event := range output {
				encoded.Write(event.Data)
			}
			if !bytes.Contains(encoded.Bytes(), []byte(`tool_calls`)) || !bytes.Contains(encoded.Bytes(), []byte(`total_tokens`)) || !bytes.Contains(encoded.Bytes(), []byte(`[DONE]`)) {
				t.Fatalf("translated stream = %s", encoded.Bytes())
			}
		})
	}
}

func TestChatResponsesRoundTripRetainsToolsAndUsage(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	request := json.RawMessage(`{
  "model":"logical-model",
  "messages":[
    {"role":"system","content":"be precise"},
    {"role":"user","content":[{"type":"text","text":"weather?"}]},
    {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Chicago\"}"}}]},
    {"role":"tool","tool_call_id":"call_1","content":"sunny"}
  ],
  "tools":[{"type":"function","function":{"name":"weather","description":"weather lookup","parameters":{"type":"object"},"strict":true}}],
  "tool_choice":"auto",
  "parallel_tool_calls":true,
  "max_completion_tokens":200
}`)
	responses, err := registry.TranslateRequest(llmprotocol.FormatOpenAIChat, llmprotocol.FormatOpenAIResponses, request, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("Chat -> Responses error = %v", err)
	}
	chat, err := registry.TranslateRequest(llmprotocol.FormatOpenAIResponses, llmprotocol.FormatOpenAIChat, responses.Body, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("Responses -> Chat error = %v\nbody=%s", err, responses.Body)
	}
	var body map[string]any
	if err := json.Unmarshal(chat.Body, &body); err != nil {
		t.Fatal(err)
	}
	messages, _ := body["messages"].([]any)
	if len(messages) != 4 {
		t.Fatalf("message count = %d, want 4; body=%s", len(messages), chat.Body)
	}
	tools, _ := body["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tool count = %d, want 1; body=%s", len(tools), chat.Body)
	}
}

func TestOpenAIUsageDetailsNormalizeAcrossFormats(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	chat := json.RawMessage(`{
  "id":"chat_1","model":"served-model",
  "choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],
  "usage":{"prompt_tokens":100,"completion_tokens":15,"total_tokens":115,
    "prompt_tokens_details":{"cached_tokens":70,"cache_creation_tokens":10},
    "completion_tokens_details":{"reasoning_tokens":5,"accepted_prediction_tokens":2,"rejected_prediction_tokens":1}}
}`)
	responses, err := registry.TranslateResponse(llmprotocol.FormatOpenAIChat, llmprotocol.FormatOpenAIResponses, chat, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("TranslateResponse() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(responses.Body, &decoded); err != nil {
		t.Fatal(err)
	}
	usage := decoded["usage"].(map[string]any)
	inputDetails := usage["input_tokens_details"].(map[string]any)
	outputDetails := usage["output_tokens_details"].(map[string]any)
	if inputDetails["cached_tokens"] != float64(70) || inputDetails["cache_creation_tokens"] != float64(10) {
		t.Fatalf("input usage details = %#v", inputDetails)
	}
	if outputDetails["reasoning_tokens"] != float64(5) || outputDetails["accepted_prediction_tokens"] != float64(2) || outputDetails["rejected_prediction_tokens"] != float64(1) {
		t.Fatalf("output usage details = %#v", outputDetails)
	}
}

func TestCacheUsageNormalizesBetweenOpenAIAndAnthropic(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	chat := json.RawMessage(`{
  "id":"chat_cache","model":"served-model",
  "choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],
  "usage":{"prompt_tokens":100,"completion_tokens":5,"total_tokens":105,
    "prompt_tokens_details":{"cached_tokens":70,"cache_write_tokens":10}}
}`)
	anthropic, err := registry.TranslateResponse(llmprotocol.FormatOpenAIChat, llmprotocol.FormatAnthropic, chat, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("Chat -> Anthropic error = %v", err)
	}
	var anthropicBody map[string]any
	if err := json.Unmarshal(anthropic.Body, &anthropicBody); err != nil {
		t.Fatal(err)
	}
	usage := anthropicBody["usage"].(map[string]any)
	if usage["input_tokens"] != float64(20) || usage["cache_read_input_tokens"] != float64(70) || usage["cache_creation_input_tokens"] != float64(10) {
		t.Fatalf("Anthropic usage = %#v", usage)
	}

	chatAgain, err := registry.TranslateResponse(llmprotocol.FormatAnthropic, llmprotocol.FormatOpenAIChat, anthropic.Body, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("Anthropic -> Chat error = %v", err)
	}
	var chatBody map[string]any
	if err := json.Unmarshal(chatAgain.Body, &chatBody); err != nil {
		t.Fatal(err)
	}
	chatUsage := chatBody["usage"].(map[string]any)
	if chatUsage["prompt_tokens"] != float64(100) || chatUsage["total_tokens"] != float64(105) {
		t.Fatalf("round-trip Chat usage = %#v", chatUsage)
	}
}

func TestJSONSchemaNormalizesBetweenOpenAIFormats(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	chat := json.RawMessage(`{
  "model":"logical-model","messages":[{"role":"user","content":"answer"}],
  "response_format":{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object"}}}
}`)
	responses, err := registry.TranslateRequest(llmprotocol.FormatOpenAIChat, llmprotocol.FormatOpenAIResponses, chat, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("Chat -> Responses error = %v", err)
	}
	var responsesBody map[string]any
	if err := json.Unmarshal(responses.Body, &responsesBody); err != nil {
		t.Fatal(err)
	}
	format := responsesBody["text"].(map[string]any)["format"].(map[string]any)
	if format["type"] != "json_schema" || format["name"] != "answer" || format["json_schema"] != nil {
		t.Fatalf("Responses text.format = %#v", format)
	}

	chatAgain, err := registry.TranslateRequest(llmprotocol.FormatOpenAIResponses, llmprotocol.FormatOpenAIChat, responses.Body, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("Responses -> Chat error = %v", err)
	}
	var chatBody map[string]any
	if err := json.Unmarshal(chatAgain.Body, &chatBody); err != nil {
		t.Fatal(err)
	}
	responseFormat := chatBody["response_format"].(map[string]any)
	schema := responseFormat["json_schema"].(map[string]any)
	if responseFormat["type"] != "json_schema" || schema["name"] != "answer" || schema["strict"] != true {
		t.Fatalf("round-trip Chat response_format = %#v", responseFormat)
	}
}

func TestStrictTranslationDoesNotForgeOrDropReasoningSignature(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	anthropic := json.RawMessage(`{
  "id":"msg_1","type":"message","role":"assistant","model":"claude-model",
  "content":[{"type":"thinking","thinking":"private","signature":"signed-by-provider"},{"type":"text","text":"answer"}],
  "stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":3}
}`)
	_, err := registry.TranslateResponse(llmprotocol.FormatAnthropic, llmprotocol.FormatOpenAIChat, anthropic, llmprotocol.StrictPolicy())
	var translationErr *llmprotocol.TranslationError
	if !errors.As(err, &translationErr) || translationErr.Code != "reasoning_signature_not_supported" {
		t.Fatalf("strict signed reasoning error = %v, want reasoning_signature_not_supported", err)
	}
	permissive, err := registry.TranslateResponse(llmprotocol.FormatAnthropic, llmprotocol.FormatOpenAIChat, anthropic, llmprotocol.PermissivePolicy())
	if err != nil {
		t.Fatalf("permissive signed reasoning error = %v", err)
	}
	if len(permissive.Diagnostics) == 0 || permissive.Diagnostics[0].Code != "reasoning_signature_not_supported" {
		t.Fatalf("diagnostics = %#v", permissive.Diagnostics)
	}
}

func TestChatStreamUsageAfterStopTranslatesToResponses(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	state := &codec.StreamState{}
	frames := []llmprotocol.WireEvent{
		{Data: json.RawMessage(`{"id":"chat_1","model":"served","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`)},
		{Data: json.RawMessage(`{"id":"chat_1","model":"served","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}`)},
		{Data: json.RawMessage(`{"id":"chat_1","model":"served","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)},
		{Data: json.RawMessage(`{"id":"chat_1","model":"served","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}`)},
		{Data: json.RawMessage(`[DONE]`)},
	}
	var translated []llmprotocol.WireEvent
	for _, frame := range frames {
		out, _, err := registry.TranslateStreamEvent(state, llmprotocol.FormatOpenAIChat, llmprotocol.FormatOpenAIResponses, frame, llmprotocol.StrictPolicy())
		if err != nil {
			t.Fatalf("TranslateStreamEvent() error = %v", err)
		}
		translated = append(translated, out...)
	}
	if len(translated) < 4 {
		t.Fatalf("translated event count = %d, want at least 4", len(translated))
	}
	if translated[0].Event != "response.created" {
		t.Fatalf("first event = %q, want response.created", translated[0].Event)
	}
	var added struct {
		Item struct {
			ID string `json:"id"`
		} `json:"item"`
	}
	if err := json.Unmarshal(translated[1].Data, &added); err != nil {
		t.Fatal(err)
	}
	if added.Item.ID == "" {
		t.Fatalf("translated output item has no stable ID: %s", translated[1].Data)
	}
	last := translated[len(translated)-1]
	if last.Event != "response.completed" {
		t.Fatalf("last event = %q, want response.completed", last.Event)
	}
	var terminal struct {
		Response struct {
			Usage struct {
				Total int64 `json:"total_tokens"`
			} `json:"usage"`
		} `json:"response"`
	}
	if err := json.Unmarshal(last.Data, &terminal); err != nil {
		t.Fatal(err)
	}
	if terminal.Response.Usage.Total != 5 {
		t.Fatalf("terminal total tokens = %d, want 5; event=%s", terminal.Response.Usage.Total, last.Data)
	}
}

func TestResponsesStateIsNeverSilentlyDiscarded(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	stateful := json.RawMessage(`{"model":"logical-model","input":"continue","store":true,"previous_response_id":"resp_1"}`)
	_, err := registry.TranslateRequest(llmprotocol.FormatOpenAIResponses, llmprotocol.FormatOpenAIChat, stateful, llmprotocol.StrictPolicy())
	var translationErr *llmprotocol.TranslationError
	if !errors.As(err, &translationErr) || translationErr.Code != "responses_state_not_supported" {
		t.Fatalf("strict state translation error = %v, want responses_state_not_supported", err)
	}

	stateless := json.RawMessage(`{"model":"logical-model","input":"hello","store":false,"background":false}`)
	translated, err := registry.TranslateRequest(llmprotocol.FormatOpenAIResponses, llmprotocol.FormatOpenAIChat, stateless, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("stateless translation error = %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(translated.Body, &body); err != nil {
		t.Fatal(err)
	}
	if stored, ok := body["store"].(bool); !ok || stored {
		t.Fatalf("translated store = %#v, want explicit false", body["store"])
	}
	if _, exists := body["background"]; exists {
		t.Fatalf("inactive Responses-only background leaked into Chat: %s", translated.Body)
	}
}

func TestProviderExtensionsRequireExplicitLossyPolicyAcrossFormats(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	request := json.RawMessage(`{"model":"logical-model","input":"hello","metadata":{"secret":"must-not-leak"}}`)
	_, err := registry.TranslateRequest(llmprotocol.FormatOpenAIResponses, llmprotocol.FormatOpenAIChat, request, llmprotocol.StrictPolicy())
	var translationErr *llmprotocol.TranslationError
	if !errors.As(err, &translationErr) || translationErr.Code != "provider_extensions_not_portable" {
		t.Fatalf("strict extension error = %v, want provider_extensions_not_portable", err)
	}

	translated, err := registry.TranslateRequest(llmprotocol.FormatOpenAIResponses, llmprotocol.FormatOpenAIChat, request, llmprotocol.PermissivePolicy())
	if err != nil {
		t.Fatalf("permissive extension translation error = %v", err)
	}
	if len(translated.Diagnostics) == 0 || translated.Diagnostics[0].Path != "$.extensions" {
		t.Fatalf("diagnostics = %#v, want bounded extension diagnostic", translated.Diagnostics)
	}
	var body map[string]any
	if err := json.Unmarshal(translated.Body, &body); err != nil {
		t.Fatal(err)
	}
	if _, exists := body["metadata"]; exists {
		t.Fatalf("provider metadata leaked across formats: %s", translated.Body)
	}
}

func TestHostedResponsesToolCannotBecomeEmptyChatFunction(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultRegistry()
	request := json.RawMessage(`{"model":"logical-model","input":"search","tools":[{"type":"web_search_preview"}]}`)
	_, err := registry.TranslateRequest(llmprotocol.FormatOpenAIResponses, llmprotocol.FormatOpenAIChat, request, llmprotocol.StrictPolicy())
	var translationErr *llmprotocol.TranslationError
	if !errors.As(err, &translationErr) || translationErr.Code != "hosted_tool_not_supported" {
		t.Fatalf("hosted tool error = %v, want hosted_tool_not_supported", err)
	}
}
