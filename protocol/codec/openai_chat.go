// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package codec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

type OpenAIChat struct{}

func (OpenAIChat) Format() llmprotocol.Format { return llmprotocol.FormatOpenAIChat }

func (OpenAIChat) DecodeRequest(body json.RawMessage, policy llmprotocol.Policy) (RequestResult, error) {
	const format = llmprotocol.FormatOpenAIChat
	object, err := decodeObject(format, body)
	if err != nil {
		return RequestResult{}, err
	}
	request := llmprotocol.Request{Preservation: preserveBody(policy, format, body, true)}
	request.Model, err = optionalString(format, object, "model")
	if err != nil {
		return RequestResult{}, err
	}
	request.Stream = false
	if stream, streamErr := optionalBool(format, object, "stream"); streamErr != nil {
		return RequestResult{}, streamErr
	} else if stream != nil {
		request.Stream = *stream
	}
	request.Sampling.Temperature, err = optionalFloat(format, object, "temperature")
	if err != nil {
		return RequestResult{}, err
	}
	request.Sampling.TopP, err = optionalFloat(format, object, "top_p")
	if err != nil {
		return RequestResult{}, err
	}
	request.Sampling.FrequencyPenalty, err = optionalFloat(format, object, "frequency_penalty")
	if err != nil {
		return RequestResult{}, err
	}
	request.Sampling.PresencePenalty, err = optionalFloat(format, object, "presence_penalty")
	if err != nil {
		return RequestResult{}, err
	}
	request.Sampling.Seed, err = optionalInt(format, object, "seed")
	if err != nil {
		return RequestResult{}, err
	}
	request.Output.MaxTokens, err = optionalInt(format, object, "max_completion_tokens")
	if err != nil {
		return RequestResult{}, err
	}
	if request.Output.MaxTokens == nil {
		request.Output.MaxTokens, err = optionalInt(format, object, "max_tokens")
		if err != nil {
			return RequestResult{}, err
		}
	} else {
		delete(object, "max_tokens")
	}
	request.Output.StopSequences, err = optionalStrings(format, object, "stop", true)
	if err != nil {
		return RequestResult{}, err
	}
	// These are transport/presentation controls. They do not change the neutral
	// request and are consumed here so a cross-format codec does not mistake
	// them for provider extensions.
	delete(object, "stream_options")
	request.Output.Choices, err = optionalInt(format, object, "n")
	if err != nil {
		return RequestResult{}, err
	}
	request.Output.Logprobs, err = optionalBool(format, object, "logprobs")
	if err != nil {
		return RequestResult{}, err
	}
	request.Output.TopLogprobs, err = optionalInt(format, object, "top_logprobs")
	if err != nil {
		return RequestResult{}, err
	}
	request.Output.Modalities, err = optionalStrings(format, object, "modalities", false)
	if err != nil {
		return RequestResult{}, err
	}
	if raw, ok := object["response_format"]; ok {
		request.Output.ResponseFormat = cloneRaw(raw)
		delete(object, "response_format")
	}
	request.Reasoning.Effort, err = optionalString(format, object, "reasoning_effort")
	if err != nil {
		return RequestResult{}, err
	}
	request.ParallelToolCalls, err = optionalBool(format, object, "parallel_tool_calls")
	if err != nil {
		return RequestResult{}, err
	}
	request.State.Store, err = optionalBool(format, object, "store")
	if err != nil {
		return RequestResult{}, err
	}

	var diagnostics []llmprotocol.Diagnostic
	if raw, ok := object["messages"]; ok {
		delete(object, "messages")
		messages, nextDiagnostics, decodeErr := decodeChatMessages(raw, policy)
		if decodeErr != nil {
			return RequestResult{}, decodeErr
		}
		diagnostics = append(diagnostics, nextDiagnostics...)
		for _, message := range messages {
			if message.Role == llmprotocol.RoleSystem || message.Role == llmprotocol.RoleDeveloper {
				request.Instructions = append(request.Instructions, llmprotocol.Instruction{Role: message.Role, Content: message.Content, Extensions: message.Extensions})
			} else {
				request.Messages = append(request.Messages, message)
			}
		}
	}
	if raw, ok := object["tools"]; ok {
		delete(object, "tools")
		request.Tools, diagnostics, err = decodeChatTools(raw, policy, diagnostics)
		if err != nil {
			return RequestResult{}, err
		}
	}
	if raw, ok := object["tool_choice"]; ok {
		delete(object, "tool_choice")
		request.ToolChoice, err = decodeChatToolChoice(raw)
		if err != nil {
			return RequestResult{}, err
		}
	}
	request.Extensions, diagnostics, err = collectAndAppendExtensions(format, object, policy, diagnostics)
	if err != nil {
		return RequestResult{}, err
	}
	return RequestResult{Request: request, Diagnostics: diagnostics}, nil
}

func (OpenAIChat) EncodeRequest(request llmprotocol.Request, policy llmprotocol.Policy) (WireResult, error) {
	const format = llmprotocol.FormatOpenAIChat
	if raw, ok := preservedRequest(request, format, policy); ok {
		return WireResult{Body: raw}, nil
	}
	object := map[string]any{"model": request.Model}
	var diagnostics []llmprotocol.Diagnostic
	if request.Stream {
		object["stream"] = true
	}
	if request.Sampling.Temperature != nil {
		object["temperature"] = *request.Sampling.Temperature
	}
	if request.Sampling.TopP != nil {
		object["top_p"] = *request.Sampling.TopP
	}
	if request.Sampling.FrequencyPenalty != nil {
		object["frequency_penalty"] = *request.Sampling.FrequencyPenalty
	}
	if request.Sampling.PresencePenalty != nil {
		object["presence_penalty"] = *request.Sampling.PresencePenalty
	}
	putInt(object, "seed", request.Sampling.Seed)
	if request.Sampling.TopK != nil {
		d, err := lossy(format, policy, "$.top_k", "top_k_not_supported")
		diagnostics = append(diagnostics, d...)
		if err != nil {
			return WireResult{}, err
		}
	}
	if request.Output.MaxTokens != nil {
		object["max_completion_tokens"] = *request.Output.MaxTokens
	}
	if len(request.Output.StopSequences) == 1 {
		object["stop"] = request.Output.StopSequences[0]
	} else if len(request.Output.StopSequences) > 1 {
		object["stop"] = request.Output.StopSequences
	}
	if request.Output.Choices != nil {
		object["n"] = *request.Output.Choices
	}
	if request.Output.Logprobs != nil {
		object["logprobs"] = *request.Output.Logprobs
	}
	putInt(object, "top_logprobs", request.Output.TopLogprobs)
	if len(request.Output.Modalities) > 0 {
		object["modalities"] = request.Output.Modalities
	}
	if len(request.Output.ResponseFormat) != 0 {
		object["response_format"] = chatResponseFormat(request.Output.ResponseFormat)
	}
	if request.Reasoning.Effort != "" {
		object["reasoning_effort"] = request.Reasoning.Effort
	}
	if request.Reasoning.BudgetTokens != nil || request.Reasoning.Include != nil || request.Reasoning.Level != "" || len(request.Reasoning.Raw) != 0 {
		d, err := lossy(format, policy, "$.reasoning", "reasoning_config_not_portable")
		diagnostics = append(diagnostics, d...)
		if err != nil {
			return WireResult{}, err
		}
	}
	if request.ParallelToolCalls != nil {
		object["parallel_tool_calls"] = *request.ParallelToolCalls
	}
	if request.State.Store != nil {
		object["store"] = *request.State.Store
	}
	if (request.State.Background != nil && *request.State.Background) || request.State.PreviousResponseID != "" || len(request.State.Conversation) != 0 {
		d, err := lossy(format, policy, "$.state", "responses_state_not_supported")
		diagnostics = append(diagnostics, d...)
		if err != nil {
			return WireResult{}, err
		}
	}
	messages := make([]any, 0, len(request.Instructions)+len(request.Messages))
	for _, instruction := range request.Instructions {
		message, next, err := encodeChatMessage(llmprotocol.Message{Role: instruction.Role, Content: instruction.Content, Extensions: instruction.Extensions}, policy)
		if err != nil {
			return WireResult{}, err
		}
		diagnostics = append(diagnostics, next...)
		messages = append(messages, message)
	}
	for _, source := range request.Messages {
		message, next, err := encodeChatMessage(source, policy)
		if err != nil {
			return WireResult{}, err
		}
		diagnostics = append(diagnostics, next...)
		messages = append(messages, message)
	}
	object["messages"] = messages
	if len(request.Tools) > 0 {
		tools := make([]any, 0, len(request.Tools))
		for _, tool := range request.Tools {
			if tool.Type != "" && tool.Type != "function" {
				d, err := lossy(format, policy, "$.tools[]", "hosted_tool_not_supported")
				diagnostics = append(diagnostics, d...)
				if err != nil {
					return WireResult{}, err
				}
				continue
			}
			function := map[string]any{"name": tool.Name, "parameters": rawOrEmptyObject(tool.Parameters)}
			if tool.Description != "" {
				function["description"] = tool.Description
			}
			if tool.Strict != nil {
				function["strict"] = *tool.Strict
			}
			mergeExtensions(function, tool.Extensions)
			tools = append(tools, map[string]any{"type": "function", "function": function})
		}
		object["tools"] = tools
	}
	if request.ToolChoice != nil {
		choice, err := encodeChatToolChoice(*request.ToolChoice)
		if err != nil {
			return WireResult{}, err
		}
		object["tool_choice"] = choice
	}
	mergeExtensions(object, request.Extensions)
	body, err := marshalObject(format, object)
	return WireResult{Body: body, Diagnostics: diagnostics}, err
}

func chatResponseFormat(raw json.RawMessage) any {
	var format map[string]json.RawMessage
	if json.Unmarshal(raw, &format) != nil {
		return json.RawMessage(cloneRaw(raw))
	}
	var typeName string
	_ = json.Unmarshal(format["type"], &typeName)
	if typeName != "json_schema" || format["json_schema"] != nil {
		return json.RawMessage(cloneRaw(raw))
	}
	schema := map[string]json.RawMessage{}
	for _, name := range []string{"name", "description", "schema", "strict"} {
		if value, ok := format[name]; ok {
			schema[name] = value
		}
	}
	return map[string]any{
		"type":        "json_schema",
		"json_schema": json.RawMessage(mustMarshal(schema)),
	}
}

func mustMarshal(value any) []byte {
	encoded, _ := json.Marshal(value)
	return encoded
}

func (OpenAIChat) DecodeResponse(body json.RawMessage, policy llmprotocol.Policy) (ResponseResult, error) {
	const format = llmprotocol.FormatOpenAIChat
	object, err := decodeObject(format, body)
	if err != nil {
		return ResponseResult{}, err
	}
	response := llmprotocol.Response{Preservation: preserveBody(policy, format, body, false)}
	response.ID, err = optionalString(format, object, "id")
	if err != nil {
		return ResponseResult{}, err
	}
	response.Model, err = optionalString(format, object, "model")
	if err != nil {
		return ResponseResult{}, err
	}
	delete(object, "object")
	delete(object, "created")
	delete(object, "service_tier")
	delete(object, "system_fingerprint")
	var diagnostics []llmprotocol.Diagnostic
	if raw, ok := object["choices"]; ok {
		delete(object, "choices")
		choices, arrayErr := decodeArray(format, "$.choices", raw)
		if arrayErr != nil {
			return ResponseResult{}, arrayErr
		}
		for index, choiceRaw := range choices {
			choice, objectErr := decodeObject(format, choiceRaw)
			if objectErr != nil {
				return ResponseResult{}, objectErr
			}
			messageRaw, exists := choice["message"]
			if !exists {
				return ResponseResult{}, translationError(format, fmt.Sprintf("$.choices[%d].message", index), "missing_message", "choice message is required")
			}
			delete(choice, "message")
			message, nextDiagnostics, messageErr := decodeChatMessage(messageRaw, policy, fmt.Sprintf("$.choices[%d].message", index))
			if messageErr != nil {
				return ResponseResult{}, messageErr
			}
			diagnostics = append(diagnostics, nextDiagnostics...)
			output := llmprotocol.ResponseOutput{Role: message.Role, Content: message.Content, Extensions: message.Extensions}
			choiceIndex, indexErr := optionalInt(format, choice, "index")
			if indexErr != nil {
				return ResponseResult{}, indexErr
			}
			if choiceIndex == nil {
				fallback := index
				output.Index = &fallback
			} else {
				value := int(*choiceIndex)
				output.Index = &value
			}
			if rawReason, exists := choice["finish_reason"]; exists && !bytes.Equal(bytes.TrimSpace(rawReason), []byte("null")) {
				delete(choice, "finish_reason")
				value, reasonErr := decodeString(format, fmt.Sprintf("$.choices[%d].finish_reason", index), rawReason)
				if reasonErr != nil {
					return ResponseResult{}, reasonErr
				}
				output.StopReason = decodeChatStopReason(value)
			} else {
				delete(choice, "finish_reason")
			}
			if rawLogprobs, exists := choice["logprobs"]; exists && !bytes.Equal(bytes.TrimSpace(rawLogprobs), []byte("null")) {
				delete(choice, "logprobs")
				output.Logprobs, err = decodeOpenAIChatLogprobs(rawLogprobs)
				if err != nil {
					return ResponseResult{}, err
				}
			} else {
				delete(choice, "logprobs")
			}
			choiceExtensions, next, extensionErr := collectExtensions(format, choice, policy)
			diagnostics = append(diagnostics, next...)
			if extensionErr != nil {
				return ResponseResult{}, extensionErr
			}
			if output.Extensions == nil {
				output.Extensions = choiceExtensions
			} else {
				for key, value := range choiceExtensions {
					output.Extensions[key] = value
				}
			}
			response.Outputs = append(response.Outputs, output)
		}
	}
	if raw, ok := object["usage"]; ok {
		delete(object, "usage")
		response.Usage, err = decodeOpenAIUsage(raw, format)
		if err != nil {
			return ResponseResult{}, err
		}
	}
	response.Extensions, diagnostics, err = collectAndAppendExtensions(format, object, policy, diagnostics)
	if err != nil {
		return ResponseResult{}, err
	}
	return ResponseResult{Response: response, Diagnostics: diagnostics}, nil
}

func (OpenAIChat) EncodeResponse(response llmprotocol.Response, policy llmprotocol.Policy) (WireResult, error) {
	const format = llmprotocol.FormatOpenAIChat
	if raw, ok := preservedResponse(response, format, policy); ok {
		return WireResult{Body: raw}, nil
	}
	object := map[string]any{"id": response.ID, "object": "chat.completion", "model": response.Model}
	choices := make([]any, 0, len(response.Outputs))
	var diagnostics []llmprotocol.Diagnostic
	for index, output := range response.Outputs {
		// output.ID is intentionally not forwarded: the completion id is set at
		// the top level above, and choices[].message carries none.
		message, next, err := encodeChatMessage(llmprotocol.Message{Role: output.Role, Content: output.Content, Extensions: output.Extensions}, policy)
		if err != nil {
			return WireResult{}, err
		}
		diagnostics = append(diagnostics, next...)
		choiceIndex := index
		if output.Index != nil {
			choiceIndex = *output.Index
		}
		choice := map[string]any{"index": choiceIndex, "message": message, "finish_reason": encodeChatStopReason(output.StopReason)}
		if len(output.Logprobs) > 0 {
			choice["logprobs"] = encodeOpenAIChatLogprobs(output.Logprobs)
		}
		choices = append(choices, choice)
	}
	object["choices"] = choices
	if hasUsage(response.Usage) {
		object["usage"] = encodeOpenAIUsage(response.Usage, true)
	}
	mergeExtensions(object, response.Extensions)
	body, err := marshalObject(format, object)
	return WireResult{Body: body, Diagnostics: diagnostics}, err
}

func decodeChatMessages(raw json.RawMessage, policy llmprotocol.Policy) ([]llmprotocol.Message, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatOpenAIChat
	values, err := decodeArray(format, "$.messages", raw)
	if err != nil {
		return nil, nil, err
	}
	messages := make([]llmprotocol.Message, 0, len(values))
	var diagnostics []llmprotocol.Diagnostic
	for index, value := range values {
		message, next, decodeErr := decodeChatMessage(value, policy, fmt.Sprintf("$.messages[%d]", index))
		if decodeErr != nil {
			return nil, diagnostics, decodeErr
		}
		diagnostics = append(diagnostics, next...)
		messages = append(messages, message)
	}
	return messages, diagnostics, nil
}

func decodeChatMessage(raw json.RawMessage, policy llmprotocol.Policy, path string) (llmprotocol.Message, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatOpenAIChat
	object, err := decodeObject(format, raw)
	if err != nil {
		return llmprotocol.Message{}, nil, err
	}
	roleName, err := optionalString(format, object, "role")
	if err != nil {
		return llmprotocol.Message{}, nil, err
	}
	role, err := validateRole(format, path+".role", roleName)
	if err != nil {
		return llmprotocol.Message{}, nil, err
	}
	message := llmprotocol.Message{Role: role}
	message.ID, err = optionalString(format, object, "id")
	if err != nil {
		return llmprotocol.Message{}, nil, err
	}
	var diagnostics []llmprotocol.Diagnostic
	if rawContent, ok := object["content"]; ok {
		delete(object, "content")
		message.Content, diagnostics, err = decodeChatContent(rawContent, policy, path+".content", diagnostics)
		if err != nil {
			return llmprotocol.Message{}, diagnostics, err
		}
	}
	// Reasoning has no OpenAI-standard field name, so the same content arrives
	// under a different key per vendor: DeepSeek/vLLM/SGLang use
	// `reasoning_content`, OpenRouter `reasoning`, Ollama `thinking`. First
	// present wins; the others are consumed so a second alias neither
	// duplicates the block nor surfaces as an unknown field. Mirrors the
	// streaming decoder's alias table (openai_chat_stream.go).
	reasoned := false
	for _, name := range []string{"reasoning_content", "reasoning", "thinking"} {
		rawReasoning, ok := object[name]
		if !ok {
			continue
		}
		delete(object, name)
		if bytes.Equal(bytes.TrimSpace(rawReasoning), []byte("null")) {
			continue
		}
		text, decodeErr := decodeString(format, path+"."+name, rawReasoning)
		if decodeErr != nil {
			return llmprotocol.Message{}, diagnostics, decodeErr
		}
		if reasoned {
			continue
		}
		message.Content = append([]llmprotocol.ContentBlock{{Type: llmprotocol.ContentReasoning, Text: text, Provider: format}}, message.Content...)
		reasoned = true
	}
	if rawRefusal, ok := object["refusal"]; ok && !bytes.Equal(bytes.TrimSpace(rawRefusal), []byte("null")) {
		delete(object, "refusal")
		text, decodeErr := decodeString(format, path+".refusal", rawRefusal)
		if decodeErr != nil {
			return llmprotocol.Message{}, diagnostics, decodeErr
		}
		message.Content = append(message.Content, llmprotocol.ContentBlock{Type: llmprotocol.ContentRefusal, Text: text, Provider: format})
	}
	if rawCalls, ok := object["tool_calls"]; ok {
		delete(object, "tool_calls")
		calls, arrayErr := decodeArray(format, path+".tool_calls", rawCalls)
		if arrayErr != nil {
			return llmprotocol.Message{}, diagnostics, arrayErr
		}
		for index, callRaw := range calls {
			call, callDiagnostics, callErr := decodeChatToolCall(callRaw, policy, fmt.Sprintf("%s.tool_calls[%d]", path, index))
			if callErr != nil {
				return llmprotocol.Message{}, diagnostics, callErr
			}
			diagnostics = append(diagnostics, callDiagnostics...)
			message.Content = append(message.Content, llmprotocol.ContentBlock{Type: llmprotocol.ContentToolCall, ToolCall: &call, Provider: format})
		}
	}
	if role == llmprotocol.RoleTool {
		callID, callErr := optionalString(format, object, "tool_call_id")
		if callErr != nil {
			return llmprotocol.Message{}, diagnostics, callErr
		}
		message.Content = []llmprotocol.ContentBlock{{Type: llmprotocol.ContentToolResult, Provider: format, Result: &llmprotocol.ToolResult{ToolCallID: callID, Content: message.Content}}}
	}
	message.Extensions, diagnostics, err = collectAndAppendExtensions(format, object, policy, diagnostics)
	if err != nil {
		return llmprotocol.Message{}, diagnostics, err
	}
	return message, diagnostics, nil
}

func decodeChatContent(raw json.RawMessage, policy llmprotocol.Policy, path string, diagnostics []llmprotocol.Diagnostic) ([]llmprotocol.ContentBlock, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatOpenAIChat
	trimmed := bytes.TrimSpace(raw)
	if bytes.Equal(trimmed, []byte("null")) || len(trimmed) == 0 {
		return nil, diagnostics, nil
	}
	if trimmed[0] == '"' {
		text, err := decodeString(format, path, raw)
		if err != nil {
			return nil, diagnostics, err
		}
		return []llmprotocol.ContentBlock{{Type: llmprotocol.ContentText, Text: text, Provider: format}}, diagnostics, nil
	}
	values, err := decodeArray(format, path, raw)
	if err != nil {
		return nil, diagnostics, err
	}
	blocks := make([]llmprotocol.ContentBlock, 0, len(values))
	for index, value := range values {
		blockPath := fmt.Sprintf("%s[%d]", path, index)
		object, objectErr := decodeObject(format, value)
		if objectErr != nil {
			return nil, diagnostics, objectErr
		}
		typeName, typeErr := optionalString(format, object, "type")
		if typeErr != nil {
			return nil, diagnostics, typeErr
		}
		switch typeName {
		case "text", "input_text", "output_text":
			text, textErr := optionalString(format, object, "text")
			if textErr != nil {
				return nil, diagnostics, textErr
			}
			blocks = append(blocks, llmprotocol.ContentBlock{Type: llmprotocol.ContentText, Text: text, Provider: format})
		case "image_url":
			source, sourceErr := decodeChatImageSource(object["image_url"], blockPath+".image_url")
			if sourceErr != nil {
				return nil, diagnostics, sourceErr
			}
			delete(object, "image_url")
			blocks = append(blocks, llmprotocol.ContentBlock{Type: llmprotocol.ContentImage, Source: &source, Provider: format})
		case "input_audio":
			var audio struct {
				Data   string `json:"data"`
				Format string `json:"format"`
			}
			if err := json.Unmarshal(object["input_audio"], &audio); err != nil {
				return nil, diagnostics, translationError(format, blockPath+".input_audio", "invalid_audio", "input_audio must be an object")
			}
			delete(object, "input_audio")
			blocks = append(blocks, llmprotocol.ContentBlock{Type: llmprotocol.ContentAudio, Source: &llmprotocol.Source{Kind: "base64", Data: audio.Data, MediaType: audio.Format}, Provider: format})
		case "file", "input_file":
			var file struct {
				FileID   string `json:"file_id"`
				FileData string `json:"file_data"`
				Filename string `json:"filename"`
			}
			fileRaw := object["file"]
			if typeName == "input_file" {
				fileRaw = value
			}
			if err := json.Unmarshal(fileRaw, &file); err != nil {
				return nil, diagnostics, translationError(format, blockPath, "invalid_file", "file content must be an object")
			}
			blocks = append(blocks, llmprotocol.ContentBlock{Type: llmprotocol.ContentFile, Source: &llmprotocol.Source{Kind: chooseSourceKind(file.FileID, file.FileData), FileID: file.FileID, Data: file.FileData, Filename: file.Filename}, Provider: format})
		default:
			block, next, unknownErr := decodeUnknownBlock(format, value, policy, blockPath)
			diagnostics = append(diagnostics, next...)
			if unknownErr != nil {
				return nil, diagnostics, unknownErr
			}
			if block != nil {
				blocks = append(blocks, *block)
			}
		}
	}
	return blocks, diagnostics, nil
}

func decodeChatImageSource(raw json.RawMessage, path string) (llmprotocol.Source, error) {
	const format = llmprotocol.FormatOpenAIChat
	if len(bytes.TrimSpace(raw)) == 0 {
		return llmprotocol.Source{}, translationError(format, path, "missing_image", "image_url is required")
	}
	if bytes.TrimSpace(raw)[0] == '"' {
		url, err := decodeString(format, path, raw)
		return llmprotocol.Source{Kind: "url", URL: url}, err
	}
	var image struct {
		URL    string `json:"url"`
		Detail string `json:"detail"`
	}
	if err := json.Unmarshal(raw, &image); err != nil {
		return llmprotocol.Source{}, translationError(format, path, "invalid_image", "image_url must be a string or object")
	}
	return llmprotocol.Source{Kind: "url", URL: image.URL, Detail: image.Detail}, nil
}

func decodeChatToolCall(raw json.RawMessage, policy llmprotocol.Policy, path string) (llmprotocol.ToolCall, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatOpenAIChat
	object, err := decodeObject(format, raw)
	if err != nil {
		return llmprotocol.ToolCall{}, nil, err
	}
	id, err := optionalString(format, object, "id")
	if err != nil {
		return llmprotocol.ToolCall{}, nil, err
	}
	typeName, err := optionalString(format, object, "type")
	if err != nil {
		return llmprotocol.ToolCall{}, nil, err
	}
	if typeName != "" && typeName != "function" {
		return llmprotocol.ToolCall{}, nil, translationError(format, path+".type", "unsupported_tool_type", "only function tools are representable")
	}
	function, err := decodeObject(format, object["function"])
	if err != nil {
		return llmprotocol.ToolCall{}, nil, err
	}
	delete(object, "function")
	name, err := optionalString(format, function, "name")
	if err != nil {
		return llmprotocol.ToolCall{}, nil, err
	}
	arguments, err := optionalString(format, function, "arguments")
	if err != nil {
		return llmprotocol.ToolCall{}, nil, err
	}
	args := json.RawMessage(arguments)
	if !json.Valid(args) {
		args = rawJSONString(arguments)
	}
	_, diagnostics, err := collectExtensions(format, object, policy)
	if err != nil {
		return llmprotocol.ToolCall{}, diagnostics, err
	}
	_, next, err := collectExtensions(format, function, policy)
	diagnostics = append(diagnostics, next...)
	if err != nil {
		return llmprotocol.ToolCall{}, diagnostics, err
	}
	return llmprotocol.ToolCall{ID: id, Name: name, Arguments: args}, diagnostics, nil
}

func decodeChatTools(raw json.RawMessage, policy llmprotocol.Policy, diagnostics []llmprotocol.Diagnostic) ([]llmprotocol.ToolDefinition, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatOpenAIChat
	values, err := decodeArray(format, "$.tools", raw)
	if err != nil {
		return nil, diagnostics, err
	}
	tools := make([]llmprotocol.ToolDefinition, 0, len(values))
	for index, value := range values {
		object, objectErr := decodeObject(format, value)
		if objectErr != nil {
			return nil, diagnostics, objectErr
		}
		typeName, typeErr := optionalString(format, object, "type")
		if typeErr != nil {
			return nil, diagnostics, typeErr
		}
		if typeName != "function" {
			return nil, diagnostics, translationError(format, fmt.Sprintf("$.tools[%d].type", index), "unsupported_tool_type", "only function tools are representable")
		}
		function, functionErr := decodeObject(format, object["function"])
		if functionErr != nil {
			return nil, diagnostics, functionErr
		}
		delete(object, "function")
		tool := llmprotocol.ToolDefinition{}
		tool.Name, err = optionalString(format, function, "name")
		if err != nil {
			return nil, diagnostics, err
		}
		tool.Description, err = optionalString(format, function, "description")
		if err != nil {
			return nil, diagnostics, err
		}
		if params, ok := function["parameters"]; ok {
			tool.Parameters = cloneRaw(params)
			delete(function, "parameters")
		}
		tool.Strict, err = optionalBool(format, function, "strict")
		if err != nil {
			return nil, diagnostics, err
		}
		tool.Extensions, diagnostics, err = collectAndAppendExtensions(format, function, policy, diagnostics)
		if err != nil {
			return nil, diagnostics, err
		}
		_, diagnostics, err = collectAndAppendExtensions(format, object, policy, diagnostics)
		if err != nil {
			return nil, diagnostics, err
		}
		tools = append(tools, tool)
	}
	return tools, diagnostics, nil
}

func decodeChatToolChoice(raw json.RawMessage) (*llmprotocol.ToolChoice, error) {
	const format = llmprotocol.FormatOpenAIChat
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, nil
	}
	if trimmed[0] == '"' {
		value, err := decodeString(format, "$.tool_choice", raw)
		if err != nil {
			return nil, err
		}
		switch value {
		case "auto", "required", "none":
			return &llmprotocol.ToolChoice{Type: llmprotocol.ToolChoiceType(value)}, nil
		default:
			return &llmprotocol.ToolChoice{Type: llmprotocol.ToolChoiceRaw, Raw: cloneRaw(raw)}, nil
		}
	}
	var object struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &object); err != nil {
		return nil, translationError(format, "$.tool_choice", "invalid_tool_choice", "tool_choice must be a string or object")
	}
	if object.Type == "function" && object.Function.Name != "" {
		return &llmprotocol.ToolChoice{Type: llmprotocol.ToolChoiceTool, Name: object.Function.Name}, nil
	}
	return &llmprotocol.ToolChoice{Type: llmprotocol.ToolChoiceRaw, Raw: cloneRaw(raw)}, nil
}

func encodeChatToolChoice(choice llmprotocol.ToolChoice) (any, error) {
	switch choice.Type {
	case llmprotocol.ToolChoiceAuto, llmprotocol.ToolChoiceRequired, llmprotocol.ToolChoiceNone:
		return string(choice.Type), nil
	case llmprotocol.ToolChoiceTool:
		return map[string]any{"type": "function", "function": map[string]any{"name": choice.Name}}, nil
	case llmprotocol.ToolChoiceRaw:
		return json.RawMessage(cloneRaw(choice.Raw)), nil
	default:
		return nil, translationError(llmprotocol.FormatOpenAIChat, "$.tool_choice", "invalid_tool_choice", "unknown neutral tool choice")
	}
}

func encodeChatMessage(message llmprotocol.Message, policy llmprotocol.Policy) (map[string]any, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatOpenAIChat
	// Message.ID is canonical-IR only and is deliberately NOT emitted: the
	// OpenAI chat dialect has no message-level id in EITHER direction. On a
	// request it is an unknown field, and strict upstreams reject the whole
	// call rather than ignoring it:
	//   400 invalid_request_error
	//   "Extra inputs are not permitted, field: 'messages[1].id'"
	// On a response the id belongs to the completion object, not to
	// choices[].message (EncodeResponse sets it at the top level).
	//
	// Decoding still reads `id` back (decodeChatMessage) — dialects such as
	// Anthropic do carry one — so the canonical field stays meaningful and
	// callers may keep populating it for correlation.
	object := map[string]any{"role": string(message.Role)}
	var content []any
	var text strings.Builder
	var reasoning strings.Builder
	var refusal strings.Builder
	var toolCalls []any
	var toolResult *llmprotocol.ToolResult
	var diagnostics []llmprotocol.Diagnostic
	for index, block := range message.Content {
		path := fmt.Sprintf("$.messages[].content[%d]", index)
		switch block.Type {
		case llmprotocol.ContentText:
			text.WriteString(block.Text)
			content = append(content, map[string]any{"type": "text", "text": block.Text})
		case llmprotocol.ContentReasoning:
			reasoning.WriteString(block.Text)
			if block.Signature != "" {
				next, err := lossy(format, policy, path+".signature", "reasoning_signature_not_supported")
				diagnostics = append(diagnostics, next...)
				if err != nil {
					return nil, diagnostics, err
				}
			}
		case llmprotocol.ContentRefusal:
			refusal.WriteString(block.Text)
		case llmprotocol.ContentImage:
			if block.Source == nil || block.Source.URL == "" {
				return nil, diagnostics, translationError(format, path, "unsupported_image_source", "Chat Completions requires an image URL")
			}
			image := map[string]any{"url": block.Source.URL}
			if block.Source.Detail != "" {
				image["detail"] = block.Source.Detail
			}
			content = append(content, map[string]any{"type": "image_url", "image_url": image})
		case llmprotocol.ContentAudio:
			if block.Source == nil || block.Source.Data == "" {
				return nil, diagnostics, translationError(format, path, "unsupported_audio_source", "Chat Completions requires inline audio")
			}
			content = append(content, map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": block.Source.Data, "format": block.Source.MediaType}})
		case llmprotocol.ContentFile:
			if block.Source == nil {
				return nil, diagnostics, translationError(format, path, "missing_file_source", "file source is required")
			}
			file := map[string]any{}
			if block.Source.FileID != "" {
				file["file_id"] = block.Source.FileID
			}
			if block.Source.Data != "" {
				file["file_data"] = block.Source.Data
			}
			if block.Source.Filename != "" {
				file["filename"] = block.Source.Filename
			}
			content = append(content, map[string]any{"type": "file", "file": file})
		case llmprotocol.ContentToolCall:
			if block.ToolCall == nil {
				return nil, diagnostics, translationError(format, path, "missing_tool_call", "tool call is required")
			}
			toolCalls = append(toolCalls, map[string]any{"id": block.ToolCall.ID, "type": "function", "function": map[string]any{"name": block.ToolCall.Name, "arguments": string(rawOrEmptyObject(block.ToolCall.Arguments))}})
		case llmprotocol.ContentToolResult:
			if block.Result == nil {
				return nil, diagnostics, translationError(format, path, "missing_tool_result", "tool result is required")
			}
			if toolResult != nil {
				return nil, diagnostics, translationError(format, path, "multiple_tool_results", "one Chat tool message cannot contain multiple tool results")
			}
			toolResult = block.Result
		case llmprotocol.ContentUnknown:
			next, err := lossy(format, policy, path, "unknown_content_block")
			diagnostics = append(diagnostics, next...)
			if err != nil {
				return nil, diagnostics, err
			}
		default:
			next, err := lossy(format, policy, path, "unsupported_content_block")
			diagnostics = append(diagnostics, next...)
			if err != nil {
				return nil, diagnostics, err
			}
		}
	}
	if toolResult != nil {
		object["role"] = "tool"
		object["tool_call_id"] = toolResult.ToolCallID
		encoded, next, err := encodeChatResultContent(toolResult.Content, policy)
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return nil, diagnostics, err
		}
		object["content"] = encoded
	} else if len(content) == 1 {
		if part, ok := content[0].(map[string]any); ok && part["type"] == "text" {
			object["content"] = part["text"]
		} else {
			object["content"] = content
		}
	} else if len(content) > 0 {
		object["content"] = content
	} else if text.Len() > 0 {
		object["content"] = text.String()
	} else {
		object["content"] = nil
	}
	if reasoning.Len() > 0 {
		object["reasoning_content"] = reasoning.String()
	}
	if refusal.Len() > 0 {
		object["refusal"] = refusal.String()
	}
	if len(toolCalls) > 0 {
		object["tool_calls"] = toolCalls
	}
	mergeExtensions(object, message.Extensions)
	delete(object, "choice_index")
	return object, diagnostics, nil
}

func encodeChatResultContent(blocks []llmprotocol.ContentBlock, policy llmprotocol.Policy) (any, []llmprotocol.Diagnostic, error) {
	var text strings.Builder
	var diagnostics []llmprotocol.Diagnostic
	for index, block := range blocks {
		if block.Type == llmprotocol.ContentText {
			text.WriteString(block.Text)
			continue
		}
		next, err := lossy(llmprotocol.FormatOpenAIChat, policy, fmt.Sprintf("$.messages[].content[%d]", index), "non_text_tool_result")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return nil, diagnostics, err
		}
	}
	return text.String(), diagnostics, nil
}

func decodeOpenAIUsage(raw json.RawMessage, format llmprotocol.Format) (llmprotocol.Usage, error) {
	object, err := decodeObject(format, raw)
	if err != nil {
		return llmprotocol.Usage{}, err
	}
	usage := llmprotocol.Usage{}
	usage.InputTokens, err = firstOptionalInt(format, object, "input_tokens", "prompt_tokens")
	if err != nil {
		return usage, err
	}
	usage.OutputTokens, err = firstOptionalInt(format, object, "output_tokens", "completion_tokens")
	if err != nil {
		return usage, err
	}
	usage.TotalTokens, err = optionalInt(format, object, "total_tokens")
	if err != nil {
		return usage, err
	}
	if rawDetails, ok := object["prompt_tokens_details"]; ok {
		delete(object, "prompt_tokens_details")
		details, detailsErr := decodeObject(format, rawDetails)
		if detailsErr != nil {
			return usage, detailsErr
		}
		usage.CachedInputTokens, err = optionalInt(format, details, "cached_tokens")
		if err != nil {
			return usage, err
		}
		usage.CacheCreationTokens, err = firstOptionalInt(format, details, "cache_creation_tokens", "cache_write_tokens")
		if err != nil {
			return usage, err
		}
	}
	if rawDetails, ok := object["input_tokens_details"]; ok {
		delete(object, "input_tokens_details")
		details, detailsErr := decodeObject(format, rawDetails)
		if detailsErr != nil {
			return usage, detailsErr
		}
		usage.CachedInputTokens, err = optionalInt(format, details, "cached_tokens")
		if err != nil {
			return usage, err
		}
		usage.CacheCreationTokens, err = firstOptionalInt(format, details, "cache_creation_tokens", "cache_write_tokens")
		if err != nil {
			return usage, err
		}
	}
	for _, name := range []string{"completion_tokens_details", "output_tokens_details"} {
		if rawDetails, ok := object[name]; ok {
			delete(object, name)
			details, detailsErr := decodeObject(format, rawDetails)
			if detailsErr != nil {
				return usage, detailsErr
			}
			usage.ReasoningTokens, err = optionalInt(format, details, "reasoning_tokens")
			if err != nil {
				return usage, err
			}
			usage.AcceptedPredictionTokens, err = optionalInt(format, details, "accepted_prediction_tokens")
			if err != nil {
				return usage, err
			}
			usage.RejectedPredictionTokens, err = optionalInt(format, details, "rejected_prediction_tokens")
			if err != nil {
				return usage, err
			}
		}
	}
	return usage, nil
}

func encodeOpenAIUsage(usage llmprotocol.Usage, chat bool) map[string]any {
	object := map[string]any{}
	inputName, outputName := "input_tokens", "output_tokens"
	inputDetails, outputDetails := "input_tokens_details", "output_tokens_details"
	if chat {
		inputName, outputName = "prompt_tokens", "completion_tokens"
		inputDetails, outputDetails = "prompt_tokens_details", "completion_tokens_details"
	}
	putInt(object, inputName, usage.InputTokens)
	putInt(object, outputName, usage.OutputTokens)
	putInt(object, "total_tokens", usage.TotalTokens)
	if usage.CachedInputTokens != nil || usage.CacheCreationTokens != nil {
		details := map[string]any{}
		putInt(details, "cached_tokens", usage.CachedInputTokens)
		putInt(details, "cache_creation_tokens", usage.CacheCreationTokens)
		object[inputDetails] = details
	}
	if usage.ReasoningTokens != nil || usage.AcceptedPredictionTokens != nil || usage.RejectedPredictionTokens != nil {
		details := map[string]any{}
		putInt(details, "reasoning_tokens", usage.ReasoningTokens)
		putInt(details, "accepted_prediction_tokens", usage.AcceptedPredictionTokens)
		putInt(details, "rejected_prediction_tokens", usage.RejectedPredictionTokens)
		object[outputDetails] = details
	}
	return object
}

func firstOptionalInt(format llmprotocol.Format, object map[string]json.RawMessage, names ...string) (*int64, error) {
	for _, name := range names {
		if _, ok := object[name]; ok {
			value, err := optionalInt(format, object, name)
			for _, other := range names {
				delete(object, other)
			}
			return value, err
		}
	}
	return nil, nil
}

func putInt(object map[string]any, name string, value *int64) {
	if value != nil {
		object[name] = *value
	}
}

func hasUsage(usage llmprotocol.Usage) bool {
	return usage.InputTokens != nil || usage.OutputTokens != nil || usage.TotalTokens != nil || usage.CachedInputTokens != nil || usage.CacheCreationTokens != nil || usage.ReasoningTokens != nil || usage.ToolUsePromptTokens != nil || usage.AcceptedPredictionTokens != nil || usage.RejectedPredictionTokens != nil
}

func decodeChatStopReason(value string) llmprotocol.StopReason {
	switch value {
	case "stop":
		return llmprotocol.StopEndTurn
	case "length":
		return llmprotocol.StopMaxTokens
	case "tool_calls", "function_call":
		return llmprotocol.StopToolUse
	case "content_filter":
		return llmprotocol.StopContentFilter
	default:
		return llmprotocol.StopUnknown
	}
}

func encodeChatStopReason(value llmprotocol.StopReason) any {
	switch value {
	case llmprotocol.StopEndTurn:
		return "stop"
	case llmprotocol.StopMaxTokens:
		return "length"
	case llmprotocol.StopToolUse:
		return "tool_calls"
	case llmprotocol.StopContentFilter:
		return "content_filter"
	case "":
		return nil
	default:
		return "stop"
	}
}

func collectAndAppendExtensions(format llmprotocol.Format, object map[string]json.RawMessage, policy llmprotocol.Policy, diagnostics []llmprotocol.Diagnostic) (llmprotocol.Extensions, []llmprotocol.Diagnostic, error) {
	extensions, next, err := collectExtensions(format, object, policy)
	return extensions, append(diagnostics, next...), err
}

func decodeUnknownBlock(format llmprotocol.Format, raw json.RawMessage, policy llmprotocol.Policy, path string) (*llmprotocol.ContentBlock, []llmprotocol.Diagnostic, error) {
	switch policy.Effective().UnknownFields {
	case llmprotocol.UnknownReject:
		return nil, nil, translationError(format, path, "unknown_content_block", "content block type is not supported")
	case llmprotocol.UnknownDrop:
		return nil, []llmprotocol.Diagnostic{{Kind: llmprotocol.DiagnosticUnknownField, Path: path, Code: "content_block_dropped"}}, nil
	default:
		return &llmprotocol.ContentBlock{Type: llmprotocol.ContentUnknown, Provider: format, Raw: cloneRaw(raw)}, nil, nil
	}
}

func rawOrEmptyObject(value json.RawMessage) json.RawMessage {
	if len(bytes.TrimSpace(value)) == 0 {
		return json.RawMessage(`{}`)
	}
	return value
}

func chooseSourceKind(fileID, data string) string {
	if fileID != "" {
		return "provider_file"
	}
	if data != "" {
		return "base64"
	}
	return "unknown"
}
