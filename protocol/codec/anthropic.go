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

type Anthropic struct{}

func (Anthropic) Format() llmprotocol.Format { return llmprotocol.FormatAnthropic }

func (Anthropic) DecodeRequest(body json.RawMessage, policy llmprotocol.Policy) (RequestResult, error) {
	const format = llmprotocol.FormatAnthropic
	object, err := decodeObject(format, body)
	if err != nil {
		return RequestResult{}, err
	}
	request := llmprotocol.Request{Preservation: preserveBody(policy, format, body, true)}
	request.Model, err = optionalString(format, object, "model")
	if err != nil {
		return RequestResult{}, err
	}
	request.Output.MaxTokens, err = optionalInt(format, object, "max_tokens")
	if err != nil {
		return RequestResult{}, err
	}
	request.Output.StopSequences, err = optionalStrings(format, object, "stop_sequences", false)
	if err != nil {
		return RequestResult{}, err
	}
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
	request.Sampling.TopK, err = optionalInt(format, object, "top_k")
	if err != nil {
		return RequestResult{}, err
	}
	var diagnostics []llmprotocol.Diagnostic
	if raw, ok := object["system"]; ok {
		delete(object, "system")
		content, next, decodeErr := decodeAnthropicContent(raw, policy, "$.system")
		diagnostics = append(diagnostics, next...)
		if decodeErr != nil {
			return RequestResult{}, decodeErr
		}
		request.Instructions = append(request.Instructions, llmprotocol.Instruction{Role: llmprotocol.RoleSystem, Content: content})
	}
	if raw, ok := object["messages"]; ok {
		delete(object, "messages")
		request.Messages, diagnostics, err = decodeAnthropicMessages(raw, policy, diagnostics)
		if err != nil {
			return RequestResult{}, err
		}
	}
	if raw, ok := object["tools"]; ok {
		delete(object, "tools")
		request.Tools, diagnostics, err = decodeAnthropicTools(raw, policy, diagnostics)
		if err != nil {
			return RequestResult{}, err
		}
	}
	if raw, ok := object["tool_choice"]; ok {
		delete(object, "tool_choice")
		request.ToolChoice, err = decodeAnthropicToolChoice(raw)
		if err != nil {
			return RequestResult{}, err
		}
	}
	if raw, ok := object["thinking"]; ok {
		delete(object, "thinking")
		thinking, thinkingErr := decodeObject(format, raw)
		if thinkingErr != nil {
			return RequestResult{}, thinkingErr
		}
		typeName, typeErr := optionalString(format, thinking, "type")
		if typeErr != nil {
			return RequestResult{}, typeErr
		}
		request.Reasoning.BudgetTokens, err = optionalInt(format, thinking, "budget_tokens")
		if err != nil {
			return RequestResult{}, err
		}
		if typeName != "" && typeName != "enabled" {
			thinking["type"] = rawJSONString(typeName)
		}
		if len(thinking) != 0 {
			request.Reasoning.Provider = format
			request.Reasoning.Raw, _ = json.Marshal(thinking)
		}
	}
	if raw, ok := object["output_config"]; ok {
		delete(object, "output_config")
		config, configErr := decodeObject(format, raw)
		if configErr != nil {
			return RequestResult{}, configErr
		}
		request.Reasoning.Effort, err = optionalString(format, config, "effort")
		if err != nil {
			return RequestResult{}, err
		}
		if rawFormat, exists := config["format"]; exists {
			delete(config, "format")
			request.Output.ResponseFormat, err = decodeAnthropicResponseFormat(rawFormat)
			if err != nil {
				return RequestResult{}, err
			}
		}
		if len(config) != 0 {
			leftover, _ := json.Marshal(config)
			object["output_config"] = leftover
		}
	}
	request.Extensions, diagnostics, err = collectAndAppendExtensions(format, object, policy, diagnostics)
	if err != nil {
		return RequestResult{}, err
	}
	return RequestResult{Request: request, Diagnostics: diagnostics}, nil
}

func (Anthropic) EncodeRequest(request llmprotocol.Request, policy llmprotocol.Policy) (WireResult, error) {
	const format = llmprotocol.FormatAnthropic
	if raw, ok := preservedRequest(request, format, policy); ok {
		return WireResult{Body: raw}, nil
	}
	object := map[string]any{"model": request.Model}
	if request.Output.MaxTokens == nil {
		return WireResult{}, translationError(format, "$.max_tokens", "missing_max_tokens", "Anthropic Messages requires max_tokens")
	}
	object["max_tokens"] = *request.Output.MaxTokens
	if len(request.Output.StopSequences) > 0 {
		object["stop_sequences"] = request.Output.StopSequences
	}
	if request.Stream {
		object["stream"] = true
	}
	if request.Sampling.Temperature != nil {
		object["temperature"] = *request.Sampling.Temperature
	}
	if request.Sampling.TopP != nil {
		object["top_p"] = *request.Sampling.TopP
	}
	if request.Sampling.TopK != nil {
		object["top_k"] = *request.Sampling.TopK
	}
	var diagnostics []llmprotocol.Diagnostic
	if err := rejectOptionalInt(format, policy, "$.sampling.seed", "seed_not_supported", request.Sampling.Seed, &diagnostics); err != nil {
		return WireResult{}, err
	}
	if err := rejectNonZeroPenalty(format, policy, "$.sampling.frequency_penalty", request.Sampling.FrequencyPenalty, &diagnostics); err != nil {
		return WireResult{}, err
	}
	if err := rejectNonZeroPenalty(format, policy, "$.sampling.presence_penalty", request.Sampling.PresencePenalty, &diagnostics); err != nil {
		return WireResult{}, err
	}
	if err := validateNeutralOutputControls(format, request.Output, policy, &diagnostics); err != nil {
		return WireResult{}, err
	}
	if providerStateActive(request.State) {
		next, err := lossy(format, policy, "$.state", "provider_state_not_supported")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{}, err
		}
	}
	if len(request.Instructions) > 0 {
		var blocks []llmprotocol.ContentBlock
		for _, instruction := range request.Instructions {
			if instruction.Role != llmprotocol.RoleSystem && instruction.Role != llmprotocol.RoleDeveloper {
				return WireResult{}, translationError(format, "$.system", "invalid_instruction_role", "Anthropic system content accepts system or developer instructions")
			}
			blocks = append(blocks, instruction.Content...)
		}
		encoded, next, err := encodeAnthropicContent(blocks, policy)
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{}, err
		}
		if text, ok := anthropicSingleText(encoded); ok {
			object["system"] = text
		} else {
			object["system"] = encoded
		}
	}
	messages := make([]any, 0, len(request.Messages))
	for index, message := range request.Messages {
		if message.Role != llmprotocol.RoleUser && message.Role != llmprotocol.RoleAssistant && message.Role != llmprotocol.RoleTool {
			return WireResult{}, translationError(format, fmt.Sprintf("$.messages[%d].role", index), "unsupported_role", "Anthropic messages accept user or assistant roles")
		}
		role := message.Role
		if role == llmprotocol.RoleTool {
			role = llmprotocol.RoleUser
		}
		content, next, err := encodeAnthropicContent(message.Content, policy)
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{}, err
		}
		value := map[string]any{"role": string(role), "content": content}
		mergeExtensions(value, message.Extensions)
		messages = append(messages, value)
	}
	object["messages"] = messages
	if len(request.Tools) > 0 {
		tools := make([]any, 0, len(request.Tools))
		for _, tool := range request.Tools {
			if tool.Type != "" && tool.Type != "function" {
				return WireResult{}, translationError(format, "$.tools", "unsupported_tool_type", "Anthropic codec supports function tools only")
			}
			value := map[string]any{"name": tool.Name, "input_schema": rawOrEmptyObject(tool.Parameters)}
			if tool.Description != "" {
				value["description"] = tool.Description
			}
			if tool.Strict != nil {
				value["strict"] = *tool.Strict
			}
			mergeExtensions(value, tool.Extensions)
			tools = append(tools, value)
		}
		object["tools"] = tools
	}
	if request.ToolChoice != nil {
		choice, err := encodeAnthropicToolChoice(*request.ToolChoice)
		if err != nil {
			return WireResult{}, err
		}
		object["tool_choice"] = choice
	}
	if request.ParallelToolCalls != nil && !*request.ParallelToolCalls {
		choice, _ := object["tool_choice"].(map[string]any)
		if choice == nil {
			choice = map[string]any{"type": "auto"}
		}
		choice["disable_parallel_tool_use"] = true
		object["tool_choice"] = choice
	}
	outputConfig := map[string]any{}
	if len(request.Output.ResponseFormat) != 0 {
		value, err := encodeAnthropicResponseFormat(request.Output.ResponseFormat)
		if err != nil {
			return WireResult{}, err
		}
		outputConfig["format"] = value
	}
	if request.Reasoning.Effort != "" {
		outputConfig["effort"] = request.Reasoning.Effort
	}
	if len(outputConfig) != 0 {
		object["output_config"] = outputConfig
	}
	if request.Reasoning.Include != nil || request.Reasoning.Level != "" {
		next, err := lossy(format, policy, "$.reasoning", "reasoning_config_not_portable")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{}, err
		}
	}
	if request.Reasoning.BudgetTokens != nil || len(request.Reasoning.Raw) != 0 {
		thinking := map[string]any{"type": "enabled"}
		if len(request.Reasoning.Raw) != 0 {
			if request.Reasoning.Provider != "" && request.Reasoning.Provider != format {
				return WireResult{}, translationError(format, "$.reasoning.raw", "reasoning_config_not_portable", "provider reasoning config cannot cross formats")
			}
			_ = json.Unmarshal(request.Reasoning.Raw, &thinking)
		}
		putInt(thinking, "budget_tokens", request.Reasoning.BudgetTokens)
		object["thinking"] = thinking
	}
	mergeExtensions(object, request.Extensions)
	body, err := marshalObject(format, object)
	return WireResult{Body: body, Diagnostics: diagnostics}, err
}

func (Anthropic) DecodeResponse(body json.RawMessage, policy llmprotocol.Policy) (ResponseResult, error) {
	const format = llmprotocol.FormatAnthropic
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
	delete(object, "type")
	delete(object, "role")
	stop, err := optionalString(format, object, "stop_reason")
	if err != nil {
		return ResponseResult{}, err
	}
	delete(object, "stop_sequence")
	var diagnostics []llmprotocol.Diagnostic
	output := llmprotocol.ResponseOutput{Role: llmprotocol.RoleAssistant, StopReason: decodeAnthropicStopReason(stop)}
	if raw, ok := object["content"]; ok {
		delete(object, "content")
		output.Content, diagnostics, err = decodeAnthropicContent(raw, policy, "$.content")
		if err != nil {
			return ResponseResult{}, err
		}
	}
	response.Outputs = []llmprotocol.ResponseOutput{output}
	if raw, ok := object["usage"]; ok {
		delete(object, "usage")
		response.Usage, err = decodeAnthropicUsage(raw)
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

func (Anthropic) EncodeResponse(response llmprotocol.Response, policy llmprotocol.Policy) (WireResult, error) {
	const format = llmprotocol.FormatAnthropic
	if raw, ok := preservedResponse(response, format, policy); ok {
		return WireResult{Body: raw}, nil
	}
	if len(response.Outputs) == 0 {
		return WireResult{}, translationError(format, "$.content", "missing_output", "Anthropic response requires one assistant output")
	}
	if len(response.Outputs) > 1 {
		return WireResult{}, translationError(format, "$.content", "multiple_outputs", "Anthropic response cannot represent multiple choices")
	}
	output := response.Outputs[0]
	var diagnostics []llmprotocol.Diagnostic
	if len(output.Logprobs) > 0 {
		next, err := lossy(format, policy, "$.outputs[].logprobs", "logprobs_not_supported")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{Diagnostics: diagnostics}, err
		}
	}
	content, next, err := encodeAnthropicContent(output.Content, policy)
	diagnostics = append(diagnostics, next...)
	if err != nil {
		return WireResult{}, err
	}
	object := map[string]any{"id": response.ID, "type": "message", "role": "assistant", "model": response.Model, "content": content, "stop_reason": encodeAnthropicStopReason(output.StopReason), "stop_sequence": nil}
	if hasUsage(response.Usage) {
		object["usage"] = encodeAnthropicUsage(response.Usage)
	}
	mergeExtensions(object, response.Extensions)
	body, err := marshalObject(format, object)
	return WireResult{Body: body, Diagnostics: diagnostics}, err
}

func decodeAnthropicMessages(raw json.RawMessage, policy llmprotocol.Policy, diagnostics []llmprotocol.Diagnostic) ([]llmprotocol.Message, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatAnthropic
	values, err := decodeArray(format, "$.messages", raw)
	if err != nil {
		return nil, diagnostics, err
	}
	messages := make([]llmprotocol.Message, 0, len(values))
	for index, value := range values {
		path := fmt.Sprintf("$.messages[%d]", index)
		object, objectErr := decodeObject(format, value)
		if objectErr != nil {
			return nil, diagnostics, objectErr
		}
		roleName, roleErr := optionalString(format, object, "role")
		if roleErr != nil {
			return nil, diagnostics, roleErr
		}
		role, roleErr := validateRole(format, path+".role", roleName)
		if roleErr != nil || (role != llmprotocol.RoleUser && role != llmprotocol.RoleAssistant) {
			return nil, diagnostics, translationError(format, path+".role", "unsupported_role", "Anthropic messages accept user or assistant roles")
		}
		contentRaw := object["content"]
		delete(object, "content")
		content, next, contentErr := decodeAnthropicContent(contentRaw, policy, path+".content")
		diagnostics = append(diagnostics, next...)
		if contentErr != nil {
			return nil, diagnostics, contentErr
		}
		extensions, next, extensionErr := collectExtensions(format, object, policy)
		diagnostics = append(diagnostics, next...)
		if extensionErr != nil {
			return nil, diagnostics, extensionErr
		}
		messages = append(messages, llmprotocol.Message{Role: role, Content: content, Extensions: extensions})
	}
	return messages, diagnostics, nil
}

func decodeAnthropicContent(raw json.RawMessage, policy llmprotocol.Policy, path string) ([]llmprotocol.ContentBlock, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatAnthropic
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, nil, nil
	}
	if trimmed[0] == '"' {
		text, err := decodeString(format, path, raw)
		return []llmprotocol.ContentBlock{llmprotocol.Text(text)}, nil, err
	}
	values, err := decodeArray(format, path, raw)
	if err != nil {
		return nil, nil, err
	}
	blocks := make([]llmprotocol.ContentBlock, 0, len(values))
	var diagnostics []llmprotocol.Diagnostic
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
		var block *llmprotocol.ContentBlock
		switch typeName {
		case "text":
			text, textErr := optionalString(format, object, "text")
			if textErr != nil {
				return nil, diagnostics, textErr
			}
			block = &llmprotocol.ContentBlock{Type: llmprotocol.ContentText, Text: text, Provider: format}
		case "thinking":
			text, textErr := optionalString(format, object, "thinking")
			if textErr != nil {
				return nil, diagnostics, textErr
			}
			signature, signatureErr := optionalString(format, object, "signature")
			if signatureErr != nil {
				return nil, diagnostics, signatureErr
			}
			block = &llmprotocol.ContentBlock{Type: llmprotocol.ContentReasoning, Text: text, Signature: signature, Provider: format}
		case "redacted_thinking":
			data, dataErr := optionalString(format, object, "data")
			if dataErr != nil {
				return nil, diagnostics, dataErr
			}
			block = &llmprotocol.ContentBlock{Type: llmprotocol.ContentReasoning, Signature: data, Provider: format}
		case "tool_use":
			id, idErr := optionalString(format, object, "id")
			if idErr != nil {
				return nil, diagnostics, idErr
			}
			name, nameErr := optionalString(format, object, "name")
			if nameErr != nil {
				return nil, diagnostics, nameErr
			}
			input := cloneRaw(object["input"])
			delete(object, "input")
			block = &llmprotocol.ContentBlock{Type: llmprotocol.ContentToolCall, Provider: format, ToolCall: &llmprotocol.ToolCall{ID: id, Name: name, Arguments: rawOrEmptyObject(input)}}
		case "tool_result":
			callID, callErr := optionalString(format, object, "tool_use_id")
			if callErr != nil {
				return nil, diagnostics, callErr
			}
			isError, boolErr := optionalBool(format, object, "is_error")
			if boolErr != nil {
				return nil, diagnostics, boolErr
			}
			contentRaw := object["content"]
			delete(object, "content")
			content, next, contentErr := decodeAnthropicContent(contentRaw, policy, blockPath+".content")
			diagnostics = append(diagnostics, next...)
			if contentErr != nil {
				return nil, diagnostics, contentErr
			}
			block = &llmprotocol.ContentBlock{Type: llmprotocol.ContentToolResult, Provider: format, Result: &llmprotocol.ToolResult{ToolCallID: callID, Content: content, IsError: isError}}
		case "image", "document":
			source, sourceErr := decodeAnthropicSource(object["source"], blockPath+".source")
			if sourceErr != nil {
				return nil, diagnostics, sourceErr
			}
			contentType := llmprotocol.ContentImage
			if typeName == "document" {
				contentType = llmprotocol.ContentFile
			}
			delete(object, "source")
			block = &llmprotocol.ContentBlock{Type: contentType, Source: &source, Provider: format}
		default:
			unknown, next, unknownErr := decodeUnknownBlock(format, value, policy, blockPath)
			diagnostics = append(diagnostics, next...)
			if unknownErr != nil {
				return nil, diagnostics, unknownErr
			}
			if unknown != nil {
				blocks = append(blocks, *unknown)
			}
			continue
		}
		if block != nil {
			block.Extensions, diagnostics, err = collectAndAppendExtensions(format, object, policy, diagnostics)
			if err != nil {
				return nil, diagnostics, err
			}
			blocks = append(blocks, *block)
		}
	}
	return blocks, diagnostics, nil
}

func decodeAnthropicSource(raw json.RawMessage, path string) (llmprotocol.Source, error) {
	const format = llmprotocol.FormatAnthropic
	object, err := decodeObject(format, raw)
	if err != nil {
		return llmprotocol.Source{}, err
	}
	typeName, err := optionalString(format, object, "type")
	if err != nil {
		return llmprotocol.Source{}, err
	}
	source := llmprotocol.Source{Kind: typeName}
	switch typeName {
	case "base64":
		source.MediaType, err = optionalString(format, object, "media_type")
		if err == nil {
			source.Data, err = optionalString(format, object, "data")
		}
	case "url":
		source.URL, err = optionalString(format, object, "url")
	case "file":
		source.FileID, err = optionalString(format, object, "file_id")
		source.Kind = "provider_file"
	default:
		source.Raw = cloneRaw(raw)
	}
	if err != nil {
		return llmprotocol.Source{}, err
	}
	_ = path
	return source, nil
}

func encodeAnthropicContent(blocks []llmprotocol.ContentBlock, policy llmprotocol.Policy) ([]any, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatAnthropic
	parts := make([]any, 0, len(blocks))
	var diagnostics []llmprotocol.Diagnostic
	for index, block := range blocks {
		path := fmt.Sprintf("$.content[%d]", index)
		switch block.Type {
		case llmprotocol.ContentText:
			value := map[string]any{"type": "text", "text": block.Text}
			mergeExtensions(value, block.Extensions)
			parts = append(parts, value)
		case llmprotocol.ContentReasoning:
			if block.Text == "" && block.Signature != "" {
				value := map[string]any{"type": "redacted_thinking", "data": block.Signature}
				mergeExtensions(value, block.Extensions)
				parts = append(parts, value)
			} else {
				value := map[string]any{"type": "thinking", "thinking": block.Text}
				if block.Signature != "" {
					value["signature"] = block.Signature
				}
				mergeExtensions(value, block.Extensions)
				parts = append(parts, value)
			}
		case llmprotocol.ContentImage, llmprotocol.ContentFile:
			if block.Source == nil {
				return nil, diagnostics, translationError(format, path, "missing_media_source", "media source is required")
			}
			typeName := "image"
			if block.Type == llmprotocol.ContentFile {
				typeName = "document"
			}
			source, err := encodeAnthropicSource(*block.Source)
			if err != nil {
				return nil, diagnostics, err
			}
			value := map[string]any{"type": typeName, "source": source}
			mergeExtensions(value, block.Extensions)
			parts = append(parts, value)
		case llmprotocol.ContentToolCall:
			if block.ToolCall == nil {
				return nil, diagnostics, translationError(format, path, "missing_tool_call", "tool call is required")
			}
			value := map[string]any{"type": "tool_use", "id": block.ToolCall.ID, "name": block.ToolCall.Name, "input": rawOrEmptyObject(block.ToolCall.Arguments)}
			mergeExtensions(value, block.Extensions)
			parts = append(parts, value)
		case llmprotocol.ContentToolResult:
			if block.Result == nil {
				return nil, diagnostics, translationError(format, path, "missing_tool_result", "tool result is required")
			}
			content, next, err := encodeAnthropicContent(block.Result.Content, policy)
			diagnostics = append(diagnostics, next...)
			if err != nil {
				return nil, diagnostics, err
			}
			value := map[string]any{"type": "tool_result", "tool_use_id": block.Result.ToolCallID, "content": content}
			if block.Result.IsError != nil {
				value["is_error"] = *block.Result.IsError
			}
			mergeExtensions(value, block.Extensions)
			parts = append(parts, value)
		case llmprotocol.ContentRefusal:
			next, err := lossy(format, policy, path, "refusal_not_distinct")
			diagnostics = append(diagnostics, next...)
			if err != nil {
				return nil, diagnostics, err
			}
			value := map[string]any{"type": "text", "text": block.Text}
			mergeExtensions(value, block.Extensions)
			parts = append(parts, value)
		case llmprotocol.ContentUnknown:
			if block.Provider == format && len(block.Raw) != 0 {
				parts = append(parts, json.RawMessage(cloneRaw(block.Raw)))
				continue
			}
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
	return parts, diagnostics, nil
}

func encodeAnthropicSource(source llmprotocol.Source) (map[string]any, error) {
	switch source.Kind {
	case "base64":
		return map[string]any{"type": "base64", "media_type": source.MediaType, "data": source.Data}, nil
	case "url":
		return map[string]any{"type": "url", "url": source.URL}, nil
	case "provider_file":
		return map[string]any{"type": "file", "file_id": source.FileID}, nil
	default:
		return nil, translationError(llmprotocol.FormatAnthropic, "$.content[].source", "unsupported_media_source", "source cannot be represented by Anthropic")
	}
}

func decodeAnthropicTools(raw json.RawMessage, policy llmprotocol.Policy, diagnostics []llmprotocol.Diagnostic) ([]llmprotocol.ToolDefinition, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatAnthropic
	values, err := decodeArray(format, "$.tools", raw)
	if err != nil {
		return nil, diagnostics, err
	}
	tools := make([]llmprotocol.ToolDefinition, 0, len(values))
	for _, value := range values {
		object, objectErr := decodeObject(format, value)
		if objectErr != nil {
			return nil, diagnostics, objectErr
		}
		tool := llmprotocol.ToolDefinition{Type: "function"}
		tool.Name, err = optionalString(format, object, "name")
		if err != nil {
			return nil, diagnostics, err
		}
		tool.Description, err = optionalString(format, object, "description")
		if err != nil {
			return nil, diagnostics, err
		}
		if rawSchema, ok := object["input_schema"]; ok {
			tool.Parameters = cloneRaw(rawSchema)
			delete(object, "input_schema")
		}
		tool.Strict, err = optionalBool(format, object, "strict")
		if err != nil {
			return nil, diagnostics, err
		}
		tool.Extensions, diagnostics, err = collectAndAppendExtensions(format, object, policy, diagnostics)
		if err != nil {
			return nil, diagnostics, err
		}
		tools = append(tools, tool)
	}
	return tools, diagnostics, nil
}

func decodeAnthropicResponseFormat(raw json.RawMessage) (json.RawMessage, error) {
	const format = llmprotocol.FormatAnthropic
	object, err := decodeObject(format, raw)
	if err != nil {
		return nil, err
	}
	typeName, err := optionalString(format, object, "type")
	if err != nil {
		return nil, err
	}
	if typeName != "json_schema" {
		return nil, translationError(format, "$.output_config.format.type", "unsupported_response_format", "Anthropic output format must be json_schema")
	}
	schema, ok := object["schema"]
	if !ok || !json.Valid(schema) {
		return nil, translationError(format, "$.output_config.format.schema", "missing_schema", "JSON-schema output format requires a schema")
	}
	delete(object, "schema")
	if len(object) != 0 {
		return nil, translationError(format, "$.output_config.format", "unsupported_response_format", "output format contains unsupported fields")
	}
	strict := true
	return encodeNeutralJSONSchema(neutralResponseFormat{Type: "json_schema", Schema: schema, Strict: &strict}), nil
}

func encodeAnthropicResponseFormat(raw json.RawMessage) (map[string]any, error) {
	const format = llmprotocol.FormatAnthropic
	value, err := decodeNeutralResponseFormat(format, "$.output.response_format", raw)
	if err != nil {
		return nil, err
	}
	switch value.Type {
	case "json_object":
		return map[string]any{"type": "json_schema", "schema": map[string]any{"type": "object"}}, nil
	case "json_schema":
		if len(value.Schema) == 0 {
			return nil, translationError(format, "$.output.response_format.schema", "missing_schema", "JSON-schema response format requires a schema")
		}
		return map[string]any{"type": "json_schema", "schema": json.RawMessage(cloneRaw(value.Schema))}, nil
	default:
		return nil, translationError(format, "$.output.response_format", "unsupported_response_format", "response format cannot be represented by Anthropic")
	}
}

func decodeAnthropicToolChoice(raw json.RawMessage) (*llmprotocol.ToolChoice, error) {
	const format = llmprotocol.FormatAnthropic
	object, err := decodeObject(format, raw)
	if err != nil {
		return nil, err
	}
	typeName, err := optionalString(format, object, "type")
	if err != nil {
		return nil, err
	}
	switch typeName {
	case "auto":
		return &llmprotocol.ToolChoice{Type: llmprotocol.ToolChoiceAuto}, nil
	case "any":
		return &llmprotocol.ToolChoice{Type: llmprotocol.ToolChoiceRequired}, nil
	case "none":
		return &llmprotocol.ToolChoice{Type: llmprotocol.ToolChoiceNone}, nil
	case "tool":
		name, nameErr := optionalString(format, object, "name")
		return &llmprotocol.ToolChoice{Type: llmprotocol.ToolChoiceTool, Name: name}, nameErr
	default:
		return &llmprotocol.ToolChoice{Type: llmprotocol.ToolChoiceRaw, Raw: cloneRaw(raw)}, nil
	}
}

func encodeAnthropicToolChoice(choice llmprotocol.ToolChoice) (map[string]any, error) {
	switch choice.Type {
	case llmprotocol.ToolChoiceAuto:
		return map[string]any{"type": "auto"}, nil
	case llmprotocol.ToolChoiceRequired:
		return map[string]any{"type": "any"}, nil
	case llmprotocol.ToolChoiceNone:
		return map[string]any{"type": "none"}, nil
	case llmprotocol.ToolChoiceTool:
		return map[string]any{"type": "tool", "name": choice.Name}, nil
	case llmprotocol.ToolChoiceRaw:
		var value map[string]any
		if err := json.Unmarshal(choice.Raw, &value); err != nil {
			return nil, translationError(llmprotocol.FormatAnthropic, "$.tool_choice", "invalid_tool_choice", "raw tool choice is invalid")
		}
		return value, nil
	default:
		return nil, translationError(llmprotocol.FormatAnthropic, "$.tool_choice", "invalid_tool_choice", "unknown neutral tool choice")
	}
}

func decodeAnthropicUsage(raw json.RawMessage) (llmprotocol.Usage, error) {
	const format = llmprotocol.FormatAnthropic
	object, err := decodeObject(format, raw)
	if err != nil {
		return llmprotocol.Usage{}, err
	}
	usage := llmprotocol.Usage{}
	uncachedInput, err := optionalInt(format, object, "input_tokens")
	if err != nil {
		return usage, err
	}
	usage.OutputTokens, err = optionalInt(format, object, "output_tokens")
	if err != nil {
		return usage, err
	}
	usage.CachedInputTokens, err = optionalInt(format, object, "cache_read_input_tokens")
	if err != nil {
		return usage, err
	}
	usage.CacheCreationTokens, err = optionalInt(format, object, "cache_creation_input_tokens")
	if err != nil {
		return usage, err
	}
	usage.InputTokens = sumUsage(uncachedInput, usage.CachedInputTokens, usage.CacheCreationTokens)
	usage.TotalTokens = sumUsage(usage.InputTokens, usage.OutputTokens)
	return usage, nil
}

func encodeAnthropicUsage(usage llmprotocol.Usage) map[string]any {
	object := map[string]any{}
	putInt(object, "input_tokens", subtractUsage(usage.InputTokens, usage.CachedInputTokens, usage.CacheCreationTokens))
	putInt(object, "output_tokens", usage.OutputTokens)
	putInt(object, "cache_read_input_tokens", usage.CachedInputTokens)
	putInt(object, "cache_creation_input_tokens", usage.CacheCreationTokens)
	return object
}

func subtractUsage(total *int64, components ...*int64) *int64 {
	if total == nil {
		return nil
	}
	value := *total
	for _, component := range components {
		if component != nil {
			value -= *component
		}
	}
	if value < 0 {
		value = 0
	}
	return &value
}

func sumUsage(values ...*int64) *int64 {
	var total int64
	found := false
	for _, value := range values {
		if value != nil {
			total += *value
			found = true
		}
	}
	if !found {
		return nil
	}
	return &total
}

func decodeAnthropicStopReason(value string) llmprotocol.StopReason {
	switch value {
	case "end_turn", "stop_sequence", "pause_turn":
		return llmprotocol.StopEndTurn
	case "max_tokens", "model_context_window_exceeded":
		return llmprotocol.StopMaxTokens
	case "tool_use":
		return llmprotocol.StopToolUse
	case "refusal":
		return llmprotocol.StopContentFilter
	case "":
		return ""
	default:
		return llmprotocol.StopUnknown
	}
}

func encodeAnthropicStopReason(value llmprotocol.StopReason) any {
	switch value {
	case llmprotocol.StopEndTurn:
		return "end_turn"
	case llmprotocol.StopMaxTokens:
		return "max_tokens"
	case llmprotocol.StopToolUse:
		return "tool_use"
	case llmprotocol.StopContentFilter:
		return "refusal"
	case "":
		return nil
	default:
		return "end_turn"
	}
}

func anthropicSingleText(parts []any) (string, bool) {
	if len(parts) != 1 {
		return "", false
	}
	value, ok := parts[0].(map[string]any)
	if !ok || value["type"] != "text" {
		return "", false
	}
	text, ok := value["text"].(string)
	return text, ok
}

func flattenText(blocks []llmprotocol.ContentBlock) string {
	var text strings.Builder
	for _, block := range blocks {
		if block.Type == llmprotocol.ContentText {
			text.WriteString(block.Text)
		}
	}
	return text.String()
}
