// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package codec

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

type OpenAIResponses struct{}

func (OpenAIResponses) Format() llmprotocol.Format { return llmprotocol.FormatOpenAIResponses }

func (OpenAIResponses) DecodeRequest(body json.RawMessage, policy llmprotocol.Policy) (RequestResult, error) {
	const format = llmprotocol.FormatOpenAIResponses
	object, err := decodeObject(format, body)
	if err != nil {
		return RequestResult{}, err
	}
	request := llmprotocol.Request{Preservation: preserveBody(policy, format, body, true)}
	request.Model, err = optionalString(format, object, "model")
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
	request.Output.MaxTokens, err = optionalInt(format, object, "max_output_tokens")
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
	request.State.Background, err = optionalBool(format, object, "background")
	if err != nil {
		return RequestResult{}, err
	}
	request.State.PreviousResponseID, err = optionalString(format, object, "previous_response_id")
	if err != nil {
		return RequestResult{}, err
	}
	if raw, ok := object["conversation"]; ok {
		delete(object, "conversation")
		if !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
			request.State.Conversation = cloneRaw(raw)
		}
	}
	var diagnostics []llmprotocol.Diagnostic
	if raw, ok := object["instructions"]; ok && !bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		delete(object, "instructions")
		content, next, decodeErr := decodeResponsesContent(raw, policy, "$.instructions", llmprotocol.RoleDeveloper)
		diagnostics = append(diagnostics, next...)
		if decodeErr != nil {
			return RequestResult{}, decodeErr
		}
		request.Instructions = append(request.Instructions, llmprotocol.Instruction{Role: llmprotocol.RoleDeveloper, Content: content})
	}
	if raw, ok := object["input"]; ok {
		delete(object, "input")
		messages, next, decodeErr := decodeResponsesInput(raw, policy)
		diagnostics = append(diagnostics, next...)
		if decodeErr != nil {
			return RequestResult{}, decodeErr
		}
		request.Messages = messages
	}
	if raw, ok := object["tools"]; ok {
		delete(object, "tools")
		request.Tools, diagnostics, err = decodeResponsesTools(raw, policy, diagnostics)
		if err != nil {
			return RequestResult{}, err
		}
	}
	if raw, ok := object["tool_choice"]; ok {
		delete(object, "tool_choice")
		request.ToolChoice, err = decodeResponsesToolChoice(raw)
		if err != nil {
			return RequestResult{}, err
		}
	}
	if raw, ok := object["text"]; ok {
		delete(object, "text")
		textObject, objectErr := decodeObject(format, raw)
		if objectErr != nil {
			return RequestResult{}, objectErr
		}
		if responseFormat, exists := textObject["format"]; exists {
			request.Output.ResponseFormat = responsesResponseFormat(responseFormat)
			delete(textObject, "format")
		}
		if len(textObject) > 0 {
			reencoded, _ := json.Marshal(textObject)
			if request.Extensions == nil {
				request.Extensions = llmprotocol.Extensions{}
			}
			request.Extensions["text"] = reencoded
		}
	}
	if raw, ok := object["reasoning"]; ok {
		delete(object, "reasoning")
		reasoningObject, objectErr := decodeObject(format, raw)
		if objectErr != nil {
			return RequestResult{}, objectErr
		}
		request.Reasoning.Effort, err = optionalString(format, reasoningObject, "effort")
		if err != nil {
			return RequestResult{}, err
		}
		if len(reasoningObject) != 0 {
			request.Reasoning.Provider = format
			request.Reasoning.Raw, _ = json.Marshal(reasoningObject)
		}
	}
	extensions, next, collectErr := collectExtensions(format, object, policy)
	diagnostics = append(diagnostics, next...)
	if collectErr != nil {
		return RequestResult{}, collectErr
	}
	if request.Extensions == nil {
		request.Extensions = extensions
	} else {
		for key, value := range extensions {
			request.Extensions[key] = value
		}
	}
	return RequestResult{Request: request, Diagnostics: diagnostics}, nil
}

func responsesResponseFormat(raw json.RawMessage) json.RawMessage {
	var format map[string]json.RawMessage
	if json.Unmarshal(raw, &format) != nil {
		return cloneRaw(raw)
	}
	var typeName string
	_ = json.Unmarshal(format["type"], &typeName)
	if typeName != "json_schema" {
		return cloneRaw(raw)
	}
	var nested map[string]json.RawMessage
	if json.Unmarshal(format["json_schema"], &nested) != nil || nested == nil {
		return cloneRaw(raw)
	}
	flattened := map[string]json.RawMessage{"type": rawJSONString("json_schema")}
	for _, name := range []string{"name", "description", "schema", "strict"} {
		if value, ok := nested[name]; ok {
			flattened[name] = value
		}
	}
	encoded, _ := json.Marshal(flattened)
	return encoded
}

func (OpenAIResponses) EncodeRequest(request llmprotocol.Request, policy llmprotocol.Policy) (WireResult, error) {
	const format = llmprotocol.FormatOpenAIResponses
	if raw, ok := preservedRequest(request, format, policy); ok {
		return WireResult{Body: raw}, nil
	}
	object := map[string]any{"model": request.Model}
	if request.Stream {
		object["stream"] = true
	}
	if request.Sampling.Temperature != nil {
		object["temperature"] = *request.Sampling.Temperature
	}
	if request.Sampling.TopP != nil {
		object["top_p"] = *request.Sampling.TopP
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
	if len(request.Output.StopSequences) > 0 {
		next, err := lossy(format, policy, "$.output.stop_sequences", "stop_sequences_not_supported")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{}, err
		}
	}
	if request.Sampling.TopK != nil {
		next, err := lossy(format, policy, "$.top_k", "top_k_not_supported")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{}, err
		}
	}
	if request.Output.MaxTokens != nil {
		object["max_output_tokens"] = *request.Output.MaxTokens
	}
	if request.ParallelToolCalls != nil {
		object["parallel_tool_calls"] = *request.ParallelToolCalls
	}
	if request.State.Store != nil {
		object["store"] = *request.State.Store
	}
	if request.State.Background != nil {
		object["background"] = *request.State.Background
	}
	if request.State.PreviousResponseID != "" {
		object["previous_response_id"] = request.State.PreviousResponseID
	}
	if len(request.State.Conversation) != 0 {
		object["conversation"] = json.RawMessage(cloneRaw(request.State.Conversation))
	}
	if len(request.Instructions) > 0 {
		var parts []any
		for _, instruction := range request.Instructions {
			if instruction.Role != llmprotocol.RoleSystem && instruction.Role != llmprotocol.RoleDeveloper {
				return WireResult{}, translationError(format, "$.instructions", "invalid_instruction_role", "instructions must use system or developer authority")
			}
			encoded, next, err := encodeResponsesContent(instruction.Content, policy, true)
			diagnostics = append(diagnostics, next...)
			if err != nil {
				return WireResult{}, err
			}
			parts = append(parts, encoded...)
		}
		if text, ok := singleTextPart(parts); ok {
			object["instructions"] = text
		} else {
			object["instructions"] = parts
		}
	}
	input := make([]any, 0, len(request.Messages))
	for index, message := range request.Messages {
		items, next, err := encodeResponsesMessage(message, policy, index)
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{}, err
		}
		input = append(input, items...)
	}
	object["input"] = input
	if len(request.Tools) > 0 {
		tools := make([]any, 0, len(request.Tools))
		for _, tool := range request.Tools {
			if tool.Type != "" && tool.Type != "function" {
				if len(tool.Raw) != 0 {
					tools = append(tools, json.RawMessage(cloneRaw(tool.Raw)))
					continue
				}
				return WireResult{}, translationError(format, "$.tools", "unsupported_tool_type", "non-function tool requires a preserved Responses definition")
			}
			value := map[string]any{"type": "function", "name": tool.Name, "parameters": rawOrEmptyObject(tool.Parameters)}
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
		value, err := encodeResponsesToolChoice(*request.ToolChoice)
		if err != nil {
			return WireResult{}, err
		}
		object["tool_choice"] = value
	}
	if len(request.Output.ResponseFormat) != 0 {
		object["text"] = map[string]any{"format": json.RawMessage(responsesResponseFormat(request.Output.ResponseFormat))}
	}
	if request.Reasoning.Effort != "" || request.Reasoning.BudgetTokens != nil || request.Reasoning.Include != nil || request.Reasoning.Level != "" || len(request.Reasoning.Raw) != 0 {
		if request.Reasoning.BudgetTokens != nil || request.Reasoning.Include != nil || request.Reasoning.Level != "" {
			next, err := lossy(format, policy, "$.reasoning", "reasoning_config_not_portable")
			diagnostics = append(diagnostics, next...)
			if err != nil {
				return WireResult{}, err
			}
		}
		reasoning := map[string]any{}
		if len(request.Reasoning.Raw) != 0 {
			if request.Reasoning.Provider != "" && request.Reasoning.Provider != format {
				return WireResult{}, translationError(format, "$.reasoning.raw", "reasoning_config_not_portable", "provider reasoning config cannot cross formats")
			}
			_ = json.Unmarshal(request.Reasoning.Raw, &reasoning)
		}
		if request.Reasoning.Effort != "" {
			reasoning["effort"] = request.Reasoning.Effort
		}
		object["reasoning"] = reasoning
	}
	mergeExtensions(object, request.Extensions)
	if rawText, exists := request.Extensions["text"]; exists {
		var preserved map[string]any
		if json.Unmarshal(rawText, &preserved) == nil {
			text, _ := object["text"].(map[string]any)
			if text == nil {
				text = map[string]any{}
			}
			for key, value := range preserved {
				if _, exists := text[key]; !exists {
					text[key] = value
				}
			}
			object["text"] = text
		}
	}
	body, err := marshalObject(format, object)
	return WireResult{Body: body, Diagnostics: diagnostics}, err
}

func (OpenAIResponses) DecodeResponse(body json.RawMessage, policy llmprotocol.Policy) (ResponseResult, error) {
	const format = llmprotocol.FormatOpenAIResponses
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
	status, err := optionalString(format, object, "status")
	if err != nil {
		return ResponseResult{}, err
	}
	var diagnostics []llmprotocol.Diagnostic
	if raw, ok := object["output"]; ok {
		delete(object, "output")
		response.Outputs, diagnostics, err = decodeResponsesOutput(raw, status, policy, diagnostics)
		if err != nil {
			return ResponseResult{}, err
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

func (OpenAIResponses) EncodeResponse(response llmprotocol.Response, policy llmprotocol.Policy) (WireResult, error) {
	const format = llmprotocol.FormatOpenAIResponses
	if raw, ok := preservedResponse(response, format, policy); ok {
		return WireResult{Body: raw}, nil
	}
	object := map[string]any{"id": response.ID, "object": "response", "model": response.Model, "status": "completed"}
	var output []any
	var diagnostics []llmprotocol.Diagnostic
	for outputIndex, source := range response.Outputs {
		if len(source.Logprobs) > 0 {
			next, err := lossy(format, policy, "$.outputs[].logprobs", "logprobs_not_supported")
			diagnostics = append(diagnostics, next...)
			if err != nil {
				return WireResult{Diagnostics: diagnostics}, err
			}
		}
		items, next, err := encodeResponsesOutput(source, policy, outputIndex)
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{}, err
		}
		output = append(output, items...)
		if source.StopReason == llmprotocol.StopMaxTokens {
			object["status"] = "incomplete"
			object["incomplete_details"] = map[string]any{"reason": "max_output_tokens"}
		}
	}
	object["output"] = output
	if hasUsage(response.Usage) {
		object["usage"] = encodeOpenAIUsage(response.Usage, false)
	}
	mergeExtensions(object, response.Extensions)
	body, err := marshalObject(format, object)
	return WireResult{Body: body, Diagnostics: diagnostics}, err
}

func decodeResponsesInput(raw json.RawMessage, policy llmprotocol.Policy) ([]llmprotocol.Message, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatOpenAIResponses
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, nil, nil
	}
	if trimmed[0] == '"' {
		text, err := decodeString(format, "$.input", raw)
		return []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.ContentBlock{llmprotocol.Text(text)}}}, nil, err
	}
	values, err := decodeArray(format, "$.input", raw)
	if err != nil {
		return nil, nil, err
	}
	messages := make([]llmprotocol.Message, 0, len(values))
	var diagnostics []llmprotocol.Diagnostic
	for index, value := range values {
		path := fmt.Sprintf("$.input[%d]", index)
		object, objectErr := decodeObject(format, value)
		if objectErr != nil {
			return nil, diagnostics, objectErr
		}
		typeName, typeErr := optionalString(format, object, "type")
		if typeErr != nil {
			return nil, diagnostics, typeErr
		}
		switch typeName {
		case "", "message":
			roleName, roleErr := optionalString(format, object, "role")
			if roleErr != nil {
				return nil, diagnostics, roleErr
			}
			role, roleErr := validateRole(format, path+".role", roleName)
			if roleErr != nil {
				return nil, diagnostics, roleErr
			}
			contentRaw := object["content"]
			delete(object, "content")
			content, next, contentErr := decodeResponsesContent(contentRaw, policy, path+".content", role)
			diagnostics = append(diagnostics, next...)
			if contentErr != nil {
				return nil, diagnostics, contentErr
			}
			id, idErr := optionalString(format, object, "id")
			if idErr != nil {
				return nil, diagnostics, idErr
			}
			extensions, next, extensionErr := collectExtensions(format, object, policy)
			diagnostics = append(diagnostics, next...)
			if extensionErr != nil {
				return nil, diagnostics, extensionErr
			}
			messages = append(messages, llmprotocol.Message{ID: id, Role: role, Content: content, Extensions: extensions})
		case "function_call":
			call, callErr := decodeResponsesFunctionCall(object, path)
			if callErr != nil {
				return nil, diagnostics, callErr
			}
			messages = append(messages, llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.ContentBlock{{Type: llmprotocol.ContentToolCall, Provider: format, ToolCall: &call}}})
		case "function_call_output":
			callID, callErr := optionalString(format, object, "call_id")
			if callErr != nil {
				return nil, diagnostics, callErr
			}
			outputRaw := object["output"]
			delete(object, "output")
			content, next, contentErr := decodeResponsesContent(outputRaw, policy, path+".output", llmprotocol.RoleTool)
			diagnostics = append(diagnostics, next...)
			if contentErr != nil {
				return nil, diagnostics, contentErr
			}
			messages = append(messages, llmprotocol.Message{Role: llmprotocol.RoleTool, Content: []llmprotocol.ContentBlock{{Type: llmprotocol.ContentToolResult, Provider: format, Result: &llmprotocol.ToolResult{ToolCallID: callID, Content: content}}}})
		case "reasoning":
			blocks, next, reasoningErr := decodeResponsesReasoning(value, policy, path)
			diagnostics = append(diagnostics, next...)
			if reasoningErr != nil {
				return nil, diagnostics, reasoningErr
			}
			messages = append(messages, llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: blocks})
		default:
			block, next, unknownErr := decodeUnknownBlock(format, value, policy, path)
			diagnostics = append(diagnostics, next...)
			if unknownErr != nil {
				return nil, diagnostics, unknownErr
			}
			if block != nil {
				messages = append(messages, llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.ContentBlock{*block}})
			}
		}
	}
	return messages, diagnostics, nil
}

func decodeResponsesContent(raw json.RawMessage, policy llmprotocol.Policy, path string, role llmprotocol.Role) ([]llmprotocol.ContentBlock, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatOpenAIResponses
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
		case "input_text", "output_text", "text":
			text, textErr := optionalString(format, object, "text")
			if textErr != nil {
				return nil, diagnostics, textErr
			}
			block = &llmprotocol.ContentBlock{Type: llmprotocol.ContentText, Text: text, Provider: format}
		case "refusal":
			text, textErr := optionalString(format, object, "refusal")
			if textErr != nil {
				return nil, diagnostics, textErr
			}
			block = &llmprotocol.ContentBlock{Type: llmprotocol.ContentRefusal, Text: text, Provider: format}
		case "input_image":
			url, urlErr := optionalString(format, object, "image_url")
			if urlErr != nil {
				return nil, diagnostics, urlErr
			}
			fileID, fileErr := optionalString(format, object, "file_id")
			if fileErr != nil {
				return nil, diagnostics, fileErr
			}
			detail, detailErr := optionalString(format, object, "detail")
			if detailErr != nil {
				return nil, diagnostics, detailErr
			}
			block = &llmprotocol.ContentBlock{Type: llmprotocol.ContentImage, Provider: format, Source: &llmprotocol.Source{Kind: chooseMediaSourceKind(url, fileID), URL: url, FileID: fileID, Detail: detail}}
		case "input_file":
			fileID, fileErr := optionalString(format, object, "file_id")
			if fileErr != nil {
				return nil, diagnostics, fileErr
			}
			fileData, fileErr := optionalString(format, object, "file_data")
			if fileErr != nil {
				return nil, diagnostics, fileErr
			}
			filename, fileErr := optionalString(format, object, "filename")
			if fileErr != nil {
				return nil, diagnostics, fileErr
			}
			block = &llmprotocol.ContentBlock{Type: llmprotocol.ContentFile, Provider: format, Source: &llmprotocol.Source{Kind: chooseSourceKind(fileID, fileData), FileID: fileID, Data: fileData, Filename: filename}}
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
		// No nil check: every case above assigns a block and the default one
		// continues, so reaching here means block is set.
		block.Extensions, diagnostics, err = collectAndAppendExtensions(format, object, policy, diagnostics)
		if err != nil {
			return nil, diagnostics, err
		}
		blocks = append(blocks, *block)
	}
	_ = role
	return blocks, diagnostics, nil
}

func decodeResponsesFunctionCall(object map[string]json.RawMessage, path string) (llmprotocol.ToolCall, error) {
	const format = llmprotocol.FormatOpenAIResponses
	callID, err := optionalString(format, object, "call_id")
	if err != nil {
		return llmprotocol.ToolCall{}, err
	}
	if callID == "" {
		callID, err = optionalString(format, object, "id")
		if err != nil {
			return llmprotocol.ToolCall{}, err
		}
	}
	name, err := optionalString(format, object, "name")
	if err != nil {
		return llmprotocol.ToolCall{}, err
	}
	arguments, err := optionalString(format, object, "arguments")
	if err != nil {
		return llmprotocol.ToolCall{}, err
	}
	args := json.RawMessage(arguments)
	if !json.Valid(args) {
		args = rawJSONString(arguments)
	}
	if name == "" {
		return llmprotocol.ToolCall{}, translationError(format, path+".name", "missing_tool_name", "function call name is required")
	}
	return llmprotocol.ToolCall{ID: callID, Name: name, Arguments: args}, nil
}

func decodeResponsesReasoning(raw json.RawMessage, policy llmprotocol.Policy, path string) ([]llmprotocol.ContentBlock, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatOpenAIResponses
	object, err := decodeObject(format, raw)
	if err != nil {
		return nil, nil, err
	}
	delete(object, "type")
	delete(object, "id")
	delete(object, "status")
	var text strings.Builder
	for _, name := range []string{"summary", "content"} {
		if rawParts, ok := object[name]; ok {
			delete(object, name)
			parts, arrayErr := decodeArray(format, path+"."+name, rawParts)
			if arrayErr != nil {
				return nil, nil, arrayErr
			}
			for _, partRaw := range parts {
				part, objectErr := decodeObject(format, partRaw)
				if objectErr != nil {
					return nil, nil, objectErr
				}
				delete(part, "type")
				value, textErr := optionalString(format, part, "text")
				if textErr != nil {
					return nil, nil, textErr
				}
				text.WriteString(value)
			}
		}
	}
	if encrypted, ok := object["encrypted_content"]; ok {
		delete(object, "encrypted_content")
		signature, signatureErr := decodeString(format, path+".encrypted_content", encrypted)
		if signatureErr != nil {
			return nil, nil, signatureErr
		}
		extensions, diagnostics, collectErr := collectExtensions(format, object, policy)
		if collectErr != nil {
			return nil, diagnostics, collectErr
		}
		return []llmprotocol.ContentBlock{{Type: llmprotocol.ContentReasoning, Text: text.String(), Signature: signature, Provider: format, Extensions: extensions}}, diagnostics, nil
	}
	extensions, diagnostics, err := collectExtensions(format, object, policy)
	if err != nil {
		return nil, diagnostics, err
	}
	return []llmprotocol.ContentBlock{{Type: llmprotocol.ContentReasoning, Text: text.String(), Provider: format, Extensions: extensions}}, diagnostics, nil
}

func decodeResponsesTools(raw json.RawMessage, policy llmprotocol.Policy, diagnostics []llmprotocol.Diagnostic) ([]llmprotocol.ToolDefinition, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatOpenAIResponses
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
			tools = append(tools, llmprotocol.ToolDefinition{Type: typeName, Raw: cloneRaw(value)})
			continue
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
		if parameters, ok := object["parameters"]; ok {
			tool.Parameters = cloneRaw(parameters)
			delete(object, "parameters")
		}
		tool.Strict, err = optionalBool(format, object, "strict")
		if err != nil {
			return nil, diagnostics, err
		}
		tool.Extensions, diagnostics, err = collectAndAppendExtensions(format, object, policy, diagnostics)
		if err != nil {
			return nil, diagnostics, fmt.Errorf("tool %d: %w", index, err)
		}
		tools = append(tools, tool)
	}
	return tools, diagnostics, nil
}

func decodeResponsesToolChoice(raw json.RawMessage) (*llmprotocol.ToolChoice, error) {
	const format = llmprotocol.FormatOpenAIResponses
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
	object, err := decodeObject(format, raw)
	if err != nil {
		return nil, err
	}
	typeName, err := optionalString(format, object, "type")
	if err != nil {
		return nil, err
	}
	name, err := optionalString(format, object, "name")
	if err != nil {
		return nil, err
	}
	if typeName == "function" && name != "" {
		return &llmprotocol.ToolChoice{Type: llmprotocol.ToolChoiceTool, Name: name}, nil
	}
	return &llmprotocol.ToolChoice{Type: llmprotocol.ToolChoiceRaw, Raw: cloneRaw(raw)}, nil
}

func encodeResponsesToolChoice(choice llmprotocol.ToolChoice) (any, error) {
	switch choice.Type {
	case llmprotocol.ToolChoiceAuto, llmprotocol.ToolChoiceRequired, llmprotocol.ToolChoiceNone:
		return string(choice.Type), nil
	case llmprotocol.ToolChoiceTool:
		return map[string]any{"type": "function", "name": choice.Name}, nil
	case llmprotocol.ToolChoiceRaw:
		return json.RawMessage(cloneRaw(choice.Raw)), nil
	default:
		return nil, translationError(llmprotocol.FormatOpenAIResponses, "$.tool_choice", "invalid_tool_choice", "unknown neutral tool choice")
	}
}

func encodeResponsesMessage(message llmprotocol.Message, policy llmprotocol.Policy, messageIndex int) ([]any, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatOpenAIResponses
	var ordinary []llmprotocol.ContentBlock
	var items []any
	var diagnostics []llmprotocol.Diagnostic
	for contentIndex, block := range message.Content {
		switch block.Type {
		case llmprotocol.ContentToolCall:
			if block.ToolCall == nil {
				return nil, diagnostics, translationError(format, "$.input", "missing_tool_call", "tool call is required")
			}
			id := block.ToolCall.ID
			if id == "" {
				id = stableID("call", messageIndex, contentIndex, block.ToolCall.Name, block.ToolCall.Arguments)
				diagnostics = append(diagnostics, llmprotocol.Diagnostic{Kind: llmprotocol.DiagnosticGeneratedID, Path: "$.input[].call_id", Code: "stable_tool_call_id"})
			}
			value := map[string]any{"type": "function_call", "call_id": id, "name": block.ToolCall.Name, "arguments": string(rawOrEmptyObject(block.ToolCall.Arguments))}
			mergeExtensions(value, block.Extensions)
			items = append(items, value)
		case llmprotocol.ContentToolResult:
			if block.Result == nil {
				return nil, diagnostics, translationError(format, "$.input", "missing_tool_result", "tool result is required")
			}
			output, next, err := encodeResponsesContent(block.Result.Content, policy, true)
			diagnostics = append(diagnostics, next...)
			if err != nil {
				return nil, diagnostics, err
			}
			value := any(output)
			if text, ok := singleTextPart(output); ok {
				value = text
			}
			item := map[string]any{"type": "function_call_output", "call_id": block.Result.ToolCallID, "output": value}
			mergeExtensions(item, block.Extensions)
			items = append(items, item)
		case llmprotocol.ContentReasoning:
			reasoning := map[string]any{"type": "reasoning", "summary": []any{map[string]any{"type": "summary_text", "text": block.Text}}}
			if block.Signature != "" {
				reasoning["encrypted_content"] = block.Signature
			}
			mergeExtensions(reasoning, block.Extensions)
			items = append(items, reasoning)
		default:
			ordinary = append(ordinary, block)
		}
	}
	if len(ordinary) > 0 || len(items) == 0 {
		content, next, err := encodeResponsesContent(ordinary, policy, message.Role == llmprotocol.RoleUser || message.Role == llmprotocol.RoleTool)
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return nil, diagnostics, err
		}
		value := map[string]any{"type": "message", "role": string(message.Role), "content": content}
		if message.ID != "" {
			value["id"] = message.ID
		}
		mergeExtensions(value, message.Extensions)
		items = append([]any{value}, items...)
	}
	return items, diagnostics, nil
}

func encodeResponsesContent(blocks []llmprotocol.ContentBlock, policy llmprotocol.Policy, input bool) ([]any, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatOpenAIResponses
	parts := make([]any, 0, len(blocks))
	var diagnostics []llmprotocol.Diagnostic
	for index, block := range blocks {
		path := fmt.Sprintf("$.content[%d]", index)
		switch block.Type {
		case llmprotocol.ContentText:
			typeName := "output_text"
			if input {
				typeName = "input_text"
			}
			value := map[string]any{"type": typeName, "text": block.Text}
			mergeExtensions(value, block.Extensions)
			parts = append(parts, value)
		case llmprotocol.ContentRefusal:
			value := map[string]any{"type": "refusal", "refusal": block.Text}
			mergeExtensions(value, block.Extensions)
			parts = append(parts, value)
		case llmprotocol.ContentImage:
			if !input || block.Source == nil {
				return nil, diagnostics, translationError(format, path, "unsupported_image", "Responses supports images only as input")
			}
			value := map[string]any{"type": "input_image"}
			if block.Source.URL != "" {
				value["image_url"] = block.Source.URL
			}
			if block.Source.FileID != "" {
				value["file_id"] = block.Source.FileID
			}
			if block.Source.Detail != "" {
				value["detail"] = block.Source.Detail
			}
			mergeExtensions(value, block.Extensions)
			parts = append(parts, value)
		case llmprotocol.ContentFile:
			if !input || block.Source == nil {
				return nil, diagnostics, translationError(format, path, "unsupported_file", "Responses supports files only as input")
			}
			value := map[string]any{"type": "input_file"}
			if block.Source.FileID != "" {
				value["file_id"] = block.Source.FileID
			}
			if block.Source.Data != "" {
				value["file_data"] = block.Source.Data
			}
			if block.Source.Filename != "" {
				value["filename"] = block.Source.Filename
			}
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

func decodeResponsesOutput(raw json.RawMessage, status string, policy llmprotocol.Policy, diagnostics []llmprotocol.Diagnostic) ([]llmprotocol.ResponseOutput, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatOpenAIResponses
	values, err := decodeArray(format, "$.output", raw)
	if err != nil {
		return nil, diagnostics, err
	}
	outputs := make([]llmprotocol.ResponseOutput, 0, len(values))
	for index, value := range values {
		path := fmt.Sprintf("$.output[%d]", index)
		object, objectErr := decodeObject(format, value)
		if objectErr != nil {
			return nil, diagnostics, objectErr
		}
		typeName, typeErr := optionalString(format, object, "type")
		if typeErr != nil {
			return nil, diagnostics, typeErr
		}
		id, idErr := optionalString(format, object, "id")
		if idErr != nil {
			return nil, diagnostics, idErr
		}
		switch typeName {
		case "message":
			roleName, roleErr := optionalString(format, object, "role")
			if roleErr != nil {
				return nil, diagnostics, roleErr
			}
			role, roleErr := validateRole(format, path+".role", roleName)
			if roleErr != nil {
				return nil, diagnostics, roleErr
			}
			contentRaw := object["content"]
			delete(object, "content")
			content, next, contentErr := decodeResponsesContent(contentRaw, policy, path+".content", role)
			diagnostics = append(diagnostics, next...)
			if contentErr != nil {
				return nil, diagnostics, contentErr
			}
			outputs = append(outputs, llmprotocol.ResponseOutput{ID: id, Role: role, Content: content, StopReason: responsesStopReason(status)})
		case "function_call":
			call, callErr := decodeResponsesFunctionCall(object, path)
			if callErr != nil {
				return nil, diagnostics, callErr
			}
			outputs = append(outputs, llmprotocol.ResponseOutput{ID: id, Role: llmprotocol.RoleAssistant, Content: []llmprotocol.ContentBlock{{Type: llmprotocol.ContentToolCall, Provider: format, ToolCall: &call}}, StopReason: llmprotocol.StopToolUse})
		case "reasoning":
			blocks, next, reasoningErr := decodeResponsesReasoning(value, policy, path)
			diagnostics = append(diagnostics, next...)
			if reasoningErr != nil {
				return nil, diagnostics, reasoningErr
			}
			outputs = append(outputs, llmprotocol.ResponseOutput{ID: id, Role: llmprotocol.RoleAssistant, Content: blocks})
		default:
			block, next, unknownErr := decodeUnknownBlock(format, value, policy, path)
			diagnostics = append(diagnostics, next...)
			if unknownErr != nil {
				return nil, diagnostics, unknownErr
			}
			if block != nil {
				outputs = append(outputs, llmprotocol.ResponseOutput{ID: id, Role: llmprotocol.RoleAssistant, Content: []llmprotocol.ContentBlock{*block}})
			}
		}
	}
	return outputs, diagnostics, nil
}

func encodeResponsesOutput(output llmprotocol.ResponseOutput, policy llmprotocol.Policy, outputIndex int) ([]any, []llmprotocol.Diagnostic, error) {
	message := llmprotocol.Message{ID: output.ID, Role: output.Role, Content: output.Content, Extensions: output.Extensions}
	return encodeResponsesMessage(message, policy, outputIndex)
}

func responsesStopReason(status string) llmprotocol.StopReason {
	switch status {
	case "completed":
		return llmprotocol.StopEndTurn
	case "incomplete":
		return llmprotocol.StopMaxTokens
	case "failed", "cancelled":
		return llmprotocol.StopError
	default:
		return llmprotocol.StopUnknown
	}
}

func singleTextPart(parts []any) (string, bool) {
	if len(parts) != 1 {
		return "", false
	}
	object, ok := parts[0].(map[string]any)
	if !ok {
		return "", false
	}
	typeName, _ := object["type"].(string)
	if typeName != "input_text" && typeName != "output_text" && typeName != "text" {
		return "", false
	}
	text, ok := object["text"].(string)
	return text, ok
}

func stableID(prefix string, indexes ...any) string {
	hash := sha256.New()
	for _, value := range indexes {
		_, _ = fmt.Fprint(hash, value, "\x00")
	}
	return prefix + "_" + hex.EncodeToString(hash.Sum(nil)[:12])
}

func chooseMediaSourceKind(url, fileID string) string {
	if fileID != "" {
		return "provider_file"
	}
	if url != "" {
		return "url"
	}
	return "unknown"
}
