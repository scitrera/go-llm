// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package codec

import (
	"encoding/json"
	"fmt"
	"strings"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

// Gemini implements the JSON contract shared by generateContent and
// streamGenerateContent. The model and operation live in the request URL and
// are deliberately not part of the encoded body.
type Gemini struct{}

func (Gemini) Format() llmprotocol.Format { return llmprotocol.FormatGemini }

func (Gemini) DecodeRequest(body json.RawMessage, policy llmprotocol.Policy) (RequestResult, error) {
	const format = llmprotocol.FormatGemini
	object, err := decodeObject(format, body)
	if err != nil {
		return RequestResult{}, err
	}
	request := llmprotocol.Request{Preservation: preserveBody(policy, format, body, true)}
	var diagnostics []llmprotocol.Diagnostic
	if raw, ok := object["systemInstruction"]; ok {
		delete(object, "systemInstruction")
		instruction, next, decodeErr := decodeGeminiContent(raw, policy, "$.systemInstruction")
		diagnostics = append(diagnostics, next...)
		if decodeErr != nil {
			return RequestResult{}, decodeErr
		}
		request.Instructions = append(request.Instructions, llmprotocol.Instruction{Role: llmprotocol.RoleSystem, Content: instruction.Content})
	}
	if raw, ok := object["contents"]; ok {
		delete(object, "contents")
		values, arrayErr := decodeArray(format, "$.contents", raw)
		if arrayErr != nil {
			return RequestResult{}, arrayErr
		}
		request.Messages = make([]llmprotocol.Message, 0, len(values))
		for index, value := range values {
			message, next, decodeErr := decodeGeminiContent(value, policy, fmt.Sprintf("$.contents[%d]", index))
			diagnostics = append(diagnostics, next...)
			if decodeErr != nil {
				return RequestResult{}, decodeErr
			}
			request.Messages = append(request.Messages, message)
		}
	}
	if raw, ok := object["generationConfig"]; ok {
		delete(object, "generationConfig")
		generation, objectErr := decodeObject(format, raw)
		if objectErr != nil {
			return RequestResult{}, objectErr
		}
		request.Output.MaxTokens, err = optionalInt(format, generation, "maxOutputTokens")
		if err != nil {
			return RequestResult{}, err
		}
		request.Output.StopSequences, err = optionalStrings(format, generation, "stopSequences", false)
		if err != nil {
			return RequestResult{}, err
		}
		request.Sampling.Temperature, err = optionalFloat(format, generation, "temperature")
		if err != nil {
			return RequestResult{}, err
		}
		request.Sampling.TopP, err = optionalFloat(format, generation, "topP")
		if err != nil {
			return RequestResult{}, err
		}
		request.Sampling.TopK, err = optionalInt(format, generation, "topK")
		if err != nil {
			return RequestResult{}, err
		}
		request.Sampling.FrequencyPenalty, err = optionalFloat(format, generation, "frequencyPenalty")
		if err != nil {
			return RequestResult{}, err
		}
		request.Sampling.PresencePenalty, err = optionalFloat(format, generation, "presencePenalty")
		if err != nil {
			return RequestResult{}, err
		}
		request.Sampling.Seed, err = optionalInt(format, generation, "seed")
		if err != nil {
			return RequestResult{}, err
		}
		request.Output.Choices, err = optionalInt(format, generation, "candidateCount")
		if err != nil {
			return RequestResult{}, err
		}
		request.Output.Logprobs, err = optionalBool(format, generation, "responseLogprobs")
		if err != nil {
			return RequestResult{}, err
		}
		request.Output.TopLogprobs, err = optionalInt(format, generation, "logprobs")
		if err != nil {
			return RequestResult{}, err
		}
		request.Output.Modalities, err = optionalStrings(format, generation, "responseModalities", false)
		if err != nil {
			return RequestResult{}, err
		}
		for index := range request.Output.Modalities {
			request.Output.Modalities[index] = strings.ToLower(request.Output.Modalities[index])
		}
		if rawThinking, ok := generation["thinkingConfig"]; ok {
			delete(generation, "thinkingConfig")
			thinking, thinkingErr := decodeObject(format, rawThinking)
			if thinkingErr != nil {
				return RequestResult{}, thinkingErr
			}
			request.Reasoning.BudgetTokens, err = optionalInt(format, thinking, "thinkingBudget")
			if err == nil {
				request.Reasoning.Include, err = optionalBool(format, thinking, "includeThoughts")
			}
			if err == nil {
				request.Reasoning.Level, err = optionalString(format, thinking, "thinkingLevel")
			}
			if err != nil {
				return RequestResult{}, err
			}
			if len(thinking) != 0 {
				request.Reasoning.Provider = format
				request.Reasoning.Raw, _ = json.Marshal(thinking)
			}
		}
		if schema, ok := generation["responseJsonSchema"]; ok {
			delete(generation, "responseJsonSchema")
			request.Output.ResponseFormat = geminiResponseFormat(schema)
		}
		delete(generation, "responseMimeType")
		_, next, collectErr := collectExtensions(format, generation, policy)
		diagnostics = append(diagnostics, next...)
		if collectErr != nil {
			return RequestResult{}, collectErr
		}
	}
	if raw, ok := object["tools"]; ok {
		delete(object, "tools")
		request.Tools, diagnostics, err = decodeGeminiTools(raw, policy, diagnostics)
		if err != nil {
			return RequestResult{}, err
		}
	}
	if raw, ok := object["toolConfig"]; ok {
		delete(object, "toolConfig")
		request.ToolChoice, err = decodeGeminiToolChoice(raw)
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

func (Gemini) EncodeRequest(request llmprotocol.Request, policy llmprotocol.Policy) (WireResult, error) {
	const format = llmprotocol.FormatGemini
	if raw, ok := preservedRequest(request, format, policy); ok {
		return WireResult{Body: raw}, nil
	}
	var diagnostics []llmprotocol.Diagnostic
	if err := validateGenerationBounds(format, request, 2, 5); err != nil {
		return WireResult{}, err
	}
	if providerStateActive(request.State) {
		next, err := lossy(format, policy, "$.state", "provider_state_not_supported")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{}, err
		}
	}
	object := map[string]any{}
	if len(request.Instructions) > 0 {
		var blocks []llmprotocol.ContentBlock
		for _, instruction := range request.Instructions {
			if instruction.Role != llmprotocol.RoleSystem && instruction.Role != llmprotocol.RoleDeveloper {
				return WireResult{}, translationError(format, "$.systemInstruction", "invalid_instruction_role", "Gemini systemInstruction accepts system or developer instructions")
			}
			blocks = append(blocks, instruction.Content...)
		}
		parts, next, err := encodeGeminiParts(blocks, nil, policy, "$.systemInstruction.parts")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{}, err
		}
		object["systemInstruction"] = map[string]any{"role": "system", "parts": parts}
	}
	toolNames := map[string]string{}
	contents := make([]any, 0, len(request.Messages))
	for index, message := range request.Messages {
		role := "user"
		switch message.Role {
		case llmprotocol.RoleUser, llmprotocol.RoleTool:
		case llmprotocol.RoleAssistant:
			role = "model"
		default:
			return WireResult{}, translationError(format, fmt.Sprintf("$.contents[%d].role", index), "unsupported_role", "Gemini contents accept user or assistant roles")
		}
		parts, next, err := encodeGeminiParts(message.Content, toolNames, policy, fmt.Sprintf("$.contents[%d].parts", index))
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{}, err
		}
		value := map[string]any{"role": role, "parts": parts}
		mergeExtensions(value, message.Extensions)
		contents = append(contents, value)
	}
	object["contents"] = contents
	generation := map[string]any{}
	putInt(generation, "maxOutputTokens", request.Output.MaxTokens)
	if len(request.Output.StopSequences) > 0 {
		generation["stopSequences"] = request.Output.StopSequences
	}
	if request.Sampling.Temperature != nil {
		generation["temperature"] = *request.Sampling.Temperature
	}
	if request.Sampling.TopP != nil {
		generation["topP"] = *request.Sampling.TopP
	}
	if request.Sampling.TopK != nil {
		generation["topK"] = *request.Sampling.TopK
	}
	if request.Sampling.FrequencyPenalty != nil {
		generation["frequencyPenalty"] = *request.Sampling.FrequencyPenalty
	}
	if request.Sampling.PresencePenalty != nil {
		generation["presencePenalty"] = *request.Sampling.PresencePenalty
	}
	putInt(generation, "seed", request.Sampling.Seed)
	if request.Output.Choices != nil {
		if *request.Output.Choices < 1 {
			return WireResult{}, translationError(format, "$.output.choices", "invalid_candidate_count", "candidate count must be positive")
		}
		generation["candidateCount"] = *request.Output.Choices
	}
	if request.Output.Logprobs != nil {
		generation["responseLogprobs"] = *request.Output.Logprobs
	}
	if request.Output.TopLogprobs != nil {
		if *request.Output.TopLogprobs < 0 || *request.Output.TopLogprobs > 20 {
			return WireResult{}, translationError(format, "$.output.top_logprobs", "logprobs_out_of_range", "Gemini logprobs must be in [0,20]")
		}
		generation["logprobs"] = *request.Output.TopLogprobs
	}
	if len(request.Output.Modalities) > 0 {
		modalities := make([]string, len(request.Output.Modalities))
		for index, modality := range request.Output.Modalities {
			switch strings.ToLower(modality) {
			case "text", "image", "audio":
				modalities[index] = strings.ToUpper(modality)
			default:
				return WireResult{}, translationError(format, "$.output.modalities", "unsupported_output_modality", "Gemini output modalities are text, image, or audio")
			}
		}
		generation["responseModalities"] = modalities
	}
	if len(request.Output.ResponseFormat) != 0 {
		schema, mimeType, err := encodeGeminiResponseFormat(request.Output.ResponseFormat)
		if err != nil {
			return WireResult{}, err
		}
		if len(schema) != 0 {
			generation["responseJsonSchema"] = schema
		}
		if mimeType != "" {
			generation["responseMimeType"] = mimeType
		}
	}
	if len(generation) > 0 {
		object["generationConfig"] = generation
	}
	if len(request.Tools) > 0 {
		declarations := make([]any, 0, len(request.Tools))
		for _, tool := range request.Tools {
			if tool.Type != "" && tool.Type != "function" {
				return WireResult{}, translationError(format, "$.tools[]", "hosted_tool_not_supported", "Gemini codec supports function tools only")
			}
			declaration := map[string]any{"name": tool.Name, "parametersJsonSchema": rawOrEmptyObject(tool.Parameters)}
			if tool.Description != "" {
				declaration["description"] = tool.Description
			}
			if tool.Strict != nil && *tool.Strict {
				next, err := lossy(format, policy, "$.tools[].strict", "strict_tool_schema_not_supported")
				diagnostics = append(diagnostics, next...)
				if err != nil {
					return WireResult{}, err
				}
			}
			mergeExtensions(declaration, tool.Extensions)
			declarations = append(declarations, declaration)
		}
		object["tools"] = []any{map[string]any{"functionDeclarations": declarations}}
	}
	if request.ToolChoice != nil {
		choice, err := encodeGeminiToolChoice(*request.ToolChoice)
		if err != nil {
			return WireResult{}, err
		}
		if choice != nil {
			object["toolConfig"] = choice
		}
	}
	if request.ParallelToolCalls != nil && !*request.ParallelToolCalls {
		next, err := lossy(format, policy, "$.parallel_tool_calls", "parallel_tool_control_not_supported")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{}, err
		}
	}
	if request.Reasoning.BudgetTokens != nil && (request.Reasoning.Level != "" || request.Reasoning.Effort != "") {
		return WireResult{}, translationError(format, "$.reasoning", "conflicting_reasoning_controls", "Gemini thinking budget and thinking level are mutually exclusive")
	}
	if request.Reasoning.BudgetTokens != nil || request.Reasoning.Include != nil || request.Reasoning.Level != "" || request.Reasoning.Effort != "" || len(request.Reasoning.Raw) != 0 {
		thinking := map[string]any{}
		if len(request.Reasoning.Raw) != 0 {
			if request.Reasoning.Provider != "" && request.Reasoning.Provider != format {
				return WireResult{}, translationError(format, "$.reasoning.raw", "reasoning_config_not_portable", "provider reasoning config cannot cross formats")
			}
			_ = json.Unmarshal(request.Reasoning.Raw, &thinking)
		}
		putInt(thinking, "thinkingBudget", request.Reasoning.BudgetTokens)
		if request.Reasoning.Include != nil {
			thinking["includeThoughts"] = *request.Reasoning.Include
		}
		level := request.Reasoning.Level
		if level == "" {
			level = request.Reasoning.Effort
		}
		if level != "" {
			thinking["thinkingLevel"] = strings.ToUpper(level)
		}
		generation["thinkingConfig"] = thinking
	}
	if len(generation) > 0 {
		object["generationConfig"] = generation
	}
	mergeExtensions(object, request.Extensions)
	body, err := marshalObject(format, object)
	return WireResult{Body: body, Diagnostics: diagnostics}, err
}

func (Gemini) DecodeResponse(body json.RawMessage, policy llmprotocol.Policy) (ResponseResult, error) {
	const format = llmprotocol.FormatGemini
	object, err := decodeObject(format, body)
	if err != nil {
		return ResponseResult{}, err
	}
	response := llmprotocol.Response{Preservation: preserveBody(policy, format, body, false)}
	response.ID, err = optionalString(format, object, "responseId")
	if err != nil {
		return ResponseResult{}, err
	}
	response.Model, err = optionalString(format, object, "modelVersion")
	if err != nil {
		return ResponseResult{}, err
	}
	var diagnostics []llmprotocol.Diagnostic
	if raw, ok := object["candidates"]; ok {
		delete(object, "candidates")
		values, arrayErr := decodeArray(format, "$.candidates", raw)
		if arrayErr != nil {
			return ResponseResult{}, arrayErr
		}
		response.Outputs = make([]llmprotocol.ResponseOutput, 0, len(values))
		for fallback, value := range values {
			candidate, objectErr := decodeObject(format, value)
			if objectErr != nil {
				return ResponseResult{}, objectErr
			}
			index := fallback
			if wireIndex, indexErr := optionalInt(format, candidate, "index"); indexErr != nil {
				return ResponseResult{}, indexErr
			} else if wireIndex != nil {
				index = int(*wireIndex)
			}
			output := llmprotocol.ResponseOutput{Index: &index, Role: llmprotocol.RoleAssistant}
			if rawContent, exists := candidate["content"]; exists {
				delete(candidate, "content")
				message, next, decodeErr := decodeGeminiContent(rawContent, policy, "$.candidates[].content")
				diagnostics = append(diagnostics, next...)
				if decodeErr != nil {
					return ResponseResult{}, decodeErr
				}
				output.Content = message.Content
			}
			finish, finishErr := optionalString(format, candidate, "finishReason")
			if finishErr != nil {
				return ResponseResult{}, finishErr
			}
			output.StopReason = decodeGeminiStopReason(finish, output.Content)
			if rawLogprobs, exists := candidate["logprobsResult"]; exists {
				delete(candidate, "logprobsResult")
				output.Logprobs, err = decodeGeminiLogprobs(rawLogprobs)
				if err != nil {
					return ResponseResult{}, err
				}
			}
			delete(candidate, "safetyRatings")
			delete(candidate, "finishMessage")
			output.Extensions, diagnostics, err = collectAndAppendExtensions(format, candidate, policy, diagnostics)
			if err != nil {
				return ResponseResult{}, err
			}
			response.Outputs = append(response.Outputs, output)
		}
	}
	if raw, ok := object["usageMetadata"]; ok {
		delete(object, "usageMetadata")
		response.Usage, err = decodeGeminiUsage(raw)
		if err != nil {
			return ResponseResult{}, err
		}
	}
	if rawFeedback, ok := object["promptFeedback"]; ok {
		delete(object, "promptFeedback")
		feedback, objectErr := decodeObject(format, rawFeedback)
		if objectErr != nil {
			return ResponseResult{}, objectErr
		}
		blockReason, reasonErr := optionalString(format, feedback, "blockReason")
		if reasonErr != nil {
			return ResponseResult{}, reasonErr
		}
		delete(feedback, "safetyRatings")
		_, next, collectErr := collectExtensions(format, feedback, policy)
		diagnostics = append(diagnostics, next...)
		if collectErr != nil {
			return ResponseResult{}, collectErr
		}
		if blockReason != "" && len(response.Outputs) == 0 {
			response.Outputs = append(response.Outputs, llmprotocol.ResponseOutput{Role: llmprotocol.RoleAssistant, StopReason: llmprotocol.StopContentFilter})
		}
	}
	response.Extensions, diagnostics, err = collectAndAppendExtensions(format, object, policy, diagnostics)
	if err != nil {
		return ResponseResult{}, err
	}
	return ResponseResult{Response: response, Diagnostics: diagnostics}, nil
}

func (Gemini) EncodeResponse(response llmprotocol.Response, policy llmprotocol.Policy) (WireResult, error) {
	const format = llmprotocol.FormatGemini
	if raw, ok := preservedResponse(response, format, policy); ok {
		return WireResult{Body: raw}, nil
	}
	object := map[string]any{}
	if response.ID != "" {
		object["responseId"] = response.ID
	}
	if response.Model != "" {
		object["modelVersion"] = response.Model
	}
	var diagnostics []llmprotocol.Diagnostic
	candidates := make([]any, 0, len(response.Outputs))
	for fallback, output := range response.Outputs {
		parts, next, err := encodeGeminiParts(output.Content, nil, policy, "$.candidates[].content.parts")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{}, err
		}
		index := fallback
		if output.Index != nil {
			index = *output.Index
		}
		candidate := map[string]any{
			"index":        index,
			"content":      map[string]any{"role": "model", "parts": parts},
			"finishReason": encodeGeminiStopReason(output.StopReason),
		}
		if len(output.Logprobs) > 0 {
			candidate["logprobsResult"] = encodeGeminiLogprobs(output.Logprobs)
		}
		mergeExtensions(candidate, output.Extensions)
		candidates = append(candidates, candidate)
	}
	object["candidates"] = candidates
	if hasUsage(response.Usage) {
		object["usageMetadata"] = encodeGeminiUsage(response.Usage)
	}
	mergeExtensions(object, response.Extensions)
	body, err := marshalObject(format, object)
	return WireResult{Body: body, Diagnostics: diagnostics}, err
}

func decodeGeminiContent(raw json.RawMessage, policy llmprotocol.Policy, path string) (llmprotocol.Message, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatGemini
	object, err := decodeObject(format, raw)
	if err != nil {
		return llmprotocol.Message{}, nil, err
	}
	roleName, err := optionalString(format, object, "role")
	if err != nil {
		return llmprotocol.Message{}, nil, err
	}
	role := llmprotocol.RoleUser
	if roleName == "model" {
		role = llmprotocol.RoleAssistant
	} else if roleName != "" && roleName != "user" && roleName != "system" {
		return llmprotocol.Message{}, nil, translationError(format, path+".role", "unsupported_role", "Gemini content role must be user, model, or system")
	} else if roleName == "system" {
		role = llmprotocol.RoleSystem
	}
	message := llmprotocol.Message{Role: role}
	var diagnostics []llmprotocol.Diagnostic
	if rawParts, ok := object["parts"]; ok {
		delete(object, "parts")
		parts, arrayErr := decodeArray(format, path+".parts", rawParts)
		if arrayErr != nil {
			return llmprotocol.Message{}, diagnostics, arrayErr
		}
		for index, rawPart := range parts {
			block, next, decodeErr := decodeGeminiPart(rawPart, policy, fmt.Sprintf("%s.parts[%d]", path, index))
			diagnostics = append(diagnostics, next...)
			if decodeErr != nil {
				return llmprotocol.Message{}, diagnostics, decodeErr
			}
			if block != nil {
				message.Content = append(message.Content, *block)
			}
		}
	}
	message.Extensions, diagnostics, err = collectAndAppendExtensions(format, object, policy, diagnostics)
	return message, diagnostics, err
}

func decodeGeminiPart(raw json.RawMessage, policy llmprotocol.Policy, path string) (*llmprotocol.ContentBlock, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatGemini
	object, err := decodeObject(format, raw)
	if err != nil {
		return nil, nil, err
	}
	if rawText, ok := object["text"]; ok {
		delete(object, "text")
		text, textErr := decodeString(format, path+".text", rawText)
		if textErr != nil {
			return nil, nil, textErr
		}
		thought := false
		if value, boolErr := optionalBool(format, object, "thought"); boolErr != nil {
			return nil, nil, boolErr
		} else if value != nil {
			thought = *value
		}
		typeName := llmprotocol.ContentText
		if thought {
			typeName = llmprotocol.ContentReasoning
		}
		block := &llmprotocol.ContentBlock{Type: typeName, Text: text, Provider: format}
		block.Signature, err = optionalString(format, object, "thoughtSignature")
		if err != nil {
			return nil, nil, err
		}
		block.Extensions, _, err = collectExtensions(format, object, policy)
		return block, nil, err
	}
	if rawCall, ok := object["functionCall"]; ok {
		delete(object, "functionCall")
		call, objectErr := decodeObject(format, rawCall)
		if objectErr != nil {
			return nil, nil, objectErr
		}
		id, idErr := optionalString(format, call, "id")
		if idErr != nil {
			return nil, nil, idErr
		}
		name, nameErr := optionalString(format, call, "name")
		if nameErr != nil {
			return nil, nil, nameErr
		}
		arguments := cloneRaw(call["args"])
		delete(call, "args")
		block := &llmprotocol.ContentBlock{Type: llmprotocol.ContentToolCall, Provider: format, ToolCall: &llmprotocol.ToolCall{ID: id, Name: name, Arguments: rawOrEmptyObject(arguments)}}
		block.Signature, err = optionalString(format, object, "thoughtSignature")
		if err != nil {
			return nil, nil, err
		}
		block.Extensions, _, err = collectExtensions(format, object, policy)
		if err != nil {
			return nil, nil, err
		}
		_, diagnostics, err := collectExtensions(format, call, policy)
		return block, diagnostics, err
	}
	if rawResponse, ok := object["functionResponse"]; ok {
		delete(object, "functionResponse")
		value, objectErr := decodeObject(format, rawResponse)
		if objectErr != nil {
			return nil, nil, objectErr
		}
		id, idErr := optionalString(format, value, "id")
		if idErr != nil {
			return nil, nil, idErr
		}
		_, nameErr := optionalString(format, value, "name")
		if nameErr != nil {
			return nil, nil, nameErr
		}
		response := cloneRaw(value["response"])
		delete(value, "response")
		content := []llmprotocol.ContentBlock{{Type: llmprotocol.ContentText, Text: string(response)}}
		result := &llmprotocol.ContentBlock{Type: llmprotocol.ContentToolResult, Provider: format, Result: &llmprotocol.ToolResult{ToolCallID: id, Content: content}}
		_, diagnostics, collectErr := collectExtensions(format, value, policy)
		return result, diagnostics, collectErr
	}
	for _, media := range []struct {
		name string
		kind llmprotocol.ContentType
	}{{"inlineData", llmprotocol.ContentFile}, {"fileData", llmprotocol.ContentFile}} {
		if rawSource, ok := object[media.name]; ok {
			sourceObject, objectErr := decodeObject(format, rawSource)
			if objectErr != nil {
				return nil, nil, objectErr
			}
			mimeType, mimeErr := optionalString(format, sourceObject, "mimeType")
			if mimeErr != nil {
				return nil, nil, mimeErr
			}
			typeName := geminiContentType(mimeType)
			source := &llmprotocol.Source{MediaType: mimeType}
			if media.name == "inlineData" {
				source.Kind = "base64"
				source.Data, err = optionalString(format, sourceObject, "data")
			} else {
				source.Kind = "url"
				source.URL, err = optionalString(format, sourceObject, "fileUri")
			}
			if err != nil {
				return nil, nil, err
			}
			return &llmprotocol.ContentBlock{Type: typeName, Source: source, Provider: format}, nil, nil
		}
	}
	block, diagnostics, err := decodeUnknownBlock(format, raw, policy, path)
	return block, diagnostics, err
}

func encodeGeminiParts(blocks []llmprotocol.ContentBlock, toolNames map[string]string, policy llmprotocol.Policy, path string) ([]any, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatGemini
	parts := make([]any, 0, len(blocks))
	var diagnostics []llmprotocol.Diagnostic
	for index, block := range blocks {
		blockPath := fmt.Sprintf("%s[%d]", path, index)
		part := map[string]any{}
		switch block.Type {
		case llmprotocol.ContentText:
			part["text"] = block.Text
		case llmprotocol.ContentReasoning:
			part["text"] = block.Text
			part["thought"] = true
			if block.Signature != "" {
				part["thoughtSignature"] = block.Signature
			}
		case llmprotocol.ContentToolCall:
			if block.ToolCall == nil {
				return nil, diagnostics, translationError(format, blockPath, "missing_tool_call", "tool call is required")
			}
			call := map[string]any{"id": block.ToolCall.ID, "name": block.ToolCall.Name, "args": rawOrEmptyObject(block.ToolCall.Arguments)}
			part["functionCall"] = call
			if block.Signature != "" {
				part["thoughtSignature"] = block.Signature
			}
			if toolNames != nil && block.ToolCall.ID != "" {
				toolNames[block.ToolCall.ID] = block.ToolCall.Name
			}
		case llmprotocol.ContentToolResult:
			if block.Result == nil {
				return nil, diagnostics, translationError(format, blockPath, "missing_tool_result", "tool result is required")
			}
			name := ""
			if toolNames != nil {
				name = toolNames[block.Result.ToolCallID]
			}
			if name == "" {
				return nil, diagnostics, translationError(format, blockPath, "missing_tool_name", "Gemini functionResponse requires a prior functionCall name")
			}
			response := geminiToolResultValue(block.Result.Content)
			part["functionResponse"] = map[string]any{"id": block.Result.ToolCallID, "name": name, "response": response}
		case llmprotocol.ContentImage, llmprotocol.ContentAudio, llmprotocol.ContentVideo, llmprotocol.ContentFile:
			if block.Source == nil {
				return nil, diagnostics, translationError(format, blockPath, "missing_media_source", "media source is required")
			}
			mimeType := block.Source.MediaType
			if mimeType == "" {
				mimeType = "application/octet-stream"
			}
			if block.Source.Data != "" || block.Source.Kind == "base64" {
				part["inlineData"] = map[string]any{"mimeType": mimeType, "data": block.Source.Data}
			} else if dataMime, data, ok := splitDataURL(block.Source.URL); ok {
				if dataMime != "" {
					mimeType = dataMime
				}
				part["inlineData"] = map[string]any{"mimeType": mimeType, "data": data}
			} else if block.Source.URL != "" {
				part["fileData"] = map[string]any{"mimeType": mimeType, "fileUri": block.Source.URL}
			} else {
				return nil, diagnostics, translationError(format, blockPath, "unsupported_media_source", "Gemini requires inline data or a file URI")
			}
		case llmprotocol.ContentUnknown:
			if block.Provider == format && json.Valid(block.Raw) {
				parts = append(parts, json.RawMessage(cloneRaw(block.Raw)))
				continue
			}
			next, err := lossy(format, policy, blockPath, "unknown_content_block")
			diagnostics = append(diagnostics, next...)
			if err != nil {
				return nil, diagnostics, err
			}
			continue
		default:
			next, err := lossy(format, policy, blockPath, "content_type_not_supported")
			diagnostics = append(diagnostics, next...)
			if err != nil {
				return nil, diagnostics, err
			}
			continue
		}
		mergeExtensions(part, block.Extensions)
		parts = append(parts, part)
	}
	return parts, diagnostics, nil
}

func decodeGeminiTools(raw json.RawMessage, policy llmprotocol.Policy, diagnostics []llmprotocol.Diagnostic) ([]llmprotocol.ToolDefinition, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatGemini
	groups, err := decodeArray(format, "$.tools", raw)
	if err != nil {
		return nil, diagnostics, err
	}
	var tools []llmprotocol.ToolDefinition
	for _, rawGroup := range groups {
		group, objectErr := decodeObject(format, rawGroup)
		if objectErr != nil {
			return nil, diagnostics, objectErr
		}
		declarations, ok := group["functionDeclarations"]
		if !ok {
			_, next, collectErr := collectExtensions(format, group, policy)
			diagnostics = append(diagnostics, next...)
			if collectErr != nil {
				return nil, diagnostics, collectErr
			}
			continue
		}
		values, arrayErr := decodeArray(format, "$.tools[].functionDeclarations", declarations)
		if arrayErr != nil {
			return nil, diagnostics, arrayErr
		}
		for _, rawDeclaration := range values {
			declaration, decodeErr := decodeObject(format, rawDeclaration)
			if decodeErr != nil {
				return nil, diagnostics, decodeErr
			}
			tool := llmprotocol.ToolDefinition{Type: "function"}
			tool.Name, err = optionalString(format, declaration, "name")
			if err != nil {
				return nil, diagnostics, err
			}
			tool.Description, err = optionalString(format, declaration, "description")
			if err != nil {
				return nil, diagnostics, err
			}
			if schema, exists := declaration["parametersJsonSchema"]; exists {
				tool.Parameters = cloneRaw(schema)
				delete(declaration, "parametersJsonSchema")
			} else if schema, exists := declaration["parameters"]; exists {
				tool.Parameters = cloneRaw(schema)
				delete(declaration, "parameters")
			}
			tool.Extensions, diagnostics, err = collectAndAppendExtensions(format, declaration, policy, diagnostics)
			if err != nil {
				return nil, diagnostics, err
			}
			tools = append(tools, tool)
		}
		delete(group, "functionDeclarations")
	}
	return tools, diagnostics, nil
}

func decodeGeminiToolChoice(raw json.RawMessage) (*llmprotocol.ToolChoice, error) {
	const format = llmprotocol.FormatGemini
	object, err := decodeObject(format, raw)
	if err != nil {
		return nil, err
	}
	function, err := decodeObject(format, object["functionCallingConfig"])
	if err != nil {
		return nil, err
	}
	mode, err := optionalString(format, function, "mode")
	if err != nil {
		return nil, err
	}
	var names []string
	if rawNames, ok := function["allowedFunctionNames"]; ok {
		if err := json.Unmarshal(rawNames, &names); err != nil {
			return nil, translationError(format, "$.toolConfig.functionCallingConfig.allowedFunctionNames", "invalid_string_array", "allowedFunctionNames must be an array of strings")
		}
	}
	if len(names) == 1 {
		return &llmprotocol.ToolChoice{Type: llmprotocol.ToolChoiceTool, Name: names[0]}, nil
	}
	switch strings.ToUpper(mode) {
	case "AUTO", "VALIDATED", "":
		return &llmprotocol.ToolChoice{Type: llmprotocol.ToolChoiceAuto}, nil
	case "ANY":
		return &llmprotocol.ToolChoice{Type: llmprotocol.ToolChoiceRequired}, nil
	case "NONE":
		return &llmprotocol.ToolChoice{Type: llmprotocol.ToolChoiceNone}, nil
	default:
		return nil, translationError(format, "$.toolConfig.functionCallingConfig.mode", "unsupported_tool_choice", "unknown Gemini function-calling mode")
	}
}

func encodeGeminiToolChoice(choice llmprotocol.ToolChoice) (any, error) {
	function := map[string]any{}
	switch choice.Type {
	case llmprotocol.ToolChoiceAuto:
		function["mode"] = "AUTO"
	case llmprotocol.ToolChoiceRequired:
		function["mode"] = "ANY"
	case llmprotocol.ToolChoiceNone:
		function["mode"] = "NONE"
	case llmprotocol.ToolChoiceTool:
		function["mode"] = "ANY"
		function["allowedFunctionNames"] = []string{choice.Name}
	default:
		return nil, translationError(llmprotocol.FormatGemini, "$.toolConfig", "unsupported_tool_choice", "tool choice cannot be represented by Gemini")
	}
	return map[string]any{"functionCallingConfig": function}, nil
}

func decodeGeminiUsage(raw json.RawMessage) (llmprotocol.Usage, error) {
	const format = llmprotocol.FormatGemini
	object, err := decodeObject(format, raw)
	if err != nil {
		return llmprotocol.Usage{}, err
	}
	usage := llmprotocol.Usage{}
	usage.InputTokens, err = optionalInt(format, object, "promptTokenCount")
	if err != nil {
		return usage, err
	}
	usage.OutputTokens, err = optionalInt(format, object, "candidatesTokenCount")
	if err != nil {
		return usage, err
	}
	usage.TotalTokens, err = optionalInt(format, object, "totalTokenCount")
	if err != nil {
		return usage, err
	}
	usage.CachedInputTokens, err = optionalInt(format, object, "cachedContentTokenCount")
	if err != nil {
		return usage, err
	}
	usage.ReasoningTokens, err = optionalInt(format, object, "thoughtsTokenCount")
	if err != nil {
		return usage, err
	}
	usage.ToolUsePromptTokens, err = optionalInt(format, object, "toolUsePromptTokenCount")
	return usage, err
}

func encodeGeminiUsage(usage llmprotocol.Usage) map[string]any {
	object := map[string]any{}
	putInt(object, "promptTokenCount", usage.InputTokens)
	putInt(object, "candidatesTokenCount", usage.OutputTokens)
	putInt(object, "totalTokenCount", usage.TotalTokens)
	putInt(object, "cachedContentTokenCount", usage.CachedInputTokens)
	putInt(object, "thoughtsTokenCount", usage.ReasoningTokens)
	putInt(object, "toolUsePromptTokenCount", usage.ToolUsePromptTokens)
	return object
}

func decodeGeminiStopReason(reason string, content []llmprotocol.ContentBlock) llmprotocol.StopReason {
	for _, block := range content {
		if block.Type == llmprotocol.ContentToolCall {
			return llmprotocol.StopToolUse
		}
	}
	switch reason {
	case "":
		return ""
	case "STOP", "FINISH_REASON_UNSPECIFIED":
		return llmprotocol.StopEndTurn
	case "MAX_TOKENS":
		return llmprotocol.StopMaxTokens
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY":
		return llmprotocol.StopContentFilter
	default:
		return llmprotocol.StopUnknown
	}
}

func encodeGeminiStopReason(reason llmprotocol.StopReason) string {
	switch reason {
	case llmprotocol.StopMaxTokens:
		return "MAX_TOKENS"
	case llmprotocol.StopContentFilter:
		return "SAFETY"
	default:
		return "STOP"
	}
}

func geminiContentType(mimeType string) llmprotocol.ContentType {
	switch {
	case strings.HasPrefix(mimeType, "image/"):
		return llmprotocol.ContentImage
	case strings.HasPrefix(mimeType, "audio/"):
		return llmprotocol.ContentAudio
	case strings.HasPrefix(mimeType, "video/"):
		return llmprotocol.ContentVideo
	default:
		return llmprotocol.ContentFile
	}
}

func splitDataURL(value string) (string, string, bool) {
	if !strings.HasPrefix(value, "data:") {
		return "", "", false
	}
	comma := strings.IndexByte(value, ',')
	if comma < 0 {
		return "", "", false
	}
	header := value[len("data:"):comma]
	mimeType, found := strings.CutSuffix(header, ";base64")
	if !found {
		return "", "", false
	}
	return mimeType, value[comma+1:], true
}

func geminiToolResultValue(blocks []llmprotocol.ContentBlock) any {
	if len(blocks) == 1 && blocks[0].Type == llmprotocol.ContentText {
		var value any
		if json.Unmarshal([]byte(blocks[0].Text), &value) == nil {
			return value
		}
		return map[string]any{"output": blocks[0].Text}
	}
	var text strings.Builder
	for _, block := range blocks {
		if block.Type == llmprotocol.ContentText {
			text.WriteString(block.Text)
		}
	}
	return map[string]any{"output": text.String()}
}

func geminiResponseFormat(schema json.RawMessage) json.RawMessage {
	value := map[string]any{"type": "json_schema", "schema": json.RawMessage(cloneRaw(schema))}
	encoded, _ := json.Marshal(value)
	return encoded
}

func encodeGeminiResponseFormat(raw json.RawMessage) (json.RawMessage, string, error) {
	var object map[string]json.RawMessage
	if json.Unmarshal(raw, &object) != nil || object == nil {
		return nil, "", translationError(llmprotocol.FormatGemini, "$.generationConfig.responseJsonSchema", "invalid_response_format", "response format must be an object")
	}
	var typeName string
	_ = json.Unmarshal(object["type"], &typeName)
	switch typeName {
	case "", "text":
		return nil, "text/plain", nil
	case "json_object":
		return json.RawMessage(`{"type":"object"}`), "application/json", nil
	case "json_schema":
		schema := object["schema"]
		if len(schema) == 0 {
			var nested map[string]json.RawMessage
			_ = json.Unmarshal(object["json_schema"], &nested)
			schema = nested["schema"]
		}
		if len(schema) == 0 {
			return nil, "", translationError(llmprotocol.FormatGemini, "$.generationConfig.responseJsonSchema", "missing_schema", "JSON-schema response format requires a schema")
		}
		return cloneRaw(schema), "application/json", nil
	default:
		return nil, "", translationError(llmprotocol.FormatGemini, "$.generationConfig.responseMimeType", "unsupported_response_format", "response format cannot be represented by Gemini")
	}
}
