// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package codec

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

// Bedrock implements the JSON payloads for Amazon Bedrock Converse. AWS SigV4,
// URL construction, credentials, and binary event-stream framing remain
// transport concerns; stream WireEvents use the Bedrock :event-type as Event.
type Bedrock struct{}

func (Bedrock) Format() llmprotocol.Format { return llmprotocol.FormatBedrock }

func (Bedrock) DecodeRequest(body json.RawMessage, policy llmprotocol.Policy) (RequestResult, error) {
	const format = llmprotocol.FormatBedrock
	object, err := decodeObject(format, body)
	if err != nil {
		return RequestResult{}, err
	}
	request := llmprotocol.Request{Preservation: preserveBody(policy, format, body, true)}
	var diagnostics []llmprotocol.Diagnostic
	if raw, ok := object["system"]; ok {
		delete(object, "system")
		blocks, next, decodeErr := decodeBedrockContent(raw, policy, "$.system")
		diagnostics = append(diagnostics, next...)
		if decodeErr != nil {
			return RequestResult{}, decodeErr
		}
		request.Instructions = append(request.Instructions, llmprotocol.Instruction{Role: llmprotocol.RoleSystem, Content: blocks})
	}
	if raw, ok := object["messages"]; ok {
		delete(object, "messages")
		values, arrayErr := decodeArray(format, "$.messages", raw)
		if arrayErr != nil {
			return RequestResult{}, arrayErr
		}
		for index, value := range values {
			messageObject, objectErr := decodeObject(format, value)
			if objectErr != nil {
				return RequestResult{}, objectErr
			}
			roleName, roleErr := optionalString(format, messageObject, "role")
			if roleErr != nil {
				return RequestResult{}, roleErr
			}
			role := llmprotocol.Role(roleName)
			if role != llmprotocol.RoleUser && role != llmprotocol.RoleAssistant {
				return RequestResult{}, translationError(format, fmt.Sprintf("$.messages[%d].role", index), "unsupported_role", "Bedrock messages accept user or assistant roles")
			}
			content, next, decodeErr := decodeBedrockContent(messageObject["content"], policy, fmt.Sprintf("$.messages[%d].content", index))
			diagnostics = append(diagnostics, next...)
			if decodeErr != nil {
				return RequestResult{}, decodeErr
			}
			delete(messageObject, "content")
			extensions, next, collectErr := collectExtensions(format, messageObject, policy)
			diagnostics = append(diagnostics, next...)
			if collectErr != nil {
				return RequestResult{}, collectErr
			}
			request.Messages = append(request.Messages, llmprotocol.Message{Role: role, Content: content, Extensions: extensions})
		}
	}
	if raw, ok := object["inferenceConfig"]; ok {
		delete(object, "inferenceConfig")
		config, objectErr := decodeObject(format, raw)
		if objectErr != nil {
			return RequestResult{}, objectErr
		}
		request.Output.MaxTokens, err = optionalInt(format, config, "maxTokens")
		if err != nil {
			return RequestResult{}, err
		}
		request.Output.StopSequences, err = optionalStrings(format, config, "stopSequences", false)
		if err != nil {
			return RequestResult{}, err
		}
		request.Sampling.Temperature, err = optionalFloat(format, config, "temperature")
		if err != nil {
			return RequestResult{}, err
		}
		request.Sampling.TopP, err = optionalFloat(format, config, "topP")
		if err != nil {
			return RequestResult{}, err
		}
		_, next, collectErr := collectExtensions(format, config, policy)
		diagnostics = append(diagnostics, next...)
		if collectErr != nil {
			return RequestResult{}, collectErr
		}
	}
	if raw, ok := object["outputConfig"]; ok {
		delete(object, "outputConfig")
		config, objectErr := decodeObject(format, raw)
		if objectErr != nil {
			return RequestResult{}, objectErr
		}
		if rawFormat, exists := config["textFormat"]; exists {
			delete(config, "textFormat")
			request.Output.ResponseFormat, err = decodeBedrockResponseFormat(rawFormat)
			if err != nil {
				return RequestResult{}, err
			}
		}
		if len(config) != 0 {
			leftover, _ := json.Marshal(config)
			object["outputConfig"] = leftover
		}
	}
	if raw, ok := object["toolConfig"]; ok {
		delete(object, "toolConfig")
		request.Tools, request.ToolChoice, diagnostics, err = decodeBedrockTools(raw, policy, diagnostics)
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

func (Bedrock) EncodeRequest(request llmprotocol.Request, policy llmprotocol.Policy) (WireResult, error) {
	const format = llmprotocol.FormatBedrock
	if raw, ok := preservedRequest(request, format, policy); ok {
		return WireResult{Body: raw}, nil
	}
	var diagnostics []llmprotocol.Diagnostic
	if err := validateGenerationBounds(format, request, 1, 4); err != nil {
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
				return WireResult{}, translationError(format, "$.system", "invalid_instruction_role", "Bedrock system accepts system or developer instructions")
			}
			blocks = append(blocks, instruction.Content...)
		}
		encoded, next, err := encodeBedrockContent(blocks, policy, "$.system")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{}, err
		}
		object["system"] = encoded
	}
	messages := make([]any, 0, len(request.Messages))
	for index, message := range request.Messages {
		role := message.Role
		if role == llmprotocol.RoleTool {
			role = llmprotocol.RoleUser
		}
		if role != llmprotocol.RoleUser && role != llmprotocol.RoleAssistant {
			return WireResult{}, translationError(format, fmt.Sprintf("$.messages[%d].role", index), "unsupported_role", "Bedrock messages accept user or assistant roles")
		}
		content, next, err := encodeBedrockContent(message.Content, policy, fmt.Sprintf("$.messages[%d].content", index))
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{}, err
		}
		value := map[string]any{"role": string(role), "content": content}
		mergeExtensions(value, message.Extensions)
		messages = append(messages, value)
	}
	object["messages"] = messages
	inference := map[string]any{}
	putInt(inference, "maxTokens", request.Output.MaxTokens)
	if len(request.Output.StopSequences) > 0 {
		inference["stopSequences"] = request.Output.StopSequences
	}
	if request.Sampling.Temperature != nil {
		inference["temperature"] = *request.Sampling.Temperature
	}
	if request.Sampling.TopP != nil {
		inference["topP"] = *request.Sampling.TopP
	}
	if request.Sampling.TopK != nil {
		next, err := lossy(format, policy, "$.sampling.top_k", "top_k_not_supported")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{}, err
		}
	}
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
	if len(inference) > 0 {
		object["inferenceConfig"] = inference
	}
	if len(request.Output.ResponseFormat) != 0 {
		value, err := encodeBedrockResponseFormat(request.Output.ResponseFormat)
		if err != nil {
			return WireResult{}, err
		}
		object["outputConfig"] = map[string]any{"textFormat": value}
	}
	if request.Reasoning.Effort != "" || request.Reasoning.BudgetTokens != nil || request.Reasoning.Include != nil || request.Reasoning.Level != "" || len(request.Reasoning.Raw) != 0 {
		next, err := lossy(format, policy, "$.reasoning", "reasoning_config_not_portable")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{}, err
		}
	}
	if len(request.Tools) > 0 {
		tools := make([]any, 0, len(request.Tools))
		for _, tool := range request.Tools {
			if tool.Type != "" && tool.Type != "function" {
				return WireResult{}, translationError(format, "$.toolConfig.tools[]", "hosted_tool_not_supported", "Bedrock codec supports function tools only")
			}
			spec := map[string]any{"name": tool.Name, "inputSchema": map[string]any{"json": rawOrEmptyObject(tool.Parameters)}}
			if tool.Description != "" {
				spec["description"] = tool.Description
			}
			if tool.Strict != nil {
				spec["strict"] = *tool.Strict
			}
			mergeExtensions(spec, tool.Extensions)
			tools = append(tools, map[string]any{"toolSpec": spec})
		}
		config := map[string]any{"tools": tools}
		if request.ToolChoice != nil {
			choice, err := encodeBedrockToolChoice(*request.ToolChoice)
			if err != nil {
				return WireResult{}, err
			}
			if choice != nil {
				config["toolChoice"] = choice
			}
		}
		object["toolConfig"] = config
	} else if request.ToolChoice != nil && request.ToolChoice.Type != llmprotocol.ToolChoiceNone {
		return WireResult{}, translationError(format, "$.toolConfig.toolChoice", "tool_choice_without_tools", "Bedrock tool choice requires tools")
	}
	if request.ParallelToolCalls != nil && !*request.ParallelToolCalls {
		next, err := lossy(format, policy, "$.parallel_tool_calls", "parallel_tool_control_not_supported")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return WireResult{}, err
		}
	}
	mergeExtensions(object, request.Extensions)
	body, err := marshalObject(format, object)
	return WireResult{Body: body, Diagnostics: diagnostics}, err
}

func (Bedrock) DecodeResponse(body json.RawMessage, policy llmprotocol.Policy) (ResponseResult, error) {
	const format = llmprotocol.FormatBedrock
	object, err := decodeObject(format, body)
	if err != nil {
		return ResponseResult{}, err
	}
	response := llmprotocol.Response{Preservation: preserveBody(policy, format, body, false)}
	var diagnostics []llmprotocol.Diagnostic
	output := llmprotocol.ResponseOutput{Role: llmprotocol.RoleAssistant}
	if rawOutput, ok := object["output"]; ok {
		delete(object, "output")
		outputObject, objectErr := decodeObject(format, rawOutput)
		if objectErr != nil {
			return ResponseResult{}, objectErr
		}
		messageObject, objectErr := decodeObject(format, outputObject["message"])
		if objectErr != nil {
			return ResponseResult{}, objectErr
		}
		roleName, roleErr := optionalString(format, messageObject, "role")
		if roleErr != nil {
			return ResponseResult{}, roleErr
		}
		if roleName != "assistant" {
			return ResponseResult{}, translationError(format, "$.output.message.role", "unsupported_role", "Bedrock output role must be assistant")
		}
		output.Content, diagnostics, err = decodeBedrockContent(messageObject["content"], policy, "$.output.message.content")
		if err != nil {
			return ResponseResult{}, err
		}
		delete(messageObject, "content")
		output.Extensions, diagnostics, err = collectAndAppendExtensions(format, messageObject, policy, diagnostics)
		if err != nil {
			return ResponseResult{}, err
		}
		delete(outputObject, "message")
	}
	stop, stopErr := optionalString(format, object, "stopReason")
	if stopErr != nil {
		return ResponseResult{}, stopErr
	}
	output.StopReason = decodeBedrockStopReason(stop)
	response.Outputs = []llmprotocol.ResponseOutput{output}
	if rawUsage, ok := object["usage"]; ok {
		delete(object, "usage")
		response.Usage, err = decodeBedrockUsage(rawUsage)
		if err != nil {
			return ResponseResult{}, err
		}
	}
	delete(object, "metrics")
	delete(object, "trace")
	delete(object, "performanceConfig")
	response.Extensions, diagnostics, err = collectAndAppendExtensions(format, object, policy, diagnostics)
	if err != nil {
		return ResponseResult{}, err
	}
	return ResponseResult{Response: response, Diagnostics: diagnostics}, nil
}

func (Bedrock) EncodeResponse(response llmprotocol.Response, policy llmprotocol.Policy) (WireResult, error) {
	const format = llmprotocol.FormatBedrock
	if raw, ok := preservedResponse(response, format, policy); ok {
		return WireResult{Body: raw}, nil
	}
	if len(response.Outputs) != 1 {
		return WireResult{}, translationError(format, "$.output", "single_output_required", "Bedrock Converse has exactly one output message")
	}
	output := response.Outputs[0]
	if output.Role != "" && output.Role != llmprotocol.RoleAssistant {
		return WireResult{}, translationError(format, "$.output.message.role", "unsupported_role", "Bedrock output role must be assistant")
	}
	content, diagnostics, err := encodeBedrockContent(output.Content, policy, "$.output.message.content")
	if err != nil {
		return WireResult{}, err
	}
	if len(output.Logprobs) > 0 {
		next, nextErr := lossy(format, policy, "$.outputs[].logprobs", "logprobs_not_supported")
		diagnostics = append(diagnostics, next...)
		if nextErr != nil {
			return WireResult{Diagnostics: diagnostics}, nextErr
		}
	}
	message := map[string]any{"role": "assistant", "content": content}
	mergeExtensions(message, output.Extensions)
	object := map[string]any{"output": map[string]any{"message": message}, "stopReason": encodeBedrockStopReason(output.StopReason)}
	if hasUsage(response.Usage) {
		object["usage"] = encodeBedrockUsage(response.Usage)
	}
	mergeExtensions(object, response.Extensions)
	body, err := marshalObject(format, object)
	return WireResult{Body: body, Diagnostics: diagnostics}, err
}

func decodeBedrockContent(raw json.RawMessage, policy llmprotocol.Policy, path string) ([]llmprotocol.ContentBlock, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatBedrock
	values, err := decodeArray(format, path, raw)
	if err != nil {
		return nil, nil, err
	}
	blocks := make([]llmprotocol.ContentBlock, 0, len(values))
	var diagnostics []llmprotocol.Diagnostic
	for index, value := range values {
		block, next, decodeErr := decodeBedrockBlock(value, policy, fmt.Sprintf("%s[%d]", path, index))
		diagnostics = append(diagnostics, next...)
		if decodeErr != nil {
			return nil, diagnostics, decodeErr
		}
		if block != nil {
			blocks = append(blocks, *block)
		}
	}
	return blocks, diagnostics, nil
}

func decodeBedrockBlock(raw json.RawMessage, policy llmprotocol.Policy, path string) (*llmprotocol.ContentBlock, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatBedrock
	object, err := decodeObject(format, raw)
	if err != nil {
		return nil, nil, err
	}
	if rawText, ok := object["text"]; ok {
		text, textErr := decodeString(format, path+".text", rawText)
		if textErr != nil {
			return nil, nil, textErr
		}
		return &llmprotocol.ContentBlock{Type: llmprotocol.ContentText, Text: text, Provider: format}, nil, nil
	}
	if rawReasoning, ok := object["reasoningContent"]; ok {
		reasoning, objectErr := decodeObject(format, rawReasoning)
		if objectErr != nil {
			return nil, nil, objectErr
		}
		textObject, objectErr := decodeObject(format, reasoning["reasoningText"])
		if objectErr != nil {
			return nil, nil, objectErr
		}
		text, textErr := optionalString(format, textObject, "text")
		if textErr != nil {
			return nil, nil, textErr
		}
		signature, signatureErr := optionalString(format, textObject, "signature")
		if signatureErr != nil {
			return nil, nil, signatureErr
		}
		return &llmprotocol.ContentBlock{Type: llmprotocol.ContentReasoning, Text: text, Signature: signature, Provider: format}, nil, nil
	}
	if rawCall, ok := object["toolUse"]; ok {
		call, objectErr := decodeObject(format, rawCall)
		if objectErr != nil {
			return nil, nil, objectErr
		}
		id, idErr := optionalString(format, call, "toolUseId")
		if idErr != nil {
			return nil, nil, idErr
		}
		name, nameErr := optionalString(format, call, "name")
		if nameErr != nil {
			return nil, nil, nameErr
		}
		arguments := cloneRaw(call["input"])
		return &llmprotocol.ContentBlock{Type: llmprotocol.ContentToolCall, Provider: format, ToolCall: &llmprotocol.ToolCall{ID: id, Name: name, Arguments: rawOrEmptyObject(arguments)}}, nil, nil
	}
	if rawResult, ok := object["toolResult"]; ok {
		result, objectErr := decodeObject(format, rawResult)
		if objectErr != nil {
			return nil, nil, objectErr
		}
		id, idErr := optionalString(format, result, "toolUseId")
		if idErr != nil {
			return nil, nil, idErr
		}
		status, statusErr := optionalString(format, result, "status")
		if statusErr != nil {
			return nil, nil, statusErr
		}
		content, next, decodeErr := decodeBedrockContent(result["content"], policy, path+".toolResult.content")
		if decodeErr != nil {
			return nil, next, decodeErr
		}
		var isError *bool
		if status != "" {
			value := status == "error"
			isError = &value
		}
		return &llmprotocol.ContentBlock{Type: llmprotocol.ContentToolResult, Provider: format, Result: &llmprotocol.ToolResult{ToolCallID: id, Content: content, IsError: isError}}, next, nil
	}
	for _, media := range []struct {
		name string
		kind llmprotocol.ContentType
	}{{"image", llmprotocol.ContentImage}, {"video", llmprotocol.ContentVideo}, {"document", llmprotocol.ContentFile}} {
		if rawMedia, ok := object[media.name]; ok {
			mediaObject, objectErr := decodeObject(format, rawMedia)
			if objectErr != nil {
				return nil, nil, objectErr
			}
			formatName, formatErr := optionalString(format, mediaObject, "format")
			if formatErr != nil {
				return nil, nil, formatErr
			}
			name, nameErr := optionalString(format, mediaObject, "name")
			if nameErr != nil {
				return nil, nil, nameErr
			}
			rawSource := cloneRaw(mediaObject["source"])
			sourceObject, sourceErr := decodeObject(format, rawSource)
			if sourceErr != nil {
				return nil, nil, sourceErr
			}
			source := &llmprotocol.Source{Filename: name, MediaType: bedrockMediaType(media.name, formatName), Raw: rawSource}
			if bytesValue, exists := sourceObject["bytes"]; exists {
				source.Kind = "base64"
				source.Data, err = decodeString(format, path+"."+media.name+".source.bytes", bytesValue)
				if err != nil {
					return nil, nil, err
				}
			} else {
				source.Kind = "provider_file"
			}
			return &llmprotocol.ContentBlock{Type: media.kind, Source: source, Provider: format}, nil, nil
		}
	}
	block, diagnostics, err := decodeUnknownBlock(format, raw, policy, path)
	return block, diagnostics, err
}

func encodeBedrockContent(blocks []llmprotocol.ContentBlock, policy llmprotocol.Policy, path string) ([]any, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatBedrock
	values := make([]any, 0, len(blocks))
	var diagnostics []llmprotocol.Diagnostic
	for index, block := range blocks {
		blockPath := fmt.Sprintf("%s[%d]", path, index)
		value := map[string]any{}
		switch block.Type {
		case llmprotocol.ContentText:
			value["text"] = block.Text
		case llmprotocol.ContentReasoning:
			reasoning := map[string]any{"text": block.Text}
			if block.Signature != "" {
				reasoning["signature"] = block.Signature
			}
			value["reasoningContent"] = map[string]any{"reasoningText": reasoning}
		case llmprotocol.ContentToolCall:
			if block.ToolCall == nil {
				return nil, diagnostics, translationError(format, blockPath, "missing_tool_call", "tool call is required")
			}
			value["toolUse"] = map[string]any{"toolUseId": block.ToolCall.ID, "name": block.ToolCall.Name, "input": rawOrEmptyObject(block.ToolCall.Arguments)}
		case llmprotocol.ContentToolResult:
			if block.Result == nil {
				return nil, diagnostics, translationError(format, blockPath, "missing_tool_result", "tool result is required")
			}
			content, next, err := encodeBedrockContent(block.Result.Content, policy, blockPath+".toolResult.content")
			diagnostics = append(diagnostics, next...)
			if err != nil {
				return nil, diagnostics, err
			}
			result := map[string]any{"toolUseId": block.Result.ToolCallID, "content": content}
			if block.Result.IsError != nil {
				if *block.Result.IsError {
					result["status"] = "error"
				} else {
					result["status"] = "success"
				}
			}
			value["toolResult"] = result
		case llmprotocol.ContentImage, llmprotocol.ContentVideo, llmprotocol.ContentFile:
			if block.Source == nil {
				return nil, diagnostics, translationError(format, blockPath, "missing_media_source", "media source is required")
			}
			if dataMime, _, ok := splitDataURL(block.Source.URL); ok && block.Source.MediaType == "" {
				block.Source.MediaType = dataMime
			}
			name, formatName := bedrockMediaIdentity(block.Type, block.Source)
			source := any(nil)
			if block.Provider == format && len(block.Source.Raw) != 0 {
				source = json.RawMessage(cloneRaw(block.Source.Raw))
			} else if block.Source.Data != "" || block.Source.Kind == "base64" {
				source = map[string]any{"bytes": block.Source.Data}
			} else if dataMime, data, ok := splitDataURL(block.Source.URL); ok {
				if block.Source.MediaType == "" {
					block.Source.MediaType = dataMime
				}
				source = map[string]any{"bytes": data}
			} else {
				return nil, diagnostics, translationError(format, blockPath, "unsupported_media_source", "Bedrock requires inline bytes or a native source")
			}
			media := map[string]any{"format": formatName, "source": source}
			if block.Type == llmprotocol.ContentFile {
				media["name"] = name
			}
			field := "image"
			if block.Type == llmprotocol.ContentVideo {
				field = "video"
			}
			if block.Type == llmprotocol.ContentFile {
				field = "document"
			}
			value[field] = media
		case llmprotocol.ContentUnknown:
			if block.Provider == format && json.Valid(block.Raw) {
				values = append(values, json.RawMessage(cloneRaw(block.Raw)))
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
		mergeExtensions(value, block.Extensions)
		values = append(values, value)
	}
	return values, diagnostics, nil
}

func decodeBedrockTools(raw json.RawMessage, policy llmprotocol.Policy, diagnostics []llmprotocol.Diagnostic) ([]llmprotocol.ToolDefinition, *llmprotocol.ToolChoice, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatBedrock
	object, err := decodeObject(format, raw)
	if err != nil {
		return nil, nil, diagnostics, err
	}
	var tools []llmprotocol.ToolDefinition
	if rawTools, ok := object["tools"]; ok {
		delete(object, "tools")
		values, arrayErr := decodeArray(format, "$.toolConfig.tools", rawTools)
		if arrayErr != nil {
			return nil, nil, diagnostics, arrayErr
		}
		for _, rawTool := range values {
			wrapper, objectErr := decodeObject(format, rawTool)
			if objectErr != nil {
				return nil, nil, diagnostics, objectErr
			}
			spec, objectErr := decodeObject(format, wrapper["toolSpec"])
			if objectErr != nil {
				return nil, nil, diagnostics, objectErr
			}
			tool := llmprotocol.ToolDefinition{Type: "function"}
			tool.Name, err = optionalString(format, spec, "name")
			if err != nil {
				return nil, nil, diagnostics, err
			}
			tool.Description, err = optionalString(format, spec, "description")
			if err != nil {
				return nil, nil, diagnostics, err
			}
			if rawSchema, exists := spec["inputSchema"]; exists {
				schemaObject, schemaErr := decodeObject(format, rawSchema)
				if schemaErr != nil {
					return nil, nil, diagnostics, schemaErr
				}
				tool.Parameters = cloneRaw(schemaObject["json"])
				delete(spec, "inputSchema")
			}
			tool.Strict, err = optionalBool(format, spec, "strict")
			if err != nil {
				return nil, nil, diagnostics, err
			}
			tool.Extensions, diagnostics, err = collectAndAppendExtensions(format, spec, policy, diagnostics)
			if err != nil {
				return nil, nil, diagnostics, err
			}
			tools = append(tools, tool)
		}
	}
	var choice *llmprotocol.ToolChoice
	if rawChoice, ok := object["toolChoice"]; ok {
		delete(object, "toolChoice")
		choice, err = decodeBedrockToolChoice(rawChoice)
		if err != nil {
			return nil, nil, diagnostics, err
		}
	}
	_, next, collectErr := collectExtensions(format, object, policy)
	diagnostics = append(diagnostics, next...)
	return tools, choice, diagnostics, collectErr
}

func decodeBedrockResponseFormat(raw json.RawMessage) (json.RawMessage, error) {
	const format = llmprotocol.FormatBedrock
	object, err := decodeObject(format, raw)
	if err != nil {
		return nil, err
	}
	typeName, err := optionalString(format, object, "type")
	if err != nil {
		return nil, err
	}
	if typeName != "json_schema" {
		return nil, translationError(format, "$.outputConfig.textFormat.type", "unsupported_response_format", "Bedrock text format must be json_schema")
	}
	structure, err := decodeObject(format, object["structure"])
	if err != nil {
		return nil, err
	}
	delete(object, "structure")
	definition, err := decodeObject(format, structure["jsonSchema"])
	if err != nil {
		return nil, err
	}
	delete(structure, "jsonSchema")
	name, err := optionalString(format, definition, "name")
	if err == nil {
		var description string
		description, err = optionalString(format, definition, "description")
		if err == nil {
			schemaText, schemaErr := optionalString(format, definition, "schema")
			if schemaErr != nil {
				return nil, schemaErr
			}
			if !json.Valid([]byte(schemaText)) {
				return nil, translationError(format, "$.outputConfig.textFormat.structure.jsonSchema.schema", "invalid_schema", "Bedrock JSON schema must contain valid JSON text")
			}
			if len(object) != 0 || len(structure) != 0 || len(definition) != 0 {
				return nil, translationError(format, "$.outputConfig.textFormat", "unsupported_response_format", "text format contains unsupported fields")
			}
			strict := true
			return encodeNeutralJSONSchema(neutralResponseFormat{Type: "json_schema", Name: name, Description: description, Schema: json.RawMessage(schemaText), Strict: &strict}), nil
		}
	}
	return nil, err
}

func encodeBedrockResponseFormat(raw json.RawMessage) (map[string]any, error) {
	const format = llmprotocol.FormatBedrock
	value, err := decodeNeutralResponseFormat(format, "$.output.response_format", raw)
	if err != nil {
		return nil, err
	}
	if value.Type == "json_object" {
		value.Type = "json_schema"
		value.Schema = json.RawMessage(`{"type":"object"}`)
	}
	if value.Type != "json_schema" || len(value.Schema) == 0 {
		return nil, translationError(format, "$.output.response_format", "unsupported_response_format", "response format cannot be represented by Bedrock")
	}
	definition := map[string]any{"schema": string(value.Schema)}
	if value.Name != "" {
		definition["name"] = value.Name
	}
	if value.Description != "" {
		definition["description"] = value.Description
	}
	return map[string]any{"type": "json_schema", "structure": map[string]any{"jsonSchema": definition}}, nil
}

func decodeBedrockToolChoice(raw json.RawMessage) (*llmprotocol.ToolChoice, error) {
	const format = llmprotocol.FormatBedrock
	object, err := decodeObject(format, raw)
	if err != nil {
		return nil, err
	}
	if _, ok := object["auto"]; ok {
		return &llmprotocol.ToolChoice{Type: llmprotocol.ToolChoiceAuto}, nil
	}
	if _, ok := object["any"]; ok {
		return &llmprotocol.ToolChoice{Type: llmprotocol.ToolChoiceRequired}, nil
	}
	if rawTool, ok := object["tool"]; ok {
		tool, objectErr := decodeObject(format, rawTool)
		if objectErr != nil {
			return nil, objectErr
		}
		name, nameErr := optionalString(format, tool, "name")
		if nameErr != nil {
			return nil, nameErr
		}
		return &llmprotocol.ToolChoice{Type: llmprotocol.ToolChoiceTool, Name: name}, nil
	}
	return nil, translationError(format, "$.toolConfig.toolChoice", "unsupported_tool_choice", "unknown Bedrock tool choice")
}

func encodeBedrockToolChoice(choice llmprotocol.ToolChoice) (any, error) {
	switch choice.Type {
	case llmprotocol.ToolChoiceAuto:
		return map[string]any{"auto": map[string]any{}}, nil
	case llmprotocol.ToolChoiceRequired:
		return map[string]any{"any": map[string]any{}}, nil
	case llmprotocol.ToolChoiceTool:
		return map[string]any{"tool": map[string]any{"name": choice.Name}}, nil
	case llmprotocol.ToolChoiceNone:
		return nil, nil
	default:
		return nil, translationError(llmprotocol.FormatBedrock, "$.toolConfig.toolChoice", "unsupported_tool_choice", "tool choice cannot be represented by Bedrock")
	}
}

func decodeBedrockUsage(raw json.RawMessage) (llmprotocol.Usage, error) {
	const format = llmprotocol.FormatBedrock
	object, err := decodeObject(format, raw)
	if err != nil {
		return llmprotocol.Usage{}, err
	}
	usage := llmprotocol.Usage{}
	usage.InputTokens, err = optionalInt(format, object, "inputTokens")
	if err != nil {
		return usage, err
	}
	usage.OutputTokens, err = optionalInt(format, object, "outputTokens")
	if err != nil {
		return usage, err
	}
	usage.TotalTokens, err = optionalInt(format, object, "totalTokens")
	if err != nil {
		return usage, err
	}
	usage.CachedInputTokens, err = optionalInt(format, object, "cacheReadInputTokens")
	if err != nil {
		return usage, err
	}
	usage.CacheCreationTokens, err = optionalInt(format, object, "cacheWriteInputTokens")
	if err != nil {
		return usage, err
	}
	return usage, nil
}

func encodeBedrockUsage(usage llmprotocol.Usage) map[string]any {
	object := map[string]any{}
	putInt(object, "inputTokens", usage.InputTokens)
	putInt(object, "outputTokens", usage.OutputTokens)
	putInt(object, "totalTokens", usage.TotalTokens)
	putInt(object, "cacheReadInputTokens", usage.CachedInputTokens)
	putInt(object, "cacheWriteInputTokens", usage.CacheCreationTokens)
	return object
}

func decodeBedrockStopReason(reason string) llmprotocol.StopReason {
	switch reason {
	case "":
		return ""
	case "end_turn", "stop_sequence":
		return llmprotocol.StopEndTurn
	case "max_tokens", "model_context_window_exceeded":
		return llmprotocol.StopMaxTokens
	case "tool_use":
		return llmprotocol.StopToolUse
	case "content_filtered", "guardrail_intervened":
		return llmprotocol.StopContentFilter
	default:
		return llmprotocol.StopUnknown
	}
}

func encodeBedrockStopReason(reason llmprotocol.StopReason) string {
	switch reason {
	case llmprotocol.StopMaxTokens:
		return "max_tokens"
	case llmprotocol.StopToolUse:
		return "tool_use"
	case llmprotocol.StopContentFilter:
		return "content_filtered"
	default:
		return "end_turn"
	}
}

func bedrockMediaType(kind, format string) string {
	if strings.Contains(format, "/") {
		return format
	}
	switch kind {
	case "image":
		return "image/" + format
	case "video":
		return "video/" + format
	case "document":
		if format == "txt" {
			return "text/plain"
		}
		if format == "pdf" {
			return "application/pdf"
		}
	}
	return "application/" + format
}

func bedrockMediaIdentity(kind llmprotocol.ContentType, source *llmprotocol.Source) (string, string) {
	formatName := strings.TrimPrefix(source.MediaType, "image/")
	formatName = strings.TrimPrefix(formatName, "video/")
	formatName = strings.TrimPrefix(formatName, "application/")
	if source.MediaType == "text/plain" {
		formatName = "txt"
	}
	if formatName == "" || strings.Contains(formatName, "/") {
		formatName = "bin"
	}
	name := strings.TrimSuffix(filepath.Base(source.Filename), filepath.Ext(source.Filename))
	if name == "" || name == "." {
		name = "document"
	}
	return name, formatName
}
