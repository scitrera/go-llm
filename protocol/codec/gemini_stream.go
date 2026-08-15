// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package codec

import (
	"bytes"
	"encoding/json"
	"fmt"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

func (Gemini) DecodeStreamEvent(state *StreamState, wire llmprotocol.WireEvent, policy llmprotocol.Policy) ([]llmprotocol.StreamEvent, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatGemini
	if len(bytes.TrimSpace(wire.Data)) == 0 {
		return nil, nil, nil
	}
	decoded, err := (Gemini{}).DecodeResponse(wire.Data, policy)
	if err != nil {
		return nil, nil, err
	}
	if decoded.Response.ID != "" {
		state.ResponseID = decoded.Response.ID
	}
	if decoded.Response.Model != "" {
		state.Model = decoded.Response.Model
	}
	values := state.valueMap()
	var events []llmprotocol.StreamEvent
	if started, _ := values["gemini_response_started"].(bool); !started {
		values["gemini_response_started"] = true
		events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamResponseStart, ResponseID: state.ResponseID, Model: state.Model})
	}
	for fallback, output := range decoded.Response.Outputs {
		index := fallback
		if output.Index != nil {
			index = *output.Index
		}
		startKey := fmt.Sprintf("gemini_output_started:%d", index)
		if started, _ := values[startKey].(bool); !started {
			values[startKey] = true
			events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamOutputStart, ResponseID: state.ResponseID, Model: state.Model, OutputIndex: index, Role: llmprotocol.RoleAssistant})
		}
		for contentIndex, block := range output.Content {
			switch block.Type {
			case llmprotocol.ContentText:
				events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamTextDelta, ResponseID: state.ResponseID, Model: state.Model, OutputIndex: index, ContentIndex: contentIndex, Delta: block.Text})
			case llmprotocol.ContentReasoning:
				events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamReasoningDelta, ResponseID: state.ResponseID, Model: state.Model, OutputIndex: index, ContentIndex: contentIndex, Delta: block.Text})
				if block.Signature != "" {
					events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamReasoningSignatureDelta, ResponseID: state.ResponseID, Model: state.Model, OutputIndex: index, ContentIndex: contentIndex, Signature: block.Signature, Provider: format})
				}
			case llmprotocol.ContentToolCall:
				if block.ToolCall != nil {
					events = append(events,
						llmprotocol.StreamEvent{Type: llmprotocol.StreamToolCallStart, ResponseID: state.ResponseID, Model: state.Model, OutputIndex: index, ContentIndex: contentIndex, ToolCallID: block.ToolCall.ID, ToolName: block.ToolCall.Name, Signature: block.Signature, Provider: format},
						llmprotocol.StreamEvent{Type: llmprotocol.StreamToolArgsDelta, ResponseID: state.ResponseID, Model: state.Model, OutputIndex: index, ContentIndex: contentIndex, ToolCallID: block.ToolCall.ID, Delta: string(rawOrEmptyObject(block.ToolCall.Arguments))},
					)
				}
			}
		}
		if len(output.Logprobs) > 0 {
			events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamLogprobs, ResponseID: state.ResponseID, Model: state.Model, OutputIndex: index, Logprobs: output.Logprobs})
		}
		if output.StopReason != "" && output.StopReason != llmprotocol.StopUnknown {
			doneKey := fmt.Sprintf("gemini_output_done:%d", index)
			if done, _ := values[doneKey].(bool); !done {
				values[doneKey] = true
				events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamOutputDone, ResponseID: state.ResponseID, Model: state.Model, OutputIndex: index, StopReason: output.StopReason})
			}
		}
	}
	if hasUsage(decoded.Response.Usage) {
		usage := decoded.Response.Usage
		events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamUsage, ResponseID: state.ResponseID, Model: state.Model, Usage: &usage})
	}
	return events, decoded.Diagnostics, nil
}

func (Gemini) EncodeStreamEvent(state *StreamState, event llmprotocol.StreamEvent, policy llmprotocol.Policy) ([]llmprotocol.WireEvent, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatGemini
	if event.ResponseID != "" {
		state.ResponseID = event.ResponseID
	}
	if event.Model != "" {
		state.Model = event.Model
	}
	base := func() map[string]any {
		value := map[string]any{}
		if state.ResponseID != "" {
			value["responseId"] = state.ResponseID
		}
		if state.Model != "" {
			value["modelVersion"] = state.Model
		}
		return value
	}
	object := base()
	values := state.valueMap()
	var candidate map[string]any
	part := map[string]any{}
	switch event.Type {
	case llmprotocol.StreamResponseStart, llmprotocol.StreamOutputStart, llmprotocol.StreamResponseDone:
		return nil, nil, nil
	case llmprotocol.StreamTextDelta:
		part["text"] = event.Delta
	case llmprotocol.StreamReasoningDelta:
		part["text"] = event.Delta
		part["thought"] = true
	case llmprotocol.StreamReasoningSignatureDelta:
		part["thought"] = true
		part["thoughtSignature"] = event.Signature
	case llmprotocol.StreamToolCallStart:
		values[streamKey("gemini_tool_id", event.OutputIndex, event.ContentIndex)] = event.ToolCallID
		values[streamKey("gemini_tool_name", event.OutputIndex, event.ContentIndex)] = event.ToolName
		if event.Signature != "" {
			values[streamKey("gemini_tool_signature", event.OutputIndex, event.ContentIndex)] = event.Signature
		}
		return nil, nil, nil
	case llmprotocol.StreamToolArgsDelta:
		name, _ := values[streamKey("gemini_tool_name", event.OutputIndex, event.ContentIndex)].(string)
		id, _ := values[streamKey("gemini_tool_id", event.OutputIndex, event.ContentIndex)].(string)
		arguments := json.RawMessage(event.Delta)
		if !json.Valid(arguments) {
			return nil, nil, translationError(format, "$stream.functionCall.args", "invalid_tool_arguments", "Gemini functionCall arguments must be complete JSON")
		}
		part["functionCall"] = map[string]any{"id": id, "name": name, "args": arguments}
		if signature, _ := values[streamKey("gemini_tool_signature", event.OutputIndex, event.ContentIndex)].(string); signature != "" {
			part["thoughtSignature"] = signature
		}
	case llmprotocol.StreamOutputDone:
		candidate = map[string]any{"index": event.OutputIndex, "finishReason": encodeGeminiStopReason(event.StopReason)}
	case llmprotocol.StreamUsage:
		if event.Usage != nil {
			object["usageMetadata"] = encodeGeminiUsage(*event.Usage)
		}
	case llmprotocol.StreamLogprobs:
		candidate = map[string]any{"index": event.OutputIndex, "logprobsResult": encodeGeminiLogprobs(event.Logprobs)}
	case llmprotocol.StreamError:
		message := "upstream stream failed"
		if event.Error != nil {
			message = event.Error.Message
		}
		object = map[string]any{"error": map[string]any{"message": message}}
	case llmprotocol.StreamUnknown:
		diagnostics, err := lossy(format, policy, "$stream", "unknown_stream_event")
		return nil, diagnostics, err
	default:
		return nil, nil, nil
	}
	if len(part) > 0 {
		candidate = map[string]any{"index": event.OutputIndex, "content": map[string]any{"role": "model", "parts": []any{part}}}
	}
	if candidate != nil {
		object["candidates"] = []any{candidate}
	}
	body, err := marshalObject(format, object)
	if err != nil {
		return nil, nil, err
	}
	return []llmprotocol.WireEvent{{Data: body}}, nil, nil
}

func (Gemini) FinishStream(_ *StreamState, _ llmprotocol.Policy) ([]llmprotocol.WireEvent, []llmprotocol.Diagnostic, error) {
	return nil, nil, nil
}
