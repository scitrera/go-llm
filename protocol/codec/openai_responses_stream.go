// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package codec

import (
	"bytes"
	"encoding/json"
	"fmt"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

func (OpenAIResponses) DecodeStreamEvent(state *StreamState, wire llmprotocol.WireEvent, policy llmprotocol.Policy) ([]llmprotocol.StreamEvent, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatOpenAIResponses
	if bytes.Equal(bytes.TrimSpace(wire.Data), []byte("[DONE]")) {
		return []llmprotocol.StreamEvent{{Type: llmprotocol.StreamResponseDone, ResponseID: state.ResponseID, Model: state.Model}}, nil, nil
	}
	object, err := decodeObject(format, wire.Data)
	if err != nil {
		return nil, nil, err
	}
	typeName := wire.Event
	if embedded, typeErr := optionalString(format, object, "type"); typeErr != nil {
		return nil, nil, typeErr
	} else if typeName == "" {
		typeName = embedded
	}
	delete(object, "sequence_number")
	base := llmprotocol.StreamEvent{ResponseID: state.ResponseID, Model: state.Model}
	if rawResponse, ok := object["response"]; ok {
		var envelope struct {
			ID    string          `json:"id"`
			Model string          `json:"model"`
			Usage json.RawMessage `json:"usage"`
		}
		if err := json.Unmarshal(rawResponse, &envelope); err != nil {
			return nil, nil, translationError(format, "$.response", "invalid_response", "response must be an object")
		}
		if envelope.ID != "" {
			state.ResponseID = envelope.ID
		}
		if envelope.Model != "" {
			state.Model = envelope.Model
		}
		base.ResponseID, base.Model = state.ResponseID, state.Model
	}
	outputIndex := 0
	if value, valueErr := optionalInt(format, object, "output_index"); valueErr != nil {
		return nil, nil, valueErr
	} else if value != nil {
		outputIndex = int(*value)
	}
	contentIndex := 0
	if value, valueErr := optionalInt(format, object, "content_index"); valueErr != nil {
		return nil, nil, valueErr
	} else if value != nil {
		contentIndex = int(*value)
	}
	base.OutputIndex, base.ContentIndex = outputIndex, contentIndex
	base.ItemID, err = optionalString(format, object, "item_id")
	if err != nil {
		return nil, nil, err
	}
	var events []llmprotocol.StreamEvent
	switch typeName {
	case "response.created", "response.in_progress", "response.queued":
		base.Type = llmprotocol.StreamResponseStart
		events = append(events, base)
	case "response.output_item.added":
		rawItem := object["item"]
		delete(object, "item")
		item, itemErr := decodeObject(format, rawItem)
		if itemErr != nil {
			return nil, nil, itemErr
		}
		itemType, typeErr := optionalString(format, item, "type")
		if typeErr != nil {
			return nil, nil, typeErr
		}
		id, idErr := optionalString(format, item, "id")
		if idErr != nil {
			return nil, nil, idErr
		}
		base.ItemID = id
		switch itemType {
		case "function_call":
			base.Type = llmprotocol.StreamToolCallStart
			base.ToolCallID, err = optionalString(format, item, "call_id")
			if err != nil {
				return nil, nil, err
			}
			base.ToolName, err = optionalString(format, item, "name")
			if err != nil {
				return nil, nil, err
			}
		case "message":
			base.Type = llmprotocol.StreamOutputStart
			role, roleErr := optionalString(format, item, "role")
			if roleErr != nil {
				return nil, nil, roleErr
			}
			base.Role, err = validateRole(format, "$.item.role", role)
			if err != nil {
				return nil, nil, err
			}
		case "reasoning":
			base.Type = llmprotocol.StreamOutputStart
			base.Role = llmprotocol.RoleAssistant
		default:
			base.Type = llmprotocol.StreamUnknown
			base.Provider = format
			base.Raw = cloneRaw(rawItem)
		}
		events = append(events, base)
	case "response.output_text.delta":
		base.Type = llmprotocol.StreamTextDelta
		base.Delta, err = optionalString(format, object, "delta")
		if err != nil {
			return nil, nil, err
		}
		events = append(events, base)
	case "response.refusal.delta":
		base.Type = llmprotocol.StreamRefusalDelta
		base.Delta, err = optionalString(format, object, "delta")
		if err != nil {
			return nil, nil, err
		}
		events = append(events, base)
	case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
		base.Type = llmprotocol.StreamReasoningDelta
		base.Delta, err = optionalString(format, object, "delta")
		if err != nil {
			return nil, nil, err
		}
		events = append(events, base)
	case "response.function_call_arguments.delta":
		base.Type = llmprotocol.StreamToolArgsDelta
		base.Delta, err = optionalString(format, object, "delta")
		if err != nil {
			return nil, nil, err
		}
		events = append(events, base)
	case "response.output_item.done":
		base.Type = llmprotocol.StreamOutputDone
		base.StopReason = llmprotocol.StopEndTurn
		if rawItem, ok := object["item"]; ok {
			var item struct{ Type, ID, CallID string }
			_ = json.Unmarshal(rawItem, &item)
			if item.Type == "function_call" {
				base.StopReason = llmprotocol.StopToolUse
			}
			if base.ItemID == "" {
				base.ItemID = item.ID
			}
			if base.ToolCallID == "" {
				base.ToolCallID = item.CallID
			}
		}
		events = append(events, base)
	case "response.completed", "response.incomplete", "response.failed", "response.cancelled":
		if rawResponse, ok := object["response"]; ok {
			var responseObject map[string]json.RawMessage
			if json.Unmarshal(rawResponse, &responseObject) == nil {
				if rawUsage, exists := responseObject["usage"]; exists && !bytes.Equal(bytes.TrimSpace(rawUsage), []byte("null")) {
					usage, usageErr := decodeOpenAIUsage(rawUsage, format)
					if usageErr != nil {
						return nil, nil, usageErr
					}
					events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamUsage, ResponseID: state.ResponseID, Model: state.Model, Usage: &usage})
				}
			}
		}
		if typeName == "response.failed" || typeName == "response.cancelled" {
			events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamError, ResponseID: state.ResponseID, Model: state.Model, Error: &llmprotocol.ProtocolError{Type: typeName, Message: "response stream did not complete"}})
		}
		events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamResponseDone, ResponseID: state.ResponseID, Model: state.Model})
	case "error":
		base.Type = llmprotocol.StreamError
		base.Error = &llmprotocol.ProtocolError{}
		base.Error.Type, _ = optionalString(format, object, "param")
		base.Error.Code, _ = optionalString(format, object, "code")
		base.Error.Message, err = optionalString(format, object, "message")
		if err != nil {
			return nil, nil, err
		}
		events = append(events, base)
	case "response.content_part.added", "response.content_part.done", "response.output_text.done", "response.refusal.done", "response.reasoning_summary_text.done", "response.function_call_arguments.done":
		// These events contain snapshots of data already carried by deltas. They
		// intentionally do not duplicate neutral deltas.
	default:
		raw, _ := json.Marshal(object)
		block, diagnostics, unknownErr := decodeUnknownBlock(format, raw, policy, "$stream")
		if unknownErr != nil {
			return nil, diagnostics, unknownErr
		}
		if block != nil {
			base.Type, base.Provider, base.Raw = llmprotocol.StreamUnknown, format, cloneRaw(wire.Data)
			events = append(events, base)
		}
		return events, diagnostics, nil
	}
	return events, nil, nil
}

func (OpenAIResponses) EncodeStreamEvent(state *StreamState, event llmprotocol.StreamEvent, policy llmprotocol.Policy) ([]llmprotocol.WireEvent, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatOpenAIResponses
	if event.ResponseID != "" {
		state.ResponseID = event.ResponseID
	}
	if event.Model != "" {
		state.Model = event.Model
	}
	var prefix []llmprotocol.WireEvent
	if started, _ := state.valueMap()["target_started"].(bool); !started && event.Type != llmprotocol.StreamResponseStart {
		state.Values["target_started"] = true
		sequence := nextSequence(state)
		object := map[string]any{
			"type":            "response.created",
			"sequence_number": sequence,
			"response":        responseStreamEnvelope(state, "in_progress", nil),
		}
		body, err := marshalObject(format, object)
		if err != nil {
			return nil, nil, err
		}
		prefix = append(prefix, llmprotocol.WireEvent{Event: "response.created", Data: body})
	}
	if event.Type == llmprotocol.StreamResponseStart {
		state.valueMap()["target_started"] = true
	}
	if event.ItemID == "" {
		switch event.Type {
		case llmprotocol.StreamOutputStart, llmprotocol.StreamTextDelta,
			llmprotocol.StreamReasoningDelta, llmprotocol.StreamRefusalDelta,
			llmprotocol.StreamOutputDone:
			event.ItemID = responseStreamItemID(state, "msg", event.OutputIndex, 0)
		case llmprotocol.StreamToolCallStart, llmprotocol.StreamToolArgsDelta:
			event.ItemID = responseStreamItemID(state, "fc", event.OutputIndex, event.ContentIndex)
		}
	}
	sequence := nextSequence(state)
	base := func(typeName string) map[string]any {
		return map[string]any{"type": typeName, "sequence_number": sequence, "output_index": event.OutputIndex, "content_index": event.ContentIndex, "item_id": event.ItemID}
	}
	var eventName string
	var object map[string]any
	var diagnostics []llmprotocol.Diagnostic
	switch event.Type {
	case llmprotocol.StreamResponseStart:
		eventName = "response.created"
		object = base(eventName)
		object["response"] = responseStreamEnvelope(state, "in_progress", nil)
	case llmprotocol.StreamOutputStart:
		eventName = "response.output_item.added"
		object = base(eventName)
		object["item"] = map[string]any{"type": "message", "id": event.ItemID, "status": "in_progress", "role": string(event.Role), "content": []any{}}
	case llmprotocol.StreamTextDelta:
		eventName = "response.output_text.delta"
		object = base(eventName)
		object["delta"] = event.Delta
	case llmprotocol.StreamRefusalDelta:
		eventName = "response.refusal.delta"
		object = base(eventName)
		object["delta"] = event.Delta
	case llmprotocol.StreamReasoningDelta:
		eventName = "response.reasoning_summary_text.delta"
		object = base(eventName)
		object["delta"] = event.Delta
	case llmprotocol.StreamToolCallStart:
		eventName = "response.output_item.added"
		object = base(eventName)
		object["item"] = map[string]any{"type": "function_call", "id": event.ItemID, "status": "in_progress", "call_id": event.ToolCallID, "name": event.ToolName, "arguments": ""}
	case llmprotocol.StreamToolArgsDelta:
		eventName = "response.function_call_arguments.delta"
		object = base(eventName)
		object["delta"] = event.Delta
	case llmprotocol.StreamOutputDone:
		eventName = "response.output_item.done"
		object = base(eventName)
		object["item"] = map[string]any{"id": event.ItemID, "status": "completed"}
	case llmprotocol.StreamUsage:
		if event.Usage != nil {
			state.valueMap()["usage"] = *event.Usage
		}
		return prefix, nil, nil
	case llmprotocol.StreamLogprobs:
		next, err := lossy(format, policy, "$stream.logprobs", "logprobs_not_supported")
		diagnostics = append(diagnostics, next...)
		return prefix, diagnostics, err
	case llmprotocol.StreamResponseDone:
		eventName = "response.completed"
		object = base(eventName)
		var usage *llmprotocol.Usage
		if value, ok := state.valueMap()["usage"].(llmprotocol.Usage); ok {
			usage = &value
		}
		object["response"] = responseStreamEnvelope(state, "completed", usage)
		state.Values["target_done"] = true
	case llmprotocol.StreamError:
		eventName = "error"
		object = base(eventName)
		if event.Error != nil {
			object["message"], object["code"], object["param"] = event.Error.Message, event.Error.Code, event.Error.Type
		} else {
			object["message"] = "upstream stream failed"
		}
	case llmprotocol.StreamUnknown:
		next, err := lossy(format, policy, "$stream", "unknown_stream_event")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return nil, diagnostics, err
		}
		return prefix, diagnostics, nil
	default:
		return prefix, nil, nil
	}
	body, err := marshalObject(format, object)
	if err != nil {
		return nil, diagnostics, err
	}
	return append(prefix, llmprotocol.WireEvent{Event: eventName, Data: body}), diagnostics, nil
}

func (OpenAIResponses) FinishStream(state *StreamState, _ llmprotocol.Policy) ([]llmprotocol.WireEvent, []llmprotocol.Diagnostic, error) {
	if done, _ := state.valueMap()["target_done"].(bool); done {
		return nil, nil, nil
	}
	state.Values["target_done"] = true
	sequence := nextSequence(state)
	object := map[string]any{"type": "response.completed", "sequence_number": sequence, "response": responseStreamEnvelope(state, "completed", nil)}
	body, err := marshalObject(llmprotocol.FormatOpenAIResponses, object)
	if err != nil {
		return nil, nil, err
	}
	return []llmprotocol.WireEvent{{Event: "response.completed", Data: body}}, nil, nil
}

func nextSequence(state *StreamState) int64 {
	values := state.valueMap()
	current, _ := values["sequence"].(int64)
	values["sequence"] = current + 1
	return current
}

func responseStreamEnvelope(state *StreamState, status string, usage *llmprotocol.Usage) map[string]any {
	object := map[string]any{"id": state.ResponseID, "object": "response", "model": state.Model, "status": status, "output": []any{}}
	if usage != nil {
		object["usage"] = encodeOpenAIUsage(*usage, false)
	}
	return object
}

func responseItemKey(output int) string { return fmt.Sprintf("response-item:%d", output) }

func responseStreamItemID(state *StreamState, prefix string, output, content int) string {
	key := fmt.Sprintf("response-item:%s:%d:%d", prefix, output, content)
	if value, ok := state.valueMap()[key].(string); ok && value != "" {
		return value
	}
	value := stableID(prefix, state.ResponseID, output, content)
	state.Values[key] = value
	return value
}
