// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package codec

import (
	"bytes"
	"encoding/json"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

func (Anthropic) DecodeStreamEvent(state *StreamState, wire llmprotocol.WireEvent, policy llmprotocol.Policy) ([]llmprotocol.StreamEvent, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatAnthropic
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
	base := llmprotocol.StreamEvent{ResponseID: state.ResponseID, Model: state.Model, Role: llmprotocol.RoleAssistant}
	index := 0
	if value, valueErr := optionalInt(format, object, "index"); valueErr != nil {
		return nil, nil, valueErr
	} else if value != nil {
		index = int(*value)
	}
	base.OutputIndex, base.ContentIndex = 0, index
	switch typeName {
	case "message_start":
		rawMessage := object["message"]
		var message struct {
			ID    string          `json:"id"`
			Model string          `json:"model"`
			Usage json.RawMessage `json:"usage"`
		}
		if err := json.Unmarshal(rawMessage, &message); err != nil {
			return nil, nil, translationError(format, "$.message", "invalid_message", "message_start requires a message object")
		}
		state.ResponseID, state.Model = message.ID, message.Model
		events := []llmprotocol.StreamEvent{{Type: llmprotocol.StreamResponseStart, ResponseID: message.ID, Model: message.Model}}
		if len(message.Usage) != 0 && !bytes.Equal(bytes.TrimSpace(message.Usage), []byte("null")) {
			usage, usageErr := decodeAnthropicUsage(message.Usage)
			if usageErr != nil {
				return nil, nil, usageErr
			}
			events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamUsage, ResponseID: message.ID, Model: message.Model, Usage: &usage})
		}
		return events, nil, nil
	case "content_block_start":
		rawBlock := object["content_block"]
		block, blockErr := decodeObject(format, rawBlock)
		if blockErr != nil {
			return nil, nil, blockErr
		}
		blockType, typeErr := optionalString(format, block, "type")
		if typeErr != nil {
			return nil, nil, typeErr
		}
		base.ItemID, _ = optionalString(format, block, "id")
		switch blockType {
		case "tool_use":
			base.Type = llmprotocol.StreamToolCallStart
			base.ToolCallID = base.ItemID
			base.ToolName, err = optionalString(format, block, "name")
		case "thinking", "redacted_thinking":
			base.Type = llmprotocol.StreamOutputStart
		case "text":
			base.Type = llmprotocol.StreamOutputStart
		default:
			base.Type, base.Provider, base.Raw = llmprotocol.StreamUnknown, format, cloneRaw(rawBlock)
		}
		if err != nil {
			return nil, nil, err
		}
		return []llmprotocol.StreamEvent{base}, nil, nil
	case "content_block_delta":
		rawDelta := object["delta"]
		delta, deltaErr := decodeObject(format, rawDelta)
		if deltaErr != nil {
			return nil, nil, deltaErr
		}
		deltaType, typeErr := optionalString(format, delta, "type")
		if typeErr != nil {
			return nil, nil, typeErr
		}
		switch deltaType {
		case "text_delta":
			base.Type = llmprotocol.StreamTextDelta
			base.Delta, err = optionalString(format, delta, "text")
		case "thinking_delta":
			base.Type = llmprotocol.StreamReasoningDelta
			base.Delta, err = optionalString(format, delta, "thinking")
		case "signature_delta":
			base.Type = llmprotocol.StreamReasoningSignatureDelta
			base.Signature, err = optionalString(format, delta, "signature")
		case "input_json_delta":
			base.Type = llmprotocol.StreamToolArgsDelta
			base.Delta, err = optionalString(format, delta, "partial_json")
		default:
			base.Type, base.Provider, base.Raw = llmprotocol.StreamUnknown, format, cloneRaw(rawDelta)
		}
		if err != nil {
			return nil, nil, err
		}
		return []llmprotocol.StreamEvent{base}, nil, nil
	case "content_block_stop":
		base.Type = llmprotocol.StreamOutputDone
		return []llmprotocol.StreamEvent{base}, nil, nil
	case "message_delta":
		var events []llmprotocol.StreamEvent
		if rawDelta, ok := object["delta"]; ok {
			delta, deltaErr := decodeObject(format, rawDelta)
			if deltaErr != nil {
				return nil, nil, deltaErr
			}
			stop, stopErr := optionalString(format, delta, "stop_reason")
			if stopErr != nil {
				return nil, nil, stopErr
			}
			if stop != "" {
				events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamOutputDone, ResponseID: state.ResponseID, Model: state.Model, StopReason: decodeAnthropicStopReason(stop)})
			}
		}
		if rawUsage, ok := object["usage"]; ok {
			usage, usageErr := decodeAnthropicUsage(rawUsage)
			if usageErr != nil {
				return nil, nil, usageErr
			}
			events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamUsage, ResponseID: state.ResponseID, Model: state.Model, Usage: &usage})
		}
		return events, nil, nil
	case "message_stop":
		return []llmprotocol.StreamEvent{{Type: llmprotocol.StreamResponseDone, ResponseID: state.ResponseID, Model: state.Model}}, nil, nil
	case "ping":
		return nil, nil, nil
	case "error":
		var envelope struct {
			Error struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal(wire.Data, &envelope); err != nil {
			return nil, nil, err
		}
		return []llmprotocol.StreamEvent{{Type: llmprotocol.StreamError, Error: &llmprotocol.ProtocolError{Type: envelope.Error.Type, Message: envelope.Error.Message}}}, nil, nil
	default:
		block, diagnostics, unknownErr := decodeUnknownBlock(format, wire.Data, policy, "$stream")
		if unknownErr != nil {
			return nil, diagnostics, unknownErr
		}
		if block == nil {
			return nil, diagnostics, nil
		}
		base.Type, base.Provider, base.Raw = llmprotocol.StreamUnknown, format, cloneRaw(wire.Data)
		return []llmprotocol.StreamEvent{base}, diagnostics, nil
	}
}

func (Anthropic) EncodeStreamEvent(state *StreamState, event llmprotocol.StreamEvent, policy llmprotocol.Policy) ([]llmprotocol.WireEvent, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatAnthropic
	if event.ResponseID != "" {
		state.ResponseID = event.ResponseID
	}
	if event.Model != "" {
		state.Model = event.Model
	}
	var eventName string
	var object map[string]any
	var diagnostics []llmprotocol.Diagnostic
	switch event.Type {
	case llmprotocol.StreamResponseStart:
		eventName = "message_start"
		object = map[string]any{"type": eventName, "message": map[string]any{"id": state.ResponseID, "type": "message", "role": "assistant", "model": state.Model, "content": []any{}, "stop_reason": nil, "stop_sequence": nil, "usage": map[string]any{"input_tokens": 0, "output_tokens": 0}}}
	case llmprotocol.StreamOutputStart:
		eventName = "content_block_start"
		object = map[string]any{"type": eventName, "index": event.ContentIndex, "content_block": map[string]any{"type": "text", "text": ""}}
	case llmprotocol.StreamTextDelta:
		eventName = "content_block_delta"
		object = map[string]any{"type": eventName, "index": event.ContentIndex, "delta": map[string]any{"type": "text_delta", "text": event.Delta}}
	case llmprotocol.StreamReasoningDelta:
		eventName = "content_block_delta"
		object = map[string]any{"type": eventName, "index": event.ContentIndex, "delta": map[string]any{"type": "thinking_delta", "thinking": event.Delta}}
	case llmprotocol.StreamReasoningSignatureDelta:
		eventName = "content_block_delta"
		object = map[string]any{"type": eventName, "index": event.ContentIndex, "delta": map[string]any{"type": "signature_delta", "signature": event.Signature}}
	case llmprotocol.StreamRefusalDelta:
		next, err := lossy(format, policy, "$stream", "refusal_not_distinct")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return nil, diagnostics, err
		}
		eventName = "content_block_delta"
		object = map[string]any{"type": eventName, "index": event.ContentIndex, "delta": map[string]any{"type": "text_delta", "text": event.Delta}}
	case llmprotocol.StreamToolCallStart:
		eventName = "content_block_start"
		object = map[string]any{"type": eventName, "index": event.ContentIndex, "content_block": map[string]any{"type": "tool_use", "id": event.ToolCallID, "name": event.ToolName, "input": map[string]any{}}}
	case llmprotocol.StreamToolArgsDelta:
		eventName = "content_block_delta"
		object = map[string]any{"type": eventName, "index": event.ContentIndex, "delta": map[string]any{"type": "input_json_delta", "partial_json": event.Delta}}
	case llmprotocol.StreamOutputDone:
		eventName = "content_block_stop"
		object = map[string]any{"type": eventName, "index": event.ContentIndex}
		state.valueMap()["stop_reason"] = event.StopReason
	case llmprotocol.StreamUsage:
		if event.Usage != nil {
			state.valueMap()["usage"] = *event.Usage
		}
		return nil, nil, nil
	case llmprotocol.StreamLogprobs:
		next, err := lossy(format, policy, "$stream.logprobs", "logprobs_not_supported")
		diagnostics = append(diagnostics, next...)
		return nil, diagnostics, err
	case llmprotocol.StreamResponseDone:
		var events []llmprotocol.WireEvent
		stop, _ := state.valueMap()["stop_reason"].(llmprotocol.StopReason)
		usage, _ := state.valueMap()["usage"].(llmprotocol.Usage)
		delta := map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": encodeAnthropicStopReason(stop), "stop_sequence": nil}, "usage": encodeAnthropicUsage(usage)}
		body, err := marshalObject(format, delta)
		if err != nil {
			return nil, nil, err
		}
		events = append(events, llmprotocol.WireEvent{Event: "message_delta", Data: body})
		stopBody, _ := marshalObject(format, map[string]any{"type": "message_stop"})
		events = append(events, llmprotocol.WireEvent{Event: "message_stop", Data: stopBody})
		state.Values["target_done"] = true
		return events, nil, nil
	case llmprotocol.StreamError:
		eventName = "error"
		providerError := map[string]any{"type": "api_error", "message": "upstream stream failed"}
		if event.Error != nil {
			providerError["message"] = event.Error.Message
			if event.Error.Type != "" {
				providerError["type"] = event.Error.Type
			}
		}
		object = map[string]any{"type": eventName, "error": providerError}
	case llmprotocol.StreamUnknown:
		next, err := lossy(format, policy, "$stream", "unknown_stream_event")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return nil, diagnostics, err
		}
		return nil, diagnostics, nil
	default:
		return nil, nil, nil
	}
	body, err := marshalObject(format, object)
	if err != nil {
		return nil, diagnostics, err
	}
	return []llmprotocol.WireEvent{{Event: eventName, Data: body}}, diagnostics, nil
}

func (Anthropic) FinishStream(state *StreamState, policy llmprotocol.Policy) ([]llmprotocol.WireEvent, []llmprotocol.Diagnostic, error) {
	if done, _ := state.valueMap()["target_done"].(bool); done {
		return nil, nil, nil
	}
	return (Anthropic{}).EncodeStreamEvent(state, llmprotocol.StreamEvent{Type: llmprotocol.StreamResponseDone}, policy)
}
