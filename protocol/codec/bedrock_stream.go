// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package codec

import (
	"encoding/json"
	"fmt"
	"strings"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

func (Bedrock) DecodeStreamEvent(state *StreamState, wire llmprotocol.WireEvent, _ llmprotocol.Policy) ([]llmprotocol.StreamEvent, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatBedrock
	eventType := strings.TrimSpace(wire.Event)
	if eventType == "" {
		var envelope map[string]json.RawMessage
		if json.Unmarshal(wire.Data, &envelope) == nil && len(envelope) == 1 {
			for name, value := range envelope {
				eventType, wire.Data = name, value
			}
		}
	}
	object, err := decodeObject(format, wire.Data)
	if err != nil {
		return nil, nil, err
	}
	values := state.valueMap()
	base := func(event llmprotocol.StreamEvent) llmprotocol.StreamEvent {
		event.ResponseID, event.Model = state.ResponseID, state.Model
		return event
	}
	switch eventType {
	case "messageStart":
		if started, _ := values["bedrock_message_started"].(bool); started {
			return nil, nil, translationError(format, "$stream.messageStart", "invalid_event_order", "messageStart is duplicated")
		}
		role, roleErr := optionalString(format, object, "role")
		if roleErr != nil {
			return nil, nil, roleErr
		}
		if role != "assistant" {
			return nil, nil, translationError(format, "$stream.messageStart.role", "unsupported_role", "Bedrock stream role must be assistant")
		}
		values["bedrock_message_started"] = true
		return []llmprotocol.StreamEvent{
			base(llmprotocol.StreamEvent{Type: llmprotocol.StreamResponseStart}),
			base(llmprotocol.StreamEvent{Type: llmprotocol.StreamOutputStart, Role: llmprotocol.RoleAssistant}),
		}, nil, nil
	case "contentBlockStart":
		if !bedrockStreamBool(values, "bedrock_message_started") || bedrockStreamBool(values, "bedrock_message_done") {
			return nil, nil, translationError(format, "$stream.contentBlockStart", "invalid_event_order", "content block started outside a message")
		}
		index, indexErr := bedrockStreamIndex(object)
		if indexErr != nil {
			return nil, nil, indexErr
		}
		start, objectErr := decodeObject(format, object["start"])
		if objectErr != nil {
			return nil, nil, objectErr
		}
		if rawTool, ok := start["toolUse"]; ok {
			tool, toolErr := decodeObject(format, rawTool)
			if toolErr != nil {
				return nil, nil, toolErr
			}
			id, idErr := optionalString(format, tool, "toolUseId")
			if idErr != nil {
				return nil, nil, idErr
			}
			name, nameErr := optionalString(format, tool, "name")
			if nameErr != nil {
				return nil, nil, nameErr
			}
			values[streamKey("bedrock_tool_id", 0, index)] = id
			values[streamKey("bedrock_tool_name", 0, index)] = name
			return []llmprotocol.StreamEvent{base(llmprotocol.StreamEvent{Type: llmprotocol.StreamToolCallStart, ContentIndex: index, ToolCallID: id, ToolName: name, Provider: format})}, nil, nil
		}
		return nil, nil, nil
	case "contentBlockDelta":
		if !bedrockStreamBool(values, "bedrock_message_started") || bedrockStreamBool(values, "bedrock_message_done") {
			return nil, nil, translationError(format, "$stream.contentBlockDelta", "invalid_event_order", "content block delta arrived outside a message")
		}
		index, indexErr := bedrockStreamIndex(object)
		if indexErr != nil {
			return nil, nil, indexErr
		}
		delta, objectErr := decodeObject(format, object["delta"])
		if objectErr != nil {
			return nil, nil, objectErr
		}
		if rawText, ok := delta["text"]; ok {
			text, textErr := decodeString(format, "$stream.contentBlockDelta.delta.text", rawText)
			if textErr != nil {
				return nil, nil, textErr
			}
			return []llmprotocol.StreamEvent{base(llmprotocol.StreamEvent{Type: llmprotocol.StreamTextDelta, ContentIndex: index, Delta: text})}, nil, nil
		}
		if rawTool, ok := delta["toolUse"]; ok {
			tool, toolErr := decodeObject(format, rawTool)
			if toolErr != nil {
				return nil, nil, toolErr
			}
			input, inputErr := optionalString(format, tool, "input")
			if inputErr != nil {
				return nil, nil, inputErr
			}
			id, _ := values[streamKey("bedrock_tool_id", 0, index)].(string)
			return []llmprotocol.StreamEvent{base(llmprotocol.StreamEvent{Type: llmprotocol.StreamToolArgsDelta, ContentIndex: index, ToolCallID: id, Delta: input})}, nil, nil
		}
		if rawReasoning, ok := delta["reasoningContent"]; ok {
			reasoning, reasoningErr := decodeObject(format, rawReasoning)
			if reasoningErr != nil {
				return nil, nil, reasoningErr
			}
			if rawText, exists := reasoning["text"]; exists {
				text, textErr := decodeString(format, "$stream.contentBlockDelta.delta.reasoningContent.text", rawText)
				if textErr != nil {
					return nil, nil, textErr
				}
				return []llmprotocol.StreamEvent{base(llmprotocol.StreamEvent{Type: llmprotocol.StreamReasoningDelta, ContentIndex: index, Delta: text, Provider: format})}, nil, nil
			}
			if rawSignature, exists := reasoning["signature"]; exists {
				signature, signatureErr := decodeString(format, "$stream.contentBlockDelta.delta.reasoningContent.signature", rawSignature)
				if signatureErr != nil {
					return nil, nil, signatureErr
				}
				return []llmprotocol.StreamEvent{base(llmprotocol.StreamEvent{Type: llmprotocol.StreamReasoningSignatureDelta, ContentIndex: index, Signature: signature, Provider: format})}, nil, nil
			}
		}
		return nil, nil, translationError(format, "$stream.contentBlockDelta.delta", "unsupported_stream_delta", "Bedrock stream delta is not representable")
	case "contentBlockStop":
		return nil, nil, nil
	case "messageStop":
		if !bedrockStreamBool(values, "bedrock_message_started") || bedrockStreamBool(values, "bedrock_message_done") {
			return nil, nil, translationError(format, "$stream.messageStop", "invalid_event_order", "messageStop is out of order")
		}
		stop, stopErr := optionalString(format, object, "stopReason")
		if stopErr != nil {
			return nil, nil, stopErr
		}
		values["bedrock_message_done"] = true
		return []llmprotocol.StreamEvent{base(llmprotocol.StreamEvent{Type: llmprotocol.StreamOutputDone, StopReason: decodeBedrockStopReason(stop)})}, nil, nil
	case "metadata":
		if rawUsage, ok := object["usage"]; ok {
			usage, usageErr := decodeBedrockUsage(rawUsage)
			if usageErr != nil {
				return nil, nil, usageErr
			}
			return []llmprotocol.StreamEvent{base(llmprotocol.StreamEvent{Type: llmprotocol.StreamUsage, Usage: &usage})}, nil, nil
		}
		return nil, nil, nil
	case "internalServerException", "modelStreamErrorException", "throttlingException", "validationException", "serviceUnavailableException":
		message, _ := optionalString(format, object, "message")
		if message == "" {
			message = "the upstream Bedrock stream failed"
		}
		return []llmprotocol.StreamEvent{base(llmprotocol.StreamEvent{Type: llmprotocol.StreamError, Error: &llmprotocol.ProtocolError{Type: eventType, Message: message, Retryable: eventType != "validationException"}})}, nil, nil
	default:
		return nil, nil, translationError(format, "$stream", "unsupported_stream_event", fmt.Sprintf("Bedrock event %q is not supported", eventType))
	}
}

func (Bedrock) EncodeStreamEvent(state *StreamState, event llmprotocol.StreamEvent, policy llmprotocol.Policy) ([]llmprotocol.WireEvent, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatBedrock
	values := state.valueMap()
	encode := func(name string, body map[string]any) ([]llmprotocol.WireEvent, []llmprotocol.Diagnostic, error) {
		raw, err := marshalObject(format, body)
		if err != nil {
			return nil, nil, err
		}
		return []llmprotocol.WireEvent{{Event: name, Data: raw}}, nil, nil
	}
	switch event.Type {
	case llmprotocol.StreamResponseStart:
		return nil, nil, nil
	case llmprotocol.StreamOutputStart:
		return encode("messageStart", map[string]any{"role": "assistant"})
	case llmprotocol.StreamTextDelta:
		return encode("contentBlockDelta", map[string]any{"contentBlockIndex": event.ContentIndex, "delta": map[string]any{"text": event.Delta}})
	case llmprotocol.StreamReasoningDelta:
		return encode("contentBlockDelta", map[string]any{"contentBlockIndex": event.ContentIndex, "delta": map[string]any{"reasoningContent": map[string]any{"text": event.Delta}}})
	case llmprotocol.StreamReasoningSignatureDelta:
		return encode("contentBlockDelta", map[string]any{"contentBlockIndex": event.ContentIndex, "delta": map[string]any{"reasoningContent": map[string]any{"signature": event.Signature}}})
	case llmprotocol.StreamToolCallStart:
		values[streamKey("bedrock_target_tool_id", event.OutputIndex, event.ContentIndex)] = event.ToolCallID
		return encode("contentBlockStart", map[string]any{"contentBlockIndex": event.ContentIndex, "start": map[string]any{"toolUse": map[string]any{"toolUseId": event.ToolCallID, "name": event.ToolName}}})
	case llmprotocol.StreamToolArgsDelta:
		return encode("contentBlockDelta", map[string]any{"contentBlockIndex": event.ContentIndex, "delta": map[string]any{"toolUse": map[string]any{"input": event.Delta}}})
	case llmprotocol.StreamOutputDone:
		return encode("messageStop", map[string]any{"stopReason": encodeBedrockStopReason(event.StopReason)})
	case llmprotocol.StreamUsage:
		body := map[string]any{}
		if event.Usage != nil {
			body["usage"] = encodeBedrockUsage(*event.Usage)
		}
		return encode("metadata", body)
	case llmprotocol.StreamLogprobs:
		diagnostics, err := lossy(format, policy, "$stream.logprobs", "logprobs_not_supported")
		return nil, diagnostics, err
	case llmprotocol.StreamResponseDone:
		return nil, nil, nil
	case llmprotocol.StreamError:
		message := "upstream stream failed"
		if event.Error != nil {
			message = event.Error.Message
		}
		return encode("modelStreamErrorException", map[string]any{"message": message})
	case llmprotocol.StreamUnknown:
		diagnostics, err := lossy(format, policy, "$stream", "unknown_stream_event")
		return nil, diagnostics, err
	default:
		return nil, nil, nil
	}
}

func (Bedrock) FinishStream(_ *StreamState, _ llmprotocol.Policy) ([]llmprotocol.WireEvent, []llmprotocol.Diagnostic, error) {
	return nil, nil, nil
}

func bedrockStreamIndex(object map[string]json.RawMessage) (int, error) {
	value, err := optionalInt(llmprotocol.FormatBedrock, object, "contentBlockIndex")
	if err != nil {
		return 0, err
	}
	if value == nil || *value < 0 {
		return 0, translationError(llmprotocol.FormatBedrock, "$stream.contentBlockIndex", "invalid_content_index", "contentBlockIndex must be non-negative")
	}
	return int(*value), nil
}

func bedrockStreamBool(values map[string]any, key string) bool {
	value, _ := values[key].(bool)
	return value
}
