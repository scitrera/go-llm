// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package codec

import (
	"bytes"
	"encoding/json"
	"fmt"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

func (OpenAIChat) DecodeStreamEvent(state *StreamState, wire llmprotocol.WireEvent, policy llmprotocol.Policy) ([]llmprotocol.StreamEvent, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatOpenAIChat
	if bytes.Equal(bytes.TrimSpace(wire.Data), []byte("[DONE]")) {
		state.valueMap()["source_done"] = true
		return []llmprotocol.StreamEvent{{Type: llmprotocol.StreamResponseDone, ResponseID: state.ResponseID, Model: state.Model}}, nil, nil
	}
	object, err := decodeObject(format, wire.Data)
	if err != nil {
		return nil, nil, err
	}
	if rawError, ok := object["error"]; ok {
		var providerError struct {
			Type    string `json:"type"`
			Code    string `json:"code"`
			Message string `json:"message"`
		}
		if err := json.Unmarshal(rawError, &providerError); err != nil {
			return nil, nil, translationError(format, "$.error", "invalid_stream_error", "error must be an object")
		}
		return []llmprotocol.StreamEvent{{Type: llmprotocol.StreamError, Error: &llmprotocol.ProtocolError{Type: providerError.Type, Code: providerError.Code, Message: providerError.Message}}}, nil, nil
	}
	if id, idErr := optionalString(format, object, "id"); idErr != nil {
		return nil, nil, idErr
	} else if id != "" {
		state.ResponseID = id
	}
	if model, modelErr := optionalString(format, object, "model"); modelErr != nil {
		return nil, nil, modelErr
	} else if model != "" {
		state.Model = model
	}
	delete(object, "object")
	delete(object, "created")
	delete(object, "service_tier")
	delete(object, "system_fingerprint")
	var events []llmprotocol.StreamEvent
	var diagnostics []llmprotocol.Diagnostic
	if rawUsage, ok := object["usage"]; ok && !bytes.Equal(bytes.TrimSpace(rawUsage), []byte("null")) {
		delete(object, "usage")
		usage, usageErr := decodeOpenAIUsage(rawUsage, format)
		if usageErr != nil {
			return nil, nil, usageErr
		}
		events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamUsage, ResponseID: state.ResponseID, Model: state.Model, Usage: &usage})
	}
	if rawChoices, ok := object["choices"]; ok {
		delete(object, "choices")
		choices, arrayErr := decodeArray(format, "$.choices", rawChoices)
		if arrayErr != nil {
			return nil, nil, arrayErr
		}
		for fallbackIndex, rawChoice := range choices {
			choice, choiceErr := decodeObject(format, rawChoice)
			if choiceErr != nil {
				return nil, nil, choiceErr
			}
			choiceIndex := fallbackIndex
			if value, valueErr := optionalInt(format, choice, "index"); valueErr != nil {
				return nil, nil, valueErr
			} else if value != nil {
				choiceIndex = int(*value)
			}
			if rawDelta, exists := choice["delta"]; exists {
				delete(choice, "delta")
				delta, deltaErr := decodeObject(format, rawDelta)
				if deltaErr != nil {
					return nil, nil, deltaErr
				}
				if role, roleErr := optionalString(format, delta, "role"); roleErr != nil {
					return nil, nil, roleErr
				} else if role != "" {
					validated, validateErr := validateRole(format, "$.choices[].delta.role", role)
					if validateErr != nil {
						return nil, nil, validateErr
					}
					events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamOutputStart, ResponseID: state.ResponseID, Model: state.Model, OutputIndex: choiceIndex, Role: validated})
				}
				delete(delta, "role") // a null role is consumed, not an unknown field
				for _, field := range []struct {
					names    []string
					typeName llmprotocol.StreamEventType
				}{
					{[]string{"content"}, llmprotocol.StreamTextDelta},
					// Reasoning has no OpenAI-standard field name, so the same
					// channel arrives under a different key per vendor:
					// DeepSeek/vLLM/SGLang use `reasoning_content`, OpenRouter
					// `reasoning`, Ollama `thinking`. First present wins; the
					// others are consumed so a second alias can neither
					// double-emit the trace nor surface as an unknown field.
					{[]string{"reasoning_content", "reasoning", "thinking"}, llmprotocol.StreamReasoningDelta},
					{[]string{"refusal"}, llmprotocol.StreamRefusalDelta},
				} {
					emitted := false
					for _, name := range field.names {
						rawValue, exists := delta[name]
						if !exists {
							continue
						}
						// Consume unconditionally: an explicit null is a known
						// field carrying no value (OpenAI sends `"content":null`
						// on tool-call and terminal chunks), not an extension.
						delete(delta, name)
						if bytes.Equal(bytes.TrimSpace(rawValue), []byte("null")) {
							continue
						}
						text, textErr := decodeString(format, "$.choices[].delta."+name, rawValue)
						if textErr != nil {
							return nil, nil, textErr
						}
						if emitted {
							continue
						}
						events = append(events, llmprotocol.StreamEvent{Type: field.typeName, ResponseID: state.ResponseID, Model: state.Model, OutputIndex: choiceIndex, Delta: text})
						emitted = true
					}
				}
				if rawCalls, exists := delta["tool_calls"]; exists {
					delete(delta, "tool_calls")
					calls, callsErr := decodeArray(format, "$.choices[].delta.tool_calls", rawCalls)
					if callsErr != nil {
						return nil, nil, callsErr
					}
					for fallbackToolIndex, rawCall := range calls {
						call, callErr := decodeObject(format, rawCall)
						if callErr != nil {
							return nil, nil, callErr
						}
						toolIndex := fallbackToolIndex
						if value, valueErr := optionalInt(format, call, "index"); valueErr != nil {
							return nil, nil, valueErr
						} else if value != nil {
							toolIndex = int(*value)
						}
						id, idErr := optionalString(format, call, "id")
						if idErr != nil {
							return nil, nil, idErr
						}
						function := map[string]json.RawMessage{}
						if rawFunction, exists := call["function"]; exists {
							function, callErr = decodeObject(format, rawFunction)
							if callErr != nil {
								return nil, nil, callErr
							}
						}
						name, nameErr := optionalString(format, function, "name")
						if nameErr != nil {
							return nil, nil, nameErr
						}
						if id != "" || name != "" {
							events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamToolCallStart, ResponseID: state.ResponseID, Model: state.Model, OutputIndex: choiceIndex, ContentIndex: toolIndex, ToolCallID: id, ToolName: name})
						}
						arguments, argsErr := optionalString(format, function, "arguments")
						if argsErr != nil {
							return nil, nil, argsErr
						}
						if arguments != "" {
							events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamToolArgsDelta, ResponseID: state.ResponseID, Model: state.Model, OutputIndex: choiceIndex, ContentIndex: toolIndex, ToolCallID: id, Delta: arguments})
						}
					}
				}
				// Whatever is left in the delta is a field this codec doesn't
				// map, so it must obey UnknownFieldPolicy like every other
				// object. Without this the delta was the ONE place unknown
				// fields vanished silently — no diagnostic under drop, no error
				// under reject — which is how a vendor's reasoning alias could
				// be dropped without a trace. StreamEvent carries no extensions,
				// so preserved values have nowhere to go (same as the
				// chunk-level collection below); the diagnostics are the signal.
				_, next, deltaExtErr := collectExtensions(format, delta, policy)
				diagnostics = append(diagnostics, next...)
				if deltaExtErr != nil {
					return nil, diagnostics, deltaExtErr
				}
			}
			if rawLogprobs, exists := choice["logprobs"]; exists && !bytes.Equal(bytes.TrimSpace(rawLogprobs), []byte("null")) {
				delete(choice, "logprobs")
				logprobs, logprobErr := decodeOpenAIChatLogprobs(rawLogprobs)
				if logprobErr != nil {
					return nil, diagnostics, logprobErr
				}
				events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamLogprobs, ResponseID: state.ResponseID, Model: state.Model, OutputIndex: choiceIndex, Logprobs: logprobs})
			} else {
				delete(choice, "logprobs")
			}
			if rawFinish, exists := choice["finish_reason"]; exists && !bytes.Equal(bytes.TrimSpace(rawFinish), []byte("null")) {
				delete(choice, "finish_reason")
				value, finishErr := decodeString(format, "$.choices[].finish_reason", rawFinish)
				if finishErr != nil {
					return nil, nil, finishErr
				}
				events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamOutputDone, ResponseID: state.ResponseID, Model: state.Model, OutputIndex: choiceIndex, StopReason: decodeChatStopReason(value)})
			} else {
				delete(choice, "finish_reason")
			}
			_, next, choiceErr := collectExtensions(format, choice, policy)
			diagnostics = append(diagnostics, next...)
			if choiceErr != nil {
				return nil, diagnostics, choiceErr
			}
		}
	}
	extensions, next, err := collectExtensions(format, object, policy)
	diagnostics = append(diagnostics, next...)
	_ = extensions
	if err != nil {
		return nil, diagnostics, err
	}
	return events, diagnostics, nil
}

func (OpenAIChat) EncodeStreamEvent(state *StreamState, event llmprotocol.StreamEvent, policy llmprotocol.Policy) ([]llmprotocol.WireEvent, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatOpenAIChat
	if event.ResponseID != "" {
		state.ResponseID = event.ResponseID
	}
	if event.Model != "" {
		state.Model = event.Model
	}
	base := func() map[string]any {
		return map[string]any{"id": state.ResponseID, "object": "chat.completion.chunk", "model": state.Model}
	}
	chunk := base()
	var diagnostics []llmprotocol.Diagnostic
	switch event.Type {
	case llmprotocol.StreamResponseStart:
		return nil, nil, nil
	case llmprotocol.StreamOutputStart:
		chunk["choices"] = []any{map[string]any{"index": event.OutputIndex, "delta": map[string]any{"role": string(event.Role)}, "finish_reason": nil}}
	case llmprotocol.StreamTextDelta:
		chunk["choices"] = []any{map[string]any{"index": event.OutputIndex, "delta": map[string]any{"content": event.Delta}, "finish_reason": nil}}
	case llmprotocol.StreamReasoningDelta:
		chunk["choices"] = []any{map[string]any{"index": event.OutputIndex, "delta": map[string]any{"reasoning_content": event.Delta}, "finish_reason": nil}}
	case llmprotocol.StreamRefusalDelta:
		chunk["choices"] = []any{map[string]any{"index": event.OutputIndex, "delta": map[string]any{"refusal": event.Delta}, "finish_reason": nil}}
	case llmprotocol.StreamToolCallStart:
		chunk["choices"] = []any{map[string]any{"index": event.OutputIndex, "delta": map[string]any{"tool_calls": []any{map[string]any{"index": event.ContentIndex, "id": event.ToolCallID, "type": "function", "function": map[string]any{"name": event.ToolName, "arguments": ""}}}}, "finish_reason": nil}}
	case llmprotocol.StreamToolArgsDelta:
		chunk["choices"] = []any{map[string]any{"index": event.OutputIndex, "delta": map[string]any{"tool_calls": []any{map[string]any{"index": event.ContentIndex, "function": map[string]any{"arguments": event.Delta}}}}, "finish_reason": nil}}
	case llmprotocol.StreamOutputDone:
		chunk["choices"] = []any{map[string]any{"index": event.OutputIndex, "delta": map[string]any{}, "finish_reason": encodeChatStopReason(event.StopReason)}}
	case llmprotocol.StreamUsage:
		chunk["choices"] = []any{}
		if event.Usage != nil {
			chunk["usage"] = encodeOpenAIUsage(*event.Usage, true)
		}
	case llmprotocol.StreamLogprobs:
		chunk["choices"] = []any{map[string]any{"index": event.OutputIndex, "delta": map[string]any{}, "logprobs": encodeOpenAIChatLogprobs(event.Logprobs), "finish_reason": nil}}
	case llmprotocol.StreamResponseDone:
		state.valueMap()["target_done"] = true
		return []llmprotocol.WireEvent{{Data: json.RawMessage(`[DONE]`)}}, nil, nil
	case llmprotocol.StreamError:
		providerError := map[string]any{"message": "upstream stream failed", "type": "upstream_error"}
		if event.Error != nil {
			providerError["message"] = event.Error.Message
			if event.Error.Type != "" {
				providerError["type"] = event.Error.Type
			}
			if event.Error.Code != "" {
				providerError["code"] = event.Error.Code
			}
		}
		chunk = map[string]any{"error": providerError}
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
	body, err := marshalObject(format, chunk)
	if err != nil {
		return nil, diagnostics, err
	}
	return []llmprotocol.WireEvent{{Data: body}}, diagnostics, nil
}

func (OpenAIChat) FinishStream(state *StreamState, _ llmprotocol.Policy) ([]llmprotocol.WireEvent, []llmprotocol.Diagnostic, error) {
	if done, _ := state.valueMap()["target_done"].(bool); done {
		return nil, nil, nil
	}
	state.Values["target_done"] = true
	return []llmprotocol.WireEvent{{Data: json.RawMessage(`[DONE]`)}}, nil, nil
}

func streamKey(prefix string, output, content int) string {
	return fmt.Sprintf("%s:%d:%d", prefix, output, content)
}
