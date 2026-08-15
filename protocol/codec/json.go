// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package codec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

func cloneRaw(value json.RawMessage) json.RawMessage {
	return append(json.RawMessage(nil), value...)
}

func decodeObject(format llmprotocol.Format, raw json.RawMessage) (map[string]json.RawMessage, error) {
	var object map[string]json.RawMessage
	if len(raw) == 0 || json.Unmarshal(raw, &object) != nil || object == nil {
		return nil, translationError(format, "$", "invalid_json_object", "body must be a JSON object")
	}
	return object, nil
}

func decodeArray(format llmprotocol.Format, path string, raw json.RawMessage) ([]json.RawMessage, error) {
	var values []json.RawMessage
	if err := json.Unmarshal(raw, &values); err != nil {
		return nil, translationError(format, path, "invalid_array", "value must be an array")
	}
	return values, nil
}

func decodeString(format llmprotocol.Format, path string, raw json.RawMessage) (string, error) {
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", translationError(format, path, "invalid_string", "value must be a string")
	}
	return value, nil
}

func optionalString(format llmprotocol.Format, object map[string]json.RawMessage, name string) (string, error) {
	raw, ok := object[name]
	if !ok || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return "", nil
	}
	delete(object, name)
	return decodeString(format, "$.'"+name+"'", raw)
}

func optionalBool(format llmprotocol.Format, object map[string]json.RawMessage, name string) (*bool, error) {
	raw, ok := object[name]
	if !ok || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	delete(object, name)
	var value bool
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, translationError(format, "$.'"+name+"'", "invalid_boolean", "value must be a boolean")
	}
	return &value, nil
}

func optionalFloat(format llmprotocol.Format, object map[string]json.RawMessage, name string) (*float64, error) {
	raw, ok := object[name]
	if !ok || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	delete(object, name)
	var value float64
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, translationError(format, "$.'"+name+"'", "invalid_number", "value must be a number")
	}
	return &value, nil
}

func optionalInt(format llmprotocol.Format, object map[string]json.RawMessage, name string) (*int64, error) {
	raw, ok := object[name]
	if !ok || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	delete(object, name)
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var number json.Number
	if err := decoder.Decode(&number); err != nil {
		return nil, translationError(format, "$.'"+name+"'", "invalid_integer", "value must be an integer")
	}
	value, err := number.Int64()
	if err != nil {
		return nil, translationError(format, "$.'"+name+"'", "invalid_integer", "value must be an integer")
	}
	return &value, nil
}

func optionalStrings(format llmprotocol.Format, object map[string]json.RawMessage, name string, allowSingle bool) ([]string, error) {
	raw, ok := object[name]
	if !ok || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	delete(object, name)
	if allowSingle && len(bytes.TrimSpace(raw)) > 0 && bytes.TrimSpace(raw)[0] == '"' {
		value, err := decodeString(format, "$."+name, raw)
		if err != nil {
			return nil, err
		}
		return []string{value}, nil
	}
	var values []string
	if err := json.Unmarshal(raw, &values); err != nil {
		return nil, translationError(format, "$."+name, "invalid_string_array", "value must be an array of strings")
	}
	return values, nil
}

func preserveBody(policy llmprotocol.Policy, format llmprotocol.Format, raw json.RawMessage, request bool) llmprotocol.Preservation {
	if policy.Effective().Preservation != llmprotocol.PreserveInMemory {
		return llmprotocol.Preservation{}
	}
	if request {
		return llmprotocol.Preservation{Requests: map[llmprotocol.Format]json.RawMessage{format: cloneRaw(raw)}}
	}
	return llmprotocol.Preservation{Responses: map[llmprotocol.Format]json.RawMessage{format: cloneRaw(raw)}}
}

func preservedRequest(request llmprotocol.Request, format llmprotocol.Format, policy llmprotocol.Policy) (json.RawMessage, bool) {
	if policy.Effective().Preservation != llmprotocol.PreserveInMemory {
		return nil, false
	}
	raw, ok := request.Preservation.Requests[format]
	return cloneRaw(raw), ok
}

func preservedResponse(response llmprotocol.Response, format llmprotocol.Format, policy llmprotocol.Policy) (json.RawMessage, bool) {
	if policy.Effective().Preservation != llmprotocol.PreserveInMemory {
		return nil, false
	}
	raw, ok := response.Preservation.Responses[format]
	return cloneRaw(raw), ok
}

func collectExtensions(format llmprotocol.Format, object map[string]json.RawMessage, policy llmprotocol.Policy) (llmprotocol.Extensions, []llmprotocol.Diagnostic, error) {
	if len(object) == 0 {
		return nil, nil, nil
	}
	keys := make([]string, 0, len(object))
	for key := range object {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	policy = policy.Effective()
	extensions := make(llmprotocol.Extensions, len(keys))
	diagnostics := make([]llmprotocol.Diagnostic, 0, len(keys))
	for _, key := range keys {
		switch policy.UnknownFields {
		case llmprotocol.UnknownReject:
			return nil, diagnostics, translationError(format, "$.'"+key+"'", "unknown_field", "field is not supported by this codec")
		case llmprotocol.UnknownDrop:
			diagnostics = append(diagnostics, llmprotocol.Diagnostic{Kind: llmprotocol.DiagnosticUnknownField, Path: "$.'" + key + "'", Code: "dropped"})
		case llmprotocol.UnknownPreserve:
			extensions[key] = cloneRaw(object[key])
		}
	}
	if len(extensions) == 0 {
		extensions = nil
	}
	return extensions, diagnostics, nil
}

func mergeExtensions(object map[string]any, extensions llmprotocol.Extensions) {
	for key, raw := range extensions {
		if _, exists := object[key]; exists {
			continue
		}
		object[key] = json.RawMessage(cloneRaw(raw))
	}
}

func marshalObject(format llmprotocol.Format, object map[string]any) (json.RawMessage, error) {
	body, err := json.Marshal(object)
	if err != nil {
		return nil, translationError(format, "$", "encode_failed", err.Error())
	}
	return body, nil
}

func translationError(format llmprotocol.Format, path, code, detail string) error {
	return &llmprotocol.TranslationError{Format: format, Path: path, Code: code, Detail: detail}
}

func lossy(format llmprotocol.Format, policy llmprotocol.Policy, path, code string) ([]llmprotocol.Diagnostic, error) {
	if policy.Effective().Lossy == llmprotocol.LossyReject {
		return nil, translationError(format, path, code, "conversion would lose provider semantics")
	}
	return []llmprotocol.Diagnostic{{Kind: llmprotocol.DiagnosticLossy, Path: path, Code: code}}, nil
}

func providerStateActive(state llmprotocol.ProviderState) bool {
	return (state.Store != nil && *state.Store) ||
		(state.Background != nil && *state.Background) ||
		state.PreviousResponseID != "" ||
		len(state.Conversation) != 0
}

func rejectNonZeroPenalty(format llmprotocol.Format, policy llmprotocol.Policy, path string, value *float64, diagnostics *[]llmprotocol.Diagnostic) error {
	if value == nil || *value == 0 {
		return nil
	}
	next, err := lossy(format, policy, path, "sampling_control_not_supported")
	*diagnostics = append(*diagnostics, next...)
	return err
}

func rejectOptionalInt(format llmprotocol.Format, policy llmprotocol.Policy, path, code string, value *int64, diagnostics *[]llmprotocol.Diagnostic) error {
	if value == nil {
		return nil
	}
	next, err := lossy(format, policy, path, code)
	*diagnostics = append(*diagnostics, next...)
	return err
}

func validateNeutralOutputControls(format llmprotocol.Format, output llmprotocol.Output, policy llmprotocol.Policy, diagnostics *[]llmprotocol.Diagnostic) error {
	if output.Choices != nil && *output.Choices != 1 {
		next, err := lossy(format, policy, "$.output.choices", "multiple_outputs_not_supported")
		*diagnostics = append(*diagnostics, next...)
		if err != nil {
			return err
		}
	}
	if output.Logprobs != nil && *output.Logprobs {
		next, err := lossy(format, policy, "$.output.logprobs", "logprobs_not_supported")
		*diagnostics = append(*diagnostics, next...)
		if err != nil {
			return err
		}
	}
	if output.TopLogprobs != nil {
		next, err := lossy(format, policy, "$.output.top_logprobs", "logprobs_not_supported")
		*diagnostics = append(*diagnostics, next...)
		if err != nil {
			return err
		}
	}
	if len(output.Modalities) > 0 && (len(output.Modalities) != 1 || output.Modalities[0] != "text") {
		next, err := lossy(format, policy, "$.output.modalities", "output_modalities_not_supported")
		*diagnostics = append(*diagnostics, next...)
		if err != nil {
			return err
		}
	}
	return nil
}

func validateGenerationBounds(format llmprotocol.Format, request llmprotocol.Request, maximumTemperature float64, maximumStops int) error {
	if request.Output.MaxTokens != nil && *request.Output.MaxTokens < 1 {
		return translationError(format, "$.output.max_tokens", "invalid_max_tokens", "maximum output tokens must be positive")
	}
	if request.Sampling.Temperature != nil && (*request.Sampling.Temperature < 0 || *request.Sampling.Temperature > maximumTemperature) {
		return translationError(format, "$.sampling.temperature", "sampling_out_of_range", fmt.Sprintf("temperature must be in [0,%v]", maximumTemperature))
	}
	if request.Sampling.TopP != nil && (*request.Sampling.TopP < 0 || *request.Sampling.TopP > 1) {
		return translationError(format, "$.sampling.top_p", "sampling_out_of_range", "top_p must be in [0,1]")
	}
	if len(request.Output.StopSequences) > maximumStops {
		return translationError(format, "$.output.stop_sequences", "too_many_stop_sequences", fmt.Sprintf("at most %d stop sequences are supported", maximumStops))
	}
	for _, value := range request.Output.StopSequences {
		if value == "" {
			return translationError(format, "$.output.stop_sequences", "invalid_stop_sequence", "stop sequences must not be empty")
		}
	}
	return nil
}

func rawJSONString(value string) json.RawMessage {
	encoded, _ := json.Marshal(value)
	return encoded
}

func validateRole(format llmprotocol.Format, path string, value string) (llmprotocol.Role, error) {
	role := llmprotocol.Role(value)
	switch role {
	case llmprotocol.RoleSystem, llmprotocol.RoleDeveloper, llmprotocol.RoleUser, llmprotocol.RoleAssistant, llmprotocol.RoleTool:
		return role, nil
	default:
		return "", translationError(format, path, "unsupported_role", fmt.Sprintf("role %q is not supported", value))
	}
}

type neutralResponseFormat struct {
	Type        string
	Name        string
	Description string
	Schema      json.RawMessage
	Strict      *bool
}

// decodeNeutralResponseFormat accepts both the flattened Responses/Gemini
// representation and Chat Completions' nested json_schema representation.
func decodeNeutralResponseFormat(format llmprotocol.Format, path string, raw json.RawMessage) (neutralResponseFormat, error) {
	object, err := decodeObject(format, raw)
	if err != nil {
		return neutralResponseFormat{}, translationError(format, path, "invalid_response_format", "response format must be an object")
	}
	result := neutralResponseFormat{}
	result.Type, err = optionalString(format, object, "type")
	if err != nil {
		return result, err
	}
	if nestedRaw, ok := object["json_schema"]; ok {
		delete(object, "json_schema")
		nested, nestedErr := decodeObject(format, nestedRaw)
		if nestedErr != nil {
			return result, translationError(format, path+".json_schema", "invalid_response_format", "json_schema must be an object")
		}
		object = nested
		if result.Type == "" {
			result.Type = "json_schema"
		}
	}
	result.Name, err = optionalString(format, object, "name")
	if err == nil {
		result.Description, err = optionalString(format, object, "description")
	}
	if err == nil {
		result.Strict, err = optionalBool(format, object, "strict")
	}
	if err != nil {
		return result, err
	}
	if schema, ok := object["schema"]; ok {
		delete(object, "schema")
		result.Schema = cloneRaw(schema)
	}
	if len(object) != 0 {
		return result, translationError(format, path, "unsupported_response_format", "response format contains provider-specific fields")
	}
	return result, nil
}

func encodeNeutralJSONSchema(value neutralResponseFormat) json.RawMessage {
	object := map[string]any{"type": "json_schema", "schema": json.RawMessage(cloneRaw(value.Schema))}
	if value.Name != "" {
		object["name"] = value.Name
	}
	if value.Description != "" {
		object["description"] = value.Description
	}
	if value.Strict != nil {
		object["strict"] = *value.Strict
	}
	encoded, _ := json.Marshal(object)
	return encoded
}
