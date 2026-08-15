// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package codec

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"sync"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

type EmbeddingRequestResult struct {
	Request     llmprotocol.EmbeddingRequest
	Diagnostics []llmprotocol.Diagnostic
}

type EmbeddingResponseResult struct {
	Response    llmprotocol.EmbeddingResponse
	Diagnostics []llmprotocol.Diagnostic
}

type EmbeddingCodec interface {
	Format() llmprotocol.Format
	DecodeEmbeddingRequest(json.RawMessage, llmprotocol.Policy) (EmbeddingRequestResult, error)
	EncodeEmbeddingRequest(llmprotocol.EmbeddingRequest, llmprotocol.Policy) (WireResult, error)
	DecodeEmbeddingResponse(json.RawMessage, llmprotocol.Policy) (EmbeddingResponseResult, error)
	EncodeEmbeddingResponse(llmprotocol.EmbeddingResponse, llmprotocol.Policy) (WireResult, error)
}

type EmbeddingRegistry struct {
	mu     sync.RWMutex
	codecs map[llmprotocol.Format]EmbeddingCodec
}

func NewEmbeddingRegistry(codecs ...EmbeddingCodec) *EmbeddingRegistry {
	registry := &EmbeddingRegistry{codecs: make(map[llmprotocol.Format]EmbeddingCodec, len(codecs))}
	for _, value := range codecs {
		registry.Register(value)
	}
	return registry
}

func NewDefaultEmbeddingRegistry() *EmbeddingRegistry {
	return NewEmbeddingRegistry(OpenAIEmbeddings{}, GeminiEmbeddings{}, BedrockTitanEmbeddings{}, BedrockCohereEmbeddings{})
}

func (r *EmbeddingRegistry) Register(value EmbeddingCodec) {
	if value == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.codecs[value.Format()] = value
}

func (r *EmbeddingRegistry) Codec(format llmprotocol.Format) (EmbeddingCodec, error) {
	r.mu.RLock()
	value := r.codecs[format]
	r.mu.RUnlock()
	if value == nil {
		return nil, fmt.Errorf("llm embedding codec %q is not registered", format)
	}
	return value, nil
}

func (r *EmbeddingRegistry) TranslateRequest(source, target llmprotocol.Format, body json.RawMessage, policy llmprotocol.Policy) (WireResult, error) {
	sourceCodec, err := r.Codec(source)
	if err != nil {
		return WireResult{}, err
	}
	targetCodec, err := r.Codec(target)
	if err != nil {
		return WireResult{}, err
	}
	decoded, err := sourceCodec.DecodeEmbeddingRequest(body, policy)
	if err != nil {
		return WireResult{}, err
	}
	if source != target {
		decoded.Request.ClearPreservation()
		if len(decoded.Request.Extensions) != 0 {
			next, nextErr := lossy(target, policy, "$.extensions", "provider_extensions_not_portable")
			decoded.Diagnostics = append(decoded.Diagnostics, next...)
			if nextErr != nil {
				return WireResult{Diagnostics: attachFormats(decoded.Diagnostics, source, target)}, nextErr
			}
			decoded.Request.Extensions = nil
		}
	}
	encoded, err := targetCodec.EncodeEmbeddingRequest(decoded.Request, policy)
	if err != nil {
		return WireResult{Diagnostics: attachFormats(decoded.Diagnostics, source, target)}, err
	}
	encoded.Diagnostics = attachFormats(append(decoded.Diagnostics, encoded.Diagnostics...), source, target)
	return encoded, nil
}

func (r *EmbeddingRegistry) TranslateResponse(source, target llmprotocol.Format, body json.RawMessage, policy llmprotocol.Policy) (WireResult, error) {
	sourceCodec, err := r.Codec(source)
	if err != nil {
		return WireResult{}, err
	}
	targetCodec, err := r.Codec(target)
	if err != nil {
		return WireResult{}, err
	}
	decoded, err := sourceCodec.DecodeEmbeddingResponse(body, policy)
	if err != nil {
		return WireResult{}, err
	}
	if source != target {
		decoded.Response.ClearPreservation()
		if len(decoded.Response.Extensions) != 0 {
			next, nextErr := lossy(target, policy, "$.extensions", "provider_extensions_not_portable")
			decoded.Diagnostics = append(decoded.Diagnostics, next...)
			if nextErr != nil {
				return WireResult{Diagnostics: attachFormats(decoded.Diagnostics, source, target)}, nextErr
			}
			decoded.Response.Extensions = nil
		}
		for index := range decoded.Response.Data {
			if len(decoded.Response.Data[index].Extensions) == 0 {
				continue
			}
			next, nextErr := lossy(target, policy, "$.data[].extensions", "provider_extensions_not_portable")
			decoded.Diagnostics = append(decoded.Diagnostics, next...)
			if nextErr != nil {
				return WireResult{Diagnostics: attachFormats(decoded.Diagnostics, source, target)}, nextErr
			}
			decoded.Response.Data[index].Extensions = nil
		}
	}
	encoded, err := targetCodec.EncodeEmbeddingResponse(decoded.Response, policy)
	if err != nil {
		return WireResult{Diagnostics: attachFormats(decoded.Diagnostics, source, target)}, err
	}
	encoded.Diagnostics = attachFormats(append(decoded.Diagnostics, encoded.Diagnostics...), source, target)
	return encoded, nil
}

func preservedEmbeddingRequest(request llmprotocol.EmbeddingRequest, format llmprotocol.Format, policy llmprotocol.Policy) (json.RawMessage, bool) {
	if policy.Effective().Preservation != llmprotocol.PreserveInMemory {
		return nil, false
	}
	raw, ok := request.Preservation.Requests[format]
	return cloneRaw(raw), ok
}

func preservedEmbeddingResponse(response llmprotocol.EmbeddingResponse, format llmprotocol.Format, policy llmprotocol.Policy) (json.RawMessage, bool) {
	if policy.Effective().Preservation != llmprotocol.PreserveInMemory {
		return nil, false
	}
	raw, ok := response.Preservation.Responses[format]
	return cloneRaw(raw), ok
}

// OpenAIEmbeddings implements POST /v1/embeddings.
type OpenAIEmbeddings struct{}

func (OpenAIEmbeddings) Format() llmprotocol.Format { return llmprotocol.FormatOpenAIEmbeddings }

func (OpenAIEmbeddings) DecodeEmbeddingRequest(body json.RawMessage, policy llmprotocol.Policy) (EmbeddingRequestResult, error) {
	const format = llmprotocol.FormatOpenAIEmbeddings
	object, err := decodeObject(format, body)
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	request := llmprotocol.EmbeddingRequest{Preservation: preserveBody(policy, format, body, true)}
	request.Model, err = optionalString(format, object, "model")
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	rawInput, exists := object["input"]
	if !exists {
		return EmbeddingRequestResult{}, translationError(format, "$.input", "missing_input", "embedding input is required")
	}
	delete(object, "input")
	request.Inputs, err = decodeOpenAIEmbeddingInputs(rawInput)
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	request.EncodingFormat, err = optionalString(format, object, "encoding_format")
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	request.Dimensions, err = optionalInt(format, object, "dimensions")
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	request.User, err = optionalString(format, object, "user")
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	if err := request.Validate(); err != nil {
		return EmbeddingRequestResult{}, translationError(format, "$.input", "invalid_input", err.Error())
	}
	extensions, diagnostics, err := collectExtensions(format, object, policy)
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	request.Extensions = extensions
	return EmbeddingRequestResult{Request: request, Diagnostics: diagnostics}, nil
}

func decodeOpenAIEmbeddingInputs(raw json.RawMessage) ([]llmprotocol.EmbeddingInput, error) {
	const format = llmprotocol.FormatOpenAIEmbeddings
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, translationError(format, "$.input", "missing_input", "embedding input is required")
	}
	if trimmed[0] == '"' {
		value, err := decodeString(format, "$.input", trimmed)
		return []llmprotocol.EmbeddingInput{{Type: llmprotocol.EmbeddingInputText, Text: value}}, err
	}
	var items []json.RawMessage
	if json.Unmarshal(trimmed, &items) != nil || len(items) == 0 {
		return nil, translationError(format, "$.input", "invalid_input", "input must be text, texts, tokens, or token arrays")
	}
	first := bytes.TrimSpace(items[0])
	if len(first) == 0 {
		return nil, translationError(format, "$.input", "invalid_input", "input contains an invalid item")
	}
	switch first[0] {
	case '"':
		result := make([]llmprotocol.EmbeddingInput, 0, len(items))
		for index, item := range items {
			text, err := decodeString(format, fmt.Sprintf("$.input[%d]", index), item)
			if err != nil {
				return nil, err
			}
			result = append(result, llmprotocol.EmbeddingInput{Type: llmprotocol.EmbeddingInputText, Text: text})
		}
		return result, nil
	case '[':
		result := make([]llmprotocol.EmbeddingInput, 0, len(items))
		for index, item := range items {
			tokens, err := decodeTokenArray(format, fmt.Sprintf("$.input[%d]", index), item)
			if err != nil {
				return nil, err
			}
			result = append(result, llmprotocol.EmbeddingInput{Type: llmprotocol.EmbeddingInputTokens, Tokens: tokens})
		}
		return result, nil
	default:
		tokens, err := decodeTokenArray(format, "$.input", trimmed)
		if err != nil {
			return nil, err
		}
		return []llmprotocol.EmbeddingInput{{Type: llmprotocol.EmbeddingInputTokens, Tokens: tokens}}, nil
	}
}

func decodeTokenArray(format llmprotocol.Format, path string, raw json.RawMessage) ([]int64, error) {
	var values []json.RawMessage
	if json.Unmarshal(raw, &values) != nil || len(values) == 0 {
		return nil, translationError(format, path, "invalid_token_array", "token input must be a non-empty integer array")
	}
	result := make([]int64, len(values))
	for index, value := range values {
		decoder := json.NewDecoder(bytes.NewReader(value))
		decoder.UseNumber()
		var number json.Number
		if decoder.Decode(&number) != nil {
			return nil, translationError(format, path, "invalid_token_array", "token input must contain integers")
		}
		parsed, err := number.Int64()
		if err != nil || parsed < 0 {
			return nil, translationError(format, path, "invalid_token_array", "token input must contain non-negative integers")
		}
		result[index] = parsed
	}
	return result, nil
}

func (OpenAIEmbeddings) EncodeEmbeddingRequest(request llmprotocol.EmbeddingRequest, policy llmprotocol.Policy) (WireResult, error) {
	const format = llmprotocol.FormatOpenAIEmbeddings
	if raw, ok := preservedEmbeddingRequest(request, format, policy); ok {
		return WireResult{Body: raw}, nil
	}
	if err := request.Validate(); err != nil {
		return WireResult{}, translationError(format, "$.input", "invalid_input", err.Error())
	}
	input, err := encodeOpenAIEmbeddingInputs(request.Inputs)
	if err != nil {
		return WireResult{}, err
	}
	object := map[string]any{"model": request.Model, "input": input}
	if request.EncodingFormat != "" {
		if request.EncodingFormat != "float" && request.EncodingFormat != "base64" {
			return WireResult{}, translationError(format, "$.encoding_format", "unsupported_encoding", "OpenAI embeddings supports float or base64 encoding")
		}
		object["encoding_format"] = request.EncodingFormat
	}
	putInt(object, "dimensions", request.Dimensions)
	if request.User != "" {
		object["user"] = request.User
	}
	var diagnostics []llmprotocol.Diagnostic
	for path, active := range map[string]bool{
		"$.task_type": request.TaskType != "", "$.title": request.Title != "", "$.input_type": request.InputType != "",
		"$.truncate": request.Truncate != "", "$.normalize": request.Normalize != nil, "$.auto_truncate": request.AutoTruncate != nil,
	} {
		if !active {
			continue
		}
		next, nextErr := lossy(format, policy, path, "embedding_control_not_supported")
		diagnostics = append(diagnostics, next...)
		if nextErr != nil {
			return WireResult{Diagnostics: diagnostics}, nextErr
		}
	}
	mergeExtensions(object, request.Extensions)
	body, err := marshalObject(format, object)
	return WireResult{Body: body, Diagnostics: diagnostics}, err
}

func encodeOpenAIEmbeddingInputs(inputs []llmprotocol.EmbeddingInput) (any, error) {
	firstType := inputs[0].Type
	for _, input := range inputs[1:] {
		if input.Type != firstType {
			return nil, translationError(llmprotocol.FormatOpenAIEmbeddings, "$.input", "mixed_input_types", "OpenAI embeddings cannot mix text and token-array inputs")
		}
	}
	if firstType == llmprotocol.EmbeddingInputText {
		if len(inputs) == 1 {
			return inputs[0].Text, nil
		}
		values := make([]string, len(inputs))
		for index := range inputs {
			values[index] = inputs[index].Text
		}
		return values, nil
	}
	if len(inputs) == 1 {
		return inputs[0].Tokens, nil
	}
	values := make([][]int64, len(inputs))
	for index := range inputs {
		values[index] = inputs[index].Tokens
	}
	return values, nil
}

func (OpenAIEmbeddings) DecodeEmbeddingResponse(body json.RawMessage, policy llmprotocol.Policy) (EmbeddingResponseResult, error) {
	const format = llmprotocol.FormatOpenAIEmbeddings
	object, err := decodeObject(format, body)
	if err != nil {
		return EmbeddingResponseResult{}, err
	}
	response := llmprotocol.EmbeddingResponse{Preservation: preserveBody(policy, format, body, false)}
	response.Model, err = optionalString(format, object, "model")
	if err != nil {
		return EmbeddingResponseResult{}, err
	}
	delete(object, "object")
	rawData, exists := object["data"]
	if !exists {
		return EmbeddingResponseResult{}, translationError(format, "$.data", "missing_embeddings", "embedding response data is required")
	}
	delete(object, "data")
	items, err := decodeArray(format, "$.data", rawData)
	if err != nil {
		return EmbeddingResponseResult{}, err
	}
	var diagnostics []llmprotocol.Diagnostic
	for fallback, raw := range items {
		item, itemErr := decodeObject(format, raw)
		if itemErr != nil {
			return EmbeddingResponseResult{}, itemErr
		}
		delete(item, "object")
		index := fallback
		if parsed, indexErr := optionalInt(format, item, "index"); indexErr != nil {
			return EmbeddingResponseResult{}, indexErr
		} else if parsed != nil {
			index = int(*parsed)
		}
		rawEmbedding, exists := item["embedding"]
		if !exists {
			return EmbeddingResponseResult{}, translationError(format, "$.data[].embedding", "missing_embedding", "embedding value is required")
		}
		delete(item, "embedding")
		tensor, encoding, tensorErr := decodeOpenAIEmbeddingTensor(rawEmbedding)
		if tensorErr != nil {
			return EmbeddingResponseResult{}, tensorErr
		}
		if response.EncodingFormat == "" {
			response.EncodingFormat = encoding
		} else if response.EncodingFormat != encoding {
			return EmbeddingResponseResult{}, translationError(format, "$.data[].embedding", "mixed_encoding", "response mixes float and base64 embeddings")
		}
		extensions, next, extensionErr := collectExtensions(format, item, policy)
		diagnostics = append(diagnostics, next...)
		if extensionErr != nil {
			return EmbeddingResponseResult{}, extensionErr
		}
		response.Data = append(response.Data, llmprotocol.EmbeddingOutput{Index: index, Embedding: tensor, Extensions: extensions})
	}
	if rawUsage, ok := object["usage"]; ok {
		delete(object, "usage")
		response.Usage, err = decodeOpenAIUsage(rawUsage, format)
		if err != nil {
			return EmbeddingResponseResult{}, err
		}
	}
	response.Extensions, diagnostics, err = collectAndAppendExtensions(format, object, policy, diagnostics)
	if err != nil {
		return EmbeddingResponseResult{}, err
	}
	return EmbeddingResponseResult{Response: response, Diagnostics: diagnostics}, nil
}

func decodeOpenAIEmbeddingTensor(raw json.RawMessage) (llmprotocol.EmbeddingTensor, string, error) {
	const format = llmprotocol.FormatOpenAIEmbeddings
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) != 0 && trimmed[0] == '"' {
		encoded, err := decodeString(format, "$.data[].embedding", trimmed)
		if err != nil {
			return llmprotocol.EmbeddingTensor{}, "", err
		}
		binaryValue, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil || len(binaryValue) == 0 || len(binaryValue)%4 != 0 {
			return llmprotocol.EmbeddingTensor{}, "", translationError(format, "$.data[].embedding", "invalid_base64_embedding", "base64 embedding must contain little-endian float32 values")
		}
		values := make([]float64, len(binaryValue)/4)
		for index := range values {
			values[index] = float64(math.Float32frombits(binary.LittleEndian.Uint32(binaryValue[index*4:])))
		}
		return llmprotocol.NewEmbeddingVector(values), "base64", nil
	}
	var values []float64
	if json.Unmarshal(trimmed, &values) != nil || len(values) == 0 {
		return llmprotocol.EmbeddingTensor{}, "", translationError(format, "$.data[].embedding", "invalid_embedding", "embedding must be a non-empty number array or base64 string")
	}
	return llmprotocol.NewEmbeddingVector(values), "float", nil
}

func (OpenAIEmbeddings) EncodeEmbeddingResponse(response llmprotocol.EmbeddingResponse, policy llmprotocol.Policy) (WireResult, error) {
	const format = llmprotocol.FormatOpenAIEmbeddings
	if raw, ok := preservedEmbeddingResponse(response, format, policy); ok {
		return WireResult{Body: raw}, nil
	}
	if err := response.Validate(); err != nil {
		return WireResult{}, translationError(format, "$.data", "invalid_embedding", err.Error())
	}
	encoding := response.EncodingFormat
	if encoding == "" {
		encoding = "float"
	}
	if encoding != "float" && encoding != "base64" {
		return WireResult{}, translationError(format, "$.encoding_format", "unsupported_encoding", "OpenAI embeddings supports float or base64 encoding")
	}
	data := make([]any, 0, len(response.Data))
	for _, output := range response.Data {
		if output.Embedding.Rank() != 1 {
			return WireResult{}, translationError(format, "$.data[].embedding", "rank_not_supported", "OpenAI embeddings cannot represent a rank-2 embedding tensor")
		}
		var embedding any = output.Embedding.Values
		if encoding == "base64" {
			binaryValue := make([]byte, len(output.Embedding.Values)*4)
			for index, value := range output.Embedding.Values {
				binary.LittleEndian.PutUint32(binaryValue[index*4:], math.Float32bits(float32(value)))
			}
			embedding = base64.StdEncoding.EncodeToString(binaryValue)
		}
		item := map[string]any{"object": "embedding", "index": output.Index, "embedding": embedding}
		mergeExtensions(item, output.Extensions)
		data = append(data, item)
	}
	object := map[string]any{"object": "list", "data": data, "model": response.Model}
	if hasUsage(response.Usage) {
		usage := map[string]any{}
		putInt(usage, "prompt_tokens", response.Usage.InputTokens)
		putInt(usage, "total_tokens", response.Usage.TotalTokens)
		object["usage"] = usage
	}
	mergeExtensions(object, response.Extensions)
	body, err := marshalObject(format, object)
	return WireResult{Body: body}, err
}

// GeminiEmbeddings implements models.embedContent and models.batchEmbedContents.
type GeminiEmbeddings struct{}

func (GeminiEmbeddings) Format() llmprotocol.Format { return llmprotocol.FormatGeminiEmbeddings }

func (GeminiEmbeddings) DecodeEmbeddingRequest(body json.RawMessage, policy llmprotocol.Policy) (EmbeddingRequestResult, error) {
	const format = llmprotocol.FormatGeminiEmbeddings
	object, err := decodeObject(format, body)
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	request := llmprotocol.EmbeddingRequest{Preservation: preserveBody(policy, format, body, true), EncodingFormat: "float"}
	var entries []map[string]json.RawMessage
	if rawRequests, ok := object["requests"]; ok {
		delete(object, "requests")
		values, arrayErr := decodeArray(format, "$.requests", rawRequests)
		if arrayErr != nil {
			return EmbeddingRequestResult{}, arrayErr
		}
		for _, value := range values {
			entry, objectErr := decodeObject(format, value)
			if objectErr != nil {
				return EmbeddingRequestResult{}, objectErr
			}
			entries = append(entries, entry)
		}
	} else {
		entries = []map[string]json.RawMessage{object}
		object = make(map[string]json.RawMessage)
	}
	var diagnostics []llmprotocol.Diagnostic
	for index, entry := range entries {
		model, modelErr := optionalString(format, entry, "model")
		if modelErr != nil {
			return EmbeddingRequestResult{}, modelErr
		}
		if request.Model == "" {
			request.Model = strings.TrimPrefix(model, "models/")
		} else if model != "" && strings.TrimPrefix(model, "models/") != request.Model {
			return EmbeddingRequestResult{}, translationError(format, "$.requests[].model", "mixed_models", "batch embedding requests must use one model")
		}
		rawContent, ok := entry["content"]
		if !ok {
			return EmbeddingRequestResult{}, translationError(format, "$.content", "missing_input", "Gemini embedding content is required")
		}
		delete(entry, "content")
		text, textErr := decodeGeminiEmbeddingContent(rawContent)
		if textErr != nil {
			return EmbeddingRequestResult{}, textErr
		}
		request.Inputs = append(request.Inputs, llmprotocol.EmbeddingInput{Type: llmprotocol.EmbeddingInputText, Text: text})
		config := entry
		nestedConfig := false
		if rawConfig, ok := entry["embedContentConfig"]; ok {
			delete(entry, "embedContentConfig")
			nestedConfig = true
			config, err = decodeObject(format, rawConfig)
			if err != nil {
				return EmbeddingRequestResult{}, err
			}
		}
		if err := mergeGeminiEmbeddingConfig(&request, config, index); err != nil {
			return EmbeddingRequestResult{}, err
		}
		_, next, extensionErr := collectExtensions(format, config, policy)
		diagnostics = append(diagnostics, next...)
		if extensionErr != nil {
			return EmbeddingRequestResult{}, extensionErr
		}
		if nestedConfig {
			_, next, extensionErr = collectExtensions(format, entry, policy)
			diagnostics = append(diagnostics, next...)
			if extensionErr != nil {
				return EmbeddingRequestResult{}, extensionErr
			}
		}
	}
	if err := request.Validate(); err != nil {
		return EmbeddingRequestResult{}, translationError(format, "$.content", "invalid_input", err.Error())
	}
	request.Extensions, diagnostics, err = collectAndAppendExtensions(format, object, policy, diagnostics)
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	return EmbeddingRequestResult{Request: request, Diagnostics: diagnostics}, nil
}

func mergeGeminiEmbeddingConfig(request *llmprotocol.EmbeddingRequest, config map[string]json.RawMessage, index int) error {
	const format = llmprotocol.FormatGeminiEmbeddings
	values := struct {
		task, title string
		dimensions  *int64
		auto        *bool
	}{}
	var err error
	values.task, err = optionalString(format, config, "taskType")
	if err == nil {
		values.title, err = optionalString(format, config, "title")
	}
	if err == nil {
		values.dimensions, err = optionalInt(format, config, "outputDimensionality")
	}
	if err == nil {
		values.auto, err = optionalBool(format, config, "autoTruncate")
	}
	if err != nil {
		return err
	}
	if index == 0 {
		request.TaskType, request.Title, request.Dimensions, request.AutoTruncate = values.task, values.title, values.dimensions, values.auto
		return nil
	}
	if values.task != request.TaskType || values.title != request.Title || !equalInt64Pointers(values.dimensions, request.Dimensions) || !equalBoolPointers(values.auto, request.AutoTruncate) {
		return translationError(format, "$.requests[]", "mixed_config", "batch embedding request controls must be identical")
	}
	return nil
}

func equalInt64Pointers(left, right *int64) bool {
	return (left == nil && right == nil) || (left != nil && right != nil && *left == *right)
}

func equalBoolPointers(left, right *bool) bool {
	return (left == nil && right == nil) || (left != nil && right != nil && *left == *right)
}

func decodeGeminiEmbeddingContent(raw json.RawMessage) (string, error) {
	const format = llmprotocol.FormatGeminiEmbeddings
	content, err := decodeObject(format, raw)
	if err != nil {
		return "", err
	}
	delete(content, "role")
	parts, err := decodeArray(format, "$.content.parts", content["parts"])
	if err != nil || len(parts) == 0 {
		return "", translationError(format, "$.content.parts", "invalid_input", "Gemini embedding content requires text parts")
	}
	var text strings.Builder
	for _, rawPart := range parts {
		part, objectErr := decodeObject(format, rawPart)
		if objectErr != nil {
			return "", objectErr
		}
		value, stringErr := optionalString(format, part, "text")
		if stringErr != nil || len(part) != 0 {
			return "", translationError(format, "$.content.parts[]", "unsupported_input", "Gemini embedding codec supports text parts only")
		}
		text.WriteString(value)
	}
	return text.String(), nil
}

func (GeminiEmbeddings) EncodeEmbeddingRequest(request llmprotocol.EmbeddingRequest, policy llmprotocol.Policy) (WireResult, error) {
	const format = llmprotocol.FormatGeminiEmbeddings
	if raw, ok := preservedEmbeddingRequest(request, format, policy); ok {
		return WireResult{Body: raw}, nil
	}
	if err := request.Validate(); err != nil {
		return WireResult{}, translationError(format, "$.content", "invalid_input", err.Error())
	}
	if request.EncodingFormat != "" && request.EncodingFormat != "float" {
		return WireResult{}, translationError(format, "$.encoding_format", "unsupported_encoding", "Gemini embeddings returns floating-point values")
	}
	if request.User != "" || request.InputType != "" || request.Truncate != "" || request.Normalize != nil {
		return WireResult{}, translationError(format, "$", "embedding_control_not_supported", "request contains controls Gemini EmbedContent cannot represent")
	}
	config := map[string]any{}
	if request.TaskType != "" {
		config["taskType"] = request.TaskType
	}
	if request.Title != "" {
		config["title"] = request.Title
	}
	putInt(config, "outputDimensionality", request.Dimensions)
	if request.AutoTruncate != nil {
		config["autoTruncate"] = *request.AutoTruncate
	}
	entries := make([]any, 0, len(request.Inputs))
	for _, input := range request.Inputs {
		if input.Type != llmprotocol.EmbeddingInputText {
			return WireResult{}, translationError(format, "$.content", "token_input_not_supported", "Gemini EmbedContent cannot represent pre-tokenized input")
		}
		entry := map[string]any{"content": map[string]any{"parts": []any{map[string]any{"text": input.Text}}}}
		if len(config) != 0 {
			entry["embedContentConfig"] = config
		}
		if request.Model != "" && len(request.Inputs) > 1 {
			entry["model"] = "models/" + strings.TrimPrefix(request.Model, "models/")
		}
		entries = append(entries, entry)
	}
	var object map[string]any
	if len(entries) == 1 {
		object = entries[0].(map[string]any)
	} else {
		object = map[string]any{"requests": entries}
	}
	mergeExtensions(object, request.Extensions)
	body, err := marshalObject(format, object)
	return WireResult{Body: body}, err
}

func (GeminiEmbeddings) DecodeEmbeddingResponse(body json.RawMessage, policy llmprotocol.Policy) (EmbeddingResponseResult, error) {
	const format = llmprotocol.FormatGeminiEmbeddings
	object, err := decodeObject(format, body)
	if err != nil {
		return EmbeddingResponseResult{}, err
	}
	response := llmprotocol.EmbeddingResponse{EncodingFormat: "float", Preservation: preserveBody(policy, format, body, false)}
	var rawEmbeddings []json.RawMessage
	if raw, ok := object["embeddings"]; ok {
		delete(object, "embeddings")
		rawEmbeddings, err = decodeArray(format, "$.embeddings", raw)
	} else if raw, ok := object["embedding"]; ok {
		delete(object, "embedding")
		rawEmbeddings = []json.RawMessage{raw}
	} else {
		return EmbeddingResponseResult{}, translationError(format, "$.embedding", "missing_embeddings", "Gemini embedding response is missing embedding data")
	}
	if err != nil {
		return EmbeddingResponseResult{}, err
	}
	var diagnostics []llmprotocol.Diagnostic
	for index, raw := range rawEmbeddings {
		tensor, extensions, next, tensorErr := decodeGeminiEmbeddingTensor(raw, policy)
		diagnostics = append(diagnostics, next...)
		if tensorErr != nil {
			return EmbeddingResponseResult{}, tensorErr
		}
		response.Data = append(response.Data, llmprotocol.EmbeddingOutput{Index: index, Embedding: tensor, Extensions: extensions})
	}
	if rawUsage, ok := object["usageMetadata"]; ok {
		delete(object, "usageMetadata")
		response.Usage, err = decodeGeminiEmbeddingUsage(rawUsage)
		if err != nil {
			return EmbeddingResponseResult{}, err
		}
	}
	response.Extensions, diagnostics, err = collectAndAppendExtensions(format, object, policy, diagnostics)
	if err != nil {
		return EmbeddingResponseResult{}, err
	}
	return EmbeddingResponseResult{Response: response, Diagnostics: diagnostics}, nil
}

func decodeGeminiEmbeddingTensor(raw json.RawMessage, policy llmprotocol.Policy) (llmprotocol.EmbeddingTensor, llmprotocol.Extensions, []llmprotocol.Diagnostic, error) {
	const format = llmprotocol.FormatGeminiEmbeddings
	object, err := decodeObject(format, raw)
	if err != nil {
		return llmprotocol.EmbeddingTensor{}, nil, nil, err
	}
	rawValues, ok := object["values"]
	if !ok {
		return llmprotocol.EmbeddingTensor{}, nil, nil, translationError(format, "$.embedding.values", "missing_embedding", "embedding values are required")
	}
	delete(object, "values")
	var values []float64
	var inferredShape []int64
	if json.Unmarshal(rawValues, &values) != nil {
		var rows [][]float64
		if json.Unmarshal(rawValues, &rows) != nil {
			return llmprotocol.EmbeddingTensor{}, nil, nil, translationError(format, "$.embedding.values", "invalid_embedding", "values must be a dense number vector or matrix")
		}
		matrix, matrixErr := llmprotocol.NewEmbeddingMatrix(rows)
		if matrixErr != nil {
			return llmprotocol.EmbeddingTensor{}, nil, nil, translationError(format, "$.embedding.values", "invalid_embedding", matrixErr.Error())
		}
		values, inferredShape = matrix.Values, matrix.Shape
	}
	shape := inferredShape
	if rawShape, exists := object["shape"]; exists {
		delete(object, "shape")
		if json.Unmarshal(rawShape, &shape) != nil {
			return llmprotocol.EmbeddingTensor{}, nil, nil, translationError(format, "$.embedding.shape", "invalid_shape", "embedding shape must be an integer array")
		}
	}
	if len(shape) == 0 {
		shape = []int64{int64(len(values))}
	}
	tensor := llmprotocol.EmbeddingTensor{Shape: shape, Values: values}
	if err := tensor.Validate(); err != nil {
		return llmprotocol.EmbeddingTensor{}, nil, nil, translationError(format, "$.embedding", "invalid_embedding", err.Error())
	}
	extensions, diagnostics, err := collectExtensions(format, object, policy)
	return tensor, extensions, diagnostics, err
}

func decodeGeminiEmbeddingUsage(raw json.RawMessage) (llmprotocol.Usage, error) {
	const format = llmprotocol.FormatGeminiEmbeddings
	object, err := decodeObject(format, raw)
	if err != nil {
		return llmprotocol.Usage{}, err
	}
	usage := llmprotocol.Usage{}
	usage.InputTokens, err = optionalInt(format, object, "promptTokenCount")
	if err != nil {
		return usage, err
	}
	usage.TotalTokens, err = optionalInt(format, object, "totalTokenCount")
	if err != nil {
		return usage, err
	}
	if usage.TotalTokens == nil && usage.InputTokens != nil {
		value := *usage.InputTokens
		usage.TotalTokens = &value
	}
	zero := int64(0)
	usage.OutputTokens = &zero
	for name, rawValue := range object {
		if usage.ProviderComponents == nil {
			usage.ProviderComponents = make(map[string]int64)
		}
		if strings.HasSuffix(name, "TokenCount") {
			var value int64
			if json.Unmarshal(rawValue, &value) == nil {
				usage.ProviderComponents[name] = value
			}
		}
	}
	return usage, nil
}

func (GeminiEmbeddings) EncodeEmbeddingResponse(response llmprotocol.EmbeddingResponse, policy llmprotocol.Policy) (WireResult, error) {
	const format = llmprotocol.FormatGeminiEmbeddings
	if raw, ok := preservedEmbeddingResponse(response, format, policy); ok {
		return WireResult{Body: raw}, nil
	}
	if err := response.Validate(); err != nil {
		return WireResult{}, translationError(format, "$.embedding", "invalid_embedding", err.Error())
	}
	encoded := make([]any, 0, len(response.Data))
	for _, output := range response.Data {
		value := map[string]any{"values": output.Embedding.Values}
		if output.Embedding.Rank() == 2 {
			value["shape"] = output.Embedding.Shape
		}
		mergeExtensions(value, output.Extensions)
		encoded = append(encoded, value)
	}
	object := map[string]any{}
	if len(encoded) == 1 {
		object["embedding"] = encoded[0]
	} else {
		object["embeddings"] = encoded
	}
	if hasUsage(response.Usage) {
		usage := map[string]any{}
		putInt(usage, "promptTokenCount", response.Usage.InputTokens)
		putInt(usage, "totalTokenCount", response.Usage.TotalTokens)
		for name, value := range response.Usage.ProviderComponents {
			if _, exists := usage[name]; !exists {
				usage[name] = value
			}
		}
		object["usageMetadata"] = usage
	}
	mergeExtensions(object, response.Extensions)
	body, err := marshalObject(format, object)
	return WireResult{Body: body}, err
}
