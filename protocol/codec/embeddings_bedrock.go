// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package codec

import (
	"bytes"
	"encoding/json"
	"strings"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

// BedrockTitanEmbeddings implements the InvokeModel payload shared by Amazon
// Titan Text Embeddings G1 and V2. Model selection remains a transport concern.
type BedrockTitanEmbeddings struct{}

func (BedrockTitanEmbeddings) Format() llmprotocol.Format {
	return llmprotocol.FormatBedrockTitanEmbeddings
}

func (BedrockTitanEmbeddings) DecodeEmbeddingRequest(body json.RawMessage, policy llmprotocol.Policy) (EmbeddingRequestResult, error) {
	const format = llmprotocol.FormatBedrockTitanEmbeddings
	object, err := decodeObject(format, body)
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	request := llmprotocol.EmbeddingRequest{EncodingFormat: "float", Preservation: preserveBody(policy, format, body, true)}
	text, err := optionalString(format, object, "inputText")
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	request.Inputs = []llmprotocol.EmbeddingInput{{Type: llmprotocol.EmbeddingInputText, Text: text}}
	request.Dimensions, err = optionalInt(format, object, "dimensions")
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	request.Normalize, err = optionalBool(format, object, "normalize")
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	if rawTypes, ok := object["embeddingTypes"]; ok {
		delete(object, "embeddingTypes")
		var types []string
		if json.Unmarshal(rawTypes, &types) != nil || len(types) != 1 || types[0] != "float" {
			object["embeddingTypes"] = rawTypes
		}
	}
	if err := request.Validate(); err != nil {
		return EmbeddingRequestResult{}, translationError(format, "$.inputText", "invalid_input", err.Error())
	}
	extensions, diagnostics, err := collectExtensions(format, object, policy)
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	request.Extensions = extensions
	return EmbeddingRequestResult{Request: request, Diagnostics: diagnostics}, nil
}

func (BedrockTitanEmbeddings) EncodeEmbeddingRequest(request llmprotocol.EmbeddingRequest, policy llmprotocol.Policy) (WireResult, error) {
	const format = llmprotocol.FormatBedrockTitanEmbeddings
	if raw, ok := preservedEmbeddingRequest(request, format, policy); ok {
		return WireResult{Body: raw}, nil
	}
	if err := request.Validate(); err != nil {
		return WireResult{}, translationError(format, "$.inputText", "invalid_input", err.Error())
	}
	if len(request.Inputs) != 1 || request.Inputs[0].Type != llmprotocol.EmbeddingInputText {
		return WireResult{}, translationError(format, "$.inputText", "input_not_supported", "Titan embeddings accepts exactly one text input per invocation")
	}
	if request.EncodingFormat != "" && request.EncodingFormat != "float" {
		return WireResult{}, translationError(format, "$.encoding_format", "unsupported_encoding", "Titan neutral translation supports float embeddings only")
	}
	if request.User != "" || request.TaskType != "" || request.Title != "" || request.InputType != "" || request.Truncate != "" || request.AutoTruncate != nil {
		return WireResult{}, translationError(format, "$", "embedding_control_not_supported", "request contains controls Titan embeddings cannot represent")
	}
	object := map[string]any{"inputText": request.Inputs[0].Text}
	titanG1 := strings.Contains(strings.ToLower(request.Model), "titan-embed-text-v1")
	if titanG1 {
		if request.Dimensions != nil || request.Normalize != nil {
			return WireResult{}, translationError(format, "$", "embedding_control_not_supported", "Titan G1 does not support dimensions or normalization controls")
		}
	} else {
		object["embeddingTypes"] = []string{"float"}
		putInt(object, "dimensions", request.Dimensions)
		if request.Normalize != nil {
			object["normalize"] = *request.Normalize
		}
	}
	mergeExtensions(object, request.Extensions)
	body, err := marshalObject(format, object)
	return WireResult{Body: body}, err
}

func (BedrockTitanEmbeddings) DecodeEmbeddingResponse(body json.RawMessage, policy llmprotocol.Policy) (EmbeddingResponseResult, error) {
	const format = llmprotocol.FormatBedrockTitanEmbeddings
	object, err := decodeObject(format, body)
	if err != nil {
		return EmbeddingResponseResult{}, err
	}
	response := llmprotocol.EmbeddingResponse{EncodingFormat: "float", Preservation: preserveBody(policy, format, body, false)}
	rawEmbedding, exists := object["embedding"]
	if exists {
		delete(object, "embedding")
	}
	if rawTyped, ok := object["embeddingsByType"]; ok {
		delete(object, "embeddingsByType")
		typed, objectErr := decodeObject(format, rawTyped)
		if objectErr != nil {
			return EmbeddingResponseResult{}, objectErr
		}
		if !exists {
			rawEmbedding, exists = typed["float"]
		}
		delete(typed, "float")
		if len(typed) != 0 {
			encoded, _ := json.Marshal(typed)
			object["embeddingsByType"] = encoded
		}
	}
	if !exists {
		return EmbeddingResponseResult{}, translationError(format, "$.embedding", "missing_embedding", "Titan response is missing a float embedding")
	}
	var values []float64
	if json.Unmarshal(rawEmbedding, &values) != nil || len(values) == 0 {
		return EmbeddingResponseResult{}, translationError(format, "$.embedding", "invalid_embedding", "Titan embedding must be a non-empty number array")
	}
	response.Data = []llmprotocol.EmbeddingOutput{{Index: 0, Embedding: llmprotocol.NewEmbeddingVector(values)}}
	response.Usage.InputTokens, err = optionalInt(format, object, "inputTextTokenCount")
	if err != nil {
		return EmbeddingResponseResult{}, err
	}
	if response.Usage.InputTokens != nil {
		value := *response.Usage.InputTokens
		zero := int64(0)
		response.Usage.OutputTokens = &zero
		response.Usage.TotalTokens = &value
	}
	extensions, diagnostics, err := collectExtensions(format, object, policy)
	if err != nil {
		return EmbeddingResponseResult{}, err
	}
	response.Extensions = extensions
	return EmbeddingResponseResult{Response: response, Diagnostics: diagnostics}, nil
}

func (BedrockTitanEmbeddings) EncodeEmbeddingResponse(response llmprotocol.EmbeddingResponse, policy llmprotocol.Policy) (WireResult, error) {
	const format = llmprotocol.FormatBedrockTitanEmbeddings
	if raw, ok := preservedEmbeddingResponse(response, format, policy); ok {
		return WireResult{Body: raw}, nil
	}
	if err := response.Validate(); err != nil {
		return WireResult{}, translationError(format, "$.embedding", "invalid_embedding", err.Error())
	}
	if len(response.Data) != 1 || response.Data[0].Embedding.Rank() != 1 {
		return WireResult{}, translationError(format, "$.embedding", "rank_not_supported", "Titan response requires exactly one rank-1 embedding")
	}
	object := map[string]any{
		"embedding":        response.Data[0].Embedding.Values,
		"embeddingsByType": map[string]any{"float": response.Data[0].Embedding.Values},
	}
	putInt(object, "inputTextTokenCount", response.Usage.InputTokens)
	mergeExtensions(object, response.Extensions)
	body, err := marshalObject(format, object)
	return WireResult{Body: body}, err
}

// BedrockCohereEmbeddings implements the text subset common to Cohere Embed v3
// and v4 InvokeModel payloads. Non-float and multimodal representations remain
// provider extensions and therefore cannot cross formats silently.
type BedrockCohereEmbeddings struct{}

func (BedrockCohereEmbeddings) Format() llmprotocol.Format {
	return llmprotocol.FormatBedrockCohereEmbeddings
}

func (BedrockCohereEmbeddings) DecodeEmbeddingRequest(body json.RawMessage, policy llmprotocol.Policy) (EmbeddingRequestResult, error) {
	const format = llmprotocol.FormatBedrockCohereEmbeddings
	object, err := decodeObject(format, body)
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	request := llmprotocol.EmbeddingRequest{EncodingFormat: "float", Preservation: preserveBody(policy, format, body, true)}
	var texts []string
	if raw, ok := object["texts"]; ok {
		delete(object, "texts")
		if json.Unmarshal(raw, &texts) != nil || len(texts) == 0 {
			return EmbeddingRequestResult{}, translationError(format, "$.texts", "invalid_input", "Cohere texts must be a non-empty string array")
		}
	} else {
		return EmbeddingRequestResult{}, translationError(format, "$.texts", "missing_input", "Cohere text embedding input is required")
	}
	for _, text := range texts {
		request.Inputs = append(request.Inputs, llmprotocol.EmbeddingInput{Type: llmprotocol.EmbeddingInputText, Text: text})
	}
	request.InputType, err = optionalString(format, object, "input_type")
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	request.Truncate, err = optionalString(format, object, "truncate")
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	request.Dimensions, err = optionalInt(format, object, "output_dimension")
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	if rawTypes, ok := object["embedding_types"]; ok {
		delete(object, "embedding_types")
		var types []string
		if json.Unmarshal(rawTypes, &types) != nil || len(types) != 1 || types[0] != "float" {
			object["embedding_types"] = rawTypes
		}
	}
	if err := request.Validate(); err != nil {
		return EmbeddingRequestResult{}, translationError(format, "$.texts", "invalid_input", err.Error())
	}
	extensions, diagnostics, err := collectExtensions(format, object, policy)
	if err != nil {
		return EmbeddingRequestResult{}, err
	}
	request.Extensions = extensions
	return EmbeddingRequestResult{Request: request, Diagnostics: diagnostics}, nil
}

func (BedrockCohereEmbeddings) EncodeEmbeddingRequest(request llmprotocol.EmbeddingRequest, policy llmprotocol.Policy) (WireResult, error) {
	const format = llmprotocol.FormatBedrockCohereEmbeddings
	if raw, ok := preservedEmbeddingRequest(request, format, policy); ok {
		return WireResult{Body: raw}, nil
	}
	if err := request.Validate(); err != nil {
		return WireResult{}, translationError(format, "$.texts", "invalid_input", err.Error())
	}
	texts := make([]string, len(request.Inputs))
	for index, input := range request.Inputs {
		if input.Type != llmprotocol.EmbeddingInputText {
			return WireResult{}, translationError(format, "$.texts", "token_input_not_supported", "Cohere embeddings cannot represent pre-tokenized input")
		}
		texts[index] = input.Text
	}
	if request.EncodingFormat != "" && request.EncodingFormat != "float" {
		return WireResult{}, translationError(format, "$.encoding_format", "unsupported_encoding", "Cohere neutral translation supports float embeddings only")
	}
	if request.User != "" || request.TaskType != "" || request.Title != "" || request.Normalize != nil || request.AutoTruncate != nil {
		return WireResult{}, translationError(format, "$", "embedding_control_not_supported", "request contains controls Cohere embeddings cannot represent")
	}
	object := map[string]any{"texts": texts, "embedding_types": []string{"float"}}
	if request.InputType != "" {
		object["input_type"] = request.InputType
	}
	if request.Truncate != "" {
		object["truncate"] = strings.ToUpper(request.Truncate)
	}
	if request.Dimensions != nil {
		cohereV4 := strings.Contains(strings.ToLower(request.Model), "embed-v4")
		if request.Model != "" && !cohereV4 {
			return WireResult{}, translationError(format, "$.dimensions", "embedding_control_not_supported", "Cohere Embed v3 does not support output_dimension")
		}
		putInt(object, "output_dimension", request.Dimensions)
	}
	mergeExtensions(object, request.Extensions)
	body, err := marshalObject(format, object)
	return WireResult{Body: body}, err
}

func (BedrockCohereEmbeddings) DecodeEmbeddingResponse(body json.RawMessage, policy llmprotocol.Policy) (EmbeddingResponseResult, error) {
	const format = llmprotocol.FormatBedrockCohereEmbeddings
	object, err := decodeObject(format, body)
	if err != nil {
		return EmbeddingResponseResult{}, err
	}
	response := llmprotocol.EmbeddingResponse{EncodingFormat: "float", Preservation: preserveBody(policy, format, body, false)}
	rawEmbeddings, ok := object["embeddings"]
	if !ok {
		return EmbeddingResponseResult{}, translationError(format, "$.embeddings", "missing_embeddings", "Cohere response is missing embeddings")
	}
	delete(object, "embeddings")
	trimmed := bytes.TrimSpace(rawEmbeddings)
	if len(trimmed) != 0 && trimmed[0] == '{' {
		typed, objectErr := decodeObject(format, rawEmbeddings)
		if objectErr != nil {
			return EmbeddingResponseResult{}, objectErr
		}
		rawEmbeddings = typed["float"]
		delete(typed, "float")
		if len(typed) != 0 {
			encoded, _ := json.Marshal(typed)
			object["embeddings_by_type"] = encoded
		}
	}
	var embeddings [][]float64
	if json.Unmarshal(rawEmbeddings, &embeddings) != nil || len(embeddings) == 0 {
		return EmbeddingResponseResult{}, translationError(format, "$.embeddings", "invalid_embeddings", "Cohere float embeddings must be a non-empty matrix")
	}
	for index, values := range embeddings {
		tensor := llmprotocol.NewEmbeddingVector(values)
		if err := tensor.Validate(); err != nil {
			return EmbeddingResponseResult{}, translationError(format, "$.embeddings", "invalid_embeddings", err.Error())
		}
		response.Data = append(response.Data, llmprotocol.EmbeddingOutput{Index: index, Embedding: tensor})
	}
	delete(object, "response_type")
	delete(object, "texts")
	extensions, diagnostics, err := collectExtensions(format, object, policy)
	if err != nil {
		return EmbeddingResponseResult{}, err
	}
	response.Extensions = extensions
	return EmbeddingResponseResult{Response: response, Diagnostics: diagnostics}, nil
}

func (BedrockCohereEmbeddings) EncodeEmbeddingResponse(response llmprotocol.EmbeddingResponse, policy llmprotocol.Policy) (WireResult, error) {
	const format = llmprotocol.FormatBedrockCohereEmbeddings
	if raw, ok := preservedEmbeddingResponse(response, format, policy); ok {
		return WireResult{Body: raw}, nil
	}
	if err := response.Validate(); err != nil {
		return WireResult{}, translationError(format, "$.embeddings", "invalid_embeddings", err.Error())
	}
	embeddings := make([][]float64, len(response.Data))
	for index, output := range response.Data {
		if output.Embedding.Rank() != 1 {
			return WireResult{}, translationError(format, "$.embeddings[]", "rank_not_supported", "Cohere embeddings cannot represent a rank-2 tensor as one embedding")
		}
		embeddings[index] = output.Embedding.Values
	}
	object := map[string]any{"response_type": "embeddings_floats", "embeddings": embeddings}
	mergeExtensions(object, response.Extensions)
	body, err := marshalObject(format, object)
	return WireResult{Body: body}, err
}
