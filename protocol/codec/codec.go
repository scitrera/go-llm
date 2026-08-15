// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

// Package codec translates provider wire formats through llmprotocol's neutral
// representation. Codecs are stateless; StreamState belongs to one response.
package codec

import (
	"encoding/json"
	"fmt"
	"sync"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

type RequestResult struct {
	Request     llmprotocol.Request
	Diagnostics []llmprotocol.Diagnostic
}

type ResponseResult struct {
	Response    llmprotocol.Response
	Diagnostics []llmprotocol.Diagnostic
}

type WireResult struct {
	Body        json.RawMessage
	Diagnostics []llmprotocol.Diagnostic
}

type StreamState struct {
	Source     llmprotocol.Format
	Target     llmprotocol.Format
	ResponseID string
	Model      string
	Values     map[string]any
}

func (s *StreamState) valueMap() map[string]any {
	if s.Values == nil {
		s.Values = make(map[string]any)
	}
	return s.Values
}

type Codec interface {
	Format() llmprotocol.Format
	DecodeRequest(json.RawMessage, llmprotocol.Policy) (RequestResult, error)
	EncodeRequest(llmprotocol.Request, llmprotocol.Policy) (WireResult, error)
	DecodeResponse(json.RawMessage, llmprotocol.Policy) (ResponseResult, error)
	EncodeResponse(llmprotocol.Response, llmprotocol.Policy) (WireResult, error)
	DecodeStreamEvent(*StreamState, llmprotocol.WireEvent, llmprotocol.Policy) ([]llmprotocol.StreamEvent, []llmprotocol.Diagnostic, error)
	EncodeStreamEvent(*StreamState, llmprotocol.StreamEvent, llmprotocol.Policy) ([]llmprotocol.WireEvent, []llmprotocol.Diagnostic, error)
	FinishStream(*StreamState, llmprotocol.Policy) ([]llmprotocol.WireEvent, []llmprotocol.Diagnostic, error)
}

type Registry struct {
	mu     sync.RWMutex
	codecs map[llmprotocol.Format]Codec
}

func NewRegistry(codecs ...Codec) *Registry {
	r := &Registry{codecs: make(map[llmprotocol.Format]Codec, len(codecs))}
	for _, c := range codecs {
		r.Register(c)
	}
	return r
}

func NewDefaultRegistry() *Registry {
	return NewRegistry(
		OpenAIChat{}, OpenAIResponses{}, Anthropic{}, Gemini{}, Bedrock{},
	)
}

func (r *Registry) Register(c Codec) {
	if c == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.codecs[c.Format()] = c
}

func (r *Registry) Codec(format llmprotocol.Format) (Codec, error) {
	r.mu.RLock()
	c := r.codecs[format]
	r.mu.RUnlock()
	if c == nil {
		return nil, fmt.Errorf("llm protocol codec %q is not registered", format)
	}
	return c, nil
}

func (r *Registry) TranslateRequest(source, target llmprotocol.Format, body json.RawMessage, policy llmprotocol.Policy) (WireResult, error) {
	sourceCodec, err := r.Codec(source)
	if err != nil {
		return WireResult{}, err
	}
	targetCodec, err := r.Codec(target)
	if err != nil {
		return WireResult{}, err
	}
	decoded, err := sourceCodec.DecodeRequest(body, policy)
	if err != nil {
		return WireResult{}, err
	}
	if source != target {
		decoded.Request.ClearPreservation()
		var next []llmprotocol.Diagnostic
		decoded.Request, next, err = prepareCrossFormatRequest(decoded.Request, source, target, policy)
		decoded.Diagnostics = append(decoded.Diagnostics, next...)
		if err != nil {
			return WireResult{Diagnostics: attachFormats(decoded.Diagnostics, source, target)}, err
		}
	}
	encoded, err := targetCodec.EncodeRequest(decoded.Request, policy)
	if err != nil {
		return WireResult{}, err
	}
	encoded.Diagnostics = attachFormats(append(decoded.Diagnostics, encoded.Diagnostics...), source, target)
	return encoded, nil
}

func (r *Registry) TranslateResponse(source, target llmprotocol.Format, body json.RawMessage, policy llmprotocol.Policy) (WireResult, error) {
	sourceCodec, err := r.Codec(source)
	if err != nil {
		return WireResult{}, err
	}
	targetCodec, err := r.Codec(target)
	if err != nil {
		return WireResult{}, err
	}
	decoded, err := sourceCodec.DecodeResponse(body, policy)
	if err != nil {
		return WireResult{}, err
	}
	if source != target {
		decoded.Response.ClearPreservation()
		var next []llmprotocol.Diagnostic
		decoded.Response, next, err = prepareCrossFormatResponse(decoded.Response, source, target, policy)
		decoded.Diagnostics = append(decoded.Diagnostics, next...)
		if err != nil {
			return WireResult{Diagnostics: attachFormats(decoded.Diagnostics, source, target)}, err
		}
	}
	encoded, err := targetCodec.EncodeResponse(decoded.Response, policy)
	if err != nil {
		return WireResult{}, err
	}
	encoded.Diagnostics = attachFormats(append(decoded.Diagnostics, encoded.Diagnostics...), source, target)
	return encoded, nil
}

// Provider extension objects are intentionally format-local. Cross-format
// codecs may preserve typed neutral fields, but they must not accidentally
// inject arbitrary source-provider JSON into a different protocol. Paths are
// schema-bounded so diagnostics remain safe for metrics and logs.
func prepareCrossFormatRequest(request llmprotocol.Request, source, target llmprotocol.Format, policy llmprotocol.Policy) (llmprotocol.Request, []llmprotocol.Diagnostic, error) {
	var diagnostics []llmprotocol.Diagnostic
	drop := func(path string, extensions *llmprotocol.Extensions) error {
		if len(*extensions) == 0 {
			return nil
		}
		next, err := lossy(target, policy, path, "provider_extensions_not_portable")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return err
		}
		*extensions = nil
		return nil
	}
	if err := drop("$.extensions", &request.Extensions); err != nil {
		return request, diagnostics, err
	}
	if len(request.Reasoning.Raw) != 0 {
		owner := request.Reasoning.Provider
		if owner == "" {
			owner = source
		}
		if owner != target {
			next, err := lossy(target, policy, "$.reasoning.raw", "reasoning_config_not_portable")
			diagnostics = append(diagnostics, next...)
			if err != nil {
				return request, diagnostics, err
			}
			request.Reasoning.Raw = nil
			request.Reasoning.Provider = ""
		}
	}
	for index := range request.Instructions {
		if err := drop("$.instructions[].extensions", &request.Instructions[index].Extensions); err != nil {
			return request, diagnostics, err
		}
		for block := range request.Instructions[index].Content {
			if err := drop("$.instructions[].content[].extensions", &request.Instructions[index].Content[block].Extensions); err != nil {
				return request, diagnostics, err
			}
		}
		cleaned, next, err := sanitizeCrossFormatBlocks(request.Instructions[index].Content, source, target, policy, "$.instructions[].content[]")
		request.Instructions[index].Content = cleaned
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return request, diagnostics, err
		}
	}
	for index := range request.Messages {
		if err := drop("$.messages[].extensions", &request.Messages[index].Extensions); err != nil {
			return request, diagnostics, err
		}
		for block := range request.Messages[index].Content {
			if err := drop("$.messages[].content[].extensions", &request.Messages[index].Content[block].Extensions); err != nil {
				return request, diagnostics, err
			}
		}
		cleaned, next, err := sanitizeCrossFormatBlocks(request.Messages[index].Content, source, target, policy, "$.messages[].content[]")
		request.Messages[index].Content = cleaned
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return request, diagnostics, err
		}
	}
	for index := range request.Tools {
		if err := drop("$.tools[].extensions", &request.Tools[index].Extensions); err != nil {
			return request, diagnostics, err
		}
	}
	return request, diagnostics, nil
}

func prepareCrossFormatResponse(response llmprotocol.Response, source, target llmprotocol.Format, policy llmprotocol.Policy) (llmprotocol.Response, []llmprotocol.Diagnostic, error) {
	var diagnostics []llmprotocol.Diagnostic
	drop := func(path string, extensions *llmprotocol.Extensions) error {
		if len(*extensions) == 0 {
			return nil
		}
		next, err := lossy(target, policy, path, "provider_extensions_not_portable")
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return err
		}
		*extensions = nil
		return nil
	}
	if err := drop("$.extensions", &response.Extensions); err != nil {
		return response, diagnostics, err
	}
	for index := range response.Outputs {
		if err := drop("$.outputs[].extensions", &response.Outputs[index].Extensions); err != nil {
			return response, diagnostics, err
		}
		for block := range response.Outputs[index].Content {
			if err := drop("$.outputs[].content[].extensions", &response.Outputs[index].Content[block].Extensions); err != nil {
				return response, diagnostics, err
			}
		}
		cleaned, next, err := sanitizeCrossFormatBlocks(response.Outputs[index].Content, source, target, policy, "$.outputs[].content[]")
		response.Outputs[index].Content = cleaned
		diagnostics = append(diagnostics, next...)
		if err != nil {
			return response, diagnostics, err
		}
	}
	return response, diagnostics, nil
}

func sanitizeCrossFormatBlocks(blocks []llmprotocol.ContentBlock, source, target llmprotocol.Format, policy llmprotocol.Policy, path string) ([]llmprotocol.ContentBlock, []llmprotocol.Diagnostic, error) {
	result := make([]llmprotocol.ContentBlock, 0, len(blocks))
	var diagnostics []llmprotocol.Diagnostic
	for _, block := range blocks {
		owner := block.Provider
		if owner == "" && (block.Type == llmprotocol.ContentUnknown || block.Signature != "" || (block.Source != nil && block.Source.Kind == "provider_file")) {
			// Older or caller-authored neutral values may omit provenance. The
			// wire format being decoded is the only safe fallback; treating an
			// opaque value as portable could forge another provider's state.
			owner = source
		}
		providerOwned := owner != "" && owner != target
		if len(block.Extensions) != 0 {
			next, err := lossy(target, policy, path+".extensions", "provider_extensions_not_portable")
			diagnostics = append(diagnostics, next...)
			if err != nil {
				return nil, diagnostics, err
			}
			block.Extensions = nil
		}
		if providerOwned && block.Type == llmprotocol.ContentUnknown {
			next, err := lossy(target, policy, path, "provider_content_not_portable")
			diagnostics = append(diagnostics, next...)
			if err != nil {
				return nil, diagnostics, err
			}
			continue
		}
		if providerOwned && block.Source != nil && block.Source.Kind == "provider_file" {
			next, err := lossy(target, policy, path+".source", "provider_resource_not_portable")
			diagnostics = append(diagnostics, next...)
			if err != nil {
				return nil, diagnostics, err
			}
			continue
		}
		if providerOwned && block.Signature != "" {
			next, err := lossy(target, policy, path+".signature", "reasoning_signature_not_supported")
			diagnostics = append(diagnostics, next...)
			if err != nil {
				return nil, diagnostics, err
			}
			block.Signature = ""
		}
		if block.Result != nil {
			cleaned, next, err := sanitizeCrossFormatBlocks(block.Result.Content, source, target, policy, path+".tool_result.content[]")
			diagnostics = append(diagnostics, next...)
			if err != nil {
				return nil, diagnostics, err
			}
			block.Result.Content = cleaned
		}
		result = append(result, block)
	}
	return result, diagnostics, nil
}

func (r *Registry) TranslateStreamEvent(state *StreamState, source, target llmprotocol.Format, event llmprotocol.WireEvent, policy llmprotocol.Policy) ([]llmprotocol.WireEvent, []llmprotocol.Diagnostic, error) {
	sourceCodec, err := r.Codec(source)
	if err != nil {
		return nil, nil, err
	}
	targetCodec, err := r.Codec(target)
	if err != nil {
		return nil, nil, err
	}
	state.Source, state.Target = source, target
	decoded, diagnostics, err := sourceCodec.DecodeStreamEvent(state, event, policy)
	if err != nil {
		return nil, attachFormats(diagnostics, source, target), err
	}
	var output []llmprotocol.WireEvent
	for _, normalized := range decoded {
		if source != target {
			var nextDiagnostics []llmprotocol.Diagnostic
			normalized, nextDiagnostics, err = sanitizeCrossFormatStreamEvent(normalized, source, target, policy)
			diagnostics = append(diagnostics, nextDiagnostics...)
			if err != nil {
				return nil, attachFormats(diagnostics, source, target), err
			}
		}
		encoded, nextDiagnostics, encodeErr := targetCodec.EncodeStreamEvent(state, normalized, policy)
		diagnostics = append(diagnostics, nextDiagnostics...)
		if encodeErr != nil {
			return nil, attachFormats(diagnostics, source, target), encodeErr
		}
		output = append(output, encoded...)
	}
	return output, attachFormats(diagnostics, source, target), nil
}

func sanitizeCrossFormatStreamEvent(event llmprotocol.StreamEvent, source, target llmprotocol.Format, policy llmprotocol.Policy) (llmprotocol.StreamEvent, []llmprotocol.Diagnostic, error) {
	owner := event.Provider
	if owner == "" && (event.Type == llmprotocol.StreamUnknown || event.Signature != "") {
		owner = source
	}
	if owner == "" || owner == target {
		return event, nil, nil
	}
	if event.Type == llmprotocol.StreamUnknown || len(event.Raw) != 0 {
		diagnostics, err := lossy(target, policy, "$.stream", "provider_content_not_portable")
		if err != nil {
			return event, diagnostics, err
		}
		event.Raw = nil
		return llmprotocol.StreamEvent{}, diagnostics, nil
	}
	if event.Signature != "" {
		diagnostics, err := lossy(target, policy, "$.stream.signature", "reasoning_signature_not_supported")
		if err != nil {
			return event, diagnostics, err
		}
		event.Signature = ""
		return event, diagnostics, nil
	}
	return event, nil, nil
}

func (r *Registry) FinishStream(state *StreamState, target llmprotocol.Format, policy llmprotocol.Policy) ([]llmprotocol.WireEvent, []llmprotocol.Diagnostic, error) {
	c, err := r.Codec(target)
	if err != nil {
		return nil, nil, err
	}
	return c.FinishStream(state, policy)
}

func attachFormats(values []llmprotocol.Diagnostic, source, target llmprotocol.Format) []llmprotocol.Diagnostic {
	for i := range values {
		values[i].Source = source
		values[i].Target = target
	}
	return values
}
