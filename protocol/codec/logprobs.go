// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package codec

import (
	"encoding/json"
	"fmt"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

func decodeOpenAIChatLogprobs(raw json.RawMessage) ([]llmprotocol.TokenLogprob, error) {
	const format = llmprotocol.FormatOpenAIChat
	object, err := decodeObject(format, raw)
	if err != nil {
		return nil, err
	}
	rawContent, ok := object["content"]
	if !ok {
		return nil, translationError(format, "$.choices[].logprobs.content", "missing_logprobs", "logprobs content is required")
	}
	delete(object, "content")
	if len(object) != 0 {
		return nil, translationError(format, "$.choices[].logprobs", "unsupported_logprobs", "logprobs contains unsupported fields")
	}
	values, err := decodeArray(format, "$.choices[].logprobs.content", rawContent)
	if err != nil {
		return nil, err
	}
	result := make([]llmprotocol.TokenLogprob, 0, len(values))
	for index, value := range values {
		item, objectErr := decodeObject(format, value)
		if objectErr != nil {
			return nil, objectErr
		}
		chosen, probabilityErr := decodeOpenAIProbability(item, fmt.Sprintf("$.choices[].logprobs.content[%d]", index))
		if probabilityErr != nil {
			return nil, probabilityErr
		}
		entry := llmprotocol.TokenLogprob{Chosen: chosen}
		if rawTop, ok := item["top_logprobs"]; ok {
			delete(item, "top_logprobs")
			top, arrayErr := decodeArray(format, "$.choices[].logprobs.content[].top_logprobs", rawTop)
			if arrayErr != nil {
				return nil, arrayErr
			}
			for _, rawCandidate := range top {
				candidate, candidateErr := decodeObject(format, rawCandidate)
				if candidateErr != nil {
					return nil, candidateErr
				}
				probability, probabilityErr := decodeOpenAIProbability(candidate, "$.choices[].logprobs.content[].top_logprobs[]")
				if probabilityErr != nil {
					return nil, probabilityErr
				}
				if len(candidate) != 0 {
					return nil, translationError(format, "$.choices[].logprobs.content[].top_logprobs[]", "unsupported_logprobs", "token probability contains unsupported fields")
				}
				entry.Top = append(entry.Top, probability)
			}
		}
		if len(item) != 0 {
			return nil, translationError(format, "$.choices[].logprobs.content[]", "unsupported_logprobs", "token logprob contains unsupported fields")
		}
		result = append(result, entry)
	}
	return result, nil
}

func decodeOpenAIProbability(object map[string]json.RawMessage, path string) (llmprotocol.TokenProbability, error) {
	const format = llmprotocol.FormatOpenAIChat
	value := llmprotocol.TokenProbability{}
	var err error
	value.Token, err = optionalString(format, object, "token")
	if err != nil {
		return value, err
	}
	logprob, err := optionalFloat(format, object, "logprob")
	if err != nil || logprob == nil {
		return value, translationError(format, path+".logprob", "invalid_logprob", "token logprob is required")
	}
	value.Logprob = *logprob
	if rawBytes, ok := object["bytes"]; ok {
		delete(object, "bytes")
		if string(rawBytes) != "null" && json.Unmarshal(rawBytes, &value.Bytes) != nil {
			return value, translationError(format, path+".bytes", "invalid_logprob", "token bytes must be an integer array or null")
		}
	}
	return value, nil
}

func encodeOpenAIChatLogprobs(values []llmprotocol.TokenLogprob) map[string]any {
	content := make([]any, 0, len(values))
	for _, value := range values {
		item := encodeOpenAIProbability(value.Chosen)
		top := make([]any, 0, len(value.Top))
		for _, candidate := range value.Top {
			top = append(top, encodeOpenAIProbability(candidate))
		}
		item["top_logprobs"] = top
		content = append(content, item)
	}
	return map[string]any{"content": content}
}

func encodeOpenAIProbability(value llmprotocol.TokenProbability) map[string]any {
	result := map[string]any{"token": value.Token, "logprob": value.Logprob}
	if value.Bytes != nil {
		result["bytes"] = value.Bytes
	} else {
		result["bytes"] = nil
	}
	return result
}

func decodeGeminiLogprobs(raw json.RawMessage) ([]llmprotocol.TokenLogprob, error) {
	const format = llmprotocol.FormatGemini
	object, err := decodeObject(format, raw)
	if err != nil {
		return nil, err
	}
	chosenRaw, ok := object["chosenCandidates"]
	if !ok {
		return nil, translationError(format, "$.candidates[].logprobsResult.chosenCandidates", "missing_logprobs", "chosen candidates are required")
	}
	delete(object, "chosenCandidates")
	chosen, err := decodeArray(format, "$.candidates[].logprobsResult.chosenCandidates", chosenRaw)
	if err != nil {
		return nil, err
	}
	var topGroups []json.RawMessage
	if topRaw, exists := object["topCandidates"]; exists {
		delete(object, "topCandidates")
		topGroups, err = decodeArray(format, "$.candidates[].logprobsResult.topCandidates", topRaw)
		if err != nil {
			return nil, err
		}
	}
	if len(object) != 0 {
		return nil, translationError(format, "$.candidates[].logprobsResult", "unsupported_logprobs", "logprobs result contains unsupported fields")
	}
	result := make([]llmprotocol.TokenLogprob, 0, len(chosen))
	for index, rawChosen := range chosen {
		value, valueErr := decodeGeminiProbability(rawChosen)
		if valueErr != nil {
			return nil, valueErr
		}
		entry := llmprotocol.TokenLogprob{Chosen: value}
		if index < len(topGroups) {
			group, groupErr := decodeObject(format, topGroups[index])
			if groupErr != nil {
				return nil, groupErr
			}
			rawCandidates, exists := group["candidates"]
			if !exists {
				return nil, translationError(format, "$.candidates[].logprobsResult.topCandidates[].candidates", "missing_logprobs", "top candidates are required")
			}
			delete(group, "candidates")
			candidates, arrayErr := decodeArray(format, "$.candidates[].logprobsResult.topCandidates[].candidates", rawCandidates)
			if arrayErr != nil {
				return nil, arrayErr
			}
			for _, candidate := range candidates {
				probability, probabilityErr := decodeGeminiProbability(candidate)
				if probabilityErr != nil {
					return nil, probabilityErr
				}
				entry.Top = append(entry.Top, probability)
			}
			if len(group) != 0 {
				return nil, translationError(format, "$.candidates[].logprobsResult.topCandidates[]", "unsupported_logprobs", "top candidate group contains unsupported fields")
			}
		}
		result = append(result, entry)
	}
	return result, nil
}

func decodeGeminiProbability(raw json.RawMessage) (llmprotocol.TokenProbability, error) {
	const format = llmprotocol.FormatGemini
	object, err := decodeObject(format, raw)
	if err != nil {
		return llmprotocol.TokenProbability{}, err
	}
	value := llmprotocol.TokenProbability{}
	value.Token, err = optionalString(format, object, "token")
	if err == nil {
		value.TokenID, err = optionalInt(format, object, "tokenId")
	}
	logprob, logprobErr := optionalFloat(format, object, "logProbability")
	if err == nil {
		err = logprobErr
	}
	if err != nil || logprob == nil {
		return value, translationError(format, "$.candidates[].logprobsResult.logProbability", "invalid_logprob", "token log probability is required")
	}
	value.Logprob = *logprob
	if len(object) != 0 {
		return value, translationError(format, "$.candidates[].logprobsResult", "unsupported_logprobs", "token probability contains unsupported fields")
	}
	return value, nil
}

func encodeGeminiLogprobs(values []llmprotocol.TokenLogprob) map[string]any {
	chosen := make([]any, 0, len(values))
	top := make([]any, 0, len(values))
	for _, value := range values {
		chosen = append(chosen, encodeGeminiProbability(value.Chosen))
		candidates := make([]any, 0, len(value.Top))
		for _, candidate := range value.Top {
			candidates = append(candidates, encodeGeminiProbability(candidate))
		}
		top = append(top, map[string]any{"candidates": candidates})
	}
	return map[string]any{"chosenCandidates": chosen, "topCandidates": top}
}

func encodeGeminiProbability(value llmprotocol.TokenProbability) map[string]any {
	result := map[string]any{"token": value.Token, "logProbability": value.Logprob}
	putInt(result, "tokenId", value.TokenID)
	return result
}
