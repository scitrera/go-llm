// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package llmprotocol

import (
	"encoding/json"
	"sort"
	"strings"
)

// Assembler folds normalized stream events into a buffered Response. It keeps
// output and content ordering stable and is safe to use for Chat, Responses, or
// Anthropic streams. One assembler is used by one goroutine for one response.
type Assembler struct {
	response Response
	outputs  map[int]*assembledOutput
	done     bool
}

type assembledOutput struct {
	id         string
	role       Role
	stopReason StopReason
	logprobs   []TokenLogprob
	blocks     map[assembledBlockKey]*assembledBlock
	nextOrder  int
}

type assembledBlockKey struct {
	content  int
	typeName ContentType
}

type assembledBlock struct {
	order     int
	block     ContentBlock
	text      strings.Builder
	arguments strings.Builder
	signature strings.Builder
}

func (a *Assembler) Apply(event StreamEvent) {
	if event.ResponseID != "" {
		a.response.ID = event.ResponseID
	}
	if event.Model != "" {
		a.response.Model = event.Model
	}
	switch event.Type {
	case StreamResponseStart:
		return
	case StreamOutputStart:
		output := a.output(event.OutputIndex)
		if event.ItemID != "" {
			output.id = event.ItemID
		}
		if event.Role != "" {
			output.role = event.Role
		}
	case StreamTextDelta:
		block := a.block(event.OutputIndex, event.ContentIndex, ContentText)
		block.text.WriteString(event.Delta)
	case StreamReasoningDelta:
		block := a.block(event.OutputIndex, event.ContentIndex, ContentReasoning)
		block.text.WriteString(event.Delta)
	case StreamReasoningSignatureDelta:
		block := a.block(event.OutputIndex, event.ContentIndex, ContentReasoning)
		block.signature.WriteString(event.Signature)
	case StreamRefusalDelta:
		block := a.block(event.OutputIndex, event.ContentIndex, ContentRefusal)
		block.text.WriteString(event.Delta)
	case StreamToolCallStart:
		block := a.block(event.OutputIndex, event.ContentIndex, ContentToolCall)
		if block.block.ToolCall == nil {
			block.block.ToolCall = &ToolCall{}
		}
		if event.ToolCallID != "" {
			block.block.ToolCall.ID = event.ToolCallID
		}
		if event.ToolName != "" {
			block.block.ToolCall.Name = event.ToolName
		}
	case StreamToolArgsDelta:
		block := a.block(event.OutputIndex, event.ContentIndex, ContentToolCall)
		if block.block.ToolCall == nil {
			block.block.ToolCall = &ToolCall{ID: event.ToolCallID}
		}
		block.arguments.WriteString(event.Delta)
	case StreamOutputDone:
		output := a.output(event.OutputIndex)
		if event.StopReason != "" {
			output.stopReason = event.StopReason
		}
	case StreamUsage:
		if event.Usage != nil {
			mergeUsage(&a.response.Usage, *event.Usage)
		}
	case StreamLogprobs:
		output := a.output(event.OutputIndex)
		output.logprobs = append(output.logprobs, event.Logprobs...)
	case StreamResponseDone:
		a.done = true
	}
}

func (a *Assembler) Done() bool { return a.done }

func (a *Assembler) Response() Response {
	result := a.response
	indexes := make([]int, 0, len(a.outputs))
	for index := range a.outputs {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	result.Outputs = make([]ResponseOutput, 0, len(indexes))
	for _, index := range indexes {
		source := a.outputs[index]
		output := ResponseOutput{ID: source.id, Role: source.role, StopReason: source.stopReason, Logprobs: append([]TokenLogprob(nil), source.logprobs...)}
		if output.Role == "" {
			output.Role = RoleAssistant
		}
		blocks := make([]*assembledBlock, 0, len(source.blocks))
		for _, block := range source.blocks {
			blocks = append(blocks, block)
		}
		sort.SliceStable(blocks, func(i, j int) bool { return blocks[i].order < blocks[j].order })
		for _, assembled := range blocks {
			block := assembled.block
			switch block.Type {
			case ContentText, ContentReasoning, ContentRefusal:
				block.Text = assembled.text.String()
				if block.Type == ContentReasoning {
					block.Signature = assembled.signature.String()
				}
			case ContentToolCall:
				if block.ToolCall == nil {
					block.ToolCall = &ToolCall{}
				}
				arguments := assembled.arguments.String()
				if arguments == "" {
					arguments = "{}"
				}
				block.ToolCall.Arguments = json.RawMessage(arguments)
			}
			output.Content = append(output.Content, block)
		}
		result.Outputs = append(result.Outputs, output)
	}
	return result
}

func (a *Assembler) output(index int) *assembledOutput {
	if a.outputs == nil {
		a.outputs = make(map[int]*assembledOutput)
	}
	output := a.outputs[index]
	if output == nil {
		output = &assembledOutput{role: RoleAssistant, blocks: make(map[assembledBlockKey]*assembledBlock)}
		a.outputs[index] = output
	}
	return output
}

func (a *Assembler) block(outputIndex, contentIndex int, typeName ContentType) *assembledBlock {
	output := a.output(outputIndex)
	key := assembledBlockKey{content: contentIndex, typeName: typeName}
	block := output.blocks[key]
	if block == nil {
		block = &assembledBlock{order: output.nextOrder, block: ContentBlock{Type: typeName}}
		output.nextOrder++
		output.blocks[key] = block
	}
	return block
}

func mergeUsage(target *Usage, source Usage) {
	if source.InputTokens != nil {
		target.InputTokens = source.InputTokens
	}
	if source.OutputTokens != nil {
		target.OutputTokens = source.OutputTokens
	}
	if source.TotalTokens != nil {
		target.TotalTokens = source.TotalTokens
	}
	if source.CachedInputTokens != nil {
		target.CachedInputTokens = source.CachedInputTokens
	}
	if source.CacheCreationTokens != nil {
		target.CacheCreationTokens = source.CacheCreationTokens
	}
	if source.ReasoningTokens != nil {
		target.ReasoningTokens = source.ReasoningTokens
	}
	if source.ToolUsePromptTokens != nil {
		target.ToolUsePromptTokens = source.ToolUsePromptTokens
	}
	if source.AcceptedPredictionTokens != nil {
		target.AcceptedPredictionTokens = source.AcceptedPredictionTokens
	}
	if source.RejectedPredictionTokens != nil {
		target.RejectedPredictionTokens = source.RejectedPredictionTokens
	}
	if source.ProviderComponents != nil {
		if target.ProviderComponents == nil {
			target.ProviderComponents = map[string]int64{}
		}
		for key, value := range source.ProviderComponents {
			target.ProviderComponents[key] = value
		}
	}
}
