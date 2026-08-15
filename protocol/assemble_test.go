// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package llmprotocol

import "testing"

func TestAssemblerPreservesReasoningTextAndToolOrdering(t *testing.T) {
	t.Parallel()
	var assembler Assembler
	assembler.Apply(StreamEvent{Type: StreamResponseStart, ResponseID: "resp_1", Model: "served"})
	assembler.Apply(StreamEvent{Type: StreamOutputStart, Role: RoleAssistant})
	assembler.Apply(StreamEvent{Type: StreamReasoningDelta, Delta: "think"})
	assembler.Apply(StreamEvent{Type: StreamTextDelta, Delta: "answer"})
	assembler.Apply(StreamEvent{Type: StreamToolCallStart, ContentIndex: 1, ToolCallID: "call_1", ToolName: "lookup"})
	assembler.Apply(StreamEvent{Type: StreamToolArgsDelta, ContentIndex: 1, Delta: `{"q":"x"}`})
	assembler.Apply(StreamEvent{Type: StreamOutputDone, StopReason: StopToolUse})
	assembler.Apply(StreamEvent{Type: StreamResponseDone})
	response := assembler.Response()
	if !assembler.Done() || response.ID != "resp_1" || len(response.Outputs) != 1 {
		t.Fatalf("response = %#v", response)
	}
	content := response.Outputs[0].Content
	if len(content) != 3 || content[0].Type != ContentReasoning || content[0].Text != "think" || content[1].Type != ContentText || content[1].Text != "answer" || content[2].ToolCall == nil || content[2].ToolCall.Name != "lookup" {
		t.Fatalf("content = %#v", content)
	}
}
