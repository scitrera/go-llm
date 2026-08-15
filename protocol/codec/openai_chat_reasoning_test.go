// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package codec_test

import (
	"encoding/json"
	"errors"
	"testing"

	llmprotocol "github.com/scitrera/go-llm/protocol"
	"github.com/scitrera/go-llm/protocol/codec"
)

// decodeChatDelta decodes one OpenAI-chat stream chunk to canonical events.
func decodeChatDelta(t *testing.T, chunk string, policy llmprotocol.Policy) ([]llmprotocol.StreamEvent, []llmprotocol.Diagnostic, error) {
	t.Helper()
	return codec.OpenAIChat{}.DecodeStreamEvent(
		&codec.StreamState{},
		llmprotocol.WireEvent{Data: json.RawMessage(chunk)},
		policy,
	)
}

func reasoningDeltas(events []llmprotocol.StreamEvent) []string {
	var out []string
	for _, event := range events {
		if event.Type == llmprotocol.StreamReasoningDelta {
			out = append(out, event.Delta)
		}
	}
	return out
}

// The reasoning channel has no OpenAI-standard field name. Every vendor alias
// must decode to the same canonical StreamReasoningDelta — otherwise the trace
// is silently lost for that provider.
func TestOpenAIChatStreamDecodesReasoningAliases(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name  string
		chunk string
	}{
		{"reasoning_content", `{"id":"a1","choices":[{"delta":{"role":"assistant","reasoning_content":"weighing"}}]}`},
		{"reasoning", `{"id":"a1","choices":[{"delta":{"role":"assistant","reasoning":"weighing"}}]}`},
		{"thinking", `{"id":"a1","choices":[{"delta":{"role":"assistant","thinking":"weighing"}}]}`},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			events, diagnostics, err := decodeChatDelta(t, test.chunk, llmprotocol.StrictPolicy())
			if err != nil {
				t.Fatalf("DecodeStreamEvent() error = %v", err)
			}
			if got := reasoningDeltas(events); len(got) != 1 || got[0] != "weighing" {
				t.Fatalf("reasoning deltas = %v, want [weighing]", got)
			}
			// The alias is a mapped field, not an extension.
			for _, d := range diagnostics {
				if d.Kind == llmprotocol.DiagnosticUnknownField {
					t.Fatalf("alias reported as unknown field: %+v", d)
				}
			}
		})
	}
}

// A chunk carrying two aliases must yield ONE trace: first present wins and the
// rest are consumed, so the text is neither duplicated nor reported as unknown.
func TestOpenAIChatStreamReasoningAliasFirstPresentWins(t *testing.T) {
	t.Parallel()
	events, diagnostics, err := decodeChatDelta(t,
		`{"id":"a1","choices":[{"delta":{"reasoning_content":"canonical","reasoning":"alias"}}]}`,
		llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("DecodeStreamEvent() error = %v", err)
	}
	if got := reasoningDeltas(events); len(got) != 1 || got[0] != "canonical" {
		t.Fatalf("reasoning deltas = %v, want [canonical]", got)
	}
	for _, d := range diagnostics {
		if d.Kind == llmprotocol.DiagnosticUnknownField {
			t.Fatalf("consumed alias reported as unknown field: %+v", d)
		}
	}
}

// Regression: leftover delta fields used to be dropped without ever reaching
// collectExtensions — the delta was the one object in the codec that escaped
// UnknownFieldPolicy entirely (no diagnostic under drop, no error under
// reject). That silence is how an unmapped vendor alias disappears unnoticed.
func TestOpenAIChatStreamUnknownDeltaFieldObeysPolicy(t *testing.T) {
	t.Parallel()
	const chunk = `{"id":"a1","choices":[{"delta":{"role":"assistant","surprise_channel":"data"}}]}`

	dropPolicy := llmprotocol.StrictPolicy()
	dropPolicy.UnknownFields = llmprotocol.UnknownDrop
	_, diagnostics, err := decodeChatDelta(t, chunk, dropPolicy)
	if err != nil {
		t.Fatalf("DecodeStreamEvent() error = %v", err)
	}
	found := false
	for _, d := range diagnostics {
		if d.Kind == llmprotocol.DiagnosticUnknownField && d.Path == "$.'surprise_channel'" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected an unknown_field diagnostic for the delta, got %+v", diagnostics)
	}

	rejectPolicy := llmprotocol.StrictPolicy()
	rejectPolicy.UnknownFields = llmprotocol.UnknownReject
	if _, _, err := decodeChatDelta(t, chunk, rejectPolicy); err == nil {
		t.Fatal("expected UnknownReject to fail on an unmapped delta field")
	}
}

// Known delta fields sent as explicit nulls (OpenAI does this on tool-call and
// terminal chunks) are consumed, not treated as unknown — otherwise closing the
// policy hole above would reject streams that work today.
func TestOpenAIChatStreamNullDeltaFieldsAreNotUnknown(t *testing.T) {
	t.Parallel()
	rejectPolicy := llmprotocol.StrictPolicy()
	rejectPolicy.UnknownFields = llmprotocol.UnknownReject
	events, _, err := decodeChatDelta(t,
		`{"choices":[{"delta":{"role":null,"content":null,"reasoning_content":null,"refusal":null},"finish_reason":"stop"}]}`,
		rejectPolicy)
	if err != nil {
		t.Fatalf("DecodeStreamEvent() error = %v", err)
	}
	for _, event := range events {
		if event.Type == llmprotocol.StreamTextDelta || event.Type == llmprotocol.StreamReasoningDelta {
			t.Fatalf("null field produced a delta event: %+v", event)
		}
	}
}

// The non-stream decoder maps the same aliases to a leading reasoning block.
func TestOpenAIChatResponseDecodesReasoningAliases(t *testing.T) {
	t.Parallel()
	for _, name := range []string{"reasoning_content", "reasoning", "thinking"} {
		name := name
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			body := json.RawMessage(`{"id":"r1","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"answer","` +
				name + `":"weighing"},"finish_reason":"stop"}]}`)
			result, err := codec.OpenAIChat{}.DecodeResponse(body, llmprotocol.StrictPolicy())
			if err != nil {
				t.Fatalf("DecodeResponse() error = %v", err)
			}
			content := result.Response.Outputs[0].Content
			if len(content) != 2 ||
				content[0].Type != llmprotocol.ContentReasoning || content[0].Text != "weighing" ||
				content[1].Type != llmprotocol.ContentText || content[1].Text != "answer" {
				t.Fatalf("content = %#v, want [reasoning, text]", content)
			}
			// Mapped, so it must not linger as a preserved extension.
			if _, ok := result.Response.Outputs[0].Extensions[name]; ok {
				t.Fatalf("mapped alias %q leaked into extensions", name)
			}
		})
	}
}

// Guard the error path stays a translation error rather than a panic when an
// alias carries a non-string value.
func TestOpenAIChatStreamReasoningAliasRejectsNonString(t *testing.T) {
	t.Parallel()
	_, _, err := decodeChatDelta(t,
		`{"id":"a1","choices":[{"delta":{"reasoning":{"unexpected":"object"}}}]}`,
		llmprotocol.StrictPolicy())
	if err == nil {
		t.Fatal("expected a decode error for a non-string reasoning alias")
	}
	var translationErr *llmprotocol.TranslationError
	if !errors.As(err, &translationErr) {
		t.Fatalf("error = %v, want a TranslationError", err)
	}
}
