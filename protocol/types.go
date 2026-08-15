// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

// Package llmprotocol defines provider-neutral LLM request, response, and
// streaming contracts. Wire-protocol codecs live in subpackages and depend only
// on these types.
package llmprotocol

import "encoding/json"

// Format identifies one provider wire contract rather than a vendor account.
type Format string

const (
	FormatOpenAIChat              Format = "openai_chat_completions"
	FormatOpenAIResponses         Format = "openai_responses"
	FormatAnthropic               Format = "anthropic_messages"
	FormatGemini                  Format = "gemini_generate_content"
	FormatBedrock                 Format = "bedrock_converse"
	FormatOpenAIEmbeddings        Format = "openai_embeddings"
	FormatGeminiEmbeddings        Format = "gemini_embed_content"
	FormatBedrockTitanEmbeddings  Format = "bedrock_titan_embeddings"
	FormatBedrockCohereEmbeddings Format = "bedrock_cohere_embeddings"
)

type Role string

const (
	RoleSystem    Role = "system"
	RoleDeveloper Role = "developer"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type ContentType string

const (
	ContentText       ContentType = "text"
	ContentReasoning  ContentType = "reasoning"
	ContentImage      ContentType = "image"
	ContentAudio      ContentType = "audio"
	ContentVideo      ContentType = "video"
	ContentFile       ContentType = "file"
	ContentToolCall   ContentType = "tool_call"
	ContentToolResult ContentType = "tool_result"
	ContentRefusal    ContentType = "refusal"
	ContentUnknown    ContentType = "unknown"
)

// Source carries one external, inline, or provider-owned media value. Data is
// base64 text when Kind is "base64"; provider_file identifies an opaque
// provider resource and must retain target affinity at the host layer.
type Source struct {
	Kind      string          `json:"kind"`
	URL       string          `json:"url,omitempty"`
	Data      string          `json:"data,omitempty"`
	MediaType string          `json:"media_type,omitempty"`
	FileID    string          `json:"file_id,omitempty"`
	Filename  string          `json:"filename,omitempty"`
	Detail    string          `json:"detail,omitempty"`
	Raw       json.RawMessage `json:"raw,omitempty"`
}

type ToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type ToolResult struct {
	ToolCallID string         `json:"tool_call_id"`
	Content    []ContentBlock `json:"content"`
	IsError    *bool          `json:"is_error,omitempty"`
}

// ContentBlock is an ordered tagged union. Exactly the fields appropriate for
// Type should be populated. Raw is retained only for ContentUnknown.
type ContentBlock struct {
	Type       ContentType     `json:"type"`
	Text       string          `json:"text,omitempty"`
	Signature  string          `json:"signature,omitempty"`
	Source     *Source         `json:"source,omitempty"`
	ToolCall   *ToolCall       `json:"tool_call,omitempty"`
	Result     *ToolResult     `json:"tool_result,omitempty"`
	Provider   Format          `json:"provider,omitempty"`
	Raw        json.RawMessage `json:"raw,omitempty"`
	Extensions Extensions      `json:"extensions,omitempty"`
}

func Text(text string) ContentBlock {
	return ContentBlock{Type: ContentText, Text: text}
}

func Refusal(text string) ContentBlock {
	return ContentBlock{Type: ContentRefusal, Text: text}
}

type Instruction struct {
	Role       Role           `json:"role"`
	Content    []ContentBlock `json:"content"`
	Extensions Extensions     `json:"extensions,omitempty"`
}

type Message struct {
	ID         string         `json:"id,omitempty"`
	Role       Role           `json:"role"`
	Content    []ContentBlock `json:"content"`
	Extensions Extensions     `json:"extensions,omitempty"`
}

type ToolDefinition struct {
	Type        string          `json:"type,omitempty"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      *bool           `json:"strict,omitempty"`
	Extensions  Extensions      `json:"extensions,omitempty"`
	Raw         json.RawMessage `json:"raw,omitempty"`
}

type ToolChoiceType string

const (
	ToolChoiceAuto     ToolChoiceType = "auto"
	ToolChoiceRequired ToolChoiceType = "required"
	ToolChoiceNone     ToolChoiceType = "none"
	ToolChoiceTool     ToolChoiceType = "tool"
	ToolChoiceRaw      ToolChoiceType = "raw"
)

type ToolChoice struct {
	Type ToolChoiceType  `json:"type"`
	Name string          `json:"name,omitempty"`
	Raw  json.RawMessage `json:"raw,omitempty"`
}

type Sampling struct {
	Temperature      *float64 `json:"temperature,omitempty"`
	TopP             *float64 `json:"top_p,omitempty"`
	TopK             *int64   `json:"top_k,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`
	Seed             *int64   `json:"seed,omitempty"`
}

type Output struct {
	MaxTokens      *int64          `json:"max_tokens,omitempty"`
	StopSequences  []string        `json:"stop_sequences,omitempty"`
	ResponseFormat json.RawMessage `json:"response_format,omitempty"`
	Choices        *int64          `json:"choices,omitempty"`
	Logprobs       *bool           `json:"logprobs,omitempty"`
	TopLogprobs    *int64          `json:"top_logprobs,omitempty"`
	Modalities     []string        `json:"modalities,omitempty"`
}

type Reasoning struct {
	Effort       string          `json:"effort,omitempty"`
	BudgetTokens *int64          `json:"budget_tokens,omitempty"`
	Include      *bool           `json:"include,omitempty"`
	Level        string          `json:"level,omitempty"`
	Provider     Format          `json:"provider,omitempty"`
	Raw          json.RawMessage `json:"raw,omitempty"`
}

// ProviderState represents request state controls shared by some, but not all,
// wire formats. A translator must never silently discard active state. False
// boolean values are intentionally retained so same-format decode/encode can
// distinguish an explicit opt-out from an omitted field.
type ProviderState struct {
	Store              *bool           `json:"store,omitempty"`
	Background         *bool           `json:"background,omitempty"`
	PreviousResponseID string          `json:"previous_response_id,omitempty"`
	Conversation       json.RawMessage `json:"conversation,omitempty"`
}

// Extensions retains top-level provider fields that have no neutral meaning.
// Values are exact JSON fragments. Hosts should use an allowlist before sending
// extensions across trust boundaries.
type Extensions map[string]json.RawMessage

// Preservation stores exact same-format bodies in memory. Code that mutates a
// decoded value must call ClearPreservation before encoding it.
type Preservation struct {
	Requests  map[Format]json.RawMessage `json:"requests,omitempty"`
	Responses map[Format]json.RawMessage `json:"responses,omitempty"`
}

type Request struct {
	Model             string           `json:"model,omitempty"`
	Instructions      []Instruction    `json:"instructions,omitempty"`
	Messages          []Message        `json:"messages,omitempty"`
	Tools             []ToolDefinition `json:"tools,omitempty"`
	ToolChoice        *ToolChoice      `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool            `json:"parallel_tool_calls,omitempty"`
	Sampling          Sampling         `json:"sampling,omitzero"`
	Output            Output           `json:"output,omitzero"`
	Reasoning         Reasoning        `json:"reasoning,omitzero"`
	State             ProviderState    `json:"state,omitzero"`
	Stream            bool             `json:"stream,omitempty"`
	Extensions        Extensions       `json:"extensions,omitempty"`
	Preservation      Preservation     `json:"preservation,omitzero"`
}

func (r *Request) ClearPreservation() {
	r.Preservation = Preservation{}
}

type Usage struct {
	// InputTokens is total input, including cached-input and cache-creation
	// components. Provider codecs normalize wire formats which report those
	// components separately.
	InputTokens              *int64           `json:"input_tokens,omitempty"`
	OutputTokens             *int64           `json:"output_tokens,omitempty"`
	TotalTokens              *int64           `json:"total_tokens,omitempty"`
	CachedInputTokens        *int64           `json:"cached_input_tokens,omitempty"`
	CacheCreationTokens      *int64           `json:"cache_creation_tokens,omitempty"`
	ReasoningTokens          *int64           `json:"reasoning_tokens,omitempty"`
	ToolUsePromptTokens      *int64           `json:"tool_use_prompt_tokens,omitempty"`
	AcceptedPredictionTokens *int64           `json:"accepted_prediction_tokens,omitempty"`
	RejectedPredictionTokens *int64           `json:"rejected_prediction_tokens,omitempty"`
	ProviderComponents       map[string]int64 `json:"provider_components,omitempty"`
}

type StopReason string

const (
	StopEndTurn       StopReason = "end_turn"
	StopMaxTokens     StopReason = "max_tokens"
	StopToolUse       StopReason = "tool_use"
	StopContentFilter StopReason = "content_filter"
	StopError         StopReason = "error"
	StopUnknown       StopReason = "unknown"
)

type TokenProbability struct {
	Token   string  `json:"token"`
	TokenID *int64  `json:"token_id,omitempty"`
	Logprob float64 `json:"logprob"`
	Bytes   []int64 `json:"bytes,omitempty"`
}

type TokenLogprob struct {
	Chosen TokenProbability   `json:"chosen"`
	Top    []TokenProbability `json:"top,omitempty"`
}

type ResponseOutput struct {
	ID         string         `json:"id,omitempty"`
	Index      *int           `json:"index,omitempty"`
	Role       Role           `json:"role"`
	Content    []ContentBlock `json:"content"`
	StopReason StopReason     `json:"stop_reason,omitempty"`
	Logprobs   []TokenLogprob `json:"logprobs,omitempty"`
	Extensions Extensions     `json:"extensions,omitempty"`
}

type Response struct {
	ID           string           `json:"id,omitempty"`
	Model        string           `json:"model,omitempty"`
	Outputs      []ResponseOutput `json:"outputs,omitempty"`
	Usage        Usage            `json:"usage,omitzero"`
	Extensions   Extensions       `json:"extensions,omitempty"`
	Preservation Preservation     `json:"preservation,omitzero"`
}

func (r *Response) ClearPreservation() {
	r.Preservation = Preservation{}
}

type StreamEventType string

const (
	StreamResponseStart           StreamEventType = "response_start"
	StreamOutputStart             StreamEventType = "output_start"
	StreamTextDelta               StreamEventType = "text_delta"
	StreamReasoningDelta          StreamEventType = "reasoning_delta"
	StreamReasoningSignatureDelta StreamEventType = "reasoning_signature_delta"
	StreamRefusalDelta            StreamEventType = "refusal_delta"
	StreamToolCallStart           StreamEventType = "tool_call_start"
	StreamToolArgsDelta           StreamEventType = "tool_arguments_delta"
	StreamOutputDone              StreamEventType = "output_done"
	StreamUsage                   StreamEventType = "usage"
	StreamLogprobs                StreamEventType = "logprobs"
	StreamResponseDone            StreamEventType = "response_done"
	StreamError                   StreamEventType = "error"
	StreamUnknown                 StreamEventType = "unknown"
)

// StreamEvent is sufficiently granular to retain OpenAI Responses item/event
// ordering while still representing Chat and Anthropic deltas.
type StreamEvent struct {
	Type         StreamEventType `json:"type"`
	ResponseID   string          `json:"response_id,omitempty"`
	Model        string          `json:"model,omitempty"`
	OutputIndex  int             `json:"output_index,omitempty"`
	ContentIndex int             `json:"content_index,omitempty"`
	ItemID       string          `json:"item_id,omitempty"`
	Role         Role            `json:"role,omitempty"`
	Delta        string          `json:"delta,omitempty"`
	Signature    string          `json:"signature,omitempty"`
	ToolCallID   string          `json:"tool_call_id,omitempty"`
	ToolName     string          `json:"tool_name,omitempty"`
	Usage        *Usage          `json:"usage,omitempty"`
	StopReason   StopReason      `json:"stop_reason,omitempty"`
	Logprobs     []TokenLogprob  `json:"logprobs,omitempty"`
	Error        *ProtocolError  `json:"error,omitempty"`
	Provider     Format          `json:"provider,omitempty"`
	Raw          json.RawMessage `json:"raw,omitempty"`
}

type ProtocolError struct {
	Type      string `json:"type,omitempty"`
	Code      string `json:"code,omitempty"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable,omitempty"`
}

// WireEvent is one decoded SSE event. Event is empty for protocols that use
// only data frames. Data excludes the SSE "data:" prefix.
type WireEvent struct {
	Event string
	Data  json.RawMessage
}
