// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package llmclient

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	llmprotocol "github.com/scitrera/go-llm/protocol"
	"github.com/scitrera/go-llm/protocol/codec"
)

type streamItem struct {
	event llmprotocol.StreamEvent
	err   error
}

type Stream struct {
	events      <-chan streamItem
	cancel      context.CancelFunc
	closeOnce   sync.Once
	done        atomic.Bool
	assembler   llmprotocol.Assembler
	metadata    CallMetadata
	metaMu      sync.RWMutex
	terminalErr error
	terminalMu  sync.Mutex
}

func (c *Client) Stream(ctx context.Context, backend Backend, request llmprotocol.Request) (*Stream, error) {
	started := time.Now()
	metadata := CallMetadata{BackendName: backend.Name, Format: backend.Format, StartedAt: started, Attempts: 1}
	effectiveModel := request.Model
	if backend.Model != "" {
		effectiveModel = backend.Model
	}
	endpoint, err := backend.endpoint(effectiveModel, true)
	if err != nil {
		return nil, &Error{Kind: ErrorInvalidConfig, Message: err.Error(), Cause: err}
	}
	metadata.Endpoint = endpoint.Redacted()
	formatCodec, err := c.registry.Codec(backend.Format)
	if err != nil {
		return nil, &Error{Kind: ErrorInvalidConfig, Message: err.Error(), Cause: err}
	}
	request.Stream = true
	if backend.Format == llmprotocol.FormatOpenAIChat && c.includeStreamUsage {
		extensions := make(llmprotocol.Extensions, len(request.Extensions)+1)
		for key, value := range request.Extensions {
			extensions[key] = append(json.RawMessage(nil), value...)
		}
		if _, exists := extensions["stream_options"]; !exists {
			extensions["stream_options"] = json.RawMessage(`{"include_usage":true}`)
		}
		request.Extensions = extensions
	}
	if backend.Model != "" && backend.Model != request.Model {
		request.Model = backend.Model
		request.ClearPreservation()
	}
	encoded, err := formatCodec.EncodeRequest(request, c.policy)
	if err != nil {
		return nil, &Error{Kind: ErrorEncode, Message: err.Error(), Cause: err}
	}
	metadata.RequestDiagnostics = encoded.Diagnostics
	body, err := MergeBodyDefaults(encoded.Body, backend.BodyDefaults)
	if err != nil {
		return nil, &Error{Kind: ErrorEncode, Message: err.Error(), Cause: err}
	}
	streamContext, cancel := context.WithCancel(ctx)
	response, err := c.do(streamContext, backend, endpoint.String(), body, true)
	if err != nil {
		cancel()
		return nil, err
	}
	metadata.StatusCode = response.StatusCode
	metadata.ResponseHeaders = response.Header.Clone()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		defer func() { _ = response.Body.Close() }()
		cancel()
		responseBody, readErr := readBounded(response.Body, c.maxResponseBytes)
		if readErr != nil {
			return nil, readErr
		}
		return nil, decodeHTTPError(response.StatusCode, responseBody)
	}
	channel := make(chan streamItem, 1)
	stream := &Stream{events: channel, cancel: cancel, metadata: metadata}
	go c.runStream(streamContext, cancel, response, formatCodec, channel, stream)
	return stream, nil
}

func (s *Stream) Next(ctx context.Context) (llmprotocol.StreamEvent, error) {
	select {
	case item, open := <-s.events:
		if !open {
			s.terminalMu.Lock()
			err := s.terminalErr
			s.terminalMu.Unlock()
			if err != nil {
				return llmprotocol.StreamEvent{}, err
			}
			return llmprotocol.StreamEvent{}, io.EOF
		}
		if item.err != nil {
			return llmprotocol.StreamEvent{}, item.err
		}
		s.assembler.Apply(item.event)
		return item.event, nil
	case <-ctx.Done():
		_ = s.Close()
		return llmprotocol.StreamEvent{}, classifyContextError(ctx.Err())
	}
}

func (s *Stream) Collect(ctx context.Context) (llmprotocol.Response, error) {
	for {
		event, err := s.Next(ctx)
		if errors.Is(err, io.EOF) {
			return s.assembler.Response(), nil
		}
		if err != nil {
			return s.assembler.Response(), err
		}
		if event.Type == llmprotocol.StreamError {
			return s.assembler.Response(), protocolStreamError(event.Error)
		}
	}
}

func (s *Stream) Response() llmprotocol.Response { return s.assembler.Response() }

func (s *Stream) Metadata() CallMetadata {
	s.metaMu.RLock()
	defer s.metaMu.RUnlock()
	metadata := s.metadata
	metadata.ResponseHeaders = metadata.ResponseHeaders.Clone()
	metadata.RequestDiagnostics = append([]llmprotocol.Diagnostic(nil), metadata.RequestDiagnostics...)
	metadata.ResponseDiagnostics = append([]llmprotocol.Diagnostic(nil), metadata.ResponseDiagnostics...)
	return metadata
}

func (s *Stream) Close() error {
	s.closeOnce.Do(func() {
		s.done.Store(true)
		s.cancel()
	})
	return nil
}

func (c *Client) runStream(ctx context.Context, cancel context.CancelFunc, response *http.Response, formatCodec codec.Codec, output chan<- streamItem, stream *Stream) {
	defer close(output)
	defer func() { _ = response.Body.Close() }()
	defer cancel()
	defer func() {
		stream.done.Store(true)
		stream.metaMu.Lock()
		stream.metadata.Duration = time.Since(stream.metadata.StartedAt)
		stream.metaMu.Unlock()
	}()
	mediaType, _, _ := mime.ParseMediaType(response.Header.Get("Content-Type"))
	if formatCodec.Format() == llmprotocol.FormatBedrock && strings.EqualFold(mediaType, "application/vnd.amazon.eventstream") {
		c.runAWSEventStream(ctx, cancel, response.Body, formatCodec, output, stream)
		return
	}
	if !strings.EqualFold(mediaType, "text/event-stream") {
		body, err := readBounded(response.Body, c.maxResponseBytes)
		if err != nil {
			sendStreamItem(ctx, output, streamItem{err: err})
			return
		}
		decoded, err := formatCodec.DecodeResponse(body, c.policy)
		if err != nil {
			sendStreamItem(ctx, output, streamItem{err: &Error{Kind: ErrorDecode, Message: err.Error(), Cause: err}})
			return
		}
		stream.metaMu.Lock()
		stream.metadata.ResponseDiagnostics = decoded.Diagnostics
		stream.metaMu.Unlock()
		for _, event := range responseToEvents(decoded.Response) {
			if !sendStreamItem(ctx, output, streamItem{event: event}) {
				return
			}
		}
		return
	}

	var timeoutReason atomic.Value
	timeoutReason.Store(error(ErrFirstEventTimeout))
	timer := newStreamTimer(c.firstEventTimeout, cancel)
	if timer == nil && c.streamIdleTimeout > 0 {
		timer = time.AfterFunc(24*time.Hour, cancel)
		timer.Stop()
	}
	defer stopTimer(timer)
	resetTimer := func(duration time.Duration, reason error) {
		if timer == nil || duration <= 0 {
			return
		}
		timer.Stop()
		timeoutReason.Store(reason)
		timer.Reset(duration)
	}
	reader := newSSEReader(response.Body, c.maxStreamLineBytes)
	state := &codec.StreamState{}
	emittedDone := false
	for {
		wire, err := reader.Next()
		if err != nil {
			if reason := timeoutReason.Load(); reason != nil && ctx.Err() != nil {
				stream.setTerminalError(&Error{Kind: ErrorTimeout, Message: reason.(error).Error(), Cause: reason.(error), Retryable: true})
				return
			}
			if errors.Is(err, io.EOF) {
				if !emittedDone {
					sendStreamItem(ctx, output, streamItem{event: llmprotocol.StreamEvent{Type: llmprotocol.StreamResponseDone, ResponseID: state.ResponseID, Model: state.Model}})
				}
				return
			}
			sendStreamItem(ctx, output, streamItem{err: &Error{Kind: ErrorStream, Message: "read upstream event stream", Cause: err, Retryable: true}})
			return
		}
		stopTimer(timer)
		events, diagnostics, decodeErr := formatCodec.DecodeStreamEvent(state, wire, c.policy)
		if len(diagnostics) > 0 {
			stream.metaMu.Lock()
			stream.metadata.ResponseDiagnostics = append(stream.metadata.ResponseDiagnostics, diagnostics...)
			stream.metaMu.Unlock()
		}
		if decodeErr != nil {
			sendStreamItem(ctx, output, streamItem{err: &Error{Kind: ErrorDecode, Message: decodeErr.Error(), Cause: decodeErr}})
			return
		}
		stopAfterOutput := false
		for _, event := range events {
			if event.Type == llmprotocol.StreamResponseDone {
				emittedDone = true
			}
			if event.Type == llmprotocol.StreamOutputDone &&
				formatCodec.Format() == llmprotocol.FormatOpenAIChat &&
				!c.includeStreamUsage {
				stopAfterOutput = true
			}
			if !sendStreamItem(ctx, output, streamItem{event: event}) {
				return
			}
		}
		if stopAfterOutput {
			return
		}
		resetTimer(c.streamIdleTimeout, ErrStreamIdleTimeout)
	}
}

func (c *Client) runAWSEventStream(ctx context.Context, cancel context.CancelFunc, reader io.Reader, formatCodec codec.Codec, output chan<- streamItem, stream *Stream) {
	var timeoutReason atomic.Value
	timeoutReason.Store(error(ErrFirstEventTimeout))
	timer := newStreamTimer(c.firstEventTimeout, cancel)
	if timer == nil && c.streamIdleTimeout > 0 {
		timer = time.AfterFunc(24*time.Hour, cancel)
		timer.Stop()
	}
	defer stopTimer(timer)
	resetTimer := func(duration time.Duration, reason error) {
		if timer == nil || duration <= 0 {
			return
		}
		timer.Stop()
		timeoutReason.Store(reason)
		timer.Reset(duration)
	}
	state := &codec.StreamState{}
	for {
		message, err := readAWSEventMessage(reader)
		if err != nil {
			if reason := timeoutReason.Load(); reason != nil && ctx.Err() != nil {
				stream.setTerminalError(&Error{Kind: ErrorTimeout, Message: reason.(error).Error(), Cause: reason.(error), Retryable: true})
				return
			}
			if errors.Is(err, io.EOF) {
				sendStreamItem(ctx, output, streamItem{event: llmprotocol.StreamEvent{Type: llmprotocol.StreamResponseDone, ResponseID: state.ResponseID, Model: state.Model}})
				return
			}
			sendStreamItem(ctx, output, streamItem{err: &Error{Kind: ErrorStream, Message: "read AWS event stream", Cause: err, Retryable: true}})
			return
		}
		stopTimer(timer)
		eventType := message.header(":event-type")
		if strings.EqualFold(message.header(":message-type"), "exception") {
			if exceptionType := message.header(":exception-type"); exceptionType != "" {
				eventType = exceptionType
			}
		}
		events, diagnostics, decodeErr := formatCodec.DecodeStreamEvent(state, llmprotocol.WireEvent{Event: eventType, Data: append(json.RawMessage(nil), message.payload...)}, c.policy)
		if len(diagnostics) > 0 {
			stream.metaMu.Lock()
			stream.metadata.ResponseDiagnostics = append(stream.metadata.ResponseDiagnostics, diagnostics...)
			stream.metaMu.Unlock()
		}
		if decodeErr != nil {
			sendStreamItem(ctx, output, streamItem{err: &Error{Kind: ErrorDecode, Message: decodeErr.Error(), Cause: decodeErr}})
			return
		}
		for _, event := range events {
			if !sendStreamItem(ctx, output, streamItem{event: event}) {
				return
			}
		}
		resetTimer(c.streamIdleTimeout, ErrStreamIdleTimeout)
	}
}

func sendStreamItem(ctx context.Context, output chan<- streamItem, item streamItem) bool {
	select {
	case output <- item:
		return true
	case <-ctx.Done():
		return false
	}
}

func (s *Stream) setTerminalError(err error) {
	s.terminalMu.Lock()
	if s.terminalErr == nil {
		s.terminalErr = err
	}
	s.terminalMu.Unlock()
}

func responseToEvents(response llmprotocol.Response) []llmprotocol.StreamEvent {
	events := []llmprotocol.StreamEvent{{Type: llmprotocol.StreamResponseStart, ResponseID: response.ID, Model: response.Model}}
	for outputIndex, output := range response.Outputs {
		events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamOutputStart, ResponseID: response.ID, Model: response.Model, OutputIndex: outputIndex, ItemID: output.ID, Role: output.Role})
		for contentIndex, block := range output.Content {
			switch block.Type {
			case llmprotocol.ContentText:
				events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamTextDelta, OutputIndex: outputIndex, ContentIndex: contentIndex, Delta: block.Text})
			case llmprotocol.ContentReasoning:
				events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamReasoningDelta, OutputIndex: outputIndex, ContentIndex: contentIndex, Delta: block.Text})
				if block.Signature != "" {
					events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamReasoningSignatureDelta, OutputIndex: outputIndex, ContentIndex: contentIndex, Signature: block.Signature})
				}
			case llmprotocol.ContentRefusal:
				events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamRefusalDelta, OutputIndex: outputIndex, ContentIndex: contentIndex, Delta: block.Text})
			case llmprotocol.ContentToolCall:
				if block.ToolCall != nil {
					events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamToolCallStart, OutputIndex: outputIndex, ContentIndex: contentIndex, ToolCallID: block.ToolCall.ID, ToolName: block.ToolCall.Name})
					events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamToolArgsDelta, OutputIndex: outputIndex, ContentIndex: contentIndex, ToolCallID: block.ToolCall.ID, Delta: string(block.ToolCall.Arguments)})
				}
			}
		}
		events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamOutputDone, OutputIndex: outputIndex, StopReason: output.StopReason})
	}
	if hasClientUsage(response.Usage) {
		usage := response.Usage
		events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamUsage, Usage: &usage})
	}
	events = append(events, llmprotocol.StreamEvent{Type: llmprotocol.StreamResponseDone, ResponseID: response.ID, Model: response.Model})
	return events
}

func hasClientUsage(value llmprotocol.Usage) bool {
	encoded, _ := json.Marshal(value)
	return string(encoded) != "{}"
}

func newStreamTimer(duration time.Duration, callback func()) *time.Timer {
	if duration <= 0 {
		return nil
	}
	return time.AfterFunc(duration, callback)
}

func stopTimer(timer *time.Timer) {
	if timer != nil {
		timer.Stop()
	}
}
