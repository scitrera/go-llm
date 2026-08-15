// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package llmclient

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
)

type ErrorKind string

const (
	ErrorInvalidConfig ErrorKind = "invalid_config"
	ErrorEncode        ErrorKind = "encode"
	ErrorTransport     ErrorKind = "transport"
	ErrorTimeout       ErrorKind = "timeout"
	ErrorCanceled      ErrorKind = "canceled"
	ErrorUpstream      ErrorKind = "upstream"
	ErrorResponseLimit ErrorKind = "response_limit"
	ErrorDecode        ErrorKind = "decode"
	ErrorStream        ErrorKind = "stream"
)

var (
	ErrFirstEventTimeout = errors.New("llm client first stream event timeout")
	ErrStreamIdleTimeout = errors.New("llm client stream idle timeout")
)

type Error struct {
	Kind       ErrorKind
	StatusCode int
	Type       string
	Code       string
	Message    string
	Retryable  bool
	Cause      error
	Body       json.RawMessage
}

func (e *Error) Error() string {
	message := e.Message
	if message == "" {
		message = string(e.Kind)
	}
	if e.StatusCode != 0 {
		return fmt.Sprintf("llm client %s: HTTP %d: %s", e.Kind, e.StatusCode, message)
	}
	return "llm client " + string(e.Kind) + ": " + message
}

func (e *Error) Unwrap() error { return e.Cause }

func statusRetryable(status int) bool {
	return status == http.StatusRequestTimeout || status == http.StatusTooManyRequests || status >= 500
}

func decodeHTTPError(status int, body json.RawMessage) *Error {
	result := &Error{Kind: ErrorUpstream, StatusCode: status, Retryable: statusRetryable(status), Body: append(json.RawMessage(nil), body...)}
	var openAI struct {
		Error struct {
			Type    string `json:"type"`
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
		Type    string `json:"type"`
		Message string `json:"message"`
	}
	if json.Unmarshal(body, &openAI) == nil {
		result.Type, result.Code, result.Message = openAI.Error.Type, openAI.Error.Code, openAI.Error.Message
		if result.Message == "" {
			result.Type, result.Message = openAI.Type, openAI.Message
		}
	}
	if result.Message == "" {
		result.Message = http.StatusText(status)
	}
	return result
}
