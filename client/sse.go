// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package llmclient

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

type sseReader struct {
	scanner *bufio.Scanner
}

func newSSEReader(reader io.Reader, maxLineBytes int) *sseReader {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), maxLineBytes)
	return &sseReader{scanner: scanner}
}

func (r *sseReader) Next() (llmprotocol.WireEvent, error) {
	var event string
	var data []string
	for r.scanner.Scan() {
		line := strings.TrimSuffix(r.scanner.Text(), "\r")
		if line == "" {
			if event != "" || len(data) > 0 {
				return llmprotocol.WireEvent{Event: event, Data: json.RawMessage(strings.Join(data, "\n"))}, nil
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		name, value, found := strings.Cut(line, ":")
		if !found {
			value = ""
		}
		value = strings.TrimPrefix(value, " ")
		switch name {
		case "event":
			event = value
		case "data":
			data = append(data, value)
		}
	}
	if err := r.scanner.Err(); err != nil {
		return llmprotocol.WireEvent{}, err
	}
	if event != "" || len(data) > 0 {
		return llmprotocol.WireEvent{Event: event, Data: json.RawMessage(strings.Join(data, "\n"))}, nil
	}
	return llmprotocol.WireEvent{}, io.EOF
}
