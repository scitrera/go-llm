// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package llmclient

import (
	"bytes"
	"context"
	"encoding/binary"
	"hash/crc32"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llmprotocol "github.com/scitrera/go-llm/protocol"
)

func TestBedrockBinaryStreamClient(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.EscapedPath() != "/model/provider%2Fmodel/converse-stream" || r.Header.Get("Accept") != "application/vnd.amazon.eventstream" || r.Header.Get("X-Test-Signature") != "signed" {
			t.Errorf("request path=%q accept=%q signature=%q", r.URL.EscapedPath(), r.Header.Get("Accept"), r.Header.Get("X-Test-Signature"))
		}
		body, _ := io.ReadAll(r.Body)
		if !bytes.Contains(body, []byte(`"messages"`)) || bytes.Contains(body, []byte(`"model"`)) {
			t.Errorf("request body = %s", body)
		}
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		for _, event := range []struct{ name, body string }{
			{"messageStart", `{"role":"assistant"}`},
			{"contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"hello"}}`},
			{"messageStop", `{"stopReason":"end_turn"}`},
			{"metadata", `{"usage":{"inputTokens":4,"outputTokens":1,"totalTokens":5}}`},
		} {
			_, _ = w.Write(encodeTestAWSEvent(event.name, []byte(event.body)))
		}
	}))
	defer server.Close()

	client := New(Options{})
	stream, err := client.Stream(context.Background(), Backend{
		Name: "bedrock", Format: llmprotocol.FormatBedrock, BaseURL: server.URL,
		Model: "provider/model",
		Auth: AuthFunc(func(_ context.Context, request *http.Request) error {
			request.Header.Set("X-Test-Signature", "signed")
			return nil
		}),
	}, llmprotocol.Request{Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.ContentBlock{llmprotocol.Text("hello")}}}})
	if err != nil {
		t.Fatal(err)
	}
	response, err := stream.Collect(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Outputs) != 1 || len(response.Outputs[0].Content) != 1 || response.Outputs[0].Content[0].Text != "hello" || response.Usage.TotalTokens == nil || *response.Usage.TotalTokens != 5 {
		t.Fatalf("response = %#v", response)
	}
}

func TestAWSEventStreamRejectsBadChecksum(t *testing.T) {
	t.Parallel()
	message := encodeTestAWSEvent("messageStart", []byte(`{"role":"assistant"}`))
	message[len(message)-1] ^= 0xff
	if _, err := readAWSEventMessage(bytes.NewReader(message)); err == nil {
		t.Fatal("readAWSEventMessage() error = nil")
	}
}

func encodeTestAWSEvent(eventType string, payload []byte) []byte {
	headers := bytes.Buffer{}
	for _, header := range []struct{ name, value string }{
		{":message-type", "event"}, {":event-type", eventType}, {":content-type", "application/json"},
	} {
		headers.WriteByte(byte(len(header.name)))
		headers.WriteString(header.name)
		headers.WriteByte(7)
		_ = binary.Write(&headers, binary.BigEndian, uint16(len(header.value)))
		headers.WriteString(header.value)
	}
	total := awsEventOverheadBytes + headers.Len() + len(payload)
	message := make([]byte, total)
	binary.BigEndian.PutUint32(message[0:4], uint32(total))
	binary.BigEndian.PutUint32(message[4:8], uint32(headers.Len()))
	binary.BigEndian.PutUint32(message[8:12], crc32.ChecksumIEEE(message[:8]))
	copy(message[12:], headers.Bytes())
	copy(message[12+headers.Len():], payload)
	binary.BigEndian.PutUint32(message[total-4:], crc32.ChecksumIEEE(message[:total-4]))
	return message
}
