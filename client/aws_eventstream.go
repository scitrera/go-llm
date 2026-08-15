// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package llmclient

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"strings"
)

const (
	awsEventPreludeBytes    = 12
	awsEventOverheadBytes   = 16
	maxAWSEventPayloadBytes = 16 << 20
	maxAWSEventHeaderBytes  = 128 << 10
	maxAWSEventMessageBytes = maxAWSEventPayloadBytes + maxAWSEventHeaderBytes + awsEventOverheadBytes
)

type awsEventMessage struct {
	headers map[string]string
	payload []byte
}

func (m awsEventMessage) header(name string) string { return m.headers[strings.ToLower(name)] }

func readAWSEventMessage(source io.Reader) (awsEventMessage, error) {
	var prelude [awsEventPreludeBytes]byte
	read, err := io.ReadFull(source, prelude[:])
	if err != nil {
		if err == io.EOF && read == 0 {
			return awsEventMessage{}, io.EOF
		}
		return awsEventMessage{}, io.ErrUnexpectedEOF
	}
	totalLength := int(binary.BigEndian.Uint32(prelude[0:4]))
	headersLength := int(binary.BigEndian.Uint32(prelude[4:8]))
	if totalLength < awsEventOverheadBytes || totalLength > maxAWSEventMessageBytes {
		return awsEventMessage{}, fmt.Errorf("AWS event-stream message length is invalid")
	}
	if headersLength > maxAWSEventHeaderBytes || headersLength > totalLength-awsEventOverheadBytes {
		return awsEventMessage{}, fmt.Errorf("AWS event-stream header length is invalid")
	}
	if totalLength-awsEventOverheadBytes-headersLength > maxAWSEventPayloadBytes {
		return awsEventMessage{}, fmt.Errorf("AWS event-stream payload length is invalid")
	}
	if crc32.ChecksumIEEE(prelude[:8]) != binary.BigEndian.Uint32(prelude[8:12]) {
		return awsEventMessage{}, fmt.Errorf("AWS event-stream prelude checksum is invalid")
	}
	raw := make([]byte, totalLength)
	copy(raw, prelude[:])
	if _, err := io.ReadFull(source, raw[awsEventPreludeBytes:]); err != nil {
		return awsEventMessage{}, io.ErrUnexpectedEOF
	}
	if crc32.ChecksumIEEE(raw[:totalLength-4]) != binary.BigEndian.Uint32(raw[totalLength-4:]) {
		return awsEventMessage{}, fmt.Errorf("AWS event-stream message checksum is invalid")
	}
	headers, err := decodeAWSEventHeaders(raw[awsEventPreludeBytes : awsEventPreludeBytes+headersLength])
	if err != nil {
		return awsEventMessage{}, err
	}
	return awsEventMessage{headers: headers, payload: append([]byte(nil), raw[awsEventPreludeBytes+headersLength:totalLength-4]...)}, nil
}

func decodeAWSEventHeaders(raw []byte) (map[string]string, error) {
	headers := make(map[string]string)
	for len(raw) != 0 {
		nameLength := int(raw[0])
		raw = raw[1:]
		if nameLength == 0 || len(raw) < nameLength+1 {
			return nil, fmt.Errorf("AWS event-stream header is truncated")
		}
		name := strings.ToLower(string(raw[:nameLength]))
		raw = raw[nameLength:]
		valueType := raw[0]
		raw = raw[1:]
		value, consumed, err := decodeAWSEventHeaderValue(valueType, raw)
		if err != nil {
			return nil, err
		}
		raw = raw[consumed:]
		if _, exists := headers[name]; exists {
			return nil, fmt.Errorf("AWS event-stream contains a duplicate header")
		}
		headers[name] = value
	}
	return headers, nil
}

func decodeAWSEventHeaderValue(valueType byte, raw []byte) (string, int, error) {
	switch valueType {
	case 0:
		return "true", 0, nil
	case 1:
		return "false", 0, nil
	case 2:
		if len(raw) >= 1 {
			return "", 1, nil
		}
	case 3:
		if len(raw) >= 2 {
			return "", 2, nil
		}
	case 4:
		if len(raw) >= 4 {
			return "", 4, nil
		}
	case 5, 8:
		if len(raw) >= 8 {
			return "", 8, nil
		}
	case 6, 7:
		if len(raw) >= 2 {
			length := int(binary.BigEndian.Uint16(raw[:2]))
			if len(raw) >= 2+length {
				if valueType == 7 {
					return string(raw[2 : 2+length]), 2 + length, nil
				}
				return "", 2 + length, nil
			}
		}
	case 9:
		if len(raw) >= 16 {
			return "", 16, nil
		}
	default:
		return "", 0, fmt.Errorf("AWS event-stream header has an unknown value type")
	}
	return "", 0, fmt.Errorf("AWS event-stream header value is truncated")
}
