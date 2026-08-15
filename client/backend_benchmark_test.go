// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package llmclient

import (
	"encoding/json"
	"testing"
)

func BenchmarkApplyBodyDefaults(b *testing.B) {
	body := json.RawMessage(`{"model":"served","messages":[{"role":"user","content":"hello"}],"temperature":0}`)
	defaults := map[string]json.RawMessage{
		"temperature":  json.RawMessage(`1`),
		"service_tier": json.RawMessage(`"priority"`),
		"top_k":        json.RawMessage(`40`),
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(body)))
	for i := 0; i < b.N; i++ {
		result, err := MergeBodyDefaults(body, defaults)
		if err != nil || len(result) == 0 {
			b.Fatalf("MergeBodyDefaults() body=%d error=%v", len(result), err)
		}
	}
}
