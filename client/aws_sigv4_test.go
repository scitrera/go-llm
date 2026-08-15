// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package llmclient

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestSigV4AuthenticatorSignsFinalBedrockRequest(t *testing.T) {
	t.Parallel()
	body := []byte(`{"messages":[{"role":"user","content":[{"text":"hello"}]}]}`)
	backend := Backend{BaseURL: "https://bedrock-runtime.us-east-1.amazonaws.com", Format: "bedrock_converse"}
	endpoint, err := backend.endpoint("arn:aws:bedrock:us-east-1:123456789012:prompt/PROMPT1234:1", false)
	if err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequest(http.MethodPost, endpoint.String(), bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	auth := SigV4Authenticator{
		Region:      "us-east-1",
		Credentials: StaticAWSCredentials(AWSCredentials{AccessKeyID: "AKIDEXAMPLE", SecretAccessKey: "secret-example", SessionToken: "session-example"}),
		Clock:       func() time.Time { return time.Date(2026, time.July, 30, 12, 34, 56, 0, time.UTC) },
	}
	if err := auth.Apply(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(body)
	if request.Header.Get("X-Amz-Date") != "20260730T123456Z" || request.Header.Get("X-Amz-Content-Sha256") != hex.EncodeToString(digest[:]) || request.Header.Get("X-Amz-Security-Token") != "session-example" {
		t.Fatalf("signed headers = %#v", request.Header)
	}
	authorization := request.Header.Get("Authorization")
	if !strings.HasPrefix(authorization, "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260730/us-east-1/bedrock/aws4_request, ") || !strings.Contains(authorization, "SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date;x-amz-security-token") || !strings.Contains(authorization, "Signature=") {
		t.Fatalf("Authorization = %q", authorization)
	}
	if request.URL.EscapedPath() != "/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Aprompt%2FPROMPT1234%3A1/converse" {
		t.Fatalf("wire path = %q", request.URL.EscapedPath())
	}
	if canonicalAWSURI(request.URL) != "/model/arn%253Aaws%253Abedrock%253Aus-east-1%253A123456789012%253Aprompt%252FPROMPT1234%253A1/converse" {
		t.Fatalf("canonical URI = %q", canonicalAWSURI(request.URL))
	}
}

func TestSigV4AuthenticatorRejectsExpiredCredentials(t *testing.T) {
	t.Parallel()
	request, _ := http.NewRequest(http.MethodPost, "https://example.test/model/m/converse", bytes.NewReader([]byte(`{}`)))
	auth := SigV4Authenticator{
		Region:      "us-east-1",
		Credentials: StaticAWSCredentials(AWSCredentials{AccessKeyID: "id", SecretAccessKey: "secret", ExpiresAt: time.Unix(10, 0)}),
		Clock:       func() time.Time { return time.Unix(10, 0) },
	}
	if err := auth.Apply(context.Background(), request); err == nil || !strings.Contains(err.Error(), "expired") {
		t.Fatalf("Apply() error = %v", err)
	}
}
