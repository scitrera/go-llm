// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package llmclient

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

const awsSigV4Algorithm = "AWS4-HMAC-SHA256"

// AWSCredentials is the minimal material required for SigV4. The client never
// discovers, refreshes, persists, or logs this value.
type AWSCredentials struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
	ExpiresAt       time.Time
}

// AWSCredentialsProvider lets a host integrate static credentials, an AWS SDK
// chain, workload identity, or another refresh mechanism without coupling this
// module to that policy.
type AWSCredentialsProvider interface {
	Retrieve(context.Context) (AWSCredentials, error)
}

type AWSCredentialsProviderFunc func(context.Context) (AWSCredentials, error)

func (f AWSCredentialsProviderFunc) Retrieve(ctx context.Context) (AWSCredentials, error) {
	return f(ctx)
}

func StaticAWSCredentials(credentials AWSCredentials) AWSCredentialsProvider {
	return AWSCredentialsProviderFunc(func(context.Context) (AWSCredentials, error) { return credentials, nil })
}

// SigV4Authenticator signs the final HTTP request, including its escaped path
// and encoded body. Service defaults to "bedrock". Clock is intended for
// deterministic tests; production callers should leave it nil.
type SigV4Authenticator struct {
	Region      string
	Service     string
	Credentials AWSCredentialsProvider
	Clock       func() time.Time
}

func (a SigV4Authenticator) Apply(ctx context.Context, request *http.Request) error {
	if a.Credentials == nil {
		return fmt.Errorf("AWS SigV4 credentials provider is required")
	}
	if strings.TrimSpace(a.Region) == "" {
		return fmt.Errorf("AWS SigV4 region is required")
	}
	service := strings.TrimSpace(a.Service)
	if service == "" {
		service = "bedrock"
	}
	credentials, err := a.Credentials.Retrieve(ctx)
	if err != nil {
		return fmt.Errorf("retrieve AWS SigV4 credentials: %w", err)
	}
	now := time.Now().UTC()
	if a.Clock != nil {
		now = a.Clock().UTC()
	}
	return signSigV4Request(request, credentials, a.Region, service, now)
}

func signSigV4Request(request *http.Request, credentials AWSCredentials, region, service string, signingTime time.Time) error {
	if credentials.AccessKeyID == "" || credentials.SecretAccessKey == "" {
		return fmt.Errorf("AWS SigV4 access key and secret are required")
	}
	if strings.ContainsAny(credentials.AccessKeyID+credentials.SecretAccessKey+credentials.SessionToken, "\r\n") {
		return fmt.Errorf("AWS SigV4 credentials contain a newline")
	}
	if !credentials.ExpiresAt.IsZero() && !signingTime.Before(credentials.ExpiresAt) {
		return fmt.Errorf("AWS SigV4 credentials are expired")
	}
	if request.GetBody == nil {
		return fmt.Errorf("AWS SigV4 request body cannot be replayed for signing")
	}
	body, err := request.GetBody()
	if err != nil {
		return fmt.Errorf("open AWS SigV4 request body: %w", err)
	}
	payloadDigest, readErr := hashReader(body)
	_ = body.Close()
	if readErr != nil {
		return fmt.Errorf("hash AWS SigV4 request body: %w", readErr)
	}
	payloadHash := hex.EncodeToString(payloadDigest[:])
	amzDate := signingTime.Format("20060102T150405Z")
	shortDate := signingTime.Format("20060102")

	request.Header.Del("Authorization")
	request.Header.Set("X-Amz-Date", amzDate)
	request.Header.Set("X-Amz-Content-Sha256", payloadHash)
	if credentials.SessionToken == "" {
		request.Header.Del("X-Amz-Security-Token")
	} else {
		request.Header.Set("X-Amz-Security-Token", credentials.SessionToken)
	}
	if request.Header.Get("Content-Type") == "" {
		request.Header.Set("Content-Type", "application/json")
	}

	canonicalURI := canonicalAWSURI(request.URL)
	canonicalQuery, err := canonicalAWSQuery(request.URL)
	if err != nil {
		return err
	}
	host := request.URL.Host
	if request.Host != "" {
		host = request.Host
	}
	headers := map[string]string{
		"content-type":         canonicalAWSHeaderValue(request.Header.Get("Content-Type")),
		"host":                 canonicalAWSHeaderValue(host),
		"x-amz-content-sha256": payloadHash,
		"x-amz-date":           amzDate,
	}
	if credentials.SessionToken != "" {
		headers["x-amz-security-token"] = canonicalAWSHeaderValue(credentials.SessionToken)
	}
	names := make([]string, 0, len(headers))
	for name := range headers {
		names = append(names, name)
	}
	sort.Strings(names)
	var canonicalHeaders strings.Builder
	for _, name := range names {
		canonicalHeaders.WriteString(name)
		canonicalHeaders.WriteByte(':')
		canonicalHeaders.WriteString(headers[name])
		canonicalHeaders.WriteByte('\n')
	}
	signedHeaders := strings.Join(names, ";")
	canonicalRequest := strings.Join([]string{request.Method, canonicalURI, canonicalQuery, canonicalHeaders.String(), signedHeaders, payloadHash}, "\n")
	canonicalDigest := sha256.Sum256([]byte(canonicalRequest))
	scope := strings.Join([]string{shortDate, region, service, "aws4_request"}, "/")
	stringToSign := strings.Join([]string{awsSigV4Algorithm, amzDate, scope, hex.EncodeToString(canonicalDigest[:])}, "\n")
	dateKey := awsHMAC([]byte("AWS4"+credentials.SecretAccessKey), shortDate)
	regionKey := awsHMAC(dateKey, region)
	serviceKey := awsHMAC(regionKey, service)
	signingKey := awsHMAC(serviceKey, "aws4_request")
	signature := hex.EncodeToString(awsHMAC(signingKey, stringToSign))
	request.Header.Set("Authorization", awsSigV4Algorithm+" Credential="+credentials.AccessKeyID+"/"+scope+", SignedHeaders="+signedHeaders+", Signature="+signature)
	return nil
}

func hashReader(reader io.Reader) ([sha256.Size]byte, error) {
	hash := sha256.New()
	_, err := io.Copy(hash, reader)
	var result [sha256.Size]byte
	copy(result[:], hash.Sum(nil))
	return result, err
}

func awsHMAC(key []byte, value string) []byte {
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte(value))
	return mac.Sum(nil)
}

func canonicalAWSURI(value *url.URL) string {
	escaped := value.EscapedPath()
	if escaped == "" {
		return "/"
	}
	return awsURIEncode(escaped, true)
}

func canonicalAWSQuery(value *url.URL) (string, error) {
	if value.RawQuery == "" {
		return "", nil
	}
	values, err := url.ParseQuery(value.RawQuery)
	if err != nil {
		return "", fmt.Errorf("decode AWS request query: %w", err)
	}
	type pair struct{ key, value string }
	var pairs []pair
	for key, items := range values {
		if len(items) == 0 {
			items = []string{""}
		}
		for _, item := range items {
			pairs = append(pairs, pair{awsURIEncode(key, false), awsURIEncode(item, false)})
		}
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].key == pairs[j].key {
			return pairs[i].value < pairs[j].value
		}
		return pairs[i].key < pairs[j].key
	})
	var result strings.Builder
	for index, item := range pairs {
		if index != 0 {
			result.WriteByte('&')
		}
		result.WriteString(item.key)
		result.WriteByte('=')
		result.WriteString(item.value)
	}
	return result.String(), nil
}

func awsURIEncode(value string, preserveSlash bool) string {
	const upperHex = "0123456789ABCDEF"
	var encoded strings.Builder
	encoded.Grow(len(value))
	for index := 0; index < len(value); index++ {
		character := value[index]
		if character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z' || character >= '0' && character <= '9' || character == '-' || character == '.' || character == '_' || character == '~' || character == '/' && preserveSlash {
			encoded.WriteByte(character)
			continue
		}
		encoded.WriteByte('%')
		encoded.WriteByte(upperHex[character>>4])
		encoded.WriteByte(upperHex[character&0x0f])
	}
	return encoded.String()
}

func canonicalAWSHeaderValue(value string) string { return strings.Join(strings.Fields(value), " ") }
