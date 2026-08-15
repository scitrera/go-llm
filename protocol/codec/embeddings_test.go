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

func TestEmbeddingSameFormatPreservationIsExact(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultEmbeddingRegistry()
	fixtures := map[llmprotocol.Format]json.RawMessage{
		llmprotocol.FormatOpenAIEmbeddings:        json.RawMessage(`{"model":"m","input":["one","two"],"metadata":{"keep":true}}`),
		llmprotocol.FormatGeminiEmbeddings:        json.RawMessage(`{"content":{"parts":[{"text":"one"}]},"embedContentConfig":{"autoTruncate":false}}`),
		llmprotocol.FormatBedrockTitanEmbeddings:  json.RawMessage(`{"inputText":"one","normalize":false}`),
		llmprotocol.FormatBedrockCohereEmbeddings: json.RawMessage(`{"texts":["one"],"input_type":"search_query"}`),
	}
	for format, body := range fixtures {
		format, body := format, body
		t.Run(string(format), func(t *testing.T) {
			t.Parallel()
			translated, err := registry.TranslateRequest(format, format, body, llmprotocol.StrictPolicy())
			if err != nil {
				t.Fatalf("TranslateRequest() error = %v", err)
			}
			if string(translated.Body) != string(body) {
				t.Fatalf("body changed\n got: %s\nwant: %s", translated.Body, body)
			}
		})
	}
}

func TestOpenAIEmbeddingRequestTranslatesToGeminiBatch(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultEmbeddingRegistry()
	translated, err := registry.TranslateRequest(
		llmprotocol.FormatOpenAIEmbeddings,
		llmprotocol.FormatGeminiEmbeddings,
		json.RawMessage(`{"model":"text-embedding","input":["one","two"],"encoding_format":"float","dimensions":256}`),
		llmprotocol.StrictPolicy(),
	)
	if err != nil {
		t.Fatalf("TranslateRequest() error = %v", err)
	}
	var body struct {
		Requests []struct {
			Model  string `json:"model"`
			Config struct {
				Dimensions int64 `json:"outputDimensionality"`
			} `json:"embedContentConfig"`
		} `json:"requests"`
	}
	if err := json.Unmarshal(translated.Body, &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Requests) != 2 || body.Requests[0].Model != "models/text-embedding" || body.Requests[1].Config.Dimensions != 256 {
		t.Fatalf("Gemini request = %s", translated.Body)
	}
}

func TestGeminiRankTwoEmbeddingIsPreservedInNeutralContract(t *testing.T) {
	t.Parallel()
	value := codec.GeminiEmbeddings{}
	decoded, err := value.DecodeEmbeddingResponse(
		json.RawMessage(`{"embedding":{"values":[1,2,3,4,5,6],"shape":[2,3]}}`),
		llmprotocol.StrictPolicy(),
	)
	if err != nil {
		t.Fatalf("DecodeEmbeddingResponse() error = %v", err)
	}
	if len(decoded.Response.Data) != 1 {
		t.Fatalf("response data = %#v", decoded.Response.Data)
	}
	tensor := decoded.Response.Data[0].Embedding
	if tensor.Rank() != 2 || tensor.Shape[0] != 2 || tensor.Shape[1] != 3 || tensor.Values[5] != 6 {
		t.Fatalf("tensor = %#v", tensor)
	}
	decoded.Response.ClearPreservation()
	encoded, err := value.EncodeEmbeddingResponse(decoded.Response, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("EncodeEmbeddingResponse() error = %v", err)
	}
	var body struct {
		Embedding struct {
			Shape []int64 `json:"shape"`
		} `json:"embedding"`
	}
	if err := json.Unmarshal(encoded.Body, &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Embedding.Shape) != 2 || body.Embedding.Shape[1] != 3 {
		t.Fatalf("encoded Gemini response = %s", encoded.Body)
	}
}

func TestRankTwoEmbeddingFailsClosedForRankOneWireFormats(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultEmbeddingRegistry()
	_, err := registry.TranslateResponse(
		llmprotocol.FormatGeminiEmbeddings,
		llmprotocol.FormatOpenAIEmbeddings,
		json.RawMessage(`{"embedding":{"values":[1,2,3,4],"shape":[2,2]}}`),
		llmprotocol.StrictPolicy(),
	)
	var translationErr *llmprotocol.TranslationError
	if !errors.As(err, &translationErr) || translationErr.Code != "rank_not_supported" {
		t.Fatalf("rank-2 Gemini -> OpenAI error = %v", err)
	}
}

func TestEmbeddingModelFamiliesTranslateWithoutConflatingBatchAndRank(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultEmbeddingRegistry()
	cohere := json.RawMessage(`{"response_type":"embeddings_floats","embeddings":[[1,2],[3,4]],"texts":["one","two"]}`)
	openai, err := registry.TranslateResponse(llmprotocol.FormatBedrockCohereEmbeddings, llmprotocol.FormatOpenAIEmbeddings, cohere, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("Cohere -> OpenAI error = %v", err)
	}
	var body struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(openai.Body, &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Data) != 2 || len(body.Data[0].Embedding) != 2 || body.Data[1].Embedding[1] != 4 {
		t.Fatalf("OpenAI response = %s", openai.Body)
	}
}

func TestProviderEmbeddingExtensionsDoNotCrossFormats(t *testing.T) {
	t.Parallel()
	registry := codec.NewDefaultEmbeddingRegistry()
	body := json.RawMessage(`{"model":"m","input":"one","provider_private":{"token":"opaque"}}`)
	_, err := registry.TranslateRequest(llmprotocol.FormatOpenAIEmbeddings, llmprotocol.FormatGeminiEmbeddings, body, llmprotocol.StrictPolicy())
	var translationErr *llmprotocol.TranslationError
	if !errors.As(err, &translationErr) || translationErr.Code != "provider_extensions_not_portable" {
		t.Fatalf("strict extension error = %v", err)
	}
}

func TestTitanEmbeddingCodecBranchesForG1AndV2(t *testing.T) {
	t.Parallel()
	value := codec.BedrockTitanEmbeddings{}
	request := llmprotocol.EmbeddingRequest{
		Model:  "amazon.titan-embed-text-v1",
		Inputs: []llmprotocol.EmbeddingInput{{Type: llmprotocol.EmbeddingInputText, Text: "one"}},
	}
	g1, err := value.EncodeEmbeddingRequest(request, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("G1 EncodeEmbeddingRequest() error = %v", err)
	}
	var g1Body map[string]any
	if err := json.Unmarshal(g1.Body, &g1Body); err != nil {
		t.Fatal(err)
	}
	if _, exists := g1Body["embeddingTypes"]; exists {
		t.Fatalf("Titan G1 body contains V2 controls: %s", g1.Body)
	}
	request.Model = "amazon.titan-embed-text-v2:0"
	v2, err := value.EncodeEmbeddingRequest(request, llmprotocol.StrictPolicy())
	if err != nil {
		t.Fatalf("V2 EncodeEmbeddingRequest() error = %v", err)
	}
	var v2Body map[string]any
	if err := json.Unmarshal(v2.Body, &v2Body); err != nil {
		t.Fatal(err)
	}
	if _, exists := v2Body["embeddingTypes"]; !exists {
		t.Fatalf("Titan V2 body = %s", v2.Body)
	}
}
