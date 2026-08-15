// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package llmprotocol

import (
	"fmt"
	"math"
)

// EmbeddingInputType identifies the portable form of one embedding input.
// Providers that accept only text must fail closed when asked to encode tokens.
type EmbeddingInputType string

const (
	EmbeddingInputText   EmbeddingInputType = "text"
	EmbeddingInputTokens EmbeddingInputType = "tokens"
)

type EmbeddingInput struct {
	Type   EmbeddingInputType `json:"type"`
	Text   string             `json:"text,omitempty"`
	Tokens []int64            `json:"tokens,omitempty"`
}

// EmbeddingTensor is an explicitly shaped, row-major dense tensor. Shape must
// have rank one or two. Keeping batch cardinality in EmbeddingResponse.Data
// means [d] is unambiguously a vector and [rows, columns] is unambiguously a
// matrix rather than a batch of vectors.
type EmbeddingTensor struct {
	Shape  []int64   `json:"shape"`
	Values []float64 `json:"values"`
}

func NewEmbeddingVector(values []float64) EmbeddingTensor {
	return EmbeddingTensor{Shape: []int64{int64(len(values))}, Values: append([]float64(nil), values...)}
}

func NewEmbeddingMatrix(rows [][]float64) (EmbeddingTensor, error) {
	if len(rows) == 0 {
		return EmbeddingTensor{}, fmt.Errorf("embedding matrix must have at least one row")
	}
	columns := len(rows[0])
	if columns == 0 {
		return EmbeddingTensor{}, fmt.Errorf("embedding matrix rows must not be empty")
	}
	values := make([]float64, 0, len(rows)*columns)
	for index, row := range rows {
		if len(row) != columns {
			return EmbeddingTensor{}, fmt.Errorf("embedding matrix row %d has %d columns; expected %d", index, len(row), columns)
		}
		values = append(values, row...)
	}
	return EmbeddingTensor{Shape: []int64{int64(len(rows)), int64(columns)}, Values: values}, nil
}

func (t EmbeddingTensor) Rank() int { return len(t.Shape) }

func (t EmbeddingTensor) Validate() error {
	if len(t.Shape) != 1 && len(t.Shape) != 2 {
		return fmt.Errorf("embedding tensor rank must be 1 or 2, got %d", len(t.Shape))
	}
	product := int64(1)
	for index, dimension := range t.Shape {
		if dimension <= 0 {
			return fmt.Errorf("embedding tensor dimension %d must be positive", index)
		}
		if product > int64(^uint(0)>>1)/dimension {
			return fmt.Errorf("embedding tensor shape is too large")
		}
		product *= dimension
	}
	if product != int64(len(t.Values)) {
		return fmt.Errorf("embedding tensor shape requires %d values, got %d", product, len(t.Values))
	}
	for index, value := range t.Values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("embedding tensor value %d must be finite", index)
		}
	}
	return nil
}

// Rows returns a defensive 2-D view. A rank-one tensor is returned as one row.
func (t EmbeddingTensor) Rows() ([][]float64, error) {
	if err := t.Validate(); err != nil {
		return nil, err
	}
	if len(t.Shape) == 1 {
		return [][]float64{append([]float64(nil), t.Values...)}, nil
	}
	rows, columns := int(t.Shape[0]), int(t.Shape[1])
	result := make([][]float64, rows)
	for row := 0; row < rows; row++ {
		start := row * columns
		result[row] = append([]float64(nil), t.Values[start:start+columns]...)
	}
	return result, nil
}

type EmbeddingRequest struct {
	Model          string           `json:"model,omitempty"`
	Inputs         []EmbeddingInput `json:"inputs"`
	EncodingFormat string           `json:"encoding_format,omitempty"`
	Dimensions     *int64           `json:"dimensions,omitempty"`
	User           string           `json:"user,omitempty"`
	TaskType       string           `json:"task_type,omitempty"`
	Title          string           `json:"title,omitempty"`
	InputType      string           `json:"input_type,omitempty"`
	Truncate       string           `json:"truncate,omitempty"`
	Normalize      *bool            `json:"normalize,omitempty"`
	AutoTruncate   *bool            `json:"auto_truncate,omitempty"`
	Extensions     Extensions       `json:"extensions,omitempty"`
	Preservation   Preservation     `json:"preservation,omitzero"`
}

func (r *EmbeddingRequest) ClearPreservation() { r.Preservation = Preservation{} }

func (r EmbeddingRequest) Validate() error {
	if len(r.Inputs) == 0 {
		return fmt.Errorf("embedding request requires at least one input")
	}
	if r.Dimensions != nil && *r.Dimensions <= 0 {
		return fmt.Errorf("embedding dimensions must be positive")
	}
	for index, input := range r.Inputs {
		switch input.Type {
		case EmbeddingInputText:
			if input.Text == "" {
				return fmt.Errorf("embedding input %d text must not be empty", index)
			}
		case EmbeddingInputTokens:
			if len(input.Tokens) == 0 {
				return fmt.Errorf("embedding input %d tokens must not be empty", index)
			}
			for _, token := range input.Tokens {
				if token < 0 {
					return fmt.Errorf("embedding input %d tokens must be non-negative", index)
				}
			}
		default:
			return fmt.Errorf("embedding input %d has unsupported type %q", index, input.Type)
		}
	}
	return nil
}

type EmbeddingOutput struct {
	Index      int             `json:"index"`
	Embedding  EmbeddingTensor `json:"embedding"`
	Extensions Extensions      `json:"extensions,omitempty"`
}

type EmbeddingResponse struct {
	Model          string            `json:"model,omitempty"`
	Data           []EmbeddingOutput `json:"data"`
	EncodingFormat string            `json:"encoding_format,omitempty"`
	Usage          Usage             `json:"usage,omitzero"`
	Extensions     Extensions        `json:"extensions,omitempty"`
	Preservation   Preservation      `json:"preservation,omitzero"`
}

func (r *EmbeddingResponse) ClearPreservation() { r.Preservation = Preservation{} }

func (r EmbeddingResponse) Validate() error {
	if len(r.Data) == 0 {
		return fmt.Errorf("embedding response requires at least one output")
	}
	for index, output := range r.Data {
		if err := output.Embedding.Validate(); err != nil {
			return fmt.Errorf("embedding output %d: %w", index, err)
		}
	}
	return nil
}
