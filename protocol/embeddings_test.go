// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package llmprotocol

import (
	"math"
	"testing"
)

func TestEmbeddingTensorSupportsVectorAndMatrix(t *testing.T) {
	t.Parallel()
	vector := NewEmbeddingVector([]float64{1, 2, 3})
	if err := vector.Validate(); err != nil || vector.Rank() != 1 || len(vector.Shape) != 1 || vector.Shape[0] != 3 {
		t.Fatalf("vector = %#v, error = %v", vector, err)
	}
	matrix, err := NewEmbeddingMatrix([][]float64{{1, 2}, {3, 4}, {5, 6}})
	if err != nil || matrix.Rank() != 2 || matrix.Shape[0] != 3 || matrix.Shape[1] != 2 {
		t.Fatalf("matrix = %#v, error = %v", matrix, err)
	}
	rows, err := matrix.Rows()
	if err != nil || len(rows) != 3 || rows[2][1] != 6 {
		t.Fatalf("rows = %#v, error = %v", rows, err)
	}
}

func TestEmbeddingTensorRejectsAmbiguousOrRaggedShapes(t *testing.T) {
	t.Parallel()
	invalid := []EmbeddingTensor{
		{},
		{Shape: []int64{4}, Values: []float64{1, 2}},
		{Shape: []int64{2, 0}, Values: []float64{}},
		{Shape: []int64{1, 1, 1}, Values: []float64{1}},
		{Shape: []int64{1}, Values: []float64{math.NaN()}},
	}
	for _, tensor := range invalid {
		if err := tensor.Validate(); err == nil {
			t.Fatalf("Validate(%#v) succeeded, want error", tensor)
		}
	}
	if _, err := NewEmbeddingMatrix([][]float64{{1, 2}, {3}}); err == nil {
		t.Fatal("NewEmbeddingMatrix() accepted ragged rows")
	}
}
