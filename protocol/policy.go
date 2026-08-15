// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Scitrera LLC

package llmprotocol

import "fmt"

type UnknownFieldPolicy string

const (
	UnknownPreserve UnknownFieldPolicy = "preserve"
	UnknownDrop     UnknownFieldPolicy = "drop"
	UnknownReject   UnknownFieldPolicy = "reject"
)

type LossyPolicy string

const (
	LossyReject LossyPolicy = "reject"
	LossyAllow  LossyPolicy = "allow_with_diagnostics"
)

type PreservationPolicy string

const (
	PreserveInMemory PreservationPolicy = "in_memory"
	PreserveDisabled PreservationPolicy = "disabled"
)

type TargetCapabilities struct {
	Tools             *bool
	Images            *bool
	Audio             *bool
	Video             *bool
	Files             *bool
	Reasoning         *bool
	JSONSchema        *bool
	ParallelToolCalls *bool
}

type Policy struct {
	UnknownFields UnknownFieldPolicy
	Lossy         LossyPolicy
	Preservation  PreservationPolicy
	Target        TargetCapabilities
}

func StrictPolicy() Policy {
	return Policy{
		UnknownFields: UnknownPreserve,
		Lossy:         LossyReject,
		Preservation:  PreserveInMemory,
	}
}

func PermissivePolicy() Policy {
	return Policy{
		UnknownFields: UnknownPreserve,
		Lossy:         LossyAllow,
		Preservation:  PreserveInMemory,
	}
}

func (p Policy) Effective() Policy {
	if p.UnknownFields == "" {
		p.UnknownFields = UnknownPreserve
	}
	if p.Lossy == "" {
		p.Lossy = LossyReject
	}
	if p.Preservation == "" {
		p.Preservation = PreserveInMemory
	}
	return p
}

type DiagnosticKind string

const (
	DiagnosticUnknownField DiagnosticKind = "unknown_field"
	DiagnosticLossy        DiagnosticKind = "lossy_conversion"
	DiagnosticGeneratedID  DiagnosticKind = "generated_id"
)

// Diagnostic is deliberately content-free and safe for logs or metrics. Path
// identifies a bounded schema location, not a caller-provided value.
type Diagnostic struct {
	Kind   DiagnosticKind `json:"kind"`
	Path   string         `json:"path,omitempty"`
	Source Format         `json:"source,omitempty"`
	Target Format         `json:"target,omitempty"`
	Code   string         `json:"code"`
}

type TranslationError struct {
	Format Format
	Path   string
	Code   string
	Detail string
}

func (e *TranslationError) Error() string {
	if e.Path == "" {
		return fmt.Sprintf("%s translation %s: %s", e.Format, e.Code, e.Detail)
	}
	return fmt.Sprintf("%s translation %s at %s: %s", e.Format, e.Code, e.Path, e.Detail)
}
