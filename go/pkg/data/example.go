// Package data provides legacy data structures.
//
// Deprecated: Use github.com/JudgmentLabs/judgeval-go/v1 instead.
// This package will be removed in a future version.
package data

import (
	"time"

	"github.com/JudgmentLabs/judgeval-go/pkg/internal/api/models"
	"github.com/google/uuid"
)

type Example struct {
	*models.Example
}

type ExampleOptions func(*Example)

func WithProperty(key string, value interface{}) ExampleOptions {
	return func(e *Example) {
		if e.AdditionalProperties == nil {
			e.AdditionalProperties = make(map[string]interface{})
		}
		e.AdditionalProperties[key] = value
	}
}

func WithName(name string) ExampleOptions {
	return func(e *Example) {
		e.Name = name
	}
}

func NewExample(options ...ExampleOptions) *Example {
	example := &Example{
		Example: &models.Example{
			AdditionalProperties: make(map[string]interface{}),
		},
	}

	example.ExampleId = uuid.New().String()
	example.CreatedAt = time.Now().Format(time.RFC3339)
	example.Name = ""

	for _, option := range options {
		option(example)
	}

	return example
}
