package v1

import (
	"maps"
	"time"

	"github.com/JudgmentLabs/judgeval-go/v1/internal/api/models"
	"github.com/google/uuid"
)

type ExampleParams struct {
	Name       *string
	Properties map[string]interface{}
}

type Example struct {
	exampleID  string
	createdAt  string
	name       *string
	properties map[string]interface{}
}

func NewExample(params ExampleParams) *Example {
	exampleID := uuid.New().String()
	createdAt := time.Now().Format(time.RFC3339)

	properties := make(map[string]interface{})
	if params.Properties != nil {
		maps.Copy(properties, params.Properties)
	}

	return &Example{
		exampleID:  exampleID,
		createdAt:  createdAt,
		name:       params.Name,
		properties: properties,
	}
}

func (e *Example) SetProperty(key string, value interface{}) *Example {
	e.properties[key] = value
	return e
}

func (e *Example) GetProperty(key string) interface{} {
	return e.properties[key]
}

func (e *Example) GetProperties() map[string]interface{} {
	propsCopy := make(map[string]interface{})
	maps.Copy(propsCopy, e.properties)
	return propsCopy
}

func (e *Example) GetExampleID() string {
	return e.exampleID
}

func (e *Example) GetCreatedAt() string {
	return e.createdAt
}

func (e *Example) GetName() *string {
	return e.name
}

func (e *Example) SetName(name string) {
	e.name = &name
}

func (e *Example) toModel() models.Example {
	result := models.Example{
		ExampleId:            e.exampleID,
		CreatedAt:            e.createdAt,
		AdditionalProperties: make(map[string]interface{}),
	}

	if e.name != nil {
		result.Name = *e.name
	}

	for k, v := range e.properties {
		result.AdditionalProperties[k] = v
	}

	return result
}
