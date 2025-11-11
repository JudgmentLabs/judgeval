package models

import (
	"encoding/json"
)

type OtelTraceSpan struct {
	OrganizationId     string        `json:"organization_id,omitempty"`
	ProjectId          string        `json:"project_id,omitempty"`
	UserId             string        `json:"user_id,omitempty"`
	Timestamp          string        `json:"timestamp,omitempty"`
	TraceId            string        `json:"trace_id,omitempty"`
	SpanId             string        `json:"span_id,omitempty"`
	ParentSpanId       string        `json:"parent_span_id,omitempty"`
	TraceState         string        `json:"trace_state,omitempty"`
	SpanName           string        `json:"span_name,omitempty"`
	SpanKind           string        `json:"span_kind,omitempty"`
	ServiceName        string        `json:"service_name,omitempty"`
	ResourceAttributes interface{}   `json:"resource_attributes,omitempty"`
	SpanAttributes     interface{}   `json:"span_attributes,omitempty"`
	Duration           int           `json:"duration,omitempty"`
	StatusCode         int           `json:"status_code,omitempty"`
	StatusMessage      string        `json:"status_message,omitempty"`
	Events             []interface{} `json:"events,omitempty"`
	Links              []interface{} `json:"links,omitempty"`

	AdditionalProperties map[string]interface{} `json:"-"`
}

func (m *OtelTraceSpan) UnmarshalJSON(data []byte) error {
	type Alias OtelTraceSpan
	aux := &struct {
		*Alias
	}{
		Alias: (*Alias)(m),
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		{
			return err
		}
	}
	m.AdditionalProperties = make(map[string]interface{})
	if err := json.Unmarshal(data, &m.AdditionalProperties); err != nil {
		{
			return err
		}
	}
	return nil
}

func (m OtelTraceSpan) MarshalJSON() ([]byte, error) {
	type Alias OtelTraceSpan
	aux := &struct {
		*Alias
	}{
		Alias: (*Alias)(&m),
	}

	result := make(map[string]interface{})

	mainBytes, err := json.Marshal(aux)
	if err != nil {
		{
			return nil, err
		}
	}

	if err := json.Unmarshal(mainBytes, &result); err != nil {
		{
			return nil, err
		}
	}

	for k, v := range m.AdditionalProperties {
		{
			result[k] = v
		}
	}

	return json.Marshal(result)
}
