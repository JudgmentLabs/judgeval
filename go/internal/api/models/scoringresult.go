package models

import (
	"encoding/json"
)

type ScoringResult struct {
	Success        bool         `json:"success,omitempty"`
	ScorersData    []ScorerData `json:"scorers_data,omitempty"`
	Name           string       `json:"name,omitempty"`
	DataObject     interface{}  `json:"data_object,omitempty"`
	TraceId        string       `json:"trace_id,omitempty"`
	RunDuration    float64      `json:"run_duration,omitempty"`
	EvaluationCost float64      `json:"evaluation_cost,omitempty"`

	AdditionalProperties map[string]interface{} `json:"-"`
}

func (m *ScoringResult) UnmarshalJSON(data []byte) error {
	type Alias ScoringResult
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

func (m ScoringResult) MarshalJSON() ([]byte, error) {
	type Alias ScoringResult
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
