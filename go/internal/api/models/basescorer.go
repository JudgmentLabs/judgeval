package models

import (
	"encoding/json"
)

type BaseScorer struct {
	ScoreType          string      `json:"score_type,omitempty"`
	Threshold          float64     `json:"threshold,omitempty"`
	Name               string      `json:"name,omitempty"`
	ClassName          string      `json:"class_name,omitempty"`
	Score              float64     `json:"score,omitempty"`
	ScoreBreakdown     interface{} `json:"score_breakdown,omitempty"`
	Reason             string      `json:"reason,omitempty"`
	UsingNativeModel   bool        `json:"using_native_model,omitempty"`
	Success            bool        `json:"success,omitempty"`
	Model              string      `json:"model,omitempty"`
	ModelClient        interface{} `json:"model_client,omitempty"`
	StrictMode         bool        `json:"strict_mode,omitempty"`
	Error              string      `json:"error,omitempty"`
	AdditionalMetadata interface{} `json:"additional_metadata,omitempty"`
	User               string      `json:"user,omitempty"`
	ServerHosted       bool        `json:"server_hosted,omitempty"`

	AdditionalProperties map[string]interface{} `json:"-"`
}

func (m *BaseScorer) UnmarshalJSON(data []byte) error {
	type Alias BaseScorer
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

func (m BaseScorer) MarshalJSON() ([]byte, error) {
	type Alias BaseScorer
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
