package models

import (
	"encoding/json"
)

type ScorerConfig struct {
	ScoreType      string      `json:"score_type,omitempty"`
	Name           string      `json:"name,omitempty"`
	Threshold      float64     `json:"threshold,omitempty"`
	Model          string      `json:"model,omitempty"`
	StrictMode     bool        `json:"strict_mode,omitempty"`
	RequiredParams []string    `json:"required_params,omitempty"`
	Kwargs         interface{} `json:"kwargs,omitempty"`

	AdditionalProperties map[string]interface{} `json:"-"`
}

func (m *ScorerConfig) UnmarshalJSON(data []byte) error {
	type Alias ScorerConfig
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

func (m ScorerConfig) MarshalJSON() ([]byte, error) {
	type Alias ScorerConfig
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
