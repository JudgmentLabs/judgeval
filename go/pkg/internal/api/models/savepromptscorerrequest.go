package models

import (
	"encoding/json"
)

type SavePromptScorerRequest struct {
	Name        string      `json:"name,omitempty"`
	Prompt      string      `json:"prompt,omitempty"`
	Threshold   float64     `json:"threshold,omitempty"`
	Model       string      `json:"model,omitempty"`
	IsTrace     bool        `json:"is_trace,omitempty"`
	Options     interface{} `json:"options,omitempty"`
	Description string      `json:"description,omitempty"`

	AdditionalProperties map[string]interface{} `json:"-"`
}

func (m *SavePromptScorerRequest) UnmarshalJSON(data []byte) error {
	type Alias SavePromptScorerRequest
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

func (m SavePromptScorerRequest) MarshalJSON() ([]byte, error) {
	type Alias SavePromptScorerRequest
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
