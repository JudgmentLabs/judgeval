package models

import (
	"encoding/json"
)

type ResolveProjectNameRequest struct {
	ProjectName string `json:"project_name,omitempty"`

	AdditionalProperties map[string]interface{} `json:"-"`
}

func (m *ResolveProjectNameRequest) UnmarshalJSON(data []byte) error {
	type Alias ResolveProjectNameRequest
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

func (m ResolveProjectNameRequest) MarshalJSON() ([]byte, error) {
	type Alias ResolveProjectNameRequest
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
