package v1

import (
	"context"
	"fmt"
	"sync"

	"github.com/JudgmentLabs/judgeval-go/pkg/env"
	"github.com/JudgmentLabs/judgeval-go/v1/internal/api"
	"github.com/JudgmentLabs/judgeval-go/v1/internal/api/models"
)

type PromptScorerFactory struct {
	client  *api.Client
	isTrace bool
	cache   sync.Map
}

type PromptScorerCreateParams struct {
	Name        string
	Prompt      string
	Threshold   float64
	Options     map[string]float64
	Model       *string
	Description *string
}

type PromptScorer struct {
	name        string
	prompt      string
	threshold   float64
	options     map[string]float64
	model       string
	description string
	isTrace     bool
}

func (f *PromptScorerFactory) Get(ctx context.Context, name string) (*PromptScorer, error) {
	cacheKey := f.buildCacheKey(name)

	if cached, ok := f.cache.Load(cacheKey); ok {
		return cached.(*PromptScorer), nil
	}

	req := &models.FetchPromptScorersRequest{
		Names: []string{name},
	}

	resp, err := f.client.FetchScorers(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch prompt scorer '%s': %w", name, err)
	}

	if len(resp.Scorers) == 0 {
		return nil, fmt.Errorf("failed to fetch prompt scorer '%s': not found", name)
	}

	scorerModel := resp.Scorers[0]
	scorerIsTrace := scorerModel.IsTrace

	if scorerIsTrace != f.isTrace {
		expectedType := "PromptScorer"
		actualType := "PromptScorer"
		if f.isTrace {
			expectedType = "TracePromptScorer"
		}
		if scorerIsTrace {
			actualType = "TracePromptScorer"
		}
		return nil, fmt.Errorf("scorer with name %s is a %s, not a %s", name, actualType, expectedType)
	}

	scorer := f.createFromModel(&scorerModel, name)
	f.cache.Store(cacheKey, scorer)

	return scorer, nil
}

func (f *PromptScorerFactory) Create(params PromptScorerCreateParams) (*PromptScorer, error) {
	if params.Name == "" {
		return nil, fmt.Errorf("name is required")
	}
	if params.Prompt == "" {
		return nil, fmt.Errorf("prompt is required")
	}

	model := env.JudgmentDefaultGPTModel
	if params.Model != nil {
		model = *params.Model
	}

	description := ""
	if params.Description != nil {
		description = *params.Description
	}

	return &PromptScorer{
		name:        params.Name,
		prompt:      params.Prompt,
		threshold:   params.Threshold,
		options:     params.Options,
		model:       model,
		description: description,
		isTrace:     f.isTrace,
	}, nil
}

func (f *PromptScorerFactory) createFromModel(model *models.PromptScorer, name string) *PromptScorer {
	options := make(map[string]float64)
	if model.Options != nil {
		if optsMap, ok := model.Options.(map[string]interface{}); ok {
			for k, v := range optsMap {
				if floatVal, ok := v.(float64); ok {
					options[k] = floatVal
				}
			}
		}
	}

	threshold := 0.5
	if model.Threshold != 0 {
		threshold = model.Threshold
	}

	modelName := env.JudgmentDefaultGPTModel
	if model.Model != "" {
		modelName = model.Model
	}

	description := ""
	if model.Description != "" {
		description = model.Description
	}

	return &PromptScorer{
		name:        name,
		prompt:      model.Prompt,
		threshold:   threshold,
		options:     options,
		model:       modelName,
		description: description,
		isTrace:     f.isTrace,
	}
}

func (f *PromptScorerFactory) buildCacheKey(name string) string {
	return fmt.Sprintf("%s:%s:%s", name, f.client.GetAPIKey(), f.client.GetOrganizationID())
}

func (s *PromptScorer) GetName() string {
	return s.name
}

func (s *PromptScorer) GetPrompt() string {
	return s.prompt
}

func (s *PromptScorer) GetThreshold() float64 {
	return s.threshold
}

func (s *PromptScorer) GetOptions() map[string]float64 {
	optsCopy := make(map[string]float64)
	for k, v := range s.options {
		optsCopy[k] = v
	}
	return optsCopy
}

func (s *PromptScorer) GetModel() string {
	return s.model
}

func (s *PromptScorer) GetDescription() string {
	return s.description
}

func (s *PromptScorer) SetThreshold(threshold float64) {
	s.threshold = threshold
}

func (s *PromptScorer) SetPrompt(prompt string) {
	s.prompt = prompt
}

func (s *PromptScorer) SetModel(model string) {
	s.model = model
}

func (s *PromptScorer) SetOptions(options map[string]float64) {
	s.options = make(map[string]float64)
	for k, v := range options {
		s.options[k] = v
	}
}

func (s *PromptScorer) SetDescription(description string) {
	s.description = description
}

func (s *PromptScorer) AppendToPrompt(addition string) {
	s.prompt = s.prompt + addition
}

func (s *PromptScorer) GetScorerConfig() *models.ScorerConfig {
	scoreType := APIScorerTypePromptScorer.String()
	if s.isTrace {
		scoreType = APIScorerTypeTracePromptScorer.String()
	}

	kwargs := map[string]interface{}{
		"prompt": s.prompt,
	}

	if len(s.options) > 0 {
		kwargs["options"] = s.options
	}
	if s.model != "" {
		kwargs["model"] = s.model
	}
	if s.description != "" {
		kwargs["description"] = s.description
	}

	return &models.ScorerConfig{
		ScoreType: scoreType,
		Threshold: s.threshold,
		Name:      s.name,
		Kwargs:    kwargs,
	}
}
