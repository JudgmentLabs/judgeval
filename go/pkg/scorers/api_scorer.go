// Package scorers provides legacy scorer functionality.
//
// Deprecated: Use github.com/JudgmentLabs/judgeval-go/v1 instead.
// This package will be removed in a future version.
package scorers

import (
	"github.com/JudgmentLabs/judgeval-go/pkg/data"
	"github.com/JudgmentLabs/judgeval-go/pkg/internal/api/models"
)

type APIScorer struct {
	*models.BaseScorer
	scoreType      data.APIScorerType
	requiredParams []string
}

type APIScorerOptions struct {
	Threshold      *float64
	Name           *string
	StrictMode     *bool
	Model          *string
	ScoreType      *string
	RequiredParams *[]string
}

type APIScorerOption func(*APIScorerOptions)

func WithThreshold(threshold float64) APIScorerOption {
	return func(opts *APIScorerOptions) {
		opts.Threshold = &threshold
	}
}

func WithName(name string) APIScorerOption {
	return func(opts *APIScorerOptions) {
		opts.Name = &name
	}
}

func WithStrictMode(strictMode bool) APIScorerOption {
	return func(opts *APIScorerOptions) {
		opts.StrictMode = &strictMode
	}
}

func WithModel(model string) APIScorerOption {
	return func(opts *APIScorerOptions) {
		opts.Model = &model
	}
}

func WithScoreType(scoreType string) APIScorerOption {
	return func(opts *APIScorerOptions) {
		opts.ScoreType = &scoreType
	}
}

func WithRequiredParams(requiredParams []string) APIScorerOption {
	return func(opts *APIScorerOptions) {
		opts.RequiredParams = &requiredParams
	}
}

func NewAPIScorer(scoreType data.APIScorerType, options ...APIScorerOption) *APIScorer {
	scorer := &APIScorer{
		BaseScorer: &models.BaseScorer{
			AdditionalProperties: make(map[string]interface{}),
			Threshold:            0.5,
			Name:                 scoreType.String(),
			ScoreType:            scoreType.String(),
		},
		scoreType:      scoreType,
		requiredParams: []string{},
	}

	opts := &APIScorerOptions{}
	for _, option := range options {
		option(opts)
	}

	if opts.Threshold != nil {
		scorer.Threshold = *opts.Threshold
	}
	if opts.Name != nil {
		scorer.Name = *opts.Name
	}
	if opts.StrictMode != nil {
		scorer.StrictMode = *opts.StrictMode
	}
	if opts.Model != nil {
		scorer.Model = *opts.Model
	}
	if opts.ScoreType != nil {
		scorer.ScoreType = *opts.ScoreType
	}
	if opts.RequiredParams != nil {
		scorer.requiredParams = *opts.RequiredParams
	}

	if scorer.StrictMode {
		scorer.Threshold = 1.0
	}

	return scorer
}

func (s *APIScorer) GetName() string {
	return s.Name
}

func (s *APIScorer) GetScorerConfig() models.ScorerConfig {
	kwargs := make(map[string]interface{})
	if s.AdditionalProperties != nil {
		for k, v := range s.AdditionalProperties {
			kwargs[k] = v
		}
	}

	return models.ScorerConfig{
		ScoreType:      s.scoreType.String(),
		Threshold:      s.Threshold,
		Name:           s.Name,
		Model:          s.Model,
		StrictMode:     s.StrictMode,
		RequiredParams: s.requiredParams,
		Kwargs:         kwargs,
	}
}

func (s *APIScorer) SuccessCheck() bool {
	if s.Error != "" {
		return false
	}
	if s.Score == 0 {
		return false
	}
	return s.Score >= s.Threshold
}
