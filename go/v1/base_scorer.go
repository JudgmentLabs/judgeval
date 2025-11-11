package v1

import (
	"maps"

	"github.com/JudgmentLabs/judgeval-go/v1/internal/api/models"
)

type BaseScorer interface {
	GetName() string
	GetScorerConfig() *models.ScorerConfig
}

type apiScorer struct {
	scoreType       APIScorerType
	threshold       float64
	name            string
	strictMode      bool
	model           string
	requiredParams  []string
	additionalProps map[string]interface{}
}

func newAPIScorer(scoreType APIScorerType, threshold float64, name string, strictMode bool, model string, requiredParams []string) *apiScorer {
	finalThreshold := threshold
	if strictMode {
		finalThreshold = 1.0
	}

	finalName := name
	if finalName == "" {
		finalName = scoreType.String()
	}

	return &apiScorer{
		scoreType:       scoreType,
		threshold:       finalThreshold,
		name:            finalName,
		strictMode:      strictMode,
		model:           model,
		requiredParams:  requiredParams,
		additionalProps: make(map[string]interface{}),
	}
}

func (s *apiScorer) GetName() string {
	return s.name
}

func (s *apiScorer) toScorerConfig(requiredParams []string) *models.ScorerConfig {
	kwargs := make(map[string]interface{})
	maps.Copy(kwargs, s.additionalProps)

	return &models.ScorerConfig{
		ScoreType:      s.scoreType.String(),
		Threshold:      s.threshold,
		Name:           s.name,
		StrictMode:     s.strictMode,
		RequiredParams: requiredParams,
		Kwargs:         kwargs,
		Model:          s.model,
	}
}
