package scorers

import (
	"github.com/JudgmentLabs/judgeval-go/pkg/internal/api/models"
)

type CustomScorer struct {
	name      string
	threshold float64
	model     string
}

func NewCustomScorer(name string, threshold float64, model string) *CustomScorer {
	return &CustomScorer{
		name:      name,
		threshold: threshold,
		model:     model,
	}
}

func (c *CustomScorer) GetName() string {
	return c.name
}

func (c *CustomScorer) GetScorerConfig() models.ScorerConfig {
	return models.ScorerConfig{
		ScoreType:      "Custom",
		Threshold:      c.threshold,
		Name:           c.name,
		Model:          c.model,
		StrictMode:     false,
		RequiredParams: []string{"input", "output"},
		Kwargs:         make(map[string]interface{}),
	}
}
