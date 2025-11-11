package scorers

import (
	"github.com/JudgmentLabs/judgeval-go/pkg/internal/api/models"
)

type BaseScorer interface {
	GetName() string

	GetScorerConfig() models.ScorerConfig
}
