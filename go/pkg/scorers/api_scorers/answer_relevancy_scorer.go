package api_scorers

import (
	"github.com/JudgmentLabs/judgeval-go/pkg/data"
	"github.com/JudgmentLabs/judgeval-go/pkg/scorers"
)

type AnswerRelevancyScorer struct {
	*scorers.APIScorer
}

func NewAnswerRelevancyScorer(options ...scorers.APIScorerOption) *AnswerRelevancyScorer {

	allOptions := append(options, scorers.WithRequiredParams([]string{
		"input",
		"actual_output",
	}))

	apiScorer := scorers.NewAPIScorer(data.AnswerRelevancy, allOptions...)

	return &AnswerRelevancyScorer{
		APIScorer: apiScorer,
	}
}
