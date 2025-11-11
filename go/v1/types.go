package v1

import "github.com/JudgmentLabs/judgeval-go/v1/internal/api/models"

type APIScorerType string

const (
	APIScorerTypePromptScorer      APIScorerType = "Prompt Scorer"
	APIScorerTypeTracePromptScorer APIScorerType = "Trace Prompt Scorer"
	APIScorerTypeFaithfulness      APIScorerType = "Faithfulness"
	APIScorerTypeAnswerRelevancy   APIScorerType = "Answer Relevancy"
	APIScorerTypeAnswerCorrectness APIScorerType = "Answer Correctness"
	APIScorerTypeCustom            APIScorerType = "Custom"
)

func (t APIScorerType) String() string {
	return string(t)
}

type ScorerConfig = models.ScorerConfig

type SerializerFunc func(interface{}) (string, error)
