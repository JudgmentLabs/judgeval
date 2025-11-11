package v1

import "github.com/JudgmentLabs/judgeval-go/v1/internal/api/models"

type BuiltInScorersFactory struct{}

type FaithfulnessScorerParams struct {
	Threshold  *float64
	Name       *string
	StrictMode *bool
	Model      *string
}

type FaithfulnessScorer struct {
	*apiScorer
	requiredParams []string
}

func (f *BuiltInScorersFactory) Faithfulness(params FaithfulnessScorerParams) *FaithfulnessScorer {
	scorer := newAPIScorer(
		APIScorerTypeFaithfulness,
		getFloat(params.Threshold, 0.5),
		getString(params.Name, ""),
		getBool(params.StrictMode, false),
		getString(params.Model, ""),
		[]string{"context", "actual_output"},
	)

	return &FaithfulnessScorer{
		apiScorer:      scorer,
		requiredParams: []string{"context", "actual_output"},
	}
}

func (s *FaithfulnessScorer) GetScorerConfig() *models.ScorerConfig {
	return s.apiScorer.toScorerConfig(s.requiredParams)
}

type AnswerCorrectnessScorerParams struct {
	Threshold  *float64
	Name       *string
	StrictMode *bool
	Model      *string
}

type AnswerCorrectnessScorer struct {
	*apiScorer
	requiredParams []string
}

func (f *BuiltInScorersFactory) AnswerCorrectness(params AnswerCorrectnessScorerParams) *AnswerCorrectnessScorer {
	scorer := newAPIScorer(
		APIScorerTypeAnswerCorrectness,
		getFloat(params.Threshold, 0.5),
		getString(params.Name, ""),
		getBool(params.StrictMode, false),
		getString(params.Model, ""),
		[]string{"input", "actual_output", "expected_output"},
	)

	return &AnswerCorrectnessScorer{
		apiScorer:      scorer,
		requiredParams: []string{"input", "actual_output", "expected_output"},
	}
}

func (s *AnswerCorrectnessScorer) GetScorerConfig() *models.ScorerConfig {
	return s.apiScorer.toScorerConfig(s.requiredParams)
}

type AnswerRelevancyScorerParams struct {
	Threshold  *float64
	Name       *string
	StrictMode *bool
	Model      *string
}

type AnswerRelevancyScorer struct {
	*apiScorer
	requiredParams []string
}

func (f *BuiltInScorersFactory) AnswerRelevancy(params AnswerRelevancyScorerParams) *AnswerRelevancyScorer {
	scorer := newAPIScorer(
		APIScorerTypeAnswerRelevancy,
		getFloat(params.Threshold, 0.5),
		getString(params.Name, ""),
		getBool(params.StrictMode, false),
		getString(params.Model, ""),
		[]string{"input", "actual_output"},
	)

	return &AnswerRelevancyScorer{
		apiScorer:      scorer,
		requiredParams: []string{"input", "actual_output"},
	}
}

func (s *AnswerRelevancyScorer) GetScorerConfig() *models.ScorerConfig {
	return s.apiScorer.toScorerConfig(s.requiredParams)
}
