package prompt_scorer

import (
	"fmt"

	"github.com/JudgmentLabs/judgeval-go/pkg/data"
	"github.com/JudgmentLabs/judgeval-go/pkg/env"
)

type PromptScorer struct {
	*BasePromptScorer
}

func GetPromptScorer(name string, opts ...ScorerOption) (*PromptScorer, error) {
	options := &ScorerOptions{
		APIURL:         env.JudgmentAPIURL,
		APIKey:         env.JudgmentAPIKey,
		OrganizationID: env.JudgmentOrgID,
	}

	for _, opt := range opts {
		opt(options)
	}

	scorerConfig, err := FetchPromptScorer(name, options.APIURL, options.APIKey, options.OrganizationID)
	if err != nil {
		return nil, err
	}

	if scorerConfig.IsTrace {
		return nil, fmt.Errorf("scorer with name %s is not a PromptScorer", name)
	}

	scorerOptions := parseScorerOptions(scorerConfig.Options)
	threshold := 0.5
	if scorerConfig.Threshold != 0 {
		threshold = scorerConfig.Threshold
	}

	return &PromptScorer{
		BasePromptScorer: NewBasePromptScorer(
			data.PromptScorer,
			name,
			scorerConfig.Prompt,
			threshold,
			scorerOptions,
			options.APIKey,
			options.OrganizationID,
		),
	}, nil
}

func (ps *PromptScorer) IsTrace() bool {
	return false
}
