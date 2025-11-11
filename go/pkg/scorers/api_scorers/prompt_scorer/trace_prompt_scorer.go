package prompt_scorer

import (
	"fmt"

	"github.com/JudgmentLabs/judgeval-go/pkg/data"
	"github.com/JudgmentLabs/judgeval-go/pkg/env"
)

type TracePromptScorer struct {
	*BasePromptScorer
}

func GetTracePromptScorer(name string, opts ...ScorerOption) (*TracePromptScorer, error) {
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

	if !scorerConfig.IsTrace {
		return nil, fmt.Errorf("scorer with name %s is not a TracePromptScorer", name)
	}

	scorerOptions := parseScorerOptions(scorerConfig.Options)
	threshold := 0.5
	if scorerConfig.Threshold != 0 {
		threshold = scorerConfig.Threshold
	}

	return &TracePromptScorer{
		BasePromptScorer: NewBasePromptScorer(
			data.TracePromptScorer,
			name,
			scorerConfig.Prompt,
			threshold,
			scorerOptions,
			options.APIKey,
			options.OrganizationID,
		),
	}, nil
}

func (tps *TracePromptScorer) IsTrace() bool {
	return true
}
