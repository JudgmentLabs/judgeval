package v1

import (
	"errors"

	"github.com/JudgmentLabs/judgeval-go/pkg/env"
	"github.com/JudgmentLabs/judgeval-go/v1/internal/api"
)

type Judgeval struct {
	apiClient  *api.Client
	Tracer     *TracerFactory
	Scorers    *ScorersFactory
	Evaluation *EvaluationFactory
}

func NewJudgeval(opts ...Option) (*Judgeval, error) {
	cfg := &clientConfig{
		apiKey: env.JudgmentAPIKey,
		orgID:  env.JudgmentOrgID,
		apiURL: env.JudgmentAPIURL,
	}

	for _, opt := range opts {
		opt.apply(cfg)
	}

	if cfg.apiKey == "" {
		return nil, errors.New("API key is required: set JUDGMENT_API_KEY environment variable or use WithAPIKey option")
	}
	if cfg.orgID == "" {
		return nil, errors.New("organization ID is required: set JUDGMENT_ORG_ID environment variable or use WithOrganizationID option")
	}
	if cfg.apiURL == "" {
		return nil, errors.New("API URL is required: set JUDGMENT_API_URL environment variable or use WithAPIURL option")
	}

	apiClient := api.NewClient(cfg.apiURL, cfg.apiKey, cfg.orgID)

	return &Judgeval{
		apiClient:  apiClient,
		Tracer:     &TracerFactory{client: apiClient},
		Scorers:    newScorersFactory(apiClient),
		Evaluation: &EvaluationFactory{client: apiClient},
	}, nil
}
