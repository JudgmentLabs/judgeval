package v1

import "github.com/JudgmentLabs/judgeval-go/v1/internal/api"

type ScorersFactory struct {
	client            *api.Client
	BuiltIn           *BuiltInScorersFactory
	PromptScorer      *PromptScorerFactory
	TracePromptScorer *PromptScorerFactory
	CustomScorer      *CustomScorerFactory
}

func newScorersFactory(client *api.Client) *ScorersFactory {
	return &ScorersFactory{
		client:            client,
		BuiltIn:           &BuiltInScorersFactory{},
		PromptScorer:      &PromptScorerFactory{client: client, isTrace: false},
		TracePromptScorer: &PromptScorerFactory{client: client, isTrace: true},
		CustomScorer:      &CustomScorerFactory{},
	}
}
