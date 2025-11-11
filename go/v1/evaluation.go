package v1

import "github.com/JudgmentLabs/judgeval-go/v1/internal/api"

type EvaluationFactory struct {
	client *api.Client
}

type EvaluationCreateParams struct {
}

type Evaluation struct {
	client *api.Client
}

func (f *EvaluationFactory) Create(params EvaluationCreateParams) *Evaluation {
	return &Evaluation{
		client: f.client,
	}
}
