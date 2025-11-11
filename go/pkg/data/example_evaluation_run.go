package data

import (
	"time"

	"github.com/JudgmentLabs/judgeval-go/pkg/internal/api/models"
	"github.com/google/uuid"
)

type ExampleEvaluationRun struct {
	*models.ExampleEvaluationRun
	OrganizationId string
}

type ExampleEvaluationRunOptions struct {
	ProjectName    string
	EvalName       string
	Examples       []*Example
	Scorers        []models.ScorerConfig
	Model          string
	OrganizationId string
	TraceId        string
	TraceSpanId    string
}

type ExampleEvaluationRunOption func(*ExampleEvaluationRunOptions)

func WithProjectName(projectName string) ExampleEvaluationRunOption {
	return func(opts *ExampleEvaluationRunOptions) {
		opts.ProjectName = projectName
	}
}

func WithEvalName(evalName string) ExampleEvaluationRunOption {
	return func(opts *ExampleEvaluationRunOptions) {
		opts.EvalName = evalName
	}
}

func WithExamples(examples []*Example) ExampleEvaluationRunOption {
	return func(opts *ExampleEvaluationRunOptions) {
		opts.Examples = examples
	}
}

func WithScorers(scorers []models.ScorerConfig) ExampleEvaluationRunOption {
	return func(opts *ExampleEvaluationRunOptions) {
		opts.Scorers = scorers
	}
}

func WithModel(model string) ExampleEvaluationRunOption {
	return func(opts *ExampleEvaluationRunOptions) {
		opts.Model = model
	}
}

func WithOrganizationId(organizationId string) ExampleEvaluationRunOption {
	return func(opts *ExampleEvaluationRunOptions) {
		opts.OrganizationId = organizationId
	}
}

func WithTraceId(traceId string) ExampleEvaluationRunOption {
	return func(opts *ExampleEvaluationRunOptions) {
		opts.TraceId = traceId
	}
}

func WithTraceSpanId(traceSpanId string) ExampleEvaluationRunOption {
	return func(opts *ExampleEvaluationRunOptions) {
		opts.TraceSpanId = traceSpanId
	}
}

func NewExampleEvaluationRun(options ...ExampleEvaluationRunOption) *ExampleEvaluationRun {
	opts := &ExampleEvaluationRunOptions{}
	for _, option := range options {
		option(opts)
	}

	run := &ExampleEvaluationRun{
		ExampleEvaluationRun: &models.ExampleEvaluationRun{},
	}

	run.Id = uuid.New().String()
	run.CreatedAt = time.Now().UTC().Format(time.RFC3339)

	run.ProjectName = opts.ProjectName
	run.EvalName = opts.EvalName
	run.Model = opts.Model
	run.OrganizationId = opts.OrganizationId
	run.TraceId = opts.TraceId
	run.TraceSpanId = opts.TraceSpanId
	run.CustomScorers = []models.BaseScorer{}
	run.JudgmentScorers = opts.Scorers

	if opts.Examples != nil {
		modelExamples := make([]models.Example, len(opts.Examples))
		for i, example := range opts.Examples {
			modelExamples[i] = *example.Example
		}
		run.Examples = modelExamples
	} else {
		run.Examples = []models.Example{}
	}

	return run
}
