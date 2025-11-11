package data

import (
	"github.com/JudgmentLabs/judgeval-go/pkg/internal/api/models"
)

type TraceEvaluationRun struct {
	*models.TraceEvaluationRun
}

type TraceEvaluationRunOptions struct {
	ProjectName string
	EvalName    string
	Scorer      models.ScorerConfig
	Model       string
	TraceId     string
	SpanId      string
}

func NewTraceEvaluationRunWithOptions(opts TraceEvaluationRunOptions) *TraceEvaluationRun {
	run := &TraceEvaluationRun{
		TraceEvaluationRun: &models.TraceEvaluationRun{},
	}

	run.ProjectName = opts.ProjectName
	run.EvalName = opts.EvalName
	run.Model = opts.Model
	run.CustomScorers = []models.BaseScorer{}
	run.JudgmentScorers = []models.ScorerConfig{opts.Scorer}
	run.TraceAndSpanIds = [][]interface{}{{opts.TraceId, opts.SpanId}}

	return run
}
