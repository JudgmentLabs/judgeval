package data

type APIScorerType string

const (
	PromptScorer         APIScorerType = "Prompt Scorer"
	TracePromptScorer    APIScorerType = "Trace Prompt Scorer"
	Faithfulness         APIScorerType = "Faithfulness"
	AnswerRelevancy      APIScorerType = "Answer Relevancy"
	AnswerCorrectness    APIScorerType = "Answer Correctness"
	InstructionAdherence APIScorerType = "Instruction Adherence"
	ExecutionOrder       APIScorerType = "Execution Order"
	Derailment           APIScorerType = "Derailment"
	ToolOrder            APIScorerType = "Tool Order"
	Classifier           APIScorerType = "Classifier"
	ToolDependency       APIScorerType = "Tool Dependency"
	Custom               APIScorerType = "Custom"
)

func (t APIScorerType) String() string {
	return string(t)
}
