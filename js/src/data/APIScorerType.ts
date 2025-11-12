export const APIScorerType = {
  PROMPT_SCORER: "Prompt Scorer",
  TRACE_PROMPT_SCORER: "Trace Prompt Scorer",
  FAITHFULNESS: "Faithfulness",
  ANSWER_RELEVANCY: "Answer Relevancy",
  ANSWER_CORRECTNESS: "Answer Correctness",
  INSTRUCTION_ADHERENCE: "Instruction Adherence",
  EXECUTION_ORDER: "Execution Order",
  DERAILMENT: "Derailment",
  CUSTOM: "Custom",
} as const;

export type APIScorerType = (typeof APIScorerType)[keyof typeof APIScorerType];
