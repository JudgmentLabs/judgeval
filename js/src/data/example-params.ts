export enum ExampleParams {
  INPUT = "input",
  ACTUAL_OUTPUT = "actual_output",
  EXPECTED_OUTPUT = "expected_output",
  CONTEXT = "context",
  RETRIEVAL_CONTEXT = "retrieval_context",
  TOOLS_CALLED = "tools_called",
  EXPECTED_TOOLS = "expected_tools",
  ADDITIONAL_METADATA = "additional_metadata",
}

export type ExampleParamKeys = keyof typeof ExampleParams;
export type ExampleParamValues = `${ExampleParams}`;
