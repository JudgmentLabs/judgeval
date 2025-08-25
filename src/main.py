from judgeval.judges.litellm_judge import LiteLLMJudge
from judgeval.tracer import Tracer


# Initialize the tracer with your project name
judgment = Tracer(project_name="errors")


# # Use the @judgment.observe decorator to trace the tool call
# @judgment.observe(span_type="tool")
# def my_tool():
#     return "Hello world!"


# # Use the @judgment.observe decorator to trace the function
# @judgment.observe(span_type="function")
# def sample_function():
#     tool_called = my_tool()
#     message = "Called my_tool() and got: " + tool_called
#     return message


# if __name__ == "__main__":
#     res = sample_function()
#     print(res)

judge = LiteLLMJudge(model="gpt-4o")

res = judge.generate("Hello.")
print(res)
