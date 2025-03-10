# from judgeval.common.tracer import Tracer, wrap
# from openai import OpenAI

# client = wrap(OpenAI())
# judgment = Tracer(project_name="my_project")

# @judgment.observe(span_type="tool")
# def my_tool():
#     return "Hello world!"

# @judgment.observe(span_type="function")
# def main():
#     res = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": f"{my_tool()}"}]
#     )
#     return res.choices[0].message.content

# if __name__ == "__main__":
#     with judgment.trace(name="my_workflow") as trace:
#         main()

from judgeval.common.tracer import Tracer, wrap
from openai import OpenAI
from together import Together
from judgeval.scorers import AnswerRelevancyScorer
judgment = Tracer(project_name="my_project")
together_client = wrap(Together())

@judgment.observe(span_type="tool")
def my_tool():
    return "Hello world!"

def main():
    with judgment.trace(name="my_workflow") as trace:
        res = together_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": f"{my_tool()}"}]
        )
        
        judgment.get_current_trace().async_evaluate(
            scorers=[AnswerRelevancyScorer(threshold=0.5)],
            input="Hello world!",
            actual_output=res.choices[0].message.content,
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        )
    
    trace.print()  # prints the state of the trace to console
    trace.save()  # saves the current state of the trace to the Judgment platform

    return res.choices[0].message.content

if __name__ == "__main__":
    main()
