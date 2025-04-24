import os
import openai
from langfuse import Langfuse
from datetime import datetime, timedelta
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import AnswerRelevancyScorer
from langfuse.decorators import langfuse_context, observe
judgment_client = JudgmentClient()

topic_suggestion = """ You're a world-class journalist, specialized
in figuring out which are the topics that excite people the most.
Your task is to give me 10 suggestions for pop-science topics that the general
public would love to read about. Make sure topics don't repeat.
The output must be a comma-separated list. Generate the list and NOTHING else.
The use of numbers is FORBIDDEN.
"""
 
output = openai.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": topic_suggestion
        }
    ],
    model="gpt-4o",
 
    temperature=1
).choices[0].message.content
 
topics = [item.strip() for item in output.split(",")]
for topic in topics:
    print(topic)

from langfuse.decorators import langfuse_context, observe
 
prompt_template = """
You're an expert science communicator, able to explain complex topics in an
approachable manner. Your task is to respond to the questions of users in an
engaging, informative, and friendly way. Stay factual, and refrain from using
jargon. Your answer should be 4 sentences at max.
Remember, keep it ENGAGING and FUN!
 
Question: {question}
"""
 
@observe()
def explain_concept(topic):
    langfuse_context.update_current_trace(
        name=f"Explanation '{topic}'",
        tags=["ext_eval_pipelines"]
    )
    prompt = prompt_template.format(question=topic)
 
 
    return openai.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o-mini",
        temperature=0.6
    ).choices[0].message.content
 
 
for topic in topics:
    print(f"Input: Please explain to me {topic.lower()}")
    print(f"Answer: {explain_concept(topic)} \n")
 
BATCH_SIZE = 10
TOTAL_TRACES = 50
 
langfuse = Langfuse(
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    host="https://us.cloud.langfuse.com"
)
 
now = datetime.now()
five_am_today = datetime(now.year, now.month, now.day, 5, 0)
five_am_yesterday = five_am_today - timedelta(days=1)
 
traces_batch = langfuse.fetch_traces(page=1,
                                     limit=BATCH_SIZE,
                                     tags="ext_eval_pipelines",
                                     from_timestamp=five_am_yesterday,
                                     to_timestamp=datetime.now()
                                   ).data
 
print(f"Traces in first batch: {len(traces_batch)}")


template_tone_eval = """
You're an expert in human emotional intelligence. You can identify with ease the
 tone in human-written text. Your task is to identify the tones present in a
 piece of <text/> with precission. Your output is a comma separated list of three
 tones. PRINT THE LIST ALONE, NOTHING ELSE.
 
<possible_tones>
neutral, confident, joyful, optimistic, friendly, urgent, analytical, respectful
</possible_tones>
 
<example_1>
Input: Citizen science plays a crucial role in research by involving everyday
people in scientific projects. This collaboration allows researchers to collect
vast amounts of data that would be impossible to gather on their own. Citizen
scientists contribute valuable observations and insights that can lead to new
discoveries and advancements in various fields. By participating in citizen
science projects, individuals can actively contribute to scientific research
and make a meaningful impact on our understanding of the world around us.
 
Output: respectful,optimistic,confident
</example_1>
 
<example_2>
Input: Bionics is a field that combines biology and engineering to create
devices that can enhance human abilities. By merging humans and machines,
bionics aims to improve quality of life for individuals with disabilities
or enhance performance for others. These technologies often mimic natural
processes in the body to create seamless integration. Overall, bionics holds
great potential for revolutionizing healthcare and technology in the future.
 
Output: optimistic,confident,analytical
</example_2>
 
<example_3>
Input: Social media can have both positive and negative impacts on mental
health. On the positive side, it can help people connect, share experiences,
and find support. However, excessive use of social media can also lead to
feelings of inadequacy, loneliness, and anxiety. It's important to find a
balance and be mindful of how social media affects your mental well-being.
Remember, it's okay to take breaks and prioritize your mental health.
 
Output: friendly,neutral,respectful
</example_3>
 
<text>
{text}
</text>
"""
 
 
test_tone_score = openai.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": template_tone_eval.format(
                text=traces_batch[1].output),
        }
    ],
    model="gpt-4o",
 
    temperature=0
).choices[0].message.content
print(f"User query: {traces_batch[1].input['args'][0]}")
print(f"Model answer: {traces_batch[1].output}")
print(f"Dominant tones: {test_tone_score}")


def use_judgment_scorer(input, expected, output):
    print(f"input: {input}\n expected: {expected}\n output: {output}")
    example = Example(
        input=input,
        expected_output=expected,
        actual_output=output
        )

    scorer = AnswerRelevancyScorer(threshold=0.5)
    results = judgment_client.run_evaluation(
        project_name="Travel Agent",
        eval_run_name="Answer Relevancy",
        examples=[example],
        scorers=[scorer],
        override=True,
        model="gpt-4o",
    )
    print(results)
    return results[0].scorers_data[0].score

for trace in traces_batch:
    langfuse.score(
        trace_id=trace.id,
        name="hallucination_marker",
        value=use_judgment_scorer(trace.input['args'][0], trace.output, trace.output)
    )