from openai import OpenAI
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.tracer import Tracer, wrap
from judgeval.scorers import ClassifierScorer

openai_client = wrap(OpenAI())
tracer = Tracer(project_name="mist-elasticsearch")
judgment_client = JudgmentClient()

@tracer.observe(span_type="function")
def generate_query(query: str) -> dict:
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": """You are an expert at converting natural language queries into Elasticsearch query DSL.
                Generate valid JSON queries that follow Elasticsearch syntax and best practices.
                """
            },
            {
                "role": "user",
                "content": f"Convert this search request to an Elasticsearch query: {query}"
            }
        ]
    )
    return {"generated_query": response.choices[0].message.content.strip()}


if __name__ == "__main__":
    examples = [
        (
            "Find documents about machine learning published in 2023",
            {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"content": "machine learning"}},
                            {"term": {"year": 2023}}
                        ]
                    }
                }
            }
        ),
        (
            "Show me articles by John Smith about neural networks",
            {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"content": "neural networks"}},
                            {"match": {"author": "John Smith"}}
                        ]
                    }
                }
            }
        )
    ]

    inputs = [input_text for input_text, _ in examples]
    outputs = [generate_query(input_text) for input_text, _ in examples]
    SCHEMA = """* content (text)
  * year (integer)
  * author (text)
  * title (text)
  * keywords (keyword)
  * publication_date (date)"""


    text_to_elasticsearch_scorer = ClassifierScorer(
        name="text-to-es",
        threshold=1.0,
        conversation=[
            {"role": "system", "content": """You are an expert at evaluating Elasticsearch queries. You will compare a generated query against a reference query and determine if they are functionally equivalent.

Evaluate the generated Elasticsearch query against the reference query using these criteria:
- Check if the query structure is valid Elasticsearch DSL
- Verify that all important search criteria from the natural query are included 
- Confirm the query would return relevant results
- Verify the query uses the correct field types according to the schema: \n{{input}}

Return "True" if the generated query is functionally equivalent to the reference query, "False" otherwise.

You will receive the reference and generated queries to compare."""},
            {
                "role": "user",
                "content": """Reference query: \n{{expected_output}} \nGenerated query: \n{{actual_output}}"""
            }
        ],
        slug="text-to-es-8891217",
        options={"True": 1.0, "False": 0.0},
    )

    examples = [
        Example(input=SCHEMA, actual_output=str(outputs[0]), expected_output=str(examples[0][1])),
        Example(input=SCHEMA, actual_output=str(outputs[1]), expected_output=str(examples[1][1])),
    ]

    import uuid

    results = judgment_client.run_evaluation(
        examples=examples,
        scorers=[text_to_elasticsearch_scorer],
        model="gpt-4o",
        project_name="text-to-es",
        eval_run_name=str(uuid.uuid4()),
    )

    for result in results:
        print(result)
