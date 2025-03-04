from openai import OpenAI
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.tracer import Tracer, wrap
from judgeval.scorers import ClassifierScorer

openai_client = wrap(OpenAI())
tracer = Tracer(project_name="mist-elasticsearch")
judgment_client = JudgmentClient()

@tracer.observe(span_type="function")
def generate_correct_query(query: str) -> dict:
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": """You are an expert at converting natural language queries into Elasticsearch query DSL.
                Generate valid JSON queries that follow Elasticsearch syntax and best practices.
                The document index has the following fields:
                - content (text): The main content of the document
                - year (integer): Publication year
                - author (text): Document author
                - title (text): Document title
                - keywords (keyword): Document keywords
                - publication_date (date): Full publication date"""
            },
            {
                "role": "user",
                "content": f"Convert this search request to an Elasticsearch query: {query}"
            }
        ]
    )
    return {"generated_query": response.choices[0].message.content.strip()}

@tracer.observe(span_type="function")
def generate_bad_query(query: str) -> dict:
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": """You are an expert at converting natural language queries into Elasticsearch query DSL.
                Generate valid JSON queries that follow Elasticsearch syntax and best practices.
                The document index has the following fields:
                - body_text (keyword): The main content of the document
                - publication_year (string): Publication year
                - creator (keyword): Document author
                - document_name (keyword): Document title
                - categories (text): Document keywords
                - timestamp (string): Full publication date"""
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
    correct_outputs = [generate_correct_query(input_text) for input_text, _ in examples]
    bad_outputs = [generate_bad_query(input_text) for input_text, _ in examples]
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

    # Create examples for correct queries
    correct_examples = [
        Example(input=SCHEMA, actual_output=str(correct_outputs[0]), expected_output=str(examples[0][1])),
        Example(input=SCHEMA, actual_output=str(correct_outputs[1]), expected_output=str(examples[1][1])),
    ]

    # Create examples for bad queries
    bad_examples = [
        Example(input=SCHEMA, actual_output=str(bad_outputs[0]), expected_output=str(examples[0][1])),
        Example(input=SCHEMA, actual_output=str(bad_outputs[1]), expected_output=str(examples[1][1])),
    ]

    import uuid

    # Evaluate correct queries
    print("Evaluating queries with proper document index:")
    correct_results = judgment_client.run_evaluation(
        examples=correct_examples,
        scorers=[text_to_elasticsearch_scorer],
        model="gpt-4o",
        project_name="text-to-es-correct",
        eval_run_name=f"correct-{str(uuid.uuid4())}",
    )

    for result in correct_results:
        print(result)
    
    # Evaluate bad queries
    print("\nEvaluating queries without proper document index:")
    bad_results = judgment_client.run_evaluation(
        examples=bad_examples,
        scorers=[text_to_elasticsearch_scorer],
        model="gpt-4o",
        project_name="text-to-es-bad",
        eval_run_name=f"bad-{str(uuid.uuid4())}",
    )

    for result in bad_results:
        print(result)
