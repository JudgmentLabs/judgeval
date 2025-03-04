from langsmith import wrappers, Client
from pydantic import BaseModel, Field
from openai import OpenAI

client = Client()
openai_client = wrappers.wrap_openai(OpenAI())

# Define mapping for document index
mapping = {
    "mappings": {
        "properties": {
            "content": {"type": "text"},
            "year": {"type": "integer"},
            "author": {"type": "text"},
            "title": {"type": "text"},
            "keywords": {"type": "keyword"},
            "publication_date": {"type": "date"}
        }
    }
}

# Create inputs and reference outputs for Elasticsearch query generation
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

inputs = [{"natural_query": input_text} for input_text, _ in examples]
outputs = [{"elasticsearch_query": output_query} for _, output_query in examples]

# Create dataset in LangSmith
dataset = client.create_dataset(
    dataset_name="Elasticsearch Query Generation Dataset (2)",
    description="Dataset for evaluating text to Elasticsearch query conversion"
)

# Add examples to dataset
client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)

# Define the agent that converts text to Elasticsearch queries
def target(inputs: dict) -> dict:
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
                "content": f"Convert this search request to an Elasticsearch query: {inputs['natural_query']}"
            }
        ]
    )
    return {"generated_query": response.choices[0].message.content.strip()}

# Define evaluation criteria
evaluation_instructions = """Evaluate the generated Elasticsearch query against the reference query:
- Check if the query structure is valid Elasticsearch DSL
- Verify that all important search criteria from the natural query are included
- Confirm the query would return relevant results
- Verify the query uses the correct field types according to the schema:
  * content (text)
  * year (integer)
  * author (text)
  * title (text)
  * keywords (keyword)
  * publication_date (date)
- Return True if the generated query is functionally equivalent to the reference, False otherwise
"""

# Evaluator function
def query_accuracy(outputs: dict, reference_outputs: dict) -> bool:
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": evaluation_instructions},
            {
                "role": "user", 
                "content": f"""Reference Query: {reference_outputs['elasticsearch_query']}
                Generated Query: {outputs['generated_query']}"""
            }
        ]
    )
    parsed = response.choices[0].message.content.strip().lower()
    print(f"PARSED: {parsed}")
    return parsed == "true"


# Run evaluation
experiment_results = client.evaluate(
    target,
    data="Elasticsearch Query Generation Dataset",
    evaluators=[
        query_accuracy,
    ],
    experiment_prefix="elasticsearch-query-generation",
    max_concurrency=2,
)
