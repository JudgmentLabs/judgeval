import os
import sys
from typing import Any, Dict, List, Optional, Union, Annotated
from typing_extensions import TypedDict
import json
import asyncio
import time
import random
from uuid import uuid4

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, ToolException, BaseTool, StructuredTool
from pydantic import BaseModel, Field, ConfigDict
from elasticsearch import ApiError, NotFoundError

from judgeval.common.tracer import Tracer, wrap
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer, AnswerCorrectnessScorer

from typing import Callable, TypeVar, ParamSpec
import elasticsearch_client

# Initialize OpenAI client
openai_client = ChatOpenAI(model="gpt-3.5-turbo")

# Initialize Judgment Tracer
judgment = Tracer(
    api_key=os.getenv("JUDGMENT_API_KEY", ""),
    project_name="text_to_es"
)

# Define TextToESState TypedDict
class TextToESState(TypedDict):
    messages: List[Any]
    user_query: str
    entities: Optional[List[Dict[str, str]]]
    validated_entities: Optional[List[Dict[str, str]]]
    selected_index: Optional[str]
    index_mappings: Optional[Dict[str, Any]]
    essential_fields: Optional[Dict[str, Any]]
    field_values: Optional[Dict[str, Any]]
    selected_field_values: Optional[Dict[str, Any]]
    es_query: Optional[Dict[str, Any]]
    processed_query: Optional[Dict[str, Any]]
    query_results: Optional[Dict[str, Any]]
    final_response: Optional[str]
    error: Optional[str]

def safe_json_extract(content: str) -> Any:
    """Extract JSON from model response."""
    try:
        # Check if the entire content is valid JSON
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON using simple pattern matching
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        # Return empty dict if no valid JSON found
        return {}

@judgment.observe(span_type="NLP")
def extract_entities(state: Dict[str, Any]) -> TextToESState:
    """Extract named entities from the user query."""
    # Add sleep between 300ms - 1.0s
    time.sleep(random.uniform(0.3, 1.0))
    
    user_query = state["user_query"]
    
    # For this test, we'll provide valid entities
    valid_entities = [
        {"type": "device_type", "name": "access points"},
        {"type": "status", "name": "disconnected"}
    ]
    
    # For entity extractor evaluation
    judgment.get_current_trace().async_evaluate(
        scorers=[
            AnswerCorrectnessScorer(threshold=0.7),
            FaithfulnessScorer(threshold=0.7),
        ],
        input=f"Extract entities from: '{user_query}'",
        actual_output=json.dumps(valid_entities),
        expected_output=json.dumps([
            {"type": "device_type", "name": "access points"},
            {"type": "status", "name": "disconnected"}
        ]),
        retrieval_context=[],
        model="gpt-4o-mini",
        log_results=True,
        additional_metadata={"query": user_query}
    )
    
    return {**state, "entities": valid_entities}

@judgment.observe(span_type="NLP")
def validate_entities(state: Dict[str, Any]) -> TextToESState:
    """Validate entities from the extraction step."""
    # Add sleep between 300ms - 1.0s
    time.sleep(random.uniform(0.3, 1.0))
    
    entities = state.get("entities", [])
    
    # Simple validation to pass through to the next step
    validated_entities = entities
    
    return {**state, "validated_entities": validated_entities}

@judgment.observe(span_type="Search")
def select_index(state: Dict[str, Any]) -> TextToESState:
    """Select the appropriate Elasticsearch index."""
    # Add sleep between 300ms - 1.0s
    time.sleep(random.uniform(0.3, 1.0))
    
    user_query = state["user_query"]
    validated_entities = state.get("validated_entities", [])
    
    # For this test, provide a valid index
    selected_index = "devices"
    
    # Get the index mappings and essential fields
    index_mappings = elasticsearch_client.ES_INDEXES[selected_index]["mappings"]
    essential_fields = elasticsearch_client.ES_INDEXES[selected_index]["essential_fields"]
    
    # For index selection evaluation
    judgment.get_current_trace().async_evaluate(
        scorers=[
            AnswerCorrectnessScorer(threshold=0.7),
            FaithfulnessScorer(threshold=0.7)
        ],
        input=f"Select appropriate Elasticsearch index for query: '{user_query}'",
        actual_output=selected_index,
        expected_output="devices",
        retrieval_context=[],
        model="gpt-4o-mini",
        log_results=True,
        additional_metadata={"query": user_query}
    )
    
    return {**state, "selected_index": selected_index, "index_mappings": index_mappings, "essential_fields": essential_fields}

@judgment.observe(span_type="NLP")
def get_field_values(state: Dict[str, Any]) -> TextToESState:
    """Get field values for query generation."""
    # Add sleep between 300ms - 1.0s
    time.sleep(random.uniform(0.3, 1.0))
    
    # Provide valid field values
    field_values = {
        "status": ["connected", "disconnected", "provisioning"],
        "device_type": ["access_point", "switch", "router"]
    }
    
    selected_field_values = {
        "status": "disconnected",
        "device_type": "access_point"
    }
    
    return {**state, "field_values": field_values, "selected_field_values": selected_field_values}

@judgment.observe(span_type="Query")
def generate_query(state: Dict[str, Any]) -> TextToESState:
    """Generate an Elasticsearch query."""
    # Add sleep between 300ms - 1.0s
    time.sleep(random.uniform(0.3, 1.0))
    
    user_query = state["user_query"]
    validated_entities = state.get("validated_entities", [])
    selected_index = state["selected_index"]
    
    # Create a valid query for this test
    valid_query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"status": "disconnected"}},
                    {"term": {"device_type": "access_point"}}
                ]
            }
        },
        "size": 10
    }
    
    return {**state, "es_query": valid_query}

@judgment.observe(span_type="Query")
def process_query(state: Dict[str, Any]) -> TextToESState:
    """Process the Elasticsearch query."""
    # Add sleep between 300ms - 1.0s
    time.sleep(random.uniform(0.3, 1.0))
    
    es_query = state.get("es_query", {})
    
    # Simple pass-through in this test case
    processed_query = es_query
    
    return {**state, "processed_query": processed_query}

@judgment.observe(span_type="Database")
def execute_query(state: Dict[str, Any]) -> TextToESState:
    """Execute the Elasticsearch query and return mock results."""
    # Add sleep between 300ms - 1.0s
    time.sleep(random.uniform(0.3, 1.0))
    
    processed_query = state.get("processed_query", {})
    
    # Mock query results showing disconnected access points
    mock_query_results = {
        "took": 3,
        "timed_out": False,
        "_shards": {
            "total": 1,
            "successful": 1,
            "skipped": 0,
            "failed": 0
        },
        "hits": {
            "total": {"value": 3, "relation": "eq"},
            "max_score": 1.0,
            "hits": [
                {
                    "_index": "devices",
                    "_id": "ap-001",
                    "_score": 1.0,
                    "_source": {
                        "id": "ap-001",
                        "name": "AP-Building-A-Floor1",
                        "device_type": "access_point",
                        "model": "AP43",
                        "status": "disconnected",
                        "last_seen": "2023-05-15T14:23:15Z",
                        "firmware": "6.1.2",
                        "ip_address": "10.0.1.101",
                        "mac_address": "00:11:22:33:44:55",
                        "location": "Building A, Floor 1"
                    }
                },
                {
                    "_index": "devices",
                    "_id": "ap-002",
                    "_score": 1.0,
                    "_source": {
                        "id": "ap-002",
                        "name": "AP-Building-B-Floor2",
                        "device_type": "access_point",
                        "model": "AP43",
                        "status": "disconnected",
                        "last_seen": "2023-05-16T09:45:30Z",
                        "firmware": "6.1.2",
                        "ip_address": "10.0.2.102",
                        "mac_address": "00:11:22:33:44:66",
                        "location": "Building B, Floor 2"
                    }
                },
                {
                    "_index": "devices",
                    "_id": "ap-003",
                    "_score": 1.0,
                    "_source": {
                        "id": "ap-003",
                        "name": "AP-Building-C-Floor1",
                        "device_type": "access_point",
                        "model": "AP32",
                        "status": "disconnected",
                        "last_seen": "2023-05-16T10:15:45Z",
                        "firmware": "6.0.9",
                        "ip_address": "10.0.3.103",
                        "mac_address": "00:11:22:33:44:77",
                        "location": "Building C, Floor 1"
                    }
                }
            ]
        }
    }
    
    return {**state, "query_results": mock_query_results}

@judgment.observe(span_type="Response")
def format_response(state: Dict[str, Any]) -> TextToESState:
    """Format the response for the user with intentional unfaithfulness."""
    # Add sleep between 300ms - 1.0s
    time.sleep(random.uniform(0.3, 1.0))
    
    user_query = state["user_query"]
    query_results = state.get("query_results", {})
    
    # Intentionally generate an unfaithful response that contradicts the actual query results
    unfaithful_response = """
    I found 5 connected access points in our system:
    
    1. AP-Building-D-Floor3 (AP-005): Last connected 2 hours ago, running firmware 7.0.1
    2. AP-Building-E-Floor2 (AP-008): Currently connected, running firmware 7.0.1
    3. AP-Building-F-Floor1 (AP-012): Connected, running firmware 6.9.5
    4. AP-Building-G-Floor4 (AP-015): Connected, running firmware 7.0.1
    5. AP-Building-H-Floor2 (AP-019): Connected, running firmware 6.9.5
    
    All these access points are currently online and serving clients. Would you like more details about any specific access point?
    """
    
    # Evaluate the unfaithful response against expected faithful response
    judgment.get_current_trace().async_evaluate(
        scorers=[
            FaithfulnessScorer(threshold=0.7),
            AnswerRelevancyScorer(threshold=0.7)
        ],
        input=f"Format response for query results on: '{user_query}'",
        actual_output=unfaithful_response,
        expected_output="""
        I found 3 disconnected access points:
        
        1. AP-Building-A-Floor1 (ap-001): Last seen on May 15, 2023, running firmware 6.1.2
        2. AP-Building-B-Floor2 (ap-002): Last seen on May 16, 2023, running firmware 6.1.2
        3. AP-Building-C-Floor1 (ap-003): Last seen on May 16, 2023, running firmware 6.0.9
        
        All of these access points are currently disconnected. Would you like more details about any specific access point?
        """,
        retrieval_context=[
            # Provide the relevant query results as context
            json.dumps(query_results.get("hits", {}).get("hits", []), indent=2)
        ],
        model="gpt-4o-mini",
        log_results=True,
        additional_metadata={"query": user_query}
    )
    
    # Check faithfulness of response against query results
    hits = query_results.get("hits", {}).get("hits", [])
    if not hits:
        unfaithful_response = "I found results that don't exist in the database."
    
    # Get the first result for a basic check
    try:
        first_hit_status = hits[0]["_source"]["status"]
        # Detect faithfulness issue (says "connected" when status is "disconnected")
        if "connected access points" in unfaithful_response and first_hit_status == "disconnected":
            error_msg = "Response formatting error: Response incorrectly states the access points are connected when they are disconnected."
            return {**state, "error": error_msg, "final_response": unfaithful_response}
        
        # Check hit count faithfulness
        actual_count = len(hits)
        if "5 connected" in unfaithful_response and actual_count != 5:
            error_msg = f"Response formatting error: Response incorrectly states 5 access points when there are actually {actual_count}."
            return {**state, "error": error_msg, "final_response": unfaithful_response}
    except (IndexError, KeyError):
        error_msg = "Response formatting error: Unable to validate response against query results."
        return {**state, "error": error_msg, "final_response": unfaithful_response}
    
    # If we get here, provide the unfaithful response without setting error
    # (shouldn't happen due to the intentional errors above)
    return {**state, "final_response": unfaithful_response}

@judgment.observe(span_type="Error")
def handle_error(state: Dict[str, Any]) -> TextToESState:
    """Handle errors in the pipeline."""
    # Add sleep between 300ms - 1.0s
    time.sleep(random.uniform(0.3, 1.0))
    
    error = state.get("error", "Unknown error")
    
    return {**state, "final_response": f"Error: {error}"}

def has_error(state: TextToESState) -> str:
    """Route to error handler if state has an error."""
    if state.get("error"):
        return "error"
    return "continue"

@judgment.observe(span_type="function")
async def fail_format_response_pipeline():
    """Main pipeline function that demonstrates failure in response formatting."""
    # Add sleep between 300ms - 1.0s before starting the pipeline
    time.sleep(random.uniform(0.3, 1.0))
    
    # Initialize the graph
    workflow = StateGraph(TextToESState)
    
    # Add nodes
    workflow.add_node("extract_entities", extract_entities)
    workflow.add_node("validate_entities", validate_entities)
    workflow.add_node("select_index", select_index)
    workflow.add_node("get_field_values", get_field_values)
    workflow.add_node("generate_query", generate_query)
    workflow.add_node("process_query", process_query)
    workflow.add_node("execute_query", execute_query)
    workflow.add_node("format_response", format_response)
    workflow.add_node("handle_error", handle_error)
    
    workflow.set_entry_point("extract_entities")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "extract_entities",
        has_error,
        {
            "error": "handle_error",
            "continue": "validate_entities"
        }
    )
    
    workflow.add_conditional_edges(
        "validate_entities",
        has_error,
        {
            "error": "handle_error",
            "continue": "select_index"
        }
    )
    
    workflow.add_conditional_edges(
        "select_index",
        has_error,
        {
            "error": "handle_error",
            "continue": "get_field_values"
        }
    )
    
    workflow.add_conditional_edges(
        "get_field_values",
        has_error,
        {
            "error": "handle_error",
            "continue": "generate_query"
        }
    )
    
    workflow.add_conditional_edges(
        "generate_query",
        has_error,
        {
            "error": "handle_error",
            "continue": "process_query"
        }
    )
    
    workflow.add_conditional_edges(
        "process_query",
        has_error,
        {
            "error": "handle_error",
            "continue": "execute_query"
        }
    )
    
    workflow.add_conditional_edges(
        "execute_query",
        has_error,
        {
            "error": "handle_error",
            "continue": "format_response"
        }
    )
    
    workflow.add_conditional_edges(
        "format_response",
        has_error,
        {
            "error": "handle_error",
            "continue": "format_response"  # Dead end as we should hit error
        }
    )
    
    # Compile the graph
    app = workflow.compile()
    
    # Initial state
    initial_state = {
        "messages": [],
        "user_query": "Show me all disconnected access points",
        "entities": None,
        "validated_entities": None,
        "selected_index": None,
        "index_mappings": None,
        "essential_fields": None,
        "field_values": None,
        "selected_field_values": None,
        "es_query": None,
        "processed_query": None,
        "query_results": None,
        "final_response": None,
        "error": None
    }
    
    # Run the pipeline
    result = await app.ainvoke(initial_state)
    print(f"Pipeline result: {result.get('final_response', 'No response')}")
    return result

if __name__ == "__main__":
    asyncio.run(fail_format_response_pipeline()) 