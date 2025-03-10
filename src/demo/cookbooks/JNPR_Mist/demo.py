import os
import sys
from typing import Any, Dict, List, Optional, Union, Annotated
from typing_extensions import TypedDict
import json
import asyncio
from uuid import uuid4

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, ToolException, BaseTool, StructuredTool
from pydantic import BaseModel, Field, ConfigDict
from elasticsearch import ApiError, NotFoundError

from judgeval.common.tracer import Tracer, JudgevalCallbackHandler
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer, AnswerCorrectnessScorer

from typing import Callable, TypeVar, ParamSpec

# Import elasticsearch functionality directly
import elasticsearch_client

# Import prompts
from prompts import (
    OPENAI_MODEL,
    ENTITY_EXTRACTION_SYSTEM,
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_VALIDATION_SYSTEM,
    ENTITY_VALIDATION_PROMPT,
    FIELD_EXTRACTION_SYSTEM,
    FIELD_VALUES_PROMPT,
    INDEX_SELECTION_SYSTEM,
    INDEX_SELECTION_PROMPT,
    QUERY_GENERATOR_SYSTEM,
    QUERY_GENERATION_PROMPT,
    RESPONSE_FORMATTER_SYSTEM,
    RESPONSE_FORMATTING_PROMPT,
    ENTITY_FIELD_MAPPINGS
)

# Initialize the Elasticsearch client
es_client = elasticsearch_client.get_elasticsearch_client()

# Helper function to check if Elasticsearch is available
def check_elasticsearch_connection() -> bool:
    """Check if Elasticsearch is available and responsive."""
    try:
        return es_client.ping()
    except Exception:
        return False

P = ParamSpec("P")
R = TypeVar("R")

load_dotenv()

# Update OpenAI model name to a valid one
OPENAI_MODEL = "gpt-4o-mini"

# Initialize clients
openai_client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=OPENAI_MODEL)
judgment = Tracer(
    api_key=os.getenv("JUDGMENT_API_KEY"),
    project_name="text_to_es"
)

# Define our state type
class TextToESState(TypedDict):
    messages: List[BaseMessage]
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

class IDMasker:
    """Helper class to mask and unmask sensitive IDs in queries and responses."""
    
    def __init__(self):
        self.masked_values = {}
        self.next_id = 1
    
    def mask_id(self, id_type: str, real_value: str) -> str:
        """
        Replace a real ID with a masked placeholder.
        
        Args:
            id_type: Type of ID (e.g., 'org', 'user', 'device')
            real_value: The actual ID value to mask
            
        Returns:
            A masked placeholder string
        """
        if not real_value:
            return real_value
            
        mask_key = f"{id_type}:{real_value}"
        
        # If we've already masked this value, return the existing mask
        if mask_key in self.masked_values:
            return self.masked_values[mask_key]
            
        # Create a new masked value
        masked_value = f"MASKED_{id_type.upper()}_ID_{self.next_id}"
        self.masked_values[mask_key] = masked_value
        self.next_id += 1
        
        return masked_value
    
    def unmask_id(self, masked_value: str) -> str:
        """
        Replace a masked placeholder with its real value.
        
        Args:
            masked_value: The masked placeholder to unmask
            
        Returns:
            The original real value if found, otherwise the masked value
        """
        # Find the key by value
        for key, value in self.masked_values.items():
            if value == masked_value:
                # Return the real value part of the key
                return key.split(":", 1)[1]
                
        # If not found, return the original masked value
        return masked_value
    
    def mask_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively mask IDs in an Elasticsearch query.
        
        Args:
            query: The Elasticsearch query to mask
            
        Returns:
            The query with sensitive IDs masked
        """
        if not isinstance(query, dict):
            return query
            
        masked_query = {}
        for key, value in query.items():
            # Handle specific ID fields
            if key in ["org_id", "site_id", "device_id", "user_id"]:
                if isinstance(value, str):
                    masked_query[key] = self.mask_id(key.split("_")[0], value)
                elif isinstance(value, list):
                    masked_query[key] = [self.mask_id(key.split("_")[0], v) for v in value]
                else:
                    masked_query[key] = value
            # Recursively process nested dictionaries
            elif isinstance(value, dict):
                masked_query[key] = self.mask_query(value)
            # Process lists
            elif isinstance(value, list):
                masked_query[key] = [
                    self.mask_query(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                masked_query[key] = value
                
        return masked_query
    
    def unmask_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively unmask IDs in an Elasticsearch query.
        
        Args:
            query: The Elasticsearch query with masked IDs
            
        Returns:
            The query with original IDs restored
        """
        if not isinstance(query, dict):
            return query
            
        unmasked_query = {}
        for key, value in query.items():
            # Handle specific ID fields
            if key in ["org_id", "site_id", "device_id", "user_id"]:
                if isinstance(value, str) and value.startswith("MASKED_"):
                    unmasked_query[key] = self.unmask_id(value)
                elif isinstance(value, list):
                    unmasked_query[key] = [
                        self.unmask_id(v) if isinstance(v, str) and v.startswith("MASKED_") else v
                        for v in value
                    ]
                else:
                    unmasked_query[key] = value
            # Recursively process nested dictionaries
            elif isinstance(value, dict):
                unmasked_query[key] = self.unmask_query(value)
            # Process lists
            elif isinstance(value, list):
                unmasked_query[key] = [
                    self.unmask_query(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                unmasked_query[key] = value
                
        return unmasked_query

# Create a global instance of the ID masker
id_masker = IDMasker()

@judgment.observe(span_type="JSON")
def safe_json_extract(content: str) -> Any:
    """Safely extract JSON content from a string, handling markdown code blocks."""
    try:
        # Try to parse the entire content as JSON
        return json.loads(content)
    except json.JSONDecodeError:
        # If that fails, look for JSON in markdown code blocks
        if "```json" in content:
            # Extract JSON from a markdown code block
            try:
                json_content = content.split("```json")[1].split("```")[0].strip()
                return json.loads(json_content)
            except (IndexError, json.JSONDecodeError):
                pass
                
        if "```" in content:
            # Extract from generic code block
            try:
                json_content = content.split("```")[1].split("```")[0].strip()
                return json.loads(json_content)
            except (IndexError, json.JSONDecodeError):
                pass
                
        # Look for { } delimited content
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx >= 0 and end_idx > start_idx:
                json_content = content[start_idx:end_idx+1]
                return json.loads(json_content)
        except json.JSONDecodeError:
            pass
            
        # Try to find JSON arrays with [ ]
        try:
            start_idx = content.find('[')
            end_idx = content.rfind(']')
            if start_idx >= 0 and end_idx > start_idx:
                json_content = content[start_idx:end_idx+1]
                return json.loads(json_content)
        except json.JSONDecodeError:
            pass
            
        # Return empty list as last resort for entity extraction
        return []

def normalize_entity(entity_type, entity_name):
    """Normalize entity names (handle plurals, capitalization, etc.)"""
    # Convert to lowercase for case-insensitive matching
    normalized_name = entity_name.lower()
    
    # Handle plurals for device types
    if entity_type == "device_type" and normalized_name.endswith("s"):
        # Convert plural to singular (e.g., "access points" -> "access_point")
        singular_name = normalized_name[:-1]  # Remove trailing 's'
        
        # Handle special cases
        if normalized_name == "access points":
            return "access_point"
        if normalized_name == "switches":
            return "switch"
        if normalized_name == "routers":
            return "router"
        
        return singular_name
    
    # Status special cases
    if entity_type == "status":
        if normalized_name == "connection status" or normalized_name == "status":
            # This is a reference to the concept, not a specific status
            return None
        
        status_map = {
            "offline": "disconnected",
            "online": "connected",
            "down": "disconnected",
            "up": "connected"
        }
        
        return status_map.get(normalized_name, normalized_name)
        
    # Event type special cases
    if entity_type == "event_type":
        if normalized_name in ["disconnect", "disconnection"]:
            return "device_disconnected"
        
    return normalized_name

def clean_query_fields(query_part, valid_fields, validated_entities):
    """Clean up query fields recursively"""
    if isinstance(query_part, dict):
        cleaned_dict = {}
        for key, value in query_part.items():
            # Handle special keys (query operators)
            if key in ["query", "bool", "must", "should", "must_not", "filter", "aggs", "aggregations"]:
                cleaned_value = clean_query_fields(value, valid_fields, validated_entities)
                # Only add non-empty lists/dicts
                if not (isinstance(cleaned_value, dict) and not cleaned_value) and not (isinstance(cleaned_value, list) and not cleaned_value):
                    cleaned_dict[key] = cleaned_value
            # Handle field references in term queries
            elif key == "term" or key == "terms":
                field_dict = {}
                for field_key, field_value in value.items():
                    # Handle masked ID values
                    if field_key == "org_id" and isinstance(field_value, str) and field_value.startswith("MASKED_"):
                        field_dict[field_key] = id_masker.unmask_id(field_value)
                    # Only include valid fields
                    elif field_key in valid_fields or field_key.endswith(".keyword"):
                        field_dict[field_key] = field_value
                if field_dict:  # Only add if not empty
                    cleaned_dict[key] = field_dict
            # Handle sort fields
            elif key == "sort":
                if isinstance(value, list):
                    cleaned_sort = []
                    for sort_item in value:
                        if isinstance(sort_item, dict):
                            sort_field = list(sort_item.keys())[0]
                            if sort_field in valid_fields:
                                cleaned_sort.append(sort_item)
                        # Keep sort direction strings (asc/desc)
                        elif isinstance(sort_item, str):
                            cleaned_sort.append(sort_item)
                    cleaned_dict[key] = cleaned_sort
                else:
                    cleaned_dict[key] = value
            # Handle size and from parameters
            elif key in ["size", "from"]:
                cleaned_dict[key] = value
            # Handle regular field references
            elif key in valid_fields or key.endswith(".keyword"):
                cleaned_dict[key] = clean_query_fields(value, valid_fields, validated_entities)
            # Skip invalid fields
            else:
                pass
        return cleaned_dict
    elif isinstance(query_part, list):
        # Filter out empty objects from lists
        cleaned_list = [
            item for item in (clean_query_fields(item, valid_fields, validated_entities) for item in query_part)
            if not (isinstance(item, dict) and not item)
        ]
        return cleaned_list
    elif isinstance(query_part, str):
        # Unmask any masked IDs
        if query_part.startswith("MASKED_"):
            return id_masker.unmask_id(query_part)
            
        # Replace entity placeholders if needed
        entity_result = query_part
        for entity in validated_entities:
            entity_type = entity["type"]
            entity_name = entity["name"]
            placeholder = f"{{entity:{entity_type}}}"
            if placeholder in entity_result:
                entity_result = entity_result.replace(placeholder, entity_name)
        return entity_result
    else:
        return query_part

@judgment.observe(span_type="NLP")
def extract_entities(state: Dict[str, Any]) -> TextToESState:
    """Extract named entities from the user query using OpenAI's model."""
    user_query = state["user_query"]
    
    prompt = ENTITY_EXTRACTION_PROMPT.format(user_query=user_query)
    
    try:
        messages = [
            SystemMessage(content=ENTITY_EXTRACTION_SYSTEM),
            HumanMessage(content=prompt)
        ]
        response = openai_client.invoke(messages)
        content = response.content
        
        entities = safe_json_extract(content)
        
        # Validate entities structure
        if not isinstance(entities, list):
            raise ValueError("Entities must be a list")
        
        for entity in entities:
            if not isinstance(entity, dict) or "type" not in entity or "name" not in entity:
                raise ValueError("Each entity must have 'type' and 'name' fields")
            
            # Normalize entity types
            if entity["type"] not in ["user", "device", "device_type", "location", "status", "timeframe", "event_type", "quantity"]:
                raise ValueError(f"Invalid entity type: {entity['type']}")
        
        # For entity extractor evaluation
        judgment.get_current_trace().async_evaluate(
            scorers=[
                AnswerRelevancyScorer(threshold=0.7),
                FaithfulnessScorer(threshold=0.7),
                AnswerCorrectnessScorer(threshold=0.7)
            ],
            input=f"Extract entities from query: '{user_query}'",
            actual_output=json.dumps(entities, indent=2),
            expected_output=json.dumps({
                "Show me all disconnected access points": [
                    {"type": "status", "name": "disconnected"},
                    {"type": "device_type", "name": "access points"}
                ],
                "Show me the connection status of user john.doe": [
                    {"type": "user", "name": "john.doe"},
                    {"type": "status", "name": "connection status"}
                ]
            }.get(user_query, []), indent=2),
            retrieval_context=[
                "Entity types and descriptions: \n" +
                "- user: A specific username or person (e.g., 'john.doe', 'Jane Smith')\n" +
                "- device: A specific device identifier (e.g., 'AP-102', 'Switch-45')\n" +
                "- device_type: Category of network device (e.g., 'access point', 'switch', 'router')\n" +
                "- location: Physical location reference (e.g., 'Building A', '3rd Floor', 'NYC Office')\n" +
                "- status: Connection or operational status (e.g., 'connected', 'disconnected', 'offline')\n" +
                "- timeframe: Time reference (e.g., 'yesterday', 'last week', '3 hours ago')\n" +
                "- event_type: Type of network event (e.g., 'disconnect', 'authentication failure')\n" +
                "- quantity: Numerical reference (e.g., 'top 5', 'more than 10')",
                
                "Sample data values: " + json.dumps({
                    'users': [doc['username'] for doc in elasticsearch_client.ES_SAMPLE_DATA.get('clients', [])[:5]],
                    'devices': [doc['name'] for doc in elasticsearch_client.ES_SAMPLE_DATA.get('devices', [])[:5]],
                    'locations': [doc['name'] for doc in elasticsearch_client.ES_SAMPLE_DATA.get('locations', [])[:5]],
                    'statuses': ['connected', 'disconnected', 'offline', 'online'],
                    'device_types': ['access_point', 'switch', 'router', 'gateway'],
                    'event_types': ['device_disconnected', 'authentication_failure', 'config_change']
                }, indent=2),
                
                "Example entity extractions:\n" +
                "Query: 'Show me all disconnected access points in Building A'\n" +
                "Entities: " + json.dumps([
                    {"type": "status", "name": "disconnected"},
                    {"type": "device_type", "name": "access points"},
                    {"type": "location", "name": "Building A"}
                ], indent=2) + "\n\n" +
                "Query: 'When did john.doe last connect from the NYC office?'\n" +
                "Entities: " + json.dumps([
                    {"type": "user", "name": "john.doe"},
                    {"type": "event_type", "name": "connect"},
                    {"type": "location", "name": "NYC office"}
                ], indent=2)
            ],
            additional_metadata={"query": user_query},
            model=OPENAI_MODEL,
            log_results=True
        )
        
        return {**state, "entities": entities}
    except Exception as e:
        error_msg = f"Entity extraction error: {str(e)}"
        return {**state, "entities": [], "error": error_msg}

@judgment.observe(span_type="NLP")
def validate_entities(state: Dict[str, Any]) -> TextToESState:
    """Validate and augment entities extracted from the user query."""
    user_query = state.get("user_query", "")
    entities = state.get("entities", [])
    
    try:
        prompt = ENTITY_VALIDATION_PROMPT.format(
            user_query=user_query,
            entities_json=json.dumps(entities, indent=2)
        )
        
        messages = [
            SystemMessage(content=ENTITY_VALIDATION_SYSTEM),
            HumanMessage(content=prompt)
        ]
        response = openai_client.invoke(messages)
        content = response.content
        
        validated_entities = safe_json_extract(content)
        
        # Validate entities structure
        if not isinstance(validated_entities, list):
            raise ValueError("Entities must be a list")
        
        for entity in validated_entities:
            if not isinstance(entity, dict) or "type" not in entity or "name" not in entity:
                raise ValueError("Each entity must have 'type' and 'name' fields")
            
            # Normalize entity types
            if entity["type"] not in ["user", "device", "device_type", "location", "status", "timeframe", "event_type", "quantity"]:
                raise ValueError(f"Invalid entity type: {entity['type']}")
        
        # For entity validation evaluation
        judgment.get_current_trace().async_evaluate(
            scorers=[
                AnswerRelevancyScorer(threshold=0.7),
                FaithfulnessScorer(threshold=0.7),
                AnswerCorrectnessScorer(threshold=0.7)
            ],
            input=f"Validate extracted entities from query: '{user_query}'",
            actual_output=json.dumps(validated_entities, indent=2),
            expected_output=json.dumps(entities, indent=2),
            retrieval_context=[
                user_query, 
                json.dumps(elasticsearch_client.ES_SAMPLE_DATA, indent=2)
            ],
            additional_metadata={"extracted_entities": entities},
            model=OPENAI_MODEL,
            log_results=True
        )
        
        return {
            **state, 
            "validated_entities": validated_entities
        }
    except Exception as e:
        error_msg = f"Entity validation error: {str(e)}"
        return {**state, "validated_entities": [], "error": error_msg}

@judgment.observe(span_type="Search")
def select_index(state: Dict[str, Any]) -> TextToESState:
    """Select the appropriate Elasticsearch index based on the user query using LLM."""
    user_query = state["user_query"]
    entities = state.get("entities", [])
    validated_entities = state.get("validated_entities", [])
    
    # Create a description of available indices for the LLM
    index_descriptions = "\n".join([
        f"- {index_name}: {index_info['description']}"
        for index_name, index_info in elasticsearch_client.ES_INDEXES.items()
    ])
    
    try:
        # Use the proper prompt from prompts.py
        prompt = INDEX_SELECTION_PROMPT.format(
            user_query=user_query,
            index_descriptions=index_descriptions,
            validated_entities_json=json.dumps(validated_entities, indent=2),
            valid_indices=", ".join(elasticsearch_client.ES_INDEXES.keys())
        )
        
        messages = [
            SystemMessage(content=INDEX_SELECTION_SYSTEM),
            HumanMessage(content=prompt)
        ]
        response = openai_client.invoke(messages)
        selected_index = response.content.strip().lower()
        
        # Validate the selected index
        if selected_index not in elasticsearch_client.ES_INDEXES:
            return {**state, "error": f"Invalid index selected: {selected_index} - not found in index configuration"}
        
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
            expected_output={
                # Ground truth mapping for test queries
                "Show me all disconnected access points": "devices",
                "How many switches are in Building B?": "devices",
                "List all devices in the Data Center": "devices",
                "Show me the connection status of user john.doe": "clients",
                "When did AP-102 last disconnect?": "events",
                "What's the status of Building A?": "locations"
            }.get(user_query, selected_index),
            retrieval_context=[
                # Pre-format complex JSON structures
                json.dumps({
                    "devices": elasticsearch_client.ES_INDEXES["devices"]["description"],
                    "clients": elasticsearch_client.ES_INDEXES["clients"]["description"],
                    "locations": elasticsearch_client.ES_INDEXES["locations"]["description"],
                    "events": elasticsearch_client.ES_INDEXES["events"]["description"]
                }, indent=2),
                f"Extracted entities: {json.dumps(validated_entities, indent=2)}"
            ],
            additional_metadata={"query": user_query, "entities": validated_entities},
            model=OPENAI_MODEL,
            log_results=True
        )
        
        return {
            **state,
            "selected_index": selected_index,
            "index_mappings": index_mappings,
            "essential_fields": essential_fields
        }
    except KeyError as e:
        return {**state, "error": f"Index configuration error: Missing key {e} in ES_INDEXES configuration"}

@judgment.observe(span_type="NLP")
def get_field_values(state: Dict[str, Any]) -> TextToESState:
    """Identify which field values are relevant to the user's query."""
    user_query = state["user_query"]
    selected_index = state["selected_index"]
    
    # Get the available field values for this index
    field_values = elasticsearch_client.ES_INDEXES[selected_index].get("field_values", {})
    
    # Format as JSON for the prompt
    field_values_json = json.dumps(field_values, indent=2)
    
    try:
        # Use the proper prompt from prompts.py
        prompt = FIELD_VALUES_PROMPT.format(
            user_query=user_query,
            field_values_json=field_values_json
        )
        
        messages = [
            SystemMessage(content=FIELD_EXTRACTION_SYSTEM),
            HumanMessage(content=prompt)
        ]
        response = openai_client.invoke(messages)
        content = response.content
        
        selected_field_values = safe_json_extract(content)
        
        # Fix: Handle the case when safe_json_extract returns a non-dictionary
        if not isinstance(selected_field_values, dict):
            selected_field_values = {}  # Default to empty dictionary
        
        # Only validate fields if we have a field_values dictionary
        if field_values:
            # Validate that all selected values are in the possible values
            for field, value in selected_field_values.items():
                if field not in field_values:
                    # Instead of returning an error, just remove the invalid field
                    del selected_field_values[field]
                    continue
                
                if isinstance(value, list):
                    selected_field_values[field] = [
                        v for v in value if v in field_values[field]
                    ]
                elif value not in field_values[field]:
                    # If the value is invalid, remove it
                    del selected_field_values[field]
        
        # For field values selection evaluation  
        judgment.get_current_trace().async_evaluate(
            scorers=[
                AnswerRelevancyScorer(threshold=0.7),
                FaithfulnessScorer(threshold=0.7)
            ],
            input=f"Determine relevant field values for query: '{user_query}' in index '{selected_index}'",
            actual_output=json.dumps(selected_field_values, indent=2),
            context=[selected_index],
            retrieval_context=[
                f"Available fields and values: {json.dumps(field_values, indent=2)}",
                f"Index schema: {json.dumps(elasticsearch_client.ES_INDEXES[selected_index]['mappings']['properties'], indent=2)}"
            ],
            additional_metadata={"essential_fields": elasticsearch_client.ES_INDEXES[selected_index].get("essential_fields", {})},
            model=OPENAI_MODEL,
            log_results=True
        )
        
        return {**state, "field_values": field_values, "selected_field_values": selected_field_values}
    except Exception as e:
        return {**state, "error": f"Field value retrieval error: {str(e)}"}

@judgment.observe(span_type="Query")
def generate_query(state: Dict[str, Any]) -> TextToESState:
    """Generate an Elasticsearch query based on the user's query and validated entities."""
    user_query = state["user_query"]
    validated_entities = state.get("validated_entities", [])
    selected_index = state["selected_index"]
    field_values = state.get("selected_field_values", {})
    index_mappings = state.get("index_mappings", {})
    
    # Get the field mapping for the selected index
    index_entity_mapping = ENTITY_FIELD_MAPPINGS.get(selected_index, {})
    
    # Format validated entities with proper field mappings
    formatted_entities = []
    for entity in validated_entities:
        entity_type = entity["type"]
        entity_name = entity["name"]
        entity_field = index_entity_mapping.get(entity_type, f"{entity_type}_id")
        
        formatted_entities.append({
            "type": entity_type,
            "name": entity_name,
            "field": entity_field
        })
    
    try:
        # Get index fields
        index_fields = list(index_mappings.get("properties", {}).keys()) if index_mappings else []
        
        prompt = QUERY_GENERATION_PROMPT.format(
            user_query=user_query,
            selected_index=selected_index,
            index_fields_json=json.dumps(index_fields, indent=2),
            entity_mappings_json=json.dumps(index_entity_mapping, indent=2),
            formatted_entities_json=json.dumps(formatted_entities, indent=2),
            field_values_json=json.dumps(field_values, indent=2),
            user_field=index_entity_mapping.get('user', 'username'),
            org_field=index_entity_mapping.get('org', 'org_id'),
            device_field=index_entity_mapping.get('device', 'id')
        )
        
        messages = [
            SystemMessage(content=QUERY_GENERATOR_SYSTEM),
            HumanMessage(content=prompt)
        ]
        response = openai_client.invoke(messages)
        query_text = response.content.strip()
        
        # Ensure we got JSON back and parse it
        if query_text.startswith("```json"):
            query_text = query_text.split("```json")[1].split("```")[0].strip()
        elif query_text.startswith("```"):
            query_text = query_text.split("```")[1].split("```")[0].strip()
        
        query = json.loads(query_text)
        
        # For query generation evaluation - with ground truth based on the sample data
        judgment.get_current_trace().async_evaluate(
            scorers=[
                AnswerRelevancyScorer(threshold=0.7),
                FaithfulnessScorer(threshold=0.7),
                AnswerCorrectnessScorer(threshold=0.7)
            ],
            input=f"Generate Elasticsearch query for: '{user_query}' on index '{selected_index}'",
            actual_output=json.dumps(query, indent=2),
            expected_output=json.dumps({
                # Examples of ground truth queries based on sample data
                "Show me all disconnected access points": {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"status": "disconnected"}},
                                {"term": {"device_type": "access_point"}}
                            ]
                        }
                    },
                    "size": 10
                },
                "Show me the connection status of user john.doe": {
                    "query": {
                        "term": {"username": "john.doe"}
                    },
                    "size": 10
                },
                "When did AP-102 last disconnect?": {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"device_name": "AP-102"}},
                                {"term": {"event_type": "device_disconnected"}}
                            ]
                        }
                    },
                    "sort": [{"timestamp": {"order": "desc"}}],
                    "size": 1
                }
            }.get(user_query, query), indent=2),
            retrieval_context=[
                f"Formatted entities: {json.dumps(formatted_entities, indent=2)}",
                f"Index schema: {json.dumps(elasticsearch_client.ES_INDEXES[selected_index]['mappings']['properties'], indent=2)}",
                f"Sample data: {json.dumps(elasticsearch_client.ES_SAMPLE_DATA.get(selected_index, [])[:2], indent=2)}"
            ],
            additional_metadata={
                "query_requires_aggregation": "count" in user_query.lower() or "how many" in user_query.lower(),
                "query_is_about_status": "status" in user_query.lower() or "connected" in user_query.lower() or "disconnected" in user_query.lower()
            },
            model=OPENAI_MODEL,
            log_results=True
        )
        
        return {**state, "es_query": query}
    except Exception as e:
        return {**state, "error": f"Query generation error: {str(e)}"}

@judgment.observe(span_type="Query")
def process_query(state: Dict[str, Any]) -> TextToESState:
    """Process the Elasticsearch query to make it executable."""
    es_query = state.get("es_query")
    index_mappings = state.get("index_mappings", {})
    validated_entities = state.get("validated_entities", [])
    selected_index = state.get("selected_index")
    user_query = state.get("user_query", "")
    
    if not es_query:
        return {**state, "error": "No query to process - query generation failed"}
    
    try:
        # Convert query to dict if it's not already (to handle ObjectApiResponse)
        if hasattr(es_query, "body"):
            es_query = es_query.body
            
        # If query is a string, parse it to a dictionary
        if isinstance(es_query, str):
            es_query = json.loads(es_query)
        
        # Get the valid field names from the index mapping
        valid_fields = set()
        if "properties" in index_mappings:
            valid_fields = set(index_mappings["properties"].keys())
            
            # Add nested fields
            for field, field_mapping in index_mappings["properties"].items():
                if isinstance(field_mapping, dict) and "properties" in field_mapping:
                    for nested_field in field_mapping["properties"].keys():
                        valid_fields.add(f"{field}.{nested_field}")
        
        # Add common fields that might not be in the mapping but are usually valid
        common_fields = ["org_id", "id", "timestamp", "event_type", "status", "name"]
        for field in common_fields:
            valid_fields.add(field)
            valid_fields.add(f"{field}.keyword")
        
        # Unmask any masked IDs in the query
        unmasked_query = id_masker.unmask_query(es_query)
            
        # Clean up the query using the extracted function
        processed_query = clean_query_fields(unmasked_query, valid_fields, validated_entities)
        
        # Always ensure basic query structure
        if not processed_query:
            processed_query = {"query": {"match_all": {}}, "size": 10}
        elif "query" not in processed_query:
            processed_query["query"] = {"match_all": {}}
        
        # Ensure we have a size parameter
        if "size" not in processed_query:
            processed_query["size"] = 10
            
        # Fix bool queries with empty filter/must/should arrays
        if "query" in processed_query and isinstance(processed_query["query"], dict):
            query = processed_query["query"]
            if "bool" in query and isinstance(query["bool"], dict):
                bool_query = query["bool"]
                for clause in ["filter", "must", "should", "must_not"]:
                    if clause in bool_query:
                        if not bool_query[clause] or (isinstance(bool_query[clause], list) and len(bool_query[clause]) == 0):
                            del bool_query[clause]
                
                # If the bool query is now empty, replace with match_all
                if not bool_query:
                    processed_query["query"] = {"match_all": {}}
        
        # For query processing evaluation
        judgment.get_current_trace().async_evaluate(
            scorers=[
                FaithfulnessScorer(threshold=0.7)
            ],
            input=f"Process Elasticsearch query for execution: {json.dumps(es_query, indent=2)}",
            actual_output=json.dumps(processed_query, indent=2),
            retrieval_context=[
                f"Index mappings: {json.dumps(index_mappings, indent=2)}",
                f"Validated entities: {json.dumps(validated_entities, indent=2)}"
            ],
            additional_metadata={
                "original_query": user_query,
                "valid_fields": list(valid_fields),
                "changes_made": "Removed invalid fields and normalized query structure"
            },
            model=OPENAI_MODEL,
            log_results=True
        )
        
        return {**state, "processed_query": processed_query}
    except Exception as e:
        # Handle query processing errors
        return {**state, "error": f"Query processing error: {str(e)}"}

@judgment.observe(span_type="Database")
def execute_query(state: Dict[str, Any]) -> TextToESState:
    """Execute the processed Elasticsearch query and return results."""
    processed_query = state.get("processed_query")
    selected_index = state.get("selected_index")
    user_query = state.get("user_query", "")
    
    if not processed_query:
        return {**state, "error": "No processed query to execute"}
    
    if not selected_index:
        return {**state, "error": "No index selected for query execution"}
    
    try:
        # Execute the query
        result = es_client.search(index=selected_index, body=processed_query)
        
        # Convert ElasticSearch response to dictionary
        if hasattr(result, "body"):
            result = result.body
            
        # Mask any sensitive IDs in the results before storing them
        masked_result = {}
        
        # Extract hits and mask IDs
        if "hits" in result:
            masked_hits = {
                "total": result["hits"].get("total", {}).get("value", 0),
                "hits": []
            }
            
            for hit in result["hits"].get("hits", []):
                source = hit.get("_source", {})
                # Mask any sensitive IDs in the source
                masked_source = {}
                for key, value in source.items():
                    if key.endswith("_id") and isinstance(value, str):
                        id_type = key.replace("_id", "")
                        masked_source[key] = id_masker.mask_id(id_type, value)
                    else:
                        masked_source[key] = value
                
                masked_hit = {**hit, "_source": masked_source}
                masked_hits["hits"].append(masked_hit)
            
            masked_result["hits"] = masked_hits
        
        # Extract aggregations and mask IDs
        if "aggregations" in result:
            masked_result["aggregations"] = result["aggregations"]  # For now, not masking IDs in aggregations
        
        # Extract results
        results = {
            "total": masked_result.get("hits", {}).get("total", 0),
            "hits": [hit.get("_source", {}) for hit in masked_result.get("hits", {}).get("hits", [])],
            "aggregations": masked_result.get("aggregations", {})
        }
        
        # For query execution evaluation
        judgment.get_current_trace().async_evaluate(
            scorers=[
                FaithfulnessScorer(threshold=0.7)
            ],
            input=f"Results of Elasticsearch query execution for: '{user_query}'",
            actual_output=json.dumps({
                "total_hits": results["total"],
                "sample_hits": results["hits"][:3] if len(results["hits"]) > 0 else [],
                "has_aggregations": bool(results["aggregations"])
            }, indent=2),
            retrieval_context=[
                json.dumps(processed_query, indent=2),
                json.dumps({
                    "Show me all disconnected access points": {
                        "total_hits": 2,  # AP-102, AP-104 from sample data
                        "expected_fields": ["id", "name", "status", "device_type"],
                        "example_hit": {"name": "AP-102", "status": "disconnected", "device_type": "access_point"}
                    },
                    "Show me the connection status of user john.doe": {
                        "total_hits": 1,
                        "expected_fields": ["username", "connection_status"],
                        "example_hit": {"username": "john.doe", "connection_status": "connected"}
                    }
                }.get(user_query, {}), indent=2)
            ],
            additional_metadata={
                "index": selected_index,
                "total_hits": results["total"],
                "has_aggregations": bool(results["aggregations"])
            },
            model=OPENAI_MODEL,
            log_results=True
        )
        
        # Store the results
        return {**state, "query_results": results}
    except NotFoundError as e:
        return {**state, "error": f"Index not found: {str(e)}"}
    except ApiError as e:
        return {**state, "error": f"Elasticsearch query error: {str(e)}"}
    except Exception as e:
        return {**state, "error": f"Response formatting error: {str(e)}"}

@judgment.observe(span_type="Response")
def format_response(state: Dict[str, Any]) -> TextToESState:
    """Format the query results into a natural language response."""
    user_query = state.get("user_query", "")
    query_results = state.get("query_results", {})
    selected_index = state.get("selected_index", "")
    validated_entities = state.get("validated_entities", [])
    processed_query = state.get("processed_query", {})
    
    if not query_results:
        return {**state, "final_response": "I couldn't find any results for your query.", "error": "No query results to format"}
    
    try:
        # Extract relevant information from the query results
        total_hits = query_results.get("total", 0)
        hits = query_results.get("hits", [])
        aggregations = query_results.get("aggregations", {})
        
        # Build a description of the validated entities
        entity_descriptions = []
        for entity in validated_entities:
            entity_type = entity["type"]
            entity_name = entity["name"]
            entity_descriptions.append(f"{entity_name} ({entity_type})")
        
        entities_text = ", ".join(entity_descriptions) if entity_descriptions else "None"
        
        # Format hits and aggregations as JSON strings
        hits_json = json.dumps(hits, indent=2)
        aggs_json = json.dumps(aggregations, indent=2)
        
        # Use the proper prompt from prompts.py
        prompt = RESPONSE_FORMATTING_PROMPT.format(
            user_query=user_query,
            selected_index=selected_index,
            total_hits=total_hits,
            hits_json=hits_json,
            aggs_json=aggs_json,
            entities_text=entities_text
        )
        
        messages = [
            SystemMessage(content=RESPONSE_FORMATTER_SYSTEM),
            HumanMessage(content=prompt)
        ]
        response = openai_client.invoke(messages)
        final_response = response.content.strip()
        
        # For response formatting evaluation
        judgment.get_current_trace().async_evaluate(
            scorers=[
                AnswerRelevancyScorer(threshold=0.7),
                FaithfulnessScorer(threshold=0.7)
            ],
            input=f"Original query: '{user_query}'",
            actual_output=final_response,
            retrieval_context=[
                f"Query results: {json.dumps({'total_hits': total_hits, 'hits': hits[:3] if hits else [], 'aggregations': aggregations}, indent=2)}",
                f"Entities mentioned: {entities_text}",
                f"Index used: {selected_index}"
            ],
            additional_metadata={
                "query_intent": "aggregation" if bool(aggregations) else "search",
                "response_length": len(final_response)
            },
            model=OPENAI_MODEL,
            log_results=True
        )
        
        return {**state, "final_response": final_response}
    except Exception as e:
        return {**state, "error": f"Response formatting error: {str(e)}"}

@judgment.observe(span_type="Error")
def handle_error(state: Dict[str, Any]) -> TextToESState:
    """Handle any errors that occur during processing."""
    error = state.get("error", "Unknown error occurred")
    return {**state, "final_response": f"I encountered an error processing your request: {error}"}

def has_error(state: TextToESState) -> str:
    """Route to error handler if state has an error."""
    if state.get("error"):
        return "error"
    return "continue"

@judgment.observe(span_type="Function")
async def text2es_pipeline():
    # Initialize Elasticsearch with sample data
    if not elasticsearch_client.init_elasticsearch():
        print("ERROR: Failed to initialize Elasticsearch. Exiting.")
        return "Elasticsearch initialization failed"
    # Initialize the graph
    workflow = StateGraph(TextToESState)
    
    # Add nodes directly without wrapping
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
    
    # Add conditional edges with correct parameter structure
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
    
    # We don't need conditional edges for format_response as it's a finish point
    
    # Set finish points
    workflow.set_finish_point("format_response")
    workflow.set_finish_point("handle_error")
    
    # Compile the workflow
    app = workflow.compile()

    handler = JudgevalCallbackHandler(judgment.get_current_trace())
    
    # Test queries to evaluate the end-to-end workflow
    test_queries = [
        "Show me all disconnected access points",
        "How many switches are in Building B?",
        "List all devices in the Data Center",
        "Show me the connection status of user john.doe",
        "When did AP-102 last disconnect?",
        "What's the status of Building A?"
    ]
    
    # Run the workflow for each test query
    config = dict(callbacks=[handler])
    
    print("=" * 50)
    print("TESTING TEXT-TO-ELASTICSEARCH PIPELINE")
    print("=" * 50)
    
    for query in test_queries[3:4]:
        print("\n" + "-" * 50)
        print(f"QUERY: {query}")
        print("-" * 50)
        
        try:
            # Run the workflow with the query
            result = await app.ainvoke(
                {
                    "messages": [HumanMessage(content=query)],
                    "user_query": query,
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
                },
                config=config
            )
            
            # Print intermediate results
            print("\nExtracted Entities:")
            print(json.dumps(result.get("entities", []), indent=2))
            
            print("\nValidated Entities:")
            print(json.dumps(result.get("validated_entities", []), indent=2))
            
            print("\nSelected Index:")
            print(result.get("selected_index", "None"))
            
            if "selected_field_values" in result:
                print("\nSelected Field Values:")
                print(json.dumps(result.get("selected_field_values", {}), indent=2))
            
            if "es_query" in result:
                print("\nGenerated ES Query:")
                print(json.dumps(result.get("es_query", {}), indent=2))
                
            if "processed_query" in result:
                print("\nProcessed ES Query:")
                print(json.dumps(result.get("processed_query", {}), indent=2))
            
            # Print the final result
            if result.get("error"):
                print(f"\nERROR: {result['error']}")
            else:
                print(f"\nFINAL RESPONSE: {result['final_response']}")
        
        except Exception as e:
            print(f"\nWorkflow execution error: {str(e)}")
    

    
    return "Pipeline testing completed"

if __name__ == "__main__":
    asyncio.run(text2es_pipeline())