from langchain_openai import ChatOpenAI
import asyncio
import os

import chromadb
from chromadb.utils import embedding_functions

from vectordbdocs import financial_data

from typing import Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ChatMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

from judgeval.common.tracer import Tracer, JudgevalCallbackHandler
from judgeval.scorers import AnswerCorrectnessScorer, FaithfulnessScorer



judgment = Tracer()

# Define our state type
class AgentState(TypedDict):
    messages: list[BaseMessage]
    category: Optional[str]
    documents: Optional[str]
    
def populate_vector_db(collection, raw_data):
    """
    Populate the vector DB with financial information.
    """
    for data in raw_data:
        collection.add(
            documents=[data['information']],
            metadatas=[{"category": data['category']}],
            ids=[f"category_{data['category'].lower().replace(' ', '_')}_{os.urandom(4).hex()}"]
        )

# Define a ChromaDB collection for document storage
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="financial_docs",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))
)

populate_vector_db(collection, financial_data)

@judgment.observe(name="pnl_retriever", span_type="retriever")
def pnl_retriever(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    results = collection.query(
        query_texts=[query],
        where={"category": "pnl"},
        n_results=3
    )
    documents = []
    for document in results["documents"]:
        documents += document

    return {"messages": state["messages"], "documents": documents}

@judgment.observe(name="balance_sheet_retriever", span_type="retriever")
def balance_sheet_retriever(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    results = collection.query(
        query_texts=[query],
        where={"category": "balance_sheets"},
        n_results=3
    )
    documents = []
    for document in results["documents"]:
        documents += document

    return {"messages": state["messages"], "documents": documents}

@judgment.observe(name="stock_retriever", span_type="retriever")
def stock_retriever(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    results = collection.query(
        query_texts=[query],
        where={"category": "stocks"},
        n_results=3
    )
    documents = []
    for document in results["documents"]:
        documents += document

    return {"messages": state["messages"], "documents": documents}

@judgment.observe(name="bad_classifier", span_type="llm")
async def bad_classifier(state: AgentState) -> AgentState:
    return {"messages": state["messages"], "category": "stocks"}

@judgment.observe(name="bad_classify")
async def bad_classify(state: AgentState) -> AgentState:
    category = await bad_classifier(state)
    
    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerCorrectnessScorer(threshold=1)],
        input=state["messages"][-1].content,
        actual_output=category["category"],
        expected_output="pnl",
        model="gpt-4o-mini"
    )
    
    return {"messages": state["messages"], "category": category["category"]}

@judgment.observe(name="bad_sql_generator", span_type="llm")
async def bad_sql_generator(state: AgentState) -> AgentState:
    ACTUAL_OUTPUT = "SELECT * FROM pnl WHERE stock_symbol = 'apppl'"
    
    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerCorrectnessScorer(threshold=1), FaithfulnessScorer(threshold=1)],
        input=state["messages"][-1].content,
        retrieval_context=state.get("documents", []),
        actual_output=ACTUAL_OUTPUT,
        expected_output=
        """
        SELECT 
            SUM(CASE 
                WHEN transaction_type = 'sell' THEN (price_per_share - (SELECT price_per_share FROM stock_transactions WHERE stock_symbol = 'aapl' AND transaction_type = 'buy' LIMIT 1)) * quantity 
                ELSE 0 
            END) AS realized_pnl
        FROM 
            stock_transactions
        WHERE 
            stock_symbol = 'aapl';
        """,
        model="gpt-4o-mini"
    )
    return {"messages": state["messages"] + [ChatMessage(content=ACTUAL_OUTPUT, role="text2sql")]}

# Create the classifier node with a system prompt
@judgment.observe(name="classify")
async def classify(state: AgentState) -> AgentState:
    messages = state["messages"]
    input_msg = [
        SystemMessage(content="""You are a financial query classifier. Your job is to classify user queries into one of three categories:
        - 'pnl' for Profit and Loss related queries
        - 'balance_sheets' for Balance Sheet related queries
        - 'stocks' for Stock market related queries
        
        Respond ONLY with the category name in lowercase, nothing else."""),
        *messages
    ]
    
    response = ChatOpenAI(model="gpt-4o-mini", temperature=0).invoke(
        input=input_msg
    )
    
    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerCorrectnessScorer(threshold=1)],
        input=str(input_msg),
        actual_output=response.content,
        expected_output="pnl",
        model="gpt-4o-mini"
    )

    return {"messages": state["messages"], "category": response.content}

# Add router node to direct flow based on classification
def router(state: AgentState) -> str:
    return state["category"]

@judgment.observe(name="generate_response")
async def generate_response(state: AgentState) -> AgentState:
    messages = state["messages"]
    documents = state.get("documents", "")

    OUTPUT = """
        SELECT 
            stock_symbol,
            SUM(CASE WHEN transaction_type = 'buy' THEN quantity ELSE -quantity END) AS total_shares,
            SUM(CASE WHEN transaction_type = 'buy' THEN quantity * price_per_share ELSE -quantity * price_per_share END) AS total_cost,
            MAX(CASE WHEN transaction_type = 'buy' THEN price_per_share END) AS current_market_price
        FROM 
            stock_transactions
        WHERE 
            stock_symbol = 'aapl'
        GROUP BY 
            stock_symbol;
        """
    
    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerCorrectnessScorer(threshold=1), FaithfulnessScorer(threshold=1)],
        input=messages[-1].content,
        actual_output=OUTPUT,
        retrieval_context=documents,
        expected_output="""
        SELECT 
            stock_symbol,
            SUM(CASE WHEN transaction_type = 'buy' THEN quantity ELSE -quantity END) AS total_shares,
            SUM(CASE WHEN transaction_type = 'buy' THEN quantity * price_per_share ELSE -quantity * price_per_share END) AS total_cost,
            MAX(CASE WHEN transaction_type = 'buy' THEN price_per_share END) AS current_market_price
        FROM 
            stock_transactions
        WHERE 
            stock_symbol = 'aapl'
        GROUP BY 
            stock_symbol;
        """,
        model="gpt-4o-mini"
    )

    return {"messages": messages + [ChatMessage(content=OUTPUT, role="text2sql")], "documents": documents}

async def main():
    with judgment.trace(
        "JP_Morgan_Run_3",
        project_name="JP_Morgan",
        overwrite=True
    ) as trace:

        # Initialize the graph
        graph_builder = StateGraph(AgentState)

        # Add classifier node
        # For failure test, pass in bad_classifier
        graph_builder.add_node("classifier", classify)
        # graph_builder.add_node("classifier", bad_classify)
        
        # Add conditional edges based on classification
        graph_builder.add_conditional_edges(
            "classifier",
            router,
            {
                "pnl": "pnl_retriever",
                "balance_sheets": "balance_sheet_retriever",
                "stocks": "stock_retriever"
            }
        )
        
        # Add retriever nodes (placeholder functions for now)
        graph_builder.add_node("pnl_retriever", pnl_retriever)
        graph_builder.add_node("balance_sheet_retriever", balance_sheet_retriever)
        graph_builder.add_node("stock_retriever", stock_retriever)

        # Add edges from retrievers to response generator
        graph_builder.add_node("response_generator", generate_response)
        # graph_builder.add_node("response_generator", bad_sql_generator)
        graph_builder.add_edge("pnl_retriever", "response_generator")
        graph_builder.add_edge("balance_sheet_retriever", "response_generator")
        graph_builder.add_edge("stock_retriever", "response_generator")
        
        graph_builder.set_entry_point("classifier")
        graph_builder.set_finish_point("response_generator")

        # Compile the graph
        graph = graph_builder.compile()
        
        handler = JudgevalCallbackHandler(trace)

        response = await graph.ainvoke({
            "messages": [HumanMessage(content="Please calculate our PNL on Apple stock. Refer to table information from documents provided.")],
            "category": None,
        }, config=dict(callbacks=[handler]))
        trace.save()
    
        print(f"Response: {response['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())
