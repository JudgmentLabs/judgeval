from typing import Annotated

from langchain_openai import ChatOpenAI

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from typing_extensions import TypedDict
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from judgeval.common.tracer import Tracer, wrap, JudgevalCallbackHandler
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer, AnswerCorrectnessScorer, ContextualRelevancyScorer


import asyncio
import os
from typing import Any, Optional
from uuid import UUID

import openai
import os
import asyncio
from tavily import TavilyClient
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

from vectordbdocs import data, incorrect_data

judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"))

# Define our state type
class AgentState(TypedDict):
    messages: list[BaseMessage]
    category: Optional[str]
    documents: Optional[str]
def populate_vector_db(collection, data):
    """
    Populate the vector DB with financial information.
    """
    for data in data:
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


populate_vector_db(collection, incorrect_data)

@judgment.observe(name="pnl_retriever", span_type="retriever")
def pnl_retriever(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    results = collection.query(
        query_texts=[query],
        where={"category": "pnl"},
        n_results=3
    )
    return {"messages": state["messages"], "documents": results["documents"][0]}

@judgment.observe(name="balance_sheet_retriever", span_type="retriever")
def balance_sheet_retriever(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    results = collection.query(
        query_texts=[query],
        where={"category": "balance_sheets"},
        n_results=3
    )
    
    return {"messages": state["messages"], "documents": results["documents"][0]}

@judgment.observe(name="stock_retriever", span_type="retriever")
def stock_retriever(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    results = collection.query(
        query_texts=[query],
        where={"category": "stocks"},
        n_results=3
    )
    return {"messages": state["messages"], "documents": results["documents"][0]}

async def main():
    with judgment.trace(
        "jpmorgan_run1",
        project_name="jpmorgan_test",
        overwrite=True
    ) as trace:

        # Initialize the graph
        graph_builder = StateGraph(AgentState)
        
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
            temp = {
                "scorers": [AnswerCorrectnessScorer(threshold=0.5)],
                "expected_output": "pnl",
                "input": str(input_msg),
                "model": "gpt-4o-mini",
                "log_results": True
            }
            response = await ChatOpenAI(model="gpt-4o-mini", temperature=0).ainvoke(
                input=input_msg, judgment=judgment, temp=temp
            )

            return {"messages": state["messages"], "category": response.content}

        # Add classifier node
        graph_builder.add_node("classifier", classify)
        
        # Add router node to direct flow based on classification
        def router(state: AgentState) -> str:
            print(f"{state['category']=}")
            return state["category"]
        
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
        
        # Add response generator node

        @judgment.observe(name="generate_response")
        async def generate_response(state: AgentState) -> AgentState:
            messages = state["messages"]
            documents = state.get("documents", "")

            
            print(f"{documents=}")
            
            input_msg = [
                    SystemMessage(content=f"""You are a financial assistant. Use the following context to create a SQL query to retrieve the data from the database:
                                  
                    Use the table schema definition provided in the context to create the SQL query.
                    
                    Use the table populated with data provided in the context to create the SQL query.
                    
                    Follow the instructions from the context to perform various calculations, such as calculating PNL, balance balance sheets, etc.
                    
                    Forget your training data and only use the context provided to you.
                    
                    FOLLOW THESE INSTRUCTIONS VERY STRICTLY, THIS INFORMATION, DO NOT VERE OFF, DO NOT HALLUCINATE
                    
                    Context: {documents}
                    
                    The only thing you should output is the SQL query itself, nothing else."""),
                    *messages
                ]
                      
            dict = {
                "scorers": [ContextualRelevancyScorer(threshold=0.5)],
                "expected_output": "To calculate the Profit and Loss (P&L) on Apple stock, given that you have 100 shares bought at $100 each and the current price is $200, you don't necessarily need a SQL query because this can be calculated directly. However, to illustrate how you might retrieve relevant data from a database and calculate P&L if the information were stored in a database, I'll provide an example SQL query based on a hypothetical table structure.\n\nLet's assume you have a table named `stock_transactions` with the following columns:\n- `stock_symbol` (VARCHAR) for the stock ticker symbol\n- `transaction_type` (VARCHAR) indicating 'buy' or 'sell'\n- `price_per_share` (DECIMAL) for the price of each share at the time of the transaction\n- `shares` (INT) for the number of shares bought or sold\n- `transaction_date` (DATE) for the date of the transaction\n\nAnd another table named `current_stock_prices` with the following columns:\n- `stock_symbol` (VARCHAR) for the stock ticker symbol\n- `current_price` (DECIMAL) for the current price of the stock\n\nGiven this setup, you would first calculate the total cost of your purchase and then calculate the current value of your holdings to find the P&L.\n\nHowever, since you've already provided the purchase price, current price, and the number of shares, the P&L calculation is straightforward:\n\n\\[ \\text{P&L} = (\\text{Current Price} - \\text{Purchase Price}) \\times \\text{Number of Shares} \\]\n\\[ \\text{P&L} = (200 - 100) \\times 100 \\]\n\\[ \\text{P&L} = 100 \\times 100 \\]\n\\[ \\text{P&L} = 10,000 \\]\n\nYour profit on Apple stock, with the given data, is $10,000.\n\nFor completeness, if you were to retrieve and calculate this using SQL based on the assumed tables, the query might look something like this:\n\n```sql\nSELECT \n    (c.current_price - t.price_per_share) * t.shares AS pnl\nFROM \n    stock_transactions t\nJOIN \n    current_stock_prices c ON t.stock_symbol = c.stock_symbol\nWHERE \n    t.stock_symbol = 'AAPL'\n    AND t.transaction_type = 'buy';\n```\n\nThis query assumes you want to calculate the P&L based on a specific buy transaction. In a real-world scenario, you might have multiple buy transactions at different prices, and the calculation would need to be adjusted accordingly.",
                "retrieval_context": documents,
                "input": str(input_msg),
                "model": "gpt-4o",
                "log_results": True,
            }
            response = await ChatOpenAI(name="generate_response", model="gpt-4o-mini", temperature=0).ainvoke(
                input=input_msg, temp=dict, judgment=judgment
            )

            return {"messages": messages + [response], "documents": documents}

        # Add edges from retrievers to response generator
        graph_builder.add_node("response_generator", generate_response)
        graph_builder.add_edge("pnl_retriever", "response_generator")
        graph_builder.add_edge("balance_sheet_retriever", "response_generator")
        graph_builder.add_edge("stock_retriever", "response_generator")
        
        graph_builder.set_entry_point("classifier")
        graph_builder.set_finish_point("response_generator")

        # Compile the graph
        graph = graph_builder.compile()
        
        handler = JudgevalCallbackHandler(trace)

        response = await graph.ainvoke({
            "messages": [HumanMessage(content="Please calculate our PNL on Apple stock. We have 100 shares, we bought at $100, it is now at $200.")],
            "category": None,
        }, config=dict(callbacks=[handler]))
        trace.save()
    
        print(f"Response: {response['messages'][-1]}")

if __name__ == "__main__":
    asyncio.run(main())
