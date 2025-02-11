import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.tools import tool
from langchain.agents import Tool, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools.render import render_text_description_and_args
from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

class DetailedDebugHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs):
        print(f"\n🔄 Chain started: {serialized.get('name', 'unnamed')}")
        print(f"Inputs: {inputs}")
    
    def on_chain_end(self, outputs: dict, **kwargs):
        print(f"\n✅ Chain finished: {outputs}")
    
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        print(f"\n🔧 Tool started: {serialized.get('name', 'unnamed')}")
        print(f"Input: {input_str}")
    
    def on_tool_end(self, output: str, **kwargs):
        print(f"Tool output: {output}")
    
    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        print(f"\n🤖 LLM started")
        print(f"Prompt: {prompts[0]}")
    
    def on_llm_end(self, response, **kwargs):
        print(f"LLM response: {response}")

class AgenticRAG:
    def __init__(self, pdf_path: str):
        # Load environment variables
        load_dotenv()
        
        # Initialize components
        self.pdf_path = pdf_path
        self.documents = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.tools = None
        self.agent_executor = None
        
        # Setup the pipeline
        self._load_and_split_documents()
        self._setup_vectorstore()
        self._setup_tools()
        self._setup_agent()

    def _load_and_split_documents(self):
        # Load PDF
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self.documents = text_splitter.split_documents(documents)

    def _setup_vectorstore(self):
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Create vectorstore
        self.vectorstore = FAISS.from_documents(self.documents, embeddings)
        self.retriever = self.vectorstore.as_retriever()

    def _setup_tools(self):
        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
        
        # Setup web search
        web_search_tool = TavilySearchResults(k=10)
        
        # Define vector search function
        def vector_search(query: str):
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever
            )
            return qa_chain.run(query)

        # Create tool decorators
        @tool
        def vector_search_tool(query: str) -> str:
            """Tool for searching the vector store."""
            return vector_search(query)

        @tool
        def web_search_tool_func(query: str) -> str:
            """Tool for performing web search."""
            return web_search_tool.run(query)

        # Define tools list
        self.tools = [
            Tool(
                name="VectorStoreSearch",
                func=vector_search_tool,
                description="Use this to search the vector store for information."
            ),
            Tool(
                name="WebSearch",
                func=web_search_tool_func,
                description="Use this to perform a web search for information."
            ),
        ]

    def _setup_agent(self):
        # Define system and human prompts
        system_prompt = (
            "Respond to the human as helpfully and accurately as possible. "
            "You have access to the following tools: {tools}\n"
            "Always try the \"VectorStoreSearch\" tool first. Only use \"WebSearch\" "
            "if the vector store does not contain the required information.\n"
            "Use a json blob to specify a tool by providing an action key (tool name) "
            "and an action_input key (tool input).\n"
            "Valid \"action\" values: \"Final Answer\" or {tool_names}\n"
            "Provide only ONE action per $JSON_BLOB, as shown:\n"
            "```\n"
            "{{\n"
            "  \"action\": $TOOL_NAME,\n"
            "  \"action_input\": $INPUT\n"
            "}}\n"
            "```\n"
            "Follow this format:\n"
            "Question: input question to answer\n"
            "Thought: consider previous and subsequent steps\n"
            "Action:\n"
            "```\n"
            "$JSON_BLOB\n"
            "```\n"
            "Observation: action result\n"
            "... (repeat Thought/Action/Observation N times)\n"
            "Thought: I know what to respond\n"
            "Action:\n"
            "```\n"
            "{{\n"
            "  \"action\": \"Final Answer\",\n"
            "  \"action_input\": \"Final response to human\"\n"
            "}}\n"
            "```\n"
            "Begin! Reminder to ALWAYS respond with a valid json blob of a single action.\n"
            "Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation"
        )

        human_prompt = (
            "{input}\n"
            "{agent_scratchpad}\n"
            "(reminder to always respond in a JSON blob)"
        )

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt),
        ])

        # Add tool descriptions to prompt
        prompt = prompt.partial(
            tools=render_text_description_and_args(list(self.tools)),
            tool_names=", ".join([t.name for t in self.tools]),
        )

        # Create the agent chain
        chain = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
            )
            | prompt
            | self.llm
            | JSONAgentOutputParser()
        )

        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=chain,
            tools=self.tools,
            handle_parsing_errors=True,
            verbose=True,
            return_intermediate_steps=True
        )

    def query(self, input_text: str, debug: bool = False) -> dict:
        """
        Query the agent with input text
        Returns a dictionary containing the input, output, and intermediate steps
        
        Args:
            input_text: The query text
            debug: If True, prints detailed chain execution including tokens and cost
        """
        callbacks = [DetailedDebugHandler()] if debug else []
        
        with get_openai_callback() as cb:
            result = self.agent_executor.invoke(
                {"input": input_text},
                callbacks=callbacks
            )

            print(f"[DEBUG] Query result: {result}")
            
            if debug:
                print("\n=== Execution Statistics ===")
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost (USD): ${cb.total_cost}")
                print("========================")
        
        return {
            "input": input_text,
            "output": result["output"],
            "intermediate_steps": result["intermediate_steps"]
        }

    def query_multiple(self, queries: list[str], debug: bool = False) -> list[dict]:
        """
        Process multiple queries and return their contexts and responses
        """
        results = []
        contexts = []
        intermediate_steps = []
        
        for query in queries:
            # Get vector store contexts
            vector_contexts = self.retriever.get_relevant_documents(query)
            if vector_contexts:
                context_texts = [doc.page_content for doc in vector_contexts]
                contexts.append(context_texts)
            else:
                print(f"[DEBUG] No relevant information in vector store for query: {query}. "
                      "Falling back to web search.")
                web_results = self.tools[1].run(query)
                contexts.append([web_results])

            # Get the agent response
            result = self.query(query, debug=debug)
            results.append(result['output'])
            intermediate_steps.append(result['intermediate_steps'])
            
        return {
            "query": queries,
            "response": results,
            "context": contexts,
            "intermediate_steps": intermediate_steps
        }

def main():
    # Example usage
    pdf_path = os.path.join(os.path.dirname(__file__), "tesla_q3.pdf") # Update with your PDF path
    rag = AgenticRAG(pdf_path)
    
    # Example queries
    queries = [
        "What milestones did the Shanghai factory achieve in Q3 2024?",
        "Tesla stock market summary for 2024?"
    ]
    
    # Run individual query
    print("\nSingle Query Example:")
    result = rag.query(queries[0], debug=True)
    print(f"Query: {queries[0]}")
    print(f"Response: {result['output']}")
    print(f"Intermediate Steps: {result['intermediate_steps']}")
    print("-" * 80)
    
    # Run multiple queries
    print("\nMultiple Queries Example:")
    results = rag.query_multiple(queries, debug=True)
    for i, (q, r) in enumerate(zip(results["query"], results["response"])):
        print(f"\nQuery: {q}")
        print(f"Response: {r}")
        print(f"Intermediate Steps: {results['intermediate_steps'][i]}")
        print("-" * 80)

if __name__ == "__main__":
    main()
