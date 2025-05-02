import os
import asyncio
from dotenv import load_dotenv
from judgeval.common.tracer import Tracer, wrap
from anthropic import AsyncAnthropic

# Load environment variables from .env file
load_dotenv()

# Ensure you have ANTHROPIC_API_KEY, JUDGMENT_API_KEY, and JUDGMENT_ORG_ID set
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
judgment_api_key = os.getenv("JUDGMENT_API_KEY")
judgment_org_id = os.getenv("JUDGMENT_ORG_ID")

if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
if not judgment_api_key:
    raise ValueError("JUDGMENT_API_KEY environment variable not set.")
if not judgment_org_id:
    raise ValueError("JUDGMENT_ORG_ID environment variable not set.")

# Instantiate the tracer
tracer = Tracer(project_name="AnthropicStreamDemo", organization_id=judgment_org_id, api_key=judgment_api_key)

# Create and wrap the Anthropic async client
client = AsyncAnthropic(api_key=anthropic_api_key)
wrapped_client = wrap(client) 

@tracer.observe(name="anthropic_stream_test_func", span_type="llm", overwrite=True)
async def stream_anthropic_response(prompt: str):
    """
    Calls the Anthropic API with streaming enabled using the .stream() context manager 
    with a wrapped client and prints the response chunks. 
    The trace should capture usage via the patched context manager.
    """
    try:
        print("\n--- Calling Anthropic API using .stream() context manager ---")
        full_response = ""
        # Use the async with client.messages.stream(...) pattern
        async with wrapped_client.messages.stream(
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100, 
        ) as stream:
            print("Streaming response:")
            async for chunk in stream: # Iterate over the stream provided by the context manager
                # Anthropic specific chunk handling based on documentation
                if chunk.type == "content_block_delta":
                     if chunk.delta.type == "text_delta":
                         text = chunk.delta.text
                         print(text, end="", flush=True)
                         full_response += text
                elif chunk.type == "message_start":
                    # Debug print to confirm usage is accessible here
                    print(f"\n(Stream started via context manager, input tokens: {chunk.message.usage.input_tokens})", end="", flush=True)
                elif chunk.type == "message_delta":
                     print(f" (Delta event via context manager, output tokens so far: {chunk.usage.output_tokens})", end="", flush=True)
                elif chunk.type == "message_stop":
                    print("\n(Stream stopped via context manager)", end="", flush=True)

        print("\n--- End of stream (context manager exited) ---")
        # The @tracer.observe decorator handles saving the trace
        print("Trace should be saved automatically by the decorator.")
        return full_response

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    test_prompt = "Write a very short poem about asynchronous streams."
    result = await stream_anthropic_response(test_prompt)
    if result:
        print(f"\n--- Final Content ---")
        print(result)
    else:
        print("\n--- Streaming failed ---")

if __name__ == "__main__":
    asyncio.run(main()) 