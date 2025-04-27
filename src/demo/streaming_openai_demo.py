import os
import openai
from dotenv import load_dotenv
from judgeval.common.tracer import Tracer, wrap
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Ensure you have OPENAI_API_KEY and JUDGMENT_API_KEY/JUDGMENT_ORG_ID set
openai_api_key = os.getenv("OPENAI_API_KEY")
judgment_api_key = os.getenv("JUDGMENT_API_KEY")
judgment_org_id = os.getenv("JUDGMENT_ORG_ID")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
if not judgment_api_key:
    raise ValueError("JUDGMENT_API_KEY environment variable not set.")
if not judgment_org_id:
    raise ValueError("JUDGMENT_ORG_ID environment variable not set.")

# Instantiate the tracer (uses singleton pattern)
tracer = Tracer()

# # Create and wrap the OpenAI client
client = OpenAI(api_key=openai_api_key)
wrapped_client = wrap(client)

@tracer.observe(name="streaming_openai_demo_trace", span_type="llm")
def stream_openai_response(prompt: str):
    """
    Calls the OpenAI API with streaming enabled using a wrapped client and prints the response chunks.
    """
    try:
        stream = wrapped_client.chat.completions.create(
            model="gpt-4", # Or your preferred model
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        print("Streaming response:")
        full_response = ""
        for chunk in stream:
            # Check if choices exist and delta content is not None before accessing
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="")
                full_response += content
        print("\n--- End of stream ---")
        return full_response

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    user_prompt = "Explain the concept of quantum entanglement in simple terms."
    final_output = stream_openai_response(user_prompt)
    # Optionally print the final accumulated output
    if final_output:
       print("\n--- Accumulated Output ---")
       print(final_output) 