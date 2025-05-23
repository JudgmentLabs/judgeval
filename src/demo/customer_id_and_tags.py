import os
import random
import json
from judgeval.common.tracer import Tracer
from dotenv import load_dotenv

load_dotenv()

judgment = Tracer(
    api_key=os.getenv("JUDGMENT_API_KEY"),
    organization_id=os.getenv("JUDGMENT_ORG_ID"),
    project_name="metadata-test"
)

def get_customer_id():
    return f"customer-{random.randint(1000, 9999)}"

@judgment.observe(name="test_metadata")
def test_metadata():
    # Generate a customer ID
    customer_id = get_customer_id()
    print(f"Generated customer ID: {customer_id}")
    
    # Define tags
    tags = ["demo", "development"]
    print(f"Using tags: {tags}")
    
    # Set both customer ID and tags at once
    judgment.set_metadata(
        customer_id=customer_id,
        tags=tags
    )
    
    # Verify the metadata was set correctly
    trace = judgment.get_current_trace()
    print(f"Trace customer_id: {trace.customer_id}")
    print(f"Trace tags: {trace.tags}")
    
    # Print all metadata
    print("\nTrace metadata:")
    judgment.print_metadata()
    
    # Do some work
    print("\nPerforming test operations...")
    result = f"Test completed with customer ID: {customer_id} and tags: {tags}"
    print(result)
    
    # Save trace data to a file for inspection
    trace_data = {
        "trace_id": trace.trace_id,
        "name": trace.name,
        "project_name": trace.project_name,
        "created_at": trace.start_time,
        "customer_id": trace.customer_id,
        "tags": trace.tags
    }
    
    with open("trace_metadata.json", "w") as f:
        json.dump(trace_data, f, indent=2)
    print("\nTrace metadata saved to trace_metadata.json for inspection")
    
    return result

if __name__ == "__main__":
    print("\n=== Metadata Test ===\n")
    
    try:
        # Run the test function
        result = test_metadata()
        print(f"\nResult: {result}")
        print("\nCheck the Judgment dashboard to verify the metadata was saved.")
    except Exception as e:
        print(f"\nError: {e}")