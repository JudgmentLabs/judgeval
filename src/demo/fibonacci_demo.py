import os
from dotenv import load_dotenv
from judgeval.tracer import Tracer, wrap

load_dotenv()

judgment = Tracer(
    api_key=os.getenv("JUDGMENT_API_KEY"),
    project_name="fibonacci_demo", 
)

def fibonacci(n: int):
    """Calculate the nth Fibonacci number recursively."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

@judgment.observe(span_type="function", deep_tracing=True)
def main(n: int):
    """Main function to calculate Fibonacci number."""
    result = fibonacci(n)

    # This should not be traced
    print(f"The {n}th Fibonacci number is: {result}")
    
    return result

if __name__ == "__main__":
    main(8)