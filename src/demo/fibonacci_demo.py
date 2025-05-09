import os
from dotenv import load_dotenv
from judgeval.tracer import Tracer, wrap

load_dotenv()

judgment = Tracer(
    api_key=os.getenv("JUDGMENT_API_KEY"),
    project_name="fibonacci_demo", 
)

class Agent:
    def fibonacci(self, n: int):
        """Calculate the nth Fibonacci number recursively."""
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            return self.fibonacci(n-1) + self.fibonacci(n-2)

@judgment.observe(span_type="function", deep_tracing=True)
def main(n: int):
    """Main function to calculate Fibonacci number."""
    agent1 = Agent()
    agent2 = Agent()
    result = agent1.fibonacci(n)
    result2 = agent2.fibonacci(n)

    # This should not be traced
    print(f"The {n}th Fibonacci number is: {result}")
    
    return result

if __name__ == "__main__":
    result, trace = main(8)
    # print(f"{result=}")
    print(f"{trace=}")
    
    for span in trace['entries']:
        if span['span_type'] == "tool":
            print(f"{span=}")
    
    # Go through the tools
    
    # Actually log the results - do we have to put it into a ScoringResult manually then call log_results... :)
