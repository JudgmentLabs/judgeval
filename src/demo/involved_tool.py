import os
import json
from dotenv import load_dotenv
from judgeval.tracer import Tracer, wrap
import asteval
import math

load_dotenv()

judgment = Tracer(
    project_name="asteval_demo", 
)

evaluator = asteval.Interpreter()

@judgment.observe(span_type="function", deep_tracing=True)
def main(code: str):
    """Main function to evaluate a mathematical expression."""
    try:
        return evaluator(code)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


code = """
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}
visited = set()
def dfs(node):
    if node not in visited:
        print(node, end=' ')
        visited.add(node)
        for neighbor in graph[node]:
            dfs(neighbor)

dfs('A')
"""

if __name__ == "__main__":
    main(code)
    