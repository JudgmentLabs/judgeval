import asyncio
import time
import sys
import os
import functools
from unittest.mock import MagicMock, patch
from typing import Dict, Optional, List
import uuid
import json

# Standard library imports needed for the new class
import concurrent.futures
import contextvars
# Needed for partial in the executor

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and mock necessary components before initializing the tracer
from judgeval.common.tracer import Tracer, JudgmentClient, TraceClient, current_trace_var, TraceEntry, TraceManagerClient, TraceThreadPoolExecutor # Import the new class

# Initialize the tracer with test values
tracer = Tracer(
    project_name="complex_async_test"
)

# In this example, we'll use a single trace with spans for all function calls
@tracer.observe(name="root_function")
async def root_function():
    print("Root function starting")
    
    # Direct await call to level 2
    result1 = await level2_function("direct")
    
    # Parallel calls (gather) to level 2 functions
    # These should be level 2 - direct children of root
    # Create two truly parallel functions that both have root_function as parent
    level2_parallel1_task = level2_parallel1("gather1")
    level2_parallel2_task = level2_parallel2("gather2")
    
    # Use trace_gather instead of asyncio.gather to preserve context
    # This ensures parent-child relationships are maintained in parallel tasks
    # result2, result3 = await trace_gather(level2_parallel1_task, level2_parallel2_task) # OLD
    result2, result3 = await asyncio.gather(level2_parallel1_task, level2_parallel2_task) # Use standard gather
    
    
    print("Root function completed")
    return f"Root results: {result1}, {result2}, {result3}"

# Level 2 - Direct child of root
# Using observe with same tracer - this will create spans in the parent trace
@tracer.observe()
async def level2_function(param):
    # Capture this function in a span within the current trace
    print(f"Level 2 function with {param}")

    # Call to level 3
    result = await level3_function(f"{param}_child")

    return f"level2:{result}"

# Level 2 - First parallel function
@tracer.observe()
async def level2_parallel1(param):
    # Capture this function in a span within the current trace
    print(f"Level 2 parallel 1 with {param}")

    # This parallel function makes another parallel call to level 3 functions
    # These should be direct children of level2_parallel1
    # r1, r2 = await trace_gather( # OLD
    r1, r2 = await asyncio.gather( # Use standard gather
        level3_parallel1(f"{param}_1"),
        level3_parallel2(f"{param}_2")
    )

    return f"level2_parallel1:{r1},{r2}"

# Level 2 - Second parallel function
@tracer.observe()
async def level2_parallel2(param):
    # Capture this function in a span within the current trace
    print(f"Level 2 parallel 2 with {param}")

    # Direct await to level 3
    result = await level3_function(f"{param}_direct")

    return f"level2_parallel2:{result}"

# Level 3 - Child of level 2 direct
@tracer.observe()
async def level3_function(param):
    # Capture this function in a span within the current trace
    print(f"Level 3 function with {param}")

    # Call to level 4
    result = await level4_function(f"{param}_deep")

    return f"level3:{result}"

# Level 3 - First parallel function called by level2_parallel1
@tracer.observe()
async def level3_parallel1(param):
    # Capture this function in a span within the current trace
    print(f"Level 3 parallel 1 with {param}")

    # This makes a nested gather call with level 4 functions
    # results = await trace_gather( # OLD
    results = await asyncio.gather( # Use standard gather
        level4_function(f"{param}_a"),
        level4_function(f"{param}_b"),
        level4_function(f"{param}_c")
    )

    return f"level3_p1:{','.join(results)}"

# Level 3 - Second parallel function called by level2_parallel1
@tracer.observe()
async def level3_parallel2(param):
    # Capture this function in a span within the current trace
    print(f"Level 3 parallel 2 with {param}")
    await asyncio.sleep(0.1)

    # Direct call to level 4
    result = await level4_deep_function(f"{param}_deep")

    return f"level3_p2:{result}"

# Level 4 - Deepest regular function
@tracer.observe()
async def level4_function(param):
    # Capture this function in a span within the current trace
    print(f"Level 4 function with {param}")
    await asyncio.sleep(0.05)

    return f"level4:{param}"

# Level 4 - Deep function that calls level 5
@tracer.observe()
async def level4_deep_function(param):
    # Capture this function in a span within the current trace
    print(f"Level 4 deep function with {param}")

    # Call to level 5 (maximum depth)
    result = await level5_function(f"{param}_final")
    test = await fib(5)
    return f"level4_deep:{result}"

@tracer.observe()
async def fib(n):
    if n <= 1:
        return n
    return await fib(n-1) + await fib(n-2)

# Level 5 - Deepest level
@tracer.observe()
async def level5_function(param):
    # Capture this function in a span within the current trace
    print(f"Level 5 function with {param}")
    await asyncio.sleep(0.05)

    return f"level5:{param}"
    
# --- Synchronous ThreadPoolExecutor Test ---

@tracer.observe(name="sync_child_task1")
def sync_child_task1(param):
    """A simple synchronous function to be run in a thread."""
    print(f"SYNC CHILD 1: Received {param}. Sleeping...")
    time.sleep(0.15)
    result = f"Result from sync_child_task1 with {param}"
    print("SYNC CHILD 1: Done.")
    return result

@tracer.observe(name="sync_child_task2")
def sync_child_task2(param1, param2):
    """Another simple synchronous function."""
    print(f"SYNC CHILD 2: Received {param1} and {param2}. Sleeping...")
    time.sleep(0.05)
    result = f"Result from sync_child_task2 with {param1}, {param2}"
    print("SYNC CHILD 2: Done.")
    return result

@tracer.observe(name="sync_parent_func")
def sync_parent_func():
    """This function uses TraceThreadPoolExecutor to run sync tasks."""
    print("SYNC PARENT: Starting...")
    results = []
    # Use the TraceThreadPoolExecutor instead of the standard one
    with TraceThreadPoolExecutor(max_workers=2) as executor:
        print("SYNC PARENT: Submitting tasks to TraceThreadPoolExecutor...")
        future1 = executor.submit(sync_child_task1, "data_for_task1")
        future2 = executor.submit(sync_child_task2, "data1_for_task2", "data2_for_task2")

        print("SYNC PARENT: Waiting for futures...")
        # Wait for futures and collect results (demonstrates typical usage)
        for future in concurrent.futures.as_completed([future1, future2]):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"SYNC PARENT: Generated an exception: {exc}")
                results.append(f"Error: {exc}")

    print("SYNC PARENT: Finished.")
    return results

# --- End Synchronous Test ---

async def main():
    # Run the root function which has deep nesting and nested parallel calls
    start_time = time.time()
    result_async = await root_function()
    end_time = time.time()
    print(f"\nAsync Final result: {result_async}")
    print(f"Async Total execution time: {end_time - start_time:.2f} seconds")
    
    print("\n" + "="*20 + " Starting Sync ThreadPool Test " + "="*20 + "\n")

    # --- Run the synchronous thread pool test ---
    # Note: We run this *outside* the async root_function's trace
    # If we wanted it nested, we'd need @tracer.observe on main or call it from root_function
    # For simplicity, let's trace it separately by calling it directly.
    # The @tracer.observe on sync_parent_func will create its own root trace.
    start_time_sync = time.time()
    result_sync = sync_parent_func() # This will be traced automatically
    end_time_sync = time.time()
    print(f"\nSync Final results: {result_sync}")
    print(f"Sync Total execution time: {end_time_sync - start_time_sync:.2f} seconds")
    # --- End synchronous test call ---

if __name__ == "__main__":
    # Run the complex async example
    asyncio.run(main()) 