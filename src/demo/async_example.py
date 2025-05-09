import asyncio
import time
import random
from typing import List, Dict, Any
import concurrent.futures
from judgeval.tracer import Tracer, wrap
from judgeval.common.tracer import TraceThreadPoolExecutor

judgment = Tracer(project_name="complex_async_test")

class DataProcessor:
    """A class with methods that will be traced"""
    
    @judgment.observe(name="process_batch", span_type="method")
    async def process_batch(self, batch_id: int, items: List[str]) -> Dict[str, Any]:
        """Process a batch of items with multiple sub-operations"""
        print(f"[Batch {batch_id}] Starting processing of {len(items)} items")
        
        
        tasks = [self.process_item(batch_id, i, item) for i, item in enumerate(items)]
        results = await asyncio.gather(*tasks)
        
        success_count = sum(1 for r in results if r["status"] == "success")
        
        await self.post_process(batch_id, results)
        
        return {
            "batch_id": batch_id,
            "total_items": len(items),
            "success_count": success_count,
            "results": results
        }
    
    @judgment.observe(name="process_item", span_type="method")
    async def process_item(self, batch_id: int, item_id: int, data: str) -> Dict[str, Any]:
        """Process a single item with simulated work"""
        print(f"[Batch {batch_id}] Processing item {item_id}: {data}")
        
        processing_time = random.uniform(0.05, 0.2)
        await asyncio.sleep(processing_time)
        
        success = random.random() > 0.1
        
        if success:
            await self.validate_result(batch_id, item_id, data)
        
        return {
            "item_id": item_id,
            "batch_id": batch_id,
            "data": data,
            "status": "success" if success else "failure",
            "processing_time": processing_time
        }
    
    @judgment.observe(name="validate_result", span_type="method")
    async def validate_result(self, batch_id: int, item_id: int, data: str) -> bool:
        """Validate the result of processing an item"""
        print(f"[Batch {batch_id}] Validating item {item_id}")
        await asyncio.sleep(0.05)
        return True
    
    @judgment.observe(name="post_process", span_type="method")
    async def post_process(self, batch_id: int, results: List[Dict[str, Any]]) -> None:
        """Perform post-processing on batch results"""
        print(f"[Batch {batch_id}] Post-processing {len(results)} results")
        await asyncio.sleep(0.1)

@judgment.observe(name="fetch_data", span_type="function")
async def fetch_data(source_id: int, limit: int = 10) -> List[str]:
    """Simulate fetching data from an external source"""
    print(f"Fetching data from source {source_id} (limit: {limit})")
    
    await asyncio.sleep(random.uniform(0.2, 0.5))
    
    data = [f"data_{source_id}_{i}" for i in range(limit)]
    
    if random.random() > 0.7:
        metadata = await fetch_metadata(source_id)
        print(f"Got metadata for source {source_id}: {metadata}")
    
    return data

@judgment.observe(name="fetch_metadata", span_type="function")
async def fetch_metadata(source_id: int) -> Dict[str, Any]:
    """Fetch metadata for a data source"""
    await asyncio.sleep(0.1)
    return {
        "source_id": source_id,
        "timestamp": time.time(),
        "record_count": random.randint(100, 1000)
    }

@judgment.observe(name="create_batches", span_type="function")
async def create_batches(data: List[str], batch_size: int = 3) -> List[List[str]]:
    """Split data into batches for processing"""
    print(f"Creating batches from {len(data)} items (batch size: {batch_size})")
    
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i+batch_size])
        
    tasks = [preprocess_batch(i, batch) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks)
    return batches

@judgment.observe(name="preprocess_batch", span_type="function")
async def preprocess_batch(batch_id: int, batch: List[str]) -> None:
    """Preprocess a batch of data"""
    print(f"Preprocessing batch {batch_id} with {len(batch)} items")
    await asyncio.sleep(0.05)

def sync_heavy_computation(data: str) -> Dict[str, Any]:
    """A CPU-bound synchronous function"""
    print(f"Performing heavy computation on: {data}")
    time.sleep(0.2)  
    return {"input": data, "result": f"processed_{data}", "timestamp": time.time()}

@judgment.observe(name="run_sync_tasks", span_type="function")
async def run_sync_tasks(data_list: List[str]) -> List[Dict[str, Any]]:
    """Run synchronous tasks in a thread pool with tracing"""
    print(f"Running {len(data_list)} synchronous tasks in thread pool")
    
    results = []
    
    with TraceThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(sync_heavy_computation, data) for data in data_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Task failed: {e}")
                results.append({"status": "error", "error": str(e)})
    
    
    await post_process_sync_results(results)
    
    return results

@judgment.observe(name="post_process_sync_results", span_type="function")
async def post_process_sync_results(results: List[Dict[str, Any]]) -> None:
    """Post-process results from synchronous computations"""
    print(f"Post-processing {len(results)} sync results")
    await asyncio.sleep(0.1)


@judgment.observe(name="error_prone_function", span_type="function")
async def error_prone_function(chance_of_error: float = 0.3) -> Dict[str, Any]:
    """A function that might throw an error, to test error handling in traces"""
    print(f"Running error-prone function (error chance: {chance_of_error})")
    await asyncio.sleep(0.1)
    
    if random.random() < chance_of_error:
        error_msg = "Simulated random failure"
        print(f"Error occurred: {error_msg}")
        raise RuntimeError(error_msg)
    
    return {"status": "success", "timestamp": time.time()}

@judgment.observe(name="with_error_handling", span_type="function")
async def with_error_handling() -> Dict[str, Any]:
    """Run multiple error-prone functions with proper error handling"""
    print("Starting function with error handling")
    results = {"successes": 0, "failures": 0}

    for i in range(3):
        try:
            await error_prone_function(0.4)
            results["successes"] += 1
        except Exception as e:
            results["failures"] += 1
            print(f"Caught error: {e}")
            
            await handle_error(str(e))
    
    return results

@judgment.observe(name="handle_error", span_type="function")
async def handle_error(error_msg: str) -> None:
    """Handle an error from another function"""
    print(f"Handling error: {error_msg}")
    await asyncio.sleep(0.05)


@judgment.observe(name="orchestrate", span_type="function")
async def orchestrate(num_sources: int = 3, items_per_source: int = 5) -> Dict[str, Any]:
    """Orchestrate the entire process end to end"""
    print(f"Starting orchestration with {num_sources} sources")
    start_time = time.time()
    
    data_sources = list(range(1, num_sources + 1))
    fetch_tasks = [fetch_data(source_id, items_per_source) for source_id in data_sources]
    all_data_lists = await asyncio.gather(*fetch_tasks)
    
    all_data = [item for sublist in all_data_lists for item in sublist]
    batches = await create_batches(all_data, batch_size=4)
    
    processor = DataProcessor()
    batch_processing_tasks = [
        processor.process_batch(i, batch) for i, batch in enumerate(batches)
    ]
    batch_results = await asyncio.gather(*batch_processing_tasks)
    
    
    sync_data = [f"sync_item_{i}" for i in range(5)]
    sync_results = await run_sync_tasks(sync_data)
    
    
    error_results = await with_error_handling()
    
    end_time = time.time()
    return {
        "sources_processed": num_sources,
        "total_items": len(all_data),
        "batch_count": len(batches),
        "batch_results": batch_results,
        "sync_results": sync_results,
        "error_handling": error_results,
        "execution_time": end_time - start_time
    }

async def main():
    """Main entry point for the application"""
    print("=== Starting Complex Async Example ===")
    
    
    results = await orchestrate(num_sources=3, items_per_source=5)
    
    print("\n=== Execution Summary ===")
    print(f"Total execution time: {results['execution_time']:.2f} seconds")
    print(f"Processed {results['total_items']} items in {results['batch_count']} batches")
    print(f"Success rate: {sum(batch['success_count'] for batch in results['batch_results'])}/{results['total_items']}")
    print(f"Error handling results: {results['error_handling']}")
    print("=== Finished Complex Async Example ===")

if __name__ == "__main__":
    asyncio.run(main())
