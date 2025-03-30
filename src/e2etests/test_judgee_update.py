"""
Tests for judgee update functionality
"""

import pytest
import os
import time
import asyncio
import sys
import httpx
from dotenv import load_dotenv

# Add the src directory to Python path to make imports work
src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

# Load environment variables from .env file
load_dotenv()

# Get server URL from environment
SERVER_URL = os.getenv("JUDGMENT_API_URL", "http://localhost:8000")
TEST_API_KEY = os.getenv("JUDGMENT_API_KEY")

if not TEST_API_KEY:
    print("❌ Error: JUDGMENT_API_KEY not set in .env file")
    sys.exit(1)

@pytest.mark.asyncio
async def verify_server(server_url: str = SERVER_URL):
    """Helper function to verify server is running."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{server_url}/health/")
            if response.status_code == 200:
                print("✓ Server health check passed")
                return True
    except Exception as e:
        print(f"❌ Server not running at {server_url}. Please start with: uvicorn server.main:app --reload")
        print(f"Error: {e}")
        return False
    return False

@pytest.mark.asyncio
async def debug_server_state(client):
    """Helper function to get server state for debugging."""
    try:
        # Check server health
        health_response = await client.get(f"{SERVER_URL}/health/")
        print("\n🔍 Server Status:")
        print(f"- Health endpoint: {'✓ OK' if health_response.status_code == 200 else '❌ Failed'}")
        print(f"- Status code: {health_response.status_code}")
        
        # Check current count
        count_response = await client.get(
            f"{SERVER_URL}/judgees/count/",
            params={"judgment_api_key": TEST_API_KEY}
        )
        print("- Current count:", count_response.json().get("judgees_ran", "Error getting count"))
        
        # Check API key
        print(f"- Using API key: {TEST_API_KEY[:8]}...")
        
        return True
    except Exception as e:
        print(f"\n❌ Failed to get server state: {str(e)}")
        print("Debug tips:")
        print("1. Check if server is running (uvicorn server.main:app --reload)")
        print("2. Verify SERVER_URL in .env file")
        print("3. Check server logs for errors")
        return False

@pytest.mark.asyncio
async def test_single_judgee_increment():
    """Test basic single judgee increment and reset."""
    print("\n" + "="*50)
    print("Test 1: Single Judgee Increment and Reset")
    print("="*50)
    
    if not await verify_server():
        return False
    
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # Add debug state before test
            print("\n📊 Pre-test State:")
            await debug_server_state(client)

            # Reset judgee count at start
            print("\n📝 Step 1: Resetting initial count...")
            response = await client.post(
                f"{SERVER_URL}/judgees/reset/",
                params={"judgment_api_key": TEST_API_KEY}
            )
            if response.status_code != 200:
                print("❌ Reset failed")
                return False
            print("✓ Reset successful")

            # Initial count verification
            print("\n📝 Step 2: Verifying initial count...")
            response = await client.get(
                f"{SERVER_URL}/judgees/count/",
                params={"judgment_api_key": TEST_API_KEY}
            )
            if response.status_code != 200:
                print("❌ Count verification failed")
                return False
            
            initial_count = response.json()["judgees_ran"]
            print(f"✓ Initial count verified: {initial_count}")
            if initial_count != 0:
                print("❌ Initial count should be 0")
                return False

            # Run evaluation with single scorer
            print("\n📝 Step 3: Running evaluation with single scorer...")
            eval_data = {
                "judgment_api_key": TEST_API_KEY,
                "examples": [{
                    "input": "test input",
                    "actual_output": "test output",
                    "expected_output": "test output",
                    "context": [],
                    "retrieval_context": [],
                    "additional_metadata": {},
                    "tools_called": [],
                    "expected_tools": []
                }],
                "scorers": [
                    {
                        "score_type": "faithfulness",
                        "threshold": 1.0,
                        "kwargs": {}
                    }
                ],
                "model": "gpt-3.5-turbo",
                "log_results": True,
                "project_name": "test_project",
                "eval_run_name": "test_single_increment"
            }

            response = await client.post(
                f"{SERVER_URL}/evaluate/",
                json=eval_data,
                timeout=60.0
            )
            if response.status_code != 200:
                print(f"\n❌ Evaluation failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                print("\nDebug Information:")
                print("1. Check request format:")
                print(f"- Examples: {len(eval_data['examples'])} provided")
                print(f"- Scorers: {len(eval_data['scorers'])} configured")
                print("2. Verify scorer configuration:")
                for scorer in eval_data["scorers"]:
                    print(f"- {scorer['score_type']}: threshold={scorer['threshold']}")
                print("3. Common issues:")
                print("- Missing required fields in request")
                print("- Invalid scorer configuration")
                print("- Server processing error (check logs)")
                return False

            # Verify increment
            print("\n📝 Step 4: Verifying single judgee increment...")
            response = await client.get(
                f"{SERVER_URL}/judgees/count/",
                params={"judgment_api_key": TEST_API_KEY}
            )
            if response.status_code != 200:
                print("❌ Count verification failed")
                return False
            
            final_count = response.json()["judgees_ran"]
            print(f"✓ Count after single increment: {final_count}")
            if final_count != 1:
                print("❌ Expected count to be 1")
                return False

            # Test reset
            print("\n📝 Step 5: Testing reset functionality...")
            response = await client.post(
                f"{SERVER_URL}/judgees/reset/",
                params={"judgment_api_key": TEST_API_KEY}
            )
            if response.status_code != 200:
                print("❌ Reset failed")
                return False
            
            response = await client.get(
                f"{SERVER_URL}/judgees/count/",
                params={"judgment_api_key": TEST_API_KEY}
            )
            reset_count = response.json()["judgees_ran"]
            print(f"✓ Count after reset: {reset_count}")
            if reset_count != 0:
                print("❌ Expected count to be 0 after reset")
                return False
            
            # Add post-test debug state
            print("\n📊 Post-test State:")
            await debug_server_state(client)

            print("\n✅ Single increment test completed successfully!")
            return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        print("\nDebug Information:")
        print(f"1. Error type: {type(e).__name__}")
        print("2. Error location:", e.__traceback__.tb_frame.f_code.co_name)
        print("3. Common causes:")
        print("- Network connectivity issues")
        print("- Server timeout")
        print("- Invalid request format")
        print("\n4. Environment Check:")
        print(f"- SERVER_URL: {SERVER_URL}")
        print(f"- API Key configured: {'Yes' if TEST_API_KEY else 'No'}")
        print("\n5. Next steps:")
        print("- Check server logs")
        print("- Verify .env configuration")
        print("- Run server in debug mode")
        return False

@pytest.mark.asyncio
async def test_multiple_judgee_increment():
    """Test multiple judgee increments with various scorers."""
    print("\n" + "="*50)
    print("Test 2: Multiple Judgee Increment")
    print("="*50)
    
    if not await verify_server():
        return False
    
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # Reset count
            print("\n📝 Step 1: Resetting initial count...")
            response = await client.post(
                f"{SERVER_URL}/judgees/reset/",
                params={"judgment_api_key": TEST_API_KEY}
            )
            if response.status_code != 200:
                print("❌ Reset failed")
                return False
            print("✓ Reset successful")

            # Run evaluation with multiple scorers
            print("\n📝 Step 2: Running evaluation with multiple scorers...")
            eval_data = {
                "judgment_api_key": TEST_API_KEY,
                "examples": [{
                    "input": "What is the capital of France?",
                    "actual_output": "Paris is the capital of France.",
                    "expected_output": "Paris",
                    "context": ["Geography"],
                    "retrieval_context": ["Paris is the capital of France"],
                    "additional_metadata": {},
                    "tools_called": [],
                    "expected_tools": []
                }],
                "scorers": [
                    {
                        "score_type": "faithfulness",
                        "threshold": 1.0,
                        "kwargs": {}
                    },
                    {
                        "score_type": "answer_relevancy",
                        "threshold": 1.0,
                        "kwargs": {}
                    },
                    {
                        "score_type": "hallucination",
                        "threshold": 1.0,
                        "kwargs": {}
                    }
                ],
                "model": "gpt-3.5-turbo",
                "log_results": True,
                "project_name": "test_project",
                "eval_run_name": "test_multiple_increment"
            }

            response = await client.post(
                f"{SERVER_URL}/evaluate/",
                json=eval_data,
                timeout=60.0
            )
            if response.status_code != 200:
                print(f"❌ Evaluation failed: {response.text}")
                return False
            print("✓ Evaluation completed successfully")

            # Verify multiple increment
            print("\n📝 Step 3: Verifying multiple judgee increment...")
            response = await client.get(
                f"{SERVER_URL}/judgees/count/",
                params={"judgment_api_key": TEST_API_KEY}
            )
            if response.status_code != 200:
                print("❌ Count verification failed")
                return False
            
            count = response.json()["judgees_ran"]
            print(f"✓ Count after multiple scorers: {count}")
            if count != 3:
                print("❌ Expected count to be 3 (one for each scorer)")
                return False

            print("\n✅ Multiple increment test completed successfully!")
            return True

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

@pytest.mark.asyncio
async def test_zero_scorer_case():
    """Test that evaluation with no scorers doesn't affect count."""
    print("\n" + "="*50)
    print("Test 3: Zero Scorer Case")
    print("="*50)
    
    if not await verify_server():
        return False
    
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # Reset and verify initial count
            print("\n📝 Step 1: Resetting initial count...")
            await client.post(
                f"{SERVER_URL}/judgees/reset/",
                params={"judgment_api_key": TEST_API_KEY}
            )
            
            initial_count = (await client.get(
                f"{SERVER_URL}/judgees/count/",
                params={"judgment_api_key": TEST_API_KEY}
            )).json()["judgees_ran"]
            print(f"✓ Initial count: {initial_count}")

            # Run evaluation with no scorers
            print("\n📝 Step 2: Running evaluation with no scorers...")
            eval_data = {
                "judgment_api_key": TEST_API_KEY,
                "examples": [{
                    "input": "test input",
                    "actual_output": "test output",
                    "expected_output": "test output",
                    "context": [],
                    "retrieval_context": [],
                    "additional_metadata": {},
                    "tools_called": [],
                    "expected_tools": []
                }],
                "scorers": [],  # Empty scorers list
                "model": "gpt-3.5-turbo",
                "log_results": True,
                "project_name": "test_project",
                "eval_run_name": "test_zero_scorer"
            }

            response = await client.post(
                f"{SERVER_URL}/evaluate/",
                json=eval_data,
                timeout=60.0
            )
            print("✓ Evaluation completed")

            # Verify count didn't change
            final_count = (await client.get(
                f"{SERVER_URL}/judgees/count/",
                params={"judgment_api_key": TEST_API_KEY}
            )).json()["judgees_ran"]
            print(f"✓ Final count: {final_count}")
            
            if final_count != initial_count:
                print(f"❌ Count changed unexpectedly: {initial_count} -> {final_count}")
                return False

            print("\n✅ Zero scorer test completed successfully!")
            return True

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

@pytest.mark.asyncio
async def test_multiple_examples():
    """Test judgee counting with multiple examples."""
    print("\n" + "="*50)
    print("Test 4: Multiple Examples")
    print("="*50)
    
    if not await verify_server():
        return False
    
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # Reset count
            print("\n📝 Step 1: Resetting count...")
            await client.post(
                f"{SERVER_URL}/judgees/reset/",
                params={"judgment_api_key": TEST_API_KEY}
            )
            
            # Run evaluation with multiple examples
            print("\n📝 Step 2: Running evaluation with multiple examples...")
            eval_data = {
                "judgment_api_key": TEST_API_KEY,
                "examples": [
                    {
                        "input": "What is 2+2?",
                        "actual_output": "4",
                        "expected_output": "4",
                        "context": ["Math"],
                        "retrieval_context": ["Basic arithmetic"],
                    },
                    {
                        "input": "What is the capital of France?",
                        "actual_output": "Paris",
                        "expected_output": "Paris",
                        "context": ["Geography"],
                        "retrieval_context": ["European capitals"],
                    }
                ],
                "scorers": [
                    {
                        "score_type": "faithfulness",
                        "threshold": 1.0,
                        "kwargs": {}
                    },
                    {
                        "score_type": "answer_relevancy",
                        "threshold": 1.0,
                        "kwargs": {}
                    }
                ],
                "model": "gpt-3.5-turbo",
                "log_results": True,
                "project_name": "test_project",
                "eval_run_name": "test_multiple_examples"
            }

            response = await client.post(
                f"{SERVER_URL}/evaluate/",
                json=eval_data,
                timeout=60.0
            )
            print("✓ Evaluation completed")

            # Verify count (should be examples × scorers)
            final_count = (await client.get(
                f"{SERVER_URL}/judgees/count/",
                params={"judgment_api_key": TEST_API_KEY}
            )).json()["judgees_ran"]
            
            expected_count = len(eval_data["examples"]) * len(eval_data["scorers"])
            print(f"\nCount Analysis:")
            print(f"- Number of examples: {len(eval_data['examples'])}")
            print(f"- Number of scorers: {len(eval_data['scorers'])}")
            print(f"- Expected count: {expected_count}")
            print(f"- Actual count: {final_count}")
            
            if final_count != expected_count:
                print("❌ Count mismatch")
                return False

            print("\n✅ Multiple examples test completed successfully!")
            return True

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

@pytest.mark.asyncio
async def test_rapid_evaluations():
    """Test rapid sequential evaluations."""
    print("\n" + "="*50)
    print("Test 5: Rapid Evaluations")
    print("="*50)
    
    if not await verify_server():
        return False
    
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # Reset count
            await client.post(
                f"{SERVER_URL}/judgees/reset/",
                params={"judgment_api_key": TEST_API_KEY}
            )
            
            # Basic evaluation data
            eval_data = {
                "judgment_api_key": TEST_API_KEY,
                "examples": [{
                    "input": "test input",
                    "actual_output": "test output",
                    "expected_output": "test output",
                    "context": [],
                    "retrieval_context": [],
                }],
                "scorers": [
                    {
                        "score_type": "faithfulness",
                        "threshold": 1.0,
                        "kwargs": {}
                    }
                ],
                "model": "gpt-3.5-turbo",
                "log_results": True,
                "project_name": "test_project",
            }

            # Run multiple evaluations rapidly
            print("\n📝 Running rapid evaluations...")
            total_runs = 5
            for i in range(total_runs):
                eval_data["eval_run_name"] = f"rapid_test_{i}"
                response = await client.post(
                    f"{SERVER_URL}/evaluate/",
                    json=eval_data,
                    timeout=60.0
                )
                print(f"✓ Evaluation {i+1}/{total_runs} completed")

            # Verify final count
            final_count = (await client.get(
                f"{SERVER_URL}/judgees/count/",
                params={"judgment_api_key": TEST_API_KEY}
            )).json()["judgees_ran"]
            
            expected_count = total_runs  # One scorer per evaluation
            print(f"\nCount Analysis:")
            print(f"- Number of evaluations: {total_runs}")
            print(f"- Expected count: {expected_count}")
            print(f"- Actual count: {final_count}")
            
            if final_count != expected_count:
                print("❌ Count mismatch in rapid evaluations")
                return False

            print("\n✅ Rapid evaluations test completed successfully!")
            return True

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

async def main():
    print("\n🚀 Starting Judgee Tracking Tests")
    print("="*50)
    
    # Environment check
    print("\n🔧 Environment Check:")
    print(f"- Server URL: {SERVER_URL}")
    print(f"- API Key: {'Configured' if TEST_API_KEY else 'Missing'}")
    
    # Run all tests with timing
    results = {}
    for test_name, test_func in {
        "Single Increment": test_single_judgee_increment,
        "Multiple Increment": test_multiple_judgee_increment,
        "Zero Scorer": test_zero_scorer_case,
        "Multiple Examples": test_multiple_examples,
        "Rapid Evaluations": test_rapid_evaluations
    }.items():
        start_time = time.time()
        success = await test_func()
        duration = time.time() - start_time
        results[test_name] = {"success": success, "duration": duration}
    
    # Print detailed summary
    print("\n"+"="*50)
    print("Detailed Test Summary")
    print("="*50)
    all_passed = True
    for test_name, result in results.items():
        status = "✅ Passed" if result["success"] else "❌ Failed"
        duration = f"{result['duration']:.2f}s"
        print(f"{test_name}: {status} ({duration})")
        all_passed = all_passed and result["success"]
    
    if not all_passed:
        print("\n❌ Some tests failed!")
        print("Common issues:")
        print("   - Server not running")
        print("   - Network connectivity")
        print("   - Invalid request format")
        print("   - Scorer configuration errors")
        sys.exit(1)
    else:
        print("\n✅ All tests passed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
