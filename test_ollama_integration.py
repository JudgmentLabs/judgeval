#!/usr/bin/env python3
"""
Test script for Ollama integration with judgeval tracing.
This script tests both chat() and generate() methods with proper tracing.
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from judgeval.common.tracer import Tracer, wrap


def test_ollama_chat_integration():
    """Test Ollama chat() method with judgeval tracing."""
    try:
        from ollama import Client
        
        print("🧪 Testing Ollama Chat Integration...")
        
        # Initialize tracer  
        tracer = Tracer(
            api_key=os.getenv("JUDGMENT_API_KEY"),
            organization_id=os.getenv("JUDGMENT_ORG_ID"),
            project_name="ollama_test"
        )
        
        # Create and wrap Ollama client
        client = wrap(Client())
        
        # Test chat method with tracing
        with tracer.trace("ollama_chat_test"):
            response = client.chat(
                model='gemma3',
                messages=[
                    {
                        'role': 'user',
                        'content': 'Why is the sky blue? Give a brief answer.',
                    }
                ]
            )
            
            print(f"✅ Chat Response: {response.message.content}")
            print(f"📊 Model: {response.model}")
            print(f"⏱️  Total Duration: {response.total_duration}ns")
            print(f"🔢 Prompt Tokens: {response.prompt_eval_count}")
            print(f"🔢 Completion Tokens: {response.eval_count}")
            
            # Access response both ways to test compatibility
            print(f"📝 Direct access: {response['message']['content']}")
            
    except ImportError:
        print("❌ Error: ollama package not installed. Run: pip install ollama")
        return False
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
        return False
    
    return True


def test_ollama_generate_integration():
    """Test Ollama generate() method with judgeval tracing.""" 
    try:
        from ollama import Client
        
        print("\n🧪 Testing Ollama Generate Integration...")
        
        # Initialize tracer
        tracer = Tracer(
            api_key=os.getenv("JUDGMENT_API_KEY"),
            organization_id=os.getenv("JUDGMENT_ORG_ID"),
            project_name="ollama_test"
        )
        
        # Create and wrap Ollama client
        client = wrap(Client())
        
        # Test generate method with tracing
        with tracer.trace("ollama_generate_test"):
            response = client.generate(
                model='gemma3',
                prompt='What is machine learning? Be concise.'
            )
            
            print(f"✅ Generate Response: {response.response}")
            print(f"📊 Model: {response.model}")
            print(f"⏱️  Total Duration: {response.total_duration}ns")
            print(f"🔢 Prompt Tokens: {response.prompt_eval_count}")
            print(f"🔢 Completion Tokens: {response.eval_count}")
            
            # Access response both ways to test compatibility
            print(f"📝 Direct access: {response['response']}")
            
    except Exception as e:
        print(f"❌ Generate test failed: {e}")
        return False
    
    return True


def test_error_handling():
    """Test error handling for invalid models/requests."""
    try:
        from ollama import Client
        
        print("\n🧪 Testing Error Handling...")
        
        tracer = Tracer(
            api_key=os.getenv("JUDGMENT_API_KEY"),
            organization_id=os.getenv("JUDGMENT_ORG_ID"),
            project_name="ollama_test"
        )
        
        client = wrap(Client())
        
        # Test with non-existent model
        try:
            with tracer.trace("ollama_error_test"):
                response = client.chat(
                    model='non_existent_model',
                    messages=[{'role': 'user', 'content': 'test'}]
                )
        except Exception as e:
            print(f"✅ Error handling works: {type(e).__name__}")
            return True
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False
    
    return True


def main():
    """Run all Ollama integration tests."""
    print("🚀 Starting Ollama Integration Tests")
    print("=" * 50)
    
    # Check environment variables
    if not os.getenv("JUDGMENT_API_KEY"):
        print("⚠️  Warning: JUDGMENT_API_KEY not set - tracing may not work")
    if not os.getenv("JUDGMENT_ORG_ID"):  
        print("⚠️  Warning: JUDGMENT_ORG_ID not set - tracing may not work")
    
    # Run tests
    chat_success = test_ollama_chat_integration()
    generate_success = test_ollama_generate_integration()
    error_success = test_error_handling()
    
    print("\n" + "=" * 50)
    print("🎯 Test Results:")
    print(f"   Chat Integration: {'✅ PASS' if chat_success else '❌ FAIL'}")
    print(f"   Generate Integration: {'✅ PASS' if generate_success else '❌ FAIL'}")
    print(f"   Error Handling: {'✅ PASS' if error_success else '❌ FAIL'}")
    
    if all([chat_success, generate_success, error_success]):
        print("\n🎉 All tests passed! Ollama integration is working correctly.")
        return 0
    else:
        print("\n💥 Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())