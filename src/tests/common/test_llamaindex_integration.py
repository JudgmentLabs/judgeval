"""
Comprehensive unit tests for LlamaIndex integration with judgeval.wrap()

This test suite addresses the gaps in PR #461 by including:
1. ReActAgent compatibility tests (the actual use case from the issue)
2. All method coverage (sync, async, streaming)
3. Error handling scenarios
4. Multi-provider support (OpenAI and Anthropic)
"""

import pytest
from unittest.mock import Mock

# Try to import LlamaIndex - skip tests if not available
try:
    from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
    from llama_index.llms.anthropic import Anthropic as LlamaIndexAnthropic
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    LlamaIndexOpenAI = None
    LlamaIndexAnthropic = None

from judgeval.common.tracer.core import wrap, _get_client_config, _format_output_data


class TestLlamaIndexIntegration:
    """Test LlamaIndex integration with judgeval"""

    @pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="LlamaIndex not available")
    def test_llamaindex_openai_client_can_be_wrapped(self):
        """Test that LlamaIndex OpenAI client can be wrapped without errors"""
        # Create LlamaIndex OpenAI client
        llm = LlamaIndexOpenAI(model="gpt-4", temperature=0.0)
        
        # Wrap the client - this should not raise an error
        wrapped_llm = wrap(llm)
        
        # Assert wrapper was created and preserves type
        assert wrapped_llm is not None
        assert hasattr(wrapped_llm, 'complete')
        assert hasattr(wrapped_llm, 'acomplete')
        assert hasattr(wrapped_llm, 'chat')
        assert hasattr(wrapped_llm, 'achat')
        assert hasattr(wrapped_llm, 'stream_complete')
        assert hasattr(wrapped_llm, 'stream_chat')
        
        # Critical: Test that the wrapped object maintains the original type
        # This is why we use monkey patching instead of a wrapper class
        assert isinstance(wrapped_llm, LlamaIndexOpenAI)
        assert type(wrapped_llm).__name__ == 'OpenAI'

    @pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="LlamaIndex not available")
    def test_llamaindex_anthropic_client_can_be_wrapped(self):
        """Test that LlamaIndex Anthropic client can be wrapped"""
        # Create LlamaIndex Anthropic client
        llm = LlamaIndexAnthropic(model="claude-3-haiku", temperature=0.0)
        
        # Wrap the client
        wrapped_llm = wrap(llm)
        
        # Assert wrapper was created
        assert wrapped_llm is not None
        assert hasattr(wrapped_llm, 'complete')
        assert hasattr(wrapped_llm, 'acomplete')
        
        # Verify type preservation
        assert isinstance(wrapped_llm, LlamaIndexAnthropic)

    @pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="LlamaIndex not available")
    def test_react_agent_compatibility(self):
        """
        Critical test: Verify that ReActAgent accepts our wrapped client.
        This is the core issue from GitHub #455.
        """
        try:
            from llama_index.core.agent import ReActAgent
            from llama_index.core.tools import FunctionTool
        except ImportError:
            pytest.skip("ReActAgent not available")
        
        # Create and wrap LLM
        llm = LlamaIndexOpenAI(model="gpt-4", temperature=0.0)
        wrapped_llm = wrap(llm)
        
        # Create a simple tool
        def dummy_tool(x: int) -> int:
            return x * 2
        
        tool = FunctionTool.from_defaults(fn=dummy_tool)
        
        # This is the critical test - ReActAgent should accept our wrapped LLM
        # because we preserved the original type through monkey patching
        try:
            agent = ReActAgent.from_tools(
                tools=[tool],
                llm=wrapped_llm
            )
            # If we get here, the agent accepted our wrapped LLM
            assert agent is not None
            # Note: ReActAgent doesn't expose the LLM as an attribute
            # The important test is that the agent was created successfully
        except Exception as e:
            pytest.fail(f"ReActAgent rejected wrapped LLM: {e}")

    @pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="LlamaIndex not available")
    def test_wrapped_methods_are_callable(self):
        """Test that all wrapped methods remain callable"""
        llm = LlamaIndexOpenAI(model="gpt-4", temperature=0.0)
        wrapped_llm = wrap(llm)
        
        # All these methods should be callable (not just exist)
        assert callable(wrapped_llm.complete)
        assert callable(wrapped_llm.acomplete)
        assert callable(wrapped_llm.chat)
        assert callable(wrapped_llm.achat)
        assert callable(wrapped_llm.stream_complete)
        assert callable(wrapped_llm.stream_chat)
        assert callable(wrapped_llm.astream_complete)
        assert callable(wrapped_llm.astream_chat)

    @pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="LlamaIndex not available")
    def test_original_attributes_preserved(self):
        """Test that wrapper preserves all original client attributes"""
        # Create client with specific attributes
        llm = LlamaIndexOpenAI(
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            api_key="test-key"
        )
        
        # Wrap the client
        wrapped_llm = wrap(llm)
        
        # All original attributes should be accessible
        assert wrapped_llm.model == "gpt-4"
        assert wrapped_llm.temperature == 0.7
        assert wrapped_llm.max_tokens == 100
        assert wrapped_llm.api_key == "test-key"

    @pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="LlamaIndex not available")
    def test_get_client_config_llamaindex(self):
        """Test _get_client_config recognizes LlamaIndex clients"""
        # Test OpenAI
        llm_openai = LlamaIndexOpenAI(model="gpt-4")
        config = _get_client_config(llm_openai)
        
        assert config[0] == "LLAMAINDEX_OPENAI_API_CALL"
        assert config[1] == llm_openai.complete
        assert config[2] == llm_openai.chat
        assert config[3] == llm_openai.stream_complete
        assert config[4] == llm_openai.stream_chat
        
        # Test Anthropic
        llm_anthropic = LlamaIndexAnthropic(model="claude-3")
        config = _get_client_config(llm_anthropic)
        
        assert config[0] == "LLAMAINDEX_ANTHROPIC_API_CALL"
        assert config[1] == llm_anthropic.complete
        assert config[2] == llm_anthropic.chat

    @pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="LlamaIndex not available")
    def test_format_output_data_llamaindex(self):
        """Test _format_output_data handles LlamaIndex responses correctly"""
        llm = LlamaIndexOpenAI(model="gpt-4")
        
        # Mock a LlamaIndex response
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_response.raw = Mock()
        mock_response.raw.usage = Mock()
        mock_response.raw.usage.prompt_tokens = 10
        mock_response.raw.usage.completion_tokens = 5
        # Mock model attribute as a string, not a Mock object
        mock_response.raw.model = "gpt-4"
        
        # For prompt_tokens_details (used for cache_read_input_tokens)
        mock_response.raw.usage.prompt_tokens_details = Mock()
        mock_response.raw.usage.prompt_tokens_details.cached_tokens = 0
        
        message_content, usage = _format_output_data(llm, mock_response)
        
        assert message_content == "Test response"
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15

    @pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="LlamaIndex not available")
    def test_module_path_detection(self):
        """Test that we correctly detect LlamaIndex clients by module path"""
        # Create a mock object that looks like LlamaIndex but isn't the imported class
        mock_llm = Mock()
        mock_llm.__class__.__module__ = 'llama_index.llms.openai.base'
        mock_llm.__class__.__name__ = 'OpenAI'
        mock_llm.complete = Mock()
        mock_llm.chat = Mock()
        mock_llm.stream_complete = Mock()
        mock_llm.stream_chat = Mock()
        
        # This should still be wrapped correctly based on module path
        wrapped = wrap(mock_llm)
        assert wrapped is not None

    def test_unsupported_client_type_error(self):
        """Test that unsupported client types raise appropriate error"""
        unsupported_client = object()
        
        with pytest.raises(ValueError, match="Unsupported client type"):
            wrap(unsupported_client)

    @pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="LlamaIndex not available")
    def test_pydantic_field_validation_bypass(self):
        """
        Test that our monkey patching approach bypasses Pydantic field validation.
        This is why we use object.__setattr__ instead of direct assignment.
        """
        llm = LlamaIndexOpenAI(model="gpt-4")
        
        # Direct assignment would fail with Pydantic
        with pytest.raises(Exception):
            llm.some_new_method = lambda: "test"
        
        # But object.__setattr__ bypasses validation
        object.__setattr__(llm, 'some_new_method', lambda: "test")
        assert llm.some_new_method() == "test"

    @pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="LlamaIndex not available")
    def test_streaming_method_preservation(self):
        """Test that streaming methods are properly wrapped"""
        llm = LlamaIndexOpenAI(model="gpt-4")
        original_stream_complete = llm.stream_complete
        original_stream_chat = llm.stream_chat
        
        wrapped_llm = wrap(llm)
        
        # Methods should be wrapped, not the originals
        assert wrapped_llm.stream_complete != original_stream_complete
        assert wrapped_llm.stream_chat != original_stream_chat
        assert callable(wrapped_llm.stream_complete)
        assert callable(wrapped_llm.stream_chat)

    @pytest.mark.skipif(LLAMAINDEX_AVAILABLE, reason="Testing without LlamaIndex")
    def test_llamaindex_not_available_handling(self):
        """Test graceful handling when LlamaIndex is not installed"""
        # When LlamaIndex is not available, any client should fall through
        # to standard client handling or raise unsupported error
        mock_client = Mock()
        mock_client.__class__.__module__ = 'llama_index.llms.openai'
        mock_client.__class__.__name__ = 'OpenAI'
        
        with pytest.raises(ValueError, match="Unsupported client type"):
            wrap(mock_client)

    @pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="LlamaIndex not available")
    def test_async_method_preservation(self):
        """Test that async methods are properly wrapped and remain async"""
        llm = LlamaIndexOpenAI(model="gpt-4")
        wrapped_llm = wrap(llm)
        
        # Async methods should still be async after wrapping
        import asyncio
        assert asyncio.iscoroutinefunction(wrapped_llm.acomplete)
        assert asyncio.iscoroutinefunction(wrapped_llm.achat)
        assert asyncio.iscoroutinefunction(wrapped_llm.astream_complete)
        assert asyncio.iscoroutinefunction(wrapped_llm.astream_chat)

    @pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="LlamaIndex not available")
    def test_wrapped_client_with_mocked_response(self):
        """Test wrapped client with mocked API responses"""
        llm = LlamaIndexOpenAI(model="gpt-4")
        
        # Mock the complete method using object.__setattr__ to bypass Pydantic validation
        mock_response = Mock()
        mock_response.text = "Mocked response"
        mock_complete = Mock(return_value=mock_response)
        object.__setattr__(llm, 'complete', mock_complete)
        
        wrapped_llm = wrap(llm)
        
        # The wrapped method should still work
        # Note: In real implementation, the wrapper adds tracing
        # but the mock simplifies testing
        result = wrapped_llm.complete("test prompt")
        assert result.text == "Mocked response"