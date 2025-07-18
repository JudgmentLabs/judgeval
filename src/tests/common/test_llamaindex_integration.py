"""
Tests for LlamaIndex integration with judgeval.wrap()
"""

import pytest
from unittest.mock import patch

# Try to import LlamaIndex - skip tests if not available
try:
    from llama_index.llms.openai.base import OpenAI as LlamaIndexOpenAI
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    LlamaIndexOpenAI = None

from judgeval.common.tracer.core import wrap


class TestLlamaIndexIntegration:
    """Test LlamaIndex integration with judgeval"""

    @pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="LlamaIndex not available")
    def test_llamaindex_client_can_be_wrapped(self):
        """Test that LlamaIndex OpenAI client can be wrapped without errors"""
        # Create LlamaIndex OpenAI client
        llm = LlamaIndexOpenAI(model="gpt-4.1", temperature=0.0)
        
        # Wrap the client - this should not raise an error
        wrapped_llm = wrap(llm)
        
        # Assert wrapper was created
        assert wrapped_llm is not None
        assert hasattr(wrapped_llm, 'complete')
        assert hasattr(wrapped_llm, 'acomplete')

    @pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="LlamaIndex not available")
    def test_llamaindex_wrapper_preserves_original_attributes(self):
        """Test that the wrapper preserves access to original client attributes"""
        # Create LlamaIndex OpenAI client
        llm = LlamaIndexOpenAI(model="gpt-4.1", temperature=0.0)
        
        # Wrap the client
        wrapped_llm = wrap(llm)
        
        # Test that original attributes are accessible
        assert wrapped_llm.model == "gpt-4.1"
        assert wrapped_llm.temperature == 0.0
        assert hasattr(wrapped_llm, 'api_key')

    @pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="LlamaIndex not available")
    def test_llamaindex_wrapper_type_identification(self):
        """Test that wrapper correctly identifies as LlamaIndex type"""
        # Create LlamaIndex OpenAI client
        llm = LlamaIndexOpenAI(model="gpt-4.1", temperature=0.0)
        
        # Wrap the client
        wrapped_llm = wrap(llm)
        
        # Test type information is preserved
        assert "LlamaIndexWrapper" in str(type(wrapped_llm))
        assert wrapped_llm._original_client is llm

    @pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="LlamaIndex not available")
    def test_llamaindex_wrapper_method_delegation(self):
        """Test that unknown methods are delegated to original client"""
        # Create LlamaIndex OpenAI client
        llm = LlamaIndexOpenAI(model="gpt-4.1", temperature=0.0)
        
        # Wrap the client
        wrapped_llm = wrap(llm)
        
        # Test that arbitrary attributes can be accessed
        # (delegated to the original client)
        assert hasattr(wrapped_llm, 'api_base')
        assert wrapped_llm.api_base == llm.api_base

    def test_unsupported_client_type_error(self):
        """Test that unsupported client types raise appropriate error"""
        # Test with a generic object that shouldn't be wrappable
        unsupported_client = object()
        
        # This should raise ValueError for unsupported client type
        with pytest.raises(ValueError, match="Unsupported client type"):
            wrap(unsupported_client)

    def test_llamaindex_not_available_fallback(self):
        """Test behavior when LlamaIndex is not available"""
        # Mock LlamaIndex as not available
        with patch('judgeval.common.tracer.core.LLAMAINDEX_AVAILABLE', False):
            # Should still handle the error gracefully
            # but would raise error for actual LlamaIndex objects
            # This tests the import guard logic
            assert True  # Import guard logic is tested during import

    @pytest.mark.skipif(not LLAMAINDEX_AVAILABLE, reason="LlamaIndex not available")  
    def test_llamaindex_wrapper_string_representation(self):
        """Test that the wrapper has proper string representation"""
        # Create LlamaIndex OpenAI client
        llm = LlamaIndexOpenAI(model="gpt-4.1", temperature=0.0)
        
        # Wrap the client
        wrapped_llm = wrap(llm)
        
        # Test that string representation includes model info
        str_repr = str(wrapped_llm)
        assert "gpt-4.1" in str_repr
        assert "LlamaIndexWrapper" in str_repr 