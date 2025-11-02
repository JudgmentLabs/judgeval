def openai_tokens_converter(
    prompt_tokens: int,
    completion_tokens: int,
    cache_read: int,
    cache_creation: int,
    total_tokens: int,
) -> tuple[int, int, int, int]:
    """
    Returns:
        tuple[int, int, int, int]:
            - judgment.usage.non_cached_input
            - judgment.usage.output_tokens
            - judgment.usage.cached_input_tokens
            - judgment.usage.cache_creation_tokens
    """
    manual_tokens = prompt_tokens + completion_tokens + cache_read + cache_creation

    if manual_tokens > total_tokens:
        # This is the openAI case where we need to subtract the cached tokens from the input tokens
        return prompt_tokens - cache_read, completion_tokens, cache_read, cache_creation
    else:
        return prompt_tokens, completion_tokens, cache_read, cache_creation
