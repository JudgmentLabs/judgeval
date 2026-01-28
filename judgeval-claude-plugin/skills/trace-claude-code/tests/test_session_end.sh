#!/bin/bash
###
# Tests for session_end.sh hook
###

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/test_helpers.sh"

# Test: SessionEnd with no trace should exit gracefully
test_session_end_no_trace() {
    # Empty state
    echo '{}' > "$STATE_FILE"
    
    local input='{"session_id": "test-session"}'
    
    echo "$input" | bash "$HOOKS_DIR/session_end.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "No current trace" "Should log no current trace"
}

# Test: SessionEnd with valid trace but no transcript
test_session_end_no_transcript() {
    create_mock_state "trace-abc123def456abc123def456" "session-456" "rootspan1234567" "project-789" "taskspan12345678"
    
    local input='{"session_id": "session-456"}'
    
    echo "$input" | bash "$HOOKS_DIR/session_end.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    # Should still try to finalize session
    assert_contains "$logs" "SessionEnd hook triggered" "Should log hook trigger"
}

# Test: SessionEnd parses transcript correctly
test_session_end_with_transcript() {
    create_mock_state "trace-abc123def456abc123def456" "session-test123" "rootspan1234567" "project-789" "taskspan12345678"
    
    # Create transcript
    local transcript_file="$TEST_TMP_DIR/.claude/projects/test-project/session-test123.jsonl"
    
    create_mock_transcript "$transcript_file" \
        "$(mock_user_message 'Hello, help me with code' '2024-01-01T10:00:00.000Z')" \
        "$(mock_assistant_message 'Of course! What do you need help with?' 'claude-opus-4-5-20250514' 100 50 '2024-01-01T10:00:01.000Z')" \
        "$(mock_user_message 'Read my file' '2024-01-01T10:00:05.000Z')" \
        "$(mock_assistant_tool_use 'tool-1' 'Read' '{"path": "test.py"}' 'Let me read that.' 'claude-opus-4-5-20250514' '2024-01-01T10:00:06.000Z')" \
        "$(mock_tool_result_message 'tool-1' 'print(\"hello\")' '2024-01-01T10:00:07.000Z')" \
        "$(mock_assistant_message 'The file contains a simple print statement.' 'claude-opus-4-5-20250514' 150 75 '2024-01-01T10:00:08.000Z')"
    
    local input
    input=$(jq -n -c \
        --arg session_id "session-test123" \
        --arg transcript_path "$transcript_file" \
        '{session_id: $session_id, transcript_path: $transcript_path}')
    
    echo "$input" | bash "$HOOKS_DIR/session_end.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "SessionEnd hook triggered" "Should log hook trigger"
    assert_contains "$logs" "Processing transcript" "Should process transcript"
}

# Test: SessionEnd handles tracing disabled
test_session_end_tracing_disabled() {
    export TRACE_TO_JUDGEVAL="false"
    
    local input='{"session_id": "test"}'
    
    echo "$input" | bash "$HOOKS_DIR/session_end.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "Tracing disabled" "Should log tracing disabled"
}

# Test: SessionEnd clears current_trace_id
test_session_end_clears_trace() {
    create_mock_state "trace-abc123def456abc123def456" "session-456" "rootspan1234567" "project-789" "taskspan12345678"
    
    local input='{"session_id": "session-456"}'
    
    echo "$input" | bash "$HOOKS_DIR/session_end.sh" 2>&1 || true
    
    # Check that current_trace_id is cleared
    local current_trace
    current_trace=$(jq -r '.current_trace_id // empty' "$STATE_FILE" 2>/dev/null)
    
    assert_equals "" "$current_trace" "current_trace_id should be cleared after session end"
}

# Test: SessionEnd with multiple LLM calls
test_session_end_multiple_llm_calls() {
    create_mock_state "trace-abc123def456abc123def456" "session-multi" "rootspan1234567" "project-789" "taskspan12345678"
    
    local transcript_file="$TEST_TMP_DIR/.claude/projects/test-project/session-multi.jsonl"
    
    # Create transcript with multiple back-and-forth
    create_mock_transcript "$transcript_file" \
        "$(mock_user_message 'First question' '2024-01-01T10:00:00.000Z')" \
        "$(mock_assistant_message 'First answer' 'claude-opus-4-5-20250514' 100 50 '2024-01-01T10:00:01.000Z')" \
        "$(mock_user_message 'Second question' '2024-01-01T10:00:10.000Z')" \
        "$(mock_assistant_message 'Second answer' 'claude-opus-4-5-20250514' 120 60 '2024-01-01T10:00:11.000Z')" \
        "$(mock_user_message 'Third question' '2024-01-01T10:00:20.000Z')" \
        "$(mock_assistant_message 'Third answer' 'claude-opus-4-5-20250514' 130 70 '2024-01-01T10:00:21.000Z')"
    
    local input
    input=$(jq -n -c \
        --arg session_id "session-multi" \
        --arg transcript_path "$transcript_file" \
        '{session_id: $session_id, transcript_path: $transcript_path}')
    
    echo "$input" | bash "$HOOKS_DIR/session_end.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "SessionEnd hook triggered" "Should log hook trigger"
}

# Test: SessionEnd handles empty transcript
test_session_end_empty_transcript() {
    create_mock_state "trace-abc123def456abc123def456" "session-empty" "rootspan1234567" "project-789" "taskspan12345678"
    
    local transcript_file="$TEST_TMP_DIR/.claude/projects/test-project/session-empty.jsonl"
    mkdir -p "$(dirname "$transcript_file")"
    touch "$transcript_file"  # Empty file
    
    local input
    input=$(jq -n -c \
        --arg session_id "session-empty" \
        --arg transcript_path "$transcript_file" \
        '{session_id: $session_id, transcript_path: $transcript_path}')
    
    echo "$input" | bash "$HOOKS_DIR/session_end.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "SessionEnd hook triggered" "Should handle empty transcript"
}

# Run all tests
echo "================================"
echo "Testing session_end.sh"
echo "================================"
echo ""

run_test "SessionEnd with no trace" test_session_end_no_trace
run_test "SessionEnd with no transcript" test_session_end_no_transcript
run_test "SessionEnd with valid transcript" test_session_end_with_transcript
run_test "SessionEnd respects tracing disabled" test_session_end_tracing_disabled
run_test "SessionEnd clears trace ID" test_session_end_clears_trace
run_test "SessionEnd with multiple LLM calls" test_session_end_multiple_llm_calls
run_test "SessionEnd handles empty transcript" test_session_end_empty_transcript

print_summary
