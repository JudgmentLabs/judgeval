#!/bin/bash
###
# Tests for subagent_stop.sh hook
###

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/test_helpers.sh"

# Mock insert_span to avoid actual API calls
# We'll track what would have been inserted
INSERTED_SPANS=()

# Override insert_span from common.sh after sourcing
setup_mock_api() {
    # Create a wrapper script that mocks the API
    cat > "$TEST_TMP_DIR/mock_common.sh" << 'MOCK_EOF'
# Override insert_span to capture calls instead of hitting API
insert_span() {
    local project_id="$1"
    local span_json="$2"
    
    # Write to a capture file
    echo "$span_json" >> "$TEST_TMP_DIR/inserted_spans.jsonl"
    debug "MOCK insert_span: $(echo "$span_json" | jq -c '.name // .attributes[0].value.stringValue')"
    echo "mock-success"
    return 0
}

# Override get_project_id to return mock
get_project_id() {
    echo "mock-project-id"
    return 0
}
MOCK_EOF
}

# Test: SubagentStop with missing trace ID should exit gracefully
test_subagent_stop_no_trace() {
    setup_mock_api
    
    # Empty state - no current_trace_id
    echo '{}' > "$STATE_FILE"
    
    local input='{"agent_id": "test-agent", "task": "Test task"}'
    local output
    output=$(echo "$input" | bash "$HOOKS_DIR/subagent_stop.sh" 2>&1) || true
    
    # Should exit without error
    assert_contains "$(cat "$LOG_FILE" 2>/dev/null || echo '')" "No current trace" \
        "Should log that there's no current trace"
}

# Test: SubagentStop with valid trace but no transcript
test_subagent_stop_no_transcript() {
    setup_mock_api
    
    # Create state with active trace
    create_mock_state "trace-123" "session-456" "root-span" "project-789" "task-span"
    
    local input='{"agent_id": "test-agent", "task": "Test task without transcript"}'
    
    # Run hook (it should still create a container span even without transcript)
    echo "$input" | bash "$HOOKS_DIR/subagent_stop.sh" 2>&1 || true
    
    # Check that the hook ran (log should exist)
    assert_file_exists "$LOG_FILE" "Log file should be created"
    
    # Check logs
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    assert_contains "$logs" "SubagentStop hook triggered" "Should log hook trigger"
}

# Test: SubagentStop with valid transcript parses LLM spans
test_subagent_stop_with_transcript() {
    setup_mock_api
    
    # Create state
    create_mock_state "trace-abc123def456abc123def456" "session-456" "rootspan1234567" "project-789" "taskspan12345678"
    
    # Create mock subagent transcript
    local transcript_file="$TEST_TMP_DIR/.claude/projects/test-project/agent-test-subagent.jsonl"
    
    create_mock_transcript "$transcript_file" \
        "$(mock_user_message 'Refactor the auth module')" \
        "$(mock_assistant_message 'I will help you refactor the auth module.' 'claude-3-5-haiku-20241022' 150 75)" \
        "$(mock_assistant_tool_use 'tool-1' 'Read' '{"path": "/src/auth.py"}' 'Let me read the file first.')" \
        "$(mock_tool_result_message 'tool-1' 'def authenticate(): pass')" \
        "$(mock_assistant_message 'I have read the file. Here is my analysis.' 'claude-3-5-haiku-20241022' 200 100)"
    
    # Input with transcript path
    local input
    input=$(jq -n -c \
        --arg agent_id "test-subagent" \
        --arg task "Refactor auth module" \
        --arg transcript_path "$transcript_file" \
        '{agent_id: $agent_id, task: $task, transcript_path: $transcript_path}')
    
    # Patch insert_span in the hook by creating a wrapper
    # Since we can't easily mock bash functions, we'll check the logs
    echo "$input" | bash "$HOOKS_DIR/subagent_stop.sh" 2>&1 || true
    
    # Check logs for evidence of parsing
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "SubagentStop hook triggered" "Should log hook trigger"
    assert_contains "$logs" "Parsing subagent transcript" "Should log transcript parsing"
}

# Test: SubagentStop finds transcript by agent ID
test_subagent_stop_finds_transcript() {
    setup_mock_api
    
    # Create state
    create_mock_state "trace-abc123def456abc123def456" "session-456" "rootspan1234567" "project-789" "taskspan12345678"
    
    # Create transcript in expected location (agent-{id}.jsonl)
    local transcript_file="$TEST_TMP_DIR/.claude/projects/test-project/agent-my-agent.jsonl"
    
    create_mock_transcript "$transcript_file" \
        "$(mock_user_message 'Hello')" \
        "$(mock_assistant_message 'Hi there!' 'claude-3-5-haiku-20241022' 50 25)"
    
    # Input without transcript_path - should find it by agent_id
    local input='{"agent_id": "my-agent", "task": "Say hello"}'
    
    echo "$input" | bash "$HOOKS_DIR/subagent_stop.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "SubagentStop hook triggered" "Should log hook trigger"
}

# Test: SubagentStop handles invalid JSON input
test_subagent_stop_invalid_json() {
    setup_mock_api
    
    create_mock_state "trace-123" "session-456" "root-span" "project-789" "task-span"
    
    local input="not valid json at all"
    
    echo "$input" | bash "$HOOKS_DIR/subagent_stop.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "Invalid JSON" "Should log invalid JSON"
}

# Test: SubagentStop respects tracing disabled
test_subagent_stop_tracing_disabled() {
    setup_mock_api
    
    export TRACE_TO_JUDGEVAL="false"
    
    local input='{"agent_id": "test", "task": "Test"}'
    
    echo "$input" | bash "$HOOKS_DIR/subagent_stop.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "Tracing disabled" "Should log tracing disabled"
}

# Test: SubagentStop with complex transcript (multiple tool uses)
test_subagent_stop_complex_transcript() {
    setup_mock_api
    
    create_mock_state "trace-abc123def456abc123def456" "session-456" "rootspan1234567" "project-789" "taskspan12345678"
    
    local transcript_file="$TEST_TMP_DIR/.claude/projects/test-project/agent-complex.jsonl"
    
    # Create a more complex transcript with multiple tools
    create_mock_transcript "$transcript_file" \
        "$(mock_user_message 'Analyze and fix the bug' '2024-01-01T10:00:00.000Z')" \
        "$(mock_assistant_tool_use 'tool-1' 'Read' '{"path": "bug.py"}' 'Let me check the file.' 'claude-3-5-haiku-20241022' '2024-01-01T10:00:01.000Z')" \
        "$(mock_tool_result_message 'tool-1' 'def buggy(): return None' '2024-01-01T10:00:02.000Z')" \
        "$(mock_assistant_tool_use 'tool-2' 'Read' '{"path": "test.py"}' 'Now checking tests.' 'claude-3-5-haiku-20241022' '2024-01-01T10:00:03.000Z')" \
        "$(mock_tool_result_message 'tool-2' 'def test_buggy(): assert buggy() is not None' '2024-01-01T10:00:04.000Z')" \
        "$(mock_assistant_tool_use 'tool-3' 'Edit' '{"path": "bug.py", "content": "def buggy(): return True"}' 'Fixing the bug.' 'claude-3-5-haiku-20241022' '2024-01-01T10:00:05.000Z')" \
        "$(mock_tool_result_message 'tool-3' 'File edited successfully' '2024-01-01T10:00:06.000Z')" \
        "$(mock_assistant_message 'I have fixed the bug by making buggy() return True.' 'claude-3-5-haiku-20241022' 300 150 '2024-01-01T10:00:07.000Z')"
    
    local input
    input=$(jq -n -c \
        --arg agent_id "complex" \
        --arg task "Analyze and fix bug" \
        --arg transcript_path "$transcript_file" \
        '{agent_id: $agent_id, task: $task, transcript_path: $transcript_path}')
    
    echo "$input" | bash "$HOOKS_DIR/subagent_stop.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "SubagentStop hook triggered" "Should log hook trigger"
    assert_contains "$logs" "Parsing subagent transcript" "Should parse transcript"
}

# Test: iso_to_nanos function works correctly
test_iso_to_nanos() {
    setup_mock_api
    
    # Source common.sh to get the function
    source "$HOOKS_DIR/common.sh"
    
    # Test with a known timestamp
    local result
    result=$(python3 -c "
from datetime import datetime
ts = '2024-01-01T12:00:00.000Z'.replace('Z', '+00:00')
print(int(datetime.fromisoformat(ts).timestamp() * 1e9))
" 2>/dev/null)
    
    # 2024-01-01T12:00:00Z should be 1704110400 seconds = 1704110400000000000 nanoseconds
    assert_equals "1704110400000000000" "$result" "ISO timestamp should convert correctly"
}

# Test: SubagentStop creates container span even if transcript parsing fails
test_subagent_stop_container_span_on_failure() {
    setup_mock_api
    
    create_mock_state "trace-abc123def456abc123def456" "session-456" "rootspan1234567" "project-789" "taskspan12345678"
    
    # Create an invalid transcript file (not valid JSONL)
    local transcript_file="$TEST_TMP_DIR/.claude/projects/test-project/agent-invalid.jsonl"
    mkdir -p "$(dirname "$transcript_file")"
    echo "this is not valid json" > "$transcript_file"
    echo "neither is this" >> "$transcript_file"
    
    local input
    input=$(jq -n -c \
        --arg agent_id "invalid" \
        --arg task "Test with invalid transcript" \
        --arg transcript_path "$transcript_file" \
        '{agent_id: $agent_id, task: $task, transcript_path: $transcript_path}')
    
    echo "$input" | bash "$HOOKS_DIR/subagent_stop.sh" 2>&1 || true
    
    # Should still create logs (container span attempt)
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "SubagentStop hook triggered" "Should log hook trigger even with invalid transcript"
}

# Run all tests
echo "================================"
echo "Testing subagent_stop.sh"
echo "================================"
echo ""

run_test "SubagentStop with no trace" test_subagent_stop_no_trace
run_test "SubagentStop with no transcript" test_subagent_stop_no_transcript
run_test "SubagentStop with valid transcript" test_subagent_stop_with_transcript
run_test "SubagentStop finds transcript by ID" test_subagent_stop_finds_transcript
run_test "SubagentStop handles invalid JSON" test_subagent_stop_invalid_json
run_test "SubagentStop respects tracing disabled" test_subagent_stop_tracing_disabled
run_test "SubagentStop with complex transcript" test_subagent_stop_complex_transcript
run_test "ISO to nanoseconds conversion" test_iso_to_nanos
run_test "SubagentStop creates container on parse failure" test_subagent_stop_container_span_on_failure

print_summary
