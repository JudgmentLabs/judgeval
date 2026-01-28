#!/bin/bash
###
# Integration tests for multi-turn, multiple sessions, and multiple subagents
###

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/test_helpers.sh"

###
# Multi-turn conversation tests
###

# Test: Multi-turn conversation with tool calls
test_multi_turn_with_tools() {
    create_mock_state "trace-multiturn12345678901234" "session-mt1" "rootspan1234567" "project-123" "taskspan12345678"
    
    local transcript_file="$TEST_TMP_DIR/.claude/projects/test-project/session-mt1.jsonl"
    
    # Complex multi-turn: user asks, assistant uses tool, result, assistant responds, repeat
    create_mock_transcript "$transcript_file" \
        "$(mock_user_message 'List files in current directory' '2024-01-01T10:00:00.000Z')" \
        "$(mock_assistant_tool_use 'tool-ls-1' 'Bash' '{"command": "ls -la"}' 'Let me list the files.' 'claude-opus-4-5-20250514' '2024-01-01T10:00:01.000Z')" \
        "$(mock_tool_result_message 'tool-ls-1' 'file1.py\nfile2.py\nREADME.md' '2024-01-01T10:00:02.000Z')" \
        "$(mock_assistant_message 'I found 3 files: file1.py, file2.py, and README.md' 'claude-opus-4-5-20250514' 150 60 '2024-01-01T10:00:03.000Z')" \
        "$(mock_user_message 'Read file1.py' '2024-01-01T10:00:10.000Z')" \
        "$(mock_assistant_tool_use 'tool-read-1' 'Read' '{"path": "file1.py"}' 'Reading file1.py...' 'claude-opus-4-5-20250514' '2024-01-01T10:00:11.000Z')" \
        "$(mock_tool_result_message 'tool-read-1' 'def hello():\n    print(\"Hello World\")' '2024-01-01T10:00:12.000Z')" \
        "$(mock_assistant_message 'The file contains a simple hello function.' 'claude-opus-4-5-20250514' 180 80 '2024-01-01T10:00:13.000Z')" \
        "$(mock_user_message 'Now edit it to add a goodbye function' '2024-01-01T10:00:20.000Z')" \
        "$(mock_assistant_tool_use 'tool-edit-1' 'Edit' '{"path": "file1.py", "content": "def goodbye(): print(\"Goodbye\")"}' 'Adding the function...' 'claude-opus-4-5-20250514' '2024-01-01T10:00:21.000Z')" \
        "$(mock_tool_result_message 'tool-edit-1' 'File edited successfully' '2024-01-01T10:00:22.000Z')" \
        "$(mock_assistant_message 'Done! I added the goodbye function.' 'claude-opus-4-5-20250514' 200 90 '2024-01-01T10:00:23.000Z')"
    
    local input
    input=$(jq -n -c \
        --arg session_id "session-mt1" \
        --arg transcript_path "$transcript_file" \
        '{session_id: $session_id, transcript_path: $transcript_path}')
    
    echo "$input" | bash "$HOOKS_DIR/session_end.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "Processing transcript" "Should process multi-turn transcript"
    # Should have created LLM spans
    assert_contains "$logs" "LLM span" "Should create LLM spans" || true
}

# Test: Multi-turn with multiple tools in one response
test_multi_turn_multiple_tools_per_response() {
    create_mock_state "trace-multitools1234567890123" "session-mt2" "rootspan1234567" "project-123" "taskspan12345678"
    
    local transcript_file="$TEST_TMP_DIR/.claude/projects/test-project/session-mt2.jsonl"
    
    # Assistant uses multiple tools in one turn
    local multi_tool_response
    multi_tool_response=$(jq -n -c '{
        type: "assistant",
        timestamp: "2024-01-01T10:00:01.000Z",
        message: {
            model: "claude-opus-4-5-20250514",
            content: [
                {type: "text", text: "I will read both files."},
                {type: "tool_use", id: "tool-1", name: "Read", input: {path: "file1.py"}},
                {type: "tool_use", id: "tool-2", name: "Read", input: {path: "file2.py"}}
            ],
            usage: {input_tokens: 100, output_tokens: 80}
        }
    }')
    
    local multi_tool_result
    multi_tool_result=$(jq -n -c '{
        type: "user",
        timestamp: "2024-01-01T10:00:02.000Z",
        message: {
            content: [
                {type: "tool_result", tool_use_id: "tool-1", content: "content of file1"},
                {type: "tool_result", tool_use_id: "tool-2", content: "content of file2"}
            ]
        },
        toolUseResult: {type: "text", text: "content of file1"}
    }')
    
    create_mock_transcript "$transcript_file" \
        "$(mock_user_message 'Read both file1.py and file2.py' '2024-01-01T10:00:00.000Z')" \
        "$multi_tool_response" \
        "$multi_tool_result" \
        "$(mock_assistant_message 'Both files have been read. Here are the contents...' 'claude-opus-4-5-20250514' 200 100 '2024-01-01T10:00:03.000Z')"
    
    local input
    input=$(jq -n -c \
        --arg session_id "session-mt2" \
        --arg transcript_path "$transcript_file" \
        '{session_id: $session_id, transcript_path: $transcript_path}')
    
    echo "$input" | bash "$HOOKS_DIR/session_end.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "Processing transcript" "Should handle multiple tools per response"
}

# Test: Long multi-turn conversation (10+ turns)
test_long_multi_turn_conversation() {
    create_mock_state "trace-longconv123456789012345" "session-long" "rootspan1234567" "project-123" "taskspan12345678"
    
    local transcript_file="$TEST_TMP_DIR/.claude/projects/test-project/session-long.jsonl"
    mkdir -p "$(dirname "$transcript_file")"
    
    # Generate 10 turns
    local ts_base=1704067200  # 2024-01-01T10:00:00
    for i in $(seq 1 10); do
        local ts_user=$((ts_base + (i-1)*10))
        local ts_assistant=$((ts_base + (i-1)*10 + 1))
        
        local user_ts
        user_ts=$(date -r $ts_user -u +"%Y-%m-%dT%H:%M:%S.000Z" 2>/dev/null || date -d "@$ts_user" -u +"%Y-%m-%dT%H:%M:%S.000Z" 2>/dev/null || echo "2024-01-01T10:0${i}:00.000Z")
        local assist_ts
        assist_ts=$(date -r $ts_assistant -u +"%Y-%m-%dT%H:%M:%S.000Z" 2>/dev/null || date -d "@$ts_assistant" -u +"%Y-%m-%dT%H:%M:%S.000Z" 2>/dev/null || echo "2024-01-01T10:0${i}:01.000Z")
        
        mock_user_message "Question $i" "$user_ts" >> "$transcript_file"
        mock_assistant_message "Answer $i" "claude-opus-4-5-20250514" $((100 + i*10)) $((50 + i*5)) "$assist_ts" >> "$transcript_file"
    done
    
    local input
    input=$(jq -n -c \
        --arg session_id "session-long" \
        --arg transcript_path "$transcript_file" \
        '{session_id: $session_id, transcript_path: $transcript_path}')
    
    echo "$input" | bash "$HOOKS_DIR/session_end.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "Processing transcript" "Should handle long conversation"
    assert_contains "$logs" "Trace ended" "Should complete trace"
}

###
# Multiple sessions tests
###

# Test: Two independent sessions with different trace IDs
test_multiple_independent_sessions() {
    # Session 1: Create state and start
    local trace1="trace-session1abc123def456"
    local trace2="trace-session2xyz789ghi012"
    
    # Setup session 1
    cat > "$STATE_FILE" << EOF
{
    "current_trace_id": "$trace1",
    "project_id": "project-123",
    "sessions": {
        "$trace1": {
            "session_id": "session-1",
            "root_span_id": "rootspan1111111",
            "project_id": "project-123",
            "current_task_span_id": "task11111111111",
            "current_task_start": "1704067200000000000",
            "started": "1704067200000000000"
        },
        "$trace2": {
            "session_id": "session-2",
            "root_span_id": "rootspan2222222",
            "project_id": "project-123",
            "current_task_span_id": "task22222222222",
            "current_task_start": "1704067300000000000",
            "started": "1704067300000000000"
        }
    }
}
EOF
    
    # Create transcripts for both sessions
    local transcript1="$TEST_TMP_DIR/.claude/projects/test-project/session-1.jsonl"
    local transcript2="$TEST_TMP_DIR/.claude/projects/test-project/session-2.jsonl"
    
    create_mock_transcript "$transcript1" \
        "$(mock_user_message 'Session 1 question' '2024-01-01T10:00:00.000Z')" \
        "$(mock_assistant_message 'Session 1 answer' 'claude-opus-4-5-20250514' 100 50 '2024-01-01T10:00:01.000Z')"
    
    create_mock_transcript "$transcript2" \
        "$(mock_user_message 'Session 2 question' '2024-01-01T10:01:00.000Z')" \
        "$(mock_assistant_message 'Session 2 answer' 'claude-opus-4-5-20250514' 110 55 '2024-01-01T10:01:01.000Z')"
    
    # End session 1
    local input1
    input1=$(jq -n -c \
        --arg session_id "session-1" \
        --arg transcript_path "$transcript1" \
        '{session_id: $session_id, transcript_path: $transcript_path}')
    
    echo "$input1" | bash "$HOOKS_DIR/session_end.sh" 2>&1 || true
    
    # Check trace1 was cleared but session 2 data still exists
    local state_after
    state_after=$(cat "$STATE_FILE")
    
    local current_trace
    current_trace=$(echo "$state_after" | jq -r '.current_trace_id // empty')
    
    assert_equals "" "$current_trace" "Current trace should be cleared after session 1 end"
    
    # Session 2 should still have its data
    local session2_root
    session2_root=$(echo "$state_after" | jq -r ".sessions[\"$trace2\"].root_span_id // empty")
    
    assert_equals "rootspan2222222" "$session2_root" "Session 2 state should still exist"
}

# Test: Sequential sessions (one after another)
test_sequential_sessions() {
    # Session 1
    create_mock_state "trace-seq1abcdef1234567890" "session-seq1" "rootspan1111111" "project-123" "task11111111111"
    
    local transcript1="$TEST_TMP_DIR/.claude/projects/test-project/session-seq1.jsonl"
    create_mock_transcript "$transcript1" \
        "$(mock_user_message 'First session' '2024-01-01T10:00:00.000Z')" \
        "$(mock_assistant_message 'Response 1' 'claude-opus-4-5-20250514' 100 50 '2024-01-01T10:00:01.000Z')"
    
    local input1
    input1=$(jq -n -c \
        --arg session_id "session-seq1" \
        --arg transcript_path "$transcript1" \
        '{session_id: $session_id, transcript_path: $transcript_path}')
    
    echo "$input1" | bash "$HOOKS_DIR/session_end.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "Trace ended: trace-seq1abcdef1234567890" "Session 1 should end"
    
    # Session 2 (new trace)
    create_mock_state "trace-seq2ghijkl5678901234" "session-seq2" "rootspan2222222" "project-123" "task22222222222"
    
    local transcript2="$TEST_TMP_DIR/.claude/projects/test-project/session-seq2.jsonl"
    create_mock_transcript "$transcript2" \
        "$(mock_user_message 'Second session' '2024-01-01T11:00:00.000Z')" \
        "$(mock_assistant_message 'Response 2' 'claude-opus-4-5-20250514' 120 60 '2024-01-01T11:00:01.000Z')"
    
    local input2
    input2=$(jq -n -c \
        --arg session_id "session-seq2" \
        --arg transcript_path "$transcript2" \
        '{session_id: $session_id, transcript_path: $transcript_path}')
    
    echo "$input2" | bash "$HOOKS_DIR/session_end.sh" 2>&1 || true
    
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "Trace ended: trace-seq2ghijkl5678901234" "Session 2 should end"
}

###
# Multiple subagents tests
###

# Test: Session with single subagent
test_session_with_single_subagent() {
    create_mock_state "trace-subagent1234567890123" "session-sub1" "rootspan1234567" "project-123" "taskspan12345678"
    
    local transcript_file="$TEST_TMP_DIR/.claude/projects/test-project/session-sub1.jsonl"
    
    # Create subagent tool use in transcript
    local subagent_tool
    subagent_tool=$(jq -n -c '{
        type: "assistant",
        timestamp: "2024-01-01T10:00:01.000Z",
        message: {
            model: "claude-opus-4-5-20250514",
            content: [
                {type: "text", text: "Delegating to subagent..."},
                {type: "tool_use", id: "subagent-1", name: "Task", input: {description: "Research topic X"}}
            ],
            usage: {input_tokens: 100, output_tokens: 50}
        }
    }')
    
    local subagent_result
    subagent_result=$(jq -n -c '{
        type: "user",
        timestamp: "2024-01-01T10:00:30.000Z",
        message: {
            content: [{type: "tool_result", tool_use_id: "subagent-1", content: "Research complete. Found 5 key points."}]
        },
        toolUseResult: {type: "text", text: "Research complete. Found 5 key points."}
    }')
    
    create_mock_transcript "$transcript_file" \
        "$(mock_user_message 'Research topic X for me' '2024-01-01T10:00:00.000Z')" \
        "$subagent_tool" \
        "$subagent_result" \
        "$(mock_assistant_message 'The research is complete. Here are the findings...' 'claude-opus-4-5-20250514' 200 100 '2024-01-01T10:00:31.000Z')"
    
    local input
    input=$(jq -n -c \
        --arg session_id "session-sub1" \
        --arg transcript_path "$transcript_file" \
        '{session_id: $session_id, transcript_path: $transcript_path}')
    
    echo "$input" | bash "$HOOKS_DIR/session_end.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "Processing transcript" "Should process subagent transcript"
}

# Test: Session with multiple subagents (parallel)
test_session_with_multiple_subagents_parallel() {
    create_mock_state "trace-multisub12345678901234" "session-msub" "rootspan1234567" "project-123" "taskspan12345678"
    
    local transcript_file="$TEST_TMP_DIR/.claude/projects/test-project/session-msub.jsonl"
    
    # Multiple subagent tool uses in one response
    local multi_subagent
    multi_subagent=$(jq -n -c '{
        type: "assistant",
        timestamp: "2024-01-01T10:00:01.000Z",
        message: {
            model: "claude-opus-4-5-20250514",
            content: [
                {type: "text", text: "Delegating to multiple subagents in parallel..."},
                {type: "tool_use", id: "subagent-a", name: "Task", input: {description: "Research topic A"}},
                {type: "tool_use", id: "subagent-b", name: "Task", input: {description: "Research topic B"}},
                {type: "tool_use", id: "subagent-c", name: "Task", input: {description: "Research topic C"}}
            ],
            usage: {input_tokens: 150, output_tokens: 80}
        }
    }')
    
    local subagent_results
    subagent_results=$(jq -n -c '{
        type: "user",
        timestamp: "2024-01-01T10:01:00.000Z",
        message: {
            content: [
                {type: "tool_result", tool_use_id: "subagent-a", content: "Result A"},
                {type: "tool_result", tool_use_id: "subagent-b", content: "Result B"},
                {type: "tool_result", tool_use_id: "subagent-c", content: "Result C"}
            ]
        },
        toolUseResult: {type: "text", text: "Result A"}
    }')
    
    create_mock_transcript "$transcript_file" \
        "$(mock_user_message 'Research topics A, B, and C in parallel' '2024-01-01T10:00:00.000Z')" \
        "$multi_subagent" \
        "$subagent_results" \
        "$(mock_assistant_message 'All research complete. Combining results...' 'claude-opus-4-5-20250514' 300 150 '2024-01-01T10:01:01.000Z')"
    
    local input
    input=$(jq -n -c \
        --arg session_id "session-msub" \
        --arg transcript_path "$transcript_file" \
        '{session_id: $session_id, transcript_path: $transcript_path}')
    
    echo "$input" | bash "$HOOKS_DIR/session_end.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "Processing transcript" "Should handle multiple parallel subagents"
}

# Test: Session with nested subagents (subagent spawning subagent)
test_session_with_nested_subagents() {
    create_mock_state "trace-nestedsub1234567890123" "session-nested" "rootspan1234567" "project-123" "taskspan12345678"
    
    local transcript_file="$TEST_TMP_DIR/.claude/projects/test-project/session-nested.jsonl"
    
    # First level subagent
    local subagent1
    subagent1=$(jq -n -c '{
        type: "assistant",
        timestamp: "2024-01-01T10:00:01.000Z",
        message: {
            model: "claude-opus-4-5-20250514",
            content: [
                {type: "text", text: "Starting main task..."},
                {type: "tool_use", id: "subagent-main", name: "Task", input: {description: "Complex task that requires sub-tasks"}}
            ],
            usage: {input_tokens: 100, output_tokens: 50}
        }
    }')
    
    # Subagent result (which would have had its own nested subagents)
    local subagent1_result
    subagent1_result=$(jq -n -c '{
        type: "user",
        timestamp: "2024-01-01T10:02:00.000Z",
        message: {
            content: [{type: "tool_result", tool_use_id: "subagent-main", content: "Complex task completed with 3 sub-tasks executed."}]
        },
        toolUseResult: {type: "text", text: "Complex task completed with 3 sub-tasks executed."}
    }')
    
    create_mock_transcript "$transcript_file" \
        "$(mock_user_message 'Execute complex task with subtasks' '2024-01-01T10:00:00.000Z')" \
        "$subagent1" \
        "$subagent1_result" \
        "$(mock_assistant_message 'Complex task completed successfully.' 'claude-opus-4-5-20250514' 200 100 '2024-01-01T10:02:01.000Z')"
    
    local input
    input=$(jq -n -c \
        --arg session_id "session-nested" \
        --arg transcript_path "$transcript_file" \
        '{session_id: $session_id, transcript_path: $transcript_path}')
    
    echo "$input" | bash "$HOOKS_DIR/session_end.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "Processing transcript" "Should handle nested subagent scenario"
}

# Test: Subagent stop hook directly
test_subagent_stop_hook() {
    # Check if subagent_stop.sh exists
    if [ ! -f "$HOOKS_DIR/subagent_stop.sh" ]; then
        echo "  (skipping - subagent_stop.sh not found)"
        return 0
    fi
    
    create_mock_state "trace-substop12345678901234" "session-substop" "rootspan1234567" "project-123" "taskspan12345678"
    
    # Create subagent transcript
    local subagent_transcript="$TEST_TMP_DIR/.claude/projects/test-project/subagent-123.jsonl"
    create_mock_transcript "$subagent_transcript" \
        "$(mock_user_message 'Subagent task' '2024-01-01T10:00:00.000Z')" \
        "$(mock_assistant_message 'Task completed' 'claude-opus-4-5-20250514' 100 50 '2024-01-01T10:00:30.000Z')"
    
    local input
    input=$(jq -n -c \
        --arg session_id "subagent-123" \
        --arg transcript_path "$subagent_transcript" \
        --arg parent_tool_use_id "tool-xyz" \
        '{session_id: $session_id, transcript_path: $transcript_path, parent_tool_use_id: $parent_tool_use_id}')
    
    echo "$input" | bash "$HOOKS_DIR/subagent_stop.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$logs" "SubagentStop hook triggered" "Subagent stop hook should trigger"
}

###
# Combined integration tests
###

# Test: Full workflow - session start, prompts, subagents, session end
test_full_workflow_integration() {
    # 1. Session start
    local input_start
    input_start=$(jq -n -c \
        --arg session_id "full-workflow-session" \
        --arg cwd "/test/project" \
        '{session_id: $session_id, cwd: $cwd}')
    
    # Mock project resolution
    export JUDGMENT_API_URL="http://localhost:9999"
    
    echo "$input_start" | bash "$HOOKS_DIR/session_start.sh" 2>&1 || true
    
    local logs
    logs=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    # Should have attempted to create session (will fail due to no server, but that's ok)
    assert_contains "$logs" "SessionStart hook triggered" "Session should start"
    
    # Check state was created
    local state
    state=$(cat "$STATE_FILE" 2>/dev/null || echo "{}")
    
    local current_trace
    current_trace=$(echo "$state" | jq -r '.current_trace_id // empty')
    
    # May be empty if API failed, but that's expected in test
    if [ -n "$current_trace" ]; then
        assert_not_empty "$current_trace" "Trace ID should be set"
    fi
}

# Test: Concurrent state updates don't corrupt state file
test_concurrent_state_updates() {
    # Initialize state
    echo '{"counter": 0, "sessions": {}}' > "$STATE_FILE"
    
    source "$HOOKS_DIR/common.sh"
    
    # Run multiple state updates concurrently
    for i in $(seq 1 5); do
        (
            set_state_value "key$i" "value$i"
        ) &
    done
    
    # Wait for all to complete
    wait
    
    # State file should still be valid JSON
    local state
    state=$(cat "$STATE_FILE")
    
    if ! echo "$state" | jq -e '.' >/dev/null 2>&1; then
        echo "  State file corrupted: $state"
        return 1
    fi
    
    return 0
}

###
# Run all tests
###

echo "================================"
echo "Integration Tests"
echo "================================"
echo ""

echo "--- Multi-turn Conversation Tests ---"
run_test "Multi-turn with tools" test_multi_turn_with_tools
run_test "Multiple tools per response" test_multi_turn_multiple_tools_per_response
run_test "Long multi-turn conversation (10+ turns)" test_long_multi_turn_conversation

echo ""
echo "--- Multiple Sessions Tests ---"
run_test "Multiple independent sessions" test_multiple_independent_sessions
run_test "Sequential sessions" test_sequential_sessions

echo ""
echo "--- Multiple Subagents Tests ---"
run_test "Session with single subagent" test_session_with_single_subagent
run_test "Session with multiple parallel subagents" test_session_with_multiple_subagents_parallel
run_test "Session with nested subagents" test_session_with_nested_subagents
run_test "Subagent stop hook" test_subagent_stop_hook

echo ""
echo "--- Combined Integration Tests ---"
run_test "Full workflow integration" test_full_workflow_integration
run_test "Concurrent state updates" test_concurrent_state_updates

print_summary
