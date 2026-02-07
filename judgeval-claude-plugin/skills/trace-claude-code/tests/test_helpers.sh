#!/bin/bash
###
# Test helpers for Judgeval Claude Code tracing hooks
###

# Test state
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
CURRENT_TEST=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Test directories
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOKS_DIR="$TEST_DIR/../hooks"
TEST_TMP_DIR=""

# Setup test environment
setup_test_env() {
    TEST_TMP_DIR=$(mktemp -d)
    export HOME="$TEST_TMP_DIR"
    export LOG_FILE="$TEST_TMP_DIR/.claude/state/judgeval_hook.log"
    export STATE_FILE="$TEST_TMP_DIR/.claude/state/judgeval_state.json"
    export QUEUE_DIR="$TEST_TMP_DIR/.claude/state/judgeval_queue"
    
    mkdir -p "$TEST_TMP_DIR/.claude/state"
    mkdir -p "$TEST_TMP_DIR/.claude/projects/test-project"
    
    # Mock environment
    export TRACE_TO_JUDGEVAL="true"
    export JUDGMENT_API_KEY="test-api-key"
    export JUDGMENT_ORG_ID="test-org-id"
    export JUDGMENT_API_URL="http://localhost:9999"
    export JUDGEVAL_CC_PROJECT="test-project"
    export JUDGEVAL_CC_DEBUG="true"
}

# Teardown test environment
teardown_test_env() {
    if [ -n "$TEST_TMP_DIR" ] && [ -d "$TEST_TMP_DIR" ]; then
        rm -rf "$TEST_TMP_DIR"
    fi
}

# Run a test function
run_test() {
    local test_name="$1"
    local test_func="$2"
    
    CURRENT_TEST="$test_name"
    TESTS_RUN=$((TESTS_RUN + 1))
    
    setup_test_env
    
    # Run the test
    local result=0
    $test_func || result=$?
    
    if [ $result -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗${NC} $test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    
    teardown_test_env
}

# Assertions
assert_equals() {
    local expected="$1"
    local actual="$2"
    local msg="${3:-Values should be equal}"
    
    if [ "$expected" != "$actual" ]; then
        echo -e "${RED}  FAIL: $msg${NC}"
        echo "    Expected: $expected"
        echo "    Actual:   $actual"
        return 1
    fi
    return 0
}

assert_contains() {
    local haystack="$1"
    local needle="$2"
    local msg="${3:-String should contain substring}"
    
    if [[ "$haystack" != *"$needle"* ]]; then
        echo -e "${RED}  FAIL: $msg${NC}"
        echo "    Looking for: $needle"
        echo "    In: $haystack"
        return 1
    fi
    return 0
}

assert_file_exists() {
    local file="$1"
    local msg="${2:-File should exist}"
    
    if [ ! -f "$file" ]; then
        echo -e "${RED}  FAIL: $msg${NC}"
        echo "    File not found: $file"
        return 1
    fi
    return 0
}

assert_not_empty() {
    local value="$1"
    local msg="${2:-Value should not be empty}"
    
    if [ -z "$value" ]; then
        echo -e "${RED}  FAIL: $msg${NC}"
        return 1
    fi
    return 0
}

assert_json_equals() {
    local json="$1"
    local path="$2"
    local expected="$3"
    local msg="${4:-JSON value should match}"
    
    local actual
    actual=$(echo "$json" | jq -r "$path" 2>/dev/null)
    
    if [ "$expected" != "$actual" ]; then
        echo -e "${RED}  FAIL: $msg${NC}"
        echo "    Path: $path"
        echo "    Expected: $expected"
        echo "    Actual:   $actual"
        return 1
    fi
    return 0
}

# Mock API server (captures requests to a file)
MOCK_API_REQUESTS=""

mock_curl() {
    # Capture the request
    local args=("$@")
    echo "${args[*]}" >> "$TEST_TMP_DIR/curl_requests.log"
    
    # Return success response
    echo '{"row_ids": ["mock-row-id"]}'
    return 0
}

# Create mock state file
create_mock_state() {
    local trace_id="$1"
    local session_id="${2:-test-session}"
    local root_span_id="${3:-root-span-123}"
    local project_id="${4:-project-123}"
    local task_span_id="${5:-task-span-456}"
    
    cat > "$STATE_FILE" << EOF
{
    "current_trace_id": "$trace_id",
    "project_id": "$project_id",
    "sessions": {
        "$trace_id": {
            "session_id": "$session_id",
            "root_span_id": "$root_span_id",
            "project_id": "$project_id",
            "current_task_span_id": "$task_span_id",
            "current_task_start": "1704067200000000000",
            "started": "1704067200000000000"
        }
    }
}
EOF
}

# Create mock transcript file
create_mock_transcript() {
    local file="$1"
    shift
    
    mkdir -p "$(dirname "$file")"
    
    # Write each line as a separate JSONL entry
    for line in "$@"; do
        echo "$line" >> "$file"
    done
}

# Create a basic user message
mock_user_message() {
    local content="$1"
    local timestamp="${2:-2024-01-01T12:00:00.000Z}"
    
    jq -n -c \
        --arg type "user" \
        --arg content "$content" \
        --arg ts "$timestamp" \
        '{type: $type, timestamp: $ts, message: {content: $content}}'
}

# Create a user message with tool result
mock_tool_result_message() {
    local tool_use_id="$1"
    local content="$2"
    local timestamp="${3:-2024-01-01T12:00:05.000Z}"
    
    jq -n -c \
        --arg type "user" \
        --arg tool_use_id "$tool_use_id" \
        --arg content "$content" \
        --arg ts "$timestamp" \
        '{
            type: $type,
            timestamp: $ts,
            message: {
                content: [{type: "tool_result", tool_use_id: $tool_use_id, content: $content}]
            },
            toolUseResult: {type: "text", text: $content}
        }'
}

# Create an assistant message with text
mock_assistant_message() {
    local text="$1"
    local model="${2:-claude-3-5-sonnet-20241022}"
    local input_tokens="${3:-100}"
    local output_tokens="${4:-50}"
    local timestamp="${5:-2024-01-01T12:00:03.000Z}"
    
    jq -n -c \
        --arg type "assistant" \
        --arg text "$text" \
        --arg model "$model" \
        --argjson input_tokens "$input_tokens" \
        --argjson output_tokens "$output_tokens" \
        --arg ts "$timestamp" \
        '{
            type: $type,
            timestamp: $ts,
            message: {
                model: $model,
                content: [{type: "text", text: $text}],
                usage: {input_tokens: $input_tokens, output_tokens: $output_tokens}
            }
        }'
}

# Create an assistant message with tool use
mock_assistant_tool_use() {
    local tool_id="$1"
    local tool_name="$2"
    local tool_input="$3"
    local text="${4:-}"
    local model="${5:-claude-3-5-sonnet-20241022}"
    local timestamp="${6:-2024-01-01T12:00:02.000Z}"
    
    if [ -n "$text" ]; then
        jq -n -c \
            --arg type "assistant" \
            --arg tool_id "$tool_id" \
            --arg tool_name "$tool_name" \
            --argjson tool_input "$tool_input" \
            --arg text "$text" \
            --arg model "$model" \
            --arg ts "$timestamp" \
            '{
                type: $type,
                timestamp: $ts,
                message: {
                    model: $model,
                    content: [
                        {type: "text", text: $text},
                        {type: "tool_use", id: $tool_id, name: $tool_name, input: $tool_input}
                    ],
                    usage: {input_tokens: 100, output_tokens: 50}
                }
            }'
    else
        jq -n -c \
            --arg type "assistant" \
            --arg tool_id "$tool_id" \
            --arg tool_name "$tool_name" \
            --argjson tool_input "$tool_input" \
            --arg model "$model" \
            --arg ts "$timestamp" \
            '{
                type: $type,
                timestamp: $ts,
                message: {
                    model: $model,
                    content: [
                        {type: "tool_use", id: $tool_id, name: $tool_name, input: $tool_input}
                    ],
                    usage: {input_tokens: 100, output_tokens: 50}
                }
            }'
    fi
}

# Print test summary
print_summary() {
    echo ""
    echo "================================"
    echo "Test Summary"
    echo "================================"
    echo "Total:  $TESTS_RUN"
    echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
    echo ""
    
    if [ $TESTS_FAILED -gt 0 ]; then
        return 1
    fi
    return 0
}
