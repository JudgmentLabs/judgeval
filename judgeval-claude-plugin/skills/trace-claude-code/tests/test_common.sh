#!/bin/bash
###
# Tests for common.sh utilities
###

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/test_helpers.sh"

# Test: generate_uuid produces valid UUIDs
test_generate_uuid() {
    source "$HOOKS_DIR/common.sh"
    
    local uuid1 uuid2
    uuid1=$(generate_uuid)
    uuid2=$(generate_uuid)
    
    # Should be non-empty
    assert_not_empty "$uuid1" "UUID should not be empty"
    
    # Should be lowercase
    local lowercase
    lowercase=$(echo "$uuid1" | tr '[:upper:]' '[:lower:]')
    assert_equals "$lowercase" "$uuid1" "UUID should be lowercase"
    
    # Two UUIDs should be different
    if [ "$uuid1" = "$uuid2" ]; then
        echo "  FAIL: Two UUIDs should be different"
        return 1
    fi
    
    return 0
}

# Test: get_time_nanos returns nanoseconds
test_get_time_nanos() {
    source "$HOOKS_DIR/common.sh"
    
    local nanos
    nanos=$(get_time_nanos)
    
    assert_not_empty "$nanos" "Nanoseconds should not be empty"
    
    # Should be a large number (> 1 trillion for any time after 1970)
    if [ "$nanos" -lt 1000000000000000000 ]; then
        echo "  FAIL: Nanoseconds value too small: $nanos"
        return 1
    fi
    
    return 0
}

# Test: state management - load and save
test_state_load_save() {
    source "$HOOKS_DIR/common.sh"
    
    # Initial state should be empty object
    local state
    state=$(load_state)
    assert_equals "{}" "$state" "Initial state should be empty object"
    
    # Save state
    save_state '{"test": "value"}'
    
    # Load should return saved state
    state=$(load_state)
    local test_val
    test_val=$(echo "$state" | jq -r '.test')
    assert_equals "value" "$test_val" "Saved state should be retrievable"
}

# Test: set_state_value and get_state_value
test_state_value_ops() {
    source "$HOOKS_DIR/common.sh"
    
    # Set a value
    set_state_value "my_key" "my_value"
    
    # Get the value
    local result
    result=$(get_state_value "my_key")
    assert_equals "my_value" "$result" "Should retrieve set value"
    
    # Get non-existent key should be empty
    result=$(get_state_value "nonexistent")
    assert_equals "" "$result" "Non-existent key should return empty"
}

# Test: session state management
test_session_state() {
    source "$HOOKS_DIR/common.sh"
    
    local session_id="test-session-123"
    
    # Set session state
    set_session_state "$session_id" "span_id" "span-456"
    set_session_state "$session_id" "project_id" "proj-789"
    
    # Get session state
    local span_id project_id
    span_id=$(get_session_state "$session_id" "span_id")
    project_id=$(get_session_state "$session_id" "project_id")
    
    assert_equals "span-456" "$span_id" "Should retrieve session span_id"
    assert_equals "proj-789" "$project_id" "Should retrieve session project_id"
}

# Test: set_session_state_batch
test_session_state_batch() {
    source "$HOOKS_DIR/common.sh"
    
    local session_id="batch-session"
    
    # Set multiple values at once
    set_session_state_batch "$session_id" \
        "key1" "value1" \
        "key2" "value2" \
        "key3" "value3"
    
    # Verify all values
    assert_equals "value1" "$(get_session_state "$session_id" "key1")" "Batch key1"
    assert_equals "value2" "$(get_session_state "$session_id" "key2")" "Batch key2"
    assert_equals "value3" "$(get_session_state "$session_id" "key3")" "Batch key3"
}

# Test: tracing_enabled function
test_tracing_enabled() {
    source "$HOOKS_DIR/common.sh"
    
    # When true
    export TRACE_TO_JUDGEVAL="true"
    if ! tracing_enabled; then
        echo "  FAIL: tracing_enabled should return true"
        return 1
    fi
    
    # When TRUE (case insensitive)
    export TRACE_TO_JUDGEVAL="TRUE"
    if ! tracing_enabled; then
        echo "  FAIL: tracing_enabled should handle uppercase"
        return 1
    fi
    
    # When false
    export TRACE_TO_JUDGEVAL="false"
    if tracing_enabled; then
        echo "  FAIL: tracing_enabled should return false"
        return 1
    fi
    
    # When empty
    export TRACE_TO_JUDGEVAL=""
    if tracing_enabled; then
        echo "  FAIL: tracing_enabled should return false when empty"
        return 1
    fi
    
    return 0
}

# Test: check_requirements
test_check_requirements() {
    # With valid API key - source with env vars set
    export JUDGMENT_API_KEY="test-key"
    export JUDGMENT_ORG_ID="test-org"
    source "$HOOKS_DIR/common.sh"
    
    if ! check_requirements; then
        echo "  FAIL: check_requirements should pass with valid config"
        return 1
    fi
    
    # Without API key - need to re-export and re-source
    export JUDGMENT_API_KEY=""
    export API_KEY=""  # Also clear the internal var
    
    if check_requirements 2>/dev/null; then
        echo "  FAIL: check_requirements should fail without API key"
        return 1
    fi
    
    return 0
}

# Test: build_otlp_attributes
test_build_otlp_attributes() {
    source "$HOOKS_DIR/common.sh"
    
    local input='{"string_key": "string_value", "int_key": 42, "bool_key": true}'
    local attrs
    attrs=$(build_otlp_attributes "$input")
    
    # Should produce an array
    local is_array
    is_array=$(echo "$attrs" | jq 'type == "array"')
    assert_equals "true" "$is_array" "Should produce an array"
    
    # Check string attribute
    local string_val
    string_val=$(echo "$attrs" | jq -r '.[] | select(.key == "string_key") | .value.stringValue')
    assert_equals "string_value" "$string_val" "String attribute should be correct"
    
    # Check int attribute
    local int_val
    int_val=$(echo "$attrs" | jq -r '.[] | select(.key == "int_key") | .value.intValue')
    assert_equals "42" "$int_val" "Int attribute should be correct"
    
    # Check bool attribute
    local bool_val
    bool_val=$(echo "$attrs" | jq -r '.[] | select(.key == "bool_key") | .value.boolValue')
    assert_equals "true" "$bool_val" "Bool attribute should be correct"
}

# Test: build_otlp_span
test_build_otlp_span() {
    source "$HOOKS_DIR/common.sh"
    
    local trace_id="abc123def456abc123def456abc123de"
    local span_id="span12345678"
    local parent_span_id="parent123456"
    local name="test-span"
    local start_time="1704067200000000000"
    local end_time="1704067201000000000"
    local attrs='[{"key": "test", "value": {"stringValue": "value"}}]'
    
    local span
    span=$(build_otlp_span "$trace_id" "$span_id" "$parent_span_id" "$name" "task" "$start_time" "$end_time" "$attrs" 0)
    
    # Verify fields
    assert_json_equals "$span" ".traceId" "$trace_id" "Trace ID"
    assert_json_equals "$span" ".spanId" "$span_id" "Span ID"
    assert_json_equals "$span" ".parentSpanId" "$parent_span_id" "Parent Span ID"
    assert_json_equals "$span" ".name" "$name" "Span name"
    assert_json_equals "$span" ".startTimeUnixNano" "$start_time" "Start time"
    assert_json_equals "$span" ".endTimeUnixNano" "$end_time" "End time"
}

# Test: logging functions
test_logging() {
    source "$HOOKS_DIR/common.sh"
    
    # Test log function
    log "INFO" "Test message"
    
    assert_file_exists "$LOG_FILE" "Log file should be created"
    
    local log_contents
    log_contents=$(cat "$LOG_FILE")
    
    assert_contains "$log_contents" "INFO" "Should contain log level"
    assert_contains "$log_contents" "Test message" "Should contain message"
}

# Test: debug logging (when enabled)
test_debug_logging_enabled() {
    export JUDGEVAL_CC_DEBUG="true"
    source "$HOOKS_DIR/common.sh"
    
    debug "Debug test message"
    
    local log_contents
    log_contents=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    assert_contains "$log_contents" "DEBUG" "Should contain DEBUG level when enabled"
    assert_contains "$log_contents" "Debug test message" "Should contain debug message"
}

# Test: debug logging (when disabled)
test_debug_logging_disabled() {
    export JUDGEVAL_CC_DEBUG="false"
    source "$HOOKS_DIR/common.sh"
    
    debug "Should not appear"
    
    local log_contents
    log_contents=$(cat "$LOG_FILE" 2>/dev/null || echo "")
    
    if [[ "$log_contents" == *"Should not appear"* ]]; then
        echo "  FAIL: Debug message should not appear when disabled"
        return 1
    fi
    
    return 0
}

# Test: file locking
test_file_locking() {
    source "$HOOKS_DIR/common.sh"
    
    # Acquire lock
    if ! acquire_lock 1; then
        echo "  FAIL: Should acquire lock"
        return 1
    fi
    
    # Lock directory should exist
    if [ ! -d "$LOCK_DIR" ]; then
        echo "  FAIL: Lock directory should exist"
        release_lock
        return 1
    fi
    
    # Release lock
    release_lock
    
    # Lock directory should be gone
    if [ -d "$LOCK_DIR" ]; then
        echo "  FAIL: Lock directory should be removed after release"
        return 1
    fi
    
    return 0
}

# Run all tests
echo "================================"
echo "Testing common.sh"
echo "================================"
echo ""

run_test "generate_uuid" test_generate_uuid
run_test "get_time_nanos" test_get_time_nanos
run_test "State load/save" test_state_load_save
run_test "State value operations" test_state_value_ops
run_test "Session state" test_session_state
run_test "Session state batch" test_session_state_batch
run_test "tracing_enabled" test_tracing_enabled
run_test "check_requirements" test_check_requirements
run_test "build_otlp_attributes" test_build_otlp_attributes
run_test "build_otlp_span" test_build_otlp_span
run_test "Logging" test_logging
run_test "Debug logging (enabled)" test_debug_logging_enabled
run_test "Debug logging (disabled)" test_debug_logging_disabled
run_test "File locking" test_file_locking

print_summary
