#!/bin/bash
###
# UserPromptSubmit Hook - Creates Task span when user submits a prompt
###

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

debug "UserPromptSubmit hook triggered"
tracing_enabled || { debug "Tracing disabled"; exit 0; }
check_requirements || exit 0

INPUT=$(cat)
debug "UserPromptSubmit input: $(echo "$INPUT" | jq -c '.' 2>/dev/null | head -c 500)"

echo "$INPUT" | jq -e '.' >/dev/null 2>&1 || { debug "Invalid JSON"; exit 0; }

SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty' 2>/dev/null)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty' 2>/dev/null)

TRACE_ID=$(get_state_value "current_trace_id")
[ -z "$TRACE_ID" ] && { debug "No current trace"; exit 0; }

ROOT_SPAN_ID=$(get_session_state "$TRACE_ID" "root_span_id")
PROJECT_ID=$(get_session_state "$TRACE_ID" "project_id")

[ -z "$ROOT_SPAN_ID" ] || [ -z "$PROJECT_ID" ] && { debug "Missing trace state"; exit 0; }

TASK_SPAN_ID=$(generate_uuid | sed 's/-//g' | head -c 16)
START_TIME=$(get_time_nanos)
PROMPT_JSON=$(echo "$PROMPT" | jq -Rs '.')

ATTRIBUTES=$(build_otlp_attributes "$(jq -n \
    --arg span_kind "task" \
    --argjson input "$PROMPT_JSON" \
    --arg session_id "$SESSION_ID" \
    '{ "judgment.span_kind": $span_kind, "judgment.input": $input, "session_id": $session_id }')")

SPAN=$(build_otlp_span "$TRACE_ID" "$TASK_SPAN_ID" "$ROOT_SPAN_ID" "Task" "task" "$START_TIME" "$START_TIME" "$ATTRIBUTES" 0)
insert_span "$PROJECT_ID" "$SPAN" || { log "ERROR" "Failed to create task span"; exit 0; }

set_session_state_batch "$TRACE_ID" \
    "current_task_span_id" "$TASK_SPAN_ID" \
    "current_task_start" "$START_TIME"

log "INFO" "Task started: span=$TASK_SPAN_ID"
exit 0
