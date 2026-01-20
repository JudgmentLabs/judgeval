#!/bin/bash
###
# PostToolUse Hook - Tracks tool usage and subagent spawning
###

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

debug "PostToolUse hook triggered"
tracing_enabled || { debug "Tracing disabled"; exit 0; }

INPUT=$(cat)
debug "PostToolUse input: $(echo "$INPUT" | jq -c '.' 2>/dev/null | head -c 500)"

echo "$INPUT" | jq -e '.' >/dev/null 2>&1 || { debug "Invalid JSON"; exit 0; }

TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty' 2>/dev/null)
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty' 2>/dev/null)
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // empty' 2>/dev/null)

[ -z "$TOOL_NAME" ] || [ -z "$SESSION_ID" ] && { debug "No tool/session"; exit 0; }

TOOL_COUNT=$(get_session_state "$SESSION_ID" "current_turn_tool_count")
TOOL_COUNT=$((${TOOL_COUNT:-0} + 1))
set_session_state "$SESSION_ID" "current_turn_tool_count" "$TOOL_COUNT"

# If this is a Task tool, extract the spawned agent_id and record the parent relationship
if [ "$TOOL_NAME" = "Task" ]; then
    TOOL_RESPONSE=$(echo "$INPUT" | jq -c '.tool_response // {}' 2>/dev/null)
    
    # Try multiple ways to extract the agent_id:
    # 1. Direct .agentId field (from toolUseResult)
    # 2. Text content containing "agentId: XXXXX"
    SPAWNED_AGENT_ID=$(echo "$TOOL_RESPONSE" | jq -r '.agentId // empty' 2>/dev/null)
    
    if [ -z "$SPAWNED_AGENT_ID" ]; then
        # Try to extract from text content
        SPAWNED_AGENT_ID=$(echo "$TOOL_RESPONSE" | jq -r '
            if type == "object" then
                (.content // [])[] | select(.type == "text") | .text
            elif type == "string" then
                .
            else
                empty
            end
        ' 2>/dev/null | grep -oE 'agentId: [a-zA-Z0-9]+' | head -1 | sed 's/agentId: //')
    fi
    
    if [ -n "$SPAWNED_AGENT_ID" ]; then
        TRACE_ID=$(get_state_value "current_trace_id")
        if [ -n "$TRACE_ID" ]; then
            # Store the parent info: which transcript spawned this agent
            # This helps SubagentStop determine the real parent
            set_session_state "$TRACE_ID" "parent_of_${SPAWNED_AGENT_ID}" "$TRANSCRIPT_PATH"
            debug "Task spawned agent $SPAWNED_AGENT_ID from $TRANSCRIPT_PATH"
        fi
    else
        debug "Task tool completed but no agentId found in response"
    fi
fi

log "INFO" "Tool used: $TOOL_NAME (count=$TOOL_COUNT)"
exit 0
