#!/bin/bash
###
# SubagentStop Hook - Creates spans for subagent execution
#
# When a subagent (Task tool) completes, this hook:
# 1. Creates a container span for the subagent
# 2. Parses the subagent's transcript for LLM/tool spans
# 3. Links everything to the parent trace
###

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

debug "SubagentStop hook triggered"
tracing_enabled || { debug "Tracing disabled"; exit 0; }
check_requirements || exit 0

INPUT=$(cat)
debug "SubagentStop input: $(echo "$INPUT" | jq -c '.' 2>/dev/null | head -c 1000)"

echo "$INPUT" | jq -e '.' >/dev/null 2>&1 || { debug "Invalid JSON"; exit 0; }

# Extract subagent info from hook input
SUBAGENT_ID=$(echo "$INPUT" | jq -r '.agent_id // .subagent_id // empty' 2>/dev/null)
# Use agent_transcript_path (subagent's own transcript), NOT transcript_path (parent session)
SUBAGENT_TRANSCRIPT=$(echo "$INPUT" | jq -r '.agent_transcript_path // empty' 2>/dev/null)
TASK_DESCRIPTION=$(echo "$INPUT" | jq -r '.task // .description // empty' 2>/dev/null)
PARENT_SESSION_ID=$(echo "$INPUT" | jq -r '.parent_session_id // .session_id // empty' 2>/dev/null)

debug "Subagent ID: $SUBAGENT_ID"
debug "Subagent transcript: $SUBAGENT_TRANSCRIPT"

# Get parent trace context
TRACE_ID=$(get_state_value "current_trace_id")
[ -z "$TRACE_ID" ] && { debug "No current trace"; exit 0; }

PARENT_TASK_SPAN_ID=$(get_session_state "$TRACE_ID" "current_task_span_id")
PROJECT_ID=$(get_session_state "$TRACE_ID" "project_id")
ROOT_SPAN_ID=$(get_session_state "$TRACE_ID" "root_span_id")

[ -z "$PROJECT_ID" ] && { debug "No project ID"; exit 0; }
[ -z "$ROOT_SPAN_ID" ] && { debug "No root span"; exit 0; }

# Look up the REAL parent from PostToolUse tracking
# PostToolUse stores which transcript spawned this agent
PARENT_TRANSCRIPT=$(get_agent_parent_transcript "$TRACE_ID" "$SUBAGENT_ID")
PARENT_AGENT_ID=""
PARENT_AGENT_SPAN=""

if [ -n "$PARENT_TRANSCRIPT" ]; then
    debug "Agent $SUBAGENT_ID was spawned from transcript: $PARENT_TRANSCRIPT"
    
    # Check if parent transcript is a subagent transcript (contains /subagents/)
    if [[ "$PARENT_TRANSCRIPT" == */subagents/* ]]; then
        # Extract parent agent_id from path: .../subagents/agent-XXXXX.jsonl
        PARENT_AGENT_ID=$(basename "$PARENT_TRANSCRIPT" .jsonl | sed 's/agent-//')
        debug "Parent is subagent: $PARENT_AGENT_ID"
        
        # Get the parent's span_id
        PARENT_AGENT_SPAN=$(get_subagent_span_id "$TRACE_ID" "$PARENT_AGENT_ID")
        if [ -n "$PARENT_AGENT_SPAN" ]; then
            debug "Found parent agent span: $PARENT_AGENT_SPAN"
        fi
    else
        debug "Parent is main session (not a subagent)"
    fi
fi

# Determine nesting depth
SUBAGENT_COUNT=$(get_subagent_count "$TRACE_ID")
if [ -n "$PARENT_AGENT_ID" ]; then
    # If we have a parent agent, we're nested
    NESTING_DEPTH=$((SUBAGENT_COUNT + 1))
else
    # First level subagent
    NESTING_DEPTH=1
fi

# For span parenting: use parent agent span if known, otherwise task span
if [ -n "$PARENT_AGENT_SPAN" ]; then
    PARENT_SPAN_ID="$PARENT_AGENT_SPAN"
    debug "Using parent agent span: $PARENT_SPAN_ID"
elif [ -n "$PARENT_TASK_SPAN_ID" ]; then
    PARENT_SPAN_ID="$PARENT_TASK_SPAN_ID"
    debug "Using task span as parent: $PARENT_SPAN_ID"
else
    PARENT_SPAN_ID="$ROOT_SPAN_ID"
    debug "Using root span as parent: $PARENT_SPAN_ID"
fi

# Generate subagent container span
SUBAGENT_SPAN_ID=$(generate_uuid | sed 's/-//g' | head -c 16)
START_TIME=$(get_time_nanos)

# Register this agent in the registry for tracking (include transcript path for nested lookups)
register_subagent "$TRACE_ID" "$SUBAGENT_ID" "$SUBAGENT_SPAN_ID" "$PARENT_AGENT_ID" "$SUBAGENT_TRANSCRIPT"
debug "Registered subagent $SUBAGENT_ID (span=$SUBAGENT_SPAN_ID, parent_agent=${PARENT_AGENT_ID:-none}, depth=$NESTING_DEPTH)"

# If no transcript path provided, try to find it
if [ -z "$SUBAGENT_TRANSCRIPT" ] || [ ! -f "$SUBAGENT_TRANSCRIPT" ]; then
    if [ -n "$SUBAGENT_ID" ]; then
        for pattern in \
            "$HOME/.claude/projects/*/agent-${SUBAGENT_ID}.jsonl" \
            "$HOME/.claude/projects/*/${SUBAGENT_ID}.jsonl" \
            "$HOME/.claude/state/agent-${SUBAGENT_ID}.jsonl"; do
            # shellcheck disable=SC2086
            FOUND=$(find $pattern -type f 2>/dev/null | head -1)
            if [ -n "$FOUND" ] && [ -f "$FOUND" ]; then
                SUBAGENT_TRANSCRIPT="$FOUND"
                debug "Found subagent transcript: $SUBAGENT_TRANSCRIPT"
                break
            fi
        done
    fi
fi

# Track subagent stats
LLM_CALLS=0
TOOL_CALLS=0
SUBAGENT_OUTPUT=""

# Parse transcript for LLM and tool spans
if [ -n "$SUBAGENT_TRANSCRIPT" ] && [ -f "$SUBAGENT_TRANSCRIPT" ]; then
    debug "Parsing subagent transcript: $SUBAGENT_TRANSCRIPT"
    
    CURRENT_OUTPUT=""
    CURRENT_MODEL=""
    CURRENT_PROMPT_TOKENS=0
    CURRENT_COMPLETION_TOKENS=0
    LLM_START_TIME=""
    LLM_END_TIME=""
    PENDING_TOOLS="{}"

    create_subagent_llm_span() {
        local output="$1" model="$2" prompt="$3" completion="$4"
        local start_time="$5" end_time="$6"
        [ -z "$output" ] && return
        
        local span_id provider input_json output_json attrs span
        span_id=$(generate_uuid | sed 's/-//g' | head -c 16)
        provider=$(detect_provider "$model")
        
        input_json=$(jq -n --arg d "$TASK_DESCRIPTION" '[{role: "user", content: $d}]' | jq -c '.' | jq -Rs '.')
        output_json=$(jq -n --arg c "$output" '[{role: "assistant", content: $c}]' | jq -c '.' | jq -Rs '.')
        
        attrs=$(build_otlp_attributes "$(jq -n \
            --arg span_kind "llm" --argjson input "$input_json" --argjson output "$output_json" \
            --arg model "${model:-claude}" --arg provider "$provider" \
            --argjson prompt "$prompt" --argjson completion "$completion" \
            '{
              "judgment.span_kind": $span_kind,
              "judgment.input": $input,
              "judgment.output": $output,
              "judgment.llm.provider": $provider,
              "judgment.llm.model": $model,
              "judgment.usage.non_cached_input_tokens": $prompt,
              "judgment.usage.output_tokens": $completion,
              "subagent_id": "'"$SUBAGENT_ID"'"
            }')")
        
        span=$(build_otlp_span "$TRACE_ID" "$span_id" "$SUBAGENT_SPAN_ID" "${model:-anthropic.messages.create}" "llm" "$start_time" "$end_time" "$attrs" 20)
        
        if insert_span "$PROJECT_ID" "$span" >/dev/null; then
            LLM_CALLS=$((LLM_CALLS + 1))
            debug "Subagent LLM span: $model"
        fi
    }

    create_subagent_tool_span() {
        local tool_name="$1" tool_input="$2" tool_output="$3" start_time="$4" end_time="$5"
        [ -z "$tool_name" ] && return
        
        local span_id input_json output_json attrs span
        span_id=$(generate_uuid | sed 's/-//g' | head -c 16)
        input_json=$(echo "$tool_input" | jq -c '.' 2>/dev/null | jq -Rs '.')
        output_json=$(echo "$tool_output" | jq -Rs '.')
        
        attrs=$(build_otlp_attributes "$(jq -n --arg span_kind "tool" --argjson input "$input_json" --argjson output "$output_json" --arg tool_name "$tool_name" --arg subagent_id "$SUBAGENT_ID" \
            '{"judgment.span_kind": $span_kind, "judgment.input": $input, "judgment.output": $output, "tool_name": $tool_name, "subagent_id": $subagent_id}')")
        
        span=$(build_otlp_span "$TRACE_ID" "$span_id" "$SUBAGENT_SPAN_ID" "$tool_name" "tool" "$start_time" "$end_time" "$attrs" 20)
        
        if insert_span "$PROJECT_ID" "$span" >/dev/null; then
            TOOL_CALLS=$((TOOL_CALLS + 1))
            debug "Subagent tool span: $tool_name"
        fi
    }

    # Parse the subagent transcript (same logic as session_end.sh)
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        
        MSG_TYPE=$(echo "$line" | jq -r '.type // empty' 2>/dev/null)
        TIMESTAMP=$(echo "$line" | jq -r '.timestamp // empty' 2>/dev/null)
        
        if [ "$MSG_TYPE" = "user" ]; then
            CONTENT=$(echo "$line" | jq -c '.message.content // empty' 2>/dev/null)
            CONTENT_TYPE=""
            if echo "$CONTENT" | jq -e 'type == "array"' >/dev/null 2>&1; then
                CONTENT_TYPE=$(echo "$CONTENT" | jq -r '.[0].type // empty' 2>/dev/null)
            fi
            
            if [ "$CONTENT_TYPE" = "tool_result" ]; then
                # Flush pending LLM span
                if [ -n "$CURRENT_OUTPUT" ]; then
                    create_subagent_llm_span "$CURRENT_OUTPUT" "$CURRENT_MODEL" "$CURRENT_PROMPT_TOKENS" "$CURRENT_COMPLETION_TOKENS" "$LLM_START_TIME" "$LLM_END_TIME"
                    CURRENT_OUTPUT=""
                fi
                LLM_START_TIME=$(iso_to_nanos "$TIMESTAMP")
                
                # Process tool results
                TOOL_USE_RESULT=$(echo "$line" | jq -c '.toolUseResult // empty' 2>/dev/null)
                
                while IFS= read -r TOOL_RESULT; do
                    [ -z "$TOOL_RESULT" ] && continue
                    TOOL_USE_ID=$(echo "$TOOL_RESULT" | jq -r '.tool_use_id // empty')
                    
                    if [ -n "$TOOL_USE_RESULT" ] && [ "$TOOL_USE_RESULT" != "null" ]; then
                        TOOL_OUT=$(echo "$TOOL_USE_RESULT" | jq -r '.text // .content // "completed"' 2>/dev/null)
                    else
                        TOOL_OUT=$(echo "$TOOL_RESULT" | jq -r '.content // "result"')
                    fi
                    
                    if [ -n "$TOOL_USE_ID" ]; then
                        PENDING=$(echo "$PENDING_TOOLS" | jq -r ".\"$TOOL_USE_ID\" // empty")
                        if [ -n "$PENDING" ] && [ "$PENDING" != "null" ]; then
                            P_NAME=$(echo "$PENDING" | jq -r '.name')
                            P_INPUT=$(echo "$PENDING" | jq -r '.input')
                            P_START=$(echo "$PENDING" | jq -r '.start')
                            END_NANOS=$(iso_to_nanos "$TIMESTAMP")
                            create_subagent_tool_span "$P_NAME" "$P_INPUT" "$TOOL_OUT" "$P_START" "$END_NANOS"
                            PENDING_TOOLS=$(echo "$PENDING_TOOLS" | jq "del(.\"$TOOL_USE_ID\")")
                        fi
                    fi
                done < <(echo "$CONTENT" | jq -c '.[]' 2>/dev/null)
                
                CURRENT_MODEL=""; CURRENT_PROMPT_TOKENS=0; CURRENT_COMPLETION_TOKENS=0
            else
                # Regular user message - flush pending LLM span
                if [ -n "$CURRENT_OUTPUT" ]; then
                    create_subagent_llm_span "$CURRENT_OUTPUT" "$CURRENT_MODEL" "$CURRENT_PROMPT_TOKENS" "$CURRENT_COMPLETION_TOKENS" "$LLM_START_TIME" "$LLM_END_TIME"
                fi
                LLM_START_TIME=$(iso_to_nanos "$TIMESTAMP")
                CURRENT_OUTPUT=""; CURRENT_MODEL=""; CURRENT_PROMPT_TOKENS=0; CURRENT_COMPLETION_TOKENS=0
            fi
            
        elif [ "$MSG_TYPE" = "assistant" ]; then
            LLM_END_TIME=$(iso_to_nanos "$TIMESTAMP")
            
            # Track tool_use blocks
            if echo "$line" | jq -e '.message.content | type == "array"' >/dev/null 2>&1; then
                while IFS= read -r TOOL_USE; do
                    [ -z "$TOOL_USE" ] && continue
                    TOOL_ID=$(echo "$TOOL_USE" | jq -r '.id // empty')
                    TOOL_NAME=$(echo "$TOOL_USE" | jq -r '.name // empty')
                    TOOL_INPUT=$(echo "$TOOL_USE" | jq -c '.input // {}')
                    if [ -n "$TOOL_ID" ] && [ -n "$TOOL_NAME" ]; then
                        PENDING_TOOLS=$(echo "$PENDING_TOOLS" | jq --arg id "$TOOL_ID" --arg name "$TOOL_NAME" --arg input "$TOOL_INPUT" --arg start "$(iso_to_nanos "$TIMESTAMP")" \
                            '.[$id] = {name: $name, input: $input, start: $start}')
                    fi
                done < <(echo "$line" | jq -c '.message.content[] | select(.type == "tool_use")' 2>/dev/null)
            fi
            
            # Extract text
            TEXT=$(echo "$line" | jq -r '.message.content | if type == "array" then [.[] | select(.type == "text") | .text] | join("\n") else . end' 2>/dev/null)
            [ -n "$TEXT" ] && CURRENT_OUTPUT="${CURRENT_OUTPUT:+$CURRENT_OUTPUT$'\n'}$TEXT"
            
            # Extract model and usage
            MODEL=$(echo "$line" | jq -r '.message.model // .model // empty' 2>/dev/null)
            [ -n "$MODEL" ] && CURRENT_MODEL="$MODEL"
            
            USAGE=$(echo "$line" | jq -c '.message.usage // .usage // {}' 2>/dev/null)
            if [ -n "$USAGE" ] && [ "$USAGE" != "{}" ]; then
                INP=$(echo "$USAGE" | jq -r '.input_tokens // 0')
                [ "$INP" != "null" ] && [ "$INP" -gt 0 ] 2>/dev/null && CURRENT_PROMPT_TOKENS=$((CURRENT_PROMPT_TOKENS + INP))
                OUT=$(echo "$USAGE" | jq -r '.output_tokens // 0')
                [ "$OUT" != "null" ] && [ "$OUT" -gt 0 ] 2>/dev/null && CURRENT_COMPLETION_TOKENS=$((CURRENT_COMPLETION_TOKENS + OUT))
            fi
            
            # Save last output for subagent summary
            SUBAGENT_OUTPUT="$CURRENT_OUTPUT"
        fi
    done < "$SUBAGENT_TRANSCRIPT"
    
    # Flush final LLM span
    [ -n "$CURRENT_OUTPUT" ] && create_subagent_llm_span "$CURRENT_OUTPUT" "$CURRENT_MODEL" "$CURRENT_PROMPT_TOKENS" "$CURRENT_COMPLETION_TOKENS" "$LLM_START_TIME" "$LLM_END_TIME"
fi

# Create the subagent container span
END_TIME=$(get_time_nanos)
TASK_INPUT_JSON=$(echo "${TASK_DESCRIPTION:-Subagent task}" | jq -Rs '.')
SUBAGENT_OUTPUT_JSON=$(echo "${SUBAGENT_OUTPUT:-Completed}" | jq -Rs '.')

# Build attributes with logical hierarchy info
SUBAGENT_ATTRS=$(build_otlp_attributes "$(jq -n \
    --arg span_kind "task" \
    --argjson input "$TASK_INPUT_JSON" \
    --argjson output "$SUBAGENT_OUTPUT_JSON" \
    --arg subagent_id "${SUBAGENT_ID:-unknown}" \
    --arg parent_agent_id "${PARENT_AGENT_ID:-}" \
    --argjson nesting_depth "$NESTING_DEPTH" \
    --argjson llm_calls "$LLM_CALLS" \
    --argjson tool_calls "$TOOL_CALLS" \
    '{
        "judgment.span_kind": $span_kind,
        "judgment.input": $input,
        "judgment.output": $output,
        "subagent_id": $subagent_id,
        "parent_agent_id": (if $parent_agent_id == "" then null else $parent_agent_id end),
        "nesting_depth": $nesting_depth,
        "llm_call_count": $llm_calls,
        "tool_count": $tool_calls
    }')")

SUBAGENT_SPAN=$(build_otlp_span "$TRACE_ID" "$SUBAGENT_SPAN_ID" "$PARENT_SPAN_ID" "Subagent: ${SUBAGENT_ID:-task}" "task" "$START_TIME" "$END_TIME" "$SUBAGENT_ATTRS" 0)

if insert_span "$PROJECT_ID" "$SUBAGENT_SPAN"; then
    log "INFO" "Subagent traced: ${SUBAGENT_ID:-unknown} (llm=$LLM_CALLS, tools=$TOOL_CALLS, depth=$NESTING_DEPTH, parent=${PARENT_AGENT_ID:-none})"
else
    log "ERROR" "Failed to create subagent span"
fi

exit 0
