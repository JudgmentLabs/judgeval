#!/bin/bash
###
# SessionEnd Hook - Creates LLM/Tool spans and finalizes trace
###

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

debug "SessionEnd hook triggered"
tracing_enabled || { debug "Tracing disabled"; exit 0; }
check_requirements || exit 0

INPUT=$(cat)
debug "SessionEnd input: $(echo "$INPUT" | jq -c '.' 2>/dev/null | head -c 500)"

echo "$INPUT" | jq -e '.' >/dev/null 2>&1 || { debug "Invalid JSON"; exit 0; }

SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty' 2>/dev/null)

TRACE_ID=$(get_state_value "current_trace_id")
[ -z "$TRACE_ID" ] && { debug "No current trace"; exit 0; }

ROOT_SPAN_ID=$(get_session_state "$TRACE_ID" "root_span_id")
PROJECT_ID=$(get_session_state "$TRACE_ID" "project_id")
TASK_SPAN_ID=$(get_session_state "$TRACE_ID" "current_task_span_id")
TASK_START=$(get_session_state "$TRACE_ID" "current_task_start")
SESSION_START=$(get_session_state "$TRACE_ID" "started")

[ -z "$ROOT_SPAN_ID" ] || [ -z "$PROJECT_ID" ] && { debug "No trace/project"; exit 0; }

CONV_FILE=$(echo "$INPUT" | jq -r '.transcript_path // empty' 2>/dev/null)
[ -z "$CONV_FILE" ] || [ ! -f "$CONV_FILE" ] && CONV_FILE=$(find "$HOME/.claude/projects" -name "${SESSION_ID}.jsonl" -type f 2>/dev/null | head -1)

WORKSPACE=$(echo "$INPUT" | jq -r '.cwd // empty' 2>/dev/null)
WORKSPACE_NAME=$(basename "$WORKSPACE" 2>/dev/null || echo "Claude Code")

if [ -n "$TASK_SPAN_ID" ] && [ -f "$CONV_FILE" ]; then
    debug "Processing transcript"
    
    # Transcript parsing state
    LLM_CALLS=0
    TOOL_CALLS=0
    CURRENT_OUTPUT=""
    CURRENT_MODEL=""
    CURRENT_PROMPT_TOKENS=0
    CURRENT_COMPLETION_TOKENS=0
    CURRENT_CACHE_CREATION=0
    CURRENT_CACHE_READ=0
    LLM_START_TIME=""
    LLM_END_TIME=""
    CONVERSATION_HISTORY="[]"
    PENDING_TOOLS="{}"
    TASK_INPUT=""

    create_llm_span() {
        local output="$1" model="$2" prompt="$3" completion="$4" history="$5"
        local cache_create="${6:-0}" cache_read="${7:-0}" start_time="$8" end_time="$9"
        [ -z "$output" ] && return
        local span_id span_start span_end input_json output_json attrs span duration_ms provider
        span_id=$(generate_uuid | sed 's/-//g' | head -c 16)
        span_start="${start_time:-$(get_time_nanos)}"
        span_end="${end_time:-$(get_time_nanos)}"
        provider=$(detect_provider "$model")
        input_json=$(echo "$history" | jq -c '.' | jq -Rs '.')
        output_json=$(jq -n --arg c "$output" '[{role: "assistant", content: $c}]' | jq -c '.' | jq -Rs '.')
        local usage_meta
        usage_meta=$(jq -n --argjson inp "$prompt" --argjson out "$completion" \
            --argjson cc "$cache_create" --argjson cr "$cache_read" \
            '{input_tokens: $inp, output_tokens: $out, cache_creation_input_tokens: $cc, cache_read_input_tokens: $cr}' | jq -c '.')
        attrs=$(build_otlp_attributes "$(jq -n \
            --arg span_kind "llm" --argjson input "$input_json" --argjson output "$output_json" \
            --arg model "${model:-claude}" --arg provider "$provider" \
            --argjson prompt "$prompt" --argjson completion "$completion" \
            --argjson cache_create "$cache_create" --argjson cache_read "$cache_read" \
            --arg usage_meta "$usage_meta" \
            '{
              "judgment.span_kind": $span_kind,
              "judgment.input": $input,
              "judgment.output": $output,
              "judgment.llm.provider": $provider,
              "judgment.llm.model": $model,
              "judgment.usage.non_cached_input_tokens": $prompt,
              "judgment.usage.output_tokens": $completion,
              "judgment.usage.cache_creation_input_tokens": $cache_create,
              "judgment.usage.cache_read_input_tokens": $cache_read,
              "judgment.usage.reasoning_tokens": 0,
              "judgment.usage.metadata": $usage_meta
            }')")
        span=$(build_otlp_span "$TRACE_ID" "$span_id" "$TASK_SPAN_ID" "${model:-anthropic.messages.create}" "llm" "$span_start" "$span_end" "$attrs" 20)
        duration_ms=$(( (span_end - span_start) / 1000000 ))
        if insert_span "$PROJECT_ID" "$span" >/dev/null; then
            LLM_CALLS=$((LLM_CALLS + 1))
            debug "LLM span: $model (${duration_ms}ms) tokens: in=$prompt out=$completion cache_create=$cache_create cache_read=$cache_read"
        fi
    }

    create_tool_span() {
        local tool_name="$1" tool_input="$2" tool_output="$3" start_time="$4" end_time="$5"
        [ -z "$tool_name" ] && return
        local span_id input_json output_json attrs span
        span_id=$(generate_uuid | sed 's/-//g' | head -c 16)
        input_json=$(echo "$tool_input" | jq -c '.' 2>/dev/null | jq -Rs '.')
        output_json=$(echo "$tool_output" | jq -Rs '.')
        attrs=$(build_otlp_attributes "$(jq -n --arg span_kind "tool" --argjson input "$input_json" --argjson output "$output_json" --arg tool_name "$tool_name" \
            '{"judgment.span_kind": $span_kind, "judgment.input": $input, "judgment.output": $output, "tool_name": $tool_name}')")
        span=$(build_otlp_span "$TRACE_ID" "$span_id" "$TASK_SPAN_ID" "$tool_name" "tool" "$start_time" "$end_time" "$attrs" 20)
        if insert_span "$PROJECT_ID" "$span" >/dev/null; then
            TOOL_CALLS=$((TOOL_CALLS + 1))
            log "INFO" "Tool span: $tool_name ($(( (end_time - start_time) / 1000000 ))ms)"
        fi
    }

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
                if [ -n "$CURRENT_OUTPUT" ]; then
                    create_llm_span "$CURRENT_OUTPUT" "$CURRENT_MODEL" "$CURRENT_PROMPT_TOKENS" "$CURRENT_COMPLETION_TOKENS" "$CONVERSATION_HISTORY" "$CURRENT_CACHE_CREATION" "$CURRENT_CACHE_READ" "$LLM_START_TIME" "$LLM_END_TIME"
                    CONVERSATION_HISTORY=$(echo "$CONVERSATION_HISTORY" | jq --arg c "$CURRENT_OUTPUT" '. += [{role: "assistant", content: $c}]')
                    CURRENT_OUTPUT=""
                fi
                LLM_START_TIME=$(iso_to_nanos "$TIMESTAMP")
                
                TOOL_USE_RESULT=$(echo "$line" | jq -c '.toolUseResult // empty' 2>/dev/null)
                
                while IFS= read -r TOOL_RESULT; do
                    [ -z "$TOOL_RESULT" ] && continue
                    TOOL_USE_ID=$(echo "$TOOL_RESULT" | jq -r '.tool_use_id // empty')
                    
                    if [ -n "$TOOL_USE_RESULT" ] && [ "$TOOL_USE_RESULT" != "null" ] && echo "$TOOL_USE_RESULT" | jq -e 'type == "object"' >/dev/null 2>&1; then
                        TOOL_TYPE=$(echo "$TOOL_USE_RESULT" | jq -r '.type // empty')
                        if [ "$TOOL_TYPE" = "text" ]; then
                            FILE_CONTENT=$(echo "$TOOL_USE_RESULT" | jq -r '.file.content // empty')
                            if [ -n "$FILE_CONTENT" ]; then
                                TOOL_OUT="$FILE_CONTENT"
                            else
                                TOOL_OUT=$(echo "$TOOL_USE_RESULT" | jq -r '.text // "completed"')
                            fi
                        else
                            TOOL_OUT=$(echo "$TOOL_USE_RESULT" | jq -c '.')
                        fi
                    elif [ -n "$TOOL_USE_RESULT" ] && [ "$TOOL_USE_RESULT" != "null" ]; then
                        TOOL_OUT=$(echo "$TOOL_USE_RESULT" | jq -r '.')
                    else
                        RAW_OUT=$(echo "$TOOL_RESULT" | jq -r '.content // "result"')
                        TOOL_OUT="${RAW_OUT#*→}"
                    fi
                    
                    if [ -n "$TOOL_USE_ID" ]; then
                        PENDING=$(echo "$PENDING_TOOLS" | jq -r ".\"$TOOL_USE_ID\" // empty")
                        if [ -n "$PENDING" ] && [ "$PENDING" != "null" ]; then
                            P_NAME=$(echo "$PENDING" | jq -r '.name')
                            P_INPUT=$(echo "$PENDING" | jq -r '.input')
                            P_START=$(echo "$PENDING" | jq -r '.start')
                            if [ -n "$P_START" ] && [ "$P_START" != "null" ] && [ "$P_START" -gt 0 ] 2>/dev/null; then
                                END_NANOS=$(iso_to_nanos "$TIMESTAMP")
                                if [ -n "$END_NANOS" ] && [ "$END_NANOS" -gt 0 ] 2>/dev/null; then
                                    create_tool_span "$P_NAME" "$P_INPUT" "$TOOL_OUT" "$P_START" "$END_NANOS"
                                fi
                            fi
                            PENDING_TOOLS=$(echo "$PENDING_TOOLS" | jq "del(.\"$TOOL_USE_ID\")")
                        fi
                    fi
                done < <(echo "$CONTENT" | jq -c '.[]' 2>/dev/null)
                CURRENT_MODEL=""; CURRENT_PROMPT_TOKENS=0; CURRENT_COMPLETION_TOKENS=0; CURRENT_CACHE_CREATION=0; CURRENT_CACHE_READ=0
            else
                if [ -n "$CURRENT_OUTPUT" ]; then
                    create_llm_span "$CURRENT_OUTPUT" "$CURRENT_MODEL" "$CURRENT_PROMPT_TOKENS" "$CURRENT_COMPLETION_TOKENS" "$CONVERSATION_HISTORY" "$CURRENT_CACHE_CREATION" "$CURRENT_CACHE_READ" "$LLM_START_TIME" "$LLM_END_TIME"
                    CONVERSATION_HISTORY=$(echo "$CONVERSATION_HISTORY" | jq --arg c "$CURRENT_OUTPUT" '. += [{role: "assistant", content: $c}]')
                fi
                if [ "$CONTENT" != "null" ] && [ -n "$CONTENT" ]; then
                    TXT="$CONTENT"
                    if echo "$CONTENT" | jq -e '.' >/dev/null 2>&1; then
                        TXT=$(echo "$CONTENT" | jq -r 'if type == "array" then [.[] | select(.type == "text") | .text] | join("\n") else . end' 2>/dev/null)
                    fi
                    [ -n "$TXT" ] && CONVERSATION_HISTORY=$(echo "$CONVERSATION_HISTORY" | jq --arg c "$TXT" '. += [{role: "user", content: $c}]')
                    [ -z "$TASK_INPUT" ] && [ -n "$TXT" ] && TASK_INPUT="$TXT"
                fi
                LLM_START_TIME=$(iso_to_nanos "$TIMESTAMP")
                CURRENT_OUTPUT=""; CURRENT_MODEL=""; CURRENT_PROMPT_TOKENS=0; CURRENT_COMPLETION_TOKENS=0; CURRENT_CACHE_CREATION=0; CURRENT_CACHE_READ=0
            fi
        elif [ "$MSG_TYPE" = "assistant" ]; then
            LLM_END_TIME=$(iso_to_nanos "$TIMESTAMP")
            
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
            TEXT=$(echo "$line" | jq -r '.message.content | if type == "array" then [.[] | select(.type == "text") | .text] | join("\n") else . end' 2>/dev/null)
            [ -n "$TEXT" ] && CURRENT_OUTPUT="${CURRENT_OUTPUT:+$CURRENT_OUTPUT$'\n'}$TEXT"
            MODEL=$(echo "$line" | jq -r '.message.model // .model // empty' 2>/dev/null)
            [ -n "$MODEL" ] && CURRENT_MODEL="$MODEL"
            USAGE=$(echo "$line" | jq -c '.message.usage // .usage // {input_tokens: .input_tokens, output_tokens: .output_tokens, cache_creation_input_tokens: .cache_creation_input_tokens, cache_read_input_tokens: .cache_read_input_tokens} | select(. != null)' 2>/dev/null)
            if [ -n "$USAGE" ] && [ "$USAGE" != "{}" ] && [ "$USAGE" != "null" ]; then
                INP=$(echo "$USAGE" | jq -r '.input_tokens // 0')
                [ "$INP" != "null" ] && [ "$INP" -gt 0 ] 2>/dev/null && CURRENT_PROMPT_TOKENS=$((CURRENT_PROMPT_TOKENS + INP))
                OUT=$(echo "$USAGE" | jq -r '.output_tokens // 0')
                [ "$OUT" != "null" ] && [ "$OUT" -gt 0 ] 2>/dev/null && CURRENT_COMPLETION_TOKENS=$((CURRENT_COMPLETION_TOKENS + OUT))
                CC=$(echo "$USAGE" | jq -r '.cache_creation_input_tokens // 0')
                [ "$CC" != "null" ] && [ "$CC" -gt 0 ] 2>/dev/null && CURRENT_CACHE_CREATION=$((CURRENT_CACHE_CREATION + CC))
                CR=$(echo "$USAGE" | jq -r '.cache_read_input_tokens // 0')
                [ "$CR" != "null" ] && [ "$CR" -gt 0 ] 2>/dev/null && CURRENT_CACHE_READ=$((CURRENT_CACHE_READ + CR))
            fi
        fi
    done < "$CONV_FILE"

    [ -n "$CURRENT_OUTPUT" ] && create_llm_span "$CURRENT_OUTPUT" "$CURRENT_MODEL" "$CURRENT_PROMPT_TOKENS" "$CURRENT_COMPLETION_TOKENS" "$CONVERSATION_HISTORY" "$CURRENT_CACHE_CREATION" "$CURRENT_CACHE_READ" "$LLM_START_TIME" "$LLM_END_TIME"

    TASK_END=$(get_time_nanos)
    TASK_INPUT_JSON=$(echo "${TASK_INPUT:-}" | jq -Rs '.')
    TASK_ATTRS=$(build_otlp_attributes "$(jq -n --arg span_kind "task" --argjson input "$TASK_INPUT_JSON" --arg output "${CURRENT_OUTPUT:-Completed}" --argjson llm "$LLM_CALLS" --argjson tools "$TOOL_CALLS" --arg session_id "$SESSION_ID" \
        '{"judgment.span_kind": $span_kind, "judgment.input": $input, "judgment.output": $output, "llm_call_count": $llm, "tool_count": $tools, "session_id": $session_id}')")
    TASK_SPAN=$(build_otlp_span "$TRACE_ID" "$TASK_SPAN_ID" "$ROOT_SPAN_ID" "Task" "task" "$TASK_START" "$TASK_END" "$TASK_ATTRS" 20)
    insert_span "$PROJECT_ID" "$TASK_SPAN" >/dev/null || debug "Failed to finalize task"

    [ "$LLM_CALLS" -gt 0 ] && log "INFO" "Created $LLM_CALLS LLM spans"
    [ "$TOOL_CALLS" -gt 0 ] && log "INFO" "Created $TOOL_CALLS tool spans"
    log "INFO" "Task finalized"
fi

END_TIME=$(get_time_nanos)
SESSION_START=${SESSION_START:-$END_TIME}
SESSION_ATTRS=$(build_otlp_attributes "$(jq -n --arg span_kind "task" --arg input "Session: $WORKSPACE_NAME" --arg output "Completed" --arg session_id "$SESSION_ID" \
    '{"judgment.span_kind": $span_kind, "judgment.input": $input, "judgment.output": $output, "session_id": $session_id, "source": "claude-code"}')")
SESSION_SPAN=$(build_otlp_span "$TRACE_ID" "$ROOT_SPAN_ID" "" "Claude Code: $WORKSPACE_NAME" "task" "$SESSION_START" "$END_TIME" "$SESSION_ATTRS" 20)
insert_span "$PROJECT_ID" "$SESSION_SPAN" || debug "Failed to finalize session"

set_state_value "current_trace_id" ""

log "INFO" "Trace ended: $TRACE_ID (session=$SESSION_ID)"
exit 0
