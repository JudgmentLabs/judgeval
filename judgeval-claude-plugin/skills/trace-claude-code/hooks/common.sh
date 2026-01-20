#!/bin/bash
# Common utilities for Judgeval Claude Code tracing hooks

# Configuration
export LOG_FILE="$HOME/.claude/state/judgeval_hook.log"
export STATE_FILE="$HOME/.claude/state/judgeval_state.json"
export LOCK_DIR="$HOME/.claude/state/judgeval.lock.d"
export DEBUG="${JUDGEVAL_CC_DEBUG:-false}"
export API_KEY="${JUDGMENT_API_KEY}"
export ORG_ID="${JUDGMENT_ORG_ID}"
export PROJECT="${JUDGEVAL_CC_PROJECT:-claude-code}"
export API_URL="${JUDGMENT_API_URL:-https://api.judgmentlabs.ai}"

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$STATE_FILE")"

# Logging
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [$1] $2" >> "$LOG_FILE"; }

debug() {
    if [ "$(echo "$DEBUG" | tr '[:upper:]' '[:lower:]')" = "true" ]; then
        log "DEBUG" "$1"
    fi
}

tracing_enabled() {
    [ "$(echo "$TRACE_TO_JUDGEVAL" | tr '[:upper:]' '[:lower:]')" = "true" ]
}

check_requirements() {
    for cmd in jq curl; do
        if ! command -v "$cmd" &>/dev/null; then
            log "ERROR" "$cmd not installed"
            return 1
        fi
    done
    if [ -z "$API_KEY" ]; then
        log "ERROR" "JUDGMENT_API_KEY not set"
        return 1
    fi
    if [ -z "$ORG_ID" ]; then
        log "ERROR" "JUDGMENT_ORG_ID not set"
        return 1
    fi
    return 0
}

# File Locking (mkdir-based, atomic on POSIX)
acquire_lock() {
    local timeout="${1:-5}"
    local count=0
    local max_attempts=$((timeout * 20))
    local lock_age
    
    while ! mkdir "$LOCK_DIR" 2>/dev/null; do
        sleep 0.05
        count=$((count + 1))
        if [ "$count" -ge "$max_attempts" ]; then
            if [ -d "$LOCK_DIR" ]; then
                lock_age=$(( $(date +%s) - $(stat -f %m "$LOCK_DIR" 2>/dev/null || stat -c %Y "$LOCK_DIR" 2>/dev/null || echo 0) ))
                if [ "$lock_age" -gt 30 ]; then
                    rmdir "$LOCK_DIR" 2>/dev/null || true
                    continue
                fi
            fi
            log "WARN" "Lock timeout after ${timeout}s"
            return 1
        fi
    done
    echo "$$" > "$LOCK_DIR/pid" 2>/dev/null || true
    return 0
}

release_lock() {
    rm -f "$LOCK_DIR/pid" 2>/dev/null || true
    rmdir "$LOCK_DIR" 2>/dev/null || true
}

with_lock() {
    if ! acquire_lock 5; then
        return 1
    fi
    local ret=0
    "$@" || ret=$?
    release_lock
    return $ret
}

# State Management
load_state() {
    if [ -f "$STATE_FILE" ]; then
        cat "$STATE_FILE" 2>/dev/null
    else
        echo "{}"
    fi
}

save_state() {
    local tmp_file="${STATE_FILE}.tmp.$$"
    echo "$1" > "$tmp_file"
    mv -f "$tmp_file" "$STATE_FILE"
}

get_state_value() {
    load_state | jq -r ".$1 // empty"
}

set_state_value() {
    with_lock _set_state_value_unsafe "$1" "$2"
}

_set_state_value_unsafe() {
    local state
    state=$(load_state)
    save_state "$(echo "$state" | jq --arg k "$1" --arg v "$2" '.[$k] = $v')"
}

get_session_state() {
    load_state | jq -r ".sessions[\"$1\"].$2 // empty"
}

set_session_state() {
    with_lock _set_session_state_unsafe "$1" "$2" "$3"
}

_set_session_state_unsafe() {
    local state
    state=$(load_state)
    save_state "$(echo "$state" | jq --arg s "$1" --arg k "$2" --arg v "$3" \
        '.sessions[$s] = (.sessions[$s] // {}) | .sessions[$s][$k] = $v')"
}

set_session_state_batch() {
    local session_id="$1"
    shift
    with_lock _set_session_state_batch_unsafe "$session_id" "$@"
}

_set_session_state_batch_unsafe() {
    local session_id="$1"
    shift
    local state key val
    state=$(load_state)
    while [ $# -ge 2 ]; do
        key="$1"
        val="$2"
        state=$(echo "$state" | jq --arg s "$session_id" --arg k "$key" --arg v "$val" \
            '.sessions[$s] = (.sessions[$s] // {}) | .sessions[$s][$k] = $v')
        shift 2
    done
    save_state "$state"
}

# Subagent Registry (for tracking agent relationships)
# Maps agent_id -> span_id so we can look up parents and build hierarchy

register_subagent() {
    local trace_id="$1" agent_id="$2" span_id="$3" parent_agent_id="$4" transcript_path="$5"
    with_lock _register_subagent_unsafe "$trace_id" "$agent_id" "$span_id" "$parent_agent_id" "$transcript_path"
}

_register_subagent_unsafe() {
    local trace_id="$1" agent_id="$2" span_id="$3" parent_agent_id="$4" transcript_path="$5"
    local state
    state=$(load_state)
    # Store agent info in a map: agents[agent_id] = {span_id, parent_agent_id, transcript_path, timestamp}
    save_state "$(echo "$state" | jq --arg s "$trace_id" --arg aid "$agent_id" --arg sid "$span_id" --arg pid "${parent_agent_id:-}" --arg tp "${transcript_path:-}" \
        '.sessions[$s].agents[$aid] = {span_id: $sid, parent_agent_id: (if $pid == "" then null else $pid end), transcript_path: (if $tp == "" then null else $tp end), timestamp: (now | tostring)}')"
}

get_subagent_span_id() {
    # Get the span_id for a given agent_id
    local trace_id="$1" agent_id="$2"
    load_state | jq -r ".sessions[\"$trace_id\"].agents[\"$agent_id\"].span_id // empty"
}

get_subagent_count() {
    local trace_id="$1"
    load_state | jq -r ".sessions[\"$trace_id\"].agents | if . == null then 0 else keys | length end"
}

# Get the parent transcript path for a spawned agent
# This is set by PostToolUse when a Task tool creates an agent
get_agent_parent_transcript() {
    local trace_id="$1" agent_id="$2"
    # Use jq with proper quoting for keys containing hyphens
    load_state | jq -r --arg t "$trace_id" --arg k "parent_of_${agent_id}" '.sessions[$t][$k] // empty'
}

# Find which agent_id owns a given transcript path
# Returns empty if it's the main session transcript
find_agent_by_transcript() {
    local trace_id="$1" transcript_path="$2"
    # Check if any registered agent has this transcript
    load_state | jq -r --arg tp "$transcript_path" "
        .sessions[\"$trace_id\"].agents 
        | to_entries 
        | map(select(.value.transcript_path == \$tp)) 
        | .[0].key // empty
    "
}

# API Operations
_build_otlp_payload() {
    local span_json="$1"
    jq -n --arg service_name "$PROJECT" --argjson span "$span_json" '{
        resourceSpans: [{
            resource: { attributes: [
                { key: "service.name", value: { stringValue: $service_name } },
                { key: "telemetry.sdk.name", value: { stringValue: "judgeval" } },
                { key: "telemetry.sdk.version", value: { stringValue: "1.0.0" } }
            ]},
            scopeSpans: [{ scope: { name: "judgeval" }, spans: [$span] }]
        }]
    }'
}

insert_span() {
    local project_id="$1" span_json="$2"
    local otlp_payload resp http_code
    
    debug "Inserting span: $(echo "$span_json" | jq -c '.name' 2>/dev/null)"
    
    otlp_payload=$(_build_otlp_payload "$span_json")
    
    resp=$(curl -s -w "\n%{http_code}" \
        --max-time 5 \
        --connect-timeout 3 \
        -X POST \
        -H "Authorization: Bearer $API_KEY" \
        -H "X-Organization-Id: $ORG_ID" \
        -H "X-Project-Id: $project_id" \
        -H "Content-Type: application/json" \
        -d "$otlp_payload" \
        "$API_URL/otel/v1/traces" 2>&1)
    
    http_code=$(echo "$resp" | tail -1)
    
    if [[ "$http_code" =~ ^20[012]$ ]]; then
        debug "OTLP insert successful (HTTP $http_code)"
        echo "success"
        return 0
    fi
    
    log "WARN" "OTLP insert failed (HTTP $http_code)"
    return 1
}

# Alias for backward compatibility
insert_span_sync() {
    insert_span "$@"
}

# Project Resolution
get_project_id() {
    local name="$1"
    local cached_id
    cached_id=$(get_state_value "project_id")
    if [ -n "$cached_id" ]; then
        echo "$cached_id"
        return 0
    fi

    debug "Resolving project: $name"
    local resp pid

    resp=$(curl -sf -X POST \
        -H "Authorization: Bearer $API_KEY" \
        -H "X-Organization-Id: $ORG_ID" \
        -H "Content-Type: application/json" \
        -d "{\"project_name\": \"$name\"}" \
        "$API_URL/projects/resolve/" 2>/dev/null) || true

    pid=$(echo "$resp" | jq -r '.project_id // empty' 2>/dev/null)
    if [ -n "$pid" ]; then
        set_state_value "project_id" "$pid"
        echo "$pid"
        return 0
    fi

    debug "Creating project: $name"
    resp=$(curl -sf -X POST \
        -H "Authorization: Bearer $API_KEY" \
        -H "X-Organization-Id: $ORG_ID" \
        -H "Content-Type: application/json" \
        -d "{\"project_name\": \"$name\"}" \
        "$API_URL/projects/add/" 2>/dev/null) || true

    pid=$(echo "$resp" | jq -r '.project_id // empty' 2>/dev/null)
    if [ -n "$pid" ]; then
        set_state_value "project_id" "$pid"
        echo "$pid"
        return 0
    fi

    log "ERROR" "Failed to get or create project: $name"
    return 1
}

# Time Utilities - Pure bash, no Python needed
get_time_nanos() {
    # Linux/Docker: date +%s%N works natively
    # macOS: %N not supported, fallback to seconds
    local nanos
    nanos=$(date +%s%N 2>/dev/null)
    if [[ "$nanos" == *N ]]; then
        # macOS - %N literal, use seconds * 1e9
        echo "$(($(date +%s) * 1000000000))"
    else
        echo "$nanos"
    fi
}

iso_to_nanos() {
    local ts="$1"
    [ -z "$ts" ] && { get_time_nanos; return; }
    
    # Try GNU date first (Linux / Docker)
    local epoch
    epoch=$(date -d "$ts" +%s%N 2>/dev/null)
    if [ -n "$epoch" ] && [[ "$epoch" != *N ]]; then
        echo "$epoch"
        return
    fi
    
    # macOS with BSD date - strip milliseconds and timezone
    local clean_ts secs
    clean_ts="${ts%%.*}"  # Remove .123Z suffix
    clean_ts="${clean_ts//T/ }"  # Replace T with space
    secs=$(date -j -f "%Y-%m-%d %H:%M:%S" "$clean_ts" +%s 2>/dev/null)
    if [ -n "$secs" ]; then
        echo "$((secs * 1000000000))"
        return
    fi
    
    # Last fallback
    get_time_nanos
}

detect_provider() {
    local model="$1"
    case "$model" in
        anthropic/*|claude-*) echo "anthropic" ;;
        openai/*|gpt-*) echo "openai" ;;
        google/*|gemini-*) echo "google" ;;
        meta-llama/*|llama-*) echo "meta" ;;
        */*) echo "openrouter" ;;
        *) echo "anthropic" ;;
    esac
}

# Span Building
build_otlp_span() {
    local trace_id="$1" span_id="$2" parent_span_id="$3" name="$4"
    local start_time="$6" end_time="$7" attributes_json="$8" update_id="${9:-0}"

    local attrs_with_update
    attrs_with_update=$(echo "$attributes_json" | jq --argjson uid "$update_id" \
        '. + [{"key": "judgment.update_id", "value": {"intValue": ($uid | tostring)}}]')

    jq -n \
        --arg trace_id "$trace_id" \
        --arg span_id "$span_id" \
        --arg parent_span_id "$parent_span_id" \
        --arg name "$name" \
        --arg start_time "$start_time" \
        --arg end_time "$end_time" \
        --argjson attributes "$attrs_with_update" \
        '{
            traceId: $trace_id,
            spanId: $span_id,
            parentSpanId: (if $parent_span_id == "" then null else $parent_span_id end),
            name: $name,
            kind: 1,
            startTimeUnixNano: $start_time,
            endTimeUnixNano: $end_time,
            attributes: $attributes,
            status: { code: 1 }
        } | with_entries(select(.value != null))'
}

build_otlp_attributes() {
    local kv_json="$1"
    echo "$kv_json" | jq '
        to_entries | map({
            key: .key,
            value: (
                if (.value | type) == "string" then { stringValue: .value }
                elif (.value | type) == "number" then
                    if (.value | floor) == .value then { intValue: (.value | tostring) }
                    else { doubleValue: .value }
                    end
                elif (.value | type) == "boolean" then { boolValue: .value }
                else { stringValue: (.value | tostring) }
                end
            )
        })
    '
}

# Utilities
generate_uuid() { uuidgen | tr '[:upper:]' '[:lower:]'; }
get_hostname() { hostname 2>/dev/null || echo "unknown"; }
get_username() { whoami 2>/dev/null || echo "unknown"; }
get_os() { uname -s 2>/dev/null || echo "unknown"; }
