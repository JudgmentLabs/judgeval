#!/bin/bash
###
# Stop Hook - Marks turn as ready for finalization
###

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

debug "Stop hook triggered"
tracing_enabled || { debug "Tracing disabled"; exit 0; }

INPUT=$(cat)
debug "Stop input: $(echo "$INPUT" | jq -c '.' 2>/dev/null | head -c 500)"

echo "$INPUT" | jq -e '.' >/dev/null 2>&1 || { debug "Invalid JSON"; exit 0; }

SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty' 2>/dev/null)
if [ -z "$SESSION_ID" ]; then
    TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // empty' 2>/dev/null)
    [ -n "$TRANSCRIPT_PATH" ] && SESSION_ID=$(basename "$TRANSCRIPT_PATH" .jsonl)
fi
[ -z "$SESSION_ID" ] && { debug "No session ID"; exit 0; }

TURN_SPAN_ID=$(get_session_state "$SESSION_ID" "current_turn_span_id")
[ -n "$TURN_SPAN_ID" ] && {
    set_session_state "$SESSION_ID" "turn_stopped" "true"
    log "INFO" "Turn marked for finalization"
}

exit 0
