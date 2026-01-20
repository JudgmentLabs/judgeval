#!/bin/bash
###
# Setup script for Judgeval Claude Code tracing
# Run this in any project directory to enable comprehensive tracing
###

set -e

echo "🧠 Judgeval Claude Code Tracing Setup"
echo "========================================"
echo ""
echo "This script will configure Claude Code to trace conversations to Judgeval."
echo "Traces include: sessions, conversation turns, and tool calls."
echo ""

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOKS_DIR="$SCRIPT_DIR/hooks"

# Verify hooks exist
for hook in common.sh session_start.sh post_tool_use.sh stop_hook.sh session_end.sh user_prompt_submit.sh subagent_stop.sh; do
    if [ ! -f "$HOOKS_DIR/$hook" ]; then
        echo "❌ Error: Missing hook script: $HOOKS_DIR/$hook"
        exit 1
    fi
done

# Check for required tools
for cmd in jq curl; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "❌ Error: $cmd is required but not installed"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "   Install with: brew install $cmd"
        else
            echo "   Install with: sudo apt-get install $cmd"
        fi
        exit 1
    fi
done

# Load API key from .env files (check current dir and parents)
load_env() {
    local dir="$PWD"
    while [ "$dir" != "/" ]; do
        if [ -f "$dir/.env" ]; then
            # Source the .env file safely (only export lines)
            while IFS= read -r line || [ -n "$line" ]; do
                # Skip comments and empty lines
                [[ "$line" =~ ^#.*$ ]] && continue
                [[ -z "$line" ]] && continue
                # Export valid variable assignments
                if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
                    export "${line?}"
                fi
            done < "$dir/.env"
            echo "  Found .env at: $dir/.env"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

# Try to load from .env
EXISTING_API_KEY=""
EXISTING_ORG_ID=""
if load_env 2>/dev/null; then
    EXISTING_API_KEY="${JUDGMENT_API_KEY:-}"
    EXISTING_ORG_ID="${JUDGMENT_ORG_ID:-}"
fi

# Prompt for API key (with default from .env if available)
if [ -n "$EXISTING_API_KEY" ]; then
    echo "Found JUDGMENT_API_KEY in .env"
    echo "Press Enter to use it, or enter a different key:"
    read -r -p "> " INPUT_KEY
    JUDGMENT_API_KEY="${INPUT_KEY:-$EXISTING_API_KEY}"
else
    echo "Enter your Judgment API key:"
    echo "  Get one at: https://app.judgmentlabs.ai/settings/api-keys"
    read -r -p "> " JUDGMENT_API_KEY
fi

if [ -z "$JUDGMENT_API_KEY" ]; then
    echo "❌ API key is required"
    exit 1
fi

# Prompt for Organization ID (with default from .env if available)
echo ""
if [ -n "$EXISTING_ORG_ID" ]; then
    echo "Found JUDGMENT_ORG_ID in .env"
    echo "Press Enter to use it, or enter a different organization ID:"
    read -r -p "> " INPUT_ORG_ID
    JUDGMENT_ORG_ID="${INPUT_ORG_ID:-$EXISTING_ORG_ID}"
else
    echo "Enter your Judgment Organization ID:"
    echo "  Find it at: https://app.judgmentlabs.ai/settings/organization"
    read -r -p "> " JUDGMENT_ORG_ID
fi

if [ -z "$JUDGMENT_ORG_ID" ]; then
    echo "❌ Organization ID is required"
    exit 1
fi

# Prompt for project name
echo ""
echo "Enter the Judgeval project name for traces (default: claude-code):"
read -r -p "> " PROJECT_NAME
PROJECT_NAME="${PROJECT_NAME:-claude-code}"

# Prompt for API URL
echo ""
echo "Enter the Judgment API URL (default: https://api.judgmentlabs.ai):"
echo "  Use https://staging.api.judgmentlabs.ai for staging"
read -r -p "> " API_URL_INPUT
JUDGMENT_API_URL="${API_URL_INPUT:-https://api.judgmentlabs.ai}"

# Prompt for debug mode
echo ""
echo "Enable debug logging? (y/N):"
read -r -p "> " ENABLE_DEBUG
if [[ "$ENABLE_DEBUG" =~ ^[Yy] ]]; then
    DEBUG_VALUE="true"
else
    DEBUG_VALUE="false"
fi

# Create .claude directory if needed
mkdir -p .claude

# Build the hooks configuration
HOOKS_CONFIG=$(cat <<EOF
{
    "SessionStart": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": "bash $HOOKS_DIR/session_start.sh"
                }
            ]
        }
    ],
    "UserPromptSubmit": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": "bash $HOOKS_DIR/user_prompt_submit.sh"
                }
            ]
        }
    ],
    "PostToolUse": [
        {
            "matcher": "*",
            "hooks": [
                {
                    "type": "command",
                    "command": "bash $HOOKS_DIR/post_tool_use.sh"
                }
            ]
        }
    ],
    "Stop": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": "bash $HOOKS_DIR/stop_hook.sh"
                }
            ]
        }
    ],
    "SessionEnd": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": "bash $HOOKS_DIR/session_end.sh"
                }
            ]
        }
    ],
    "SubagentStop": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": "bash $HOOKS_DIR/subagent_stop.sh"
                }
            ]
        }
    ]
}
EOF
)

# Build environment config
ENV_CONFIG=$(jq -n \
    --arg key "$JUDGMENT_API_KEY" \
    --arg org "$JUDGMENT_ORG_ID" \
    --arg url "$JUDGMENT_API_URL" \
    --arg proj "$PROJECT_NAME" \
    --arg debug "$DEBUG_VALUE" \
    '{
        "TRACE_TO_JUDGEVAL": "true",
        "JUDGMENT_API_KEY": $key,
        "JUDGMENT_ORG_ID": $org,
        "JUDGMENT_API_URL": $url,
        "JUDGEVAL_CC_PROJECT": $proj,
        "JUDGEVAL_CC_DEBUG": $debug
    }')

# Check if settings.local.json exists
SETTINGS_FILE=".claude/settings.local.json"
if [ -f "$SETTINGS_FILE" ]; then
    echo ""
    echo "Found existing $SETTINGS_FILE"

    # Read existing settings and merge
    EXISTING=$(cat "$SETTINGS_FILE")

    UPDATED=$(echo "$EXISTING" | jq \
        --argjson hooks "$HOOKS_CONFIG" \
        --argjson env "$ENV_CONFIG" \
        '.hooks = $hooks | .env = (.env // {}) + $env')

    echo "$UPDATED" > "$SETTINGS_FILE"
else
    # Create new settings file
    jq -n \
        --argjson hooks "$HOOKS_CONFIG" \
        --argjson env "$ENV_CONFIG" \
        '{hooks: $hooks, env: $env}' > "$SETTINGS_FILE"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Configuration saved to: $SETTINGS_FILE"
echo ""
echo "Hooks configured:"
echo "  • SessionStart      - Creates trace root when session begins"
echo "  • UserPromptSubmit  - Creates Turn container for each user message"
echo "  • PostToolUse       - Captures tool calls as children of Turn"
echo "  • Stop              - Creates LLM span and finalizes Turn"
echo "  • SubagentStop      - Traces subagent (Task tool) execution"
echo "  • SessionEnd        - Finalizes trace when session ends"
echo ""
echo "Settings:"
echo "  • API URL: $JUDGMENT_API_URL"
echo "  • Project: $PROJECT_NAME"
echo "  • Debug:   $DEBUG_VALUE"
echo ""
echo "Next steps:"
echo "  1. Start Claude Code in this directory: claude"
echo "  2. Have a conversation"
echo "  3. View traces at: https://app.judgmentlabs.ai/projects/$PROJECT_NAME/traces"
echo ""
echo "To view hook logs:"
echo "  tail -f ~/.claude/state/judgeval_hook.log"
echo ""

# Test API connection
echo "Testing API connection..."

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
    -H "Authorization: Bearer $JUDGMENT_API_KEY" \
    -H "X-Organization-Id: $JUDGMENT_ORG_ID" \
    -H "Content-Type: application/json" \
    -d "{\"project_name\": \"$PROJECT_NAME\"}" \
    "$JUDGMENT_API_URL/projects/resolve/" 2>&1)

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)

if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ API connection successful"
elif [ "$HTTP_CODE" = "401" ]; then
    echo "⚠️  API authentication failed (HTTP 401)"
    echo "   Check your API key and try again"
else
    echo "⚠️  API connection issue (HTTP $HTTP_CODE)"
    echo "   Check your API key and organization ID"
fi
