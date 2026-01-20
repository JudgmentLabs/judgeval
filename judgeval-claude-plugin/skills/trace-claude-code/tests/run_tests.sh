#!/bin/bash
###
# Test runner for Judgeval Claude Code tracing hooks
#
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh common       # Run only common.sh tests
#   ./run_tests.sh subagent     # Run only subagent_stop.sh tests
#   ./run_tests.sh session_end  # Run only session_end.sh tests
###

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║     Judgeval Claude Code Tracing - Test Suite             ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check dependencies
echo "Checking dependencies..."
for cmd in jq bash; do
    if ! command -v "$cmd" &>/dev/null; then
        echo -e "${RED}Error: $cmd is required but not installed${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓ Dependencies OK${NC}"
echo ""

# Track overall results
TOTAL_SUITES=0
PASSED_SUITES=0
FAILED_SUITES=0

run_test_suite() {
    local suite_name="$1"
    local suite_file="$2"
    
    TOTAL_SUITES=$((TOTAL_SUITES + 1))
    
    echo -e "${YELLOW}Running: $suite_name${NC}"
    echo "----------------------------------------"
    
    if bash "$suite_file"; then
        PASSED_SUITES=$((PASSED_SUITES + 1))
        echo -e "${GREEN}Suite passed!${NC}"
    else
        FAILED_SUITES=$((FAILED_SUITES + 1))
        echo -e "${RED}Suite failed!${NC}"
    fi
    echo ""
}

# Determine which tests to run
TEST_FILTER="${1:-all}"

case "$TEST_FILTER" in
    common)
        run_test_suite "common.sh utilities" "$SCRIPT_DIR/test_common.sh"
        ;;
    subagent)
        run_test_suite "subagent_stop.sh" "$SCRIPT_DIR/test_subagent_stop.sh"
        ;;
    session_end)
        run_test_suite "session_end.sh" "$SCRIPT_DIR/test_session_end.sh"
        ;;
    integration)
        run_test_suite "integration tests" "$SCRIPT_DIR/test_integration.sh"
        ;;
    all|*)
        run_test_suite "common.sh utilities" "$SCRIPT_DIR/test_common.sh"
        run_test_suite "subagent_stop.sh" "$SCRIPT_DIR/test_subagent_stop.sh"
        run_test_suite "session_end.sh" "$SCRIPT_DIR/test_session_end.sh"
        run_test_suite "integration tests" "$SCRIPT_DIR/test_integration.sh"
        ;;
esac

# Final summary
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                    Final Summary                          ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "Test suites run: $TOTAL_SUITES"
echo -e "Passed: ${GREEN}$PASSED_SUITES${NC}"
echo -e "Failed: ${RED}$FAILED_SUITES${NC}"
echo ""

if [ $FAILED_SUITES -gt 0 ]; then
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
