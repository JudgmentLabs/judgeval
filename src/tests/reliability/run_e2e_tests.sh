#!/bin/bash
# Helper script to run E2E tests against staging
#
# Usage:
#   ./run_e2e_tests.sh [test-selector]
#
# Examples:
#   ./run_e2e_tests.sh                                    # Run all non-slow E2E tests
#   ./run_e2e_tests.sh "e2e"                             # Run all E2E tests (including slow)
#   ./run_e2e_tests.sh "test_e2e_100_spans_baseline"    # Run specific test
#
# Prerequisites:
#   export JUDGMENT_API_KEY="your-api-key"
#   export JUDGMENT_ORG_ID="your-org-id"
#   export JUDGMENT_API_URL="https://staging.judgmentlabs.ai/"

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check environment variables
if [ -z "$JUDGMENT_API_KEY" ]; then
    echo -e "${RED}ERROR: JUDGMENT_API_KEY environment variable not set${NC}"
    echo "Please set it with: export JUDGMENT_API_KEY='your-api-key'"
    exit 1
fi

if [ -z "$JUDGMENT_ORG_ID" ]; then
    echo -e "${RED}ERROR: JUDGMENT_ORG_ID environment variable not set${NC}"
    echo "Please set it with: export JUDGMENT_ORG_ID='your-org-id'"
    exit 1
fi

# Default to staging URL if not set
if [ -z "$JUDGMENT_API_URL" ]; then
    export JUDGMENT_API_URL="https://staging.judgmentlabs.ai/"
    echo -e "${YELLOW}Using default API URL: $JUDGMENT_API_URL${NC}"
fi

# Determine test selector
TEST_SELECTOR="${1:-e2e and not slow}"

echo -e "${GREEN}Running E2E tests against: $JUDGMENT_API_URL${NC}"
echo -e "${GREEN}Test selector: $TEST_SELECTOR${NC}"
echo ""

# Run tests
pytest src/tests/reliability/test_e2e_staging.py \
    -v \
    -s \
    -m "$TEST_SELECTOR" \
    --timeout=1800 \
    --tb=short

echo ""
echo -e "${GREEN}✓ E2E tests completed${NC}"
