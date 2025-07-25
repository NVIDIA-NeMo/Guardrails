#!/bin/bash

# Test script for the release process
# This script allows you to test the release process locally without actually creating a release

set -e

echo "🧪 Testing NeMo Guardrails Release Process"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if git-cliff is installed
if ! command -v git-cliff &> /dev/null; then
    echo -e "${RED}❌ git-cliff is not installed${NC}"
    echo "Please install git-cliff first:"
    echo "  brew install git-cliff  # macOS"
    echo "  cargo install git-cliff # or via Rust"
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}❌ Not in a git repository${NC}"
    exit 1
fi

# Test 1: Generate changelog preview
echo -e "\n${YELLOW}📝 Test 1: Generating changelog preview...${NC}"
if git cliff --unreleased --strip all > /tmp/changelog_preview.md; then
    echo -e "${GREEN}✅ Changelog preview generated successfully${NC}"
    echo "Preview content:"
    echo "---"
    cat /tmp/changelog_preview.md
    echo "---"
else
    echo -e "${RED}❌ Failed to generate changelog preview${NC}"
fi

# Test 2: Check version bumping
echo -e "\n${YELLOW}🔢 Test 2: Checking version bump...${NC}"
CURRENT_VERSION=$(grep -m1 'version = ' pyproject.toml | sed 's/.*version = "\(.*\)"/\1/')
NEXT_VERSION=$(git cliff --bumped-version)
echo "Current version: $CURRENT_VERSION"
echo "Next version would be: $NEXT_VERSION"

# Test 3: Verify cliff.toml configuration
echo -e "\n${YELLOW}⚙️  Test 3: Verifying cliff.toml configuration...${NC}"
if [ -f cliff.toml ] && git cliff --latest > /dev/null 2>&1; then
    echo -e "${GREEN}✅ cliff.toml is valid and working${NC}"
else
    echo -e "${RED}❌ cliff.toml has errors or is missing${NC}"
fi

# Test 4: Check conventional commits
echo -e "\n${YELLOW}📋 Test 4: Checking recent commits format...${NC}"
echo "Last 5 commits:"
git log --oneline -5

# Test 5: Dry run of changelog generation
echo -e "\n${YELLOW}🚀 Test 5: Dry run of full changelog generation...${NC}"
if git cliff -o /tmp/test_changelog.md; then
    echo -e "${GREEN}✅ Full changelog generated successfully${NC}"
    echo "Generated $(wc -l < /tmp/test_changelog.md) lines"

    # Check if NOTE section is preserved
    if grep -q "NOTE:" /tmp/test_changelog.md; then
        echo -e "${GREEN}✅ NOTE section is preserved${NC}"
    else
        echo -e "${RED}❌ NOTE section might be missing${NC}"
    fi
else
    echo -e "${RED}❌ Failed to generate full changelog${NC}"
fi

# Test 6: Poetry availability test
echo -e "\n${YELLOW}🐍 Test 6: Checking Poetry installation...${NC}"
if command -v poetry &> /dev/null; then
    echo -e "${GREEN}✅ Poetry is installed${NC}"
    poetry --version

    # Test Poetry version command
    echo -e "\n${YELLOW}Testing Poetry version command...${NC}"
    CURRENT_POETRY_VERSION=$(poetry version -s)
    echo "Current project version: $CURRENT_POETRY_VERSION"

    # Test dry-run of version bump
    echo -e "\nDry run of version bump:"
    poetry version patch --dry-run || true
else
    echo -e "${RED}❌ Poetry is not installed${NC}"
    echo "Please install Poetry first:"
    echo "  curl -sSL https://install.python-poetry.org | python3 -"
fi

echo -e "\n${GREEN}✨ Release process testing complete!${NC}"
echo -e "\nTo perform an actual release, use the GitHub Actions workflow:"
echo "  1. Go to Actions → 📦 Release"
echo "  2. Click 'Run workflow'"
echo "  3. Enter the version number"
echo "  4. Click 'Run workflow'"
