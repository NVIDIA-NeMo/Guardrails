.PHONY: help
.PHONY: test test-parallel test-serial test-benchmark test-watch test-coverage test-profile warm-fastembed-cache
.PHONY: docs-fern docs-fern-strict docs-fern-live docs-fern-preview-watch docs-fern-generate-sdk docs-fern-fix-empty-links docs-check-links docs-check-redirects docs-fern-publish-staging docs-fern-publish-public
.PHONY: pre-commit

.DEFAULT_GOAL := help

TEST ?=
ARGS ?=
WORKERS ?= auto
# pytest-xdist --dist strategy for $(PYTEST) -n $(WORKERS) --dist $(DIST) $(ARGS) $(TEST).
# worksteal dynamically rebalances queued tests; override DIST when debugging or grouping matters.
DIST ?= worksteal

PYTEST ?= poetry run pytest
# These targets assume a Unix-like shell for env -u; use bash, Git Bash, or WSL on Windows.
UNIT_TEST_ENV ?= env -u OPENAI_API_KEY -u NVIDIA_API_KEY \
	-u LIVE_TEST -u LIVE_TEST_MODE -u TEST_LIVE_MODE

FASTEMBED_CACHE ?= .cache/fastembed
FASTEMBED_MODEL ?= sentence-transformers/all-MiniLM-L6-v2
FASTEMBED_ENV ?= env FASTEMBED_CACHE_PATH=$(FASTEMBED_CACHE)
FERN_STAGING_INSTANCE ?= nvidia-nemo-guardrails-staging.docs.buildwithfern.com/nemo/guardrails
FERN_PUBLIC_INSTANCE ?= nvidia-nemo-guardrails.docs.buildwithfern.com/nemo/guardrails

test:
	$(UNIT_TEST_ENV) $(PYTEST) -n $(WORKERS) --dist $(DIST) $(ARGS) $(TEST)

test-parallel: test

test-serial:
	$(PYTEST) $(ARGS) $(TEST)

test-benchmark:
	$(PYTEST) $(ARGS) benchmark/tests

test-watch:
	poetry run ptw --snapshot-update --now . -- -vv $(ARGS) $(TEST)

test-coverage:
	$(UNIT_TEST_ENV) $(PYTEST) -n $(WORKERS) --dist $(DIST) --cov=nemoguardrails --cov-report=xml:coverage.xml $(ARGS) $(TEST)

test-profile:
	$(PYTEST) -vv --profile-svg $(ARGS) $(TEST)

warm-fastembed-cache:
	$(FASTEMBED_ENV) poetry run python -c 'from fastembed import TextEmbedding; model = TextEmbedding("$(FASTEMBED_MODEL)"); next(model.embed(["warmup"]))'

docs-fern: docs-fern-strict

docs-fern-strict: docs-fern-generate-sdk
	FERN_VERSION=$$(node -p "require('./fern/fern.config.json').version") && cd fern && npx --yes "fern-api@$${FERN_VERSION}" check

docs-fern-live: docs-fern-generate-sdk
	FERN_VERSION=$$(node -p "require('./fern/fern.config.json').version") && cd fern && npx --yes "fern-api@$${FERN_VERSION}" docs dev

docs-fern-publish-staging: docs-fern-generate-sdk
	FERN_VERSION=$$(node -p "require('./fern/fern.config.json').version") && cd fern && npx --yes "fern-api@$${FERN_VERSION}" generate --docs --instance "$(FERN_STAGING_INSTANCE)"

docs-fern-publish-public: docs-fern-generate-sdk
	FERN_VERSION=$$(node -p "require('./fern/fern.config.json').version") && cd fern && npx --yes "fern-api@$${FERN_VERSION}" generate --docs --instance "$(FERN_PUBLIC_INSTANCE)"

docs-fern-preview-watch: docs-fern-generate-sdk
	node scripts/watch-fern-preview.mjs

docs-fern-generate-sdk:
	FERN_VERSION=$$(node -p "require('./fern/fern.config.json').version") && cd fern && npx --yes "fern-api@$${FERN_VERSION}" docs md generate --library guardrails-python-sdk
	node scripts/normalize-fern-sdk-reference.mjs

docs-fern-fix-empty-links:
	node scripts/fix-empty-fern-links.mjs

docs-check-links:
	bash scripts/check-docs-links.sh --local-only

docs-check-redirects:
	cd docs && poetry run python scripts/validate_redirects.py

pre-commit:
	poetry run pre-commit install
	poetry run pre-commit run --all-files

help:
	@printf '%s\n' \
		'' \
		'Usage:' \
		'  make test [TEST=path] [WORKERS=auto] [ARGS="-q --tb=short"]' \
		'  make test-serial [TEST=path] [ARGS="-q"]' \
		'  make test-benchmark [ARGS="-q"]' \
		'  make test-parallel [TEST=path] [WORKERS=auto] [ARGS="-q --tb=short"]' \
		'  make test-watch [TEST=path]' \
		'' \
		'Tests:' \
		'  test                  Run pytest.ini testpaths with pytest-xdist' \
		'  test-parallel         Alias for test' \
		'  test-serial           Run pytest without xdist or env filtering' \
		'  test-benchmark        Run benchmark tooling tests' \
		'  test-watch            Run pytest in watch mode' \
		'  test-coverage         Run pytest with coverage' \
		'  test-profile          Run pytest with profiling' \
		'  warm-fastembed-cache  Prime the repo-local FastEmbed cache' \
		'' \
		'Docs:' \
		'  docs-fern             Check Fern docs using the pinned Fern CLI' \
		'  docs-fern-strict      Check Fern docs using the pinned Fern CLI' \
		'  docs-fern-live        Serve Fern docs locally' \
		'  docs-fern-publish-staging Publish Fern docs to the staging instance' \
		'  docs-fern-publish-public Publish Fern docs to the public instance' \
		'  docs-fern-preview-watch Watch and publish Fern preview for the current branch' \
		'  docs-fern-generate-sdk Regenerate Python SDK reference pages with Fern' \
		'  docs-fern-fix-empty-links Replace empty Markdown links with titles from Fern navigation' \
		'  docs-check-links     Validate Markdown and MDX links locally' \
		'  docs-check-redirects  Validate docs redirects' \
		'' \
		'Maintenance:' \
		'  pre-commit            Install and run pre-commit hooks'
