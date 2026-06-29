.PHONY: help
.PHONY: test test-parallel test-serial test-benchmark test-watch test-coverage test-profile record-cassettes rewrite-cassettes replay-cassettes snapshot-cassettes check-record-test-env warm-fastembed-cache
.PHONY: docs-fern docs-fern-strict docs-fern-live docs-fern-preview-watch docs-fern-generate-sdk docs-fern-fix-empty-links docs-check-redirects docs-fern-publish-staging docs-fern-publish-public
.PHONY: pre-commit

.DEFAULT_GOAL := help

TEST ?=
ARGS ?=
WORKERS ?= auto
# pytest-xdist --dist strategy for $(PYTEST) -n $(WORKERS) --dist $(DIST) $(ARGS) $(TEST).
# worksteal dynamically rebalances queued tests; override DIST when debugging or grouping matters.
DIST ?= worksteal

PYTEST ?= poetry run pytest
RECORDED_TESTS ?= tests/recorded
RECORDED_RECORD_MODE ?= once
RECORDED_SNAPSHOT_MODE ?= create
RECORDED_REQUIRED_KEYS ?= OPENAI_API_KEY NVIDIA_API_KEY
RECORDED_REPLAY_ENV ?= env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy
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

record-cassettes: check-record-test-env
	$(PYTEST) $(RECORDED_TESTS) --record-mode=$(RECORDED_RECORD_MODE) -m "not fake_cassette"
	$(RECORDED_REPLAY_ENV) $(PYTEST) $(RECORDED_TESTS) --block-network --inline-snapshot=$(RECORDED_SNAPSHOT_MODE)
	$(RECORDED_REPLAY_ENV) $(PYTEST) $(RECORDED_TESTS) --block-network

rewrite-cassettes:
	$(MAKE) record-cassettes RECORDED_RECORD_MODE=rewrite RECORDED_SNAPSHOT_MODE=fix

replay-cassettes:
	$(RECORDED_REPLAY_ENV) $(PYTEST) $(RECORDED_TESTS) --block-network

snapshot-cassettes:
	$(RECORDED_REPLAY_ENV) $(PYTEST) $(RECORDED_TESTS) --block-network --inline-snapshot=fix

check-record-test-env:
	@missing=""; \
	for key in $(RECORDED_REQUIRED_KEYS); do \
		if [ -z "$$(printenv "$$key")" ]; then \
			missing="$$missing $$key"; \
		fi; \
	done; \
	if [ -n "$$missing" ]; then \
		printf '%s\n' "Missing required env var(s):$$missing" \
			"Set them before make record-cassettes, or override RECORDED_REQUIRED_KEYS for a focused refresh."; \
		exit 2; \
	fi

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
		'  make record-cassettes [RECORDED_TESTS=tests/recorded] [RECORDED_RECORD_MODE=once] [RECORDED_SNAPSHOT_MODE=create] [RECORDED_REQUIRED_KEYS="OPENAI_API_KEY NVIDIA_API_KEY"]' \
		'  make rewrite-cassettes [RECORDED_TESTS=tests/recorded] [RECORDED_REQUIRED_KEYS="OPENAI_API_KEY NVIDIA_API_KEY"]' \
		'  make replay-cassettes [RECORDED_TESTS=tests/recorded]' \
		'  make snapshot-cassettes [RECORDED_TESTS=tests/recorded]' \
		'' \
		'Tests:' \
		'  test                  Run pytest.ini testpaths with pytest-xdist' \
		'  test-parallel         Alias for test' \
		'  test-serial           Run pytest without xdist or env filtering' \
		'  test-benchmark        Run benchmark tooling tests' \
		'  test-watch            Run pytest in watch mode' \
		'  test-coverage         Run pytest with coverage' \
		'  test-profile          Run pytest with profiling' \
		'  record-cassettes      Record missing or selected cassettes, fill snapshots, and verify replay' \
		'  rewrite-cassettes     Rewrite selected cassettes, fill snapshots, and verify replay' \
		'  replay-cassettes      Verify selected cassettes offline without recording' \
		'  snapshot-cassettes    Update inline snapshots from existing cassettes offline' \
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
		'  docs-check-redirects  Validate docs redirects' \
		'' \
		'Maintenance:' \
		'  pre-commit            Install and run pre-commit hooks'
