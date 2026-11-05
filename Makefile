.PHONY: help install
.PHONY: test test-parallel test-serial test-benchmark test-watch test-coverage test-profile test-hooks-agents test-relay-plugin record-cassettes rewrite-cassettes replay-cassettes snapshot-cassettes check-record-test-env warm-fastembed-cache
.PHONY: test-docs-scripts docs-fern-node-dependencies docs-fern docs-fern-strict docs-fern-live docs-fern-preview-watch docs-fern-generate-openapi docs-fern-generate-references docs-fern-generate-sdk docs-fern-fix-empty-links docs-fern-publish-staging docs-fern-publish-public
.PHONY: pre-commit pre-commit-install

.DEFAULT_GOAL := help

TEST ?=
ARGS ?=
WORKERS ?= auto
UV_LOCKED ?= true
export UV_LOCKED
# pytest-xdist --dist strategy for $(PYTEST) -n $(WORKERS) --dist $(DIST) $(ARGS) $(TEST).
# worksteal dynamically rebalances queued tests; override DIST when debugging or grouping matters.
DIST ?= worksteal

PYTEST ?= uv run pytest
RECORDED_TESTS ?= tests/recorded
RECORDED_RECORD_MODE ?= once
RECORDED_SNAPSHOT_MODE ?= create
RECORDED_REQUIRED_KEYS ?= OPENAI_API_KEY NVIDIA_API_KEY
RECORDED_REPLAY_ENV ?= env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy
# These targets assume a Unix-like shell for env -u; use bash, Git Bash, or WSL on Windows.
UNIT_TEST_ENV ?= env -u OPENAI_API_KEY -u NVIDIA_API_KEY -u NVIDIA_INFERENCE_KEY \
	-u LIVE_TEST -u LIVE_TEST_MODE -u TEST_LIVE_MODE

# The hooks-server tests that run Claude Code and Codex take both agents from their published wheels, installed into a
# directory of their own rather than .venv, so neither is a Guardrails dependency. `uv pip sync` makes the directory
# match requirements.txt and checks every wheel against its hash. The tests run in .venv, with live provider keys unset
# as for `make test`, and find the agents through HOOKS_AGENTS_DIR.
HOOKS_AGENTS_TESTS ?= integration_tests/hooks_agents
HOOKS_AGENTS_DIR ?= .cache/hooks_agents

# The NeMo Relay plugin's end-to-end tests run in their own Python 3.11 environment without Guardrails, while the hooks
# servers they start run from .venv. `uv pip sync --require-hashes` makes that environment match requirements.txt and
# checks every package against its hash, which `uv run --with-requirements` doesn't. nemo-relay is exempt from
# [tool.uv]'s exclude-newer window, which would filter out a recent pin. Live provider keys are unset as for `make test`.
RELAY_PLUGIN_TESTS ?= integration_tests/relay_plugin
RELAY_PLUGIN_VENV ?= .venv-relay-plugin
RELAY_PLUGIN_PYTEST ?= $(UNIT_TEST_ENV) $(RELAY_PLUGIN_VENV)/bin/pytest

FASTEMBED_CACHE ?= .cache/fastembed
FASTEMBED_MODEL ?= sentence-transformers/all-MiniLM-L6-v2
FASTEMBED_ENV ?= env FASTEMBED_CACHE_PATH=$(FASTEMBED_CACHE)
FERN_STAGING_INSTANCE ?= nvidia-nemo-guardrails-staging.docs.buildwithfern.com/nemo/guardrails
FERN_PUBLIC_INSTANCE ?= nvidia-nemo-guardrails.docs.buildwithfern.com/nemo/guardrails
DOCS_NODE_STAMP := node_modules/.package-lock.json

install:
	uv sync --group dev

test:
	$(UNIT_TEST_ENV) $(PYTEST) -n $(WORKERS) --dist $(DIST) $(ARGS) $(TEST)

test-parallel: test

test-serial:
	$(PYTEST) $(ARGS) $(TEST)

test-benchmark:
	$(PYTEST) $(ARGS) benchmark/tests

test-watch:
	uv run ptw --snapshot-update --now . -- -vv $(ARGS) $(TEST)

test-coverage:
	$(UNIT_TEST_ENV) $(PYTEST) -n $(WORKERS) --dist $(DIST) --cov=nemoguardrails --cov-report=xml:coverage.xml $(ARGS) $(TEST)

test-profile:
	$(PYTEST) -vv --profile-svg $(ARGS) $(TEST)

test-hooks-agents:
	uv pip sync --target $(HOOKS_AGENTS_DIR) --require-hashes $(HOOKS_AGENTS_TESTS)/requirements.txt
	HOOKS_AGENTS_DIR=$(abspath $(HOOKS_AGENTS_DIR)) $(UNIT_TEST_ENV) $(PYTEST) -n $(WORKERS) --dist $(DIST) $(ARGS) \
		$(or $(TEST),$(HOOKS_AGENTS_TESTS))

# The first sync is the one `uv run` makes of .venv for the other targets; the next two build the tests' environment.
test-relay-plugin:
	uv sync --inexact
	uv venv --quiet --allow-existing --python 3.11 $(RELAY_PLUGIN_VENV)
	uv pip sync --python $(RELAY_PLUGIN_VENV)/bin/python --require-hashes --exclude-newer-package nemo-relay=false \
		$(RELAY_PLUGIN_TESTS)/requirements.txt
	$(RELAY_PLUGIN_PYTEST) $(ARGS) $(or $(TEST),$(RELAY_PLUGIN_TESTS))

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
	$(FASTEMBED_ENV) uv run python -c 'from fastembed import TextEmbedding; model = TextEmbedding("$(FASTEMBED_MODEL)"); next(model.embed(["warmup"]))'

test-docs-scripts: docs-fern-node-dependencies
	npm run test:docs-scripts

docs-fern-node-dependencies: $(DOCS_NODE_STAMP)

$(DOCS_NODE_STAMP): package.json package-lock.json
	npm ci

docs-fern: docs-fern-strict

docs-fern-strict: docs-fern-generate-references
	node scripts/run-fern-with-ref-sdk.mjs check

docs-fern-live: docs-fern-generate-references
	FERN_VERSION=$$(node -p "require('./fern/fern.config.json').version") && cd fern && npx --yes "fern-api@$${FERN_VERSION}" docs dev

docs-fern-publish-staging: docs-fern-generate-references
	node scripts/run-fern-with-ref-sdk.mjs generate --docs --instance "$(FERN_STAGING_INSTANCE)"

docs-fern-publish-public: docs-fern-generate-references
	node scripts/run-fern-with-ref-sdk.mjs generate --docs --instance "$(FERN_PUBLIC_INSTANCE)"

docs-fern-preview-watch: docs-fern-node-dependencies
	node scripts/watch-fern-preview.mjs

docs-fern-generate-references: docs-fern-generate-sdk

docs-fern-generate-openapi:
	# Importing the server schema does not require publishing credentials.
	env -u FERN_TOKEN -u DOCS_FERN_TOKEN uv run --locked --extra server python scripts/export_openapi.py
	env -u FERN_TOKEN -u DOCS_FERN_TOKEN uv run --locked --extra server python scripts/export_proxy_openapi.py

docs-fern-generate-sdk: docs-fern-generate-openapi docs-fern-node-dependencies
	FERN_VERSION=$$(node -p "require('./fern/fern.config.json').version") && cd fern && npx --yes "fern-api@$${FERN_VERSION}" docs md generate --library guardrails-python-sdk
	node scripts/normalize-fern-sdk-reference.mjs

docs-fern-fix-empty-links:
	node scripts/fix-empty-fern-links.mjs

pre-commit:
	uv run pre-commit run --all-files

pre-commit-install:
	uv run pre-commit install

help:
	@printf '%s\n' \
		'' \
		'Usage:' \
		'  make install' \
		'  make test [TEST=path] [WORKERS=auto] [ARGS="-q --tb=short"]' \
		'  make test-serial [TEST=path] [ARGS="-q"]' \
		'  make test-benchmark [ARGS="-q"]' \
		'  make test-parallel [TEST=path] [WORKERS=auto] [ARGS="-q --tb=short"]' \
		'  make test-watch [TEST=path]' \
		'  make test-hooks-agents [TEST=integration_tests/hooks_agents/...] [WORKERS=auto] [ARGS="-v"]' \
		'  make test-relay-plugin [TEST=integration_tests/relay_plugin/...] [ARGS="-v"]' \
		'  make record-cassettes [RECORDED_TESTS=tests/recorded] [RECORDED_RECORD_MODE=once] [RECORDED_SNAPSHOT_MODE=create] [RECORDED_REQUIRED_KEYS="OPENAI_API_KEY NVIDIA_API_KEY"]' \
		'  make rewrite-cassettes [RECORDED_TESTS=tests/recorded] [RECORDED_REQUIRED_KEYS="OPENAI_API_KEY NVIDIA_API_KEY"]' \
		'  make replay-cassettes [RECORDED_TESTS=tests/recorded]' \
		'  make snapshot-cassettes [RECORDED_TESTS=tests/recorded]' \
		'' \
		'Setup:' \
		'  install               Install locked development dependencies' \
		'' \
		'Tests:' \
		'  test                  Run pytest.ini testpaths with pytest-xdist' \
		'  test-parallel         Alias for test' \
		'  test-serial           Run pytest without xdist or env filtering' \
		'  test-benchmark        Run benchmark tooling tests' \
		'  test-watch            Run pytest in watch mode' \
		'  test-coverage         Run pytest with coverage' \
		'  test-profile          Run pytest with profiling' \
		'  test-hooks-agents     Run the hooks-server tests with real Claude Code and Codex (Linux, macOS)' \
		'  test-relay-plugin     Run the NeMo Relay plugin end-to-end tests (needs a Rust toolchain)' \
		'  record-cassettes      Record missing or selected cassettes, fill snapshots, and verify replay' \
		'  rewrite-cassettes     Rewrite selected cassettes, fill snapshots, and verify replay' \
		'  replay-cassettes      Verify selected cassettes offline without recording' \
		'  snapshot-cassettes    Update inline snapshots from existing cassettes offline' \
		'  warm-fastembed-cache  Prime the repo-local FastEmbed cache' \
		'' \
		'Docs:' \
		'  test-docs-scripts     Run unit tests for Fern documentation scripts' \
		'  docs-fern             Check Fern docs using the pinned Fern CLI' \
		'  docs-fern-strict      Check Fern docs using the pinned Fern CLI' \
		'  docs-fern-live        Serve Fern docs locally' \
		'  docs-fern-publish-staging Publish Fern docs to the staging instance' \
		'  docs-fern-publish-public Publish Fern docs to the public instance' \
		'  docs-fern-preview-watch Watch and publish Fern preview for the current branch' \
		'  docs-fern-generate-references Regenerate OpenAPI and Python SDK references' \
		'  docs-fern-generate-openapi Regenerate the OpenAPI specification from FastAPI' \
		'  docs-fern-generate-sdk Regenerate Python SDK reference pages with Fern' \
		'  docs-fern-fix-empty-links Replace empty Markdown links with titles from Fern navigation' \
		'' \
		'Maintenance:' \
		'  pre-commit            Run pre-commit hooks against all files' \
		'  pre-commit-install    Install the Git pre-commit hook'
