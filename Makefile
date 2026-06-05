.PHONY: help
.PHONY: test test-parallel test-serial test-benchmark test-watch test-coverage test-profile warm-fastembed-cache
.PHONY: docs docs-strict docs-serve docs-update-cards docs-check-cards docs-watch-cards docs-check-redirects
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

docs:
	poetry run sphinx-build -b html docs _build/docs

docs-strict:
	poetry run sphinx-build -b html -W --keep-going docs _build/docs

docs-serve:
	cd docs && poetry run sphinx-autobuild . _build/html --port 8000 --open-browser

docs-update-cards:
	cd docs && poetry run python scripts/update_cards/update_cards.py

docs-check-cards:
	cd docs && poetry run python scripts/update_cards/update_cards.py --dry-run

docs-watch-cards:
	cd docs && poetry run python scripts/update_cards/update_cards.py watch

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
		'  docs                  Build docs' \
		'  docs-strict           Build docs with warnings as errors' \
		'  docs-serve            Serve docs locally' \
		'  docs-update-cards     Update generated docs cards' \
		'  docs-check-cards      Check generated docs cards' \
		'  docs-watch-cards      Watch and update generated docs cards' \
		'  docs-check-redirects  Validate docs redirects' \
		'' \
		'Maintenance:' \
		'  pre-commit            Install and run pre-commit hooks'
