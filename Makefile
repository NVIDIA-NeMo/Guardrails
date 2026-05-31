.PHONY: all test tests warm_fastembed_cache test_parallel test_fast test_watch test_coverage test_profile docs docs-strict docs-serve docs-update-cards docs-check-cards docs-watch-cards pre_commit help

# Default target executed when no specific target is provided to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/
PYTEST_ARGS ?=
PYTEST_WORKERS ?= 8
PYTEST_DIST ?= worksteal
PYTEST_FAST_MARK_EXPR ?= not slow and not live
PYTEST_FAST_ARGS ?= --tb=short -q
FAST_TEST_FILE ?= \
	tests/test_utils.py \
	tests/test_cache_utils.py \
	tests/test_clavata_utils.py \
	tests/test_parser_utils.py \
	tests/test_config_loading.py \
	tests/test_config_validation.py \
	tests/test_rails_llm_config.py \
	tests/test_rails_llm_utils.py \
	tests/test_provider_selection.py \
	tests/test_imports.py \
	tests/test_types.py \
	tests/test_types_exports.py \
	tests/test_output_parsers.py \
	tests/test_content_safety_output_parsers.py \
	tests/test_jailbreak_config.py \
	tests/test_jailbreak_models.py \
	tests/test_guardrails_ai_config.py \
	tests/test_clavata_models.py \
	tests/test_buffer_strategy.py \
	tests/test_action_params_types.py \
	tests/test_actions_output_mapping.py \
	tests/test_actions_validation.py \
	tests/test_action_error.py \
	tests/llm/frameworks/test_registry.py \
	tests/server/test_schema_utils.py \
	tests/tracing/spans/test_span_format_enum.py \
	tests/tracing/spans/test_spans.py \
	tests/tracing/test_span_formatting.py \
	tests/_compat/test_langchain_kwargs.py
FASTEMBED_CACHE_PATH ?= .cache/fastembed
FASTEMBED_MODEL ?= sentence-transformers/all-MiniLM-L6-v2
PYTEST_PARALLEL_ENV ?= env -u OPENAI_API_KEY -u NVIDIA_API_KEY FASTEMBED_CACHE_PATH=$(FASTEMBED_CACHE_PATH)

test:
	poetry run pytest $(PYTEST_ARGS) $(TEST_FILE)

tests:
	poetry run pytest $(PYTEST_ARGS) $(TEST_FILE)

warm_fastembed_cache:
	$(PYTEST_PARALLEL_ENV) poetry run python -c 'from fastembed import TextEmbedding; model = TextEmbedding("$(FASTEMBED_MODEL)"); next(model.embed(["warmup"]))'

test_parallel: warm_fastembed_cache
	$(PYTEST_PARALLEL_ENV) poetry run pytest -n $(PYTEST_WORKERS) --dist $(PYTEST_DIST) $(PYTEST_ARGS) $(TEST_FILE)

test_fast:
	$(PYTEST_PARALLEL_ENV) poetry run pytest -n $(PYTEST_WORKERS) --dist $(PYTEST_DIST) -m "$(PYTEST_FAST_MARK_EXPR)" $(PYTEST_FAST_ARGS) $(PYTEST_ARGS) $(FAST_TEST_FILE)

test_watch:
	poetry run ptw --snapshot-update --now . -- -vv $(TEST_FILE)

test_coverage:
	poetry run pytest --cov=$(TEST_FILE) --cov-report=term-missing

test_profile:
	poetry run pytest -vv tests/ --profile-svg

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

pre_commit:
	pre-commit install
	pre-commit run --all-files


# HELP

help:
	@echo '----'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in given file'
	@echo 'warm_fastembed_cache         - prepare the repo-local FastEmbed cache for parallel tests'
	@echo 'test_parallel                - run unit tests with pytest-xdist using a warmed FastEmbed cache'
	@echo 'test_fast                    - run curated fast parallel test subset'
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo 'test_coverage                - run unit tests with coverage'
	@echo 'docs                         - build docs, if you installed the docs dependencies'
	@echo 'docs-strict                  - build docs with warnings as errors (used in CI)'
	@echo 'docs-serve                   - serve docs locally with auto-rebuild on changes'
	@echo 'docs-update-cards            - update grid cards in index files from linked pages'
	@echo 'docs-check-cards             - check if grid cards are up to date (dry run)'
	@echo 'docs-watch-cards             - watch for file changes and auto-update cards'
	@echo 'docs-check-redirects         - validate that all redirect targets exist'
	@echo 'pre_commit                   - run pre-commit hooks'
