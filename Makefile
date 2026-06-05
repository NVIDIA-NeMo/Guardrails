.PHONY: all test tests test_watch test_coverage test_profile docs docs-strict docs-serve docs-fern docs-fern-strict docs-fern-live docs-fern-preview-watch docs-fern-generate-sdk docs-fern-fix-empty-links docs-update-cards docs-check-cards docs-watch-cards pre_commit help

# Default target executed when no specific target is provided to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/

test:
	poetry run pytest $(TEST_FILE)

tests:
	poetry run pytest $(TEST_FILE)

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

docs-fern: docs-fern-strict

docs-fern-strict: docs-fern-generate-sdk
	FERN_VERSION=$$(node -p "require('./fern/fern.config.json').version") && cd fern && npx --yes "fern-api@$${FERN_VERSION}" check

docs-fern-live: docs-fern-generate-sdk
	FERN_VERSION=$$(node -p "require('./fern/fern.config.json').version") && cd fern && npx --yes "fern-api@$${FERN_VERSION}" docs dev

docs-fern-preview-watch: docs-fern-generate-sdk
	node scripts/watch-fern-preview.mjs

docs-fern-generate-sdk:
	FERN_VERSION=$$(node -p "require('./fern/fern.config.json').version") && cd fern && npx --yes "fern-api@$${FERN_VERSION}" docs md generate --library guardrails-python-sdk
	node scripts/normalize-fern-sdk-reference.mjs

docs-fern-fix-empty-links:
	node scripts/fix-empty-fern-links.mjs

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
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo 'test_coverage                - run unit tests with coverage'
	@echo 'docs                         - build docs, if you installed the docs dependencies'
	@echo 'docs-strict                  - build docs with warnings as errors (used in CI)'
	@echo 'docs-serve                   - serve docs locally with auto-rebuild on changes'
	@echo 'docs-fern                    - check Fern docs using the pinned Fern CLI'
	@echo 'docs-fern-strict             - check Fern docs using the pinned Fern CLI'
	@echo 'docs-fern-live               - serve Fern docs locally'
	@echo 'docs-fern-preview-watch      - watch and publish Fern preview for the current branch'
	@echo 'docs-fern-generate-sdk       - regenerate Python SDK reference pages with Fern'
	@echo 'docs-fern-fix-empty-links    - replace empty Markdown links with titles from Fern navigation'
	@echo 'docs-update-cards            - update grid cards in index files from linked pages'
	@echo 'docs-check-cards             - check if grid cards are up to date (dry run)'
	@echo 'docs-watch-cards             - watch for file changes and auto-update cards'
	@echo 'docs-check-redirects         - validate that all redirect targets exist'
	@echo 'pre_commit                   - run pre-commit hooks'
