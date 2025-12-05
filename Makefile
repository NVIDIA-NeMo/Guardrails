.PHONY: all test tests test_watch test_coverage test_profile docs docs-serve docs-update-cards docs-check-cards docs-watch-cards pre_commit help

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

docs-serve:
	cd docs && poetry run sphinx-autobuild . _build/html --port 8000 --open-browser

docs-update-cards:
	cd docs && poetry run python scripts/update_cards/update_cards.py

docs-check-cards:
	cd docs && poetry run python scripts/update_cards/update_cards.py --dry-run

docs-watch-cards:
	cd docs && poetry run python scripts/update_cards/update_cards.py watch

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
	@echo 'docs-serve                   - serve docs locally with auto-rebuild on changes'
	@echo 'docs-update-cards            - update grid cards in index files from linked pages'
	@echo 'docs-check-cards             - check if grid cards are up to date (dry run)'
	@echo 'docs-watch-cards             - watch for file changes and auto-update cards'
	@echo 'pre_commit                   - run pre-commit hooks'
