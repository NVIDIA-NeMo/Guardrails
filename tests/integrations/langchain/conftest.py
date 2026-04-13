import pytest


@pytest.fixture(autouse=True)
def _langchain_tests_use_langchain(langchain_framework):
    pass
