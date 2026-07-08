import sys
import os
import pytest

# Add backend root to path so agents/tools/etc. are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Hard guarantee: no test may hit a real LLM endpoint.
import pydantic_ai.models

pydantic_ai.models.ALLOW_MODEL_REQUESTS = False

MOCK_CFG = {
    "base_url": "http://localhost:1234/v1",
    "model": "test-model",
    "max_tokens": 4096,
    "max_response_tokens": 1024,
}


@pytest.fixture
def cfg():
    return MOCK_CFG.copy()
