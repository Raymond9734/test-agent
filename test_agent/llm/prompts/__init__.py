# test_agent/llm/prompts/__init__.py

from .language_detection import LANGUAGE_DETECTION_PROMPT
from .test_generation import TEST_GENERATION_PROMPT
from .test_fixing import TEST_FIXING_PROMPT

__all__ = ["LANGUAGE_DETECTION_PROMPT", "TEST_GENERATION_PROMPT", "TEST_FIXING_PROMPT"]
