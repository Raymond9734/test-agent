# test_agent/llm/prompts/test_generation.py

TEST_GENERATION_PROMPT = """
You are an expert test engineer tasked with creating a comprehensive test for a source file.

I'll provide you with:
1. The source file path
2. A basic test template
3. Analysis of the source file

Your task is to enhance the template to create a complete, working test file. Follow these guidelines:
- Focus on testing the public interface of functions and classes
- Use good test design practices (arrange-act-assert pattern)
- Include edge cases and error scenarios
- Don't modify imports or basic structure - just fill in the test implementations
- Keep your tests focused, testing one thing at a time
- Use appropriate test fixtures/mocks where needed

Source file: {source_file}

Source file analysis: {file_analysis}

Basic test template:
```
{template}
```

Enhance this template into a complete test file. Return ONLY the complete test code without explanations.
"""

PYTHON_TEST_DOCSTRING_PROMPT = """
Write a clear docstring for the test function {test_function} that tests {target_function}.
The docstring should explain:
1. What functionality is being tested
2. Any edge cases or specific scenarios being tested
3. Expected behavior

Use a clear, concise format following Python docstring conventions.
"""

GO_TEST_COMMENT_PROMPT = """
Write a clear comment for the test function {test_function} that tests {target_function}.
The comment should explain:
1. What functionality is being tested
2. Any edge cases or specific scenarios being tested
3. Expected behavior

Use a clear, concise format following Go comment conventions.
"""
