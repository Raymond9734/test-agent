# test_agent/llm/prompts/test_fixing.py

TEST_FIXING_PROMPT = """
I need help fixing a failing test. I'll provide:
1. The test content
2. The error output
3. An analysis of the error

Test file: {test_file}
Source file: {source_file}

Error output:
{error_output}

Error analysis:
- Error type: {error_type}
- Error message: {error_message}
- Has syntax error: {has_syntax_error}
- Has import error: {has_import_error}
- Has assertion error: {has_assertion_error}
- Has exception: {has_exception}

Current test content:
```
{test_content}
```

Please fix the test based on the error output. Return ONLY the corrected test code without explanations.
"""
