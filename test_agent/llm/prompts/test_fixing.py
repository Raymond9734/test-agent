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

IMPORT_ERROR_FIXING_PROMPT = """
This test is failing due to an import error. The missing module is: {missing_module}

Please fix the import statement or provide a valid alternative approach. Consider:
1. Correcting the import path
2. Using a mock or stub if appropriate
3. Adding a dependency check or skip test if the module is optional

Return ONLY the corrected code without explanations.
"""

ASSERTION_ERROR_FIXING_PROMPT = """
This test is failing due to an assertion error. The assertion that failed is:

{assertion_error}

Please modify the test to fix this assertion. Consider:
1. Checking if the expected value is correct
2. Updating the assertion to match the actual behavior if it's valid
3. Fixing any setup code that might be affecting the test

Return ONLY the corrected code without explanations.
"""
