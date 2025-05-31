# test_agent/llm/prompts/language_detection.py

LANGUAGE_DETECTION_PROMPT = """
You are an expert programmer tasked with identifying the programming language used in a project.

Project directory: {project_dir}

I will provide you with information about the files in this project, and you need to determine the primary programming language used.

File extensions found: {extensions}
Key files: {key_files}

Based on this information, which programming language is most likely used in this project?

Respond with just the language name (e.g., "python", "go", "javascript", etc.) and a brief explanation.
"""
