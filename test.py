#!/usr/bin/env python3
"""
Test script to verify the fixes for import and location issues
"""

import os
import sys
from test_agent.language.python.adapter import PythonAdapter


def test_fixes():
    """Test the fixed functionality"""

    # Test the adapter
    adapter = PythonAdapter()

    # Test project directory from the logs
    project_dir = "/home/rayzy/Berrijam Internship/gojiberri/gojiberri/ui/utils"
    source_file = os.path.join(project_dir, "api_utils.py")

    print("=" * 60)
    print("TESTING FIXES FOR IMPORT AND LOCATION ISSUES")
    print("=" * 60)

    # Test 1: Pattern Detection
    print("\n1. Testing Pattern Detection:")
    print(f"Project directory: {project_dir}")

    if os.path.exists(project_dir):
        pattern = adapter.detect_project_structure(project_dir)
        print(f"Detected location pattern: {pattern.get('location_pattern')}")
        print(f"Detected naming convention: {pattern.get('naming_convention')}")
        print(f"Primary test directory: {pattern.get('primary_test_dir')}")
        print(f"Test directories found: {pattern.get('test_directories')}")

        # Expected: location_pattern should be 'tests_subdirectory'
        if pattern.get("location_pattern") == "tests_subdirectory":
            print("✅ Pattern detection FIXED - correctly detected tests_subdirectory")
        else:
            print(
                f"❌ Pattern detection still broken - got {pattern.get('location_pattern')}"
            )
    else:
        print(f"❌ Project directory not found: {project_dir}")
        return

    # Test 2: Test Path Generation
    print("\n2. Testing Test Path Generation:")
    if os.path.exists(source_file):
        test_directory = pattern.get(
            "primary_test_dir", os.path.join(project_dir, "tests")
        )
        test_path = adapter.generate_test_path(source_file, test_directory, pattern)
        print(f"Source file: {source_file}")
        print(f"Generated test path: {test_path}")

        # Expected: test should be in tests subdirectory
        expected_path = os.path.join(project_dir, "tests", "test_api_utils.py")
        if test_path == expected_path:
            print("✅ Test path generation FIXED - correct location")
        else:
            print(f"❌ Test path generation still broken")
            print(f"   Expected: {expected_path}")
            print(f"   Got:      {test_path}")
    else:
        print(f"❌ Source file not found: {source_file}")
        return

    # Test 3: Import Statement Generation
    print("\n3. Testing Import Statement Generation:")
    try:
        # Mock analysis data
        analysis = {
            "functions": [
                {"name": "get_current_session_id"},
                {"name": "set_current_session_id"},
            ],
            "classes": [],
        }

        import_statement = adapter._generate_import_statement(
            source_file, test_path, analysis
        )
        print(f"Generated import: {import_statement}")

        # Expected: should use full module path
        if "gojiberri" in import_statement and "ui.utils.api_utils" in import_statement:
            print("✅ Import generation FIXED - using full module path")
        elif "from ..api_utils import" in import_statement:
            print("✅ Import generation FIXED - using relative import")
        else:
            print(f"❌ Import generation still broken - not using proper module path")

    except Exception as e:
        print(f"❌ Error testing import generation: {e}")

    # Test 4: Template Generation
    print("\n4. Testing Template Generation:")
    try:
        template = adapter.generate_test_template(source_file, analysis, pattern)
        print("Template generated successfully")

        # Check if template contains proper imports
        if "from gojiberri" in template or "from .." in template:
            print("✅ Template generation FIXED - contains proper imports")
        else:
            print("❌ Template generation still broken - imports look wrong")
            print("First few lines of template:")
            print("\n".join(template.split("\n")[:10]))

    except Exception as e:
        print(f"❌ Error testing template generation: {e}")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_fixes()
