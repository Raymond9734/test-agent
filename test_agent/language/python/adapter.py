# test_agent/language/python/adapter.py

import os
import ast
import tempfile
import logging

from typing import List, Dict, Any, Optional


from ..base import LanguageAdapter, registry

# Configure logging
logger = logging.getLogger(__name__)


class PythonAdapter(LanguageAdapter):
    """Python language adapter with advanced AST-based analysis"""

    def __init__(self):
        """Initialize the Python adapter"""
        self._project_roots_cache = {}
        self._test_patterns_cache = {}

    @property
    def language_name(self) -> str:
        """Return the name of the language"""
        return "python"

    @property
    def file_extensions(self) -> List[str]:
        """Return the file extensions for this language"""
        return [".py"]

    @property
    def test_file_prefix(self) -> str:
        """Return the prefix for test files"""
        return "test_"

    @property
    def test_command(self) -> List[str]:
        """Return the command to run tests"""
        return ["python", "-m", "pytest"]

    def analyze_source_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python source file using AST"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse the file into an AST
            tree = ast.parse(content)

            # Extract information
            classes = []
            functions = []
            imports = []
            module_docstring = ast.get_docstring(tree)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    class_docstring = ast.get_docstring(node)

                    for m in node.body:
                        if isinstance(m, ast.FunctionDef):
                            method_docstring = ast.get_docstring(m)
                            method_args = self._extract_function_args(m)
                            method_returns = self._extract_function_returns(m)

                            methods.append(
                                {
                                    "name": m.name,
                                    "args": method_args,
                                    "returns": method_returns,
                                    "docstring": method_docstring,
                                    "lineno": m.lineno,
                                    "decorators": [
                                        d.id
                                        for d in m.decorator_list
                                        if isinstance(d, ast.Name)
                                    ],
                                }
                            )

                    # Skip internal/private classes
                    if not node.name.startswith("_"):
                        classes.append(
                            {
                                "name": node.name,
                                "methods": methods,
                                "docstring": class_docstring,
                                "lineno": node.lineno,
                                "decorators": [
                                    d.id
                                    for d in node.decorator_list
                                    if isinstance(d, ast.Name)
                                ],
                            }
                        )

                elif isinstance(node, ast.FunctionDef):
                    # Only include top-level functions
                    if isinstance(node.parent, ast.Module):
                        # Skip internal/private functions for testing
                        if not node.name.startswith("_"):
                            docstring = ast.get_docstring(node)
                            args = self._extract_function_args(node)
                            returns = self._extract_function_returns(node)

                            functions.append(
                                {
                                    "name": node.name,
                                    "args": args,
                                    "returns": returns,
                                    "docstring": docstring,
                                    "lineno": node.lineno,
                                    "decorators": [
                                        d.id
                                        for d in node.decorator_list
                                        if isinstance(d, ast.Name)
                                    ],
                                }
                            )

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.append(
                                {
                                    "module": name.name,
                                    "name": name.name.split(".")[-1],
                                    "alias": name.asname,
                                    "lineno": node.lineno,
                                }
                            )
                    else:  # ImportFrom
                        module = node.module or ""
                        for name in node.names:
                            imports.append(
                                {
                                    "module": module,
                                    "name": name.name,
                                    "alias": name.asname,
                                    "lineno": node.lineno,
                                }
                            )

            # Extract type annotations
            type_annotations = self._extract_type_annotations(tree)

            return {
                "file": file_path,
                "module_docstring": module_docstring,
                "classes": classes,
                "functions": functions,
                "imports": imports,
                "type_annotations": type_annotations,
            }

        except Exception as e:
            return {"file": file_path, "error": str(e)}

    def _extract_function_args(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract function arguments with type annotations"""
        args = []

        for arg in node.args.args:
            arg_info = {"name": arg.arg, "type": None}

            # Extract type annotation if present
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_info["type"] = arg.annotation.id
                elif isinstance(arg.annotation, ast.Attribute):
                    arg_info["type"] = self._extract_attribute(arg.annotation)
                elif isinstance(arg.annotation, ast.Subscript):
                    arg_info["type"] = self._extract_subscript(arg.annotation)

            args.append(arg_info)

        # Handle keyword arguments
        if node.args.kwarg:
            args.append(
                {"name": node.args.kwarg.arg, "type": "dict", "is_kwargs": True}
            )

        # Handle *args
        if node.args.vararg:
            args.append(
                {"name": node.args.vararg.arg, "type": "tuple", "is_vararg": True}
            )

        return args

    def _extract_function_returns(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract function return type annotation"""
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return node.returns.id
            elif isinstance(node.returns, ast.Attribute):
                return self._extract_attribute(node.returns)
            elif isinstance(node.returns, ast.Subscript):
                return self._extract_subscript(node.returns)

        # Try to find return annotation from return statements
        return_values = []
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value:
                if isinstance(child.value, ast.Name):
                    return_values.append(child.value.id)
                elif isinstance(child.value, ast.Constant):
                    return_values.append(type(child.value.value).__name__)

        if return_values:
            return ", ".join(set(return_values))

        return None

    def _extract_attribute(self, node: ast.Attribute) -> str:
        """Extract attribute name from AST node"""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._extract_attribute(node.value)}.{node.attr}"
        return node.attr

    def _extract_subscript(self, node: ast.Subscript) -> str:
        """Extract subscript expression from AST node"""
        value = ""
        if isinstance(node.value, ast.Name):
            value = node.value.id
        elif isinstance(node.value, ast.Attribute):
            value = self._extract_attribute(node.value)

        # Handle different Python versions
        if hasattr(node, "slice"):
            if isinstance(node.slice, ast.Index):
                if hasattr(node.slice, "value"):
                    slice_str = self._extract_slice_value(node.slice.value)
                    return f"{value}[{slice_str}]"
            elif isinstance(node.slice, ast.Name):
                return f"{value}[{node.slice.id}]"
            elif isinstance(node.slice, ast.Constant):
                return f"{value}[{node.slice.value}]"
            elif isinstance(node.slice, ast.Tuple):
                elts = []
                for elt in node.slice.elts:
                    if isinstance(elt, ast.Name):
                        elts.append(elt.id)
                    elif isinstance(elt, ast.Constant):
                        elts.append(str(elt.value))
                return f"{value}[{', '.join(elts)}]"

        return f"{value}[...]"

    def _extract_slice_value(self, node: ast.AST) -> str:
        """Extract string representation of a slice value"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._extract_attribute(node)
        elif isinstance(node, ast.Tuple):
            elts = []
            for elt in node.elts:
                if isinstance(elt, ast.Name):
                    elts.append(elt.id)
                elif isinstance(elt, ast.Attribute):
                    elts.append(self._extract_attribute(elt))
            return ", ".join(elts)
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return "..."

    def _extract_type_annotations(self, tree: ast.Module) -> List[Dict[str, Any]]:
        """Extract variable type annotations from AST"""
        annotations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and node.annotation:
                if isinstance(node.target, ast.Name):
                    anno_type = None
                    if isinstance(node.annotation, ast.Name):
                        anno_type = node.annotation.id
                    elif isinstance(node.annotation, ast.Attribute):
                        anno_type = self._extract_attribute(node.annotation)
                    elif isinstance(node.annotation, ast.Subscript):
                        anno_type = self._extract_subscript(node.annotation)

                    if anno_type:
                        annotations.append(
                            {
                                "name": node.target.id,
                                "type": anno_type,
                                "lineno": node.lineno,
                            }
                        )

        return annotations

    def analyze_test_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Python test file to extract test cases and framework"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse the file into an AST
            tree = ast.parse(content)

            test_functions = []
            test_classes = []
            imports = []
            framework = self._detect_test_framework_from_content(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith("test_"):
                        docstring = ast.get_docstring(node)
                        test_functions.append(
                            {
                                "name": node.name,
                                "docstring": docstring,
                                "lineno": node.lineno,
                            }
                        )

                elif isinstance(node, ast.ClassDef):
                    if node.name.startswith("Test"):
                        class_docstring = ast.get_docstring(node)
                        methods = []

                        for m in node.body:
                            if isinstance(m, ast.FunctionDef) and (
                                m.name.startswith("test_")
                                or (
                                    framework == "unittest"
                                    and m.name.startswith("test")
                                )
                            ):
                                method_docstring = ast.get_docstring(m)
                                methods.append(
                                    {
                                        "name": m.name,
                                        "docstring": method_docstring,
                                        "lineno": m.lineno,
                                    }
                                )

                        test_classes.append(
                            {
                                "name": node.name,
                                "methods": methods,
                                "docstring": class_docstring,
                                "lineno": node.lineno,
                            }
                        )

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.append(name.name)
                    else:  # ImportFrom
                        module = node.module or ""
                        for name in node.names:
                            imports.append(f"{module}.{name.name}")

            # Find source file being tested
            project_root = self._find_project_root(file_path)
            source_file = self.find_corresponding_source(file_path, project_root)

            return {
                "file": file_path,
                "source_file": source_file,
                "test_functions": test_functions,
                "test_classes": test_classes,
                "imports": imports,
                "framework": framework,
            }

        except Exception as e:
            return {"file": file_path, "error": str(e)}

    def _safe_read_file(self, file_path: str) -> Optional[str]:
        """
        Safely read a file with multiple encoding attempts.

        Args:
            file_path: Path to the file to read

        Returns:
            File content as string, or None if reading fails
        """
        # Try different encodings
        encodings_to_try = ["utf-8", "latin-1", "cp1252", "iso-8859-1", "utf-16"]

        for encoding in encodings_to_try:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.debug(f"Error reading {file_path} with {encoding}: {str(e)}")
                continue

        # If all encodings fail, try to read as binary and detect encoding
        try:
            import chardet

            with open(file_path, "rb") as f:
                raw_data = f.read()

            detected = chardet.detect(raw_data)
            if detected["encoding"]:
                try:
                    return raw_data.decode(detected["encoding"])
                except Exception:
                    pass
        except ImportError:
            # chardet not available
            pass
        except Exception:
            pass

        logger.debug(f"Could not read file {file_path} with any encoding")
        return None


    def check_is_test_file(self, file_path: str) -> bool:
        """Check if a file is a test file"""
        file_name = os.path.basename(file_path)

        # Don't consider __init__.py as a test file
        if file_name == "__init__.py":
            return False

        # Check if it's a Python file first
        if not file_name.endswith(".py"):
            return False

        # Common test file patterns for Python
        if (
            file_name.startswith("test_")
            or file_name.endswith("_test.py")
            or (
                file_name.startswith("test") and file_name != "test.py"
            )  # handles testXXX.py
        ):
            return True

        # Check if file is in a directory with "test" in the name
        dir_path = os.path.dirname(file_path).lower()
        if "test" in dir_path:
            # If it's in a test directory, it's likely a test file even without test prefix
            return True

        return False

    def detect_project_structure(self, project_dir: str) -> Dict[str, Any]:
        """Detect project structure, test patterns, and framework information"""
        # Check cache first
        project_dir = os.path.abspath(project_dir)
        if project_dir in self._test_patterns_cache:
            return self._test_patterns_cache[project_dir]

        # Analyze project structure
        test_directories = []
        test_files = []

        # Look for test directories
        for root, dirs, _ in os.walk(project_dir):
            for dir_name in dirs:
                if dir_name.lower() in ["test", "tests", "testing"]:
                    test_directories.append(os.path.join(root, dir_name))

        # Look for test files
        for root, _, files in os.walk(project_dir):
            for file_name in files:
                if self.check_is_test_file(file_name):
                    test_files.append(os.path.join(root, file_name))

        # Analyze test location patterns with preference for tests_subdirectory
        location_pattern = "tests_subdirectory"  # Default preference
        naming_convention = "test_prefix"  # Default

        if test_files:
            # Count location patterns
            location_counts = {
                "same_directory": 0,
                "tests_directory": 0,
                "tests_subdirectory": 0,
                "mirror_under_tests": 0,
            }

            # Count naming patterns
            naming_counts = {"test_prefix": 0, "suffix_test": 0}

            for test_file in test_files:
                # Check location pattern
                rel_path = os.path.relpath(test_file, project_dir)
                parts = rel_path.split(os.sep)
                test_dir_name = os.path.basename(os.path.dirname(test_file)).lower()

                # Priority: Check if test file is in a tests subdirectory
                if test_dir_name in ["tests", "test", "testing"]:
                    location_counts["tests_subdirectory"] += 1
                elif parts[0].lower() in ["tests", "test"]:
                    if len(parts) > 2:
                        location_counts["mirror_under_tests"] += 1
                    else:
                        location_counts["tests_directory"] += 1
                else:
                    location_counts["same_directory"] += 1

                # Check naming pattern
                file_name = os.path.basename(test_file)
                if file_name.startswith("test_"):
                    naming_counts["test_prefix"] += 1
                elif "_test" in file_name:
                    naming_counts["suffix_test"] += 1

            # Determine primary patterns - prioritize tests_subdirectory
            if location_counts["tests_subdirectory"] > 0:
                location_pattern = "tests_subdirectory"
            elif location_counts:
                location_pattern = max(location_counts.items(), key=lambda x: x[1])[0]

            if naming_counts:
                naming_convention = max(naming_counts.items(), key=lambda x: x[1])[0]

        # If we found test directories but no clear pattern, prefer tests_subdirectory
        if test_directories and location_pattern not in [
            "tests_subdirectory",
            "mirror_under_tests",
        ]:
            # Check if any test directories are subdirectories of source dirs
            for test_dir in test_directories:
                test_parent = os.path.dirname(test_dir)
                if test_parent != project_dir:  # It's a subdirectory somewhere
                    location_pattern = "tests_subdirectory"
                    break

        # Detect test framework
        framework = "pytest"  # Default

        # Check first few test files for framework indicators
        files_to_check = test_files[:5]  # Check first 5 test files

        for test_file in files_to_check:
            try:
                # Use safe file reading
                content = self._safe_read_file(test_file)
                if content is None:
                    logger.debug(
                        f"Skipping framework detection for unreadable file: {test_file}"
                    )
                    continue

                detected_framework = self._detect_test_framework_from_content(content)
                if detected_framework == "unittest":
                    framework = "unittest"
                    break
            except Exception as e:
                logger.debug(f"Error detecting framework in {test_file}: {str(e)}")
                continue

        # Determine primary test directory - prefer existing tests subdirectory
        primary_test_dir = None

        # First, look for tests directories that are subdirectories of the project dir
        for test_dir in test_directories:
            if os.path.dirname(test_dir) == project_dir and os.path.basename(
                test_dir
            ).lower() in ["tests", "test"]:
                primary_test_dir = test_dir
                break

        # If no direct subdirectory found, use the first test directory
        if not primary_test_dir and test_directories:
            primary_test_dir = test_directories[0]

        # Default fallback
        if not primary_test_dir:
            primary_test_dir = os.path.join(project_dir, "tests")

        # Create result
        result = {
            "test_directories": test_directories,
            "test_files": test_files,
            "location_pattern": location_pattern,
            "naming_convention": naming_convention,
            "framework": framework,
            "primary_test_dir": primary_test_dir,
        }

        # Cache result
        self._test_patterns_cache[project_dir] = result

        return result

    def generate_test_path(
        self,
        source_file: str,
        test_directory: str,
        pattern: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate the proper path for a test file given a source file and pattern"""
        # Ensure absolute paths
        source_file = os.path.abspath(source_file)
        test_directory = os.path.abspath(test_directory)

        # First, check if test already exists
        project_root = self._find_project_root(source_file)
        existing_test = self.find_corresponding_test(source_file, project_root)
        if existing_test and os.path.exists(existing_test):
            return existing_test

        # Use provided pattern or detect it
        if pattern is None:
            pattern = self.detect_project_structure(project_root)

        # Extract filename components
        source_name = os.path.basename(source_file)
        source_base, source_ext = os.path.splitext(source_name)

        # Determine test filename based on naming convention
        naming_convention = pattern.get("naming_convention", "test_prefix")

        if naming_convention == "test_prefix":
            test_filename = f"test_{source_name}"
        else:  # suffix_test
            test_filename = f"{source_base}_test{source_ext}"

        # Generate path based on location pattern
        location_pattern = pattern.get("location_pattern", "tests_subdirectory")

        logger.debug(f"Generating test path for {source_file}")
        logger.debug(f"Location pattern: {location_pattern}")
        logger.debug(f"Test directory: {test_directory}")

        if location_pattern == "tests_subdirectory":
            # Create tests subdirectory in the same directory as the source file
            source_dir = os.path.dirname(source_file)
            test_dir = os.path.join(source_dir, "tests")
            os.makedirs(test_dir, exist_ok=True)
            result_path = os.path.join(test_dir, test_filename)
            logger.debug(f"tests_subdirectory: {result_path}")
            return result_path

        elif location_pattern == "same_directory":
            result_path = os.path.join(os.path.dirname(source_file), test_filename)
            logger.debug(f"same_directory: {result_path}")
            return result_path

        elif location_pattern == "mirror_under_tests":
            # Get path relative to project root
            rel_path = os.path.relpath(source_file, project_root)
            rel_dir = os.path.dirname(rel_path)

            # Create mirror directory under test_directory
            mirror_dir = os.path.join(test_directory, rel_dir)
            os.makedirs(mirror_dir, exist_ok=True)
            result_path = os.path.join(mirror_dir, test_filename)
            logger.debug(f"mirror_under_tests: {result_path}")
            return result_path

        else:  # tests_directory or fallback
            source_dir = os.path.dirname(source_file)

            # If the source file is not in the project root, prefer tests_subdirectory
            if source_dir != project_root:
                test_dir = os.path.join(source_dir, "tests")
                os.makedirs(test_dir, exist_ok=True)
                result_path = os.path.join(test_dir, test_filename)
                logger.debug(f"fallback to tests_subdirectory: {result_path}")
                return result_path

            # Otherwise, use the global test directory for root-level files
            os.makedirs(test_directory, exist_ok=True)
            result_path = os.path.join(test_directory, test_filename)
            logger.debug(f"global test directory: {result_path}")
            return result_path

    def _detect_test_framework_from_content(self, content: str) -> str:
        """Detect which test framework is being used in the content"""
        # Handle None or empty content
        if not content:
            return "pytest"  # Default fallback

        pytest_indicators = [
            "import pytest",
            "from pytest",
            "@pytest",
            "pytest.fixture",
            "pytest.mark",
            "def test_",
        ]

        unittest_indicators = [
            "import unittest",
            "from unittest",
            "TestCase",
            "self.assert",
            "setUp",
            "tearDown",
        ]

        pytest_score = sum(1 for indicator in pytest_indicators if indicator in content)
        unittest_score = sum(
            1 for indicator in unittest_indicators if indicator in content
        )

        return "pytest" if pytest_score >= unittest_score else "unittest"

    def _generate_import_statement(
        self, source_file: str, test_path: str, analysis: Dict[str, Any]
    ) -> str:
        """
        Generate the appropriate import statement for the test file.

        Args:
            source_file: Path to the source file being tested
            test_path: Path where the test file will be created
            analysis: Analysis results containing classes and functions

        Returns:
            Import statement string
        """
        # Extract classes and functions to import
        classes = [cls["name"] for cls in analysis.get("classes", [])]
        functions = [func["name"] for func in analysis.get("functions", [])]

        # Combine all imports
        imports_to_make = classes + functions

        # Get module name from source file
        source_name = os.path.splitext(os.path.basename(source_file))[0]

        # Find the Python package root by looking for the outermost __init__.py
        def find_package_root(file_path):
            """Find the root of the Python package by traversing up directories."""
            current_dir = os.path.dirname(os.path.abspath(file_path))
            package_parts = []

            while current_dir and current_dir != "/":
                if os.path.exists(os.path.join(current_dir, "__init__.py")):
                    # This directory is part of a package
                    package_parts.insert(0, os.path.basename(current_dir))
                    current_dir = os.path.dirname(current_dir)
                else:
                    # No __init__.py found, stop here
                    break

            return package_parts

        # Get the package path from the source file
        try:
            package_parts = find_package_root(source_file)

            if package_parts:
                # Build the full module path
                module_path = ".".join(package_parts + [source_name])

                # Generate import statement
                if imports_to_make:
                    import_list = ", ".join(imports_to_make)
                    return f"from {module_path} import {import_list}"
                else:
                    # Import the module itself if no specific items found
                    return f"from {module_path} import *"
            else:
                # Fallback to relative import if no package structure found
                # Calculate the relative path from test to source
                test_dir = os.path.dirname(test_path)
                source_dir = os.path.dirname(source_file)

                # Check if test is in a subdirectory of source directory
                if test_dir.startswith(source_dir):
                    # Test is in a subdirectory (like tests/), use parent import
                    if imports_to_make:
                        import_list = ", ".join(imports_to_make)
                        return f"from ..{source_name} import {import_list}"
                    else:
                        return f"from ..{source_name} import *"
                else:
                    # Same level or other structure
                    if imports_to_make:
                        import_list = ", ".join(imports_to_make)
                        return f"from .{source_name} import {import_list}"
                    else:
                        return f"from .{source_name} import *"

        except Exception as e:
            logger.debug(f"Error generating import statement: {str(e)}")
            # Ultimate fallback
            if imports_to_make:
                import_list = ", ".join(imports_to_make)
                return f"from {source_name} import {import_list}"
            else:
                return f"from {source_name} import *"

    def generate_test_template(
        self,
        source_file: str,
        analysis: Dict[str, Any],
        pattern: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a test template based on source analysis and project patterns"""
        # Use provided pattern or detect it
        if pattern is None:
            project_dir = self._find_project_root(source_file)
            pattern = self.detect_project_structure(project_dir)

        # Generate test path to determine import structure
        test_directory = pattern.get("primary_test_dir") or os.path.join(
            self._find_project_root(source_file), "tests"
        )
        test_path = self.generate_test_path(source_file, test_directory, pattern)

        framework = pattern.get("framework", "pytest")
        naming_convention = pattern.get("naming_convention", "test_prefix")

        if framework == "unittest":
            return self._get_unittest_template(
                source_file, test_path, analysis, naming_convention
            )
        else:
            return self._get_pytest_template(
                source_file, test_path, analysis, naming_convention
            )

    def _get_pytest_template(
        self,
        source_file: str,
        test_path: str,
        analysis: Dict[str, Any],
        naming_convention: str,
    ) -> str:
        """Generate a pytest-style test template"""

        # Get module name
        module_name = os.path.splitext(os.path.basename(source_file))[0]

        # Generate import statement
        import_statement = self._generate_import_statement(
            source_file, test_path, analysis
        )

        template = f"""# Generated test file for {module_name}
    import pytest
    from unittest.mock import Mock, patch, MagicMock
    {import_statement}


    """

        # Add test functions for classes
        for cls in analysis.get("classes", []):
            if naming_convention in ["test_classes", "mixed"]:
                template += f"""
    class Test{cls['name']}:
        def test_{cls['name'].lower()}_initialization(self):
            \"\"\"Test that {cls['name']} can be initialized.\"\"\"
            # TODO: Add appropriate initialization parameters
            # instance = {cls['name']}()
            # assert instance is not None
            pass
    """
                # Add tests for methods
                for method in cls.get("methods", []):
                    if method["name"] != "__init__":
                        template += f"""
        def test_{method['name']}(self):
            \"\"\"Test the {method['name']} method of {cls['name']}.\"\"\"
            # TODO: Create instance and test {method['name']}
            # instance = {cls['name']}()
            # result = instance.{method['name']}()
            # assert result == expected_result
            pass
    """
            else:
                # Add function-style tests for class methods
                template += f"""
    def test_{cls['name'].lower()}_initialization():
        \"\"\"Test that {cls['name']} can be initialized.\"\"\"
        # TODO: Add appropriate initialization parameters
        # instance = {cls['name']}()
        # assert instance is not None
        pass
    """
                # Add tests for methods
                for method in cls.get("methods", []):
                    if method["name"] != "__init__":
                        template += f"""
    def test_{cls['name'].lower()}_{method['name']}():
        \"\"\"Test the {method['name']} method of {cls['name']}.\"\"\"
        # TODO: Create instance and test {method['name']}
        # instance = {cls['name']}()
        # result = instance.{method['name']}()
        # assert result == expected_result
        pass
    """

        # Add test functions for standalone functions
        for func in analysis.get("functions", []):
            template += f"""
    def test_{func['name']}():
        \"\"\"Test the {func['name']} function.\"\"\"
        # TODO: Add proper test for {func['name']}
        # result = {func['name']}()
        # assert result == expected_value
        pass
    """

        # Add parametrized test example if there are functions
        if analysis.get("functions") or analysis.get("classes"):
            template += """

    # Example parametrized test - remove if not needed
    @pytest.mark.parametrize("input_value,expected", [
        ("test_input", "expected_output"),
        # Add more test cases here
    ])
    def test_parametrized_example(input_value, expected):
        \"\"\"Example of parametrized test.\"\"\"
        # TODO: Replace with actual test logic
        # result = your_function(input_value)
        # assert result == expected
        pass
    """

        return template

    def _get_unittest_template(
        self,
        source_file: str,
        test_path: str,
        analysis: Dict[str, Any],
        naming_convention: str,
    ) -> str:
        """Generate a unittest-style test template"""

        # Get module name
        module_name = os.path.splitext(os.path.basename(source_file))[0]

        # Generate import statement
        import_statement = self._generate_import_statement(
            source_file, test_path, analysis
        )

        template = f"""# Generated test file for {module_name}
    import unittest
    from unittest.mock import Mock, patch, MagicMock
    {import_statement}


    """

        # Add test classes for classes
        for cls in analysis.get("classes", []):
            template += f"""class Test{cls['name']}(unittest.TestCase):
        def test_{cls['name'].lower()}_initialization(self):
            \"\"\"Test that {cls['name']} can be initialized.\"\"\"
            # TODO: Add appropriate initialization parameters
            # instance = {cls['name']}()
            # self.assertIsNotNone(instance)
            pass
    """
            # Add tests for methods
            for method in cls.get("methods", []):
                if method["name"] != "__init__":
                    template += f"""
        def test_{method['name']}(self):
            \"\"\"Test the {method['name']} method of {cls['name']}.\"\"\"
            # TODO: Create instance and test {method['name']}
            # instance = {cls['name']}()
            # result = instance.{method['name']}()
            # self.assertEqual(result, expected_result)
            pass
    """
            template += "\n"

        # Add test class for standalone functions
        if analysis.get("functions"):
            template += (
                f"""class Test{module_name.capitalize()}Functions(unittest.TestCase):"""
            )

            # Add test methods for standalone functions
            for func in analysis.get("functions", []):
                template += f"""
        def test_{func['name']}(self):
            \"\"\"Test the {func['name']} function.\"\"\"
            # TODO: Add proper test for {func['name']}
            # result = {func['name']}()
            # self.assertEqual(result, expected_value)
            pass
    """
            template += "\n"

        template += """
    if __name__ == '__main__':
        unittest.main()
    """

        return template

    def find_corresponding_source(
        self, test_file: str, project_dir: str
    ) -> Optional[str]:
        """Find the source file that a test file is testing"""
        if not self.check_is_test_file(test_file):
            return None

        file_name = os.path.basename(test_file)
        # file_ext = os.path.splitext(file_name)[1]

        # Remove test prefix/suffix based on language conventions
        if file_name.startswith("test_"):
            source_name = file_name[5:]  # Remove "test_"
        elif file_name.endswith("_test.py"):
            source_name = file_name[:-8] + ".py"  # Remove "_test.py" and add ".py"
        else:
            # Can't determine source file name
            return None

        # First look in the same directory
        test_dir = os.path.dirname(test_file)
        potential_source = os.path.join(test_dir, source_name)
        if os.path.exists(potential_source):
            return potential_source

        # Then look in parent directory if test is in a test directory
        test_dir_name = os.path.basename(test_dir).lower()
        if test_dir_name in ["test", "tests", "testing"]:
            parent_dir = os.path.dirname(test_dir)
            potential_source = os.path.join(parent_dir, source_name)
            if os.path.exists(potential_source):
                return potential_source

        # Look for source files in the project with the same base name
        source_base = os.path.splitext(source_name)[0]
        for root, _, files in os.walk(project_dir):
            for file in files:
                if file == source_name:
                    return os.path.join(root, file)
                elif os.path.splitext(file)[0] == source_base and file.endswith(".py"):
                    return os.path.join(root, file)

        return None

    def find_corresponding_test(
        self, source_file: str, project_dir: str
    ) -> Optional[str]:
        """
        Find the test file for a given source file.

        Args:
            source_file: Path to the source file
            project_dir: Root directory of the project

        Returns:
            Optional[str]: Path to the corresponding test file, or None if not found
        """
        # Skip if it's already a test file
        if self.check_is_test_file(source_file):
            return None

        file_name = os.path.basename(source_file)
        file_base, file_ext = os.path.splitext(file_name)
        source_dir = os.path.dirname(source_file)

        # Ensure we're only looking for .py test files
        if file_ext != ".py":
            logger.debug(f"Skipping non-Python file: {source_file}")
            return None

        # Generate possible test file names (ensuring they end with .py)
        test_names = [
            f"test_{file_name}",  # test_file.py
            f"{file_base}_test.py",  # file_test.py
        ]

        # Check in different locations based on common patterns, prioritizing tests subdirectory
        potential_locations = [
            os.path.join(source_dir, "tests"),  # tests/ subdirectory (PRIORITY)
            source_dir,  # Same directory
            os.path.join(source_dir, "test"),  # test/ subdirectory
            os.path.join(project_dir, "tests"),  # tests/ in project root
            os.path.join(project_dir, "test"),  # test/ in project root
        ]

        # Add mirror pattern: directory structure under tests/
        rel_path = os.path.relpath(source_dir, project_dir)
        if rel_path != ".":
            potential_locations.append(os.path.join(project_dir, "tests", rel_path))
            potential_locations.append(os.path.join(project_dir, "test", rel_path))

        # Check each potential location
        for test_name in test_names:
            for location in potential_locations:
                if os.path.exists(location):
                    test_path = os.path.join(location, test_name)
                    if os.path.exists(test_path):
                        logger.debug(f"Found test file at {test_path}")
                        return test_path

        # If no exact match found, try a more comprehensive search but exclude __pycache__
        logger.debug(
            f"No direct match found, performing comprehensive search for {file_base}"
        )

        # Define directories to exclude
        excluded_dirs = [
            "__pycache__",
            ".git",
            ".idea",
            ".vscode",
            "node_modules",
            "venv",
            "env",
        ]

        for root, dirs, files in os.walk(project_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in excluded_dirs]

            # Only check directories with "test" in the name
            if "test" in root.lower() or "tests" in root.lower():
                for file in files:
                    # Ensure file is a Python file
                    if not file.endswith(".py"):
                        continue

                    # Check if file name matches test patterns
                    if (file.startswith("test_") and file_base in file) or (
                        file.endswith("_test.py") and file_base in file
                    ):
                        try:
                            test_file_path = os.path.join(root, file)
                            logger.debug(
                                f"Checking potential test file: {test_file_path}"
                            )

                            # Try different encodings when reading files
                            for encoding in [
                                "utf-8",
                                "latin-1",
                                "cp1252",
                                "iso-8859-1",
                            ]:
                                try:
                                    with open(
                                        test_file_path, "r", encoding=encoding
                                    ) as f:
                                        content = f.read()
                                        # Look for imports or references to the source file
                                        if file_base in content or file_name in content:
                                            logger.debug(
                                                f"Found matching test file: {test_file_path}"
                                            )
                                            return test_file_path
                                    # If we get here, the file was read successfully
                                    break
                                except UnicodeDecodeError:
                                    # Try the next encoding
                                    continue
                                except Exception as e:
                                    # Any other error, just log and continue
                                    logger.debug(f"Error reading file {file}: {str(e)}")
                        except Exception as e:
                            logger.debug(f"Error processing file {file}: {str(e)}")

        logger.debug(f"No corresponding test found for {source_file}")
        return None

    def get_environment_location(self) -> str:
        """
        Get the recommended location for the virtual environment
        based on the project structure
        """
        # Use system temp directory for virtual environment
        base_dir = tempfile.gettempdir()
        venv_dir = os.path.join(base_dir, "test_agent_venv")
        return venv_dir


# Add parent references to AST nodes for better analysis
# This modifies the ast module to track parent relationships
def _add_parent_references(tree):
    """Add parent references to all nodes in the AST"""
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    return tree


# Patch ast.parse to add parent references
original_parse = ast.parse


def patched_parse(*args, **kwargs):
    """Patched version of ast.parse that adds parent references"""
    tree = original_parse(*args, **kwargs)
    return _add_parent_references(tree)


# Apply the patch
ast.parse = patched_parse

# Register the adapter
registry.register([".py"], PythonAdapter)
