# test_agent/language/python/parser.py

import os
import ast
import logging
from typing import Dict, List, Any, Optional, Set, Union

# Configure logging
logger = logging.getLogger(__name__)


class PythonParser:
    """
    Parser for Python source code.

    Provides methods to extract structure and content from Python files
    using AST (Abstract Syntax Tree) analysis.
    """

    def __init__(self):
        """Initialize the Python parser."""
        self._ast_cache = {}  # Cache parsed AST trees

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a Python file and extract its structure.

        Args:
            file_path: Path to the Python file to parse

        Returns:
            Dictionary with parsed file structure
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}

            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            tree = self._get_ast(file_path, content)

            # Extract file structure
            return self._analyze_ast(tree, file_path)

        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {str(e)}")
            return {"error": f"Syntax error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            return {"error": str(e)}

    def parse_content(self, content: str, filename: str = "<string>") -> Dict[str, Any]:
        """
        Parse Python code content directly.

        Args:
            content: Python code as string
            filename: Optional virtual filename for the content

        Returns:
            Dictionary with parsed content structure
        """
        try:
            # Parse AST
            tree = ast.parse(content, filename=filename)

            # Add parent references to AST nodes
            tree = self._add_parent_references(tree)

            # Extract structure
            return self._analyze_ast(tree, filename)

        except SyntaxError as e:
            logger.error(f"Syntax error in content: {str(e)}")
            return {"error": f"Syntax error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error parsing content: {str(e)}")
            return {"error": str(e)}

    def _get_ast(self, file_path: str, content: str) -> ast.Module:
        """
        Get AST for file content with caching.

        Args:
            file_path: Path to the file (used as cache key)
            content: File content to parse

        Returns:
            AST Module node
        """
        # Check cache first
        if file_path in self._ast_cache:
            return self._ast_cache[file_path]

        # Parse content
        tree = ast.parse(content, filename=file_path)

        # Add parent references to AST nodes
        tree = self._add_parent_references(tree)

        # Cache the result
        self._ast_cache[file_path] = tree

        return tree

    def _add_parent_references(self, tree: ast.Module) -> ast.Module:
        """
        Add parent references to AST nodes for easier traversal.

        Args:
            tree: AST tree to modify

        Returns:
            Modified AST tree with parent references
        """
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node
        return tree

    def _analyze_ast(self, tree: ast.Module, file_path: str) -> Dict[str, Any]:
        """
        Analyze AST to extract code structure.

        Args:
            tree: AST tree to analyze
            file_path: Path to the file being analyzed

        Returns:
            Dictionary with code structure
        """
        # Initialize result
        result = {
            "file": file_path,
            "type": "module",
            "imports": [],
            "classes": [],
            "functions": [],
            "variables": [],
            "module_docstring": None,
        }

        # Get module docstring
        result["module_docstring"] = ast.get_docstring(tree)

        # Process top-level nodes
        for node in tree.body:
            # Extract imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports = self._extract_imports(node)
                result["imports"].extend(imports)

            # Extract classes
            elif isinstance(node, ast.ClassDef):
                class_info = self._extract_class(node)
                result["classes"].append(class_info)

            # Extract functions
            elif isinstance(node, ast.FunctionDef):
                func_info = self._extract_function(node)
                result["functions"].append(func_info)

            # Extract variables
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                vars_info = self._extract_variables(node)
                result["variables"].extend(vars_info)

        return result

    def _extract_imports(
        self, node: Union[ast.Import, ast.ImportFrom]
    ) -> List[Dict[str, Any]]:
        """
        Extract import information from import nodes.

        Args:
            node: Import or ImportFrom AST node

        Returns:
            List of import dictionaries
        """
        imports = []

        if isinstance(node, ast.Import):
            # Regular imports (import x, import y)
            for name in node.names:
                imports.append(
                    {
                        "type": "import",
                        "name": name.name,
                        "alias": name.asname,
                        "lineno": node.lineno,
                    }
                )
        else:  # ImportFrom
            # From imports (from x import y)
            module = node.module or ""
            for name in node.names:
                imports.append(
                    {
                        "type": "import_from",
                        "module": module,
                        "name": name.name,
                        "alias": name.asname,
                        "lineno": node.lineno,
                        "level": node.level,  # Relative import level
                    }
                )

        return imports

    def _extract_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """
        Extract class information from a ClassDef node.

        Args:
            node: ClassDef AST node

        Returns:
            Dictionary with class information
        """
        class_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "methods": [],
            "class_variables": [],
            "bases": [],
            "decorators": [],
            "lineno": node.lineno,
        }

        # Extract base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                class_info["bases"].append(base.id)
            elif isinstance(base, ast.Attribute):
                class_info["bases"].append(self._extract_attribute(base))

        # Extract decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                class_info["decorators"].append(decorator.id)
            elif isinstance(decorator, ast.Call) and isinstance(
                decorator.func, ast.Name
            ):
                class_info["decorators"].append(decorator.func.id)
            elif isinstance(decorator, ast.Attribute):
                class_info["decorators"].append(self._extract_attribute(decorator))

        # Extract class body elements
        for item in node.body:
            # Methods
            if isinstance(item, ast.FunctionDef):
                method_info = self._extract_function(item, is_method=True)
                class_info["methods"].append(method_info)

            # Class variables
            elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                vars_info = self._extract_variables(item, class_var=True)
                class_info["class_variables"].extend(vars_info)

        return class_info

    def _extract_function(
        self, node: ast.FunctionDef, is_method: bool = False
    ) -> Dict[str, Any]:
        """
        Extract function/method information from a FunctionDef node.

        Args:
            node: FunctionDef AST node
            is_method: Whether this function is a class method

        Returns:
            Dictionary with function information
        """
        func_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "args": [],
            "returns": None,
            "decorators": [],
            "lineno": node.lineno,
            "is_method": is_method,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
        }

        # Extract arguments
        for arg in node.args.args:
            arg_info = {"name": arg.arg, "type": None}
            if arg.annotation:
                arg_info["type"] = self._extract_annotation(arg.annotation)
            func_info["args"].append(arg_info)

        # Extract *args argument
        if node.args.vararg:
            vararg = {"name": f"*{node.args.vararg.arg}", "type": None}
            if node.args.vararg.annotation:
                vararg["type"] = self._extract_annotation(node.args.vararg.annotation)
            func_info["args"].append(vararg)

        # Extract **kwargs argument
        if node.args.kwarg:
            kwarg = {"name": f"**{node.args.kwarg.arg}", "type": None}
            if node.args.kwarg.annotation:
                kwarg["type"] = self._extract_annotation(node.args.kwarg.annotation)
            func_info["args"].append(kwarg)

        # Extract return type annotation
        if node.returns:
            func_info["returns"] = self._extract_annotation(node.returns)

        # Extract decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                func_info["decorators"].append(decorator.id)
            elif isinstance(decorator, ast.Call) and isinstance(
                decorator.func, ast.Name
            ):
                func_info["decorators"].append(decorator.func.id)
            elif isinstance(decorator, ast.Attribute):
                func_info["decorators"].append(self._extract_attribute(decorator))

        return func_info

    def _extract_variables(
        self, node: Union[ast.Assign, ast.AnnAssign], class_var: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Extract variable assignments.

        Args:
            node: Assign or AnnAssign AST node
            class_var: Whether this is a class variable

        Returns:
            List of variable dictionaries
        """
        variables = []

        if isinstance(node, ast.Assign):
            # Extract targets
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_info = {
                        "name": target.id,
                        "type": None,
                        "lineno": node.lineno,
                        "class_var": class_var,
                    }
                    variables.append(var_info)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            var_info = {
                                "name": elt.id,
                                "type": None,
                                "lineno": node.lineno,
                                "class_var": class_var,
                            }
                            variables.append(var_info)

        elif isinstance(node, ast.AnnAssign):
            # Annotated assignment (x: type = value)
            if isinstance(node.target, ast.Name):
                var_info = {
                    "name": node.target.id,
                    "type": self._extract_annotation(node.annotation),
                    "lineno": node.lineno,
                    "class_var": class_var,
                }
                variables.append(var_info)

        return variables

    def _extract_annotation(self, node: ast.AST) -> Optional[str]:
        """
        Extract type annotation from an AST node.

        Args:
            node: AST node containing type annotation

        Returns:
            String representation of the type annotation
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._extract_attribute(node)
        elif isinstance(node, ast.Subscript):
            value = self._extract_annotation(node.value)

            # Handle subscript (e.g., List[int])
            if hasattr(node, "slice"):
                # Python 3.9+
                if isinstance(node.slice, ast.Index):
                    if hasattr(node.slice, "value"):
                        # Python 3.8-
                        slice_str = self._extract_subscript_slice(node.slice.value)
                        return f"{value}[{slice_str}]"
                    else:
                        # Python 3.9+
                        slice_str = self._extract_subscript_slice(node.slice)
                        return f"{value}[{slice_str}]"
                else:
                    # Direct slice in Python 3.9+
                    slice_str = self._extract_subscript_slice(node.slice)
                    return f"{value}[{slice_str}]"

            return f"{value}[...]"
        elif isinstance(node, ast.Constant):
            # Python 3.8+ uses Constant
            return repr(node.value)
        elif isinstance(node, (ast.Str, ast.Num, ast.Bytes, ast.NameConstant)):
            # Python 3.7 and earlier
            if isinstance(node, ast.Str):
                return repr(node.s)
            elif isinstance(node, ast.Num):
                return repr(node.n)
            elif isinstance(node, ast.Bytes):
                return repr(node.s)
            elif isinstance(node, ast.NameConstant):
                return repr(node.value)

        # Default for unknown annotation types
        return "Any"

    def _extract_attribute(self, node: ast.Attribute) -> str:
        """
        Extract full attribute name (e.g., module.submodule.attr).

        Args:
            node: Attribute AST node

        Returns:
            Full attribute name as string
        """
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._extract_attribute(node.value)}.{node.attr}"
        return node.attr

    def _extract_subscript_slice(self, node: ast.AST) -> str:
        """
        Extract the slice part of a subscript (e.g., the 'int' in List[int]).

        Args:
            node: AST node for the slice part

        Returns:
            String representation of the slice
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._extract_attribute(node)
        elif isinstance(node, ast.Tuple):
            elements = []
            for elt in node.elts:
                elements.append(self._extract_annotation(elt))
            return ", ".join(elements)
        elif isinstance(node, ast.Subscript):
            # Nested subscript (e.g., Dict[str, List[int]])
            value = self._extract_annotation(node.value)
            slice_str = self._extract_subscript_slice(node.slice)
            return f"{value}[{slice_str}]"
        elif isinstance(node, ast.Constant):
            # Python 3.8+
            return repr(node.value)
        elif isinstance(node, (ast.Str, ast.Num, ast.Bytes, ast.NameConstant)):
            # Python 3.7 and earlier
            if isinstance(node, ast.Str):
                return repr(node.s)
            elif isinstance(node, ast.Num):
                return repr(node.n)
            elif isinstance(node, ast.Bytes):
                return repr(node.s)
            elif isinstance(node, ast.NameConstant):
                return repr(node.value)
        elif isinstance(node, ast.Index):
            # Python 3.8-
            return self._extract_subscript_slice(node.value)

        return "..."

    def find_dependencies(self, tree: ast.Module) -> Set[str]:
        """
        Find external package dependencies in the AST.

        Args:
            tree: AST Module node

        Returns:
            Set of dependency package names
        """
        dependencies = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    # Extract root package (e.g., 'pandas' from 'pandas.DataFrame')
                    root_package = name.name.split(".")[0]
                    dependencies.add(root_package)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Extract root package
                    root_package = node.module.split(".")[0]
                    dependencies.add(root_package)

        # Remove standard library modules
        std_libs = {
            "os",
            "sys",
            "re",
            "math",
            "random",
            "datetime",
            "time",
            "json",
            "csv",
            "collections",
            "itertools",
            "functools",
            "typing",
            "pathlib",
            "io",
            "shutil",
            "glob",
            "argparse",
            "logging",
            "unittest",
            "abc",
            "ast",
            "copy",
            "hashlib",
            "pickle",
            "tempfile",
            "warnings",
            "zipfile",
            "importlib",
        }

        return dependencies - std_libs

    def extract_test_functions(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract test functions from a Python test file.

        Args:
            file_path: Path to the test file

        Returns:
            List of test function dictionaries
        """
        # Parse the file
        analysis = self.parse_file(file_path)

        if "error" in analysis:
            return []

        test_functions = []

        # Extract top-level test functions
        for func in analysis.get("functions", []):
            if func["name"].startswith("test_"):
                test_functions.append(func)

        # Extract test methods in test classes
        for cls in analysis.get("classes", []):
            if cls["name"].startswith("Test"):
                for method in cls.get("methods", []):
                    if method["name"].startswith("test_"):
                        # Add class context to the method
                        method["class"] = cls["name"]
                        test_functions.append(method)

        return test_functions

    def extract_test_classes(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract test classes from a Python test file.

        Args:
            file_path: Path to the test file

        Returns:
            List of test class dictionaries
        """
        # Parse the file
        analysis = self.parse_file(file_path)

        if "error" in analysis:
            return []

        test_classes = []

        # Extract test classes
        for cls in analysis.get("classes", []):
            if cls["name"].startswith("Test"):
                # Only include classes with test methods
                has_test_methods = any(
                    method["name"].startswith("test_")
                    for method in cls.get("methods", [])
                )

                if has_test_methods:
                    test_classes.append(cls)

        return test_classes

    def detect_test_framework(self, file_path: str) -> str:
        """
        Detect the test framework used in a Python test file.

        Args:
            file_path: Path to the test file

        Returns:
            Detected framework name ("pytest", "unittest", or "unknown")
        """
        # Parse the file
        analysis = self.parse_file(file_path)

        if "error" in analysis:
            return "unknown"

        # Check imports for pytest or unittest
        imports = analysis.get("imports", [])

        for imp in imports:
            if imp.get("name") == "pytest" or imp.get("module") == "pytest":
                return "pytest"
            if imp.get("name") == "unittest" or imp.get("module") == "unittest":
                return "unittest"

        # Check class inheritance for unittest.TestCase
        for cls in analysis.get("classes", []):
            if "TestCase" in cls.get("bases", []):
                return "unittest"

        # Check for pytest fixtures
        for func in analysis.get("functions", []):
            if "fixture" in func.get("decorators", []):
                return "pytest"

        # Check for pytest-style test functions
        has_pytest_tests = any(
            func["name"].startswith("test_") for func in analysis.get("functions", [])
        )

        if has_pytest_tests:
            return "pytest"

        # Check for unittest-style test methods
        for cls in analysis.get("classes", []):
            has_unittest_tests = any(
                method["name"].startswith("test") for method in cls.get("methods", [])
            )

            if has_unittest_tests and "TestCase" in cls.get("bases", []):
                return "unittest"

        return "unknown"


# Create a singleton instance
python_parser = PythonParser()
