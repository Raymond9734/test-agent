# test_agent/language/go/adapter.py

import os
import re
from typing import List, Dict, Any, Optional

from ..base import LanguageAdapter, registry


class GoAdapter(LanguageAdapter):
    """Go language adapter with enhanced analysis and project detection"""

    def __init__(self):
        """Initialize the Go adapter"""
        self._project_roots_cache = {}
        self._test_patterns_cache = {}
        self._module_name_cache = {}

    @property
    def language_name(self) -> str:
        """Return the name of the language"""
        return "go"

    @property
    def file_extensions(self) -> List[str]:
        """Return the file extensions for this language"""
        return [".go"]

    @property
    def test_file_prefix(self) -> str:
        """
        Return the prefix for test files.
        Go uses a suffix (_test.go) rather than a prefix.
        """
        return ""  # No prefix - Go uses suffix _test.go

    @property
    def test_command(self) -> List[str]:
        """Return the command to run tests"""
        return ["go", "test"]

    def analyze_source_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Go source file to extract structure"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Basic parsing using regular expressions

            # Extract package name
            package_match = re.search(r"package\s+(\w+)", content)
            package = package_match.group(1) if package_match else "unknown"

            # Extract imports
            imports = []
            import_blocks = re.findall(r"import\s+\((.*?)\)", content, re.DOTALL)
            for block in import_blocks:
                for line in block.strip().split("\n"):
                    line = line.strip()
                    if line and not line.startswith("//"):
                        # Extract quoted import path
                        import_match = re.search(r'"([^"]+)"', line)
                        if import_match:
                            imports.append(import_match.group(1))

            # Also catch single line imports
            single_imports = re.findall(r'import\s+"([^"]+)"', content)
            imports.extend(single_imports)

            # Extract function definitions (not methods)
            function_matches = re.findall(
                r"func\s+(\w+)\s*\((.*?)\)(.*?){", content, re.DOTALL
            )
            functions = []
            for name, args, ret in function_matches:
                args_clean = self._clean_go_params(args)
                ret_clean = self._clean_go_returns(ret)

                functions.append(
                    {"name": name, "args": args_clean, "returns": ret_clean}
                )

            # Extract struct definitions
            struct_matches = re.findall(
                r"type\s+(\w+)\s+struct\s+{(.*?)}", content, re.DOTALL
            )
            structs = []
            for name, fields_str in struct_matches:
                fields = []
                # Match field names and types - handles multi-line definitions
                field_matches = re.findall(r"(\w+)(?:\s+([*\[\]\w\.]+))", fields_str)
                for field_name, field_type in field_matches:
                    fields.append({"name": field_name, "type": field_type.strip()})

                structs.append({"name": name, "fields": fields})

            # Extract method definitions (functions with receivers)
            method_matches = re.findall(
                r"func\s+\(([\w\s*]+)\)\s+(\w+)\s*\((.*?)\)(.*?){", content, re.DOTALL
            )
            methods = []
            for receiver, name, args, ret in method_matches:
                receiver_parts = receiver.strip().split()
                receiver_var = receiver_parts[0] if len(receiver_parts) > 0 else ""
                receiver_type = (
                    receiver_parts[-1].replace("*", "")
                    if len(receiver_parts) > 1
                    else receiver_parts[0].replace("*", "")
                )

                args_clean = self._clean_go_params(args)
                ret_clean = self._clean_go_returns(ret)

                methods.append(
                    {
                        "receiver_var": receiver_var,
                        "receiver_type": receiver_type,
                        "name": name,
                        "args": args_clean,
                        "returns": ret_clean,
                    }
                )

            # Extract interfaces
            interface_matches = re.findall(
                r"type\s+(\w+)\s+interface\s+{(.*?)}", content, re.DOTALL
            )
            interfaces = []
            for name, methods_str in interface_matches:
                interface_methods = []

                # Extract method signatures
                method_matches = re.findall(
                    r"(\w+)(?:\((.*?)\))(?:\s*(.*?))?(?:$|\n)",
                    methods_str,
                    re.MULTILINE,
                )
                for method_match in method_matches:
                    method_name = method_match[0]
                    method_args = (
                        self._clean_go_params(method_match[1])
                        if len(method_match) > 1
                        else []
                    )
                    method_returns = (
                        self._clean_go_returns(method_match[2])
                        if len(method_match) > 2
                        else None
                    )

                    interface_methods.append(
                        {
                            "name": method_name,
                            "args": method_args,
                            "returns": method_returns,
                        }
                    )

                interfaces.append({"name": name, "methods": interface_methods})

            # Get module info from go.mod
            module_name = self._get_module_name(file_path)

            return {
                "file": file_path,
                "package": package,
                "module": module_name,
                "imports": imports,
                "functions": functions,
                "structs": structs,
                "methods": methods,
                "interfaces": interfaces,
            }

        except Exception as e:
            return {"file": file_path, "error": str(e)}

    def _clean_go_params(self, params_str: str) -> List[Dict[str, str]]:
        """Clean and parse Go parameters"""
        params = []

        if not params_str.strip():
            return params

        # Split by commas, but handle multiple parameters of the same type
        param_groups = re.findall(r"((?:\w+(?:,\s*\w+)*)\s+[\*\[\]\w\.]+)", params_str)

        for group in param_groups:
            parts = group.split()
            if len(parts) >= 2:
                param_type = parts[-1]
                param_names = parts[0].split(",")

                for name in param_names:
                    name = name.strip()
                    if name:
                        params.append({"name": name, "type": param_type})

        return params

    def _clean_go_returns(self, returns_str: str) -> Optional[List[Dict[str, str]]]:
        """Clean and parse Go return values"""
        returns_str = returns_str.strip()

        if not returns_str:
            return None

        returns = []

        # Handle single unnamed return
        if "(" not in returns_str and "," not in returns_str:
            return [{"type": returns_str.strip()}]

        # Handle named returns or multiple returns
        if returns_str.startswith("("):
            # Remove parentheses
            returns_str = (
                returns_str[1:-1] if returns_str.endswith(")") else returns_str[1:]
            )

        # Split by commas
        for ret in returns_str.split(","):
            ret = ret.strip()
            if ret:
                # Check for named return
                parts = ret.split()
                if len(parts) == 2:
                    returns.append({"name": parts[0], "type": parts[1]})
                else:
                    returns.append({"type": ret})

        return returns

    def analyze_test_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a Go test file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract package name
            package_match = re.search(r"package\s+(\w+)", content)
            package = package_match.group(1) if package_match else "unknown"

            # Extract test functions
            test_funcs = re.findall(r"func\s+(Test\w+)\s*\(", content)
            benchmark_funcs = re.findall(r"func\s+(Benchmark\w+)\s*\(", content)
            example_funcs = re.findall(r"func\s+(Example\w+)\s*\(", content)

            # Extract import paths for source package
            imports = []
            import_blocks = re.findall(r"import\s+\((.*?)\)", content, re.DOTALL)
            for block in import_blocks:
                for line in block.strip().split("\n"):
                    line = line.strip()
                    if line and not line.startswith("//"):
                        # Extract quoted import path
                        import_match = re.search(r'"([^"]+)"', line)
                        if import_match:
                            imports.append(import_match.group(1))

            # Also catch single line imports
            single_imports = re.findall(r'import\s+"([^"]+)"', content)
            imports.extend(single_imports)

            # Determine if it uses the _test package convention
            is_test_package = "_test" in package

            # Find source file being tested
            project_root = self._find_project_root(file_path)
            source_file = self.find_corresponding_source(file_path, project_root)

            return {
                "file": file_path,
                "source_file": source_file,
                "package": package,
                "is_test_package": is_test_package,
                "test_functions": test_funcs,
                "benchmark_functions": benchmark_funcs,
                "example_functions": example_funcs,
                "imports": imports,
            }

        except Exception as e:
            return {"file": file_path, "error": str(e)}

    def _get_module_name(self, file_path: str) -> str:
        """
        Extract the Go module name from go.mod file

        Args:
            file_path: Path to any file in the Go project

        Returns:
            Module name or empty string if not found
        """
        # Check cache first
        project_root = self._find_project_root(file_path)
        if project_root in self._module_name_cache:
            return self._module_name_cache[project_root]

        # Find go.mod file
        go_mod_path = os.path.join(project_root, "go.mod")
        if not os.path.exists(go_mod_path):
            self._module_name_cache[project_root] = ""
            return ""

        try:
            with open(go_mod_path, "r") as f:
                content = f.read()

            # Extract module name
            module_match = re.search(r"module\s+([^\s]+)", content)
            if module_match:
                module_name = module_match.group(1)
                self._module_name_cache[project_root] = module_name
                return module_name
        except Exception:
            pass

        self._module_name_cache[project_root] = ""
        return ""

    def detect_project_structure(self, project_dir: str) -> Dict[str, Any]:
        """Detect project structure, test patterns, and framework information"""
        # Check cache first
        project_dir = os.path.abspath(project_dir)
        if project_dir in self._test_patterns_cache:
            return self._test_patterns_cache[project_dir]

        # Go projects typically have tests in the same directory as source files
        # But some larger projects might have a /test directory
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
                if file_name.endswith("_test.go"):
                    test_files.append(os.path.join(root, file_name))

        # Analyze test location patterns
        location_pattern = "same_directory"  # Default for Go

        if test_files:
            # Count location patterns
            location_counts = {
                "same_directory": 0,
                "tests_directory": 0,
            }

            # Also check for test package pattern
            package_patterns = {
                "same_package": 0,
                "package_test": 0,
            }

            for test_file in test_files:
                # Check location pattern
                test_dir = os.path.dirname(test_file)
                dir_name = os.path.basename(test_dir).lower()

                if dir_name in ["test", "tests", "testing"]:
                    location_counts["tests_directory"] += 1
                else:
                    location_counts["same_directory"] += 1

                    analysis = self.analyze_test_file(test_file)
                    if analysis.get("is_test_package"):
                        package_patterns["package_test"] += 1
                    else:
                        package_patterns["same_package"] += 1

            # Determine primary patterns
            if location_counts:
                location_pattern = max(location_counts.items(), key=lambda x: x[1])[0]

            # Determine package pattern
            package_pattern = "same_package"  # Default
            if package_patterns["package_test"] > package_patterns["same_package"]:
                package_pattern = "package_test"

        # Get module name
        module_name = self._get_module_name(project_dir)

        # Create result
        result = {
            "test_directories": test_directories,
            "test_files": test_files,
            "location_pattern": location_pattern,
            "package_pattern": package_pattern,
            "module": module_name,
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

        # For Go, the test file is always named {source_base}_test.go
        test_filename = f"{source_base}_test{source_ext}"

        # Generate path based on location pattern
        location_pattern = pattern.get("location_pattern", "same_directory")

        if location_pattern == "same_directory":
            # Go standard is tests in the same directory
            return os.path.join(os.path.dirname(source_file), test_filename)
        else:  # tests_directory
            # Use test_directory - but need to match package structure
            rel_path = os.path.relpath(os.path.dirname(source_file), project_root)
            test_path = os.path.join(test_directory, rel_path)
            os.makedirs(test_path, exist_ok=True)
            return os.path.join(test_path, test_filename)

    def generate_test_template(
        self,
        source_file: str,
        analysis: Dict[str, Any],
        pattern: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a test template based on project patterns and file analysis"""
        # Use provided pattern or detect it
        if pattern is None:
            project_dir = self._find_project_root(source_file)
            pattern = self.detect_project_structure(project_dir)

        package_name = analysis.get("package", "main")
        package_pattern = pattern.get("package_pattern", "same_package")

        # Determine the package for the test file
        if package_pattern == "package_test":
            test_package = f"{package_name}_test"
        else:
            test_package = package_name

        # Check if we need to import the source package
        import_source = package_pattern == "package_test"

        module_name = pattern.get("module", "")
        source_import = ""
        if import_source and module_name:
            # If module name is available, use it to construct the import path
            project_dir = self._find_project_root(source_file)
            rel_source_dir = os.path.dirname(os.path.relpath(source_file, project_dir))
            if rel_source_dir:
                source_import = f'\n\t"{module_name}/{rel_source_dir}"'
            else:
                source_import = f'\n\t"{module_name}"'

        template = f"""package {test_package}

import (
\t"testing"{source_import}
)

// Generated test file for {os.path.basename(source_file)}
"""

        # Add tests for standalone functions
        for func in analysis.get("functions", []):
            template += f"""
func Test{func['name']}(t *testing.T) {{
\t// TODO: Write test for {func['name']}
\tt.Run("{func['name']}", func(t *testing.T) {{
\t\t// TODO: Add test cases
\t\t// result := {package_name if import_source else ""}.{func['name']}()
\t\t// if result != expected {{
\t\t//     t.Errorf("Expected %v, got %v", expected, result)
\t\t// }}
\t}})
}}
"""

        # Add tests for methods
        for method in analysis.get("methods", []):
            receiver_type = method.get("receiver_type", "")
            method_name = method.get("name", "")

            if receiver_type and method_name:
                template += f"""
func Test{receiver_type}_{method_name}(t *testing.T) {{
\t// TODO: Write test for {receiver_type}.{method_name}
\tt.Run("{method_name}", func(t *testing.T) {{
\t\t// Create an instance of {receiver_type}
\t\t// instance := {package_name + "." if import_source else ""}{receiver_type}{{}}
\t\t
\t\t// TODO: Add test cases
\t\t// result := instance.{method_name}()
\t\t// if result != expected {{
\t\t//     t.Errorf("Expected %v, got %v", expected, result)
\t\t// }}
\t}})
}}
"""

        # Add table-driven test example if there are functions or methods
        if analysis.get("functions") or analysis.get("methods"):
            template += """
// Example of table-driven tests
func TestTableDriven(t *testing.T) {
\t// Define test cases
\ttests := []struct {
\t\tname     string
\t\tinput    string
\t\texpected string
\t}{
\t\t{"test case 1", "input1", "expected1"},
\t\t{"test case 2", "input2", "expected2"},
\t}

\t// Run test cases
\tfor _, tc := range tests {
\t\tt.Run(tc.name, func(t *testing.T) {
\t\t\t// TODO: Call the function/method being tested
\t\t\t// result := YourFunction(tc.input)
\t\t\t
\t\t\t// if result != tc.expected {
\t\t\t//     t.Errorf("Expected %v, got %v", tc.expected, result)
\t\t\t// }
\t\t})
\t}
}
"""

        return template

    def check_is_test_file(self, file_path: str) -> bool:
        """Check if a file is a Go test file"""
        return file_path.endswith("_test.go")

    def find_corresponding_source(
        self, test_file: str, project_dir: str
    ) -> Optional[str]:
        """Find the source file that a test file is testing"""
        if not self.check_is_test_file(test_file):
            return None

        # Go test files are usually in the same directory as the source file
        # with the naming pattern: source.go -> source_test.go
        file_name = os.path.basename(test_file)

        # Extract source file name: remove _test.go suffix
        if file_name.endswith("_test.go"):
            source_name = file_name[:-8] + ".go"  # Replace _test.go with .go

            # Check in the same directory
            source_path = os.path.join(os.path.dirname(test_file), source_name)
            if os.path.exists(source_path):
                return source_path

        # If source not found and test is in a test directory, check parent directory
        test_dir = os.path.dirname(test_file)
        test_dir_name = os.path.basename(test_dir).lower()
        if test_dir_name in ["test", "tests", "testing"]:
            parent_dir = os.path.dirname(test_dir)
            source_path = os.path.join(parent_dir, source_name)
            if os.path.exists(source_path):
                return source_path

        # If still not found, search the project
        for root, _, files in os.walk(project_dir):
            if source_name in files:
                return os.path.join(root, source_name)

        return None

    def find_corresponding_test(
        self, source_file: str, project_dir: str
    ) -> Optional[str]:
        """Find the test file for a given source file"""
        # Skip if it's already a test file
        if self.check_is_test_file(source_file):
            return None

        file_name = os.path.basename(source_file)
        source_base, source_ext = os.path.splitext(file_name)
        source_dir = os.path.dirname(source_file)

        # Generate test file name (Go standard pattern)
        test_name = f"{source_base}_test{source_ext}"

        # Check in the same directory (Go standard)
        test_path = os.path.join(source_dir, test_name)
        if os.path.exists(test_path):
            return test_path

        # Check in a test directory with the same relative path
        test_dirs = []

        # Look for test directories in project
        for root, dirs, _ in os.walk(project_dir):
            for dir_name in dirs:
                if dir_name.lower() in ["test", "tests", "testing"]:
                    test_dirs.append(os.path.join(root, dir_name))

        # Check each test directory
        rel_source_dir = os.path.relpath(source_dir, project_dir)
        for test_dir in test_dirs:
            if os.path.isdir(test_dir):
                # Check for direct match in test dir
                test_path = os.path.join(test_dir, test_name)
                if os.path.exists(test_path):
                    return test_path

                # Check with matching directory structure
                mirror_dir = os.path.join(test_dir, rel_source_dir)
                if os.path.isdir(mirror_dir):
                    test_path = os.path.join(mirror_dir, test_name)
                    if os.path.exists(test_path):
                        return test_path

        return None

    def get_environment_location(self) -> str:
        """
        Get the recommended location for the build environment.
        For Go, we don't really need a special environment as 'go test' handles it.
        """
        # No special environment needed for Go
        return ""


# Register the adapter
registry.register([".go"], GoAdapter)
