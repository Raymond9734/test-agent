# test_agent/language/go/parser.py

import os
import logging
import re
import subprocess
from typing import Dict, List, Any, Set

# Configure logging
logger = logging.getLogger(__name__)


class GoParser:
    """
    Parser for Go source code.

    Provides methods to extract structure and content from Go files
    using regular expressions and Go's built-in tools when available.
    """

    def __init__(self):
        """Initialize the Go parser."""
        self._parse_cache = {}  # Cache parsed file structures
        self._go_installed = self._check_go_installed()

    def _check_go_installed(self) -> bool:
        """
        Check if Go is installed and available.

        Returns:
            True if Go is installed, False otherwise
        """
        try:
            subprocess.run(
                ["go", "version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning(
                "Go is not installed or not in PATH. Using regex-based parsing only."
            )
            return False

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a Go file and extract its structure.

        Args:
            file_path: Path to the Go file to parse

        Returns:
            Dictionary with parsed file structure
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}

            # Check cache first
            if file_path in self._parse_cache:
                return self._parse_cache[file_path]

            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Try to use Go's built-in tooling if available
            if self._go_installed:
                try:
                    ast_result = self._parse_with_go_ast(file_path)
                    if ast_result and not ast_result.get("error"):
                        self._parse_cache[file_path] = ast_result
                        return ast_result
                except Exception as e:
                    logger.debug(
                        f"Error using Go AST parser: {str(e)}. Falling back to regex parser."
                    )

            # Fall back to regex-based parsing
            result = self._parse_with_regex(content, file_path)
            self._parse_cache[file_path] = result
            return result

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            return {"error": str(e)}

    def parse_content(self, content: str, filename: str = "<string>") -> Dict[str, Any]:
        """
        Parse Go code content directly.

        Args:
            content: Go code as string
            filename: Optional virtual filename for the content

        Returns:
            Dictionary with parsed content structure
        """
        try:
            # Use regex-based parsing for content
            return self._parse_with_regex(content, filename)

        except Exception as e:
            logger.error(f"Error parsing content: {str(e)}")
            return {"error": str(e)}

    def _parse_with_go_ast(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a Go file using Go's built-in AST tools.

        Args:
            file_path: Path to the Go file

        Returns:
            Dictionary with parsed file structure
        """
        # Create a temporary Go file to extract the AST
        temp_dir = os.path.dirname(os.path.abspath(file_path))
        ast_tool_path = os.path.join(temp_dir, "_temp_ast_tool.go")

        # Simple Go program to print the AST as JSON
        ast_tool_content = """
package main

import (
    "encoding/json"
    "fmt"
    "go/ast"
    "go/parser"
    "go/token"
    "os"
)

type SimpleFunc struct {
    Name       string   `json:"name"`
    DocString  string   `json:"docstring,omitempty"`
    Parameters []string `json:"parameters,omitempty"`
    Results    []string `json:"results,omitempty"`
    IsMethod   bool     `json:"is_method"`
    Receiver   string   `json:"receiver,omitempty"`
}

type SimpleStruct struct {
    Name      string   `json:"name"`
    DocString string   `json:"docstring,omitempty"`
    Fields    []string `json:"fields,omitempty"`
}

type SimpleInterface struct {
    Name      string      `json:"name"`
    DocString string      `json:"docstring,omitempty"`
    Methods   []SimpleFunc `json:"methods,omitempty"`
}

type SimpleFile struct {
    Package    string           `json:"package"`
    Imports    []string         `json:"imports,omitempty"`
    Functions  []SimpleFunc     `json:"functions,omitempty"`
    Structs    []SimpleStruct   `json:"structs,omitempty"`
    Interfaces []SimpleInterface `json:"interfaces,omitempty"`
}

func main() {
    if len(os.Args) < 2 {
        fmt.Println("Usage: go run ast_tool.go <file.go>")
        os.Exit(1)
    }

    filePath := os.Args[1]
    fset := token.NewFileSet()
    f, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
    if err != nil {
        fmt.Printf(`{"error": "%s"}`, err)
        os.Exit(1)
    }

    result := SimpleFile{
        Package: f.Name.Name,
    }

    // Extract imports
    for _, i := range f.Imports {
        path := i.Path.Value
        result.Imports = append(result.Imports, path)
    }

    // Extract declarations
    for _, decl := range f.Decls {
        switch d := decl.(type) {
        case *ast.FuncDecl:
            // Get function/method info
            fn := SimpleFunc{
                Name:      d.Name.Name,
                IsMethod:  d.Recv != nil,
            }

            // Extract doc string
            if d.Doc != nil {
                fn.DocString = d.Doc.Text()
            }

            // Extract receiver for methods
            if d.Recv != nil && len(d.Recv.List) > 0 {
                // Get receiver type
                expr := d.Recv.List[0].Type
                if star, ok := expr.(*ast.StarExpr); ok {
                    expr = star.X  // Dereference pointer
                }
                if ident, ok := expr.(*ast.Ident); ok {
                    fn.Receiver = ident.Name
                }
            }

            // Extract parameters
            if d.Type.Params != nil {
                for _, param := range d.Type.Params.List {
                    typeStr := ""
                    // Convert type to string (simplified)
                    switch t := param.Type.(type) {
                    case *ast.Ident:
                        typeStr = t.Name
                    case *ast.StarExpr:
                        if ident, ok := t.X.(*ast.Ident); ok {
                            typeStr = "*" + ident.Name
                        }
                    }
                    
                    // Add parameters
                    for _, name := range param.Names {
                        paramStr := name.Name + " " + typeStr
                        fn.Parameters = append(fn.Parameters, paramStr)
                    }
                }
            }

            // Extract return values
            if d.Type.Results != nil {
                for _, result := range d.Type.Results.List {
                    typeStr := ""
                    // Convert type to string (simplified)
                    switch t := result.Type.(type) {
                    case *ast.Ident:
                        typeStr = t.Name
                    case *ast.StarExpr:
                        if ident, ok := t.X.(*ast.Ident); ok {
                            typeStr = "*" + ident.Name
                        }
                    }
                    
                    // Add named return or just type
                    if len(result.Names) > 0 {
                        for _, name := range result.Names {
                            resultStr := name.Name + " " + typeStr
                            fn.Results = append(fn.Results, resultStr)
                        }
                    } else {
                        fn.Results = append(fn.Results, typeStr)
                    }
                }
            }

            result.Functions = append(result.Functions, fn)

        case *ast.GenDecl:
            for _, spec := range d.Specs {
                switch s := spec.(type) {
                case *ast.TypeSpec:
                    // Check if it's a struct
                    if structType, ok := s.Type.(*ast.StructType); ok {
                        st := SimpleStruct{
                            Name: s.Name.Name,
                        }
                        
                        // Extract doc string
                        if d.Doc != nil {
                            st.DocString = d.Doc.Text()
                        }
                        
                        // Extract fields
                        if structType.Fields != nil {
                            for _, field := range structType.Fields.List {
                                typeStr := ""
                                // Convert type to string (simplified)
                                switch t := field.Type.(type) {
                                case *ast.Ident:
                                    typeStr = t.Name
                                case *ast.StarExpr:
                                    if ident, ok := t.X.(*ast.Ident); ok {
                                        typeStr = "*" + ident.Name
                                    }
                                }
                                
                                // Add fields
                                if len(field.Names) > 0 {
                                    for _, name := range field.Names {
                                        fieldStr := name.Name + " " + typeStr
                                        st.Fields = append(st.Fields, fieldStr)
                                    }
                                } else {
                                    // Embedded field
                                    st.Fields = append(st.Fields, typeStr)
                                }
                            }
                        }
                        
                        result.Structs = append(result.Structs, st)
                    }
                    
                    // Check if it's an interface
                    if interfaceType, ok := s.Type.(*ast.InterfaceType); ok {
                        iface := SimpleInterface{
                            Name: s.Name.Name,
                        }
                        
                        // Extract doc string
                        if d.Doc != nil {
                            iface.DocString = d.Doc.Text()
                        }
                        
                        // Extract methods
                        if interfaceType.Methods != nil {
                            for _, method := range interfaceType.Methods.List {
                                if len(method.Names) > 0 {
                                    // It's a method
                                    fnType, ok := method.Type.(*ast.FuncType)
                                    if ok {
                                        fn := SimpleFunc{
                                            Name:     method.Names[0].Name,
                                            IsMethod: true,
                                        }
                                        
                                        // Extract parameters
                                        if fnType.Params != nil {
                                            for _, param := range fnType.Params.List {
                                                typeStr := ""
                                                // Get parameter type (simplified)
                                                switch t := param.Type.(type) {
                                                case *ast.Ident:
                                                    typeStr = t.Name
                                                case *ast.StarExpr:
                                                    if ident, ok := t.X.(*ast.Ident); ok {
                                                        typeStr = "*" + ident.Name
                                                    }
                                                }
                                                
                                                // Add parameters
                                                if len(param.Names) > 0 {
                                                    for _, name := range param.Names {
                                                        paramStr := name.Name + " " + typeStr
                                                        fn.Parameters = append(fn.Parameters, paramStr)
                                                    }
                                                } else {
                                                    fn.Parameters = append(fn.Parameters, typeStr)
                                                }
                                            }
                                        }
                                        
                                        // Extract results
                                        if fnType.Results != nil {
                                            for _, result := range fnType.Results.List {
                                                typeStr := ""
                                                // Get result type (simplified)
                                                switch t := result.Type.(type) {
                                                case *ast.Ident:
                                                    typeStr = t.Name
                                                case *ast.StarExpr:
                                                    if ident, ok := t.X.(*ast.Ident); ok {
                                                        typeStr = "*" + ident.Name
                                                    }
                                                }
                                                
                                                // Add results
                                                if len(result.Names) > 0 {
                                                    for _, name := range result.Names {
                                                        resultStr := name.Name + " " + typeStr
                                                        fn.Results = append(fn.Results, resultStr)
                                                    }
                                                } else {
                                                    fn.Results = append(fn.Results, typeStr)
                                                }
                                            }
                                        }
                                        
                                        iface.Methods = append(iface.Methods, fn)
                                    }
                                }
                            }
                        }
                        
                        result.Interfaces = append(result.Interfaces, iface)
                    }
                }
            }
        }
    }

    // Output as JSON
    jsonData, err := json.MarshalIndent(result, "", "  ")
    if err != nil {
        fmt.Printf(`{"error": "%s"}`, err)
        os.Exit(1)
    }
    
    fmt.Println(string(jsonData))
}
"""

        try:
            # Write the temporary Go file
            with open(ast_tool_path, "w") as f:
                f.write(ast_tool_content)

            # Run the Go program to extract AST
            process = subprocess.run(
                ["go", "run", ast_tool_path, file_path],
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse the JSON output
            import json

            ast_data = json.loads(process.stdout)

            if os.path.exists(ast_tool_path):
                os.remove(ast_tool_path)

            # Convert to our standard format
            result = {
                "file": file_path,
                "package": ast_data.get("package", ""),
                "imports": [],
                "functions": [],
                "structs": [],
                "interfaces": [],
            }

            # Process imports
            for imp in ast_data.get("imports", []):
                # Clean up quotes
                cleaned = imp.strip("\"'")
                result["imports"].append({"path": cleaned})

            # Process functions
            for func in ast_data.get("functions", []):
                func_info = {
                    "name": func.get("name", ""),
                    "docstring": func.get("docstring", ""),
                    "parameters": [],
                    "returns": [],
                    "is_method": func.get("is_method", False),
                    "receiver": func.get("receiver", ""),
                }

                # Process parameters
                for param in func.get("parameters", []):
                    parts = param.split(" ", 1)
                    if len(parts) == 2:
                        func_info["parameters"].append(
                            {
                                "name": parts[0],
                                "type": parts[1],
                            }
                        )

                # Process returns
                for ret in func.get("results", []):
                    parts = ret.split(" ", 1)
                    if len(parts) == 2:
                        func_info["returns"].append(
                            {
                                "name": parts[0],
                                "type": parts[1],
                            }
                        )
                    else:
                        func_info["returns"].append(
                            {
                                "type": ret,
                            }
                        )

                result["functions"].append(func_info)

            # Process structs
            for struct in ast_data.get("structs", []):
                struct_info = {
                    "name": struct.get("name", ""),
                    "docstring": struct.get("docstring", ""),
                    "fields": [],
                }

                # Process fields
                for field in struct.get("fields", []):
                    parts = field.split(" ", 1)
                    if len(parts) == 2:
                        struct_info["fields"].append(
                            {
                                "name": parts[0],
                                "type": parts[1],
                            }
                        )
                    else:
                        # Embedded field
                        struct_info["fields"].append(
                            {
                                "type": field,
                                "embedded": True,
                            }
                        )

                result["structs"].append(struct_info)

            # Process interfaces
            for iface in ast_data.get("interfaces", []):
                iface_info = {
                    "name": iface.get("name", ""),
                    "docstring": iface.get("docstring", ""),
                    "methods": [],
                }

                # Process methods
                for method in iface.get("methods", []):
                    method_info = {
                        "name": method.get("name", ""),
                        "parameters": [],
                        "returns": [],
                    }

                    # Process parameters
                    for param in method.get("parameters", []):
                        parts = param.split(" ", 1)
                        if len(parts) == 2:
                            method_info["parameters"].append(
                                {
                                    "name": parts[0],
                                    "type": parts[1],
                                }
                            )
                        else:
                            method_info["parameters"].append(
                                {
                                    "type": param,
                                }
                            )

                    # Process returns
                    for ret in method.get("results", []):
                        parts = ret.split(" ", 1)
                        if len(parts) == 2:
                            method_info["returns"].append(
                                {
                                    "name": parts[0],
                                    "type": parts[1],
                                }
                            )
                        else:
                            method_info["returns"].append(
                                {
                                    "type": ret,
                                }
                            )

                    iface_info["methods"].append(method_info)

                result["interfaces"].append(iface_info)

            return result

        except subprocess.CalledProcessError as e:
            logger.debug(f"Go AST extraction failed: {e.stderr}")
            raise
        finally:

            if os.path.exists(ast_tool_path):
                os.remove(ast_tool_path)

    def _parse_with_regex(self, content: str, file_path: str) -> Dict[str, Any]:
        """
        Parse Go code using regular expressions.

        Args:
            content: Go code content
            file_path: Path to the file (for reference)

        Returns:
            Dictionary with parsed file structure
        """
        result = {
            "file": file_path,
            "package": "",
            "imports": [],
            "functions": [],
            "structs": [],
            "interfaces": [],
        }

        # Extract package name
        package_match = re.search(r"package\s+(\w+)", content)
        if package_match:
            result["package"] = package_match.group(1)

        # Extract imports
        import_blocks = re.findall(r"import\s+\((.*?)\)", content, re.DOTALL)
        for block in import_blocks:
            for line in block.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("//"):
                    # Extract quoted import path
                    import_match = re.search(r'"([^"]+)"', line)
                    if import_match:
                        import_path = import_match.group(1)
                        result["imports"].append({"path": import_path})

        # Also catch single line imports
        single_imports = re.findall(r'import\s+"([^"]+)"', content)
        for import_path in single_imports:
            result["imports"].append({"path": import_path})

        # Extract function definitions (not methods)
        function_matches = re.findall(
            r"func\s+(\w+)\s*\((.*?)\)(.*?){", content, re.DOTALL
        )
        for name, args, ret in function_matches:
            # Parse arguments
            params = self._parse_go_params(args)

            # Parse return values
            returns = self._parse_go_returns(ret)

            # Add to results
            result["functions"].append(
                {
                    "name": name,
                    "parameters": params,
                    "returns": returns,
                    "is_method": False,
                }
            )

        # Extract method definitions
        method_matches = re.findall(
            r"func\s+\(([\w\s*]+)\)\s+(\w+)\s*\((.*?)\)(.*?){", content, re.DOTALL
        )
        for receiver, name, args, ret in method_matches:
            # Parse receiver
            receiver_parts = receiver.strip().split()
            receiver_var = receiver_parts[0] if len(receiver_parts) > 0 else ""
            receiver_type = (
                receiver_parts[-1].replace("*", "")
                if len(receiver_parts) > 1
                else receiver_parts[0].replace("*", "")
            )

            # Parse arguments
            params = self._parse_go_params(args)

            # Parse return values
            returns = self._parse_go_returns(ret)

            # Add to results
            result["functions"].append(
                {
                    "name": name,
                    "parameters": params,
                    "returns": returns,
                    "is_method": True,
                    "receiver": receiver_type,
                    "receiver_var": receiver_var,
                }
            )

        # Extract struct definitions
        struct_matches = re.findall(
            r"type\s+(\w+)\s+struct\s+{(.*?)}", content, re.DOTALL
        )
        for name, fields_str in struct_matches:
            fields = []

            # Match field names and types
            field_matches = re.findall(r"(\w+)(?:\s+([*\[\]\w\.]+))", fields_str)
            for field_name, field_type in field_matches:
                fields.append(
                    {
                        "name": field_name,
                        "type": field_type.strip(),
                    }
                )

            result["structs"].append(
                {
                    "name": name,
                    "fields": fields,
                }
            )

        # Extract interface definitions
        interface_matches = re.findall(
            r"type\s+(\w+)\s+interface\s+{(.*?)}", content, re.DOTALL
        )
        for name, methods_str in interface_matches:
            methods = []

            # Extract method signatures
            method_matches = re.findall(
                r"(\w+)(?:\((.*?)\))(?:\s*(.*?))?(?:$|\n)", methods_str, re.MULTILINE
            )

            for method_match in method_matches:
                method_name = method_match[0]

                # Parse parameters
                method_params = []
                if len(method_match) > 1:
                    method_params = self._parse_go_params(method_match[1])

                # Parse return values
                method_returns = []
                if len(method_match) > 2:
                    method_returns = self._parse_go_returns(method_match[2])

                methods.append(
                    {
                        "name": method_name,
                        "parameters": method_params,
                        "returns": method_returns,
                    }
                )

            result["interfaces"].append(
                {
                    "name": name,
                    "methods": methods,
                }
            )

        return result

    def _parse_go_params(self, params_str: str) -> List[Dict[str, str]]:
        """
        Parse Go function/method parameters.

        Args:
            params_str: Parameter string (e.g., "a int, b string")

        Returns:
            List of parameter dictionaries
        """
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

    def _parse_go_returns(self, returns_str: str) -> List[Dict[str, str]]:
        """
        Parse Go function/method return values.

        Args:
            returns_str: Return string (e.g., "int, error")

        Returns:
            List of return value dictionaries
        """
        returns_str = returns_str.strip()

        if not returns_str:
            return []

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

    def find_dependencies(self, content: str) -> Set[str]:
        """
        Find external package dependencies in Go code.

        Args:
            content: Go code content

        Returns:
            Set of dependency package names
        """
        dependencies = set()

        # Extract import paths
        import_blocks = re.findall(r"import\s+\((.*?)\)", content, re.DOTALL)
        for block in import_blocks:
            for line in block.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("//"):
                    # Extract quoted import path
                    import_match = re.search(r'"([^"]+)"', line)
                    if import_match:
                        import_path = import_match.group(1)
                        # Extract package name (last component of path)
                        parts = import_path.split("/")
                        if parts:
                            dependencies.add(parts[-1])

        # Also catch single line imports
        single_imports = re.findall(r'import\s+"([^"]+)"', content)
        for import_path in single_imports:
            parts = import_path.split("/")
            if parts:
                dependencies.add(parts[-1])

        # Remove standard library packages
        std_libs = {
            "fmt",
            "os",
            "io",
            "strings",
            "strconv",
            "time",
            "math",
            "bytes",
            "path",
            "net",
            "http",
            "encoding",
            "json",
            "regexp",
            "context",
            "sync",
            "errors",
            "log",
            "flag",
            "testing",
            "sort",
            "bufio",
            "io/ioutil",
            "path/filepath",
        }

        return dependencies - std_libs

    def extract_test_functions(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract test functions from a Go test file.

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

        # Extract test functions (Test*)
        for func in analysis.get("functions", []):
            if func["name"].startswith("Test"):
                test_functions.append(func)

        return test_functions

    def extract_benchmark_functions(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract benchmark functions from a Go test file.

        Args:
            file_path: Path to the test file

        Returns:
            List of benchmark function dictionaries
        """
        # Parse the file
        analysis = self.parse_file(file_path)

        if "error" in analysis:
            return []

        benchmark_functions = []

        # Extract benchmark functions (Benchmark*)
        for func in analysis.get("functions", []):
            if func["name"].startswith("Benchmark"):
                benchmark_functions.append(func)

        return benchmark_functions

    def extract_example_functions(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract example functions from a Go test file.

        Args:
            file_path: Path to the test file

        Returns:
            List of example function dictionaries
        """
        # Parse the file
        analysis = self.parse_file(file_path)

        if "error" in analysis:
            return []

        example_functions = []

        # Extract example functions (Example*)
        for func in analysis.get("functions", []):
            if func["name"].startswith("Example"):
                example_functions.append(func)

        return example_functions

    def detect_test_pattern(self, file_path: str) -> str:
        """
        Detect if a Go test file uses table-driven tests.

        Args:
            file_path: Path to the test file

        Returns:
            Pattern name: "table_driven" or "standard"
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for table-driven test patterns
            table_patterns = [
                r"tests\s*:=\s*\[\]struct",  # tests := []struct{...}
                r"testCases\s*:=\s*\[\]struct",  # testCases := []struct{...}
                r"cases\s*:=\s*\[\]struct",  # cases := []struct{...}
                r"for\s*,\s*tc\s*:=\s*range\s+",  # for _, tc := range ...
                r"for\s*,\s*tt\s*:=\s*range\s+",  # for _, tt := range ...
                r"for\s*,\s*test\s*:=\s*range\s+",  # for _, test := range ...
            ]

            for pattern in table_patterns:
                if re.search(pattern, content):
                    return "table_driven"

            return "standard"

        except Exception as e:
            logger.error(f"Error detecting test pattern in {file_path}: {str(e)}")
            return "standard"


# Create a singleton instance
go_parser = GoParser()
