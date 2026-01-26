#!/usr/bin/env python3
"""
Script to automatically generate API documentation for all modules in randan.
"""

import os
import sys
import subprocess
from pathlib import Path

def find_python_modules(root_dir):
    """Find all Python modules in the package."""
    modules = []
    root_path = Path(root_dir)
    
    for py_file in root_path.rglob("*.py"):
        if "__pycache__" in str(py_file) or "test" in str(py_file).lower():
            continue
            
        # Convert file path to module path
        rel_path = py_file.relative_to(root_path)
        module_path = str(rel_path.with_suffix('')).replace(os.sep, '.')
        
        if module_path.endswith(".__init__"):
            module_path = module_path[:-9]
        
        modules.append(f"randan.{module_path}")
    
    return sorted(set(modules))

def main():
    # Get all modules
    modules = find_python_modules("../randan")
    
    # Create autoapi.rst with all modules
    with open("source/autoapi.rst", "w", encoding="utf-8") as f:
        f.write("Auto-Generated API Documentation\n")
        f.write("================================\n\n")
        
        f.write(".. currentmodule:: randan\n\n")
        
        # Main module
        f.write(".. automodule:: randan\n")
        f.write("   :members:\n")
        f.write("   :undoc-members:\n")
        f.write("   :show-inheritance:\n\n")
        
        # All submodules
        f.write("Submodules\n")
        f.write("----------\n\n")
        
        f.write(".. autosummary::\n")
        f.write("   :toctree: generated\n")
        f.write("   :template: module.rst\n")
        f.write("   :recursive:\n\n")
        
        for module in modules:
            f.write(f"   {module}\n")

if __name__ == "__main__":
    main()
