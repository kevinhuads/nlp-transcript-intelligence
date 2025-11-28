# tests/conftest.py
import os
import sys

# Absolute path to this file (tests/conftest.py)
current_file = os.path.abspath(__file__)

# Project root: go up one directory from "tests"
project_root = os.path.dirname(os.path.dirname(current_file))

# Ensure the project root is on sys.path so "src" can be imported
if project_root not in sys.path:
    sys.path.insert(0, project_root)
