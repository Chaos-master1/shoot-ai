"""Early stub installation for all unit tests.

This conftest runs before any test file in this directory is imported,
ensuring kawkab.core.paths and kawkab.core.logging stubs are in place
before storage_service.py (or any other module) is loaded.

NOTE: Test files should NOT import from this conftest. The parent
tests/conftest.py is loaded automatically by pytest and provides
install_kawkab_stubs, load_service_module, etc.
"""

import importlib.util
import os
import sys

_tests_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_dir = os.path.join(_tests_dir, "..", "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Load the parent conftest module to get the stubs
_conftest_path = os.path.join(_tests_dir, "conftest.py")
_spec = importlib.util.spec_from_file_location("tests.conftest", _conftest_path)
_parent_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parent_mod)
sys.modules["tests.conftest"] = _parent_mod

# Re-export names that test files import from conftest
install_kawkab_stubs = _parent_mod.install_kawkab_stubs
install_loguru_stub = _parent_mod.install_loguru_stub
install_httpx_stub = _parent_mod.install_httpx_stub
load_service_module = _parent_mod.load_service_module
_ensure_package_loaded = _parent_mod._ensure_package_loaded

# Install stubs immediately
install_kawkab_stubs()
