"""CLI viewer (python -m debug_viz.viewer <path>) tests.

The end-to-end snapshot → replay path is covered in test_server.py;
this file just smoke-tests the CLI module.
"""
from debug_viz import viewer


def test_viewer_module_exposes_main():
    assert callable(viewer.main)
