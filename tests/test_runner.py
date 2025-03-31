import pytest
from planner.core.runner import run_planner_with_backend

def test_mock_backend(capsys):
    prompt = "Plan a meeting on Tuesday at 10am"
    run_planner_with_backend(prompt, backend="mock")
    captured = capsys.readouterr()
    assert "Mock planner response" in captured.out

def test_backend_dispatch_invalid(capsys):
    prompt = "Invalid test"
    run_planner_with_backend(prompt, backend="nonexistent")
    captured = capsys.readouterr()
    assert "Mock planner response" in captured.out