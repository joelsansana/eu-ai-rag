"""Tests for the CI eval gate (scripts/eval_gate.py).

The gate's contract:
- current < baseline - tolerance  -> exit non-zero (GATE FAILED)
- current >= baseline - tolerance -> pass, prints the comparison
- baseline unfetchable / invalid  -> pass with warning (bootstrap case)
- current file broken or metric missing -> exit non-zero (GATE ERROR)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from eval_gate import BaselineNotFoundError, fetch_baseline, load_current, main


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def current_file(tmp_path: Path) -> Path:
    return _write(tmp_path / "current.json", {"hit_at_5": 0.81, "n_questions": 31})


class TestFetchBaseline:
    def test_parses_valid_baseline(self, tmp_path, monkeypatch):
        baseline_file = _write(tmp_path / "baseline.json", {"hit_at_5": 0.84})
        monkeypatch.setattr(
            "eval_gate.requests.get",
            lambda url, timeout: type("R", (), {
                "text": baseline_file.read_text(),
                "raise_for_status": lambda self: None,
            })(),
        )
        assert fetch_baseline("http://example.com/baseline.json") == {"hit_at_5": 0.84}

    def test_network_error_raises_baseline_not_found(self, monkeypatch):
        import requests as _requests

        def boom(url, timeout):
            raise _requests.ConnectionError("no route to host")

        monkeypatch.setattr("eval_gate.requests.get", boom)
        with pytest.raises(BaselineNotFoundError):
            fetch_baseline("http://example.com/baseline.json")

    def test_invalid_json_raises_baseline_not_found(self, monkeypatch):
        monkeypatch.setattr(
            "eval_gate.requests.get",
            lambda url, timeout: type("R", (), {
                "text": "not json",
                "raise_for_status": lambda self: None,
            })(),
        )
        with pytest.raises(BaselineNotFoundError):
            fetch_baseline("http://example.com/baseline.json")


class TestLoadCurrent:
    def test_loads_current(self, current_file):
        assert load_current(current_file)["hit_at_5"] == 0.81

    def test_missing_current_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            load_current(tmp_path / "nope.json")

    def test_malformed_current_file_exits(self, tmp_path):
        (tmp_path / "bad.json").write_text("{broken")
        with pytest.raises(SystemExit):
            load_current(tmp_path / "bad.json")


class TestMainGate:
    def _run(self, monkeypatch, tmp_path, baseline, current):
        baseline_file = _write(tmp_path / "baseline.json", baseline)
        current_file = _write(tmp_path / "current.json", current)
        monkeypatch.setattr(
            "eval_gate.requests.get",
            lambda url, timeout: type("R", (), {
                "text": baseline_file.read_text(),
                "raise_for_status": lambda self: None,
            })(),
        )
        argv = [
            "eval_gate.py",
            "--current", str(current_file),
            "--baseline-url", "http://example.com/baseline.json",
            "--metric", "hit_at_5",
            "--tolerance", "0.02",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        main()

    def test_passes_when_within_tolerance(self, monkeypatch, tmp_path, current_file):
        # 0.83 vs baseline 0.84 = a 1-point drop, inside the 2-point tolerance
        self._run(monkeypatch, tmp_path, {"hit_at_5": 0.84}, {"hit_at_5": 0.83})

    def test_passes_at_exactly_tolerance_boundary(self, monkeypatch, tmp_path):
        self._run(monkeypatch, tmp_path, {"hit_at_5": 0.84}, {"hit_at_5": 0.82})

    def test_fails_when_below_tolerance(self, monkeypatch, tmp_path):
        # sys.exit(str) carries the message as the exit code; the interpreter
        # prints it to stderr only at top level, so assert on the code.
        with pytest.raises(SystemExit) as exc:
            self._run(monkeypatch, tmp_path, {"hit_at_5": 0.84}, {"hit_at_5": 0.79})
        assert exc.value.code != 0
        assert "GATE FAILED" in str(exc.value.code)

    def test_missing_baseline_passes_with_warning(self, monkeypatch, tmp_path, current_file, capsys):
        import requests as _requests

        def boom(url, timeout):
            raise _requests.ConnectionError("404")

        monkeypatch.setattr("eval_gate.requests.get", boom)
        monkeypatch.setattr(sys, "argv", [
            "eval_gate.py",
            "--current", str(current_file),
            "--baseline-url", "http://example.com/baseline.json",
            "--metric", "hit_at_5",
        ])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0  # explicit pass
        assert "No baseline found" in capsys.readouterr().out

    def test_metric_missing_from_current_is_gate_error(self, monkeypatch, tmp_path):
        with pytest.raises(SystemExit) as exc:
            self._run(monkeypatch, tmp_path, {"hit_at_5": 0.84}, {"n_questions": 5})
        assert "GATE ERROR" in str(exc.value) or exc.value.code != 0
