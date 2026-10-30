"""Notebook setup and bounded replay use current APIs, with no tuning side effects."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import make_rows, write_dataset

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebook").glob("*.ipynb"))
REMOVED = {
    "CandleBuilder", "Candle", "compute_from_candle", "compute_from_candles", "build_from_candle",
    "check_candle", "min_aggressive_volume", "max_price_displacement_ticks", "big_trade_filter",
    "min_delta_threshold", "min_price_displacement_ticks", "min_price_extension_ticks",
}


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_code_parses_and_uses_current_api(path):
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names.update(n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute))
        assert not names & REMOVED
        assert not any(isinstance(n, ast.Attribute) and n.attr in ("check_bar", "process_signal")
                       for n in ast.walk(tree)), "Replay must route through InfoBarPipeline.process_bar"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_setup_and_small_replay_execute(path, tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat", reason="Notebook execution smoke requires optional nbformat")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    write_dataset(tmp_path, make_rows(120, t0="2023-11-19T23:30:00"))
    monkeypatch.setenv("ORDERFLOW_NOTEBOOK_ROOT", str(tmp_path))
    monkeypatch.setenv("ORDERFLOW_NOTEBOOK_START", "2023-11-19")
    monkeypatch.setenv("ORDERFLOW_NOTEBOOK_END", "2023-11-21")
    monkeypatch.chdir(ROOT)
    namespace = {"__name__": "__notebook_smoke__"}
    executed = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        if any(stop in cell.source for stop in ("# TUNING", "study.optimize", "optuna")):
            break
        exec(compile(cell.source, str(path), "exec"), namespace)
        executed += 1
    assert executed >= 3
    assert namespace["pipeline"].bars_processed == 120
    assert namespace["ledger"].__class__.__module__ == "orderflow_system.ledger"


@pytest.mark.parametrize("argument", ["--help", "--dry-run"])
def test_retired_diagnostic_explains_replacement(argument):
    result = subprocess.run([sys.executable, str(ROOT / "notebook/diagnostics/low_trade_count_four_arm.py"), argument],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "validate_infobars" in result.stdout or "TradeLedger" in result.stdout
