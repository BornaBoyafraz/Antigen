"""Tests for self-contained browser-demo interaction wiring."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node.js is required for browser-logic tests")
def test_threshold_decision_logic_changes_at_selected_operating_point() -> None:
    assert NODE is not None
    script = """
const decision = require("./webapp/decision.js");
process.stdout.write(JSON.stringify({
  defaultDecision: decision.decisionForScore(0.62, 0.5),
  raisedDecision: decision.decisionForScore(0.62, 0.75),
  boundaryDecision: decision.decisionForScore(0.62, 0.62),
  lowMode: decision.operatingPointForThreshold(0.2).name,
  middleMode: decision.operatingPointForThreshold(0.5).name,
  highMode: decision.operatingPointForThreshold(0.8).name
}));
"""
    completed = subprocess.run(
        [NODE, "-e", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "defaultDecision": "injection",
        "raisedDecision": "benign",
        "boundaryDecision": "injection",
        "lowMode": "Recall-first",
        "middleMode": "Balanced",
        "highMode": "Selective",
    }


def test_threshold_control_is_accessible_and_self_contained() -> None:
    html = (ROOT / "webapp" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "webapp" / "app.js").read_text(encoding="utf-8")

    assert 'id="decision-threshold"' in html
    assert 'min="0"' in html
    assert 'max="1"' in html
    assert 'step="0.01"' in html
    assert 'aria-describedby="threshold-guidance"' in html
    assert html.index('src="/decision.js"') < html.index('src="/app.js"')
    assert "decisionThreshold.addEventListener(\"input\", updateDecisionThreshold)" in script
    assert "renderDecision();" in script
    assert "http://" not in html
    assert "https://" not in html
