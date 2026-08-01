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


def test_threshold_control_is_accessible_and_remote_code_free() -> None:
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
    assert '<script src="http' not in html
    assert '<img src="http' not in html
    assert 'href="https://github.com/BornaBoyafraz/Antigen"' in html


@pytest.mark.skipif(NODE is None, reason="Node.js is required for browser-logic tests")
def test_conversation_logic_preserves_per_turn_escalation_evidence() -> None:
    assert NODE is not None
    script = """
const conversation = require("./webapp/conversation.js");
const result = conversation.normalizeConversationPayload({
  label: "injection",
  score: 0.9,
  escalation_detected: true,
  turns: [
    {text: "setup", label: "injection", score: 0.8, codeword_setups: ["pineapple"]},
    {text: "pineapple", label: "injection", score: 0.9, codeword_invoked: "pineapple"}
  ]
});
process.stdout.write(JSON.stringify({
  escalationDetected: result.escalationDetected,
  invoked: result.turns[1].codewordInvoked,
  summary: conversation.escalationSummary(result)
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
    output = json.loads(completed.stdout)
    assert output["escalationDetected"] is True
    assert output["invoked"] == "pineapple"
    assert output["summary"]["title"] == "Codeword-smuggling escalation detected"
    assert "pineapple" in output["summary"]["detail"]


def test_conversation_mode_posts_to_existing_endpoint_and_renders_turns() -> None:
    html = (ROOT / "webapp" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "webapp" / "app.js").read_text(encoding="utf-8")

    assert 'id="conversation-turn-inputs"' in html
    assert 'id="conversation-turn-results"' in html
    assert 'id="conversation-escalation"' in html
    assert html.index('src="/conversation.js"') < html.index('src="/app.js"')
    assert 'fetch("/api/classify_conversation"' in script
    assert "JSON.stringify({ turns })" in script
    assert "result.turns.map(renderConversationTurn)" in script
    assert 'dataset.state =\n    result.escalationDetected ? "detected" : "clear"' in script
