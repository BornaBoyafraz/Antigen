"""Command-line access to the classifier for scripts and local checks.

The command trains the existing scoring and explanation pipelines once at
startup, then classifies positional text or piped standard input without
mixing status output into the result stream.
"""
from __future__ import annotations

import argparse
import json
import sys

from explain import explain, fit_explainer
from model import load_dataset, train

# Training is deferred to first use rather than done at import time, so that
# --help and argument errors return instantly instead of paying for a full
# training run, and so importing this module has no side effects.
_STATE: dict = {}

_SIGNAL_LABELS = (
    ("triggered_phrases", "Phrases"),
    ("triggered_role_markers", "Role markers"),
    ("triggered_addressed_phrases", "Addressed phrases"),
    ("triggered_indirect_frames", "Indirect frames"),
    ("discussion_context_phrases", "Discussion context"),
)


def _load_pipelines():
    """Train the scoring and explanation pipelines once, caching them for the
    life of the process."""
    if not _STATE:
        examples = load_dataset()
        _STATE["pipeline"] = train(examples)
        _STATE["explainer"] = fit_explainer(examples)
    return _STATE["pipeline"], _STATE["explainer"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify text as prompt injection or benign content.",
    )
    parser.add_argument("text", nargs="*", help="text to classify; reads stdin when omitted")
    parser.add_argument("--json", action="store_true", help="emit the complete explanation as JSON")
    return parser


def _render_human(explanation: dict) -> str:
    lines = [f"{explanation['label'].upper()} {explanation['score']:.3f}"]
    for field, label in _SIGNAL_LABELS:
        signals = explanation[field]
        if signals:
            lines.append(f"  {label}: {', '.join(signals)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Classify command-line or piped text and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.text:
        text = " ".join(args.text)
    elif sys.stdin.isatty():
        # Nothing piped in and nothing on the command line: fail fast instead
        # of silently blocking on an interactive terminal waiting for input.
        text = ""
    else:
        text = sys.stdin.read()

    if not text.strip():
        print(f"{parser.prog}: error: no text provided", file=sys.stderr)
        return 2

    pipeline, explainer = _load_pipelines()
    explanation = explain(pipeline, explainer, text).to_dict()
    if args.json:
        print(json.dumps(explanation))
    else:
        print(_render_human(explanation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
