"""Engineered feature extraction for prompt-injection detection.

Every feature here is independently interpretable: each one answers a
concrete question about the text ("does it contain a fake system-role
marker?", "is there a high density of base64-looking spans?") rather than
being a latent dimension in an embedding. That's the point — the
explanations in explain.py are built directly out of these.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, fields

# --- Pattern banks -----------------------------------------------------

OVERRIDE_PHRASES = [
    r"ignore (?:all|any|the)?\s*(?:previous|prior|above)\s*instructions?",
    r"disregard (?:all|any|the)?\s*(?:previous|prior|above)",
    r"forget (?:all|everything|what)\s*(?:you (?:were|have been) told|above)?",
    r"new instructions?\s*[:\-]",
    r"override\s+(?:your|the|all)\s+(?:instructions?|rules?|guidelines?)",
    r"you (?:are|'re) now\b",
    r"act as (?:a|an)\b",
    r"pretend (?:to be|you are)\b",
    r"developer mode",
    r"do anything now",
    r"\bDAN\b",
    r"jailbreak",
    r"reveal (?:your|the) (?:system prompt|instructions|prompt)",
    r"repeat (?:your|the) (?:system prompt|instructions|prompt|previous text)",
    r"what (?:is|are) your (?:system prompt|instructions)",
    r"from now on\b",
    r"without any (?:restrictions?|limitations?|filters?)",
    r"no (?:longer|more) (?:bound|restricted|limited) by",
]

ROLE_SPOOF_MARKERS = [
    r"\[\s*system\s*\]",
    r"\[\s*/?\s*(?:inst|assistant|user)\s*\]",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|system\|>",
    r"###\s*(?:system|instruction)s?\b",
    r"^\s*system\s*:",
    r"<system>",
]

ADDRESSED_TO_ASSISTANT = [
    r"\bdear\s+(?:ai|assistant|chatbot|model|agent)\b",
    r"\battention\s+(?:ai|assistant|model|agent)\b",
    r"\bto the (?:ai|assistant|language model|llm|agent)\b",
    r"\bif you are an (?:ai|assistant|language model|llm)\b",
    r"\bnote to (?:the )?(?:ai|assistant|model|agent)\b",
    r"\bhey (?:ai|assistant|chatgpt|claude|gemini)\b",
]

IMPERATIVE_VERBS = [
    "ignore", "disregard", "forget", "override", "reveal", "repeat", "output",
    "print", "send", "email", "post", "delete", "execute", "run", "download",
    "visit", "navigate", "click", "submit", "bypass", "disable", "act",
    "pretend", "respond", "reply", "translate", "decode", "leak",
]

_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_HEX_ESCAPE_RE = re.compile(r"(?:\\x[0-9a-fA-F]{2}){4,}|(?:%[0-9a-fA-F]{2}){4,}")
_URL_RE = re.compile(r"https?://\S+")
_ZERO_WIDTH_RE = re.compile(r"[​‌‍⁠﻿]")

_override_re = [re.compile(p, re.IGNORECASE) for p in OVERRIDE_PHRASES]
_role_re = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in ROLE_SPOOF_MARKERS]
_addressed_re = [re.compile(p, re.IGNORECASE) for p in ADDRESSED_TO_ASSISTANT]

INDIRECT_FRAME_MARKERS = [
    r"<(?:webpage|page|document|email|tool_result|tool_output|search_result)>",
    r"\[(?:webpage|document|email|tool result|tool output|search result)\]",
    r"^\s*(?:from|subject)\s*:",
    r"here is the (?:content of the|webpage|document|email|search result)",
]
_indirect_frame_re = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in INDIRECT_FRAME_MARKERS]


@dataclass
class FeatureResult:
    """One row of engineered numeric features plus the raw matches that
    produced them, so callers can explain a prediction, not just score it.
    """
    override_phrase_count: int
    role_spoof_count: int
    addressed_to_assistant_count: int
    imperative_density: float
    base64_span_ratio: float
    hex_or_url_escape_count: int
    zero_width_char_count: int
    url_count: int
    indirect_frame_marker_count: int
    length_chars: int
    uppercase_ratio: float

    matched_override_phrases: list[str]
    matched_role_markers: list[str]
    matched_addressed_phrases: list[str]
    matched_indirect_frames: list[str]

    def to_vector(self) -> list[float]:
        return [
            self.override_phrase_count,
            self.role_spoof_count,
            self.addressed_to_assistant_count,
            self.imperative_density,
            self.base64_span_ratio,
            self.hex_or_url_escape_count,
            self.zero_width_char_count,
            self.url_count,
            self.indirect_frame_marker_count,
            self.length_chars,
            self.uppercase_ratio,
        ]

    @staticmethod
    def vector_names() -> list[str]:
        return [
            "override_phrase_count",
            "role_spoof_count",
            "addressed_to_assistant_count",
            "imperative_density",
            "base64_span_ratio",
            "hex_or_url_escape_count",
            "zero_width_char_count",
            "url_count",
            "indirect_frame_marker_count",
            "length_chars",
            "uppercase_ratio",
        ]


def _find_all(patterns: list[re.Pattern], text: str) -> list[str]:
    hits: list[str] = []
    for p in patterns:
        for m in p.finditer(text):
            hits.append(m.group(0))
    return hits


def _imperative_density(text: str) -> float:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in IMPERATIVE_VERBS)
    return hits / len(words)


def _base64_span_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = sum(len(m.group(0)) for m in _BASE64_RE.finditer(text))
    return min(1.0, total / max(1, len(text)))


def _uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)


def extract_features(text: str) -> FeatureResult:
    """Extract the full engineered feature set for one piece of text.

    `text` is treated as a single block of content destined for an LLM's
    context window — a user prompt, or third-party content (a tool result,
    a webpage excerpt, an email body) that will be inserted into that
    context. The feature set is deliberately the same either way: the
    threat model is "does this text try to steer the assistant," regardless
    of which channel it arrived through.
    """
    normalized = unicodedata.normalize("NFKC", text)

    override_hits = _find_all(_override_re, normalized)
    role_hits = _find_all(_role_re, normalized)
    addressed_hits = _find_all(_addressed_re, normalized)
    indirect_hits = _find_all(_indirect_frame_re, normalized)

    hex_or_url_escapes = len(_HEX_ESCAPE_RE.findall(normalized))
    zero_width = len(_ZERO_WIDTH_RE.findall(text))
    urls = len(_URL_RE.findall(normalized))

    return FeatureResult(
        override_phrase_count=len(override_hits),
        role_spoof_count=len(role_hits),
        addressed_to_assistant_count=len(addressed_hits),
        imperative_density=_imperative_density(normalized),
        base64_span_ratio=_base64_span_ratio(normalized),
        hex_or_url_escape_count=hex_or_url_escapes,
        zero_width_char_count=zero_width,
        url_count=urls,
        indirect_frame_marker_count=len(indirect_hits),
        length_chars=len(text),
        uppercase_ratio=_uppercase_ratio(normalized),
        matched_override_phrases=override_hits,
        matched_role_markers=role_hits,
        matched_addressed_phrases=addressed_hits,
        matched_indirect_frames=indirect_hits,
    )
