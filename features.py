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
    r"ignore (?:all|any|the|your|my|our|this|these|that)?\s*"
    r"(?:previous|prior|above|current)\s*instructions?",
    r"disregard (?:all|any|the|your|my|our|this|these|that)?\s*(?:previous|prior|above|current)",
    r"forget (?:all|everything|what)\s*(?:you (?:were|have been) told|above)?",
    r"new instructions?\s*[:\-]",
    r"override\s+(?:your|the|all)\s+(?:instructions?|rules?|guidelines?)",
    r"you (?:are|'re) now\b",
    r"act as (?:a|an|my|your|the)?\s*\S+",
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

# A matched trigger phrase inside quotes, or preceded by language that
# frames it as something being quoted/discussed/tested rather than issued
# as an instruction, is the model's single biggest known false-positive
# class (see README "Honest scope" / Roadmap). This feature exists to give
# the classifier a signal to distinguish "ignore previous instructions" as
# an attack from 'ignore previous instructions' as the subject of a
# sentence about attacks.
#
# Quoted-span detection has to work even when the matched trigger phrase is
# only a fragment of a longer quoted clause (e.g. the match is "DAN" at the
# tail end of a 40-character quote) — a fixed few-character window around
# just the match doesn't cover that. It also has to not treat contraction
# apostrophes ("don't", "let's") as quote delimiters, since those are far
# more common than real quotes in ordinary English and would otherwise
# spam false positives on this feature. So: neutralize letter-apostrophe-
# letter contractions first, then find actual quote-delimited spans, then
# check whether the match falls inside one of those spans.
_CONTRACTION_APOSTROPHE_RE = re.compile(r"(?<=[A-Za-z])'(?=[A-Za-z])")
_QUOTE_PAIR_RES = [
    re.compile(r"'([^'\n]{1,300}?)'"),
    re.compile(r'"([^"\n]{1,300}?)"'),
    re.compile(r"‘([^’\n]{1,300}?)’"),
    re.compile(r"“([^”\n]{1,300}?)”"),
]


def _quoted_spans(text: str) -> list[tuple[int, int]]:
    neutralized = _CONTRACTION_APOSTROPHE_RE.sub("\0", text)
    spans = []
    for pattern in _QUOTE_PAIR_RES:
        for m in pattern.finditer(neutralized):
            spans.append((m.start(1), m.end(1)))
    return spans


def _is_within_any_span(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(s <= start and end <= e for s, e in spans)


DISCUSSION_CUES = [
    r"\bquot(?:e|ed|es|ing)\b",
    r"\bexample(?:s)?\s+(?:of|like)\b",
    r"\b(?:is|was|are)\s+call(?:ed)?\b",
    r"\bknown as\b",
    r"\bliterally\b",
    r"\bthe (?:phrase|string|term|line|words?)\b",
    r"\bcanonical\b",
    r"\bwell[- ]known\b",
    r"\bfor (?:my|a|this|our) (?:class|thesis|blog post|fiction|unit test|test|slide deck)\b",
    r"\bnot an actual instruction\b",
    r"\bfact[- ]check\b",
    r"\bcitation\b",
    r"\bheadline\b",
    r"\bsample input\b",
    r"\bstyle guide\b",
    r"\bhigh level\b",
    r"\bexplain(?:s|ing)?\b",
    r"\btest suite\b",
]
_discussion_cue_re = [re.compile(p, re.IGNORECASE) for p in DISCUSSION_CUES]
_DISCUSSION_CUE_WINDOW = 80


def _in_discussion_context(text: str, match: re.Match, quoted_spans: list[tuple[int, int]]) -> bool:
    if _is_within_any_span(quoted_spans, match.start(), match.end()):
        return True
    context = text[max(0, match.start() - _DISCUSSION_CUE_WINDOW):match.start()]
    return any(p.search(context) for p in _discussion_cue_re)


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
    discussion_context_count: int

    matched_override_phrases: list[str]
    matched_role_markers: list[str]
    matched_addressed_phrases: list[str]
    matched_indirect_frames: list[str]
    matched_discussion_context_phrases: list[str]

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
            self.discussion_context_count,
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
            "discussion_context_count",
        ]


def _find_all(patterns: list[re.Pattern], text: str) -> list[str]:
    hits: list[str] = []
    for p in patterns:
        for m in p.finditer(text):
            hits.append(m.group(0))
    return hits


def _find_all_matches(patterns: list[re.Pattern], text: str) -> list[re.Match]:
    matches: list[re.Match] = []
    for p in patterns:
        matches.extend(p.finditer(text))
    return matches


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

    override_matches = _find_all_matches(_override_re, normalized)
    role_matches = _find_all_matches(_role_re, normalized)
    addressed_matches = _find_all_matches(_addressed_re, normalized)
    indirect_hits = _find_all(_indirect_frame_re, normalized)

    override_hits = [m.group(0) for m in override_matches]
    role_hits = [m.group(0) for m in role_matches]
    addressed_hits = [m.group(0) for m in addressed_matches]

    quoted_spans = _quoted_spans(normalized)
    discussion_matches = [
        m for m in override_matches + role_matches + addressed_matches
        if _in_discussion_context(normalized, m, quoted_spans)
    ]
    discussion_hits = [m.group(0) for m in discussion_matches]

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
        discussion_context_count=len(discussion_hits),
        matched_override_phrases=override_hits,
        matched_role_markers=role_hits,
        matched_addressed_phrases=addressed_hits,
        matched_indirect_frames=indirect_hits,
        matched_discussion_context_phrases=discussion_hits,
    )
