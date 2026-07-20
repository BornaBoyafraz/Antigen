# Antigen — architecture

## Threat model

Antigen classifies a single block of text — a user prompt, or third-party
content (a webpage excerpt, an email body, a tool/function-call result,
a file being read by a coding agent) that is about to be inserted into an
LLM's context window — as `benign` or `injection`. The underlying question
is always the same regardless of channel: **does this text attempt to
steer the assistant's behavior in a way the legitimate user didn't ask
for?**

That framing is deliberate. Classic ("direct") prompt injection — a user
typing "ignore your instructions" straight into the chat box — and
indirect prompt injection — an attacker planting "ignore your
instructions" inside a webpage an agent later fetches — are the same
attack from the classifier's point of view: text with instruction-like
content addressed at the assistant. What differs is *how the text arrived*,
which is metadata the caller (the application wrapping the LLM) has and
the classifier doesn't need. Antigen's job is narrower and more tractable:
score the text itself.

Category taxonomy (`direct_injection`, `indirect_injection` — further
split into `tool_use` / `browser_use` / `coding_agent` framings in the
adversarial suite —, `jailbreak_roleplay`, `obfuscated`) is loosely
modeled on the public category descriptions from Gray Swan Arena's
Indirect Prompt Injection Challenge (sponsored by UK AISI, OpenAI,
Anthropic, Amazon, Meta, and Google DeepMind). No text in this repository
is drawn from that or any other live challenge — see "Honest scope" below
and the dataset generation scripts themselves.

## Pipeline

```
text ──▶ features.py (engineered signals) ──┐
                                              ├──▶ FeatureUnion ──▶ classifier ──▶ label, score
text ──▶ TfidfVectorizer (char 3-5-grams) ──┘
```

- **`features.py`** — regex-bank and structural detectors: instruction-override
  phrases ("ignore previous instructions", "developer mode", ...), fake
  role/system markers (`[SYSTEM]`, `<|im_start|>`, ...), text addressed
  directly at "the AI"/"the assistant" (a strong indirect-injection tell —
  legitimate webpages and emails essentially never talk to a language
  model), indirect-content framing markers (`<tool_result>`, "here is the
  content of the webpage...", ...), encoding-smuggling signals (base64-shaped
  spans, hex/URL escapes, zero-width/homoglyph unicode), imperative-mood
  word density, and basic length/casing stats.
- **char n-gram TF-IDF (`analyzer="char_wb"`, n=3-5)** — robust to
  word-level obfuscation (spacing, punctuation insertion, minor
  misspellings) in a way a word-level vectorizer isn't, and it's what lets
  the model pick up on injection-flavored substrings even when they don't
  match a hand-written regex exactly.
- **Classifier** — `LogisticRegression` wrapped in `CalibratedClassifierCV`
  (sigmoid calibration, 3-fold), over the concatenated feature union. A
  linear model was chosen over gradient boosting or a fine-tuned
  transformer encoder specifically because every prediction is
  attributable to a fixed, human-readable feature (see `explain.py`) — for
  a security classifier whose job includes *explaining itself to a human
  reviewing a flagged case*, that property is worth more than the few
  points of accuracy a heavier model might add. The path to swapping in a
  heavier model behind the same `predict_one(pipeline, text) -> (label,
  score)` interface is open; see Roadmap in the README.
- **`explain.py`** — a second, uncalibrated `LogisticRegression` fit on the
  same feature union, used only to extract a single clean coefficient
  vector (the calibrated model internally holds several classifier copies,
  one per CV fold, which is convenient for scoring but awkward to
  introspect for one coherent explanation). Given a piece of text, it
  reports which heuristic patterns matched verbatim and which TF-IDF
  character n-grams contributed most to the score.

## Dataset

`data/generate_dataset.py` produces `data/prompts.jsonl` (247 rows at time
of writing — rerun the script and check the printed counts for the current
number). It is original, hand-authored content, generated two ways:

1. **Hand-written seed examples** for each category (benign, benign-but-
   tricky, direct injection, indirect injection, jailbreak, obfuscated).
2. **Template combinatorics** for the higher-volume categories (direct
   injection: opener × target phrase combinations; indirect injection: a
   "carrier" structure — tool result / webpage / email / calendar entry /
   search result / PR description / config file / document — crossed with
   an injected directive; a matching set of *benign* carrier templates
   with no injected directive, so the model sees the same structural
   framing markers in both classes and can't just learn "any `<tool_result>`
   tag means attack").

None of this is scraped from Gray Swan Arena, any other live red-teaming
challenge, or any other dataset. It's synthetic by construction — the
scripts that generate it are the actual specification of what's in it.

`eval/build_adversarial_suite.py` produces a second, much smaller (30-row)
benchmark that shares zero code or phrasing templates with the training
generator, hand-written specifically to test generalization past the
training distribution's specific wording — including three
indirect-injection sub-categories (`tool_use`, `browser_use`,
`coding_agent`) named after Gray Swan's public track descriptions, as a
taxonomy reference only.

## Evaluation methodology

`eval/harness.py` does a stratified 75/25 train/test split of the main
dataset (`train_test_split(..., stratify=labels)`), trains on the 75%, and
reports precision/recall/F1/ROC-AUC plus a confusion matrix and
per-category accuracy on both:

1. the held-out 25% split, and
2. the fully independent adversarial suite.

Both numbers are reported — not just the easier held-out split — because a
random split of template-generated data still lets phrasing patterns leak
between train and test; the adversarial suite is the harder, more honest
number.

Run `.venv/bin/python examples/demo.py` for the current numbers, or
`.venv/bin/pytest -q` — `test_held_out_accuracy_meets_minimum_bar` and
`test_adversarial_suite_accuracy_meets_minimum_bar` in
`tests/test_model.py` assert fixed minimum bars (0.80 / 0.70) so a future
change that quietly tanks accuracy fails CI instead of shipping.

## Honest scope

What's real and independently checkable: the feature extractors are
regular regex/structural matching over real text (`tests/test_features.py`
checks each one against known cases); the classifier is a real, trained
scikit-learn pipeline scored against a genuinely held-out split and a
disjoint adversarial benchmark (not just re-reported training accuracy);
the explanations are real model internals (matched regex spans, actual
logistic-regression coefficients × TF-IDF weights), not a templated
after-the-fact justification.

What this is *not*: a production security product, a submission to any
live competition, or a claim of state-of-the-art accuracy. Specific,
known limitations:

- **The dataset is small and synthetic** (247 training rows). It covers a
  reasonable breadth of attack phrasing but not the volume or diversity a
  production classifier would need. The per-category accuracy breakdown in
  the eval report is intentionally granular so this isn't hidden behind a
  single aggregate number.
- **The model's most persistent error mode is exactly what you'd predict**:
  benign text that *quotes or discusses* an injection phrase (a security
  class example, a phishing-awareness ticket, a research annotation) is
  hard to distinguish from live injection using n-grams and phrase
  matching alone, since the surface text is nearly identical — the
  difference is quotation/discussion context. `features.py` now has a
  `discussion_context_count` signal for exactly this (quote-span detection
  — robust to the match being only part of a longer quoted clause, and to
  contraction apostrophes like "don't" not being mistaken for quote marks
  — plus a small discussion-cue phrase bank: "the phrase", "known as",
  "for a class", etc.). `model.py` also has a deterministic follow-up,
  `_apply_discussion_override`: when every trigger match in the text is a
  discussion/quoted one (zero bare matches) *and* the text carries no
  indirect-content framing marker, it caps the score instead of trusting a
  single learned coefficient to out-vote the raw trigger-phrase hit. That
  second condition matters — without it, an attacker could wrap a live
  indirect-injection payload in quote marks inside fetched content and get
  it waved through; `test_discussion_override_does_not_fire_inside_indirect_frame`
  in `tests/test_model.py` checks exactly that. Together, this is a real,
  measured improvement, not a full fix: held-out
  `category_accuracy["benign_hard"]` went from 0.0 to 1.0, and the
  independent adversarial suite's `benign_hard` accuracy went from 0.25 to
  0.5. The remaining gap is specifically the cases the override correctly
  declines to touch — text with one quoted example *and* separate unquoted
  wording nearby, where a bare match legitimately exists — which still
  depends on the learned signal alone and needs more training rows of that
  shape, not another override. See Roadmap.
- **No multi-turn context.** Each call scores one block of text in
  isolation; a slow-escalation attack spread across several turns isn't
  modeled.
- **No live LLM-judge fallback in the default path.** The architecture
  could add one (send ambiguous-score cases to an actual model for a
  second opinion) but that would require an API key and network access,
  which this project deliberately avoids so it runs anywhere with no
  setup beyond `pip install`.
- **CPU-only, single-process, in-memory model — by design**, not a
  limitation being hidden: this is a portfolio/demo artifact, not a
  deployed service. `api/app.py` is representative of what a serving layer
  looks like, not a claim that it's production-hardened (no auth, no rate
  limiting, no model versioning).
- **The `Dockerfile` was not verified with an actual `docker build`** — no
  Docker daemon in the environment it was written in. What *was* verified:
  copying the exact file set the image `COPY`s into a clean directory,
  `pip install .` (the same non-editable, non-dev install the image runs)
  in a fresh virtualenv from only those files, then starting `uvicorn` from
  that install and hitting `/api/health`, `/`, and `/api/classify` — which
  catches the packaging failure modes (missing files, import errors) most
  likely to break the real build. It's a strong proxy, not a substitute for
  actually building and running the image before relying on it.
