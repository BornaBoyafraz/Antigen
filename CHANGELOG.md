# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The repository has no release tags, and its package version has remained
`0.1.0` throughout its history, so the current development history is grouped
under a combined Unreleased / 0.1.0 heading rather than assigned an invented
release date.

## [Unreleased / 0.1.0]

### Added

- An interpretable local prompt-injection classifier combining character
  n-gram TF-IDF, engineered signals, calibrated logistic regression,
  per-prediction explanations, synthetic data generation, held-out and
  adversarial evaluation, a FastAPI service, and automated tests.
- An interactive browser demo, subsequently extended with live decision-
  threshold controls, ordered multi-turn analysis, and an explanatory landing
  page.
- Narrow multi-turn codeword-smuggling detection and the
  `POST /api/classify_conversation` endpoint.
- Structured-context signals for directives hidden in code comments,
  JSON/tool-argument string values, and URL query parameters, including
  matched spans in prediction explanations.
- A transparent regex-only baseline with side-by-side model lift reporting.
- Evaluation tools for operating-threshold sweeps, category-grouped error
  analysis, engineered-feature coefficients, reproducible dataset statistics,
  latency and throughput measurement, and adversarial evasion retention.
- Batch classification for up to 100 texts, request metrics, configurable
  per-IP rate limiting, and richer generated API documentation.
- Shell and module command-line entry points, plus trusted-local model
  persistence and cache reuse.
- Docker packaging, a container health check, Make workflows, multi-version
  test CI, Ruff and mypy checks, branch-coverage reporting, pre-commit hooks,
  CI failure annotations, and runtime packaging guards.
- A model card, responsible disclosure policy, contributor guide, and issue
  and pull-request templates.

### Changed

- Expanded the synthetic dataset with more benign-but-difficult examples and
  indirect-injection carriers, weighting growth toward the documented false-
  positive weak spot.
- Rebuilt the browser demo as an explanatory landing site while preserving its
  classification workflows.
- Hardened container packaging with explicit exclusions and a health check.

### Fixed

- Reduced false positives for benign text that quotes or discusses attack
  language through discussion-context detection, a guarded deterministic score
  cap, wider `act as` handling, and targeted mixed-phrasing data.
- Prevented the discussion cap from suppressing live injections inside JSON
  and tool results by distinguishing structured string syntax from prose
  quotation and recognizing bare tool-output framing.
- Closed detection gaps for exact system-prompt extraction wording and variants
  such as `ignore your previous instructions`.
- Restored Python 3.10 CI test collection.
- Removed unintended horizontal scrolling on mobile in the landing page.
