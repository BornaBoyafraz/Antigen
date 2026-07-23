"""Persist trained classifier pipelines for faster repeated local runs.

The cache is intentionally explicit and path-based: callers choose when two
runs share an artifact, while this module handles atomic writes and validates
the loaded object type. Joblib artifacts use pickle semantics and must only be
loaded from trusted sources.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from model import Example, train


def save_pipeline(pipeline: Pipeline, path: Path) -> Path:
    """Atomically save a trained pipeline and return its destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        joblib.dump(pipeline, temporary_path, compress=3)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    return destination


def load_pipeline(path: Path) -> Pipeline:
    """Load a pipeline from a trusted local artifact.

    Joblib can execute code while deserializing. Never pass a file received
    from an untrusted source.
    """
    loaded = joblib.load(Path(path))
    if not isinstance(loaded, Pipeline):
        raise TypeError(f"expected a scikit-learn Pipeline in {path}")
    return loaded


def train_and_cache(
    examples: list[Example],
    path: Path,
    *,
    force_retrain: bool = False,
) -> Pipeline:
    """Load an existing cache or train and persist a new pipeline.

    Cache freshness is controlled by the caller: an existing artifact is
    reused without comparing dataset or source-code versions. Set
    ``force_retrain`` or choose a versioned path when those inputs change.
    """
    cache_path = Path(path)
    if cache_path.is_file() and not force_retrain:
        return load_pipeline(cache_path)

    pipeline = train(examples)
    save_pipeline(pipeline, cache_path)
    return pipeline
