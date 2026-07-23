"""FastAPI service exposing the Antigen classifier.

On startup, trains the scoring pipeline and the (separate, uncalibrated)
explanation pipeline once on the full dataset and keeps them in memory —
this is a single-process local demo, not a production model-serving setup;
see docs/ARCHITECTURE.md for what a real deployment would add (versioned
model artifacts, async inference workers, concurrency controls).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.examples import EXAMPLE_GALLERY
from conversation import score_conversation
from explain import explain, fit_explainer
from model import load_dataset, train

WEBAPP_DIR = Path(__file__).parent.parent / "webapp"

_state: dict = {}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    examples = load_dataset()
    _state["pipeline"] = train(examples)
    _state["explainer"] = fit_explainer(examples)
    _state["n_training_examples"] = len(examples)
    yield
    _state.clear()


app = FastAPI(title="Antigen", description="Prompt-injection classifier API", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)


class ClassifyResponse(BaseModel):
    label: str
    score: float
    explanation: dict


class ClassifyBatchRequest(BaseModel):
    texts: list[Annotated[str, Field(min_length=1, max_length=20000)]] = Field(
        ..., min_length=1, max_length=100
    )


class ClassifyConversationRequest(BaseModel):
    turns: list[str] = Field(..., min_length=1, max_length=50)


class ClassifyConversationResponse(BaseModel):
    label: str
    score: float
    escalation_detected: bool
    turns: list[dict]


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "trained": "pipeline" in _state,
        "n_training_examples": _state.get("n_training_examples"),
    }


@app.get("/api/examples")
def examples() -> list[dict]:
    return EXAMPLE_GALLERY


def _classify_text(text: str) -> ClassifyResponse:
    pipeline = _state["pipeline"]
    explainer = _state["explainer"]
    exp = explain(pipeline, explainer, text)
    return ClassifyResponse(label=exp.label, score=exp.score, explanation=exp.to_dict())


@app.post("/api/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    return _classify_text(req.text)


@app.post("/api/classify_batch", response_model=list[ClassifyResponse])
def classify_batch(req: ClassifyBatchRequest) -> list[ClassifyResponse]:
    """Classify up to 100 independent text blocks in request order."""
    return [_classify_text(text) for text in req.texts]


@app.post("/api/classify_conversation", response_model=ClassifyConversationResponse)
def classify_conversation(req: ClassifyConversationRequest) -> ClassifyConversationResponse:
    """Scores a conversation window turn-by-turn and additionally flags a
    later turn that invokes a covert trigger word an earlier turn defined
    -- see conversation.py for exactly what this does and doesn't catch."""
    pipeline = _state["pipeline"]
    result = score_conversation(pipeline, req.turns)
    return ClassifyConversationResponse(**result.to_dict())


if WEBAPP_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEBAPP_DIR), html=True), name="webapp")
