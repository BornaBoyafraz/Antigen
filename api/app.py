"""Small FastAPI app that serves the classifier and the demo page.

The model is trained once when the app starts up and kept in memory, so
every request just reuses it. This is only meant for the local demo, not
for real production use.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.examples import EXAMPLE_GALLERY
from conversation import score_conversation
from explain import explain, fit_explainer
from model import load_dataset, train

WEBAPP_DIR = Path(__file__).parent.parent / "webapp"

# The trained model and explainer live here after startup.
_state: dict[str, Any] = {}


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    examples = load_dataset()
    _state["pipeline"] = train(examples)
    _state["explainer"] = fit_explainer(examples)
    _state["n_training_examples"] = len(examples)
    yield
    _state.clear()


app = FastAPI(title="Antigen", lifespan=_lifespan)

# Allow the demo page (and quick experiments) to call the API from anywhere.
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
    explanation: dict[str, Any]


class ClassifyBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class ClassifyConversationRequest(BaseModel):
    turns: list[str] = Field(..., min_length=1, max_length=50)


class ClassifyConversationResponse(BaseModel):
    label: str
    score: float
    escalation_detected: bool
    turns: list[dict[str, Any]]


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "trained": "pipeline" in _state,
        "n_training_examples": _state.get("n_training_examples"),
    }


@app.get("/api/examples")
def examples() -> list[dict[str, str]]:
    return EXAMPLE_GALLERY


def _classify_text(text: str) -> ClassifyResponse:
    exp = explain(_state["pipeline"], _state["explainer"], text)
    return ClassifyResponse(label=exp.label, score=exp.score, explanation=exp.to_dict())


@app.post("/api/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    return _classify_text(req.text)


@app.post("/api/classify_batch", response_model=list[ClassifyResponse])
def classify_batch(req: ClassifyBatchRequest) -> list[ClassifyResponse]:
    return [_classify_text(text) for text in req.texts]


@app.post("/api/classify_conversation", response_model=ClassifyConversationResponse)
def classify_conversation(req: ClassifyConversationRequest) -> ClassifyConversationResponse:
    result = score_conversation(_state["pipeline"], req.turns)
    return ClassifyConversationResponse(**result.to_dict())


# Serve the demo page and its assets at the root.
if WEBAPP_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEBAPP_DIR), html=True), name="webapp")
