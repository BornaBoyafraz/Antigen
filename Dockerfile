FROM python:3.12-slim

WORKDIR /app

# Copied explicitly rather than `COPY . .` -- keeps tests/, docs/, .venv/,
# and any local tooling directories out of the image instead of relying on
# a .dockerignore to catch them after the fact.
COPY pyproject.toml README.md ./
COPY features.py model.py explain.py conversation.py ./
COPY api ./api
COPY data ./data
COPY eval ./eval
COPY webapp ./webapp

RUN pip install --no-cache-dir .

EXPOSE 8000

# The model trains in-process at startup (api/app.py's lifespan) against
# data/prompts.jsonl -- there's no separate model-artifact build step here,
# by design (see docs/ARCHITECTURE.md "Honest scope" for what a heavier
# deployment would add on top of this).
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
