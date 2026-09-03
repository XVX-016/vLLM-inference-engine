"""
Phase 5 — FastAPI serving layer. STUB.

Implement using the Phase 5 prompt: OpenAI-compatible /v1/completions
endpoint, streaming via Server-Sent Events, backed by the Scheduler +
engine from Phases 2-4.

Run with: uvicorn engine.serving.api_server:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Paged-KV Inference Engine")


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 64
    stream: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    # TODO(Phase 5): submit req to Scheduler.add_request(), drive engine
    # .step() loop until this sequence finishes, stream tokens back via
    # SSE if req.stream else return full completion.
    raise NotImplementedError("Wire this up to the Scheduler once Phase 4 is complete.")
