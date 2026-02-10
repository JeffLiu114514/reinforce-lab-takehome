from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .orchestrator import run as run_agent


WEB_DIR = Path(__file__).resolve().parent / "web"


@dataclass
class RunState:
    status: str = "running"
    progress: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None


_RUNS: dict[str, RunState] = {}
_RUNS_LOCK = threading.Lock()


class RunRequest(BaseModel):
    prompt: str
    k_per_query: int = Field(default=8, ge=1, le=10)
    max_urls: int = Field(default=30, ge=1, le=100)
    out_dir: str = "artifacts"
    config_path: str = "config/source_weights.json"
    google_model: str | None = None
    google_api_key: str | None = None


class RunStartResponse(BaseModel):
    ok: bool
    run_id: str


class RunResponse(BaseModel):
    ok: bool
    out_dir: str
    created_at: str
    summary: dict[str, Any]
    metrics: dict[str, Any]
    report_markdown: str
    graph_mermaid: str
    ledger: dict[str, Any]
    trace: dict[str, Any] | None
    files: dict[str, str]


class RunStatusResponse(BaseModel):
    status: str
    progress: list[str]
    error: str | None = None
    result: RunResponse | None = None


def _restore_env_var(key: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = previous


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_job(run_id: str, payload: RunRequest) -> None:
    def hook(message: str) -> None:
        with _RUNS_LOCK:
            state = _RUNS.get(run_id)
            if state:
                state.progress.append(message)

    prev_model = os.getenv("GOOGLE_MODEL")
    prev_api_key = os.getenv("GOOGLE_API_KEY")

    if payload.google_model and payload.google_model.strip():
        os.environ["GOOGLE_MODEL"] = payload.google_model.strip()
    if payload.google_api_key and payload.google_api_key.strip():
        os.environ["GOOGLE_API_KEY"] = payload.google_api_key.strip()

    try:
        ledger = run_agent(
            prompt=payload.prompt,
            k_per_query=payload.k_per_query,
            max_urls=payload.max_urls,
            out_dir=payload.out_dir,
            config_path=payload.config_path,
            progress_hook=hook,
        )

        out_dir = Path(payload.out_dir)
        report_path = out_dir / "report.md"
        graph_path = out_dir / "graph.mmd"
        ledger_path = out_dir / "ledger.json"
        trace_path = out_dir / "trace.json"

        summary = {
            "sources": len(ledger.sources),
            "evidence": len(ledger.evidence),
            "claims": len(ledger.graph.claims),
            "edges": len(ledger.graph.edges),
            "resolutions": len(ledger.resolutions),
        }

        result = RunResponse(
            ok=True,
            out_dir=str(out_dir),
            created_at=ledger.created_at,
            summary=summary,
            metrics=ledger.metrics,
            report_markdown=_read_text(report_path),
            graph_mermaid=_read_text(graph_path),
            ledger=ledger.model_dump(mode="json"),
            trace=_read_json(trace_path),
            files={
                "report": str(report_path.resolve()),
                "graph": str(graph_path.resolve()),
                "ledger": str(ledger_path.resolve()),
                "trace": str(trace_path.resolve()),
            },
        )

        with _RUNS_LOCK:
            state = _RUNS.get(run_id)
            if state:
                state.status = "complete"
                state.result = result.model_dump(mode="json")
    except Exception as exc:
        with _RUNS_LOCK:
            state = _RUNS.get(run_id)
            if state:
                state.status = "error"
                state.error = str(exc)
    finally:
        _restore_env_var("GOOGLE_MODEL", prev_model)
        _restore_env_var("GOOGLE_API_KEY", prev_api_key)


def create_app() -> FastAPI:
    app = FastAPI(title="Research Agent UI", version="0.1.0")
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        index_path = WEB_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=500, detail="UI assets are missing")
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.post("/api/run", response_model=RunStartResponse)
    def run_api(payload: RunRequest) -> RunStartResponse:
        prompt = payload.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt must not be empty")

        run_id = uuid.uuid4().hex
        with _RUNS_LOCK:
            _RUNS[run_id] = RunState(status="running", progress=["[init] run queued"])

        thread = threading.Thread(target=_run_job, args=(run_id, payload), daemon=True)
        thread.start()
        return RunStartResponse(ok=True, run_id=run_id)

    @app.get("/api/status/{run_id}", response_model=RunStatusResponse)
    def run_status(run_id: str) -> RunStatusResponse:
        with _RUNS_LOCK:
            state = _RUNS.get(run_id)
            if not state:
                raise HTTPException(status_code=404, detail="run_id not found")
            result = state.result
            return RunStatusResponse(
                status=state.status,
                progress=list(state.progress),
                error=state.error,
                result=RunResponse(**result) if result else None,
            )

    return app


app = create_app()
